# ODS Model Resolver

Deterministic, offline model selection for local LLM inference based on hardware constraints and user preferences.

## Overview

The **Open Data Systems (ODS)** model resolver is a vendored, ported component from [github.com/Osmantic/ODS](https://github.com/Osmantic/ODS). It powers model-fitting decisions in the aither-adk without requiring external binaries, network calls, or third-party services.

### Design Principles

- **Deterministic**: Same hardware + constraints → same model, every time
- **Offline**: No network calls; all decisions from vendored catalog
- **Fail-closed**: Raises `OdsError` on any logic failure; never silent/empty
- **Profile-aware**: Routes `auto` profile to best family for detected hardware (gemma4 on Apple, qwen elsewhere)
- **Extensible**: Edge rules (unified-memory substitution, context floor, Spark override) baked in; easy to add more

## API

### OdsResolver

```python
from adk.ods import OdsResolver, OdsRecommendation, OdsError

resolver = OdsResolver()  # Loads catalog from package data on first call

recommendation = resolver.resolve(
    backend='nvidia',           # nvidia|amd|apple|cpu|unknown
    memory_type='discrete',     # discrete|unified
    vram_mb=24576,
    ram_gb=32,
    profile='qwen',             # qwen|gemma4|auto
    tier='3',                   # hardware tier; None = auto-detect
    host_arch='x86_64',         # x86_64|arm64|unknown
    max_size_mb=None,           # max model size; None = no limit
    installable_only=False,     # if True, only recommend install_recommendation=True models
)

print(recommendation.selected.name)
print(recommendation.reason)
print(recommendation.confidence)
# → OdsRecommendation with policy, source='ods', selected model, reason, confidence, alternatives
```

### Return Shape

```python
@dataclass
class OdsRecommendation:
    policy: str              # e.g. 'unified-memory-coder-next-a3b-v1', 'default-tier3-qwen'
    source: str              # always 'ods'
    confidence: float        # 0.0-1.0 (0.95+ high confidence)
    profile: str             # resolved profile (qwen or gemma4)
    host_arch: str           # echoed from input
    memory_capacity_gb: float # usable capacity (35% CPU, 55% unified, 95% discrete)
    memory_label: str        # human-readable (e.g. 'NVIDIA A100 discrete (24GB)')
    selected: ModelRecord    # the recommended model
    reason: str              # human-readable reason
    alternatives: list[ModelRecord]  # top 3 alternatives
```

## Integration with recommend_config()

The resolver is **PRIMARY** in `LLMFitClient.recommend_config()`:

```python
from adk.llmfit import LLMFitClient

client = LLMFitClient()

# ODS is tried first (deterministic, offline, source='ods')
result = client.recommend_config(
    backend='nvidia',
    vram_mb=24576,
    ram_gb=32,
    profile='qwen',
    tier='3',
    use_llmfit=False  # (default) try ODS first
)

# result = {
#     'hardware': { ... },
#     'fast': { 'model': 'qwen3.5-7b-q4', 'provider': 'ods', 'score': 0.92, 'tps': 22, ... },
#     'balanced': { ... },
#     'reasoning': { ... },
#     'coding': { ... },
#     'embedding': { ... },
# }
```

### Path Priority (recommend_config)

```
recommend_config(use_llmfit=False) called
  ├─→ Path 1: ODS Resolver (PRIMARY) ──────────────────────
  │   ├─ Catch OdsError on import → log, fall to Path 2
  │   ├─ Call system_info() → get hw dict
  │   ├─ If hw is None → log, fall to Path 2
  │   ├─ Call OdsResolver.resolve() 5 times (one per tier)
  │   ├─ Assemble config with provider='ods' tag
  │   └─ Return config (success path)
  │
  ├─ Path 2: llmfit (FALLBACK) ─────────────────────────
  │   ├─ Call system_info() if not cached
  │   ├─ Call llmfit.top_models(use_case=tier) for each tier
  │   ├─ Assemble config with provider='llmfit' tag
  │   └─ Return config (or error dict if unavailable)
  │
  └─ Return config dict (never None, always has 'hardware' + tier dicts or 'error' key)
```

Return shape is **unchanged** from llmfit era (backward compatible):
- Each tier (fast, balanced, reasoning, coding, embedding) is `{...}` or `None`
- New `provider` field indicates source: `'ods'` or `'llmfit'`
- Scoring range (0.0-1.0) is normalized across both paths for caller compatibility

If ODS is unavailable or `use_llmfit=True` is passed, the call falls back to llmfit.

## Traceability & Versioning

To identify which ODS version is running:

```python
from adk.ods import ODS_VENDORED_COMMIT, ODS_LAST_VENDORED_DATE

print(f"Using ODS catalog {ODS_VENDORED_COMMIT} vendored {ODS_LAST_VENDORED_DATE}")
# Output: Using ODS catalog abc123def456 vendored 2026-07-25
```

Each tier recommendation includes:
- `source: 'ods'` — indicates ODS resolver was used (deterministic)
- `score: float` — confidence 0.0-1.0 (ODS always >= 0.75 unless fallback)
- Optional `source_metadata` — raw policy, reason, alternatives for debugging

**Catalog Metadata:** model-library.json may include top-level metadata:
```json
{
  "version": "1.0",
  "metadata": {
    "upstream_commit": "abc123...",
    "upstream_date": "2026-07-25T...",
    "vendored_from": "https://github.com/Osmantic/ODS"
  },
  "models": [...]
}
```

This allows downstream tools to validate which upstream ODS version generated recommendations.

## Hardware Routing Rules

### Profile Routing (auto)

| Backend      | Memory Type | Profile → |
|--------------|-------------|-----------|
| apple        | unified     | gemma4    |
| nvidia       | discrete    | qwen      |
| amd          | discrete    | qwen      |
| cpu          | —           | qwen      |
| unknown      | —           | qwen      |

### Tier Detection

| Backend | VRAM / RAM Range         | Tier          |
|---------|--------------------------|---------------|
| nvidia  | 0-2 GB                   | 0 (CPU-like)  |
| nvidia  | 2-8 GB                   | 2 (entry)     |
| nvidia  | 8-20 GB                  | 3 (mid)       |
| nvidia  | 20+ GB                   | 4 (high)      |
| apple   | unified 8-16 GB          | SH_COMPACT    |
| apple   | unified 16+ GB           | SH_LARGE      |
| cpu     | 4-8 GB RAM               | 0             |
| cpu     | 8-32 GB RAM              | 1             |
| cpu     | 32+ GB RAM               | 2             |

### Edge Rules

#### 1. Unified-Memory Coder Substitution (Apple Silicon)

When:
- `backend='apple'` AND `memory_type='unified'` AND `profile='qwen'` AND tier in ['SH_LARGE']
- Model family = qwen AND specialty = 'Code'

Then:
- **Substitute** to `qwen3.6-35b-a3b-ud-q4` (Unified Density variant)
- **Policy**: `'unified-memory-coder-next-a3b-v1'`
- **Reason**: "Unified-memory optimization for Apple Silicon code workload"

#### 2. Hermes Context Floor (Reasoning Models)

When:
- Model family = 'hermes' AND tier in ['3', '4', 'NV_ULTRA', 'SH_LARGE']

Then:
- **Clamp** model.context_length to min(131072, catalog value)
- **Reason**: "Hermes reasoning models support extended context; tier allows it"

#### 3. Spark aarch64 Override (ARM64 Discrete)

When:
- `host_arch='arm64'` AND `backend='nvidia'` AND tier='NV_ULTRA'

Then:
- **Prefer** models with `specialty='Spark'` or ARM-optimized quantizations
- **Policy**: `'spark-aarch64-nv-ultra-a3b-v1'`
- **Reason**: "ARM64 hardware with high-end GPU; Spark variant optimal"

#### 4. Bootstrap Model Fallback

When:
- `installable_only=True` AND `max_size_mb` too small for profile/family
- OR no candidates after profile/family filtering

Then:
- **Fall back** to `qwen3.5-2b-q4` (bootstrap model, ~1.5 GB)
- **Policy**: `'bootstrap-fallback'`
- **Confidence**: < 0.75
- **Reason**: "No suitable candidates; bootstrap model ensures installation works"

## Scoring & Ranking

Each candidate model is scored based on:

1. **Memory fit** (base score)
   - Candidate VRAM ≤ available VRAM → fit_ratio = candidate / available
   - fit_ratio < (1 - VRAM_FIT_TOLERANCE_GB / available) → excluded
   - fit_ratio >= 0.98 (very tight fit) → apply -0.35 penalty
   - Otherwise → score += fit_ratio * 0.50

2. **Profile match**
   - Model family matches profile → score += 0.30
   - Model specialty matches profile → score += 0.20

3. **Context length** (reasoning workloads)
   - Model context >= 32000 → score += 0.15
   - Model context >= 128000 → score += 0.25

4. **Quantization**
   - Q4 quantization (good balance) → score += 0.10
   - Q5/Q6 (lossless) → score += 0.15

5. **Throughput** (if profile defines runtime_profiles)
   - Extrapolate tokens/sec from catalog
   - Higher TPS → score += 0.10

Final score = min(1.0, base + adjustments).

## Catalog Schema

### model-library.json

```json
{
  "version": "1.0",
  "models": [
    {
      "id": "qwen3.5-2b-q4",
      "name": "Qwen 3.5 2B Q4",
      "family": "qwen",
      "gguf_file": "qwen3.5-2b-q4.gguf",
      "gguf_url": "https://huggingface.co/...",
      "gguf_sha256": "abc123...",
      "size_mb": 1500,
      "vram_required_gb": 1.2,
      "context_length": 32000,
      "quantization": "q4",
      "specialty": "Bootstrap",
      "llm_model_name": "qwen3.5-2b-instruct",
      "install_recommendation": true,
      "runtime_profiles": {
        "qwen": { "tps": 18, "tokens_per_batch": 256 }
      },
      "app_compatibility": ["General", "Code", "Chat"]
    }
  ]
}
```

### gpu-database.json

```json
{
  "known_gpus": { "nvidia": { "10de:2184": {...} } },
  "heuristic_classes": [ { "vendor": "nvidia", "vram_min_gb": 20, "tier": "3" } ],
  "known_gpu_bandwidth": { "nvidia": { "A100-80GB": 2039.0 } },
  "defaults": {
    "vram_fit_tolerance_gb": 0.25,
    "tight_fit_threshold": 0.98,
    "profile_default": "auto"
  }
}
```

## Re-vendor Procedure

To update the catalog from upstream ODS:

### 1. Fetch upstream

```bash
cd /tmp
git clone https://github.com/Osmantic/ODS.git
cd ODS
git checkout main  # or a specific tag/commit
TARGET_COMMIT=$(git rev-parse --short HEAD)
```

### 2. Copy files

```bash
cp config/model-library.json /path/to/aither-adk/adk/ods/
cp config/gpu-database.json /path/to/aither-adk/adk/ods/
cp config/hardware-classes.json /path/to/aither-adk/adk/ods/
```

### 3. Update metadata

Edit `/path/to/aither-adk/adk/ods/__init__.py`:

```python
ODS_VENDORED_COMMIT = "abc123def456"  # from step 1
ODS_VENDORED_URL = "https://github.com/Osmantic/ODS"
```

### 4. Validate schema

```bash
cd /path/to/aither-adk
python -c "from adk.ods import load_catalog; load_catalog()"
```

### 5. Run tests

```bash
python -m pytest adk/ods/tests/test_resolver.py -v
python -m pytest adk/ods/tests/test_integration.py -v
```

Expected: All POSITIVE assertions pass (known hardware profiles return expected models).

### 6. Commit

```bash
git add adk/ods/*.json adk/ods/__init__.py
git commit -m "chore: vendor ODS catalog $(TARGET_COMMIT)"
```

## Constants & Configuration

```python
# In adk/ods/__init__.py
ODS_VENDORED_COMMIT = None          # unknown: vendored from a zipball, not a clone
ODS_VENDORED_REF = "main"
ODS_VENDORED_RELEASE = "2.5.3"      # ods/manifest.json at time of vendoring
ODS_VENDORED_URL = "https://github.com/Osmantic/ODS"
ODS_VENDORED_SHA256 = {...}         # per-file integrity anchor; see validate_catalog.py

# In adk/ods/data.py
VRAM_FIT_TOLERANCE_GB = 0.25        # Models within 0.25 GB are considered "fit"
TIGHT_FIT_THRESHOLD = 0.98          # fit_ratio > 0.98 triggers penalty
TIGHT_FIT_PENALTY = -0.35           # Score penalty for tight fits
MEMORY_USAGE_CPU_PERCENT = 0.35     # CPU: use 35% of RAM for models
MEMORY_USAGE_UNIFIED_PERCENT = 0.55 # Unified: use 55% for models
MEMORY_USAGE_DISCRETE_PERCENT = 0.95 # Discrete: use 95% of VRAM for models
```

## Error Handling

```python
from adk.ods import OdsError

try:
    recommendation = resolver.resolve(...)
except OdsError as e:
    # Catalog missing, corrupt, or no viable candidates
    # Fallback: print error message to user, suggest manual model selection
    print(f"Model selection failed: {e}")
    # Do NOT return None or empty dict — fail-closed
    raise
```

Common errors:

| Error | Cause | Action |
|-------|-------|--------|
| "Catalog missing or invalid schema" | JSON files not found or corrupt | Re-run installation; check package integrity |
| "No viable candidates after exhausting fallback pool" | Constraints too tight (e.g., max_size_mb=100 MB on GPU) | Loosen constraints; manually select model |
| "Hardware detection failed" | Backend/memory_type mismatch | Double-check hardware specs; pass explicit tier |

## Backward Compatibility

The return shape of `LLMFitClient.recommend_config()` is **unchanged**:

```python
{
  'hardware': { 'backend': 'nvidia', 'vram_gb': 24, ... },
  'fast': { 'model': 'qwen3.5-7b-q4', 'score': 0.92, 'tps': 22 },
  'balanced': { 'model': 'qwen3-37b-q4', 'score': 0.85, 'tps': 16 },
  'reasoning': { 'model': 'qwen3.6-35b-q4', 'score': 0.88, 'tps': 14 },
  'coding': { 'model': 'qwen3-coder-37b-q4', 'score': 0.91, 'tps': 12 },
  'embedding': { 'model': 'bge-small-en-q4', 'score': 0.80, ... },
}
```

Callers do NOT need to change; ODS→llmfit transformation is transparent.

## License

Vendored from [Osmantic/ODS](https://github.com/Osmantic/ODS) under Apache License 2.0.
See `NOTICE` and repository `LICENSE` for details.
