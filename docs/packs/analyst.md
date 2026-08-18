# Analyst Studio

`analyst` · version `3.5.0` · 5.3 KB

**[Download analyst-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/analyst-3.5.0.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/analyst-3.5.0.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/analyst-3.5.0.tar.gz
tar xzf analyst-3.5.0.tar.gz
python analyst/install.py
```

Installs to `~/.aither/packs/analyst/`, which adk discovers with no
configuration. The installer verifies the pack is discoverable rather than
assuming it. adk itself:

```bash
pip install aither-adk
```

## About

A data-analysis agent that classifies, regresses, and forecasts over
structured data using zero-shot foundation models (TabFM + TimesFM), and
reasons about the results. Adapts to new labeled data in-context (support
set) instead of gradient training.

## Skills

- `anomaly-detection`
- `structured-inference`

## Contents

```
agent.yaml
brain_pack.yaml
skills/anomaly-detection.md
skills/structured-inference.md
```

---

sha256 `4fd77be22dec8b054c1960443cb3c21dd7a7058807e25eab5549fa6e2fa42933`  
Built from `v3.5.0` (adk 3.5.0). [All packs](../packs.md)
