# Changelog

All notable changes to aither-adk will be documented in this file.

## [2.6.4] - 2026-06-05

### Fixes
- **Loop-guard nudges corrupted the message history → DeepSeek 400 mid-run.**
  When the ReAct loop emitted multiple tool calls in one assistant turn and the
  loop guard fired a WARN/BLOCK/CIRCUIT_BREAK on one of them, it spliced a
  standalone `system` nudge message *between* the tool results. DeepSeek (and
  strict vLLM chat templates) require an assistant `tool_calls` message to be
  followed by exactly one `tool` result per `tool_call_id`, contiguously — so the
  interleaved message produced `400 … insufficient tool messages following
  tool_calls message` and aborted the turn (this is reliably hit after crossing
  the 15-call circuit-break threshold, where every subsequent call WARNs).
  Per-call guidance is now folded into that call's own `tool` result, and any
  standalone steering is deferred to a single message AFTER the whole batch —
  preserving the `tool_calls`↔`tool_results` pairing every backend requires.
  Regression test: `deep-research-agent/tests/test_msg_structure.py`
  (loopguard-flood scenario).

## [2.6.3] - 2026-06-04

### Fixes
- **Transient provider errors killed a run on the first blip.** A single
  `502 Bad Gateway` / `503` / `529 overloaded` / `429` from the upstream API
  aborted the whole request — the Anthropic provider had **no retry at all**, and
  the OpenAI-compatible provider only retried `429`. Both now retry transient
  statuses (`408/429/500/502/503/504/529`) with `Retry-After`-aware exponential
  backoff (3 attempts), and surface the provider's body if all retries are
  exhausted. Non-transient `4xx` (e.g. `400`) still fails fast. Applies to
  `chat` and `chat_stream` on both providers.

## [2.6.2] - 2026-06-04

### Fixes
- **DeepSeek (and strict OpenAI-compatible backends) 400'd on long tool
  conversations.** The ReAct loop legitimately injects mid-conversation
  `system` steering messages (the "[DIMINISHING RETURNS]" hint after several
  low-output tool iterations, loop-guard nudges). OpenAI tolerates a non-leading
  `system` message; DeepSeek rejects it with a 400. Long research runs
  (many `fetch_url` calls) reliably tripped this. Fix: non-leading `system`
  messages are demoted to `user` at the OpenAI-compatible provider boundary
  (`_demote_nonleading_system`) — steering intent preserved, payload valid on
  every backend; leading system prompt(s) untouched.
- **Opaque HTTP errors.** `resp.raise_for_status()` discarded the provider's
  response body, so a 400/422 surfaced as a bare `Client error '400 Bad
  Request'` with no cause. `_ensure_ok` now includes the provider's actual
  message (e.g. `context_length_exceeded`, `Invalid 'messages'`) in the raised
  error, for both `chat` and `chat_stream`.

## [2.6.1] - 2026-06-04

### Fixes
- **`MCPServer.mount()` returned HTTP 422 for every `/mcp` call.** Under
  `from __future__ import annotations` the route handler's `request: Request`
  annotation is a string that FastAPI resolves against the *module* globals, but
  `Request` was imported locally inside `mount()` — so FastAPI couldn't resolve
  it and treated `request` as a required query param. The MCP server endpoint
  (`initialize`, `tools/list`, `tools/call`) was unusable when mounted. FastAPI
  symbols are now imported at module level (guarded; the `[node]` extra still
  gates actual use). Any agent's tools can now be exposed as a working MCP
  server with `MCPServer(...).mount(app)`.

## [2.6.0] - 2026-06-04

### Streaming + scaffolding
- **`AitherAgent.stream_chat`** — reliable streaming sibling of `chat()`: native
  tool-calling loop (concludes on its own, unlike the text-protocol `stream_react`)
  that emits live `tool`/`tool_result` events and streams the answer as `token`
  events. `token_delay` paces tokens for a typing effect.
- **`adk new <template>`** — scaffold a full, runnable template app. Ships the
  `deep-research` template (a cited web-research agent: pack + server + web UI):
  `adk new deep-research`.

### Fixes
- **BYO-key no longer community-capped.** The free-tier monthly token cap now applies
  only to the metered Aitherium gateway backend, never to a user's own provider key
  (Anthropic/OpenAI/DeepSeek/Ollama/vLLM). Previously it could brick a self-hosted
  agent after ~100k tokens on the user's own key.
- **Stronger, grounded memory recall.** `chat()` now injects the top-6 knowledge-graph
  hits (untruncated) with a hard no-fabrication instruction — was top-3 truncated to
  200 chars, which surfaced partial facts and let weaker models invent the rest.
- **Tool-arg coercion.** `ToolRegistry.execute` coerces LLM-supplied args toward each
  parameter's annotated type, so a tool no longer crashes when a model passes a list
  or string where an int/float/bool was expected.

### Docs
- New `docs/AGENT_DEV_GUIDE.md` — the golden path for building agents/packs.

## [2.5.0] - 2026-06-04

### Continual learning — agents that learn from runs and reuse memory

Agents now improve the second time they do a task instead of starting from scratch.

### Added
- **Recall-before**: `AitherAgent.chat()` searches the local `SkillStore` and injects the
  top matching learned skills as a system block, so the model reuses proven procedures.
- **Learn-after**: `_learn_after()` runs after each response — reinforces (`success_count++`)
  any skills that were recalled+reused, and extracts + saves a NEW skill from successful
  multi-tool runs via `SkillExtractor`. Wired at both `chat()` return paths.
- Gated by `AITHER_SKILLS` (default on). The previously-unused `SkillStore`/`SkillExtractor`
  are now connected to the run loop.

### Notes
- Pairs with the platform-side LearnedProcedure substrate (recipes → scored procedures →
  skills/tools/A2A/packs → MCTS priors → evolution). The SDK shares memory via the Spirit
  bridge when configured.

## [2.0.0] - 2026-06-02

### Open-core boundary (BREAKING)

This release draws a single, enforced free/paid line. The free tier stays
genuinely useful (real agent, typed + graph + code memory, ReAct, ~essential
tools); paid tiers unlock the *scale* capabilities via portal.aitherium.com.

### Added
- **`adk/licensing.py`** — the entitlement keystone. Resolves a tier
  (COMMUNITY by default), verifies optional portal-signed `~/.aither/license.json`
  (Ed25519), and answers `can_use_fleet/channels/cron/swarm/auto_neurons`,
  `is_agent_licensed`, `max_effort`, etc. **Fail-closed**: no/invalid/expired
  license → free tier; an unsigned license never grants premium.
  `AITHER_TENANT_SLUG=aitherium` → unrestricted INTERNAL tier.
- **Moat guard** (`scripts/check_moat_boundary.py`) wired into the publish
  workflow — blocks any wheel that leaks the moat or drops the keystone.
- Tests: `tests/test_licensing.py`, `tests/test_moat_boundary.py`.

### Changed (gating — fail-closed)
- Free tier effort is capped at 3 (reasoning effort 7–10 needs Professional).
- Monthly token budget hard-block on the free tier (no silent overspend).
- Gated behind a paid tier: fleet mode (`FleetConfig`), cross-agent delegation
  (`AgentForge.delegate`), channel adapters, cron scheduler, proactive
  auto-neurons, and `swarm`/`swarm_code` dispatch.

### Removed
- **`adk/nanogpt.py`** (on-device training IP) removed from the published SDK
  and the public repo; relocated to the internal AitherOS moat. Excluded from
  the wheel as defense-in-depth.
- Pre-2.0 releases shipped this IP and ungated capabilities and are being
  yanked (PyPI) / removed (GitHub releases). See `scripts/yank_leaky_releases.py`
  and `scripts/purge_public_leaks.sh`.

## [1.20.0] - 2026-05-27

### Fixed
- **Pack discovery path mismatch** — `_find_local_packs()` now scans both
  `~/.aither/packs/` and `~/.aitheros/packs/`, so packs installed via
  `adk pack install` are discovered by `adk-workspace`.

### Added
- **Auto-register on pack install** — `adk pack install` now auto-registers
  with the portal if the pack contains an `agent.yaml` with a `portal` section.
  One command does download + extract + register.
- **Flux event on agent registration** — `developer_portal.py` emits
  `agent.registered` Flux event so the portal fleet UI updates in real-time.

## [1.15.0] - 2026-05-26

### Added
- **Cloud memory** — `adk quickstart --cloud` now tests gateway memory and
  auto-configures Spirit URL so memories persist across devices via
  `gateway.aitherium.com/v1/memory/teach` and `/v1/memory/recall`.
- **Config→env export for cloud mode** — `Config.from_env()` now exports
  `AITHER_CLOUD_MODE`, `AITHER_SPIRIT_URL`, and spirit path env vars from
  saved config so Memory and other modules pick them up without needing
  explicit env vars set by the user.
- **Cloud Quick Start section** in README — 3-command setup for users who
  just want a Claude/GPT-4/DeepSeek API key and the full agent harness.
- **`--memory` profile for node deployment** — `adk deploy node --memory`
  adds Spirit (8087) + WorkingMemory (8101) containers for persistent
  vector memory. Without this, agents use local SQLite only.
- **Configurable vLLM in node compose** — `AITHER_VLLM_MODEL`,
  `AITHER_VLLM_GPU_UTIL`, `AITHER_VLLM_CTX_LEN`, `AITHER_VLLM_QUANT_ARGS`
  env vars let `adk setup` write GPU-detected tier settings to `.env`
  and have the node compose pick them up (TQ4 for small GPUs, BNB for large).

### Fixed
- **Memory cloud routing** — `Memory._spirit_teach()` and `_spirit_recall()`
  now use configurable paths and auth headers for gateway proxy. Previously
  cloud-only users silently lost all Spirit memory calls.

## [1.9.0] - 2026-05-24

### Added
- **Composable ToolPack system** — agents can load tool packs from
  `.toolpack.yaml` directories. Drop a pack in `~/.aitheros/packs/` or
  set `AITHER_PACKS_DIR`, and any agent discovers it automatically.
- **`tools.packs` in agent.yaml** — list tool pack IDs to auto-load
  (e.g. `packs: [crm-tools, calendar-pro]`).
- **`register_tool_packs(agent, pack_ids, packs_dir)`** in `builtin_tools` —
  programmatic pack loading for agents.
- **`AITHER_TOOL_PACKS` env var** — comma-separated pack IDs loaded on
  agent init (alternative to YAML config).
- **Unknown builtin categories resolve as pack IDs** — `tools.builtin:
  [file_io, crm-tools]` now works: `crm-tools` is looked up as a pack.

## [1.8.0] - 2026-05-24

### Removed
- **11 dead modules** (3,023 lines) — `agent_deploy`, `cloud_deploy`, `create`,
  `phonehome`, `sdk_bridge`, `sovereign`, `telemetry`, `telemetry_config`,
  `error_reporter`, `anonymizer`, `package`. None were imported anywhere.
- **29 ghost exports** from `__init__.py` — `secrets`, `otel`, `fs_sandbox`,
  and `vector_memory` modules were listed in `__all__` and `__getattr__` but
  the backing files never existed. Accessing them raised `ModuleNotFoundError`.
- **Stale dist/ artifacts** — removed old wheels/tarballs (v0.4.0–v1.4.1, 11 MB).
- **Orphan test scripts** — `test_phase1.py` and `test_phase23.py` (one-off
  validation scripts not part of the pytest suite).

### Fixed
- **Version mismatch** — `__init__.__version__` now matches `pyproject.toml`
  (was stuck at 1.6.0).

## [1.5.0] - 2026-05-23

### Added
- **`--sovereign` flag for `adk deploy node`** — registers the node with the
  Aitherium hub via federation after deployment. Saves federation credentials
  to `~/.aither/.env.federation`. Accepts `--hub` (custom hub URL) and
  `--tenant` (tenant slug) options. Non-fatal: node runs standalone if
  registration fails.
- **Federation → fleet cross-post** (Genesis) — sovereign nodes that register
  or heartbeat via `/federation/register` and `/federation/heartbeat` now
  automatically appear in the fleet dashboard (`/fleet/deployments`) with
  `node_type: "sovereign"` and live status/metrics.
- **"Deploy Sovereign Node" section on Connect page** (AitherVeil) — Section 8
  with ADK deploy commands, fleet dashboard links, and provision page links.
- **`deploy-node` action in tunnel proxy** (AitherVeil) — Connect page and
  external callers can now request bootstrap scripts via
  `POST /api/tunnel/devworkspace?action=deploy-node`.

### Fixed
- **Fleet provision page** — API URL corrected from `/api/fleet/provision` to
  `/api/bridge/genesis/fleet/provision` (matches bridge proxy pattern).
- **Fleet detail page** — API URL corrected from `/api/fleet/deployments/` to
  `/api/bridge/genesis/fleet/deployments/` (same fix).
- **Dockerfile** — entry point corrected from `aither-serve` to `adk-serve`
  (matches `pyproject.toml` script registration). Version label updated.

## [1.4.1] - 2026-05-22

### Fixed
- **macOS executable builds** — dropped `--target-arch universal2` from
  `packaging/build_executable.py`. PyPI wheels for `pydantic_core` and other
  native deps are arch-specific, not fat binaries, so universal2 always failed
  with `IncompatibleBinaryArchError`. Now builds for the runner's native arch
  (`aither-macos-arm64`).
- **Release workflow strategy** — added `fail-fast: false` so a single
  executable build failure no longer cancels the other OS jobs.
- **PyPI publish** — added `skip-existing: true` so re-runs of the same
  version don't fail when the artifact already exists.
- **GitHub Release job** — now runs whenever `pypi` succeeds even if some
  executable matrix entries fail, so the release page is always created with
  whatever binaries did build successfully.

## [1.4.0] - 2026-05-22

### Added
- **Sovereign template now ships with `agent_core` 5-tier memory** — generated
  backends include the canonical `agent_core/` package (working → episodic →
  semantic → procedural → identity) plus `agent_memory.py` helper, vendored
  from `portal-kit-backend`.
- `aiosqlite>=0.20.0` added to sovereign template `requirements.txt` for
  local agent_core SQLite storage.
- Document tombstoning: `evict_cached_conversations_referencing` is wired
  through document-delete handlers in consumer apps so memories citing
  deleted documents are blocked from recall.
- New tests: 13 smoke tests for `portal-kit-backend/agent_core` covering
  promotion thresholds, decay rates, store roundtrip, tombstone recall
  blocking, circuit breaker state machine, and chat-turn recording.

### Changed
- `scripts/vendor_agent_core.py` now syncs canonical `agent_core` to 4
  targets: WorkspaceRuntime, ADK sovereign template, and the GargBot +
  Chelle consumer mirrors (previously only 2).

## [1.2.1] - 2026-05-21

### Added
- `adk train` command group — 6 subcommands: status, launch, logs, cancel, runs, register-gpu
- `/slash-commands` endpoint — ADK server auto-exposes all 35 CLI commands as structured JSON manifest for AitherShell
- `/cli/execute` endpoint — AitherShell can run any CLI command through the server
- `build_command_manifest()` — introspects full argparse tree (commands, args, choices, subcommands)
- `_register_commands()` extracted from main() for shared parser construction

### Changed
- `_get_genesis_url()` now probes: Genesis → ADK server → Aitherium cloud gateway (works without Genesis)

## [1.2.0] - 2026-05-21

### Added
- **Runtime backend switching** — `agent.switch_backend("deepseek")` and `agent.llm.switch_backend()` change provider without recreating agent
- **Hybrid reasoning** — `agent.set_reasoning_backend("anthropic")` routes effort 7+ to cloud API while keeping local orchestrator
- **First-class DeepSeek provider** — `deepseek` in effort model table with `deepseek-chat` and `deepseek-reasoner`
- **TQ4 quantization tiers** — `nano` (6GB, TurboQuant 4-bit), `standard-tq4` (12GB, both models TQ4), `hybrid-tq4` (6GB + cloud reasoning)
- **`adk backend` CLI** — `list`, `set`, `set-reasoning`, `test` subcommands
- **`adk quickstart`** — unified first-run wizard (setup + auth + shell)
- **`adk tools`** — list local + MCP tools with tier markers
- **`adk backup`** — tarball export of `~/.aither/`
- **`adk ingest`** — feed docs/files into knowledge graph
- **`adk setup nemotron`** shortcut — alias for `--tier lite`
- **`--reasoning-api`** flag in setup — `anthropic`, `openai`, `deepseek`, `gateway`
- **`--dgx-spark URL`** flag — remote vLLM endpoint configuration
- **DGX Spark auto-detection** — scans `spark.local`, `192.168.0.33`, `AITHER_DGX_URL` env
- **Post-setup smoke test** — sends real inference request after container startup
- **Schema migration system** — version table + migration runner in `memory.py` and `graph_memory.py`
- **Configurable vLLM port scan** — `AITHER_VLLM_PORTS` env var for custom ports
- **`aithershell` entry point** — alias for `adk` in pyproject.toml
- Shell auto-downloads binary on first `adk shell` (no `--install` needed)
- Shell pre-flight check warns if no backend detected
- Shell passes full config via env vars (API key, tenant, inference URL)
- DGX/Remote and Cloud API checks added to `adk doctor`
- Config fields: `reasoning_backend`, `reasoning_api_key`, `deepseek_api_key`, `dgx_url`, `vllm_extra_ports`

### Fixed
- `test_aeon_chat_basic` timeout — mock scope was exiting before request
- `adk doctor` check_disk timeout — cap rglob at 5000 files

## [1.1.7] - 2026-05-19

### Added
- Remote agent pipeline — `adk onboard` → fleet dispatch → remote inference
- Agent-to-agent federation with mesh relay
- `adk connect` for desktop mesh enrollment

## [1.1.6] - 2026-05-19

### Fixed
- Tool call execution recovery across all backends
- Hermes tool call fallback parsing for vLLM

## [1.1.5] - 2026-05-19

### Fixed
- Tool call execution + Hermes fallback parsing improvements

## [1.1.0] - 2026-05-18

### Added
- Multi-channel gateway (Telegram, Discord, Slack, webhook)
- Aeon group chat with 7 presets
- Skills auto-extraction from multi-step sessions
- MCP stdio server for Claude Code integration
- `adk gateway`, `adk cron`, `adk skills`, `adk soul` commands
- SOUL.md import/export for portable agent identity
- Elysium cloud device-flow authentication

## [1.0.0] - 2026-05-18

### Breaking
- Package consolidated — `aither_adk` namespace replaced by `adk` + `adk.platform`
- 93 duplicate files removed from `AitherOS/packages/aither_adk/`

### Added
- Standalone core primitives (`adk.core`: Agent, Tool, Memory, ModelBackend, Capability, Trace)
- `adk.platform` sub-package merging all platform toolkit modules
- Compatibility shims for legacy `from aither_adk.*` imports

### Changed
- Version synced across pyproject.toml, `adk/__init__.py`, `adk/platform/__init__.py`, npm package
- Classifier promoted from Alpha to Beta

## [0.16.0] - 2026-04-16

### Swarm Coding Engine & Repowise Integration
- **`repowise_search` tool** — Semantic + keyword hybrid code search via Repowise, with ripgrep fallback
- **`swarm_code` tool** — Dispatch complex tasks to 11-agent swarm pipeline (ARCHITECT->SWARM->REVIEW->JUDGE)
- **`agent.swarm()`** — Async convenience method with configurable mode, effort, timeout
- **`agent.code_search()`** — Async convenience method returning structured results
- New tool categories: `repowise` and `swarm` in TOOL_CATEGORIES
- `repowise_search` and `swarm_code` added to `__init__.py` exports with lazy loading
- `IDENTITY_DEFAULTS` updated: repowise for code-focused agents, swarm for orchestration agents
- Standalone graceful degradation: repowise falls back to ripgrep, swarm returns structured error

## [0.13.0] - 2026-04-02

### Graph Faculties — Local Knowledge for Every Agent
- **CodeGraph** (2,799 lines) — Full Python AST indexer with call graph, keyword/semantic/hybrid query, embedding matrix cache, incremental re-indexing, multi-hop chain expansion
- **MemoryGraph** (1,339 lines) — Graph-based persistent agent memory with 10 edge types, hybrid query (keyword + semantic + graph expansion), multi-hop recall, pickle persistence
- **EmbeddingProvider** — 4-backend fallback chain: sentence-transformers (GPU/CPU) -> Ollama -> Elysium cloud -> feature hashing (zero deps)
- **BaseFacultyGraph** — Abstract base with HMAC-SHA256 validated pickle persistence

### Agent Integration
- `agent.set_code_graph(cg)` — auto-registers `code_search` + `code_context` tools
- `agent.set_memory_graph(mg)` — auto-registers `remember` + `recall` + `memory_stats` tools
- Both graphs inject context into LLM prompts automatically during chat

### Zero-Config Onboarding
- `adk start` / `adk` (no args) — auto-detect project, index code, detect LLM, persistent memory, interactive chat
- Works for any directory: Python codebases, doc folders, mixed workspaces
- Auto-detects LLM: Ollama -> vLLM -> Elysium -> OpenAI -> Anthropic
- Per-project persistent memory in `~/.aither/memory/<project>`
- `adk index <path>` — standalone indexing with progress bar and stats

### MCP Gateway
- `POST /v1/embeddings` — OpenAI-compatible embedding proxy to vLLM-embeddings:8209
- Available to all tiers (embeddings are free)
- ADK EmbeddingProvider uses this as Elysium cloud fallback

### Optional Dependencies
- `pip install aither-adk[graphs]` — numpy for 10x cosine similarity speedup
- `pip install aither-adk[embedding]` — sentence-transformers + torch for local GPU embeddings
- `pip install aither-adk` alone — graphs work with feature hashing (zero deps)

## [0.12.0] - 2026-04-01

### Bootstrap & Service Discovery
- Version handshake -- ADK checks major.minor compatibility with Genesis/Node
- Background reconnect loop -- ServiceBridge re-probes every 30s in standalone mode
- Port 8080/8090 documented -- MCP vs OpenAI-compat clearly separated
- Genesis URL configurable via GENESIS_URL env var (was hardcoded)
- Standalone mode warning -- visible stderr alert when AitherOS not detected
- Auto-reconnect on startup when services come online

### Elysium Cloud Inference
- Unified gateway URL -- gateway.aitherium.com handles auth + billing + inference
- Streaming inference -- SSE passthrough for /v1/chat/completions with stream=true
- Auth proxy routes -- /v1/auth/register, /v1/auth/login, /v1/auth/me
- Billing proxy -- /v1/billing/balance through gateway
- AitherConnect Elysium fallback -- cloud inference when local Genesis is down
- AitherDesktop Elysium fallback -- third-tier chat fallback after Node

### Infrastructure
- /discovery endpoint on Genesis -- unified service URLs/versions/health
- /api/config/services on Veil -- runtime port config for client-side JS
- Veil healthcheck gates on Genesis -- unhealthy when backend is down
- Desktop crash detection 90s->60s (threshold 3->2)

## [0.11.0] - 2026-04-01

### Agent Execution Quality (Claude Code Parity)
- Raise loop guard block threshold from 3 to 4 -- agents get more room for iterative search
- Soft synthesis nudge for effort >= 4 (no tool stripping, trust the model)
- max_output_tokens escalation -- retry up to 3x when response is truncated
- Tool result pairing guarantee -- synthetic error for orphaned tool_use blocks
- Micro-compaction of old tool results -- save context tokens on long sessions
- First-turn tool forcing only for effort >= 6 (trust model for lower effort)
- Diminishing returns detection -- nudge agent when 3+ turns produce < 500 tokens
- Message normalization -- merge consecutive same-role messages, strip empties
- LLM retry with exponential backoff (5 retries, 500ms-16s, jitter)

## [0.9.0] - 2026-03-16

The "connected world" release. Cross-platform identity pairing, voice capabilities, and multi-channel integration.

### Added
- **Pairing**: Cross-platform identity linking (`adk/pairing.py`)
  - `PairingManager` — SQLite-backed identity linking with 6-char pairing codes
  - Link users across Telegram, Discord, Slack, WhatsApp with 10min TTL codes
  - Canonical session IDs for cross-channel conversation continuity
  - `get_session_id()` returns "user-{id}" for paired users
- **Voice**: Speech-to-text and text-to-speech client (`adk/voice.py`)
  - `VoiceClient` — async STT/TTS/emotion via AitherVoice service
  - Convenience functions: `hear()`, `say()`, `feel()`
  - 6 voice options: alloy, echo, fable, nova, onyx, shimmer
  - Emotion detection with intensity scoring
- New exports: `PairingManager`, `PairingResult`, `PlatformIdentity`, `VoiceClient`, `TranscriptionResult`, `SynthesisResult`, `EmotionResult`

### Changed
- `__init__.py` exports expanded with pairing and voice symbols

## [0.6.0] - 2026-03-13

The "group mind" release. Multi-agent group chat, creative tools, and Iris identity.

### Added
- **Aeon**: Multi-agent group chat engine (`adk/aeon.py`)
  - `AeonSession` — persistent group chat with parallel agent execution
  - 7 presets: balanced, creative, technical, security, minimal, duo_code, research
  - Orchestrator synthesis: Aither summarizes all agent responses
  - Serial execution for Ollama, parallel for vLLM/cloud
  - ConversationStore persistence with `type: "aeon"` metadata
  - `group_chat()` one-shot convenience function
- `aither aeon` CLI command — interactive terminal group chat with color-coded agents
  - `-p/--preset`, `-a/--agents`, `-r/--rounds`, `--no-synthesize` flags
  - `reset` and `quit` commands
- Server endpoints: `POST /aeon/chat`, `GET /aeon/presets`, `GET /aeon/sessions/{id}`
- Creative tools in builtin_tools: `image_generate`, `image_refine`, `image_search`, `video_generate`
- Iris agent identity with visual generation capabilities
- 48 Aeon tests (data models, presets, chat, context, persistence, server, exports)

### Changed
- `__init__.py` exports: `AeonSession`, `AeonResponse`, `AeonMessage`, `group_chat`, `AEON_PRESETS`

## [0.5.0] - 2026-03-13

The "tenant-ready" release. Multi-tenant admin, permission grants, safety profiles, and a full setup wizard.

### Added
- `aither setup` interactive setup wizard with hardware detection, model selection, identity config
- Strata storage backend: SQLite WAL persistence for conversations, memories, knowledge graphs
- CLI test runner: `aither test` with auto-discovery and parallel execution
- Permission grants system for MCP tool access control
- MCP account management tools
- LLM provider auto-detection for Ollama and vLLM endpoints
- Apache-2.0 LICENSE file
- Elysium desktop sync module
- Comprehensive CLI test suite (493 tests)
- Strata storage test suite (972 tests)
- LLM provider tests (82 tests)

### Changed
- CLI expanded: `aither setup`, `aither test`, `aither bugreport`, `aither doctor`
- Strata module rewritten as full local-first storage engine
- Server startup includes Strata initialization
- README rewritten with clearer quickstart and architecture docs

### Fixed
- Dead documentation links
- Python 3.11 compatibility issues
- Ruff per-file-ignores configuration
- Identity provisioning edge cases
- Docker node-gyp native dependency builds (python3 + build tools)

## [0.4.0] - 2026-03-13

The "own your AI" release. Self-hosted agent OS for people who don't want their data on someone else's servers.

### Added
- Muse agent identity (creative/artistic generation)
- Port 8120 to vLLM scan for ExoNodes discovery
- Public roadmap with milestone-based porting schedule
- Competitive positioning: self-hosted alternative to cloud-locked AI appliances

### Changed
- LLM router: compute-aware effort routing (replaces effort-level context gating)
- README rewritten for sovereignty-first messaging
- Package description updated
- Promoted from alpha to stable release

### Fixed
- Version string consistency between pyproject.toml and __init__.py
- vLLM port scanning now includes port 8120 (ExoNodes)

## [0.3.1] - 2026-03-09

### Added
- GraphMemory: SQLite knowledge graph with Ollama embeddings and hybrid search
- NeuronPool: auto-fire pre-LLM data gathering agents
- NanoGPT: pure Python char-level transformer with LoRA fine-tuning
- Safety gates: IntakeGuard (input), LoopGuard (recursion), Sandbox (code exec)
- Event system: EventEmitter with chat/tool/forge event types
- Builtin tools: identity-based tool selection
- ServiceBridge: auto-discovery of AitherOS services
- Streaming: chat_stream with safety gate integration
- Auth middleware: Bearer token authentication
- CLI: `aither init` and `aither serve` commands

### Changed
- Wired 5 previously disconnected modules into agent loop
- 522 total passing tests (up from 85)

## [0.3.0] - 2026-03-07

### Added
- Clean-room agent development kit
- Multi-backend LLM providers (Ollama, OpenAI, Anthropic)
- AitherAgent class with @tool decorator
- SQLite conversation memory
- OpenAI-compatible server (`aither-serve`)
- 16 agent identities as package data
- MCP bridge to mcp.aitherium.com
- Privacy-centric opt-in telemetry
- Bug reporting CLI and API
- FastAPI server with OpenAI-compatible endpoints
- Hardware auto-detection (5 tiers, 11 profiles)
- Fleet orchestration and multi-agent coordination
- A2A mesh protocol
- Federation protocol for cross-instance agent dispatch

### Initial release
- 85 passing tests
- Apache-2.0 license
