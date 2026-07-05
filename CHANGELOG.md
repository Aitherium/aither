# Changelog

All notable changes to aither-adk will be documented in this file.

## [2.15.0] - 2026-07-04

### Added — built-in web admin console + portal-profile settings sync

- **Web admin console.** The page `adk up` opens (served at `/` and `/ui`) is now a full,
  self-contained console — Chat plus six admin tabs: **Backend** (view/switch LLM provider,
  set keys, test connection), **Packs** (list/enable/disable/reload tool packs),
  **Sessions** (browse/delete conversations), **Logs** (redacted agent-log tail),
  **Graph** (knowledge-graph search/neighborhood/stats), and **MCP** (add/remove external
  MCP servers). No build step, no CDN — a single packaged HTML asset. The original minimal
  streaming chat page stays at `/chat` for back-compat.
- **Admin API (`adk/admin_api.py`).** ~20 endpoints under `/admin/*`, all bearer-gated by the
  existing middleware (never in `_skip_auth_paths`). Operates on the live agent — LLM backend
  swap (`LLMRouter.switch_backend`), pack reload, and MCP registration take effect without a
  restart.
- **Settings sync (`adk/settings_sync.py`).** The user's portal profile is the source of
  truth: on startup the agent pulls `preferences.adk` and applies it over the local
  `~/.aither/config.yaml` cache; admin-console edits push a debounced snapshot back to
  `PUT /api/settings/preferences`. Provider API keys and MCP auth headers are device-local
  secrets and are never included in the pushed snapshot. Opt out with `AITHER_SETTINGS_SYNC=false`.

### Security

- **Timing-safe token comparison.** The server bearer check now uses `hmac.compare_digest`
  (the token is reachable over the public tunnel; a byte-wise `!=` leaked it to timing attacks).
- **MCP add is SSRF-guarded + confirmed.** Adding an external MCP server is a two-step
  prepare→confirm flow; URLs resolving to loopback / private / link-local / metadata addresses
  are rejected (override with `AITHER_ALLOW_PRIVATE_MCP=1` for trusted local servers).
- **Config editor is allowlisted.** `PATCH /admin/config` writes only safe scalar prefs and
  hard-denies fields that could execute code or redirect traffic at startup (e.g.
  `aither_toolpack_dirs`, backend base URLs).
- **Log tail is scrubbed.** Bearer tokens, `#k=` fragments, and `sk-`/`aither_sk_` keys are
  redacted from the log-tail endpoint; the tunnel log is excluded.

## [2.14.8] - 2026-07-04

### Added — `adk up` for normal humans (no API key, no prereqs, live chat)

- **Hosted-brain default — no API key required.** When there's no local model and no
  provider key, `adk up` signs in (device-flow) and routes inference through the gateway
  using the portal token (`AITHER_LLM_BACKEND=gateway` + bearer). A non-technical user
  never needs to obtain or paste an LLM API key. Explicit `--provider`/local model still
  take precedence; unattended `--yes` with no token/backend fails cleanly (exit 3).
- **cloudflared auto-download.** If the tunnel binary is missing, `adk up` downloads the
  right platform asset to `~/.aither/bin` (SHA256-verified against the release; opt out with
  `AITHER_CLOUDFLARED_REQUIRE_CHECKSUM=0`) instead of exiting with an install hint.
- **Built-in streaming chat page.** The agent now serves a self-contained chat UI at `/`
  (and `/chat`) with a live "thinking…" indicator and token-by-token streaming — so a human
  has somewhere to talk to it with immediate feedback instead of a blank wait. `adk up` opens
  it locally and passes the callback bearer in the URL fragment (`#k=…`), so `/chat/stream`
  stays authenticated (not an open proxy) with nothing to paste.
- **Human-readable failures.** `adk up` errors now carry a plain-English `next_action`
  (kept machine-readable in `--yes` JSON).

### Fixed

- `adk up` chat link pointed at a non-existent `/settings/agent` route; it now points at the
  agent's own streaming chat page.

## [2.14.7] - 2026-07-04

### Added

- **`adk up` — one command to run a persistent, fleet-connected agent.** Collapses
  the scattered onboarding surface (`host`/`quickstart`/`connect`/`login`/`setup`)
  into a single command that starts `aither-serve` (single agent, identity `aither`
  by default), opens a Cloudflare quick-tunnel for fleet reachability, registers with
  the portal, and installs a reboot-persistent autostart entry (Windows Task Scheduler
  / systemd-user / launchd). Detaches by default so the terminal is freed; `--foreground`
  blocks. Reuses the existing device-flow login, cloudflared discovery, provider-key
  store, and `_preflight_check` backend detection.
- **Fully non-interactive path for autonomous agents.** `adk up --yes` (or any
  non-TTY invocation) takes ZERO prompts — everything resolves from flags, env, and
  saved config — and emits a single machine-readable JSON status line with meaningful
  exit codes (`0` ok, `2` bad-input, `3` no-backend, `4` tunnel-missing, `5` register-failed).
  Degrades to local-only (warn, not fail) when a portal token is absent unless
  `--require-register` is set.
- **`adk down` / `adk status --json` companions.** `down` stops the agent + tunnel,
  best-effort deregisters, and removes the autostart entry; `status` reports the running
  agent (liveness + `/health` + tunnel + registration), with `--json` for agents/CI.
- **`adk stack`** — the previous `adk up` behaviour (supervise the Room + Ollama consumer
  stack) now lives here; `adk up` is the connected-agent command.

### Internal

- New `adk/agent_daemon.py`: detached process spawn, status file
  (`~/.aither/adk-up.json`), pid liveness/teardown, and cross-platform autostart install.

## [2.14.6] - 2026-07-04

### Added

- **`adk.preflight` — self-bootstrap capability probe.** `run_preflight(agent,
  spec)` wakes an agent up knowing what it is actually plugged into: a bounded
  (1.5s) parallel `LivenessProbe` over the router's resolved backends
  (primary/reasoning), the embeddings winner, remote MCP tools, the structured-ML
  / `/ml/teach` endpoints, and voice — returning an honest `CapabilityReport`.
  Status vocabulary distinguishes reachability from entitlement
  (`OK`/`MISSING`/`UNREACHABLE`/`AUTH`/`TIER_DENIED`/`UNSUPPORTED`), a hosted slot
  never prints a bare `OK`, and a required-but-unsatisfied slot aborts before the
  loop instead of blind-firing. Headless task/rules resolution + an allowed-roots
  policy binding (`builtin_tools.set_allowed_roots`, which resets the memoized
  cache so the override actually enforces).
- **Multimodal (image) messages.** `Message.content` now accepts an OpenAI-style
  content-part list (`str | list[dict]`), passed through verbatim to the
  OpenAI-compatible body (gateway / vLLM / OpenAI). New `adk.llm.multimodal`
  helpers — `image_message()`, `image_content_parts()`, `to_image_url()` — build
  the parts and encode local image bytes/paths into `data:` URIs.
- **Real vision preflight probe.** The vision slot is no longer unconditionally
  `UNSUPPORTED`: it sends a discriminable image (a solid red square) and requires
  the model to name the color, so a blind/text-only model cannot false-pass. `OK`
  only on a real round-trip; `AUTH` on 401/403; `UNSUPPORTED` when there is no
  OpenAI-compatible provider or the model cannot see the image.

## [2.14.5] - 2026-07-04

### Added

- **`adk.reasoning.mcts` — generic Monte-Carlo Tree Search library.** A reusable
  MCTS engine (ported from the internal `UnifiedMCTS`, with all internal couplings
  removed) that any agent can drive over its own environment:
  - `MCTSEnvironment` protocol (`get_state_hash` / `get_actions` / `step` /
    `evaluate` / `clone`) — implement it and search.
  - Three **optional, default-off** model seams — `TransitionModel`, `PolicyModel`
    (PUCT priors), `ValueModel` — so a learned world-model / policy / value can be
    injected without touching the core; with all three `None` the engine is a plain
    UCT search.
  - `ObservedTransitionModel` adapter — a cheap online lookup world-model (hit =
    exact stored transition; miss = identity + negative "unknown" reward +
    `is_uncertain`, biasing exploration toward learning), with JSONL save/load for
    cross-run reuse.
  - `UnifiedMCTS(config).search(env)` returns an `MCTSResult` (best action, value,
    confidence, principal-variation path) and an optional `MCTSTrace`
    (visit-distribution policy target) for distillation.
- **Self-bootstrapping agent machinery.** Build an agent from a declarative spec
  instead of bespoke wiring:
  - `adk.bootstrap.build_agent_from_spec(path) -> (AitherAgent, RunCtx)` — resolves
    prompts, activates required/optional tool packs, and carries a `memory_map`
    (a `node_type -> (role, tier)` override for memory writes, never a gate).
  - `adk.prompts.PromptBridge` — resolves an agent's prompt bindings from files
    (relative to the spec) or `AitherPrompts` dot-keys (optional import).
  - `adk.packs.PackActivator` — eager activation of installed tool packs by name
    over `register_tool_packs`; a required pack that registers zero tools raises
    `PackUnavailable`, optional packs soft-degrade.
  - `python -m adk.bootstrap.generic_agent <spec.yaml>` inspects a spec (resolved
    name, activated packs + tool counts, prompt keys, memory map) without running a
    live loop.
- **`deep_research` tool pack** — a full multi-engine web-research workflow as
  `dr_*` agent tools (keyless search, page fetch + cache, site mirror, a citable
  findings knowledge graph, and markdown/PDF/DOCX report generation), activatable
  by name via the tool-pack loader.

## [2.14.4] - 2026-07-04

### Added

- **`structured_ml` tool domain** — zero-shot structured-data inference for agents,
  backed by TabFM (tabular classification/regression) and TimesFM (time-series
  forecasting). Three tools:
  - `tabular_classify(support_rows, target, query_rows)` — classify rows from a
    labeled support set, in-context (no training step; up to 10 classes).
  - `tabular_regress(support_rows, target, query_rows)` — predict a numeric target.
  - `timeseries_forecast(series, horizon)` — forecast future values of a series.

  The tools POST to a structured-ML inference service resolved from
  `AITHER_STRUCTURED_ML_URL` (default `http://localhost:8192`), reject non-http(s)
  URLs, and summarise oversized responses to protect the agent's context.
- **`Capability.STRUCTURED_INFERENCE`** capability token and a new **`analyst`**
  identity/pack that ships the `structured_ml` domain by default. The domain is
  opt-in — only identities that enable the `structured_ml` category get the tools.

## [2.14.3] - 2026-07-03

### Fixed

All four bugs below were found by actually booting `adk run` against a real
chat model and driving a real tool-calling request end-to-end (not inferred
from reading the code) — each one silently broke the "local agent uses its
configured model and its tools actually work" path.

- **`AITHER_MODEL` / `config.model` silently ignored by the OpenAI-compatible
  backend**: `LLMRouter.__init__` never read `config.model` into `self._model`
  unless the caller passed `model=` directly to the constructor — which nothing
  did. Every explicit model choice (env var, config file) was dropped, the
  provider fell back to a hardcoded per-backend default (`gpt-4o-mini` for the
  `openai` backend), and an unrecognized model name got silently remapped
  server-side to the wrong backing model. Concretely: setting
  `AITHER_MODEL=aither-orchestrator` had no effect — every request actually
  went to a different model that was never meant to be called directly and
  can't reliably use tools.
- **`OPENAI_BASE_URL` ignored for explicit `--backend`/`AITHER_LLM_BACKEND`
  selection**: `LLMRouter._auto_detect()`'s explicit-backend branch hardcoded
  `base_url=None` when constructing the provider, instead of reading
  `config.openai_base_url` (which two other call sites in the same file
  already did correctly). Any self-hosted OpenAI-compatible endpoint
  (MicroScheduler, vLLM, LM Studio) configured via `OPENAI_BASE_URL` was
  silently ignored in favor of `https://api.openai.com/v1`.
- **Tool-call steering never engaged on default-effort chat**: the retry that
  nudges a model back into structured tool-calling when it responds with prose
  instead of a function call was gated behind `effort >= 6`. Default `effort`
  for a plain `agent.chat(message)` call resolves to `5`, so the safety net —
  built specifically for this failure mode — never fired for default requests.
  Also now forces `tool_choice="required"` on the retry instead of relying on a
  text nudge alone.
- **Every non-string builtin-tool parameter silently typed as `"string"` in
  the generated tool schema**: `builtin_tools.py` uses
  `from __future__ import annotations`, so `_extract_parameters()` in
  `tools.py` was reading postponed (string) type hints (`"int"`, not `int`)
  straight out of `fn.__annotations__`, which never matched
  `_type_to_schema`'s type-object keys and silently fell through to
  `{"type": "string"}`. Every builtin tool with an `int`/`float`/`bool`/`list`
  parameter was affected — e.g. `file_read(start_line: int, end_line: int)`
  advertised both as strings, so a model correctly calling the tool with
  `"start_line": "0"` crashed the actual implementation with
  `unsupported operand type(s) for -: 'str' and 'int'`. Fixed by resolving
  hints through `typing.get_type_hints()` instead of the raw annotations dict.
  `_coerce_arguments()` — the runtime safety net that's supposed to coerce a
  model-supplied `"0"` string into `0` before a tool call executes — had the
  exact same raw-`__annotations__` bug and silently no-opped for every builtin
  tool; fixed the same way.
- **Streaming tool-loop could hang a connection forever**: `chat_stream()`'s
  fallback to the sync tool loop (tool calls can't stream token-by-token) had
  no timeout, so a model that didn't cleanly terminate a tool-calling round
  trip left the SSE connection open indefinitely with zero bytes sent. Bounded
  with a 60s `asyncio.wait_for`; on timeout the stream yields a clear message
  instead of hanging.

## [2.14.2] - 2026-07-03

### Fixed
- **AitherShell session amnesia**: the REPL generated a fresh `session_id` for
  EVERY message, so conversation history never followed across turns (the
  server stored each turn under a different session). One stable id is now
  minted per shell run; `/resume` and `/new` semantics unchanged, and the
  cross-restart auto-restore of the last session now actually works.

## [2.14.1] - 2026-07-02

### Added
- **Account-tied licensing via device-flow login** — `adk login` now persists the account
  license returned by the device-flow handshake to `~/.aither/license.json` (base64
  `license_key` → `{payload, signature}` envelope), so entitlements from the authenticated
  account carry into every later `adk` invocation without re-auth. Pairs with the fleet-side
  node tool-pack gate that grants packs by plan-tier rank.
- **`can_use_formbridge` / `can_use_untether` entitlements** (PROFESSIONAL+ rank ≥ 3) backing
  the FormBridge and UNTETHER agent tool-packs.

### Fixed
- `adk` CLI: guard the optional `memory_graph` singleton (was a latent `NameError` on the
  `/stats`, `/memory`, and save paths when memory was not constructed).

## [2.14.0] - 2026-07-02

### Added
- **`adk.memory_wiki` — LyraWiki-style self-managed semantic knowledge** (opt-in; nothing constructs it by default): `MemoryWiki(graph_memory)` maintains curated wiki ARTICLES as `wiki_article` nodes (title/slug, LLM-owned markdown with `[[wikilinks]]`, embedding, durable tier, `CONSOLIDATED_FROM` edges to source memories, governance-ledgered revisions).
  - `consolidate(llm, since=, budget=)` clusters unconsolidated episodic/fact nodes (embedding + keyword), drafts/UPDATEs articles via the INJECTED llm callable (`str -> str` or `messages -> str`, sync or async — no hard model dependency), cites source node ids, marks contradicted sources superseded, then DEMOTES consolidated sources to a fast-decay tier via `promote()` (represented knowledge doesn't need raw copies).
  - `lint(llm=None)` — deterministic health checks (orphan wikilinks, empty/stale articles, live-but-contradicted pairs) + an optional LLM audit pass.
  - `prune(relevance_floor=0.05, hard_delete_after_days=14)` — article relevance = tier freshness × reinforcement (reinforce-on-recall) + link-degree to LIVE memories; below the floor → reversible governance tombstone; entombed past the window → **HARD-DELETE** (the tombstone snapshot itself is purged — content and embedding irrecoverable; the true-deletion path). A node is never deleted without a tombstone first.
  - `recall(query, limit)` — article-first RAG (curated articles rank above raw nodes) that reinforces returned articles.
- **`adk.routines` — agent heartbeat + self-programmed routines**: `RoutineStore` is a durable registry (`~/.aither/routines/{agent}.json`, `AITHER_DATA_DIR`-aware, injectable path) of cron-scheduled self-prompts `{name, cron, instruction, enabled, last_run, last_result, tags}` that rehydrates into the existing `adk.cron.CronScheduler` on `start()`. A fire runs `agent.chat(instruction)` with the agent's full toolset — the agent programs its future self with its own adk. Guardrails: `max_routines` (12), a 5-minute min fire interval per routine, a per-fire timeout, and every result ledgered (truncated) to `last_result`. Direct-callable routines (`register_direct`) fire a bound method instead of a self-prompt. Self-management tools (`routine_create/list/update/pause/resume/delete/run_now`) ship as OpenAI tool defs + handlers (`build_routine_tools` / `register_routine_tools` / `routine_tool_defs`); the handlers ONLY touch the RoutineStore — they can never modify agent config or safety-relevant settings (the leash principle).
- **`AitherAgent(routines=True, memory_maintenance=True)`** (both default **False** → the default agent is byte-identical): `routines=True` builds the RoutineStore + scheduler, registers the self-management tools, and starts the heartbeat lazily on the first `chat()` (or via explicit `start_routines()`). `memory_maintenance=True` registers DEFAULT routines — `wiki_consolidate` (every 2 h), `wiki_lint` / `wiki_prune` / `graph_sweep` (daily) — as DIRECT method fires so memory upkeep runs even on tiny models, while staying visible/manageable via `routine_list`.
- `TombstoneStore.purge(tombstone_id)` — the hard-delete endpoint: irrecoverably removes a tombstone snapshot and rewrites the persisted store without it (consumed by `memory_wiki.prune`).

## [2.13.6] - 2026-07-02

### Added
- **Qdrant backend for GraphMemory dataplane sync — TRUE two-way, verified live.** Set `AITHER_FLEET_QDRANT_URL` and the sync uses Qdrant (`upsert` + `scroll` + filtered `search`) instead of Nexus. This is what actually delivers cross-agent memory sharing: a node is upserted with its OWN embedding under a deterministic per-(tenant,node) point id (idempotent re-push), and `fleet_pull` reliably SCROLLS a tenant's points back — the enumerate Nexus lacks (its `/search` doesn't return ingested docs and `/export` needs lancedb, which has a hard dependency conflict in the fleet). **Proven live with 2 agents:** agent-1 ingests → auto-syncs to the tenant dataplane; a fresh agent-2 in the same tenant `fleet_pull`s all of agent-1's nodes and can search them; a different tenant pulls nothing (isolation). Dimension-safe (never mixes vector dims in a collection). Local SQLite stays source of truth; best-effort, never raises.

## [2.13.5] - 2026-07-02

### Fixed
- **CRITICAL — GraphMemory dataplane sync push was 100% failing.** `_fleet_push_node` sent `source_type="graph"` / `content_type="graph_node"`, which AitherNexus rejects as invalid enums (HTTP 500) — so NO graph node ever replicated in 2.13.4. Now sends the accepted `source_type="manual"` / `content_type="text"` (graph provenance is carried in `metadata.synced_from`). **Verified live** against a fleet Nexus: a 4-node ingest replicates with `pending==0` and the collection count reflects it.

### Added
- **Swarm-awareness notification** — on a successful sync batch, `GraphMemory` emits a best-effort `graph.synced` event (tenant/workspace/agent/collection/count) to `AITHER_FLEET_EVENTS_URL` so OTHER agents in the tenant/swarm can pull the fresh data and deconflict in-flight work (push-based awareness instead of polling). No-op unless the env is set.

### Known limitation
- Cross-agent **rehydration** (`fleet_pull`) is only reliable when the fleet Nexus is backed by lancedb (persistent + exportable). An in-memory Nexus stores on `/ingest` but neither its `/search` nor `/export` returns the docs — two-way read needs the persistent Nexus.

## [2.13.4] - 2026-07-02

### Added
- **Memory-maintenance primitives on `GraphMemory`** (all opt-in; default behaviour byte-identical):
  - **Reinforce-on-recall**: `recall_with_activation(..., reinforce=True)` bumps `reinforcement_count`/`last_reinforced` on every RETURNED node's metadata mirror (one UPDATE per node, WAL-safe) — making the `MemoryRecord` reinforcement fields the activation scorer reads LIVE. Default `False` = zero writes.
  - **`sweep(now, archive_below=0.05, max_nodes=None, dry_run=False)`**: read-time decay enforced at rest — archives nodes past their tier TTL with freshness×reinforcement score below `archive_below` and `reinforcement_count <= 1`; `max_nodes` additionally archives the lowest-scored overflow. PERMANENT tier immune. Every archive is a reversible governance tombstone (`TombstoneStore.recover`) + a FORGET ledger entry; a node is never deleted without a tombstone. `dry_run` returns the would-archive list without acting.
  - **`promote(node_id, tier=?, role=?)`**: re-tier/re-role a node IN PLACE (tier/role columns + metadata mirror) — edges, embedding and ledger history survive; ledgers an UPDATE entry when governance is on.
- `MutationType.UPDATE` ledger entry kind (in-place re-tier/re-role).
- **Per-tenant dataplane sync for `GraphMemory`** — replicate the local graph to a per-tenant vector dataplane (Nexus/Qdrant) so a sovereign agent's memory is durable + rehydratable across restarts and hosting boundaries. `fleet_push_all_nodes()` (catch-up), `fleet_pull()` (best-effort semantic top-up), `fleet_sync_pending()`, and auto-sync on ingest (`_maybe_autosync` + `drain_sync()`). **On by default** (`AITHER_FLEET_SYNC=auto`; only `false/0/off` disables) with an inferred target (local Nexus `:8122` or the gateway in cloud mode) and a SEPARATE collection (`AITHER_FLEET_GRAPH_COLLECTION`, default `graph_memory`) from the KV memory sync. New `GraphMemory(tenant_id=, workspace_id=, fleet_url=, auto_sync=)` constructor args override env (required for multi-tenant host processes). Node `synced` column + schema v4 migration. TLS verify is CA-bundle-aware (`AITHER_CA_BUNDLE`) for the internal-TLS mesh. Local SQLite stays the source of truth; every sync path is best-effort and never raises into ingest.

## [2.13.3] - 2026-07-02

### Added
- **Canonical self-deploying embeddings provider** (`adk.embeddings`): one embedding provider for the whole SDK so vectors are portable across scopes (platform / tenant / sovereign) — same model + dimension everywhere. Pinned to `nomic-embed-text` (768-d; the vLLM `--served-model-name` for nomic-embed-text-v1.5). Lazy, single-flight resolution chain: explicit `AITHER_EMBEDDINGS_URL` → local vLLM (`:8209` then `:8120`, HTTPS-then-HTTP) → local Ollama → gateway (`AITHER_GATEWAY_EMBEDDINGS_URL`) → auto-deploy a local vLLM embeddings container if a GPU + Docker are present (opt out with `AITHER_EMBED_AUTODEPLOY=0`) → CPU sentence-transformers (384-d, degraded) → feature-hash (384-d, degraded, always works). Every batch is dimension-tagged so callers never silently mix 768-d and 384-d vectors. `get_default_embedder()`, `embed_texts()`, `embed_one()`, `get_provider()`, `reset_provider()`.

### Changed
- **`GraphMemory` now defaults its embedder to the canonical provider** (opt out with `AITHER_GRAPH_EMBEDDER=legacy`), so every adk agent shares one 768-d space. Added a meta-table dimension guard: the index is pinned to its first embedding's dimension and later different-dimension vectors are refused (kills 768↔384 mixing that silently poisons cosine similarity).

## [2.13.2] - 2026-07-01

### Added
- **Session persistence**: AitherShell now auto-saves the active session_id to `~/.aither/config.yaml` after each chat turn and auto-restores it on the next shell launch. Multi-turn conversations survive across restarts — the messages were always persisted in SQLite + JSON, only the session pointer was missing.
- `/new` command to start a fresh session (clears saved session_id).
- `/sessions` command to list recent sessions with agent name, message count, and timestamp. Resume any with `/resume <id>`.

## [2.13.1] - 2026-07-01

### Fixed
- **CRITICAL:** `deploy_connect()` now sends a HEAD request to verify the GitHub release asset exists before downloading, with a clear error if the asset is missing.
- **CRITICAL:** Compose file URLs are now pinned to a stable commit hash (configurable via `AITHER_COMPOSE_PIN` env) instead of tracking the mutable `main` branch.
- **CRITICAL:** `deploy_adk_node()` now falls back to starting `adk-serve` natively when Docker is unavailable, instead of blocking with "Docker not installed".
- **HIGH:** GHCR login failure now blocks deployment immediately with actionable guidance, instead of warning and continuing to a confusing image-pull failure.
- **HIGH:** Health checks now stream the last 30 lines of Docker Compose logs on failure, so the user can see why a container didn't start.
- **HIGH:** vLLM port allocation now probes for conflicts before assigning; if port 8200 is busy, workers are automatically remapped to the next free port.
- **MEDIUM:** Compose file downloads are now cached with ETag; re-running `adk deploy node` skips re-downloading an unchanged file.
- **MEDIUM:** DGX Spark discovery now respects `AITHER_DGX_HOSTS` env (comma-separated) for custom network topologies, instead of only checking `spark.local` and `192.168.0.33`.
- **MEDIUM:** Cloud API keys (Anthropic, OpenAI, DeepSeek) are now validated with a lightweight probe during `adk setup` infra scan, so invalid keys surface early.
- **MEDIUM:** `deploy_connect()` now validates the extracted extension has a `manifest.json` and warns if it's missing.

## [2.13.0] - 2026-07-01

### Removed
- **`adk.faculties.memory_graph`** — the pickle-based `MemoryGraph` module is deleted. `from adk.faculties import MemoryGraph` now resolves to the canonical `adk.graph_memory.GraphMemory` for back-compat.
- **`adk.platform.memory`** — the entire legacy memory subsystem (`MemoryManager`, `UnifiedMemorySystem`, `GameEngine`, `StoryboardEngine`, `AnchorGenerator`) is deleted. Agent memory is `adk.memory.Memory` + `adk.graph_memory.GraphMemory`.

### Changed
- `adk work` command now uses `GraphMemory` instead of the removed pickle-based faculty graph.
- `adk.MemoryGraph` and `adk.faculties.MemoryGraph` lazy exports now resolve to `adk.graph_memory.GraphMemory`.
- `agent.set_memory_graph()` kept for back-compat but documented as deprecated; agents already wire `GraphMemory` automatically in `__init__`.
- Platform startup (`adk.platform.infrastructure.startup`) gracefully handles missing `MemoryManager`.
- Platform CLI module listing and health check updated to reflect removed memory module.

## [2.12.7] - 2026-07-01

### Deprecated
- `adk.faculties.MemoryGraph` — emits `DeprecationWarning` on instantiation; use `adk.graph_memory.GraphMemory` (SQLite-backed, governed, embedding-native) instead.
- `adk.platform.memory` package — the legacy `MemoryManager`, `UnifiedMemorySystem`, `GameEngine`, and `StoryboardEngine` modules now emit `DeprecationWarning` on import; use `adk.memory.Memory` and `adk.graph_memory.GraphMemory` for new code.

### Changed
- Updated `adk.faculties` docstring and usage examples to point at the canonical memory surface.

## [2.12.6] - 2026-07-01

### Changed
- Trimmed public package debt by removing grid runbook artifacts and shell/room binary placeholder directories from published artifacts.
- Removed stale public-sync allowlist entries for deleted phase test files and added explicit public-sync guards for binary placeholder directories and grid artifacts.
- Updated the public README grid note so it no longer links to runbook files excluded from the SDK package.

## [2.12.5] - 2026-07-01

### Added
- Added best-effort fleet memory sync from local ADK memory into Nexus/Qdrant-compatible RAG stores, including batch catch-up and vector-search recall hooks.
- Added saved config/env support for fleet memory URL, collection, and sync mode.

### Changed
- Excluded FormBridge runtime/test internals from public ADK packaging and public repo sync.

## [2.12.4] - 2026-07-01

### Added
- Added a hot-reload endpoint for installed agent packs so newly applied pack tools can be picked up without restarting `aither-serve`.

### Fixed
- Fixed synchronous `ChatRelay` calls on Python 3.12 when no event loop is running; async WebSocket delivery is still scheduled when a loop exists.
- Aligned gateway inference tier tests with the current public ladder: free, starter, pro, enterprise, platform.

## [2.12.0] - 2026-06-28

### Added — one-command local inference + upgrade path (click-to-run)
- **`adk quickstart-local`** — detects hardware, picks a backend
  (Ollama / llama.cpp / vLLM), installs it, downloads a model, verifies, and
  prints next steps. Bootstraps Ollama itself via winget/brew/install.sh when
  missing (no "go install it" dead-ends). Default model `gemma4:e2b`.
- **`adk backend switch <ollama|llamacpp|vllm>`** (+ `backend status`) — migrate
  the active deployment between engines; re-points config and smoke-tests.
- **`adk install pack:<name>`** (+ `adk install list` / `adk packs`) — install
  ready-made agent packs (openclaw, hermes, claude-code) into `~/.aither/agents`.

### Fixed (found via live end-to-end testing on Windows + CUDA)
- quickstart-local reported success on a dead endpoint (smoke test was warn-only);
  a failed smoke now fails the command, with a service-readiness wait.
- `--model` resolved to `None` (getattr default never fired) → uses the real default.
- Smoke test: 30s→180s timeout for cold model load; accepts reasoning-model output
  (empty `content` at low token budgets is no longer a false failure).
- `adk install pack:<name>` colon syntax (argparse rejected it before dispatch).
- `adk backend status` health probe used a nonexistent `/health` path → 404; now
  probes `/v1/models` (uniform OpenAI-compatible liveness for all three backends).
- Windows: removed non-ASCII glyphs from printed strings that crashed the cp1252
  console (a `✓` even made a successful install report failure).

## [2.11.1] - 2026-06-14

### Fixed — `adk host` / `adk login` device flow (autonomous onboarding)
Found driving `adk host` against production; all broke real device-flow login:
- **Device flow hit the portal FRONTEND, not AitherIdentity.** `adk login`/`adk host`
  POSTed `/auth/device/code` to portal/veil.aitherium.com, which 307-redirects API auth
  to `/login`. Added `_resolve_identity_url()` mapping the aitherium.com topology to
  `idp.aitherium.com` (the Identity API); localhost/bespoke pass through unchanged.
- **Cloudflare 403'd urllib.** `idp.*` sits behind Cloudflare, which blocks urllib's
  default `Python-urllib/3.x` UA. Device-code + poll requests now send a real
  User-Agent + Accept (verified urllib→idp 200, was 403).
- **`adk host` now self-authenticates via device flow** when there's no token, and
  **retries via device flow on a 401/403** (a stale saved token no longer dead-ends
  registration).

## [2.11.0] - 2026-06-14

### Added — attach your own MCP "hands" to a self-hosted agent
A self-hosted agent now has the full brain (your model) · body (this loop) · **hands**
(your tools) trifecta. The AitherOS platform relays the MCP servers you registered for
your agent in the `/stream` request body as `mcp_endpoints`, and the loop wires their
tools into the ReAct turn:
- **`adk/mcp_endpoint_tools.register_mcp_endpoint_tools(agent, endpoints)`** — speaks MCP
  JSON-RPC (`tools/list` to discover, `tools/call` to invoke) and registers a namespaced
  proxy tool (`{endpoint}__{tool}`) for each advertised tool. Mirrors `app_proxy_tools`.
  Best-effort: an unreachable endpoint registers nothing and never raises.
- **`POST /stream` and `POST /chat/stream`** accept an optional `mcp_endpoints` list
  (`[{name, url, [headers]}]`); `_aitheros_stream` attaches them to the agent before the
  turn. Existing callers are unaffected (the field is optional).

## [2.10.1] - 2026-06-14

### Fixed — self-hosted `aither-serve` was crash-on-startup (2.10.0 regression)
A fresh `pip install aither-adk && aither-serve --backend deepseek` could not boot. Found
and fixed while standing up a real self-hosted DeepSeek agent end-to-end:
- **Startup `NameError: name 'port' is not defined`** in three lifespan helpers
  (`_join_aithernet`, `_init_a2a_server`, `_start_mesh_hosting`) — they referenced a bare
  `port` no longer in scope. Now use `config.server_port`, and `main()` writes the resolved
  `--port`/`--host` back to `config` so every lifespan helper and the invoke_url see the
  actual bound port.
- **Fleet self-registration crashed the whole server** — the registration handlers caught
  `(ImportError, RuntimeError, OSError, ConnectionError)` but **not** `httpx.HTTPError`, so a
  `RemoteProtocolError`/connect error talking to the control plane took the agent down. All
  network-facing lifespan handlers now also catch `httpx.HTTPError` (non-fatal, as intended).
- **`--backend` was ignored when `AITHER_API_KEY` was set** — `_auto_detect` picked the cloud
  gateway before the explicit backend, and `_try_elysium_fallback` then replaced the chosen
  provider. An explicit backend (deepseek/openai/anthropic/your own vllm/ollama) now always
  wins; "auto"/"gateway" still flow through detection. The operator's own brain is never
  silently hijacked.
- **Approved-tool trace truncated after `/sessions/{id}/confirm`** — the SSE relay treated
  `AgentResponse.tool_calls_made` (a `list[str]`) as dicts (`tc.get(...)`), raised
  `AttributeError`, and an over-narrow `except` swallowed it so the client saw only
  `session_start` + heartbeats. The relay now handles string tool names, the `except` is
  broad (always emits a typed `error` + terminal `complete`), and the module-level
  `_strata_ingest` `NameError` + Chronicle `session_id` kwarg mismatch are fixed.

## [2.10.0] - 2026-06-13

### Added — human-in-the-loop tool approval (`adk/approval.py`)
- **Pause before a gated tool, resume on decision.** An agent can now pause its turn before
  executing a sensitive tool and wait for a human's allow/deny — so a managed control plane
  (or any operator) can gate tool use without forking the loop.
- **Policy via `AITHER_TOOL_APPROVAL`** — a comma-separated list of tool names (optionally
  `agent:tool`), or `*` for every tool. Empty = no gating (unchanged behavior).
- **Survives restarts / days-later approvals.** The paused turn (user message + pending tool
  calls) is persisted to disk keyed by `session_id`; resume is a fresh request, not a held
  connection. Decisions are recorded per `(session_id, tool_name)`.
- **`AgentResponse.requires_action` / `.pending`** — set when a turn paused; `pending` lists
  the gated tool calls awaiting a decision.
- **`AitherAgent.resume(session_id, decisions)`** — records the decisions and re-runs the
  paused turn (the gate consumes them: allowed tools execute, denied tools get a denial
  observation, the turn continues — and may pause again on a different gated tool).
- **Server:** `POST /sessions/{id}/confirm` records decisions and streams the resumed turn;
  `/stream` emits a `requires_action` event when a turn pauses.

## [2.9.0] - 2026-06-13

### Added — universal LLM continuation primitive (`adk/llm/continuation.py`)
- **`run_continuation()` + `stitch()`** — one shared continue-until-complete primitive. When a
  completion stops because it hit the output token cap (`finish_reason == "length"`), it continues
  the generation and STITCHES the chunks (overlap-dedup + restart-detect) into a complete answer,
  instead of every call-site hand-rolling its own retry. Bounded three ways (max rounds AND total
  output ceiling AND a no-growth break), and it never mutates the caller's message history.
- **`LLMProvider.chat_with_continuation()`** and **`LLMRouter.chat_with_continuation()`** — every
  provider AND the router inherit the primitive, so any call-site gets continuation for free.
- Kill-switch `ADK_LLM_CONTINUATION=off`; tunables `ADK_LLM_MAX_CONTINUATIONS`,
  `ADK_LLM_MAX_TOTAL_OUTPUT_TOKENS` (defaults: 2 rounds / 8192 tokens).

### Changed
- The ReAct loop's truncation recovery now uses the shared primitive. The previous inline
  doubling-retry REPLACED the partial response (losing earlier text in `resp.content`) and injected
  continuation scaffolding into the live message history; it now STITCHES and leaves history clean.
  Tool-call turns are never continued as text.

### Included — previously-committed-but-unreleased changes
- 2.8.2: TLS verification on by default (`verify=False` removed).
- 2.8.1: baked license + pack-signing public keys.
- 2.8.0: version-bump release.

## [2.7.0] - 2026-06-05

### Added — provider-agnostic LLM cost levers (at the `LLMProvider` ABC)
- **Prompt caching.** New `ProviderCapabilities` (`prompt_cache: explicit|automatic|none`, `batch`) and
  normalized `LLMResponse.cache_read_tokens`/`cache_write_tokens`, so the "tokens saved" meter never
  depends on which backend served the turn. Anthropic inserts `cache_control` breakpoints over the stable
  system+tools prefix; OpenAI/DeepSeek caching is automatic and their `cached_tokens` are now surfaced
  (previously discarded). Default-on in `AitherAgent.chat`; override with `cache=False`.
- **ResponseCache** (`adk/llm/cache.py`) — exact-match response cache with a pluggable backend, wired into
  `LLMRouter` as an opt-in (`LLMRouter(response_cache=…)` + per-call `cacheable=True`). OFF by default — no
  determinism change unless opted in.
- **Gemini provider** (`adk/llm/gemini.py`) with explicit `cachedContents` caching; registered in the router.
- **Async batch runner** (`adk/llm/batch_runner.py`) — `run_batch()` over the Anthropic Message Batches /
  OpenAI Batch APIs (~50% off) with a transparent fallback to concurrent `chat()` where unsupported.
- **Graph-RAG retriever interface** (`adk/graph_retriever.py`).

### Added — principal authority (prompt-injection defense enforced outside the model)
- `adk/auth.py`: `Principal` / `AuthContext` / `PrincipalResolver` (+ deny-all default).
- `ToolRegistry.execute(name, args, auth=…)` — a default-deny authorization gate at the tool boundary, so a
  forbidden tool call is blocked regardless of what the LLM emits. `ToolDef` gains `required_clearance` /
  `action_class`. Threaded through `AitherAgent.chat`; `auth=None` is fully backward-compatible.
- `adk/channel_auth.py`: Telegram webhook-secret + CEO-id checks and HMAC-signed approvals.

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
