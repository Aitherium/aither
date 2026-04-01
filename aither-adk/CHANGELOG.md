# Changelog

All notable changes to aither-adk will be documented in this file.

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
