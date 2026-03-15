# Changelog

All notable changes to aither-adk will be documented in this file.

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
