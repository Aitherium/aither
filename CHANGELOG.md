# Changelog

All notable changes to AitherOS are documented here.

## [0.3.0] - 2026-03-08

### Added — Knowledge Graph Memory
- `adk.graph_memory` — SQLite-backed knowledge graph with embedding-based search
- Ollama `nomic-embed-text` embeddings with feature-hashing fallback (works offline)
- Hybrid search: keyword inverted index + semantic cosine similarity, query-type-weighted
- Entity extraction: CamelCase services, capitalized phrases, file paths, code identifiers
- Relation extraction: "X uses Y", "X depends on Y", "X contains Y" triples
- Auto-edge detection: TAG_SIBLING (shared tags), SAME_SESSION, RELATED (embedding sim)
- BFS graph traversal: `get_related("entity", depth=2)` for multi-hop exploration
- Conversation auto-ingestion: entities and relations extracted after every chat()
- Agent convenience: `graph_remember()`, `graph_query()`, `graph_stats()`

### Added — Neuron Architecture
- `adk.neurons` — Auto-firing context neurons before LLM calls
- WebSearchNeuron (DuckDuckGo, no API key), MemoryNeuron, GraphNeuron
- NeuronPool: manages and fires neurons in parallel with timeout protection
- AutoNeuronFire: 6-category pattern detection, result caching, auto-injects context
- Custom neuron registration via `BaseNeuron` ABC

### Added — NanoGPT Trainer
- `adk.nanogpt` — Zero-dependency character-level transformer
- Pure Python autograd engine (Value class with backward pass)
- Multi-head attention, RMSNorm, MLP with ReLU
- LoRA hypernetwork adapters for document-specific memory
- Async training via `asyncio.to_thread()` (non-blocking)
- Save/load model weights to JSON, generation with temperature sampling
- Use cases: topic classification, anomaly detection, intent prediction

### Added — Full Pipeline Wiring
- **Safety**: IntakeGuard wired into chat() input (blocks injection), output (redacts leaks), forge dispatch, streaming
- **Context**: ContextManager token-aware truncation replaces manual message assembly
- **Events**: EventEmitter fires chat_request/response, tool_call/result, forge_dispatch/complete
- **Builtin Tools**: Identity-based registration (12 tools: file_io, shell, python, web, secrets)
- **ServiceBridge**: Auto-discovery at server startup (Node -> Genesis -> Gateway -> standalone)

### Added — Agent Features
- `chat_stream()` — agent-level streaming with full safety pipeline, tool fallback to sync
- Server auth middleware: `AITHER_SERVER_API_KEY` env var, Bearer token validation
- CLI scaffolding: `aither init <name>` creates project, `aither run` starts server
- Skip-auth paths: /health, /docs, /openapi.json, /metrics, /demo, /redoc

### Tests
- 183 new tests across 9 test files
- 522 total ADK tests passing (0 regressions)

## [0.2.0] - 2026-03-08

### Added — AitherMesh Gateway Connection
- Connect standalone/Alpha instances to `gateway.aitherium.com` for remote backend services
- Gateway API routes: `/api/gateway/connect`, `/api/gateway/status`, `/api/gateway/proxy`
- Gateway proxy forwards to 16 whitelisted backend services with bearer auth
- Settings UI page at `/settings/gateway` with connection form, status display, feature cards
- 3-tier connection detection: local → gateway → full-stack
- `useCapabilities` hook: parallel node + gateway status fetch with feature merging
- `FeatureGate` + `FeatureChip`: per-widget graceful degradation with gateway awareness
- `LocalCapabilityBanner`: context-aware banners with "Connect to AitherMesh" CTA
- 10+ Veil pages wrapped with `FeatureGate` for tier-appropriate degradation

### Improved — ADK (Agent Development Kit)
- `adk.server`: enhanced FastAPI server with OpenAI-compatible endpoints
- `adk.config`: improved configuration management
- `adk.identity`: expanded identity resolution
- `adk.setup`: comprehensive hardware detection and profile matching
- New modules: `builtin_tools`, `context`, `events`, `safety`, `services`
- Updated hardware profiles for latest GPU generations
- Expanded test coverage (test_setup.py: 500+ lines)

### Added — ADK New Modules
- `adk.builtin_tools` — built-in tool implementations
- `adk.context` — context management for agent sessions
- `adk.events` — event system for agent lifecycle
- `adk.safety` — safety checks and guardrails
- `adk.services` — service discovery and health checks

## [0.1.0-alpha.1] - 2026-03-07

### Added — AitherADK (Agent Development Kit)
- Clean-room Python package: `pip install aither-adk`
- Multi-backend LLM providers: Ollama, OpenAI, Anthropic, vLLM, LM Studio
- `AitherAgent` class with @tool decorator and ReAct-style tool loops
- `LLMRouter` with auto-detection and effort-based model selection
- `Memory` — SQLite-backed conversation history and KV store
- `Server` — FastAPI with OpenAI-compatible `/v1/chat/completions`
- `MCPBridge` — connect to AitherOS tools via mcp.aitherium.com
- 16 agent identities shipped as package data
- `aither-serve` CLI entry point
- `aither-bug` CLI for built-in bug reporting
- Privacy-centric opt-in telemetry (phonehome)
- Gateway client for agent registration and discovery
- 85 passing tests

### Added — Standalone Mode
- AitherNode mounts ADK server when Genesis unreachable
- AitherDesktop falls back to Node:8090 via OpenAI-compatible format
- AitherConnect sidepanel supports standalone mode toggle
- Node ADK endpoint at `/v1/chat/completions` for universal client access

### Added — Dark Factory (Autonomous Loops)
- Background loops: neuron firing (4hr), orchestrator (daily), finetune (12hr)
- Pulse SSE pain subscriber with auto-reconnect
- SelfModification pipeline with post-merge verification
- IntentChainRunner for multi-agent dispatch chains
- StrataFeedback for effort/model calibration
- SessionLearner with pattern extraction and memory promotion
- PlaybookEngine with YAML operational runbooks and auto-generation
- ContextTierManager OODA loop (observe/orient/decide/act)
- ClosedLoopController with model hot-swap on deploy

### Added — Expedition Manager
- Multi-session project orchestration with SQLite persistence
- SOW analysis, phase decomposition, parallel task dispatch
- Human-in-the-loop gate approvals (CODE_REVIEW, DEPLOYMENT, SOW, TECH)
- Forge artifact extraction with auto-REVISE
- Phase integration tests, Athena security review
- Cost/token tracking per phase, budget variance alerts
- Auto-deploy with rollback on smoke failure
- 250 passing tests

### Added — Swarm Coding Engine
- 11 specialized agents, 4-phase pipeline (ARCHITECT->SWARM->REVIEW->JUDGE)
- 3 execution modes: LLM (fast), FORGE (full tools), PLAN_ONLY
- AgentForge-backed dispatch with identity mapping
- Sandbox testing and bundle delivery

### Added — Security & Multi-Tenancy
- Caller isolation: Platform/Public/Demo/Tenant/Anonymous types
- Public tenant auto-routing for external requests
- Multi-tenant graph isolation across all graph subsystems
- Pipeline prompt injection defense (3 insertion points)
- RBAC SQLite backend with JSON seed data migration
- GDPR Art.17 data erasure endpoints

### Added — Infrastructure
- Content Production Pipeline (8 artifact types, agent dispatch, quality gates)
- Agent-to-Agent full mesh (forge_dispatch tools, heartbeat, MCP indexing)
- Social graph (friends, groups, directory, MySpace-style profiles)
- Package Manager (APM) with tenant enablements and HMAC entitlements
- Frontier Judge with Anthropic Claude quality gates
- Agent email system (20 Proton Mail addresses)
- Context synthesis replacing hard truncation
- Local voice (faster-whisper STT, Piper TTS, VRAM coordination)
- MCP SaaS Gateway at mcp.aitherium.com
- LTX-2.3 video generation via Iris + ComfyUI

### Infrastructure
- 203 microservices (23 compound containers, 65 Docker total)
- 2600+ passing tests across 120+ test files
- 170+ PowerShell automation scripts
- Multi-model vLLM deployment (4 workers)
- Hardware profile auto-detection (5 tiers)

---

## [Pre-Alpha] - 2024-12 through 2026-02

### Foundation
- Initial 203 microservices built across 12 architectural layers
- Genesis bootloader with 7-phase boot sequence
- AitherZero PowerShell automation framework
- AitherVeil Next.js dashboard
- Pain system and self-healing infrastructure
- Five-tier memory architecture
- Multi-model LLM routing via MicroScheduler
- FluxEmitter event bus for inter-agent communication
