# Changelog

All notable changes to AitherOS are documented here.

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
- 97 microservices (23 compound containers, 65 Docker total)
- 2600+ passing tests across 120+ test files
- 170+ PowerShell automation scripts
- Multi-model vLLM deployment (4 workers)
- Hardware profile auto-detection (5 tiers)

---

## [Pre-Alpha] - 2024-12 through 2026-02

### Foundation
- Initial 97 microservices built across 12 architectural layers
- Genesis bootloader with 7-phase boot sequence
- AitherZero PowerShell automation framework
- AitherVeil Next.js dashboard
- Pain system and self-healing infrastructure
- Five-tier memory architecture
- Multi-model LLM routing via MicroScheduler
- FluxEmitter event bus for inter-agent communication
