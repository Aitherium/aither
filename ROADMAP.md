# AitherOS Roadmap

> Updated March 2026. This is the public-facing roadmap.

## Done

### Core Platform
- [x] 97 FastAPI microservices across 12 architectural layers
- [x] 23 compound Docker containers (absorbing 81 sub-services)
- [x] 65 Docker containers total
- [x] 2600+ passing tests across 120+ test files
- [x] Genesis bootloader with topological-sort boot orchestration
- [x] AitherZero automation framework (170+ scripts)

### Agent Infrastructure
- [x] 16 specialist agents with persistent identity and memory
- [x] AgentForge dispatch with effort-based model selection
- [x] CapabilityRegistry with heartbeat and tool-level tracking
- [x] ToolGraph 3-tier selection (NanoGPT, hybrid, full)
- [x] ReAct-style tool use loops with shared workspace

### Dark Factory (Autonomous Loops)
- [x] Neuron firing — proactive data gathering (33 patterns)
- [x] Pain-driven remediation — detect, fix, deploy, verify
- [x] Session learning — pattern extraction, memory promotion
- [x] Playbook auto-generation after 3 successful occurrences
- [x] Model finetuning via daydream corpus
- [x] Strata feedback for effort/model calibration
- [x] Intent chain routing with alternative agent fallback
- [x] Escalation chain (retry with context, upgrade model)

### Orchestration
- [x] Expedition Manager — multi-session project orchestration
- [x] Swarm Coding Engine — 11 agents, 4-phase pipeline
- [x] Content Production Pipeline — 8 artifact types
- [x] Frontier Judge — cloud-based quality gate (Anthropic)
- [x] Clarification Gate — agent-to-human email workflow

### Security & Multi-Tenancy
- [x] RBAC with SQLite backend and PostgreSQL option
- [x] Caller isolation (Platform/Public/Demo/Tenant/Anonymous)
- [x] Multi-tenant graph isolation
- [x] Pipeline prompt injection defense
- [x] HMAC-SHA256 capability tokens (default-deny)
- [x] GDPR Art.17 data erasure

### Infrastructure
- [x] MicroScheduler VRAM coordination
- [x] Multi-model vLLM deployment (4 workers)
- [x] FluxEmitter pub/sub event bus
- [x] Context synthesis (extractive compression, no hard truncation)
- [x] Context spillover + OODA tier manager
- [x] Local voice (faster-whisper STT, Piper TTS)

### Social & Communication
- [x] Social graph (friends, groups, directory)
- [x] MySpace-style customizable profile pages
- [x] Agent email system (20 Proton Mail addresses)
- [x] AitherRelay IRC bridge

### Developer Experience
- [x] MCP SaaS Gateway at mcp.aitherium.com
- [x] Package Manager (APM) with tenant enablements
- [x] Partner profile system (white-label deployment)
- [x] 100+ MCP tools via AitherNode

---

## Alpha Release (Current Sprint)

### AitherADK
- [x] Clean-room agent development kit (pip install aither-adk)
- [x] Multi-backend LLM providers (Ollama, OpenAI, Anthropic)
- [x] Agent class with @tool decorator and SQLite memory
- [x] OpenAI-compatible server (aither-serve)
- [x] 16 agent identities as package data
- [x] MCP bridge to mcp.aitherium.com
- [x] Privacy-centric opt-in telemetry
- [x] Built-in bug reporting (CLI + API)
- [x] 85 passing tests

### Integration
- [x] AitherNode standalone mode with ADK server
- [x] Desktop fallback to Node when Genesis unavailable
- [x] Connect extension standalone mode toggle
- [ ] Hardware profile auto-detection (5 tiers)
- [ ] Auto-installer for drivers/CUDA/PyTorch/models

### Gateway Infrastructure
- [ ] Cloudflare Worker at gateway.aitherium.com
- [ ] User registration and email verification
- [ ] Agent capability advertisement and discovery
- [ ] Bug report ingestion and GitHub issue creation

### Documentation
- [x] Alpha repo README rewrite
- [x] Roadmap update
- [ ] GETTING_STARTED.md replacing COMING_SOON.md
- [ ] CHANGELOG.md with real release notes
- [ ] Landing page update (hello@aitherium.com)

---

## Next

### Post-Alpha
- [ ] Mobile companion app (React Native)
- [ ] JS/TS SDK for browser and Node.js
- [ ] Agent marketplace
- [ ] Federation between AitherOS instances
- [ ] Plugin/extension system for community agents
- [ ] vLLM pre-built Docker images per hardware tier

### Future
- [ ] Kubernetes deployment with Helm charts
- [ ] Multi-node distributed agents via WireGuard overlay
- [ ] Voice interface (wake word, continuous conversation)
- [ ] Managed cloud offering
- [ ] Visual workflow builder

---

*Want to influence the roadmap? Star the repo and open a Discussion.*
