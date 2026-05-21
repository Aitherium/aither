# AitherOS Roadmap

Updated May 2026.

---

## Milestone 1 — Alpha Foundation (v0.3.x) [DONE]

Everything needed for `pip install aither-adk` to work end-to-end.

| Feature | Status |
|---------|--------|
| Agent class with @tool decorator | Done |
| Multi-backend LLM (Ollama, OpenAI, Anthropic, vLLM) | Done |
| SQLite conversation memory | Done |
| OpenAI-compatible server (`aither-serve`) | Done |
| 17 agent identities as package data | Done |
| CLI (`aither init`, `aither serve`, `aither bug`) | Done |
| Hardware auto-detection (5 tiers, 11 profiles) | Done |
| Safety gates (IntakeGuard, LoopGuard, Sandbox) | Done |
| Privacy-centric opt-in telemetry | Done |
| MCP bridge to mcp.aitherium.com | Done |
| 522 passing tests | Done |

---

## Milestone 2 — Developer Experience (v0.4-v0.16) [DONE]

SDK polish: documentation, onboarding, code intelligence, and memory graph.

| Feature | Status |
|---------|--------|
| API reference docs (auto-generated from docstrings) | Done |
| Tutorial suite (First Agent, Multi-Agent Fleet, vLLM) | Done |
| GETTING_STARTED.md and CHANGELOG.md | Done |
| `aither init` interactive wizard | Done |
| Auto-install Ollama and pull default model | Done |
| Cross-platform CI (Windows/macOS/Linux) | Done |
| PyPI publish workflow (GitHub Actions trusted publishing) | Done |
| CodeGraph indexing (62,867 chunks, 2,318 files) | Done |
| MemoryGraph with hybrid search (keyword + semantic) | Done |
| MCP gateway and tool auto-discovery | Done |
| Examples: RAG agent, code review agent, MCP tool server | Done |

---

## Milestone 3 — Platform Merge (v1.0) [DONE]

Package consolidation under the `adk` namespace. Single install, single import.

| Feature | Status |
|---------|--------|
| Unified `adk` namespace (agent, forge, chat, tools) | Done |
| Context pipeline (token-budget-aware, 12-stage assembly) | Done |
| Knowledge graph with cross-session persistence | Done |
| Neuron system (web search, memory recall, code context) | Done |
| Agent Forge with ReAct dispatch loop | Done |
| Effort-based model routing (EffortScaler) | Done |
| Swarm Coding Engine (4-phase, 11 roles) | Done |
| A2A mesh and capability registry | Done |
| RBAC, API key auth, token metering | Done |
| Docker Compose for ADK + vLLM | Done |
| Structured logging, health monitoring, Prometheus metrics | Done |

---

## Milestone 4 — Multi-Channel & Federation (v1.1) [DONE]

Multi-channel agent access, group collaboration, and cross-instance federation.

| Feature | Status |
|---------|--------|
| Multi-channel gateway (web, Discord, Slack, API) | Done |
| Aeon group chat (multi-agent conversations) | Done |
| Skills system (composable agent capabilities) | Done |
| MCP stdio transport | Done |
| SOUL.md (agent personality and behavior definitions) | Done |
| Elysium auth (OAuth2 + tenant isolation) | Done |
| Remote agent pipeline (provision, deploy, monitor) | Done |
| Federation mesh (cross-instance agent communication) | Done |
| CallerContext isolation (PLATFORM/PUBLIC/DEMO/TENANT) | Done |

---

## Milestone 5 — Runtime & Inference (v1.2) [DONE]

Runtime backend flexibility, quantization, and CLI expansion.

| Feature | Status |
|---------|--------|
| Runtime backend switching (hot-swap vLLM instances) | Done |
| Hybrid reasoning (local reasoning model + cloud fallback) | Done |
| TQ4 quantization support (TRITON_ATTN backend) | Done |
| DeepSeek provider integration | Done |
| DGX Spark support (128GB unified memory, remote LAN) | Done |
| `adk train` CLI command | Done |
| Slash-command bridge | Done |
| Schema migrations framework | Done |
| 7 new CLI commands (aithershell entry point) | Done |
| GoalWire integration (7 goal tiers, 5 escalation paths) | Done |
| 1,246+ passing tests | Done |

---

## Milestone 6 — Agent Experience (v1.3) [CURRENT]

Target: June 2026

Interactive agent workflows, marketplace, and expanded language support.

| Feature | Status |
|---------|--------|
| AitherShell interactive mode improvements | In Progress |
| Agent marketplace integration (publish/discover agents) | In Progress |
| CodeGraph multi-language support (JS/TS/Rust/Go) | Planned |
| Real-time voice agent mode | Planned |
| Fleet auto-scaling | Planned |

---

## Milestone 7 — Platform v2 (v2.0) [PLANNED]

Target: Q3 2026

Breaking changes and next-generation architecture.

| Feature | Status |
|---------|--------|
| Async-first API (breaking migration) | Planned |
| Plugin system (loadable agent extensions) | Planned |
| A2A v2 protocol standardization | Planned |
| Distributed training with federated learning | Planned |
| Self-improving agent loops | Planned |

---

## AitherZero Roadmap

AitherZero is the PowerShell automation framework. Ships separately as a module.

### Current (v2.x) [DONE]

| Feature | Status |
|---------|--------|
| 170+ numbered automation scripts | Done |
| PSScriptAnalyzer CI integration | Done |
| Pester test suite | Done |
| Module build pipeline (public + private merge) | Done |
| Bootstrap with Minimal/Full profiles | Done |
| Dashboard TUI | Done |
| AitherOS service management cmdlets | Done |

### Next (v3.0) [PLANNED]

Target: Q3 2026

| Feature | Priority |
|---------|----------|
| PSGallery publish workflow | P0 |
| Cross-platform test matrix (Windows + Linux + macOS) | P0 |
| Agent interaction cmdlets (Invoke-AitherAgent) | P1 |
| Pipeline DSL for automation workflows | P1 |
| Integration with ADK Python agents | P2 |

---

## Release Schedule

| Date | Version | Milestone | Status |
|------|---------|-----------|--------|
| March 2026 | v0.3.x | Alpha Foundation | DONE |
| April 2026 | v0.4-v0.16 | Developer Experience | DONE |
| April 2026 | v1.0 | Platform Merge | DONE |
| May 2026 | v1.1 | Multi-Channel & Federation | DONE |
| May 2026 | v1.2 | Runtime & Inference | DONE |
| June 2026 | v1.3 | Agent Experience | CURRENT |
| Q3 2026 | v2.0 | Platform v2 | PLANNED |

---

Want to influence the roadmap? Star the repo and open a Discussion.
