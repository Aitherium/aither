# AitherOS Roadmap

Updated March 2026.

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

## Milestone 2 — Developer Experience (v0.4.0) [IN PROGRESS]

Target: April 2026

Make the SDK something people can actually pick up and use without reading source code.

### 2a. Documentation & Onboarding (Week 1-2)

| Task | What to port/build | Priority |
|------|---------------------|----------|
| API reference docs | Auto-generate from docstrings (pdoc/mkdocs) | P0 |
| Tutorial: "Your First Agent in 5 Minutes" | New — hello world to custom tools | P0 |
| Tutorial: "Multi-Agent Fleet" | New — based on examples/multi_agent.py | P0 |
| Tutorial: "Connect to Your Own vLLM" | Expand from setup-vllm.py | P1 |
| GETTING_STARTED.md | Replace COMING_SOON.md placeholder | P0 |
| CHANGELOG.md with semver entries | New — retroactive from git log | P0 |
| llms.txt / llms-full.txt | Already done, keep current | Done |

### 2b. Installer & Setup Hardening (Week 2-3)

| Task | What to port/build | Priority |
|------|---------------------|----------|
| `aither init` interactive wizard | Improve existing cli.py | P0 |
| Auto-install Ollama if missing | Port from AitherOS setup.py patterns | P1 |
| Auto-pull default model on first run | Port from AitherOS boot/models.py | P1 |
| Windows/macOS/Linux install testing | CI matrix in sync-alpha.yml | P0 |
| PyPI publish workflow | New — build + twine + test.pypi gate | P0 |

### 2c. Examples Expansion (Week 3-4)

| Task | What to port/build | Priority |
|------|---------------------|----------|
| RAG agent (file ingestion + Q&A) | New — uses graph_memory.py | P1 |
| Code review agent | Port pattern from Hydra identity | P1 |
| Slack/Discord bot template | New — webhook + aither-serve | P2 |
| MCP tool server example | Expand from mcp_server.py | P1 |
| Jupyter notebook walkthrough | New | P2 |

---

## Milestone 3 — Agent Intelligence (v0.5.0)

Target: May 2026

Port the core cognitive loop from AitherOS to make agents actually smart.

### 3a. Context Pipeline (Week 1-2)

| Task | What to port | Source |
|------|-------------|--------|
| Token-budget-aware context assembly | lib/orchestration/ContextPipeline.py | Stages 1-8 |
| Query-conditioned re-scoring | ContextPipeline Stage 7.5 | New in M1 |
| Dynamic budget by complexity | NeuronScaler complexity classes | lib/cognitive/ |
| Conversation summarization | Context synthesis (extractive) | lib/core/ |

### 3b. Knowledge Graph (Week 2-3)

| Task | What to port | Source |
|------|-------------|--------|
| Graph memory with embeddings | Already in adk, needs hardening | adk/graph_memory.py |
| Cross-session knowledge persistence | MemoryGraph patterns | lib/memory/ |
| Hybrid search (keyword + semantic) | Already in adk | adk/graph_memory.py |
| Faculty graph pattern | BaseFacultyGraph abstraction | lib/faculties/ |

### 3c. Neuron System (Week 3-4)

| Task | What to port | Source |
|------|-------------|--------|
| Pre-LLM data gathering (NeuronPool) | Already in adk, needs more patterns | adk/neurons.py |
| Web search neuron | Port WebSearchNeuron | lib/cognitive/ |
| Memory recall neuron | Port MemoryNeuron | lib/cognitive/ |
| Code context neuron | Port CodeGraphNeuron | lib/cognitive/ |

---

## Milestone 4 — Multi-Agent Orchestration (v0.6.0)

Target: June 2026

Port the swarm and forge systems for real multi-agent workflows.

### 4a. Agent Forge (Week 1-2)

| Task | What to port | Source |
|------|-------------|--------|
| ReAct dispatch loop | Already in adk/forge.py, needs hardening | lib/orchestration/AgentForge.py |
| Effort-based model routing | EffortScaler | lib/core/AgentKernel.py |
| Tool-calling with structured output | vLLM tool_call + XML fallback | lib/core/UnifiedChatBackend.py |
| Shared workspace between agents | SharedRLMWorkspace | lib/cognitive/ |

### 4b. Swarm Coding Engine (Week 2-4)

| Task | What to port | Source |
|------|-------------|--------|
| 4-phase pipeline (Architect/Swarm/Review/Judge) | SwarmCodingEngine | lib/orchestration/ |
| 11 swarm roles with identity mapping | swarm_roles.yaml | config/ |
| Parallel agent execution | asyncio.gather dispatch | SwarmCodingEngine |
| Delivery pipeline (package + sandbox test) | SwarmDeliveryPipeline | lib/orchestration/ |

### 4c. Fleet Management

| Task | What to port | Source |
|------|-------------|--------|
| Agent-to-Agent mesh | Already in adk/a2a.py | lib/core/AgentBus.py |
| Federation protocol | Already in adk/federation.py | New in ADK |
| Capability registry | Port heartbeat + TTL patterns | lib/core/ |

---

## Milestone 5 — Production Readiness (v0.7.0)

Target: July 2026

Everything needed to run agents in production.

### 5a. Observability

| Task | What to port | Source |
|------|-------------|--------|
| Structured logging (Chronicle) | Already in adk | adk/chronicle.py |
| Health monitoring (Watch) | Already in adk | adk/watch.py |
| Prometheus metrics export | Already in adk | adk/metrics.py |
| Context X-Ray (debug context assembly) | ContextXray | lib/core/ |
| Distributed tracing | Already in adk | adk/trace.py |

### 5b. Security

| Task | What to port | Source |
|------|-------------|--------|
| RBAC with role/group/permission model | Simplified from AitherRBAC | lib/security/ |
| API key authentication | Already in adk | adk middleware |
| Token metering and rate limiting | Already in adk | adk/metering.py |
| Prompt injection defense | Pipeline safety gates | lib/core/ |

### 5c. Deployment

| Task | What to port/build | Priority |
|------|---------------------|----------|
| Docker Compose for ADK + vLLM | Already exists, needs testing | P0 |
| Helm chart for Kubernetes | New | P1 |
| Pre-built Docker images per hardware tier | New — GHCR publish | P1 |
| Systemd/launchd service files | New | P2 |

---

## Milestone 6 — Ecosystem (v1.0.0)

Target: Q3 2026

Public release and ecosystem growth.

| Feature | Description |
|---------|-------------|
| JS/TS SDK | Browser and Node.js client for ADK servers |
| Agent marketplace | Publish/discover community agents |
| Plugin system | Loadable agent extensions |
| Visual workflow builder | Drag-and-drop agent pipelines (Veil-based) |
| Mobile companion | React Native app for agent interaction |
| Managed cloud offering | Hosted ADK at aitherium.com |

---

## Porting Dependency Map

Modules listed in order of dependency. Port bottom-up.

```
Layer 0 (no internal deps — copy directly):
  adk/tools.py, adk/config.py, adk/identity.py, adk/memory.py
  adk/llm/*.py, adk/safety.py, adk/chronicle.py

Layer 1 (depends on Layer 0):
  adk/context.py, adk/neurons.py, adk/graph_memory.py
  adk/events.py, adk/metering.py

Layer 2 (depends on Layer 1):
  adk/agent.py, adk/forge.py, adk/chat.py
  adk/builtin_tools.py

Layer 3 (depends on Layer 2):
  adk/a2a.py, adk/federation.py, adk/fleet.py
  adk/server.py, adk/mcp.py

Layer 4 (depends on Layer 3):
  adk/setup.py, adk/elysium.py
```

---

## AitherZero Roadmap

AitherZero is the PowerShell automation framework. Ships separately as a module.

### Current (v2.0.0)

| Feature | Status |
|---------|--------|
| 170+ numbered automation scripts | Done |
| PSScriptAnalyzer CI integration | Done |
| Pester test suite | Done |
| Module build pipeline (public + private merge) | Done |
| Bootstrap with Minimal/Full profiles | Done |
| Dashboard TUI | Done |

### Next (v2.1.0) — April 2026

| Feature | Priority |
|---------|----------|
| PSGallery publish workflow | P0 |
| Cross-platform test matrix (Windows + Linux + macOS) | P0 |
| Script dependency graph visualization | P1 |
| AitherOS service management cmdlets | P1 |
| Docker management cmdlets | P2 |

### Future (v3.0.0)

| Feature | Priority |
|---------|----------|
| Agent interaction cmdlets (Invoke-AitherAgent) | P1 |
| Pipeline DSL for automation workflows | P1 |
| Integration with ADK Python agents | P2 |

---

## Release Schedule

| Date | ADK Version | AitherZero | Milestone |
|------|-------------|------------|-----------|
| March 2026 | v0.4.0-alpha | v2.0.1 | Docs + installer + CI |
| April 2026 | v0.4.0 | v2.1.0 | Developer Experience |
| May 2026 | v0.5.0 | — | Agent Intelligence |
| June 2026 | v0.6.0 | — | Multi-Agent Orchestration |
| July 2026 | v0.7.0 | v2.2.0 | Production Readiness |
| Q3 2026 | v1.0.0 | v3.0.0 | Public Ecosystem |

---

Want to influence the roadmap? Star the repo and open a Discussion.
