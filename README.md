<div align="center">

<img src="assets/aitheros-logo.jpg" alt="AitherOS" width="200" />

# AitherOS

**An Agentic Operating System — not a chatbot with tools.**

203 microservices · 16 specialist agents · Dark factory autonomous loops
2600+ tests · Expedition orchestration · Self-improving feedback loops

[![Status](https://img.shields.io/badge/status-alpha-blueviolet?style=flat-square)](https://aitherium.com)
[![Built By](https://img.shields.io/badge/built%20by-one%20person-cyan?style=flat-square)](#)
[![Services](https://img.shields.io/badge/services-97-blue?style=flat-square)](#architecture)
[![Tests](https://img.shields.io/badge/tests-2600%2B-green?style=flat-square)](#)
[![Agents](https://img.shields.io/badge/agents-16-purple?style=flat-square)](#agents)

**Start building agents today:**

```bash
pip install aither-adk
```

[Get Started](#get-started) |
[Architecture](#architecture) |
[Agents](#the-agents) |
[ADK](#aither-adk) |
[Roadmap](#roadmap) |
[Stay Updated](#stay-updated)

</div>

---

> [!CAUTION]
> ## 🚧 This Is Alpha. Real Alpha. Read This First.
>
> **AitherOS-Alpha is a public port of a massive internal monorepo.** This is not a polished release. This is not a demo. This is not ready for production use. **Most of it will not work out of the box.**
>
> To understand why, consider what you're looking at: a system with **203 microservices** across **12 architectural layers**, **268 automation scripts**, **21 service groups**, **18 specialist AI agents** with persistent identity and memory, multi-model inference orchestration, distributed cognition pipelines, self-healing pain loops, VRAM coordination, a full Next.js dashboard, MCP tool integration, and thousands of internal cross-references — all built by one person over 15+ months. This is not a weekend project. This is an operating system.
>
> Porting something of this complexity into a clean, standalone public repository is a **multi-month migration**, not a git push. Here's what that means for you right now:
>
> **🔴 Expect breakage everywhere:**
> - Services will fail to start — missing configs, hardcoded internal paths, unresolved dependencies
> - Docker Compose profiles are partially validated — some containers won't build or connect
> - Documentation references internal tooling, endpoints, and infrastructure that doesn't exist here yet
> - Python imports may reference modules that haven't been ported
> - PowerShell scripts may assume paths and environment variables from the internal repo
> - Database schemas, migrations, and seed data are incomplete
> - Agent identities, memory tiers, and cognitive pipelines require services that may not be running
> - Tests exist but many will fail without the full service mesh
>
> **🟡 What DOES work today:**
> - **AitherADK** — the agent development kit is standalone and pip-installable
> - **Architecture documentation** — the design and thinking are real and documented
> - **Individual service code** — the Python services are real, production-grade code
> - **Agent identities** — all 16 ship with ADK as package data
>
> **Why ship it broken?** Because building in the open matters more than waiting for perfection. The architecture is real. The code is real. The agents are real. We're porting it piece by piece, validating as we go. If you're here, you're early — and that means things will be rough.
>
> **If something doesn't work, it's not abandoned — it just hasn't been migrated yet.**
>
> 📋 **Track migration progress:** [ROADMAP.md](ROADMAP.md) · 🐛 **Report issues:** [GitHub Issues](https://github.com/Aitherium/AitherOS-Alpha/issues)

---

## Get Started

### Option 1: Build an Agent (5 minutes)

```bash
pip install aither-adk
```

```python
import asyncio
from adk import AitherAgent

async def main():
    agent = AitherAgent("aither")  # Auto-detects Ollama on localhost
    response = await agent.chat("Hello! What can you help me with?")
    print(response.content)

asyncio.run(main())
```

Works with Ollama, OpenAI, Anthropic, vLLM, LM Studio, or any OpenAI-compatible API.

### Option 2: Run the Server

```bash
aither-serve --identity aither --port 8080
```

Now any client can talk to your agent via OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"hello"}]}'
```

### Option 3: Full AitherOS Stack (Docker)

```bash
git clone https://github.com/Aitherium/AitherOS-Alpha.git
cd AitherOS-Alpha
python install.py        # Detects hardware, installs deps, pulls models
docker compose up -d     # Start everything
# Dashboard at http://localhost:3000
```

---

## What Is This?

AitherOS is a full-stack **agentic operating system** built solo from the ground up. Not an AI wrapper, not a prompt chain, not a demo — it's 97 FastAPI microservices running across 12 architectural layers with 16 specialist AI agents that coordinate, remember, feel pain, and improve themselves.

> *From Greek aither — the primordial god of light and the upper air. The invisible medium that makes creation possible.*

**AitherADK** is the standalone agent development kit extracted from this system. Build agents that work with any LLM backend, optionally connect back to the full AitherOS infrastructure.

---

## Architecture

```
LAYER 10  UI          AitherVeil (Next.js dashboard)
LAYER 9   TRAINING    Model lifecycle, daydream corpus, session harvesting
LAYER 8.5 MESH        Distributed node network, GPU sharing
LAYER 8   SECURITY    RBAC, secrets, flux monitoring, recovery
LAYER 7   AUTOMATION  Scheduler, demand, autonomic routines
LAYER 6   GPU         VRAM coordination, parallel inference, ComfyUI
LAYER 5   AGENTS      Agent council, forge dispatch, genesis orchestration
LAYER 3   COGNITION   Reasoning, judgment, flow control, will policies
LAYER 2   PERCEPTION  Voice, vision, portal, reflex
LAYER 1   CORE        Node, Pulse, Watch, MicroScheduler
LAYER 0   INFRA       Chronicle, Secrets, Nexus, Strata
```

**By the numbers:**
- 203 microservices (23 compound containers absorbing 81 sub-services)
- 65 Docker containers total
- 2600+ passing tests across 120+ test files
- 16 specialist agents with persistent identity
- 170+ PowerShell automation scripts
- 15 months of solo development

---

## What Makes It an "Agentic OS"

| Requirement | How AitherOS Does It |
|---|---|
| **Persistent Identity** | 16 agents with memory, personality, skills, and domain expertise |
| **Autonomous Action** | Dark factory loops — neuron firing, pain-driven remediation, model finetuning |
| **Multi-Agent Coordination** | Expedition Manager orchestrates multi-session projects with gate approvals |
| **Tool Use** | 100+ MCP tools, @tool decorator, function calling with any model |
| **Environmental Awareness** | Pain system, health monitoring, context tiers, VRAM coordination |
| **Self-Improvement** | Session learning, playbook auto-generation, model hot-swap |
| **Human Governance** | Frontier Judge quality gates, human-in-the-loop expedition approvals |

---

## The Agents

| Agent | Role | Specialty |
|-------|------|-----------|
| **Aither** | Orchestrator | System coordination, delegation, awareness synthesis |
| **Atlas** | Project Manager | Roadmaps, research delegation, executive reporting |
| **Demiurge** | Code Craftsman | Code generation, refactoring, architecture |
| **Lyra** | Researcher | Knowledge synthesis, deep-dive analysis |
| **Athena** | Security Oracle | Vulnerability analysis, threat assessment |
| **Hydra** | Code Guardian | Multi-perspective code review, quality assurance |
| **Prometheus** | Infra Titan | Infrastructure, deployment, scaling |
| **Apollo** | Performance | Optimization, benchmarking, profiling |
| **Iris** | Creative Muse | Image/video generation via ComfyUI |
| **Viviane** | Memory Guardian | Knowledge retrieval, context preservation |
| **Vera** | Content Creator | Writing, editing, social media |
| **Hera** | Community | Social engagement, publishing |
| **Morgana** | Secrets Keeper | Encryption, secure configuration |
| **Saga** | Documentation | Technical writing, knowledge base |
| **Themis** | Compliance | Ethics, fairness, policy enforcement |
| **Chaos** | Chaos Engineer | Resilience testing, failure injection |

All 16 identities ship with AitherADK as package data.

---

## AitherADK

The **Agent Development Kit** lets you build AI agents without the full AitherOS stack:

```python
from adk import AitherAgent, tool
from adk.llm import LLMRouter

# Custom tool
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Agent with any backend
agent = AitherAgent(
    "research-bot",
    identity="lyra",
    llm=LLMRouter(provider="openai", api_key="sk-..."),
)

response = await agent.run("Research AI agent frameworks in 2026")
```

**Features:**
- Multi-backend LLM (Ollama, OpenAI, Anthropic, vLLM, LM Studio)
- `@tool` decorator for function calling
- SQLite memory (conversations + KV store)
- OpenAI-compatible server
- MCP bridge to AitherOS tools via `mcp.aitherium.com`
- 16 pre-built agent identities
- Privacy-first opt-in telemetry

See the [ADK documentation](aither-adk/README.md) for full details.

---

## Key Systems

### Dark Factory
Autonomous loops that run without human intervention:
- **Neuron firing** — Proactive data gathering based on conversation patterns
- **Pain-driven remediation** — Detects failures, generates fixes, deploys patches
- **Session learning** — Extracts patterns from conversations, promotes to memory
- **Playbook auto-generation** — Successful operations become reusable playbooks
- **Model finetuning** — Daydream corpus feeds next training cycle

### Expedition Manager
Multi-session project orchestration:
- SOW analysis and phase decomposition
- Parallel task dispatch via AgentForge
- Human-in-the-loop gate approvals
- Security review, code review, deployment gates
- Cost/token tracking per phase

### Swarm Coding Engine
11 specialized agents in a 4-phase pipeline:
- ARCHITECT (design) -> SWARM (8 parallel workers) -> REVIEW -> JUDGE
- 3 coders, 2 testers, 2 security reviewers, 1 documentation writer

### Content Production Pipeline
Universal artifact factory:
- Presentations, articles, code projects, marketing, video, documents
- Agent dispatch, quality gates, assembly, delivery

---

## Hardware Profiles

| Profile | GPU VRAM | RAM | Models |
|---------|----------|-----|--------|
| **CPU Only** | None | 8 GB | Cloud API fallback |
| **Minimal** | 8-12 GB | 16 GB | llama3.2:3b, nomic-embed |
| **Standard** | 24 GB | 32 GB | llama3.1:8b, deepseek-r1:8b |
| **Workstation** | 48 GB+ | 64 GB | llama3.1:70b, deepseek-r1:14b |
| **Server** | 80 GB+ | 128 GB+ | Multi-model vLLM deployment |

The installer auto-detects your hardware and selects the right profile.

---

## Roadmap

### Done
- 203 microservices across 12 layers
- 16 specialist agents with persistent identity
- Dark factory autonomous loops (neuron, pain, learning, playbook, finetune)
- Expedition Manager for multi-session projects
- Swarm Coding Engine (11 agents, 4 phases)
- Content Production Pipeline
- Social graph (profiles, friends, groups)
- Package Manager (APM)
- MCP SaaS Gateway at mcp.aitherium.com
- RBAC with SQLite backend
- Multi-tenant isolation
- Pipeline prompt injection defense
- 2600+ passing tests

### Alpha (Now)
- AitherADK — pip-installable agent development kit
- AitherNode standalone mode with ADK server
- Desktop/Connect fallback to Node when Genesis unavailable
- Hardware profile auto-detection
- Auto-installer for drivers, CUDA, PyTorch, models
- Gateway at gateway.aitherium.com (Cloudflare Worker)

### Next
- Mobile companion app
- JS/TS SDK
- Federation between AitherOS instances
- Agent marketplace
- Kubernetes deployment option

---

## Stay Updated

1. **Star** this repository
2. **Watch** -> "Releases only" for minimal noise
3. **Sign up** at [aitherium.com](https://aitherium.com) for alpha access

Contact: hello@aitherium.com

---

## License

Source code will be released under a dual license:
- **AGPL-3.0** for open source use
- **Commercial license** for enterprise deployments

AitherADK is **Apache-2.0** (fully permissive).

---

*Built solo. Shipping real.*
