<p align="center">
  <img src="https://raw.githubusercontent.com/Aitherium/AitherOS-Alpha/main/assets/logo.png" alt="AitherOS" width="200"/>
</p>

<h1 align="center">AitherOS</h1>

<p align="center">
  <strong>A full-stack agentic operating system.</strong><br/>
  97 microservices. 16 specialist agents. 2,600+ passing tests.<br/>
  Built by one person over 15 months.
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/status-alpha-orange?style=flat-square" alt="Alpha"/></a>
  <a href="#"><img src="https://img.shields.io/badge/services-97-blue?style=flat-square" alt="97 Services"/></a>
  <a href="#"><img src="https://img.shields.io/badge/agents-16-blue?style=flat-square" alt="16 Agents"/></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-2600%2B_passing-green?style=flat-square" alt="2600+ Tests"/></a>
  <a href="#"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License"/></a>
  <a href="https://pypi.org/project/aither-adk/"><img src="https://img.shields.io/pypi/v/aither-adk?style=flat-square&label=aither-adk" alt="PyPI"/></a>
</p>

<p align="center">
  <a href="https://playground.aitherium.com">Playground</a> |
  <a href="https://mcp.aitherium.com">MCP Server</a> |
  <a href="https://gateway.aitherium.com">Gateway</a> |
  <a href="https://chat.aitherium.com">Community</a> |
  <a href="docs/GETTING_STARTED.md">Get Started</a>
</p>

---

## What Is This?

AitherOS is an AI-powered operating system built on 97 microservices organized in 12 architectural layers. It runs 16 specialist AI agents that collaborate to handle everything from code generation and security audits to infrastructure management and creative tasks.

The **AitherADK** (Agent Development Kit) is the open-source entry point. It lets you run any of the 16 agents locally with 4 pip dependencies, connect to cloud tools through the gateway, or self-host the full stack on your own hardware.

This is the alpha release. Everything works. Some edges are rough.

---

## Quick Start

Three ways to get running, from simplest to most powerful.

### Path A: pip install (simplest)

```bash
pip install aither-adk
aither-serve --identity atlas --port 8080
```

Your agent is now live at `http://localhost:8080`. It auto-detects Ollama if running, or falls back to a built-in response. See the [Getting Started guide](docs/GETTING_STARTED.md) for full setup including Ollama.

### Path B: Docker

```bash
git clone https://github.com/Aitherium/AitherOS-Alpha.git
cd AitherOS-Alpha
docker compose -f docker/docker-compose.alpha.yml up -d
```

Serves an agent on port 8080. Connects to Ollama on your host machine by default. See [Self-Hosting](docs/SELF_HOSTING.md) for GPU-accelerated vLLM setup.

### Path C: Cloud (no local setup)

Connect to **gateway.aitherium.com** and use AitherOS tools directly from your IDE or the browser-based playground. No local install required.

- **Playground**: [playground.aitherium.com](https://playground.aitherium.com) -- chat with agents in your browser
- **MCP Server**: Add `mcp.aitherium.com` to your IDE (see below)
- **API**: `POST https://gateway.aitherium.com/api/v1/chat` with your API key

---

## MCP Server

The AitherOS MCP server at **mcp.aitherium.com** gives your IDE access to AitherOS tools -- code search, agent delegation, memory, and more. It works with any editor that supports the Model Context Protocol.

### Claude Code

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "aitheros": {
      "type": "sse",
      "url": "https://mcp.aitherium.com/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "aitheros": {
      "type": "sse",
      "url": "https://mcp.aitherium.com/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

### VS Code + Continue

Add to your Continue configuration (`~/.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "aitheros",
      "transport": {
        "type": "sse",
        "url": "https://mcp.aitherium.com/sse",
        "headers": {
          "Authorization": "Bearer YOUR_API_KEY"
        }
      }
    }
  ]
}
```

### Windsurf

Add to your Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "aitheros": {
      "serverUrl": "https://mcp.aitherium.com/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

### Generic / Other Editors

Any MCP-compatible client can connect to the SSE endpoint:

```
URL: https://mcp.aitherium.com/sse
Auth: Bearer token in Authorization header
```

Get an API key by registering at [gateway.aitherium.com](https://gateway.aitherium.com) or through the [Playground](https://playground.aitherium.com).

---

## Aither Playground

[playground.aitherium.com](https://playground.aitherium.com) is a browser-based chat interface for talking to AitherOS agents. No install, no API key management -- just open the page and start a conversation.

The playground connects to the same gateway infrastructure that powers the MCP server and API, so you get the full agent experience including tool use, memory, and multi-agent delegation.

---

## AitherConnect

AitherConnect is a Chrome extension that brings AitherOS into your browser. It connects through the gateway (or to a local AitherOS instance) and provides:

- In-page agent assistance
- Context-aware tool access
- Quick access to any of the 16 agents

Install from the Chrome Web Store or build from source in the `connect/` directory.

---

## Community

Join the conversation at [chat.aitherium.com](https://chat.aitherium.com) -- an RCS-style chat and forum for AitherOS users and contributors. Discuss features, share agent configurations, report issues, and connect with other builders.

---

## Architecture

```
LAYER 10  UI            AitherVeil (web dashboard)
LAYER 9   TRAINING      Model fine-tuning, data harvesting
LAYER 8.5 MESH          Service mesh, deployment, external gateway
LAYER 8   SECURITY      Identity, auth, secrets, recovery
LAYER 7   AUTOMATION    Scheduling, demand routing, autonomic ops
LAYER 6   GPU           VRAM coordination, acceleration, offload
LAYER 5   AGENTS        Council, agent-to-agent mesh, orchestrator
LAYER 3   COGNITION     Reasoning, judgment, intent, flow control
LAYER 2   PERCEPTION    Voice, vision, portal, reflexes
LAYER 1   CORE          Node, Pulse (heartbeat), Watch, MicroScheduler
LAYER 0   INFRA         Chronicle (logging), Secrets, Nexus, Strata
```

97 microservices across 12 layers. 16 specialist agents. All services are defined in a single source-of-truth configuration file and boot in dependency order through a topological sort.

---

## The 16 Agents

| Agent | Domain | Description |
|-------|--------|-------------|
| **Aither** | Orchestrator | System overseer -- coordination, synthesis, delegation |
| **Atlas** | Project Management | Planning, tracking, service discovery, reporting |
| **Demiurge** | Code | Code generation, refactoring, architecture |
| **Lyra** | Research | Knowledge synthesis, deep research, analysis |
| **Athena** | Security | Security audits, vulnerability analysis, threat modeling |
| **Hydra** | Code Review | Quality assurance, test coverage, code review |
| **Prometheus** | Infrastructure | Deployment, scaling, infrastructure management |
| **Apollo** | Performance | Optimization, benchmarking, profiling |
| **Iris** | Creative | Image generation, design, visual content |
| **Viviane** | Memory | Knowledge retrieval, context management, recall |
| **Vera** | Content | Writing, editing, social media, content strategy |
| **Hera** | Community | Social engagement, publishing, outreach |
| **Morgana** | Secrets | Encryption, key management, secure operations |
| **Saga** | Documentation | Technical writing, API docs, guides |
| **Themis** | Compliance | Ethics review, policy enforcement, fairness |
| **Chaos** | Chaos Engineering | Resilience testing, failure injection, recovery |

Every agent has a distinct identity (system prompt, skills, personality) defined in YAML. You can use any of them through the ADK, the MCP server, or the playground.

---

## What Works Offline vs Connected

| Feature | Offline (ADK only) | Connected (Gateway) |
|---------|-------------------|-------------------|
| Chat with agents | Yes (local LLM) | Yes (cloud + local) |
| Tool calling | Yes (local tools) | Yes (100+ MCP tools) |
| Conversation memory | Yes (local SQLite) | Yes (persistent graph) |
| Code search | No | Yes (CodeGraph) |
| Multi-agent delegation | No | Yes (AgentForge) |
| Model fine-tuning | No | Yes (training pipeline) |
| Web dashboard | No | Yes (AitherVeil) |
| OpenAI-compatible API | Yes | Yes |
| MCP server | No | Yes (mcp.aitherium.com) |
| Agent identities | Yes (all 16) | Yes (all 16) |

---

## AitherDesktop

AitherDesktop is a native Windows application (PyQt6 WebView shell) that provides a desktop-native interface to AitherOS. It connects to a local AitherOS instance or to the gateway for cloud access.

Features:
- System tray integration
- Native notifications
- Offline package verification
- Auto-update

---

## Hardware Profiles

The setup wizard (`python setup-vllm.py`) detects your hardware and recommends the best configuration. Five GPU tiers are supported for vLLM, plus an Ollama-only mode that works on any hardware:

| Tier | VRAM | Backend | Models | Use Case |
|------|------|---------|--------|----------|
| **nano** | 8-12 GB | vLLM | Qwen3-8B | Budget GPU, basic chat |
| **lite** | 12-16 GB | vLLM | Nemotron Orchestrator 8B | Mid-range, good tool use |
| **standard** | 20-24 GB | vLLM | Orchestrator + DeepSeek R1 14B | Enthusiast, chat + reasoning |
| **full** | 24+ GB | vLLM | Orchestrator + R1 + Embeddings | Full stack, all capabilities |
| **ollama** | Any | Ollama | Your choice | Any GPU (NVIDIA/AMD/Apple) or CPU |

11 hardware profiles are included for fine-grained configuration: minimal, standard, workstation, nvidia_low, nvidia_mid, nvidia_high, nvidia_ultra, server, amd, apple_silicon, cpu_only.

See [Self-Hosting](docs/SELF_HOSTING.md) for detailed setup instructions.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Step-by-step quickstart guide |
| [Gateway](docs/GATEWAY.md) | Cloud gateway connection and MCP setup |
| [Self-Hosting](docs/SELF_HOSTING.md) | Running AitherOS on your own hardware |
| [Roadmap](ROADMAP.md) | What is done, what is next |
| [Changelog](CHANGELOG.md) | Release notes |
| [ADK README](aither-adk/README.md) | AitherADK package documentation |

---

## Contributing

AitherOS is in alpha. Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests: `cd aither-adk && python -m pytest tests/`
5. Submit a pull request

For bugs, use `aither-bug "description"` (ships with the ADK) or open a GitHub issue.

Join the discussion at [chat.aitherium.com](https://chat.aitherium.com).

---

## License

Apache-2.0. See [LICENSE](LICENSE) for the full text.

---

<p align="center">
  <a href="https://github.com/Aitherium/AitherOS-Alpha">Star this repo</a> to follow development.
  <br/>
  <a href="https://github.com/Aitherium/AitherOS-Alpha/subscription">Watch</a> for release notifications.
</p>
