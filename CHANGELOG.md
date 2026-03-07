# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0-alpha.1] -- 2026-03-07

First public alpha release.

### Added

- **AitherADK**: Agent Development Kit -- 19 modules, 16 agent identities, pip-installable with 4 dependencies (httpx, pyyaml, fastapi, uvicorn)
- **Gateway**: gateway.aitherium.com -- authentication, agent mesh, MCP tool proxying, chat API, event streaming (13 lifecycle phases, 8 security layers)
- **MCP Server**: mcp.aitherium.com -- SSE endpoint configurable in Claude Code, Cursor, VS Code + Continue, Windsurf, and any MCP-compatible editor
- **Playground**: playground.aitherium.com -- browser-based chat interface with all 16 agents
- **AitherConnect**: Chrome browser extension for in-page agent assistance (connects via gateway or local instance)
- **AitherDesktop**: Native Windows application (PyQt6 WebView shell) with system tray, notifications, and offline package verification
- **Setup wizard**: `setup-vllm.py` -- interactive hardware-aware setup for 5 GPU tiers (nano, lite, standard, full, ollama)
- **11 hardware profiles**: minimal, standard, workstation, nvidia_low, nvidia_mid, nvidia_high, nvidia_ultra, server, amd, apple_silicon, cpu_only
- **Docker support**: Dockerfile + docker-compose.alpha.yml with optional vLLM GPU profile
- **7 examples**: hello_agent, custom_tools, openclaw_agent, openai_agent, multi_agent, federation_demo, full_lifecycle_test
- **OpenAI-compatible API**: `/v1/chat/completions`, `/v1/models`, `/chat`, `/health`, `/docs`
- **Multi-backend LLM routing**: Ollama, vLLM, OpenAI, Anthropic, LM Studio, llama.cpp, Groq, Together, and any OpenAI-compatible API
- **Streaming support**: Server-sent events for real-time response streaming
- **Local SQLite memory**: Conversation history and key-value store, no external database required
- **MCP bridge**: Connect to AitherOS cloud tools from local agents via `mcp.aitherium.com`
- **16 agent identities**: Aither (orchestrator), Atlas (project management), Demiurge (code), Lyra (research), Athena (security), Hydra (code review), Prometheus (infrastructure), Apollo (performance), Iris (creative), Viviane (memory), Vera (content), Hera (community), Morgana (secrets), Saga (documentation), Themis (compliance), Chaos (chaos engineering)
- **Privacy-first telemetry**: Opt-in, anonymized, no prompts or responses ever transmitted
- **Bug report CLI**: `aither-bug "description"` for easy issue reporting
- **Federation client**: Connect to AitherOS mesh for multi-instance communication

### Infrastructure

- 97 microservices across 12 architectural layers (core system, not yet publicly containerized)
- 2,600+ passing tests
- Apache-2.0 license
