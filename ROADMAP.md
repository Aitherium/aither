# AitherOS Roadmap

---

## Done (Alpha -- shipped 2026-03-07)

- AitherADK: Agent Development Kit (19 modules, 16 identities, pip-installable, 4 dependencies)
- Gateway: gateway.aitherium.com (auth, mesh, MCP tools, chat, 13 lifecycle phases, 8 security layers)
- MCP Server: mcp.aitherium.com (configurable in Claude Code, Cursor, VS Code, Windsurf, and any SSE-compatible editor)
- Playground: playground.aitherium.com (browser-based chat with all 16 agents)
- AitherConnect: Chrome extension for browser integration
- AitherDesktop: Native Windows app (PyQt6 WebView shell)
- Setup wizard: `setup-vllm.py` with 5 GPU tiers (nano, lite, standard, full, ollama)
- 11 hardware profiles (minimal through nvidia_ultra, plus AMD, Apple Silicon, CPU-only)
- Docker support (Dockerfile + compose with optional vLLM GPU profile)
- 7 working examples (hello agent, custom tools, OpenClaw, OpenAI backend, multi-agent, federation, lifecycle test)
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Multi-backend LLM routing (Ollama, vLLM, OpenAI, Anthropic, LM Studio, any OpenAI-compatible)
- Streaming support
- Local SQLite memory (conversation history + key-value store)
- MCP bridge for cloud tool access
- Privacy-first opt-in telemetry
- Bug report CLI (`aither-bug`)
- 97 microservices across 12 architectural layers
- 16 specialist AI agents
- 2,600+ passing tests
- Apache-2.0 license

---

## Alpha (current release)

What ships today:

- The ADK works standalone with any LLM backend
- The gateway handles auth, chat, and MCP tool proxying
- The MCP server is live and configurable in major IDEs
- The playground is live for browser-based interaction
- Self-hosting works with Ollama or vLLM on your own hardware
- AitherDesktop connects to local or cloud instances
- All 16 agent identities are included and functional

Known limitations:

- The full 97-service stack is not yet containerized for public deployment
- AitherNode (the 100+ tool MCP server) runs inside the full stack only
- Some advanced features (training pipeline, multi-agent swarm, expedition system) are internal-only for now
- Documentation is improving continuously

---

## Next (Beta)

- **AitherNode Docker image** -- The full 100+ tool MCP server as a standalone container
- **Full stack containers** -- Docker Compose for all 97 services (tiered profiles for different hardware)
- **Mobile app** -- iOS and Android companion
- **ComfyUI model wizard** -- Guided setup for image/video generation models
- **Package marketplace** -- Install and share agent packages, tools, and workflows
- **AitherVeil themes** -- Customizable web dashboard
- **Federation protocol** -- Connect multiple AitherOS instances into a mesh
- **Improved onboarding** -- Interactive tutorial in the playground

---

## Future

- **Distributed training** -- Fine-tune models across federated nodes
- **Edge deployment** -- Run agents on Raspberry Pi, Jetson, and other edge devices
- **Multi-tenant SaaS** -- Hosted AitherOS for teams and organizations
- **Plugin SDK** -- Third-party service integrations
- **Voice interface** -- Local STT/TTS with VRAM coordination
- **Autonomous operations** -- Full dark factory mode for self-managing infrastructure
