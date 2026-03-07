# AitherADK — Agent Development Kit

Build AI agents that work with **any LLM backend** — Ollama, OpenAI, Anthropic, vLLM, LM Studio, or any OpenAI-compatible API.

```bash
pip install aither-adk
```

## Quick Start

```python
import asyncio
from adk import AitherAgent

async def main():
    agent = AitherAgent("aither")  # Auto-detects Ollama on localhost
    response = await agent.chat("Hello! What can you help me with?")
    print(response.content)

asyncio.run(main())
```

## Features

- **Multi-backend LLM** — Ollama, OpenAI, Anthropic, vLLM, LM Studio, llama.cpp, Groq, Together
- **16 agent identities** — Pre-built personas (orchestrator, researcher, coder, security, etc.)
- **Tool system** — `@tool` decorator for function calling with any model
- **Memory** — Local SQLite conversation history + key-value store
- **OpenAI-compatible server** — Serve any agent via `/v1/chat/completions`
- **MCP bridge** — Connect to AitherOS tools via `mcp.aitherium.com`
- **Privacy-first telemetry** — Opt-in, anonymized, no prompts/responses ever sent

## Choose Your Backend

```python
from adk import AitherAgent
from adk.llm import LLMRouter

# Ollama (auto-detected if running)
agent = AitherAgent("atlas")

# OpenAI
agent = AitherAgent("atlas", llm=LLMRouter(provider="openai", api_key="sk-..."))

# Anthropic
agent = AitherAgent("atlas", llm=LLMRouter(provider="anthropic", api_key="sk-ant-..."))

# vLLM / LM Studio / any OpenAI-compatible
agent = AitherAgent("atlas", llm=LLMRouter(
    provider="openai",
    base_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3.2-3B-Instruct",
))
```

## Add Tools

```python
from adk import AitherAgent, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = AitherAgent("atlas", tools=[get_global_registry()])
response = await agent.chat("What's 42 * 17?")  # Uses calculate tool automatically
```

## Serve as API

```bash
# CLI
aither-serve --identity aither --port 8080 --backend ollama

# Then use from any client:
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"hello"}]}'
```

The server exposes:
- `POST /chat` — Genesis-compatible format
- `POST /v1/chat/completions` — OpenAI-compatible format
- `GET /v1/models` — List available models
- `GET /health` — Health check
- `GET /docs` — Swagger UI

## Connect to AitherOS (Optional)

Use AitherOS tools without running the full stack:

```python
from adk.mcp import MCPBridge

bridge = MCPBridge(api_key="your-key")
tools = await bridge.list_tools()
result = await bridge.call_tool("explore_code", {"query": "agent dispatch"})

# Or register all MCP tools into your agent
await bridge.register_tools(agent)
```

## Agent Identities

16 pre-built identities ship with the package:

| Identity | Role | Best For |
|----------|------|----------|
| `aither` | Orchestrator | System coordination, delegation |
| `atlas` | Project Manager | Planning, tracking, reporting |
| `demiurge` | Code Craftsman | Code generation, refactoring |
| `lyra` | Researcher | Research, knowledge synthesis |
| `athena` | Security Oracle | Security audits, vulnerability analysis |
| `hydra` | Code Guardian | Code review, quality assurance |
| `prometheus` | Infra Titan | Infrastructure, deployment, scaling |
| `apollo` | Performance | Optimization, benchmarking |
| `iris` | Creative | Image generation, design |
| `viviane` | Memory | Knowledge retrieval, context |
| `vera` | Content | Writing, editing, social media |
| `hera` | Community | Social engagement, publishing |
| `morgana` | Secrets | Security, encryption |
| `saga` | Documentation | Technical writing |
| `themis` | Compliance | Ethics, policy, fairness |
| `chaos` | Chaos Engineer | Resilience testing |

## Examples

See the `examples/` directory:
- `hello_agent.py` — Minimal 20-line agent
- `custom_tools.py` — Agent with `@tool` functions
- `openclaw_agent.py` — Web research agent
- `openai_agent.py` — Using different LLM backends
- `multi_agent.py` — Two agents collaborating

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AITHER_LLM_BACKEND` | `auto` | Backend: `ollama`, `openai`, `anthropic`, `auto` |
| `AITHER_MODEL` | (auto) | Default model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | | OpenAI API key |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `AITHER_API_KEY` | | AitherOS gateway API key |
| `AITHER_PORT` | `8080` | Server port |
| `AITHER_PHONEHOME` | `false` | Enable opt-in telemetry |

## Bug Reports

```bash
# CLI
aither-bug "description of the issue"
aither-bug --dry-run  # See what would be sent

# Programmatic
await agent.report_bug("Tool X fails with Y error")
```

## License

Apache-2.0
