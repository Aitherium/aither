# Getting Started with AitherOS

## Quick Start (ADK Only)

The fastest way to start building with AitherOS agents:

```bash
# Install the Agent Development Kit
pip install aither-adk

# Start a server with the default agent
aither-serve --identity aither --port 8080
```

Your agent is now running at `http://localhost:8080` with:
- `POST /chat` — Genesis-compatible chat
- `POST /v1/chat/completions` — OpenAI-compatible chat
- `GET /v1/models` — List available models
- `GET /health` — Health check
- `GET /docs` — Swagger UI

### Talk to Your Agent

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'
```

### Build a Custom Agent

```python
import asyncio
from adk import AitherAgent, tool

@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

async def main():
    agent = AitherAgent("my-agent", identity="lyra")

    @agent.tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))

    response = await agent.chat("What's 42 * 17?")
    print(response.content)

asyncio.run(main())
```

## Choose Your LLM Backend

### Ollama (Local, Free)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
aither-serve  # Auto-detects Ollama
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
aither-serve --backend openai
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
aither-serve --backend anthropic
```

### vLLM / LM Studio / Any OpenAI-Compatible

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
aither-serve --backend openai
```

## Full AitherOS Stack (Docker)

For the complete 97-service deployment:

```bash
git clone https://github.com/Aitherium/AitherOS-Alpha.git
cd AitherOS-Alpha

# Auto-detect hardware and install dependencies
python install.py

# Start core services
docker compose -f docker-compose.aitheros.yml up -d

# Dashboard
open http://localhost:3000
```

### Hardware Requirements

| Profile | GPU VRAM | RAM | What You Get |
|---------|----------|-----|--------------|
| **CPU Only** | None | 8 GB | ADK server with cloud LLM fallback |
| **Minimal** | 8 GB | 16 GB | Local inference with llama3.2:3b |
| **Standard** | 24 GB | 32 GB | Full stack with 8B models |
| **Workstation** | 48 GB+ | 64 GB | Multi-model with 70B models |
| **Server** | 80 GB+ | 128 GB+ | Full vLLM multi-worker deployment |

## Connect to AitherOS Cloud (Optional)

Use AitherOS tools without running the full stack:

```python
from adk.mcp import MCPBridge

bridge = MCPBridge(api_key="your-key")  # From gateway.aitherium.com
tools = await bridge.list_tools()
result = await bridge.call_tool("explore_code", {"query": "agent dispatch"})
```

## Desktop App

AitherDesktop connects to your local agent server automatically:

1. Install AitherDesktop
2. It tries Genesis:8001 first (full mode)
3. Falls back to AitherNode:8090 (standalone mode)
4. All features work in both modes

## Browser Extension

AitherConnect adds AI to your browser:

1. Install AitherConnect from the Chrome Web Store
2. Enable "Standalone Mode" in options to use your local ADK server
3. Sidepanel chat works with any running aither-serve instance

## Report Issues

```bash
# Built-in bug reporter
aither-bug "description of the issue"

# Or programmatically
await agent.report_bug("Tool X fails with Y error")
```

## Next Steps

- Browse the [examples/](aither-adk/examples/) directory
- Read the [ADK README](aither-adk/README.md)
- Check the [ROADMAP](ROADMAP.md)
- Sign up at [aitherium.com](https://aitherium.com)

Contact: hello@aitherium.com
