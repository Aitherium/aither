# Getting Started with AitherOS

This guide gets you from zero to a working AI agent in under 5 minutes.

---

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- **Ollama** (recommended for local inference) -- [ollama.com](https://ollama.com)
- OR an API key for OpenAI, Anthropic, or any OpenAI-compatible provider

---

## 1. Install the ADK

```bash
pip install aither-adk
```

This installs the AitherOS Agent Development Kit with 4 dependencies (httpx, pyyaml, fastapi, uvicorn). No GPU drivers, no CUDA, no heavy frameworks.

For cloud LLM providers, install optional extras:

```bash
pip install aither-adk[openai]      # OpenAI support
pip install aither-adk[anthropic]   # Anthropic support
pip install aither-adk[all]         # Both
```

---

## 2. Start Ollama

If you want to run models locally (recommended):

```bash
ollama serve
ollama pull llama3.2:3b
```

The `llama3.2:3b` model needs about 2 GB of RAM and runs on any modern machine. For better quality, pull a larger model:

```bash
ollama pull llama3.1:8b        # General purpose (5 GB)
ollama pull deepseek-r1:14b    # Deep reasoning (9 GB)
ollama pull qwen2.5-coder:7b   # Code generation (5 GB)
```

Skip this step if you plan to use a cloud provider instead.

---

## 3. Run an Agent

```bash
aither-serve --identity atlas --port 8080
```

This starts the Atlas agent (project management and planning) as an OpenAI-compatible API server. It auto-detects Ollama on localhost.

To use a different agent, swap the identity:

```bash
aither-serve --identity demiurge --port 8080   # Code generation
aither-serve --identity lyra --port 8080       # Research
aither-serve --identity athena --port 8080     # Security
```

All 16 identities are listed in the [main README](../README.md#the-16-agents).

---

## 4. Test It

### Health check

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{"status": "healthy", "identity": "atlas", "version": "0.1.0a1"}
```

### Chat (Genesis format)

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you help me with?"}'
```

### Chat (OpenAI-compatible format)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello, who are you?"}]
  }'
```

### List models

```bash
curl http://localhost:8080/v1/models
```

### Swagger docs

Open [http://localhost:8080/docs](http://localhost:8080/docs) in your browser for the interactive API reference.

---

## 5. Connect to the Gateway (optional)

The AitherOS gateway at **gateway.aitherium.com** gives your local agent access to cloud tools, persistent memory, and multi-agent delegation.

1. Register at [playground.aitherium.com](https://playground.aitherium.com) or via the API
2. Get your API key from the dashboard
3. Set the environment variable and restart:

```bash
export AITHER_API_KEY=your-key-here
aither-serve --identity atlas --port 8080
```

Your agent now has access to AitherOS MCP tools through the gateway. See [GATEWAY.md](GATEWAY.md) for details.

---

## 6. Try the Playground

[playground.aitherium.com](https://playground.aitherium.com) is a browser-based chat interface. No install needed. Open the page, pick an agent, and start talking.

The playground is connected to the same infrastructure as the gateway and MCP server, so it has full access to all 16 agents and their tools.

---

## 7. Add MCP to Your IDE

The MCP server at **mcp.aitherium.com** lets your code editor use AitherOS tools (code search, agent dispatch, memory queries, and more).

### Claude Code

Add to `.mcp.json` in your project root:

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

Add to `~/.continue/config.json`:

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

Add to your Windsurf MCP config:

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

---

## 8. Docker Alternative

If you prefer containers:

```bash
git clone https://github.com/Aitherium/AitherOS-Alpha.git
cd AitherOS-Alpha

# Copy and edit environment config
cp docker/.env.example docker/.env

# Start the ADK server
docker compose -f docker/docker-compose.alpha.yml up -d

# Check health
curl http://localhost:8080/health
```

The Docker setup assumes Ollama is running on the host. The container reaches it via `host.docker.internal`. Edit `docker/.env` to switch backends or configure API keys.

For GPU users with vLLM:

```bash
docker compose -f docker/docker-compose.alpha.yml --profile gpu up -d
```

---

## 9. Use as a Python Library

The ADK is also a Python library you can import directly:

```python
import asyncio
from adk import AitherAgent

async def main():
    agent = AitherAgent("atlas")
    response = await agent.chat("Summarize the key risks in this project.")
    print(response.content)

asyncio.run(main())
```

Add custom tools:

```python
from adk import AitherAgent, tool

@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    return open(path).read()

agent = AitherAgent("demiurge", tools=[read_file])
response = await agent.chat("Review the code in main.py")
```

See the `examples/` directory for more:

| Example | Description |
|---------|-------------|
| `hello_agent.py` | Minimal 20-line agent |
| `custom_tools.py` | Agent with `@tool` functions |
| `openclaw_agent.py` | Web research agent |
| `openai_agent.py` | Using different LLM backends |
| `multi_agent.py` | Two agents collaborating |
| `federation_demo.py` | Connecting to AitherOS mesh |
| `full_lifecycle_test.py` | End-to-end lifecycle test |

---

## Next Steps

- **[GATEWAY.md](GATEWAY.md)** -- Connect to the cloud gateway, configure MCP, manage API keys
- **[SELF_HOSTING.md](SELF_HOSTING.md)** -- Run the full stack on your own hardware with vLLM
- **[ROADMAP.md](../ROADMAP.md)** -- See what is shipping next
- **[chat.aitherium.com](https://chat.aitherium.com)** -- Join the community

---

## Troubleshooting

### "Connection refused" on port 8080

The server is not running. Check that `aither-serve` started without errors. If using Docker, check `docker logs adk-server`.

### "No LLM backend available"

No Ollama detected and no cloud API key set. Either:
- Start Ollama: `ollama serve`
- Or set a cloud key: `export OPENAI_API_KEY=sk-...`

### Ollama model not found

Pull the model first: `ollama pull llama3.2:3b`

### Port already in use

Change the port: `aither-serve --port 8081`

### Docker cannot reach Ollama

Make sure Ollama is listening on all interfaces: `OLLAMA_HOST=0.0.0.0 ollama serve`
