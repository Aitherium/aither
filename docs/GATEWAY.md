# AitherOS Gateway

The AitherOS gateway at **gateway.aitherium.com** is the cloud bridge between your local agents and the full AitherOS platform. It handles authentication, MCP tool access, agent mesh communication, telemetry, and event streaming.

The gateway runs 13 lifecycle phases and 8 security layers. You do not need to understand any of that to use it.

---

## What Is the Gateway?

When you run an agent locally with `aither-serve`, it works standalone with your local LLM. Connecting to the gateway adds:

- **MCP tools** -- 100+ tools for code search, memory, agent delegation, and more
- **Multi-agent mesh** -- Your local agent can delegate tasks to other agents in the cloud
- **Persistent memory** -- Conversations and knowledge survive restarts
- **Telemetry** -- Anonymized, opt-in usage data that helps improve the system
- **Event streaming** -- Real-time events from the AitherOS platform

The gateway is optional. Everything works offline without it.

---

## Registration

### Via the Playground

1. Go to [playground.aitherium.com](https://playground.aitherium.com)
2. Click "Sign Up"
3. Enter your email and password
4. Verify your email
5. Your API key is available in the dashboard

### Via the API

```bash
curl -X POST https://gateway.aitherium.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "your-password"}'
```

---

## API Key Management

Your API key authenticates all requests to the gateway, MCP server, and API endpoints.

### Get your key

After registration, retrieve your key:

```bash
curl -X POST https://gateway.aitherium.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "your-password"}'
```

The response includes your API key and clearance level.

### Use your key

Set it as an environment variable:

```bash
export AITHER_API_KEY=your-key-here
```

Or pass it directly:

```bash
aither-serve --identity atlas --port 8080
# The server reads AITHER_API_KEY from the environment
```

In API calls, pass it as a Bearer token:

```bash
curl -X POST https://gateway.aitherium.com/api/v1/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

---

## What the Gateway Provides

### Chat

Send messages to any of the 16 agents:

```bash
curl -X POST https://gateway.aitherium.com/api/v1/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the security of this API design", "agent": "athena"}'
```

### MCP Tools

Access AitherOS tools from any MCP-compatible editor. See the MCP Server section below.

### Agent Mesh

Your local agent can dispatch tasks to cloud agents:

```python
from adk.gateway import GatewayClient

client = GatewayClient(api_key="your-key")
result = await client.dispatch("demiurge", "Refactor this function for readability")
```

### Telemetry

Opt-in, anonymized usage data. No prompts, responses, or personal data are ever transmitted. Enable with:

```bash
export AITHER_PHONEHOME=true
```

### Events

Stream real-time events from the platform:

```bash
curl -N https://gateway.aitherium.com/api/v1/events/stream \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## MCP Server

The MCP server at **mcp.aitherium.com** exposes AitherOS tools to your IDE through the Model Context Protocol. It is a separate endpoint from the gateway API, but uses the same API key.

### Endpoint

```
SSE: https://mcp.aitherium.com/sse
Auth: Bearer token in Authorization header
```

### IDE Configuration

#### Claude Code

`.mcp.json` in your project root:

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

#### Cursor

`.cursor/mcp.json`:

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

#### VS Code + Continue

`~/.continue/config.json`:

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

#### Windsurf

Windsurf MCP configuration:

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

#### Other Editors

Any MCP-compatible client can connect to `https://mcp.aitherium.com/sse` with a Bearer token.

### Available Tool Categories

| Category | Examples |
|----------|---------|
| Code search | `explore_code`, `codegraph_search`, `codegraph_get_context` |
| Agent dispatch | `ask_agent`, `forge_subagent`, `swarm_code` |
| Memory | `query_memory`, `recall`, `remember` |
| File operations | `fs_read_file`, `fs_write_file` |
| System | `get_system_status`, `get_service_status` |
| Git | `git_status`, `git_diff`, `git_add`, `git_commit` |
| Search | `web_search`, `unified_search` |

---

## Aither Playground

[playground.aitherium.com](https://playground.aitherium.com) is a browser-based chat interface. It connects to the gateway and provides:

- Conversation with any of the 16 agents
- Tool use (agents can search code, read files, run commands)
- Persistent conversation history
- No local setup required

The playground is free to use during the alpha period.

---

## AitherConnect

AitherConnect is a Chrome extension that integrates AitherOS into your browser. It can connect to:

- **The gateway** (cloud mode) -- works anywhere, needs an API key
- **A local AitherOS instance** (local mode) -- for self-hosted setups

Install from the Chrome Web Store or build from source.

---

## Rate Limits

Default rate limits during the alpha period:

| Limit | Value |
|-------|-------|
| Requests per minute | 60 |
| Requests per day | 5,000 |
| Max message length | 32,000 characters |
| Max concurrent sessions | 5 |

These limits may change. Higher limits are available at higher clearance levels.

---

## Privacy

AitherOS takes privacy seriously. Here is what flows where:

### Sent to the gateway

- Your API key (for authentication)
- Chat messages you send to cloud agents
- MCP tool calls and their parameters
- Opt-in telemetry (if enabled): anonymized usage stats, no message content

### Stays local

- Messages to your local agent (when not using the gateway)
- Local SQLite memory and conversation history
- Your Ollama models and configuration
- Any files on your machine

### Never collected

- Prompts or responses (unless you explicitly send them to a cloud agent)
- Personal data beyond your registration email
- Browsing history, clipboard, or system information

---

## Clearance Levels

Gateway access is organized into clearance levels:

| Level | Access |
|-------|--------|
| **OBSERVER** | Read-only API access, playground chat, basic MCP tools |
| **CONTRIBUTOR** | Full chat, all MCP tools, agent dispatch, memory |
| **OPERATOR** | Service management, deployment tools, advanced operations |
| **ADMIN** | Full platform access, user management, configuration |

New registrations start at OBSERVER. Clearance upgrades are granted based on usage and contribution.

---

## Troubleshooting

### "Unauthorized" (401)

Your API key is missing or invalid. Check that:
- The `AITHER_API_KEY` environment variable is set
- The key has not been revoked
- You are using `Bearer YOUR_KEY` in the Authorization header

### "Rate limited" (429)

You have exceeded the request limit. Wait and retry. Check the `Retry-After` header for timing.

### MCP tools not appearing in IDE

1. Verify your API key works: `curl -H "Authorization: Bearer YOUR_KEY" https://mcp.aitherium.com/sse`
2. Restart your IDE after changing MCP configuration
3. Check that the MCP config file is in the correct location for your editor

### Gateway unreachable

The gateway is at `gateway.aitherium.com`. If it is down, your local agent continues to work normally -- it just loses access to cloud tools and memory.
