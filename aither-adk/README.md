# AitherOS Alpha

A standalone AI agent platform. Build agent fleets with **GPU-optimized local inference** — auto-detects your hardware, spins up vLLM containers with paged attention and continuous batching, and routes models by effort level.

One agent or twenty. **vLLM first**, Ollama fallback, cloud when needed. Your agents, your GPU, your rules.

**Works standalone. Works with Elysium. Works hybrid.** Start with Alpha on your laptop, connect to Elysium when you need the full stack — 97 microservices, training pipelines, mesh compute, and the Dark Factory. Alpha is the on-ramp.

```bash
pip install aither-adk
```

## Quick Start

### Single Agent

```python
import asyncio
from adk import AitherAgent

async def main():
    agent = AitherAgent("aither")  # Auto-detects vLLM/Ollama on localhost
    response = await agent.chat("Hello! What can you help me with?")
    print(response.content)

asyncio.run(main())
```

### Fleet Mode — Multiple Agents

```python
import asyncio
from adk.fleet import load_fleet

async def main():
    fleet = load_fleet(agent_names=["aither", "lyra", "demiurge", "hydra"])
    orchestrator = fleet.get_orchestrator()  # aither

    # Chat with the orchestrator — it can delegate to other agents
    response = await orchestrator.chat("Review the auth module for security issues")
    print(response.content)

    # Or talk to a specific agent directly
    lyra = fleet.get_agent("lyra")
    response = await lyra.chat("Research the latest trends in agent frameworks")
    print(response.content)

asyncio.run(main())
```

### Serve as API

```bash
# Single agent
aither-serve --identity aither --port 8080

# Fleet mode — multiple agents
aither-serve --agents aither,lyra,demiurge,hydra --port 8080

# Fleet from YAML config
aither-serve --fleet fleet.yaml --port 8080
```

## Fleet Mode

The key differentiator: any agent can call any other agent. When you create a fleet, every agent automatically gets `ask_agent` and `list_agents` tools.

### From the CLI

```bash
aither-serve --agents aither,lyra,demiurge,hydra,athena
```

### From a YAML file

```yaml
# fleet.yaml
name: my-fleet
orchestrator: aither    # gets all delegation requests by default
agents:
  - identity: aither
  - identity: lyra
  - identity: demiurge
  - identity: hydra
  - identity: athena
  - name: my-custom-agent
    system_prompt: "You are a specialized data analysis agent..."
```

```bash
aither-serve --fleet fleet.yaml
```

### Fleet API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agents` | GET | List all agents in the fleet |
| `/agents/{name}/chat` | POST | Chat with a specific agent |
| `/agents/{name}/sessions` | GET | List sessions for an agent |
| `/forge/dispatch` | POST | Dispatch via AgentForge (auto-routing) |
| `/chat` | POST | Chat with orchestrator (Genesis-compatible) |
| `/v1/chat/completions` | POST | OpenAI-compatible (routes to orchestrator) |

## Orchestration

Agents delegate to each other through the built-in `ask_agent` tool. When an agent needs help from a specialist, it calls `ask_agent("demiurge", "Write a Python function that...")` and gets the result back.

```python
from adk.forge import AgentForge, ForgeSpec

forge = AgentForge()

# Auto-route to best agent
result = await forge.dispatch(ForgeSpec(
    agent_type="auto",
    task="Review this code for security vulnerabilities: ...",
))
# Routes to athena based on keyword matching

# Explicit dispatch
result = await forge.dispatch(ForgeSpec(
    agent_type="demiurge",
    task="Refactor the auth module to use async/await",
    timeout=180.0,
))
```

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
    model="nvidia/Nemotron-Orchestrator-8B",
))
```

## Architecture

### Effort-Based Model Routing

AitherOS Alpha automatically selects the right model based on task complexity:

| Effort | vLLM (primary) | Ollama (fallback) | OpenAI | Anthropic | Use Case |
|--------|----------------|-------------------|--------|-----------|----------|
| 1-3 (small) | `Llama-3.2-3B` | `llama3.2:3b` | `gpt-4o-mini` | `claude-haiku` | Quick lookups, simple Q&A |
| 4-6 (medium) | `Nemotron-Orchestrator-8B` | `nemotron-orchestrator-8b` | `gpt-4o` | `claude-sonnet` | Most tasks, orchestration |
| 7-10 (large) | `deepseek-r1:14b` | `deepseek-r1:14b` | `o1` | `claude-opus` | Complex reasoning, code review |

### GPU Auto-Detection

`auto_setup()` detects your GPU and configures the optimal backend:

1. **NVIDIA + Docker** → Starts vLLM containers (paged attention, continuous batching, tensor parallelism)
2. **AMD / Apple Silicon / No Docker** → Falls back to Ollama
3. **No GPU** → Uses cloud APIs (gateway.aitherium.com or OpenAI/Anthropic direct)

```python
from adk.setup import auto_setup
report = await auto_setup()  # Detects GPU, starts vLLM, ready to go
```

### Core Components

```
AitherAgent          — Agent with identity, tools, memory, LLM
  AgentRegistry      — In-process registry of running agents
  AgentForge         — Dispatch agents by type or auto-route
  FleetConfig        — Multi-agent fleet from YAML or CLI
  ConversationStore  — JSON file persistence for conversations
  LLMRouter          — Multi-backend auto-detecting router
  Memory             — SQLite KV store + conversation history
  ToolRegistry       — @tool decorator, OpenAI function calling format
  Identity           — 16 YAML-based agent personas
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
response = await agent.chat("What's 42 * 17?")  # Uses calculate tool
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

## AitherOS Alpha vs Elysium

AitherOS Alpha is the standalone agent platform. **Elysium** is the full AitherOS deployment with 97 microservices. Alpha connects to Elysium when available but works completely standalone.

| Capability | Alpha (Standalone) | Elysium (Full AitherOS) |
|-----------|-------------------|------------------------|
| Agents | 16 identities, custom agents, fleet mode | 29 agent cards, full AgentKernel |
| Orchestration | In-process AgentForge, ask_agent delegation | SwarmCodingEngine (11 roles), Expeditions |
| LLM Routing | Ollama/OpenAI/Anthropic auto-detect, effort tiers | MicroScheduler VRAM coordination, vLLM multi-worker |
| Memory | SQLite KV + JSON conversation files | Unified knowledge graph, embeddings, MemoryGraph |
| Persistence | Local SQLite + JSON files (~/.aither/) | ConversationStore + crystallization + graph nodes |
| Tools | @tool decorator, tool registry | 100+ MCP tools, ToolGraph 3-tier, CodeGraph |
| Server | OpenAI-compatible API, fleet endpoints | Genesis orchestrator (97 microservices) |
| Training | -- | Prism, Trainer, Harvest, DaydreamCorpus |
| Creative | -- | ComfyUI, LTX video, Iris agent |
| Voice | -- | faster-whisper STT, Piper TTS |
| Autonomy | -- | Dark Factory, closed-loop learning |
| Security | -- | Full RBAC, capability tokens, HMAC-SHA256 |
| Multi-tenant | -- | Tenant isolation, caller context |
| Mesh | -- | AitherMesh, distributed compute, ExoNodes |
| Social | -- | MySpace pages, social graph, groups |
| Connect to Elysium | MCP bridge + federation client | N/A (IS Elysium) |

## Hardware Profiles

AitherOS Alpha auto-detects your hardware and selects the right models:

| Profile | GPU VRAM | Default Model | Reasoning Model | Coding Model |
|---------|----------|---------------|-----------------|--------------|
| `cpu_only` | None | Cloud (gateway) | Cloud | Cloud |
| `minimal` | 8-12 GB | `llama3.2:3b` | -- | -- |
| `nvidia_mid` | 8-12 GB | `nemotron-orchestrator-8b` | `deepseek-r1:8b` | -- |
| `nvidia_high` | 16-24 GB | `nemotron-orchestrator-8b` | `deepseek-r1:14b` | `qwen2.5-coder:14b` |
| `nvidia_ultra` | 32+ GB | `nemotron-orchestrator-8b` | `deepseek-r1:32b` | `qwen2.5-coder:32b` |
| `apple_silicon` | M1/M2/M3/M4 | `nemotron-orchestrator-8b` | `deepseek-r1:8b` | -- |
| `amd` | ROCm | `nemotron-orchestrator-8b` | `deepseek-r1:8b` | -- |

## Connect to Elysium

Alpha is designed as the gateway to Elysium. Three operating modes:

### Standalone (no Elysium needed)
Everything runs locally — agents, LLM, memory, tools. Zero network dependencies.

### Hybrid (best of both worlds)
Run agents locally but use Elysium for the heavy lifting — MCP tools, knowledge graph, training data, mesh compute. Your agents keep local autonomy but gain access to 100+ tools and the full AitherOS infrastructure.

```python
from adk import AitherAgent
from adk.mcp import MCPBridge

# Create a local agent
agent = AitherAgent("atlas")

# Connect to Elysium's MCP tools
bridge = MCPBridge(api_key="your-key")
await bridge.register_tools(agent)  # Now your agent has 100+ Elysium tools

# Agent can now use explore_code, query_memory, get_system_status, etc.
response = await agent.chat("Search the codebase for authentication bugs")
```

### Full Federation (join the mesh)
Register your Alpha node with Elysium. Your agents appear in the mesh, can receive delegated tasks, and contribute compute.

```python
from adk import connect_federation

fed = connect_federation(host="http://elysium.local")
await fed.register("my-alpha-node", api_key="your-key")
await fed.join_mesh(capabilities=["text_gen", "code_review"])

# Your agents are now part of the Elysium fleet
status = await fed.get_system_status()
```

### Gateway Inference
No local GPU? Use the AitherOS gateway for inference — same API, cloud-hosted models.

```bash
export AITHER_API_KEY=your-key
aither-serve --identity aither  # Uses gateway.aitherium.com for LLM
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AITHER_LLM_BACKEND` | `auto` | Backend: `ollama`, `openai`, `anthropic`, `auto` |
| `AITHER_MODEL` | (auto) | Default model name |
| `AITHER_PREFER_LOCAL` | `false` | Try Ollama before gateway |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | | OpenAI API key |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `AITHER_API_KEY` | | AitherOS gateway API key |
| `AITHER_PORT` | `8080` | Server port |
| `AITHER_HOST` | `0.0.0.0` | Server bind address |
| `AITHER_DATA_DIR` | `~/.aither` | Data directory for memory/conversations |
| `AITHER_PHONEHOME` | `false` | Enable opt-in telemetry |

## Examples

See the `examples/` directory:
- `hello_agent.py` — Minimal 20-line agent
- `custom_tools.py` — Agent with `@tool` functions
- `openclaw_agent.py` — Web research agent
- `openai_agent.py` — Using different LLM backends
- `multi_agent.py` — Two agents collaborating
- `federation_demo.py` — Connecting to Elysium

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
