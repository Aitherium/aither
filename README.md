# AitherADK — Agent Development Kit

**Build multi-agent systems that run on your hardware.**

```bash
pip install aither-adk
aither init my-agent && cd my-agent && python agent.py
```

AitherADK gives you 48 specialized agent identities, effort-based model routing, knowledge graph memory, and fleet orchestration. Works with vLLM, Ollama, OpenAI, Anthropic — same code, any backend.

## Quick Start

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
from adk.fleet import load_fleet

fleet = load_fleet(agent_names=["aither", "lyra", "demiurge", "hydra"])
orchestrator = fleet.get_orchestrator()
response = await orchestrator.chat("Review the auth module for security issues")
# aither delegates to hydra (code review) and athena (security) automatically
```

### Serve as API

```bash
# Single agent — OpenAI-compatible endpoint
aither-serve --identity aither --port 8080

# Fleet of specialists
aither-serve --agents aither,lyra,demiurge,hydra,athena --port 8080

# Drop-in replacement for OpenAI
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"aither","messages":[{"role":"user","content":"hello"}]}'
```

## Choose Your Backend

```python
from adk import AitherAgent
from adk.llm import LLMRouter

# Ollama (auto-detected if running)
agent = AitherAgent("atlas")

# vLLM / LM Studio / any OpenAI-compatible
agent = AitherAgent("atlas", llm=LLMRouter(
    provider="openai",
    base_url="http://localhost:8120/v1",
    model="aither-orchestrator",
))

# OpenAI
agent = AitherAgent("atlas", llm=LLMRouter(provider="openai", api_key="sk-..."))

# Anthropic
agent = AitherAgent("atlas", llm=LLMRouter(provider="anthropic", api_key="sk-ant-..."))
```

## Local Inference with vLLM + TurboQuant

For maximum performance, run vLLM with [aither-kvcache](https://github.com/Aitherium/aitherkvcache) — sub-byte KV cache compression that fits 3.8x more KV data per byte than FP16 (up to 7.1x at 2-bit).

```bash
# Auto-setup: detects GPU, pulls models, starts vLLM containers
from adk.setup import auto_setup
report = await auto_setup()

# Or manually with Docker:
docker compose -f docker-compose.adk-vllm.yml up -d

# vLLM with TurboQuant (4-bit KV cache):
# --attention-backend CUSTOM activates TQ compressed storage
# --kv-cache-dtype fp8_e4m3 for FP8 baseline
```

### Hardware Profiles

`auto_setup()` detects your GPU and configures the optimal stack:

| Profile | VRAM | Primary Model | Reasoning |
|---------|------|---------------|-----------|
| `cpu_only` | None | Ollama llama3.2:3b | — |
| `nvidia_mid` | 8-12 GB | Nemotron-Orchestrator-8B | deepseek-r1:8b |
| `nvidia_high` | 16-24 GB | Nemotron-Orchestrator-8B | deepseek-r1:14b |
| `nvidia_ultra` | 32+ GB | Nemotron-Orchestrator-8B (BF16) | deepseek-r1:32b |
| `apple_silicon` | M1-M4 | Nemotron-Orchestrator-8B | deepseek-r1:8b |

## Effort-Based Model Routing

Tasks are automatically routed to the right model based on complexity:

| Effort | Model | Use Case |
|--------|-------|----------|
| 1-3 | llama3.2:3b | Quick lookups, simple Q&A |
| 4-6 | Nemotron-Orchestrator-8B | Most tasks, tool calling, orchestration |
| 7-10 | deepseek-r1:14b | Complex reasoning, code review |

## Agent Identities

48 pre-built identities with distinct capabilities:

| Identity | Role | Best For |
|----------|------|----------|
| `aither` | Orchestrator | System coordination, delegation |
| `atlas` | Project Manager | Planning, tracking, reporting |
| `demiurge` | Code Craftsman | Code generation, refactoring |
| `lyra` | Researcher | Research, knowledge synthesis |
| `athena` | Security Oracle | Security audits, vulnerability analysis |
| `hydra` | Code Guardian | Code review, quality assurance |
| `prometheus` | Worldbuilder | Simulation, procedural generation |
| `apollo` | Performance | Optimization, benchmarking |
| `iris` | Creative | Image generation, design |
| `viviane` | Memory | Knowledge retrieval, context |
| `vera` | Content | Writing, editing |
| `morgana` | Secrets | Security, encryption |
| `saga` | Documentation | Technical writing |
| `themis` | Compliance | Ethics, policy |
| `chaos` | Chaos Engineer | Resilience testing |
| `terra` | Infrastructure | Disk, memory, health monitoring |

Any agent can delegate to any other via `ask_agent("demiurge", "Write a function that...")`.

## Tools

```python
from adk import AitherAgent, tool

@tool
def search_codebase(query: str) -> str:
    """Search the codebase for matching code."""
    return run_search(query)

@tool
def run_tests(path: str) -> str:
    """Run tests at the given path."""
    return subprocess.check_output(["pytest", path]).decode()

agent = AitherAgent("demiurge", tools=[search_codebase, run_tests])
response = await agent.chat("Find and fix the auth bug")
```

## Knowledge Graph Memory

Every agent has a local knowledge graph — SQLite-backed, embedding-aware, zero external deps.

```python
agent = AitherAgent("atlas")

# Store knowledge
await agent.graph_remember("AitherOS", "uses", "vLLM")
await agent.graph_remember("TurboQuant", "compresses", "KV cache")

# Query
results = await agent.graph_query("What compresses the KV cache?")

# Auto-ingests from conversations
response = await agent.chat("Tell me about the agent dispatch system")
# Entities extracted and stored automatically
```

Features: hybrid keyword + semantic search, entity extraction, relation triples, BFS traversal, conversation auto-ingestion.

## Neuron Architecture

Neurons auto-fire before LLM calls to gather relevant context:

```python
from adk.neurons import BaseNeuron, NeuronResult

class DatabaseNeuron(BaseNeuron):
    name = "database"
    async def fire(self, query, **kwargs):
        results = await db.search(query)
        return NeuronResult(neuron=self.name, content=results, relevance=0.9)

agent._auto_neurons.pool.register(DatabaseNeuron())
```

Built-in: WebSearchNeuron (DuckDuckGo), MemoryNeuron, GraphNeuron.

## The Full Stack

AitherADK is the SDK layer. The full AitherOS stack adds deployment, monitoring, and infrastructure:

| Component | What It Does | Install |
|-----------|-------------|---------|
| **AitherADK** | Agent SDK — identities, tools, memory, fleet | `pip install aither-adk` |
| **AitherZero** | PowerShell automation — deploy services, manage infra | [github.com/Aitherium/AitherZero](https://github.com/Aitherium/AitherZero) |
| **AitherVeil** | Web dashboard — chat, monitoring, agent management | `docker compose up aither-veil` |
| **AitherNode** | MCP server — 100+ tools for Claude Code, Cursor, Copilot | Port 8080 |
| **AitherDesktop** | Native Windows app — Win+A hotkey, system tray | Installer |
| **AitherConnect** | Chrome extension — AI on any webpage | Extension store |
| **aither-kvcache** | KV cache compression — 3.8x vs FP16 (up to 7.1x at 2-bit) | `pip install aither-kvcache` |
| **vLLM containers** | Local inference — Nemotron-8B, DeepSeek-R1, vision | Docker Compose |

### Deploying with AitherZero

[AitherZero](https://github.com/Aitherium/AitherZero) is the automation layer — 170+ PowerShell scripts for deploying and managing the full stack:

```powershell
# Deploy vLLM + orchestrator
./AitherZero/library/automation-scripts/10-core/0101_Start-CoreServices.ps1

# Deploy AitherVeil dashboard
./AitherZero/library/automation-scripts/50-deployment/0501_Deploy-Veil.ps1

# Health check everything
./AitherZero/library/automation-scripts/80-testing/0801_Run-HealthChecks.ps1
```

## Streaming

```python
async for chunk in agent.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

## Safety

Input/output safety runs on every `chat()` call — regex-based prompt injection detection (14 patterns), API key leak detection, system prompt leak detection. Non-fatal: agent works if safety module fails.

## NanoGPT

Zero-dependency character-level transformer for local fine-tuning. Pure Python autograd (no PyTorch required). Train on conversation patterns for intent prediction, anomaly detection, topic classification.

```python
from adk.nanogpt import NanoGPT

model = NanoGPT(n_layer=1, n_embd=16, block_size=16, n_head=4)
await model.train(["training data here"], num_steps=500)
loss = model.evaluate("test input")  # High loss = unfamiliar content
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AITHER_LLM_BACKEND` | `auto` | `ollama`, `openai`, `anthropic`, `auto` |
| `AITHER_MODEL` | (auto) | Default model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OPENAI_API_KEY` | | OpenAI API key |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `AITHER_PORT` | `8080` | Server port |
| `AITHER_DATA_DIR` | `~/.aither` | Data directory |

## License

**Business Source License 1.1** (BSL-1.1)

Free for individuals, internal use, building products, research, and education. Converts to **AGPL-3.0** on 2030-03-13.

See [LICENSE](LICENSE) for full terms.
