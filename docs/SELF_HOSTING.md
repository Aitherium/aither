# Self-Hosting AitherOS

This guide covers running AitherOS on your own hardware with local inference. No cloud connection required.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free | 100 GB SSD |
| GPU | None (CPU inference works) | NVIDIA with 8+ GB VRAM |
| OS | Windows 10+, Linux, macOS | Linux (for vLLM) or Windows (for AitherDesktop) |

GPU is not required. Ollama runs on CPU, Apple Silicon, AMD GPUs, and NVIDIA GPUs. vLLM requires an NVIDIA GPU with CUDA support.

---

## 1. Install Ollama

Ollama is the simplest way to run models locally. It works on every platform.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS

Download from [ollama.com](https://ollama.com) or:

```bash
brew install ollama
```

### Windows

Download from [ollama.com](https://ollama.com).

### Pull a model

```bash
ollama serve
ollama pull llama3.2:3b
```

---

## 2. Setup Wizard

The setup wizard detects your hardware and configures the best backend automatically.

```bash
git clone https://github.com/Aitherium/AitherOS-Alpha.git
cd AitherOS-Alpha/aither-adk

python setup-vllm.py
```

The wizard will:

1. Detect your GPU vendor, model, and VRAM
2. Recommend a tier (nano, lite, standard, full, or ollama)
3. Let you pick models that fit your hardware
4. Generate a Docker Compose file (if using vLLM)
5. Pull models and start services

### Non-interactive mode

```bash
python setup-vllm.py --tier full          # Skip prompts, go full stack
python setup-vllm.py --tier ollama        # Use Ollama only
python setup-vllm.py --tier nano          # Minimal vLLM setup
python setup-vllm.py --dry-run            # Show what would happen
python setup-vllm.py --generate           # Generate compose file only
```

---

## 3. Hardware Profiles

The setup wizard maps your hardware to one of five tiers:

### vLLM Tiers (NVIDIA GPU required)

| Tier | VRAM | Models | Notes |
|------|------|--------|-------|
| **nano** | 8-12 GB | Qwen3-8B | Single small model. RTX 3060, RTX 4060, Arc A770. |
| **lite** | 12-16 GB | Nemotron Orchestrator 8B | Good tool use. RTX 3080, RTX 4070. |
| **standard** | 20-24 GB | Orchestrator + DeepSeek R1 14B | Chat + reasoning. RTX 3090, RTX 4090. |
| **full** | 24+ GB | Orchestrator + R1 + Embeddings | Full stack. RTX 4090, A5000, A100. |

### Ollama Mode (any hardware)

| Hardware | Recommended Models |
|----------|-------------------|
| CPU only | llama3.2:3b |
| 8 GB VRAM | llama3.2:3b, nomic-embed-text |
| 12 GB VRAM | llama3.1:8b, deepseek-r1:7b |
| 16-24 GB VRAM | llama3.1:8b, deepseek-r1:14b, qwen2.5-coder:14b |
| 32+ GB VRAM | llama3.1:70b-q4, deepseek-r1:32b |
| Apple Silicon (M1-M4) | llama3.1:8b, deepseek-r1:14b |
| AMD GPU | llama3.1:8b (ROCm support) |

### Profile files

11 pre-built hardware profiles are included in `aither-adk/profiles/`:

- `minimal.yaml` -- 8-12 GB VRAM, single GPU
- `standard.yaml` -- General purpose
- `workstation.yaml` -- Development workstation
- `nvidia_low.yaml` -- NVIDIA 8-12 GB
- `nvidia_mid.yaml` -- NVIDIA 12-16 GB
- `nvidia_high.yaml` -- NVIDIA 16-24 GB
- `nvidia_ultra.yaml` -- NVIDIA 32+ GB
- `server.yaml` -- Multi-GPU server
- `amd.yaml` -- AMD GPU (ROCm)
- `apple_silicon.yaml` -- Apple M-series
- `cpu_only.yaml` -- No GPU

---

## 4. Docker Compose for vLLM

If the setup wizard generates a Docker Compose file, or you want to set it up manually:

```bash
cd AitherOS-Alpha

# Copy and edit environment config
cp docker/.env.example docker/.env

# Edit .env to set your backend and model preferences
# Then start:
docker compose -f docker/docker-compose.alpha.yml up -d
```

For GPU-accelerated vLLM:

```bash
docker compose -f docker/docker-compose.alpha.yml --profile gpu up -d
```

This starts:

- **adk-server** on port 8080 -- your agent API
- **vllm-primary** on port 8120 -- local vLLM inference (GPU profile only)

### Environment variables

Edit `docker/.env`:

```bash
# Backend: ollama, vllm, openai, anthropic
AITHER_LLM_BACKEND=ollama

# Ollama host (Docker reaches host via host.docker.internal)
OLLAMA_HOST=http://host.docker.internal:11434

# vLLM model (GPU profile)
VLLM_MODEL=meta-llama/Llama-3.2-3B-Instruct

# Optional cloud keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Optional gateway connection
AITHER_API_KEY=
```

---

## 5. Connecting AitherDesktop to Your Server

AitherDesktop (the native Windows app) can connect to your self-hosted instance instead of the cloud gateway.

1. Start your local agent: `aither-serve --identity aither --port 8080`
2. In AitherDesktop settings, set the server URL to `http://localhost:8080`
3. AitherDesktop will use your local agent for all operations

If both local and gateway connections are configured, AitherDesktop uses local first and falls back to the gateway.

---

## 6. Running Fully Offline

AitherOS works completely offline. No internet connection is needed after initial setup.

### What you need

1. Ollama (or vLLM) with models already pulled
2. The ADK installed (`pip install aither-adk`)
3. That is it

### Start offline

```bash
# Make sure Ollama has your models
ollama list

# Start the agent (no gateway, no telemetry)
aither-serve --identity atlas --port 8080
```

### What you lose offline

- MCP tools from mcp.aitherium.com
- Cloud agent delegation
- Persistent cloud memory
- Playground access
- Telemetry reporting

### What still works offline

- All 16 agent identities
- Local chat with your LLM
- Local SQLite memory
- Custom tools (`@tool` decorator)
- OpenAI-compatible API (`/v1/chat/completions`)
- Docker deployment

---

## Troubleshooting

### vLLM: "CUDA out of memory"

Your model is too large for your GPU. Drop to a lower tier:

```bash
python setup-vllm.py --tier nano
```

Or use Ollama instead, which handles memory management automatically.

### Ollama: model runs slowly

If inference is slow, check that Ollama is using your GPU:

```bash
ollama ps   # Shows which device is being used
```

On Linux, ensure NVIDIA drivers and CUDA are installed. On macOS, Metal acceleration is automatic on Apple Silicon.

### Docker: cannot reach Ollama

The Docker container reaches Ollama on the host via `host.docker.internal`. Make sure Ollama is listening on all interfaces:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

### Port conflicts

Change the port in your start command:

```bash
aither-serve --port 8081
```

Or in Docker, edit the port mapping in `docker-compose.alpha.yml`.
