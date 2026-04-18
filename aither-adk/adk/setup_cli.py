"""aither setup — Interactive CLI for local inference + optional AitherOS stack.

Primary goal: get vLLM containers running so agents can use true concurrent
inference with continuous batching. This is what enables running parallel
agent fleets on a single GPU — Ollama serializes requests, vLLM batches them.

Usage:
    aither setup                    # Auto-detect GPU, set up vLLM
    aither setup --tier lite        # Force a specific tier
    aither setup --tier ollama      # Fallback: Ollama for non-NVIDIA GPUs
    aither setup --dry-run          # Show what would happen
    aither setup --stack core       # Also deploy AitherOS core services
    aither setup --stack full       # Deploy full AitherOS stack via AitherZero
    aither setup --non-interactive  # No prompts (CI/automation)

Pure stdlib — no pip dependencies required.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
        except Exception:
            return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR = _supports_color()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text

def bold(t: str) -> str: return _c("1", t)
def green(t: str) -> str: return _c("92", t)
def yellow(t: str) -> str: return _c("93", t)
def red(t: str) -> str: return _c("91", t)
def cyan(t: str) -> str: return _c("96", t)
def dim(t: str) -> str: return _c("2", t)

def info(msg: str) -> None: print(f"  {green('+')} {msg}")
def warn(msg: str) -> None: print(f"  {yellow('!')} {msg}")
def err(msg: str) -> None: print(f"  {red('x')} {msg}")
def step(n: int, total: int, msg: str) -> None:
    print(f"\n  {bold(f'[{n}/{total}]')} {bold(msg)}")


# ---------------------------------------------------------------------------
# GPU Detection (sync, for CLI use)
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    vendor: str = "none"
    name: str = "Unknown"
    vram_mb: int = 0          # Best single GPU VRAM (for Ollama profile selection)
    cuda_version: str = ""
    driver_version: str = ""
    gpu_count: int = 0
    total_vram_mb: int = 0    # Sum of all GPUs (for vLLM tier selection)
    all_gpus: list = field(default_factory=list)  # List of {"name": str, "vram_mb": int}


def _run(cmd: list[str], timeout: int = 10) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def detect_gpu() -> GPUInfo:
    """Detect GPU: NVIDIA > AMD > Apple Silicon > none."""
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run([smi, "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
        if out:
            lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
            cuda_out = _run([smi])
            cuda_ver = ""
            if cuda_out:
                m = re.search(r"CUDA Version:\s*([\d.]+)", cuda_out)
                if m:
                    cuda_ver = m.group(1)

            # Parse all GPUs, find best (max VRAM), sum total
            all_gpus = []
            best_name = "NVIDIA GPU"
            best_vram = 0
            total_vram = 0
            best_driver = ""
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                g_name = parts[0] if parts else "NVIDIA GPU"
                g_vram = int(float(parts[1])) if len(parts) > 1 else 0
                g_driver = parts[2] if len(parts) > 2 else ""
                all_gpus.append({"name": g_name, "vram_mb": g_vram})
                total_vram += g_vram
                if g_vram > best_vram:
                    best_vram = g_vram
                    best_name = g_name
                    best_driver = g_driver

            return GPUInfo(
                vendor="nvidia",
                name=best_name,
                vram_mb=best_vram,
                driver_version=best_driver,
                cuda_version=cuda_ver,
                gpu_count=len(lines),
                total_vram_mb=total_vram,
                all_gpus=all_gpus,
            )

    rocm = shutil.which("rocm-smi")
    if rocm:
        out = _run([rocm, "--showproductname"])
        if out:
            name = "AMD GPU"
            for line in out.split("\n"):
                if any(k in line for k in ("GPU", "Radeon", "Instinct")):
                    name = line.strip().split(":")[-1].strip() if ":" in line else line.strip()
                    break
            return GPUInfo(vendor="amd", name=name)

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        return GPUInfo(
            vendor="apple",
            name=chip.strip() if chip else "Apple Silicon",
            vram_mb=int(int(mem.strip()) / (1024 * 1024)) if mem else 0,
        )

    return GPUInfo()


# ---------------------------------------------------------------------------
# vLLM Tier Definitions
# ---------------------------------------------------------------------------

@dataclass
class VLLMWorker:
    name: str
    model: str
    served_name: str
    port: int
    gpu_mem: float
    ctx_len: int
    extra_args: list[str] = field(default_factory=list)
    description: str = ""
    download_gb: float = 0.0
    vram_gb: float = 0.0


TIERS: dict[str, dict] = {
    "nano": {
        "name": "Nano",
        "desc": "Qwen3-8B via vLLM — for 8-12GB GPUs",
        "min_vram_gb": 6,
        "workers": [
            VLLMWorker("orchestrator", "Qwen/Qwen3-8B", "aither-orchestrator",
                       8200, 0.80, 16384,
                       ["--quantization bitsandbytes", "--load-format bitsandbytes",
                        "--enable-auto-tool-choice", "--tool-call-parser hermes",
                        "--enable-prefix-caching"],
                       "Qwen3-8B — capable chat + tool calling", 8.0, 5.5),
        ],
    },
    "lite": {
        "name": "Lite",
        "desc": "Nemotron Orchestrator — for 12-16GB GPUs",
        "min_vram_gb": 10,
        "workers": [
            VLLMWorker("orchestrator", "nvidia/Nemotron-Orchestrator-8B", "aither-orchestrator",
                       8200, 0.80, 32768,
                       ["--quantization bitsandbytes", "--load-format bitsandbytes",
                        "--enable-auto-tool-choice", "--tool-call-parser hermes",
                        "--enable-prefix-caching"],
                       "Nemotron-Orchestrator-8B — outperforms GPT-4o on tool use", 16.0, 6.5),
        ],
    },
    "standard": {
        "name": "Standard",
        "desc": "Orchestrator + Reasoning — for 20-24GB GPUs. True parallel agents.",
        "min_vram_gb": 18,
        "workers": [
            VLLMWorker("orchestrator", "nvidia/Nemotron-Orchestrator-8B", "aither-orchestrator",
                       8200, 0.35, 32768,
                       ["--quantization bitsandbytes", "--load-format bitsandbytes",
                        "--enable-auto-tool-choice", "--tool-call-parser hermes",
                        "--enable-prefix-caching", "--enable-sleep-mode"],
                       "Nemotron-Orchestrator-8B — handles 80% of agent requests", 16.0, 6.5),
            VLLMWorker("reasoning", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-r1:14b",
                       8201, 0.55, 16384,
                       ["--quantization bitsandbytes", "--load-format bitsandbytes",
                        "--enable-auto-tool-choice", "--tool-call-parser hermes",
                        "--reasoning-parser deepseek_r1",
                        "--enable-prefix-caching", "--enable-sleep-mode"],
                       "DeepSeek-R1 14B — deep thinking for complex tasks", 28.0, 12.0),
        ],
    },
    "full": {
        "name": "Full",
        "desc": "Orchestrator + Reasoning + Embeddings — 24GB+ GPUs. Full fleet support.",
        "min_vram_gb": 20,
        "workers": [
            VLLMWorker("orchestrator", "nvidia/Nemotron-Orchestrator-8B", "aither-orchestrator",
                       8200, 0.35, 32768,
                       ["--quantization bitsandbytes", "--load-format bitsandbytes",
                        "--enable-auto-tool-choice", "--tool-call-parser hermes",
                        "--enable-prefix-caching", "--enable-sleep-mode"],
                       "Nemotron-Orchestrator-8B", 16.0, 6.5),
            VLLMWorker("reasoning", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-r1:14b",
                       8201, 0.55, 16384,
                       ["--quantization bitsandbytes", "--load-format bitsandbytes",
                        "--enable-auto-tool-choice", "--tool-call-parser hermes",
                        "--reasoning-parser deepseek_r1",
                        "--enable-prefix-caching", "--enable-sleep-mode"],
                       "DeepSeek-R1 14B", 28.0, 12.0),
            VLLMWorker("embeddings", "nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text",
                       8209, 0.05, 2048,
                       ["--dtype float16", "--max-num-seqs 64"],
                       "Nomic Embed v1.5 — vector search", 0.5, 0.5),
        ],
    },
}


def recommend_tier(gpu: GPUInfo) -> str:
    if gpu.vendor != "nvidia":
        return "ollama"
    # Use total VRAM for vLLM tier — workers spread across GPUs
    vram = (gpu.total_vram_mb or gpu.vram_mb) / 1024 * 0.85
    if vram >= 24: return "full"
    if vram >= 18: return "standard"
    if vram >= 10: return "lite"
    if vram >= 6: return "nano"
    return "ollama"


# ---------------------------------------------------------------------------
# Docker Compose Generation
# ---------------------------------------------------------------------------

def generate_compose(tier_id: str, hf_token: str = "") -> str:
    tier = TIERS[tier_id]
    workers = tier["workers"]
    services = []
    for w in workers:
        extra = " ".join(w.extra_args)
        env_block = "      NVIDIA_VISIBLE_DEVICES: all\n      VLLM_NO_USAGE_STATS: '1'"
        if hf_token:
            env_block += f"\n      HF_TOKEN: \"{hf_token}\"\n      HUGGING_FACE_HUB_TOKEN: \"{hf_token}\""
        svc = textwrap.dedent(f"""\
  adk-vllm-{w.name}:
    image: vllm/vllm-openai:latest
    container_name: adk-vllm-{w.name}
    shm_size: '4gb'
    environment:
{env_block}
    command: >
      --model {w.model}
      --host 0.0.0.0 --port 8000
      --gpu-memory-utilization {w.gpu_mem}
      --max-model-len {w.ctx_len}
      --enforce-eager --dtype auto
      --max-num-seqs 8
      --trust-remote-code
      --served-model-name {w.served_name}
      {extra}
    ports:
      - "{w.port}:8000"
    volumes:
      - adk-hf-cache:/root/.cache/huggingface
    healthcheck:
      interval: 30s
      timeout: 10s
      start_period: 900s
      retries: 5
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]""")
        services.append(svc)

    total_dl = sum(w.download_gb for w in workers)
    return textwrap.dedent(f"""\
# AitherOS vLLM Inference Stack — Generated by `aither setup`
# Tier: {tier_id} ({tier['name']}) — {tier['desc']}
#
# Why vLLM over Ollama?
#   vLLM uses continuous batching — multiple agents share the GPU simultaneously.
#   Ollama serializes requests — agents wait in line. For parallel agent fleets,
#   vLLM is 3-10x faster under concurrent load.
#
# Usage:
#   docker compose -f docker-compose.vllm.yml up -d
#   docker compose -f docker-compose.vllm.yml logs -f
#   docker compose -f docker-compose.vllm.yml down
#
# First run downloads ~{total_dl:.0f}GB of model weights (cached after).

services:
{chr(10).join(services)}

volumes:
  adk-hf-cache:
    name: adk-hf-cache
""")


# ---------------------------------------------------------------------------
# Docker + Container Management
# ---------------------------------------------------------------------------

def check_docker() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if not docker:
        return False, "Docker not installed"
    out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if not out:
        return False, "Docker daemon not running"
    return True, f"Docker {out}"


def start_containers(compose_path: Path, dry_run: bool = False) -> bool:
    if dry_run:
        info(f"Would run: docker compose -f {compose_path} up -d")
        return True
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "up", "-d"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            err(f"docker compose up failed: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        warn("docker compose up timed out — containers may still be starting")
        return True
    except Exception as e:
        err(f"Failed: {e}")
        return False


def wait_for_health(port: int, name: str, timeout: int = 300) -> bool:
    import urllib.request
    import urllib.error

    start = time.time()
    dots = 0
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    if dots > 0:
                        print()
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        if dots == 0:
            print(f"    Waiting for {name}", end="", flush=True)
        print(".", end="", flush=True)
        dots += 1
        time.sleep(5)
    if dots > 0:
        print()
    return False


# ---------------------------------------------------------------------------
# Ollama Fallback
# ---------------------------------------------------------------------------

OLLAMA_RECOMMENDED = [
    ("nemotron-orchestrator-8b", "NVIDIA Nemotron — best tool use", 8),
    ("qwen3:8b", "Qwen 3 8B — strong multilingual", 8),
    ("deepseek-r1:7b", "DeepSeek-R1 7B — reasoning", 8),
    ("nomic-embed-text", "Nomic Embed — vector search", 2),
]


def setup_ollama(gpu: GPUInfo, dry_run: bool = False) -> int:
    ollama = shutil.which("ollama")
    if not ollama:
        err("Ollama not installed")
        print(f"    Install: {cyan('https://ollama.com/download')}")
        return 1

    vram_gb = gpu.vram_mb / 1024 if gpu.vram_mb else 0
    info("Ollama found")
    if gpu.vendor != "none":
        info(f"GPU: {gpu.name} ({vram_gb:.0f}GB)")

    print()
    warn("Ollama serializes requests — agents wait in line.")
    warn("For parallel agent fleets, use an NVIDIA GPU + vLLM.")
    print()

    models_to_pull = []
    for name, desc, min_vram in OLLAMA_RECOMMENDED:
        if vram_gb >= min_vram or gpu.vendor == "none":
            models_to_pull.append(name)
            info(f"Will pull: {name} — {desc}")

    if not models_to_pull:
        models_to_pull = ["llama3.2:3b", "nomic-embed-text"]
        info(f"Low VRAM — using compact models: {', '.join(models_to_pull)}")

    running = _run(["ollama", "list"])
    if running is None:
        warn("Ollama not running — starting...")
        if not dry_run:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            time.sleep(3)

    existing = set()
    out = _run(["ollama", "list"])
    if out:
        for line in out.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                existing.add(parts[0].split(":")[0])

    for model in models_to_pull:
        base = model.split(":")[0]
        if base in existing or model in existing:
            info(f"Already have: {model}")
            continue
        if dry_run:
            info(f"Would pull: {model}")
        else:
            info(f"Pulling: {bold(model)}...")
            try:
                subprocess.run(["ollama", "pull", model], timeout=1800)
            except Exception as e:
                warn(f"Failed to pull {model}: {e}")

    _save_config("ollama", None, gpu)
    print()
    info("Ollama ready")
    print(f"  {dim('For true parallel agents, use vLLM with an NVIDIA GPU.')}")
    print()
    return 0


# ---------------------------------------------------------------------------
# AitherZero Bridge
# ---------------------------------------------------------------------------

def find_aitherzero() -> Optional[Path]:
    """Locate AitherZero scripts directory."""
    candidates = []
    env_path = os.environ.get("AITHERZERO_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Relative to adk package
    adk_dir = Path(__file__).resolve().parent
    for depth in range(1, 5):
        parent = adk_dir
        for _ in range(depth):
            parent = parent.parent
        candidates.append(parent / "AitherZero")

    candidates.extend([
        Path.home() / "AitherOS" / "AitherZero",
        Path.home() / "AitherOS-Fresh" / "AitherZero",
    ])

    for p in candidates:
        if p.is_dir() and (p / "library" / "automation-scripts").is_dir():
            return p
    return None


def deploy_stack(profile: str, dry_run: bool = False, api_key: str = "") -> int:
    """Deploy AitherOS services via AitherZero OneClick script.

    Profiles: minimal, core, full, headless, gpu, agents
    """
    az_root = find_aitherzero()
    if not az_root:
        warn("AitherZero not found — cannot deploy AitherOS stack")
        print()
        print(f"  To deploy AitherOS, clone the repo:")
        print(f"    {cyan('git clone https://github.com/Aitherium/AitherOS AitherOS-Fresh')}")
        print(f"    {cyan('cd AitherOS-Fresh && aither setup --stack ' + profile)}")
        print()
        print(f"  Or set {cyan('AITHERZERO_PATH')} to your AitherZero directory.")
        return 1

    pwsh = shutil.which("pwsh")
    if not pwsh:
        warn("PowerShell 7 (pwsh) required for AitherZero scripts")
        if sys.platform == "win32":
            print(f"    {cyan('winget install Microsoft.PowerShell')}")
        elif sys.platform == "darwin":
            print(f"    {cyan('brew install powershell')}")
        else:
            print(f"    {cyan('https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell')}")
        return 1

    deploy_script = az_root / "library" / "automation-scripts" / "30-deploy" / "3020_Deploy-OneClick.ps1"
    if not deploy_script.exists():
        err(f"Deploy script not found: {deploy_script}")
        return 1

    info(f"AitherZero: {az_root}")
    info(f"Profile: {profile}")

    cmd = [str(pwsh), "-NoProfile", "-File", str(deploy_script),
           "-Profile", profile, "-NonInteractive"]

    if api_key:
        os.environ["AITHER_API_KEY"] = api_key

    if dry_run:
        info(f"Would run: {' '.join(cmd)}")
        return 0

    print()
    print(f"  {bold('Deploying AitherOS')} ({profile} profile)...")
    print(f"  {dim('This may take several minutes on first run.')}")
    print()

    try:
        result = subprocess.run(cmd, timeout=1800)
        return result.returncode
    except subprocess.TimeoutExpired:
        err("Deployment timed out after 30 minutes")
        return 1
    except Exception as e:
        err(f"Deployment failed: {e}")
        return 1


# ---------------------------------------------------------------------------
# Config Persistence
# ---------------------------------------------------------------------------

def _save_config(backend: str, tier_id: Optional[str], gpu: GPUInfo):
    """Save setup results to ~/.aither/config.json."""
    config_path = Path.home() / ".aither" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            pass

    config["default_backend"] = "openai" if backend == "vllm" else "ollama"
    config["setup_backend"] = backend
    config["gpu_vendor"] = gpu.vendor
    config["gpu_name"] = gpu.name

    if tier_id:
        config["setup_tier"] = tier_id
        workers = TIERS[tier_id]["workers"]
        config["inference_url"] = f"http://localhost:{workers[0].port}/v1"
        if len(workers) > 1:
            reasoning = [w for w in workers if w.name == "reasoning"]
            if reasoning:
                config["reasoning_url"] = f"http://localhost:{reasoning[0].port}/v1"
    elif backend == "ollama":
        config["inference_url"] = "http://localhost:11434/v1"

    config_path.write_text(json.dumps(config, indent=2))
    info(f"Config saved: {config_path}")


# ---------------------------------------------------------------------------
# Interactive Prompt
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "", choices: list[str] = None) -> str:
    if choices:
        full = f"  {bold('?')} {prompt} [{'/'.join(choices)}]"
    elif default:
        full = f"  {bold('?')} {prompt} [{default}]"
    else:
        full = f"  {bold('?')} {prompt}"
    while True:
        try:
            answer = input(f"{full}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if not answer and default:
            return default
        if choices and answer.lower() not in [c.lower() for c in choices]:
            print(f"    {yellow('Choose:')} {', '.join(choices)}")
            continue
        if answer:
            return answer.lower() if choices else answer
        if not default and not choices:
            return ""


# ---------------------------------------------------------------------------
# Main: aither setup
# ---------------------------------------------------------------------------

def cmd_setup(args) -> int:
    """Main entry point for `aither setup`."""
    dry_run: bool = args.dry_run
    compose_path = Path(args.output)
    non_interactive: bool = args.non_interactive

    print()
    print(bold("  ============================================================"))
    print(bold("    AitherOS Setup"))
    print(dim("    GPU detection -> vLLM containers -> parallel agent fleets"))
    print(bold("  ============================================================"))
    print()
    print(f"  {dim('vLLM uses continuous batching — multiple agents share the GPU')}")
    print(f"  {dim('simultaneously. Ollama serializes. vLLM parallelizes.')}")
    print()

    if dry_run:
        print(f"  {yellow('DRY RUN — no changes will be made')}\n")

    total_steps = 5
    if args.stack:
        total_steps = 6

    # ── Step 1: Detect GPU ────────────────────────────────────────
    step(1, total_steps, "Detecting GPU hardware")
    gpu = detect_gpu()
    vram_gb = gpu.vram_mb / 1024 if gpu.vram_mb else 0

    if gpu.vendor == "nvidia":
        if gpu.gpu_count > 1 and gpu.all_gpus:
            info(f"GPUs: {bold(str(gpu.gpu_count))} detected")
            for i, g in enumerate(gpu.all_gpus):
                g_vram = g['vram_mb'] / 1024
                info(f"  GPU {i}: {g['name']} ({g_vram:.0f}GB)")
            total_gb = gpu.total_vram_mb / 1024 if gpu.total_vram_mb else vram_gb
            info(f"Best GPU: {bold(gpu.name)} ({vram_gb:.0f}GB)")
            info(f"Total VRAM: {bold(f'{total_gb:.0f}GB')}")
        else:
            info(f"GPU: {bold(gpu.name)}")
            info(f"VRAM: {bold(f'{vram_gb:.0f}GB')}")
        if gpu.cuda_version:
            info(f"CUDA: {gpu.cuda_version}")
    elif gpu.vendor == "amd":
        info(f"GPU: {bold(gpu.name)} (AMD)")
        warn("AMD GPUs work with Ollama. vLLM requires NVIDIA CUDA.")
    elif gpu.vendor == "apple":
        info(f"GPU: {bold(gpu.name)} (Apple Silicon)")
        warn("Apple Silicon works with Ollama. vLLM requires NVIDIA CUDA.")
    else:
        warn("No GPU detected — will use Ollama with CPU inference")

    # ── Step 2: Check Docker ──────────────────────────────────────
    step(2, total_steps, "Checking prerequisites")
    docker_ok, docker_msg = check_docker()
    if docker_ok:
        info(f"Docker: {docker_msg}")
    else:
        warn(f"Docker: {docker_msg}")

    ollama_ok = bool(shutil.which("ollama"))
    if ollama_ok:
        info("Ollama: installed (fallback)")
    else:
        info("Ollama: not installed (not needed with vLLM)")

    # Auto-install llmfit for hardware-aware model selection
    llmfit_ok = bool(shutil.which("llmfit"))
    if llmfit_ok:
        info("llmfit: installed (hardware-aware model selection)")
    elif not dry_run:
        info("llmfit: not found — installing for smart model selection...")
        import asyncio as _aio
        from adk.setup import AgentSetup as _AS
        _setup = _AS()
        try:
            llmfit_ok = _aio.get_event_loop().run_until_complete(_setup.ensure_llmfit())
            if llmfit_ok:
                info(f"llmfit: {green('installed')}")
            else:
                warn("llmfit: install failed (model selection will use static profiles)")
        except Exception:
            warn("llmfit: install failed (model selection will use static profiles)")

    # Decide path
    can_vllm = docker_ok and gpu.vendor == "nvidia" and vram_gb >= 6
    forced_tier = args.tier

    if forced_tier == "ollama":
        can_vllm = False
    elif forced_tier and forced_tier != "ollama" and not can_vllm:
        err(f"Tier '{forced_tier}' requires Docker + NVIDIA GPU")
        return 1

    if not can_vllm and not ollama_ok:
        err("Need Docker + NVIDIA GPU (for vLLM) or Ollama installed")
        print(f"    Docker: {cyan('https://docker.com/products/docker-desktop')}")
        print(f"    Ollama: {cyan('https://ollama.com/download')}")
        return 1

    # ── Step 3: Select Tier ───────────────────────────────────────
    step(3, total_steps, "Selecting inference tier")

    if not can_vllm:
        info("Using Ollama (no NVIDIA GPU or Docker)")
        step(4, total_steps, "Setting up Ollama")
        result = setup_ollama(gpu, dry_run)
        if args.stack:
            step(total_steps, total_steps, f"Deploying AitherOS ({args.stack})")
            deploy_stack(args.stack, dry_run, args.api_key or "")
        return result

    # vLLM path
    print()
    print(f"  {bold('Why vLLM?')}")
    print(f"  Ollama: one request at a time. Agents queue up, wait their turn.")
    print(f"  vLLM:   {bold('continuous batching')} — all agents run {green('simultaneously')}.")
    print(f"  Result: {green('3-10x faster')} with concurrent agent fleets.")
    print()

    recommended = forced_tier or recommend_tier(gpu)

    if forced_tier:
        tier_id = forced_tier
        info(f"Using tier: {bold(tier_id)} (from --tier)")
    elif non_interactive:
        tier_id = recommended
        info(f"Auto-selected: {bold(tier_id)}")
    else:
        # Tier comparison table
        print(f"  {'Tier':<12} {'Workers':<35} {'VRAM':<10} {'Download'}")
        print(f"  {'-'*12} {'-'*35} {'-'*10} {'-'*10}")
        for tid, tier in TIERS.items():
            workers_str = " + ".join(w.name for w in tier["workers"])
            vram_need = sum(w.vram_gb for w in tier["workers"])
            dl = sum(w.download_gb for w in tier["workers"])
            fits = vram_gb * 0.85 >= tier["min_vram_gb"]
            status = green("fits") if fits else red("too big")
            print(f"  {bold(tid):<20} {workers_str:<35} ~{vram_need:.0f}GB {status:<18} ~{dl:.0f}GB")
        print()
        info(f"Recommended: {bold(recommended)}")
        tier_id = ask("Select tier", default=recommended, choices=list(TIERS.keys()))

    tier = TIERS[tier_id]
    info(f"{tier['name']}: {tier['desc']}")

    total_dl = sum(w.download_gb for w in tier["workers"])
    total_vram = sum(w.vram_gb for w in tier["workers"])
    info(f"Download: ~{total_dl:.0f}GB (cached after first run)")
    info(f"VRAM: ~{total_vram:.0f}GB / {vram_gb:.0f}GB")

    # ── Step 4: HuggingFace Token ─────────────────────────────────
    step(4, total_steps, "Checking HuggingFace access")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if hf_token:
        info(f"HF token: {hf_token[:8]}...{hf_token[-4:]}")
    elif not non_interactive:
        print(f"  Some models need a HuggingFace token (free account).")
        print(f"  Get one at: {cyan('https://huggingface.co/settings/tokens')}")
        hf_token = ask("HuggingFace token (Enter to skip)", default="")
        if hf_token:
            info(f"Token set: {hf_token[:8]}...")
    if not hf_token:
        warn("No HF token — gated models may fail to download")

    # ── Step 5: Generate + Start ──────────────────────────────────
    step(5, total_steps, "Starting vLLM containers")

    compose_content = generate_compose(tier_id, hf_token)

    if dry_run:
        info(f"Would write: {compose_path}")
        info(f"Would start {len(tier['workers'])} container(s)")
    else:
        compose_path.write_text(compose_content)
        info(f"Compose file: {compose_path}")

        if not start_containers(compose_path, dry_run):
            return 1
        info("Containers started")

        # Wait for health
        for w in tier["workers"]:
            info(f"Checking {w.name} (:{w.port})...")
            if wait_for_health(w.port, w.name, timeout=600):
                info(f"{w.name}: {green('healthy')}")
            else:
                warn(f"{w.name}: still loading (docker logs adk-vllm-{w.name})")

    _save_config("vllm", tier_id, gpu)

    # ── Optional: Deploy AitherOS stack ───────────────────────────
    if args.stack:
        step(total_steps, total_steps, f"Deploying AitherOS ({args.stack})")
        deploy_stack(args.stack, dry_run, args.api_key or "")

    # ── Summary ───────────────────────────────────────────────────
    orch = tier["workers"][0]
    print()
    print(bold("  ============================================================"))
    print()
    for w in tier["workers"]:
        print(f"  {green('*')} {bold(w.name)}: http://localhost:{w.port}/v1")
        print(f"    Model: {w.model} ({w.served_name})")
    print()
    print(f"  {bold('Run your agent:')}")
    print(f"    {cyan('aither run')}")
    print(f"    {dim('(auto-detects vLLM on port 8200)')}")
    print()
    print(f"  {bold('Run parallel agents:')}")
    print(f"    {cyan('aither run --agents lyra,atlas,demiurge')}")
    print(f"    {dim('All agents share the GPU via continuous batching.')}")
    print()

    if args.stack:
        print(f"  {bold('AitherOS:')}")
        print(f"    Dashboard: {cyan('http://localhost:3000')}")
        print(f"    Genesis:   {cyan('http://localhost:8001')}")
        print()

    if tier_id in ("standard", "full"):
        print(f"  {bold('Gaming mode:')}")
        print(f"    {cyan(f'docker compose -f {compose_path} stop')}  {dim('(free VRAM)')}")
        print(f"    {cyan(f'docker compose -f {compose_path} start')} {dim('(resume)')}")
        print()

    print(f"  {bold('Manage:')}")
    print(f"    Logs:   {cyan(f'docker compose -f {compose_path} logs -f')}")
    print(f"    Stop:   {cyan(f'docker compose -f {compose_path} stop')}")
    print(f"    Update: {cyan(f'docker compose -f {compose_path} pull && docker compose -f {compose_path} up -d')}")
    print()
    print(f"  {green(bold('Ready!'))}")
    print(f"  {dim('Your GPU is now an inference server for parallel agent fleets.')}")
    print()
    return 0
