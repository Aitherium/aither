"""aither deploy — Component deployment for the AitherOS ecosystem.

Deploy individual components (Ollama, vLLM, AitherNode, core stack, full stack)
or companion apps (AitherConnect, AitherDesktop) from public GitHub releases.

Usage (via CLI):
    aither deploy ollama                   # Install + pull models for your GPU
    aither deploy ollama --models qwen3:8b,phi4  # Pull specific models
    aither deploy vllm                     # Set up vLLM inference (delegates to setup)
    aither deploy node                     # AitherNode + Genesis via Docker
    aither deploy node --gpu --dashboard   # Node with GPU + Veil dashboard
    aither deploy core                     # Core services (Node, Pulse, Watch, Genesis)
    aither deploy full                     # Full AitherOS stack (~55 containers)
    aither deploy full --profile chat-full # Specific chat profile
    aither deploy connect                  # AitherConnect browser extension
    aither deploy desktop                  # AitherDesktop native app
    aither deploy stop node                # Stop a running deployment

Pure stdlib -- no pip dependencies required.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from adk.setup_cli import (
    bold,
    cyan,
    dim,
    green,
    red,
    yellow,
    info,
    warn,
    err,
    step,
    detect_gpu,
    GPUInfo,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_RAW = "https://raw.githubusercontent.com/Aitherium/AitherOS/main"
GITHUB_RELEASES = "https://github.com/Aitherium/AitherOS/releases/latest/download"
AITHER_DIR = Path.home() / ".aither"
REGISTRY = "ghcr.io/aitherium"

# Compose file URLs (private repo — requires auth)
COMPOSE_NODE_URL = f"{GITHUB_RAW}/docker-compose.node.yml"
COMPOSE_FULL_URL = f"{GITHUB_RAW}/docker-compose.aitheros.yml"

# Gateway for auth validation
GATEWAY_URL = "https://gateway.aitherium.com"

# Health-check endpoints (port -> service name)
HEALTH_ENDPOINTS = {
    8001: "Genesis",
    8080: "AitherNode",
    8081: "Pulse",
    8082: "Watch",
    3000: "AitherVeil",
}

# Ollama model tiers keyed by VRAM range
_OLLAMA_MODELS_BY_VRAM = {
    "none": ["gemma4:4b", "nomic-embed-text"],
    "low":  ["gemma4:4b", "nomic-embed-text"],           # <8GB
    "mid":  ["nemotron-orchestrator-8b", "nomic-embed-text"],  # 8-12GB
    "high": [                                              # 12-24GB
        "nemotron-orchestrator-8b",
        "deepseek-r1:14b",
        "nomic-embed-text",
    ],
    "ultra": [                                             # 24GB+
        "nemotron-orchestrator-8b",
        "deepseek-r1:14b",
        "gemma4:27b",
        "nomic-embed-text",
    ],
}


# ---------------------------------------------------------------------------
# Helper: Authentication gate
# ---------------------------------------------------------------------------

def _get_api_key(api_key_arg: Optional[str] = None) -> Optional[str]:
    """Resolve API key from arg > env > saved config. Returns None if not found."""
    key = api_key_arg or os.environ.get("AITHER_API_KEY", "")
    if key:
        return key
    config_path = AITHER_DIR / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return data.get("api_key", "") or None
        except Exception:
            pass
    # Also check YAML config
    yaml_path = AITHER_DIR / "config.yaml"
    if yaml_path.exists():
        try:
            text = yaml_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.strip().startswith("api_key:"):
                    val = line.split(":", 1)[1].strip().strip("'\"")
                    if val:
                        return val
        except Exception:
            pass
    return None


def _validate_api_key(api_key: str) -> bool:
    """Validate an API key against the Aitherium gateway.

    Returns True if valid, False otherwise. Falls back to format check
    if gateway is unreachable (offline deployments still work with a key).
    """
    # Format check: must look like an Aitherium key
    if not (api_key.startswith("aither_") or api_key.startswith("ak_") or len(api_key) >= 32):
        return False

    # Try gateway validation (non-blocking — offline deploys still allowed)
    try:
        req = Request(
            f"{GATEWAY_URL}/v1/auth/validate",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "AitherADK/1.0",
            },
        )
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        # Gateway unreachable — allow deployment with valid-format key
        # This enables offline/air-gapped sovereign deployments
        return True


def _require_auth(api_key_arg: Optional[str] = None) -> Optional[str]:
    """Gate function for AitherOS component deployment.

    Returns the API key if valid, or None (and prints error) if not.
    """
    key = _get_api_key(api_key_arg)
    if not key:
        err("AitherOS deployment requires an API key")
        print()
        print(f"  Get one (free):")
        print(f"    {cyan('aither register')}")
        print()
        print(f"  Or set it:")
        print(f"    {cyan('export AITHER_API_KEY=your_key')}")
        print(f"    {cyan('aither deploy node --api-key your_key')}")
        print()
        return None

    if not _validate_api_key(key):
        err("Invalid API key")
        print()
        print(f"  Register for a new key: {cyan('aither register')}")
        print()
        return None

    return key


def _docker_login_ghcr(api_key: str) -> bool:
    """Authenticate with GHCR using the API key for private image pulls."""
    try:
        result = subprocess.run(
            ["docker", "login", REGISTRY, "-u", "aither", "--password-stdin"],
            input=api_key,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helper: subprocess runner
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 30) -> Optional[str]:
    """Run a command and return stdout on success, None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: file download
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Download a file using urllib.request. Returns True on success."""
    if dry_run:
        info(f"Would download: {url}")
        info(f"  -> {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    info(f"Downloading: {url}")
    try:
        req = Request(url, headers={"User-Agent": "AitherADK/1.0"})
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        size_kb = len(data) / 1024
        if size_kb > 1024:
            info(f"  -> {dest} ({size_kb / 1024:.1f} MB)")
        else:
            info(f"  -> {dest} ({size_kb:.0f} KB)")
        return True
    except (HTTPError, URLError, OSError) as exc:
        err(f"Download failed: {exc}")
        return False


def _download_bytes(url: str) -> Optional[bytes]:
    """Download a URL and return raw bytes, or None on failure."""
    try:
        req = Request(url, headers={"User-Agent": "AitherADK/1.0"})
        with urlopen(req, timeout=120) as resp:
            return resp.read()
    except (HTTPError, URLError, OSError) as exc:
        err(f"Download failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helper: Docker Compose
# ---------------------------------------------------------------------------

def _docker_compose(
    compose_file: Path,
    profiles: list[str],
    action: str,
    dry_run: bool,
    timeout: int = 300,
) -> int:
    """Run docker compose with given profiles and action (pull/up -d/down/ps).

    Returns the subprocess exit code, or 0 on dry run.
    """
    cmd = ["docker", "compose", "-f", str(compose_file)]
    for p in profiles:
        cmd += ["--profile", p]

    # Split action into parts (e.g. "up -d" -> ["up", "-d"])
    cmd += action.split()

    if dry_run:
        info(f"Would run: {' '.join(cmd)}")
        return 0

    info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, timeout=timeout)
        return result.returncode
    except subprocess.TimeoutExpired:
        warn(f"Command timed out after {timeout}s -- containers may still be starting")
        return 0
    except FileNotFoundError:
        err("docker command not found")
        return 1
    except Exception as exc:
        err(f"Command failed: {exc}")
        return 1


# ---------------------------------------------------------------------------
# Helper: Health Check
# ---------------------------------------------------------------------------

def _health_check(url: str, timeout: int = 120) -> bool:
    """Poll a health endpoint until it returns 200 or timeout (seconds)."""
    start = time.time()
    dots = 0
    service_name = url.rsplit("/", 1)[0].rsplit(":", 1)[-1] if ":" in url else url
    while time.time() - start < timeout:
        try:
            req = Request(url, headers={"User-Agent": "AitherADK/1.0"})
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    if dots > 0:
                        print()
                    return True
        except (HTTPError, URLError, ConnectionError, OSError):
            pass
        if dots == 0:
            print(f"    Waiting for {url}", end="", flush=True)
        print(".", end="", flush=True)
        dots += 1
        time.sleep(3)
    if dots > 0:
        print()
    return False


# ---------------------------------------------------------------------------
# Helper: Docker prerequisite check
# ---------------------------------------------------------------------------

def _check_docker() -> tuple[bool, str]:
    """Check if Docker is installed and daemon is running.

    Returns (ok, message).
    """
    docker = shutil.which("docker")
    if not docker:
        return False, "Docker is not installed"
    out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if not out:
        return False, "Docker daemon is not running"
    return True, f"Docker {out}"


# ---------------------------------------------------------------------------
# Helper: VRAM tier selection
# ---------------------------------------------------------------------------

def _vram_tier(gpu: GPUInfo) -> str:
    """Classify GPU VRAM into an Ollama model tier key."""
    if gpu.vendor == "none":
        return "none"
    vram_gb = gpu.vram_mb / 1024 if gpu.vram_mb else 0
    if vram_gb >= 24:
        return "ultra"
    if vram_gb >= 12:
        return "high"
    if vram_gb >= 8:
        return "mid"
    return "low"


# ---------------------------------------------------------------------------
# Helper: Platform detection
# ---------------------------------------------------------------------------

def _platform_name() -> str:
    """Return a normalized platform name: linux, macos, windows."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows" or sys.platform == "win32":
        return "windows"
    return "linux"


# ===========================================================================
# Component: Ollama
# ===========================================================================

def deploy_ollama(dry_run: bool = False, models: Optional[list[str]] = None) -> int:
    """Deploy Ollama with appropriate models for the detected GPU.

    Args:
        dry_run: If True, show what would happen without executing.
        models: Explicit model list to pull. If None, auto-select by GPU VRAM.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    print()
    print(bold("  Ollama Deployment"))
    print(dim("  Local LLM inference with automatic GPU offloading"))
    print()

    if dry_run:
        print(f"  {yellow('DRY RUN -- no changes will be made')}\n")

    total_steps = 4
    plat = _platform_name()

    # -- Step 1: Check installation -------------------------------------------
    step(1, total_steps, "Checking Ollama installation")

    ollama = shutil.which("ollama")
    if not ollama:
        warn("Ollama is not installed")
        print()
        if plat == "linux":
            print(f"  Install with:")
            print(f"    {cyan('curl -fsSL https://ollama.com/install.sh | sh')}")
        elif plat == "macos":
            print(f"  Install with:")
            print(f"    {cyan('brew install ollama')}")
        elif plat == "windows":
            print(f"  Install with:")
            print(f"    {cyan('winget install Ollama.Ollama')}")
        print()
        print(f"  Or download from: {cyan('https://ollama.com/download')}")
        return 1

    info(f"Ollama binary: {ollama}")

    # -- Step 2: Detect GPU ---------------------------------------------------
    step(2, total_steps, "Detecting GPU hardware")

    gpu = detect_gpu()
    vram_gb = gpu.vram_mb / 1024 if gpu.vram_mb else 0

    if gpu.vendor == "nvidia":
        info(f"GPU: {bold(gpu.name)} ({vram_gb:.0f} GB VRAM)")
        if gpu.cuda_version:
            info(f"CUDA: {gpu.cuda_version}")
    elif gpu.vendor == "amd":
        info(f"GPU: {bold(gpu.name)} (AMD ROCm)")
    elif gpu.vendor == "apple":
        info(f"GPU: {bold(gpu.name)} (unified memory: {vram_gb:.0f} GB)")
    else:
        warn("No GPU detected -- Ollama will use CPU inference (slow)")

    tier = _vram_tier(gpu)
    info(f"VRAM tier: {bold(tier)}")

    # -- Step 3: Ensure Ollama is running ------------------------------------
    step(3, total_steps, "Starting Ollama service")

    running = _run(["ollama", "list"])
    if running is not None:
        info("Ollama is already running")
    else:
        warn("Ollama is not running -- starting...")
        if not dry_run:
            # Start ollama serve in background
            if plat == "windows":
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            time.sleep(3)
            # Verify it started
            verify = _run(["ollama", "list"])
            if verify is not None:
                info("Ollama started successfully")
            else:
                warn("Ollama may still be starting -- continuing anyway")
        else:
            info("Would start: ollama serve (background)")

    # Build set of already-installed models
    existing_models: set[str] = set()
    model_list_out = _run(["ollama", "list"])
    if model_list_out:
        for line in model_list_out.strip().split("\n")[1:]:  # skip header
            parts = line.split()
            if parts:
                # Models appear as "name:tag" -- store both full and base
                full_name = parts[0]
                existing_models.add(full_name)
                existing_models.add(full_name.split(":")[0])

    # -- Step 4: Pull models --------------------------------------------------
    step(4, total_steps, "Pulling models")

    if models:
        models_to_pull = list(models)
        info(f"Using explicit model list: {', '.join(models_to_pull)}")
    else:
        models_to_pull = list(_OLLAMA_MODELS_BY_VRAM.get(tier, _OLLAMA_MODELS_BY_VRAM["none"]))
        info(f"Auto-selected for {bold(tier)} tier: {', '.join(models_to_pull)}")

    # Estimate resource usage
    total_disk_gb = 0.0
    size_estimates = {
        "gemma4:4b": 2.5, "gemma4:27b": 16.0,
        "nemotron-orchestrator-8b": 5.0, "deepseek-r1:14b": 9.0,
        "deepseek-r1:7b": 4.5, "nomic-embed-text": 0.3,
        "qwen3:8b": 5.0, "llama3.2:3b": 2.0,
    }
    for m in models_to_pull:
        total_disk_gb += size_estimates.get(m, 4.0)

    print()
    info(f"Estimated disk usage: ~{total_disk_gb:.1f} GB")
    if tier in ("mid", "high", "ultra"):
        info(f"Estimated VRAM at runtime: models loaded on demand, one at a time")
    else:
        info(f"CPU inference: expect 5-20 tokens/sec depending on model size")
    print()

    pulled = 0
    skipped = 0
    failed = 0

    for model in models_to_pull:
        base = model.split(":")[0]
        if base in existing_models or model in existing_models:
            info(f"Already installed: {model}")
            skipped += 1
            continue

        if dry_run:
            info(f"Would pull: {bold(model)}")
            pulled += 1
            continue

        info(f"Pulling: {bold(model)} ...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                timeout=1800,  # 30-minute timeout per model
            )
            if result.returncode == 0:
                info(f"  {green('OK')}")
                pulled += 1
            else:
                warn(f"  Pull returned exit code {result.returncode}")
                failed += 1
        except subprocess.TimeoutExpired:
            warn(f"  Pull timed out for {model} (30 min limit)")
            failed += 1
        except Exception as exc:
            warn(f"  Failed to pull {model}: {exc}")
            failed += 1

    # -- Summary --------------------------------------------------------------
    print()
    print(bold("  " + "-" * 60))
    print()
    info(f"Pulled: {pulled}  Skipped (already installed): {skipped}  Failed: {failed}")
    if failed > 0:
        warn("Some models failed to pull -- re-run to retry")
    print()
    info(f"Ollama API: {cyan('http://localhost:11434')}")
    info(f"OpenAI-compatible: {cyan('http://localhost:11434/v1')}")
    print()
    info(f"Test it: {cyan('ollama run ' + models_to_pull[0])}")
    print()
    if not dry_run:
        info(f"{green(bold('Ollama is ready!'))}")
    return 1 if failed > 0 and pulled == 0 else 0


# ===========================================================================
# Component: vLLM
# ===========================================================================

def deploy_vllm(dry_run: bool = False, tier: Optional[str] = None, hf_token: str = "") -> int:
    """Deploy vLLM inference containers.

    Delegates to the existing setup_cli logic which handles GPU detection,
    tier selection, compose generation, and container startup.

    Args:
        dry_run: If True, show what would happen without executing.
        tier: Force a specific vLLM tier (nano/lite/standard/full/ollama).
        hf_token: HuggingFace token for gated model access.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    from adk.setup_cli import cmd_setup

    # Synthesize an args namespace matching what cmd_setup expects
    class _SetupArgs:
        pass

    args = _SetupArgs()
    args.dry_run = dry_run
    args.tier = tier
    args.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
    args.non_interactive = bool(tier)  # non-interactive if tier is forced
    args.output = str(AITHER_DIR / "docker-compose.vllm.yml")
    args.stack = None
    args.api_key = ""

    return cmd_setup(args)


# ===========================================================================
# Component: AitherNode
# ===========================================================================

def deploy_node(
    dry_run: bool = False,
    tag: str = "latest",
    gpu: bool = False,
    dashboard: bool = False,
    mesh: bool = False,
    api_key_arg: Optional[str] = None,
) -> int:
    """Deploy AitherNode (MCP server) + Genesis orchestrator via Docker Compose.

    Downloads the node compose file from GitHub and starts the containers with
    selected profile flags. Requires a valid Aitherium API key.

    Resource estimates:
        Base:      ~2 GB RAM,  ~4 GB disk (images)
        + GPU:     +1 GB RAM,  +2 GB disk
        + Dashboard: +512 MB RAM, +1 GB disk
        + Mesh:    +256 MB RAM

    Args:
        dry_run: If True, show what would happen without executing.
        tag: Docker image tag (default: latest).
        gpu: Enable GPU-accelerated services.
        dashboard: Enable AitherVeil web dashboard (port 3000).
        mesh: Enable mesh networking for multi-node setups.
        api_key_arg: Explicit API key (falls back to env/config).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    print()
    print(bold("  AitherNode Deployment"))
    print(dim("  MCP server (8080) + Genesis orchestrator (8001)"))
    print()

    # Auth gate
    api_key = _require_auth(api_key_arg)
    if not api_key:
        return 1

    if dry_run:
        print(f"  {yellow('DRY RUN -- no changes will be made')}\n")

    total_steps = 4

    # -- Step 1: Prerequisites ------------------------------------------------
    step(1, total_steps, "Checking prerequisites")

    docker_ok, docker_msg = _check_docker()
    if docker_ok:
        info(f"{docker_msg}")
    else:
        err(docker_msg)
        print()
        print(f"  Install Docker: {cyan('https://docker.com/products/docker-desktop')}")
        return 1

    # Authenticate with GHCR for private image pulls
    if not dry_run:
        if _docker_login_ghcr(api_key):
            info(f"Authenticated with {REGISTRY}")
        else:
            warn(f"GHCR login failed -- image pulls may fail if images are private")

    # -- Step 2: Download compose file ----------------------------------------
    step(2, total_steps, "Downloading compose configuration")

    compose_file = AITHER_DIR / "docker-compose.node.yml"
    if not _download(COMPOSE_NODE_URL, compose_file, dry_run):
        err("Failed to download compose file")
        return 1

    # -- Step 3: Pull and start -----------------------------------------------
    step(3, total_steps, "Starting containers")

    profiles = []
    if gpu:
        profiles.append("gpu")
        info("Profile: gpu (GPU-accelerated services)")
    if dashboard:
        profiles.append("dashboard")
        info("Profile: dashboard (AitherVeil on port 3000)")
    if mesh:
        profiles.append("mesh")
        info("Profile: mesh (multi-node networking)")

    if not profiles:
        info("Profile: default (Node + Genesis)")

    # Resource estimates
    ram_gb = 2.0 + (1.0 if gpu else 0) + (0.5 if dashboard else 0) + (0.25 if mesh else 0)
    disk_gb = 4.0 + (2.0 if gpu else 0) + (1.0 if dashboard else 0)
    print()
    info(f"Estimated resources: ~{ram_gb:.1f} GB RAM, ~{disk_gb:.0f} GB disk (images)")
    print()

    # Set environment variables for the compose file
    env = os.environ.copy()
    env["AITHEROS_IMAGE_TAG"] = tag
    env["AITHEROS_REGISTRY"] = REGISTRY

    # Pull images
    rc = _docker_compose(compose_file, profiles, "pull", dry_run, timeout=600)
    if rc != 0:
        err("Failed to pull images")
        return 1

    # Start containers
    rc = _docker_compose(compose_file, profiles, "up -d", dry_run, timeout=120)
    if rc != 0:
        err("Failed to start containers")
        return 1

    # -- Step 4: Health checks ------------------------------------------------
    step(4, total_steps, "Verifying services")

    if dry_run:
        info("Would check: http://localhost:8080/health (AitherNode)")
        info("Would check: http://localhost:8001/health (Genesis)")
        if dashboard:
            info("Would check: http://localhost:3000 (AitherVeil)")
    else:
        endpoints = [
            (8080, "AitherNode"),
            (8001, "Genesis"),
        ]
        if dashboard:
            endpoints.append((3000, "AitherVeil"))

        all_healthy = True
        for port, name in endpoints:
            if _health_check(f"http://localhost:{port}/health", timeout=120):
                info(f"{name} (:{port}): {green('healthy')}")
            else:
                warn(f"{name} (:{port}): not responding yet -- may still be starting")
                all_healthy = False

    # -- Summary --------------------------------------------------------------
    print()
    print(bold("  " + "-" * 60))
    print()
    info(f"AitherNode: {cyan('http://localhost:8080')}")
    info(f"Genesis:    {cyan('http://localhost:8001')}")
    if dashboard:
        info(f"Dashboard:  {cyan('http://localhost:3000')}")
    print()
    info(f"Manage:")
    info(f"  Logs:  {cyan(f'docker compose -f {compose_file} logs -f')}")
    info(f"  Stop:  {cyan(f'aither deploy stop node')}")
    info(f"  PS:    {cyan(f'docker compose -f {compose_file} ps')}")
    print()
    if not dry_run:
        info(f"{green(bold('AitherNode is ready!'))}")
    return 0


# ===========================================================================
# Component: Core Stack
# ===========================================================================

def deploy_core(dry_run: bool = False, tag: str = "latest", api_key_arg: Optional[str] = None) -> int:
    """Deploy AitherOS core services (Node, Genesis, Pulse, Watch).

    Same as deploy_node but always includes GPU and dashboard profiles,
    and health-checks all core ports. Requires a valid Aitherium API key.

    Resource estimates:
        ~4 GB RAM, ~8 GB disk (images)

    Args:
        dry_run: If True, show what would happen without executing.
        tag: Docker image tag (default: latest).
        api_key_arg: Explicit API key (falls back to env/config).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    print()
    print(bold("  AitherOS Core Deployment"))
    print(dim("  Genesis (8001) + Node (8080) + Pulse (8081) + Watch (8082)"))
    print()

    # Auth gate
    api_key = _require_auth(api_key_arg)
    if not api_key:
        return 1

    if dry_run:
        print(f"  {yellow('DRY RUN -- no changes will be made')}\n")

    total_steps = 4

    # -- Step 1: Prerequisites ------------------------------------------------
    step(1, total_steps, "Checking prerequisites")

    docker_ok, docker_msg = _check_docker()
    if docker_ok:
        info(docker_msg)
    else:
        err(docker_msg)
        print()
        print(f"  Install Docker: {cyan('https://docker.com/products/docker-desktop')}")
        return 1

    # Authenticate with GHCR
    if not dry_run:
        if _docker_login_ghcr(api_key):
            info(f"Authenticated with {REGISTRY}")
        else:
            warn(f"GHCR login failed -- image pulls may fail if images are private")

    # Resource estimate
    print()
    info("Estimated resources: ~4 GB RAM, ~8 GB disk (images)")
    print()

    # -- Step 2: Download compose file ----------------------------------------
    step(2, total_steps, "Downloading compose configuration")

    compose_file = AITHER_DIR / "docker-compose.node.yml"
    if not _download(COMPOSE_NODE_URL, compose_file, dry_run):
        err("Failed to download compose file")
        return 1

    # -- Step 3: Pull and start -----------------------------------------------
    step(3, total_steps, "Starting core containers")

    profiles = ["gpu", "dashboard"]
    info("Profiles: gpu, dashboard (full core stack)")

    env = os.environ.copy()
    env["AITHEROS_IMAGE_TAG"] = tag
    env["AITHEROS_REGISTRY"] = REGISTRY

    rc = _docker_compose(compose_file, profiles, "pull", dry_run, timeout=600)
    if rc != 0:
        err("Failed to pull images")
        return 1

    rc = _docker_compose(compose_file, profiles, "up -d", dry_run, timeout=120)
    if rc != 0:
        err("Failed to start containers")
        return 1

    # -- Step 4: Health checks ------------------------------------------------
    step(4, total_steps, "Verifying core services")

    core_ports = [
        (8001, "Genesis"),
        (8080, "AitherNode"),
        (8081, "Pulse"),
        (8082, "Watch"),
        (3000, "AitherVeil"),
    ]

    if dry_run:
        for port, name in core_ports:
            info(f"Would check: http://localhost:{port}/health ({name})")
    else:
        for port, name in core_ports:
            if _health_check(f"http://localhost:{port}/health", timeout=120):
                info(f"{name} (:{port}): {green('healthy')}")
            else:
                warn(f"{name} (:{port}): not responding yet -- may still be starting")

    # -- Summary --------------------------------------------------------------
    print()
    print(bold("  " + "-" * 60))
    print()
    for port, name in core_ports:
        info(f"{name:15s} {cyan(f'http://localhost:{port}')}")
    print()
    info(f"Manage:")
    info(f"  Logs:  {cyan(f'docker compose -f {compose_file} logs -f')}")
    info(f"  Stop:  {cyan('aither deploy stop core')}")
    info(f"  PS:    {cyan(f'docker compose -f {compose_file} ps')}")
    print()
    if not dry_run:
        info(f"{green(bold('AitherOS core is ready!'))}")
    return 0


# ===========================================================================
# Component: Full Stack
# ===========================================================================

def deploy_full(
    dry_run: bool = False,
    tag: str = "latest",
    profile: str = "chat-agents",
    api_key_arg: Optional[str] = None,
) -> int:
    """Deploy the full AitherOS stack via docker compose.

    WARNING: This is a large deployment (~55 containers). Ensure sufficient
    system resources before proceeding. Requires a valid Aitherium API key.

    Resource estimates by profile:
        chat-minimal:  ~20 containers, ~8 GB RAM,  ~15 GB disk
        chat-full:     ~29 containers, ~12 GB RAM, ~20 GB disk
        chat-agents:   ~31 containers, ~14 GB RAM, ~22 GB disk

    Args:
        dry_run: If True, show what would happen without executing.
        tag: Docker image tag (default: latest).
        profile: Docker Compose profile (default: chat-agents).
        api_key_arg: Explicit API key (falls back to env/config).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    profile_resources = {
        "chat-minimal": ("~20 containers", "~8 GB RAM", "~15 GB disk"),
        "chat-full":    ("~29 containers", "~12 GB RAM", "~20 GB disk"),
        "chat-agents":  ("~31 containers", "~14 GB RAM", "~22 GB disk"),
    }
    containers, ram, disk = profile_resources.get(
        profile, ("unknown", "~14 GB RAM", "~22 GB disk")
    )

    print()
    print(bold("  AitherOS Full Stack Deployment"))
    print(dim(f"  Profile: {profile} ({containers})"))
    print()

    # Auth gate
    api_key = _require_auth(api_key_arg)
    if not api_key:
        return 1

    print(f"  {yellow('WARNING: This is a large deployment.')}")
    print(f"  {yellow(f'Resource requirements: {ram}, {disk} (images)')}")
    print()

    if dry_run:
        print(f"  {yellow('DRY RUN -- no changes will be made')}\n")

    total_steps = 4

    # -- Step 1: Prerequisites ------------------------------------------------
    step(1, total_steps, "Checking prerequisites")

    docker_ok, docker_msg = _check_docker()
    if docker_ok:
        info(docker_msg)
    else:
        err(docker_msg)
        print()
        print(f"  Install Docker: {cyan('https://docker.com/products/docker-desktop')}")
        return 1

    # Authenticate with GHCR
    if not dry_run:
        if _docker_login_ghcr(api_key):
            info(f"Authenticated with {REGISTRY}")
        else:
            warn(f"GHCR login failed -- image pulls may fail if images are private")

    # Check available RAM (best-effort)
    try:
        if _platform_name() == "linux":
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    if mem_gb < 12:
                        warn(f"System RAM: {mem_gb:.0f} GB (recommended: 16+ GB for full stack)")
                    else:
                        info(f"System RAM: {mem_gb:.0f} GB")
                    break
    except Exception:
        pass

    # -- Step 2: Download compose file ----------------------------------------
    step(2, total_steps, "Downloading compose configuration")

    compose_file = AITHER_DIR / "docker-compose.aitheros.yml"
    if not _download(COMPOSE_FULL_URL, compose_file, dry_run):
        err("Failed to download compose file")
        return 1

    # -- Step 3: Pull and start -----------------------------------------------
    step(3, total_steps, f"Starting {profile} stack")

    profiles = [profile]
    info(f"Profile: {bold(profile)}")

    env = os.environ.copy()
    env["AITHEROS_IMAGE_TAG"] = tag
    env["AITHEROS_REGISTRY"] = REGISTRY

    # Pull images (longer timeout for many images)
    info("Pulling images (this may take several minutes on first run)...")
    rc = _docker_compose(compose_file, profiles, "pull", dry_run, timeout=900)
    if rc != 0:
        err("Failed to pull images")
        return 1

    rc = _docker_compose(compose_file, profiles, "up -d", dry_run, timeout=300)
    if rc != 0:
        err("Failed to start containers")
        return 1

    # -- Step 4: Health checks ------------------------------------------------
    step(4, total_steps, "Verifying key services")

    check_ports = [
        (8001, "Genesis"),
        (8080, "AitherNode"),
        (3000, "AitherVeil"),
    ]

    if dry_run:
        for port, name in check_ports:
            info(f"Would check: http://localhost:{port}/health ({name})")
    else:
        for port, name in check_ports:
            if _health_check(f"http://localhost:{port}/health", timeout=180):
                info(f"{name} (:{port}): {green('healthy')}")
            else:
                warn(f"{name} (:{port}): not responding yet -- may still be starting")

        # Show running container count
        ps_out = _run(["docker", "compose", "-f", str(compose_file),
                        "--profile", profile, "ps", "--format", "json"])
        if ps_out:
            try:
                # docker compose ps --format json outputs one JSON object per line
                running = 0
                for line in ps_out.strip().split("\n"):
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("State") == "running":
                            running += 1
                info(f"Running containers: {bold(str(running))}")
            except (json.JSONDecodeError, KeyError):
                pass

    # -- Summary --------------------------------------------------------------
    print()
    print(bold("  " + "-" * 60))
    print()
    info(f"Genesis:    {cyan('http://localhost:8001')}")
    info(f"AitherNode: {cyan('http://localhost:8080')}")
    info(f"Dashboard:  {cyan('http://localhost:3000')}")
    print()
    info(f"Manage:")
    info(f"  Logs:   {cyan(f'docker compose -f {compose_file} --profile {profile} logs -f')}")
    info(f"  Stop:   {cyan('aither deploy stop full')}")
    info(f"  PS:     {cyan(f'docker compose -f {compose_file} --profile {profile} ps')}")
    info(f"  Update: {cyan(f'docker compose -f {compose_file} --profile {profile} pull')}")
    print()
    if not dry_run:
        info(f"{green(bold('AitherOS full stack is ready!'))}")
    return 0


# ===========================================================================
# Component: AitherConnect (browser extension)
# ===========================================================================

def deploy_connect(dry_run: bool = False, api_key_arg: Optional[str] = None) -> int:
    """Download and extract the AitherConnect browser extension.

    The extension is downloaded from the latest GitHub release and extracted
    to ~/.aither/AitherConnect/. Requires a valid Aitherium API key.

    Args:
        dry_run: If True, show what would happen without executing.
        api_key_arg: Explicit API key (falls back to env/config).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    print()
    print(bold("  AitherConnect Browser Extension"))
    print(dim("  Chrome extension for AitherOS integration"))
    print()

    # Auth gate
    if not _require_auth(api_key_arg):
        return 1

    if dry_run:
        print(f"  {yellow('DRY RUN -- no changes will be made')}\n")

    total_steps = 2
    connect_url = f"{GITHUB_RELEASES}/AitherConnect.zip"
    dest_dir = AITHER_DIR / "AitherConnect"

    # -- Step 1: Download -----------------------------------------------------
    step(1, total_steps, "Downloading AitherConnect")

    if dry_run:
        info(f"Would download: {connect_url}")
        info(f"Would extract to: {dest_dir}")
    else:
        data = _download_bytes(connect_url)
        if data is None:
            err("Failed to download AitherConnect")
            print()
            print(f"  Check releases: {cyan('https://github.com/Aitherium/AitherOS/releases')}")
            return 1

        # Extract zip
        info(f"Extracting to {dest_dir}...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(BytesIO(data)) as zf:
                zf.extractall(dest_dir)
            info(f"Extracted {len(list(dest_dir.rglob('*')))} files")
        except zipfile.BadZipFile:
            err("Downloaded file is not a valid zip archive")
            return 1

    # -- Step 2: Instructions -------------------------------------------------
    step(2, total_steps, "Installation instructions")

    print()
    print(f"  To install the extension in Chrome:")
    print()
    print(f"    1. Open {cyan('chrome://extensions')}")
    print(f"    2. Enable {bold('Developer mode')} (toggle in top-right)")
    print(f"    3. Click {bold('Load unpacked')}")
    print(f"    4. Select: {cyan(str(dest_dir))}")
    print()
    print(f"  For Edge: use {cyan('edge://extensions')} (same steps)")
    print()
    if not dry_run:
        info(f"{green(bold('AitherConnect downloaded!'))}")
    return 0


# ===========================================================================
# Component: AitherDesktop
# ===========================================================================

def deploy_desktop(dry_run: bool = False, api_key_arg: Optional[str] = None) -> int:
    """Download the AitherDesktop native application.

    Platform availability:
        Windows: Portable .exe from GitHub releases
        Linux:   Bootc ISO image (see documentation)
        macOS:   Not yet available

    Requires a valid Aitherium API key.

    Args:
        dry_run: If True, show what would happen without executing.
        api_key_arg: Explicit API key (falls back to env/config).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    print()
    print(bold("  AitherDesktop Native Application"))
    print()

    # Auth gate
    if not _require_auth(api_key_arg):
        return 1

    if dry_run:
        print(f"  {yellow('DRY RUN -- no changes will be made')}\n")

    plat = _platform_name()

    if plat == "windows":
        total_steps = 2
        desktop_url = f"{GITHUB_RELEASES}/AitherDesktop-win64.zip"
        dest_dir = AITHER_DIR / "AitherDesktop"

        step(1, total_steps, "Downloading AitherDesktop for Windows")

        if dry_run:
            info(f"Would download: {desktop_url}")
            info(f"Would extract to: {dest_dir}")
        else:
            data = _download_bytes(desktop_url)
            if data is None:
                err("Failed to download AitherDesktop")
                print()
                print(f"  Check releases: {cyan('https://github.com/Aitherium/AitherOS/releases')}")
                return 1

            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(BytesIO(data)) as zf:
                    zf.extractall(dest_dir)
                info(f"Extracted to {dest_dir}")
            except zipfile.BadZipFile:
                err("Downloaded file is not a valid zip archive")
                return 1

        step(2, total_steps, "Installation complete")

        # Locate .exe
        exe_path = dest_dir / "AitherDesktop.exe"
        if not dry_run and not exe_path.exists():
            # Search for it in subdirectories
            exe_candidates = list(dest_dir.rglob("AitherDesktop.exe"))
            if exe_candidates:
                exe_path = exe_candidates[0]

        print()
        info(f"Executable: {cyan(str(exe_path))}")
        info(f"Run: {cyan(str(exe_path))}")
        print()
        if not dry_run:
            info(f"{green(bold('AitherDesktop downloaded!'))}")
        return 0

    elif plat == "linux":
        print(f"  AitherDesktop for Linux is available as a {bold('Bootc ISO image')}.")
        print()
        print(f"  This provides a full AitherOS-powered desktop environment")
        print(f"  built on Fedora CoreOS with atomic updates.")
        print()
        print(f"  For more information:")
        print(f"    {cyan('https://github.com/Aitherium/AitherOS/wiki/AitherDesktop-Linux')}")
        print()
        print(f"  To download the ISO:")
        print(f"    {cyan(f'{GITHUB_RELEASES}/AitherDesktop-bootc.iso')}")
        print()
        return 0

    elif plat == "macos":
        print(f"  AitherDesktop for macOS is {yellow('not yet available')}.")
        print()
        print(f"  In the meantime, you can use:")
        print(f"    - {cyan('aither deploy node --dashboard')} (Docker-based, runs in browser)")
        print(f"    - {cyan('aither deploy connect')} (Chrome extension)")
        print()
        print(f"  Follow progress: {cyan('https://github.com/Aitherium/AitherOS/issues')}")
        print()
        return 0

    else:
        err(f"Unsupported platform: {plat}")
        return 1


# ===========================================================================
# Stop / Teardown
# ===========================================================================

def deploy_stop(component: str) -> int:
    """Stop a deployed AitherOS component.

    Args:
        component: One of "ollama", "node", "core", "full", "vllm", "all".

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    print()
    print(bold(f"  Stopping: {component}"))
    print()

    if component == "ollama":
        # Try graceful stop, then kill
        info("Stopping Ollama...")
        result = _run(["ollama", "stop"])
        if result is not None:
            info("Ollama stopped")
            return 0

        # Fallback: find and kill the process
        plat = _platform_name()
        if plat == "windows":
            _run(["taskkill", "/F", "/IM", "ollama.exe"])
            # Also stop the app if running
            _run(["taskkill", "/F", "/IM", "ollama app.exe"])
        else:
            _run(["pkill", "-f", "ollama serve"])
        info("Ollama process terminated")
        return 0

    elif component in ("node", "core"):
        compose_file = AITHER_DIR / "docker-compose.node.yml"
        if not compose_file.exists():
            warn(f"Compose file not found: {compose_file}")
            warn("Nothing to stop (was it deployed with 'aither deploy node/core'?)")
            return 1
        rc = _docker_compose(compose_file, [], "down", dry_run=False, timeout=120)
        if rc == 0:
            info(f"{component} stack stopped")
        return rc

    elif component == "full":
        compose_file = AITHER_DIR / "docker-compose.aitheros.yml"
        if not compose_file.exists():
            warn(f"Compose file not found: {compose_file}")
            warn("Nothing to stop (was it deployed with 'aither deploy full'?)")
            return 1
        rc = _docker_compose(compose_file, [], "down", dry_run=False, timeout=120)
        if rc == 0:
            info("Full stack stopped")
        return rc

    elif component == "vllm":
        # Check both possible compose file locations
        compose_candidates = [
            AITHER_DIR / "docker-compose.vllm.yml",
            Path("docker-compose.vllm.yml"),
        ]
        for compose_file in compose_candidates:
            if compose_file.exists():
                rc = _docker_compose(compose_file, [], "down", dry_run=False, timeout=120)
                if rc == 0:
                    info(f"vLLM stack stopped (from {compose_file})")
                return rc
        warn("No vLLM compose file found")
        warn("Nothing to stop (was it deployed with 'aither setup' or 'aither deploy vllm'?)")
        return 1

    elif component == "all":
        info("Stopping all AitherOS components...")
        exit_code = 0

        # Stop full stack
        full_compose = AITHER_DIR / "docker-compose.aitheros.yml"
        if full_compose.exists():
            rc = _docker_compose(full_compose, [], "down", dry_run=False, timeout=120)
            if rc == 0:
                info("Full stack stopped")
            else:
                exit_code = 1

        # Stop node/core stack
        node_compose = AITHER_DIR / "docker-compose.node.yml"
        if node_compose.exists():
            rc = _docker_compose(node_compose, [], "down", dry_run=False, timeout=120)
            if rc == 0:
                info("Node stack stopped")
            else:
                exit_code = 1

        # Stop vLLM
        for vllm_path in [AITHER_DIR / "docker-compose.vllm.yml",
                          Path("docker-compose.vllm.yml")]:
            if vllm_path.exists():
                rc = _docker_compose(vllm_path, [], "down", dry_run=False, timeout=120)
                if rc == 0:
                    info(f"vLLM stopped (from {vllm_path})")
                else:
                    exit_code = 1

        # Stop Ollama
        if shutil.which("ollama"):
            deploy_stop("ollama")

        print()
        if exit_code == 0:
            info(f"{green(bold('All components stopped'))}")
        else:
            warn("Some components may not have stopped cleanly")
        return exit_code

    else:
        err(f"Unknown component: {component}")
        print()
        print(f"  Valid components: ollama, vllm, node, core, full, all")
        return 1


# ===========================================================================
# CLI entry point
# ===========================================================================

def cmd_deploy_component(args) -> int:
    """Main entry point for `aither deploy <component>`.

    Called from cli.py when the deploy-stack or component deploy command
    is invoked. This dispatches to the appropriate deploy_* function.
    """
    component = getattr(args, "component", None)
    dry_run = getattr(args, "dry_run", False)

    if not component:
        print()
        print(bold("  AitherOS Component Deployment"))
        print()
        print(f"  Usage: {cyan('aither deploy <component> [options]')}")
        print()
        print(f"  Components:")
        print(f"    {bold('ollama')}     Install Ollama + pull models for your GPU")
        print(f"    {bold('vllm')}       Deploy vLLM inference containers (NVIDIA GPU)")
        print(f"    {bold('node')}       AitherNode MCP server + Genesis orchestrator")
        print(f"    {bold('core')}       Core services (Node, Pulse, Watch, Genesis, Veil)")
        print(f"    {bold('full')}       Full AitherOS stack (~31 containers)")
        print(f"    {bold('connect')}    AitherConnect browser extension")
        print(f"    {bold('desktop')}    AitherDesktop native application")
        print(f"    {bold('stop')}       Stop a running deployment")
        print()
        print(f"  Examples:")
        print(f"    {dim('aither deploy ollama')}")
        print(f"    {dim('aither deploy ollama --models qwen3:8b,phi4')}")
        print(f"    {dim('aither deploy node --gpu --dashboard')}")
        print(f"    {dim('aither deploy full --profile chat-full')}")
        print(f"    {dim('aither deploy stop all')}")
        print()
        return 1

    if component == "ollama":
        models = None
        models_str = getattr(args, "models", None)
        if models_str:
            models = [m.strip() for m in models_str.split(",") if m.strip()]
        return deploy_ollama(dry_run=dry_run, models=models)

    elif component == "vllm":
        tier = getattr(args, "tier", None)
        hf_token = getattr(args, "hf_token", "") or ""
        return deploy_vllm(dry_run=dry_run, tier=tier, hf_token=hf_token)

    elif component == "node":
        tag = getattr(args, "tag", "latest") or "latest"
        gpu = getattr(args, "gpu", False)
        dashboard = getattr(args, "dashboard", False)
        mesh = getattr(args, "mesh", False)
        api_key = getattr(args, "api_key", None)
        return deploy_node(dry_run=dry_run, tag=tag, gpu=gpu,
                           dashboard=dashboard, mesh=mesh, api_key_arg=api_key)

    elif component == "core":
        tag = getattr(args, "tag", "latest") or "latest"
        api_key = getattr(args, "api_key", None)
        return deploy_core(dry_run=dry_run, tag=tag, api_key_arg=api_key)

    elif component == "full":
        tag = getattr(args, "tag", "latest") or "latest"
        profile = getattr(args, "profile", "chat-agents") or "chat-agents"
        api_key = getattr(args, "api_key", None)
        return deploy_full(dry_run=dry_run, tag=tag, profile=profile, api_key_arg=api_key)

    elif component == "connect":
        api_key = getattr(args, "api_key", None)
        return deploy_connect(dry_run=dry_run, api_key_arg=api_key)

    elif component == "desktop":
        api_key = getattr(args, "api_key", None)
        return deploy_desktop(dry_run=dry_run, api_key_arg=api_key)

    elif component == "stop":
        stop_target = getattr(args, "stop_target", None)
        if not stop_target:
            err("Specify what to stop: ollama, vllm, node, core, full, all")
            return 1
        return deploy_stop(stop_target)

    else:
        err(f"Unknown component: {component}")
        print(f"  Run {cyan('aither deploy')} to see available components.")
        return 1
