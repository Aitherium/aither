"""
AitherADK Local Orchestrator Setup (llama.cpp + Nemotron-Orchestrator-8B)
==========================================================================

Cross-platform provisioner that:
  1. Detects GPU/accelerator (CUDA / Vulkan / Metal / CPU)
  2. Downloads matching llama.cpp prebuilt binary from GitHub releases
  3. Downloads quantized Nemotron-Orchestrator-8B GGUF from HuggingFace
  4. Installs llama-server as a background service (systemd / launchd / Task Scheduler)
  5. Registers OpenAI-compatible endpoint in ~/.aither/config.json

Designed for endpoint laptops/workstations that:
  - Have <16GB VRAM (vLLM doesn't fit)
  - Don't have Docker (or shouldn't run Docker for an inference daemon)
  - Need an always-on local orchestrator that survives reboots
  - Use Intel Arc / AMD / Apple Silicon (no CUDA, vLLM won't work)

Pure stdlib — no pip dependencies. Importable from CLI, MCP tools, and
PowerShell wrappers via subprocess.

Public API:
    detect_accel() -> AccelInfo
    pick_quant(vram_gb, ram_gb) -> str  ("Q3_K_M" / "Q4_K_M" / "Q5_K_M" / "Q6_K" / "Q8_0")
    install(quant=None, port=8200, model_repo=DEFAULT_MODEL_REPO,
            service=True, dry_run=False) -> InstallResult
    status(port=8200) -> StatusResult
    uninstall(port=8200, purge=False) -> bool

CLI entrypoint:
    python -m adk.llamacpp_setup [--quant Q4_K_M] [--port 8200] [--no-service]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_REPO = "bartowski/nvidia_Nemotron-Orchestrator-8B-GGUF"
DEFAULT_MODEL_DISPLAY = "nemotron-orchestrator-8b"
DEFAULT_SERVED_NAME = "aither-orchestrator"
DEFAULT_PORT = 8200
DEFAULT_CTX = 8192
LLAMACPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
HF_RESOLVE_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"

# Quant catalog with approximate sizes (Nemotron-Orchestrator-8B specific)
QUANTS = {
    "Q3_K_M": {"size_gb": 4.0, "ram_gb": 5.0, "quality": "OK for orchestrator-only"},
    "Q4_K_M": {"size_gb": 4.9, "ram_gb": 6.0, "quality": "Recommended — sweet spot"},
    "Q5_K_M": {"size_gb": 5.7, "ram_gb": 7.0, "quality": "Higher fidelity"},
    "Q6_K":   {"size_gb": 6.6, "ram_gb": 8.0, "quality": "Near-lossless"},
    "Q8_0":   {"size_gb": 8.5, "ram_gb": 10.0, "quality": "Effectively lossless"},
}

AITHER_HOME = Path.home() / ".aither"
LLAMACPP_DIR = AITHER_HOME / "llamacpp"
MODELS_DIR = AITHER_HOME / "models"
LOG_DIR = AITHER_HOME / "logs"
CONFIG_PATH = AITHER_HOME / "config.json"


# ---------------------------------------------------------------------------
# Accelerator detection
# ---------------------------------------------------------------------------

@dataclass
class AccelInfo:
    kind: str = "cpu"   # cuda | vulkan | metal | cpu
    name: str = "Unknown"
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    cuda_version: str = ""
    os_family: str = ""  # windows | linux | macos
    arch: str = ""       # x64 | arm64
    notes: list = field(default_factory=list)


def _run(cmd: list[str], timeout: int = 8) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _detect_ram_gb() -> float:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 * 1024)
        elif sys.platform == "darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return int(out) / (1024 ** 3) if out else 0.0
        elif sys.platform == "win32":
            out = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"])
            if out:
                m = re.search(r"=(\d+)", out)
                if m:
                    return int(m.group(1)) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def detect_accel() -> AccelInfo:
    info = AccelInfo(ram_gb=_detect_ram_gb())

    if sys.platform == "win32":
        info.os_family = "windows"
    elif sys.platform == "darwin":
        info.os_family = "macos"
    else:
        info.os_family = "linux"

    info.arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"

    # macOS / Apple Silicon -> Metal
    if info.os_family == "macos" and info.arch == "arm64":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
        info.kind = "metal"
        info.name = chip
        info.vram_gb = info.ram_gb  # unified memory
        return info

    # NVIDIA -> CUDA
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits"])
        if out:
            best_vram = 0
            best_name = "NVIDIA GPU"
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                vram = int(float(parts[1])) if len(parts) > 1 else 0
                if vram > best_vram:
                    best_vram = vram
                    best_name = parts[0]
            cuda_ver = ""
            smi_out = _run(["nvidia-smi"])
            if smi_out:
                m = re.search(r"CUDA Version:\s*([\d.]+)", smi_out)
                if m:
                    cuda_ver = m.group(1)
            info.kind = "cuda"
            info.name = best_name
            info.vram_gb = best_vram / 1024
            info.cuda_version = cuda_ver
            return info

    # AMD ROCm -> Vulkan (llama.cpp ROCm builds exist but Vulkan is more portable)
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        info.kind = "vulkan"
        info.name = "AMD GPU"
        info.notes.append("AMD detected — using Vulkan build")
        return info

    # Intel Arc / iGPU on Windows -> Vulkan
    if info.os_family == "windows":
        # dxdiag check for Intel Arc / Iris Xe
        gpu_check = _run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
        if gpu_check:
            names = gpu_check.strip().split("\n")
            for n in names:
                n_clean = n.strip()
                if any(x in n_clean.lower() for x in ("arc", "iris", "intel", "radeon", "amd")):
                    info.kind = "vulkan"
                    info.name = n_clean
                    info.notes.append("Using Vulkan build for non-NVIDIA GPU")
                    return info

    # Linux Intel Arc -> Vulkan (vulkaninfo present)
    if shutil.which("vulkaninfo"):
        info.kind = "vulkan"
        info.name = "Vulkan-capable GPU"
        return info

    info.kind = "cpu"
    info.name = platform.processor() or "CPU"
    return info


# ---------------------------------------------------------------------------
# Quantization selection
# ---------------------------------------------------------------------------

def pick_quant(vram_gb: float, ram_gb: float, accel_kind: str = "cuda") -> str:
    """Pick the largest quant that fits in available accelerator memory."""
    # For CPU/Vulkan-on-iGPU, we use system RAM as the memory pool
    if accel_kind in ("cpu", "vulkan") and vram_gb < 2:
        pool_gb = ram_gb * 0.5  # leave half RAM for OS + apps
    elif accel_kind == "metal":
        pool_gb = ram_gb * 0.6  # macOS unified memory, generous
    else:
        pool_gb = vram_gb * 0.85  # leave headroom for KV cache

    # Pick largest quant whose RAM estimate fits
    for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"):
        if QUANTS[q]["ram_gb"] <= pool_gb:
            return q
    return "Q3_K_M"  # fallback — smallest


# ---------------------------------------------------------------------------
# llama.cpp binary acquisition
# ---------------------------------------------------------------------------

def _pick_release_asset(assets: list, accel: AccelInfo) -> Optional[str]:
    """Pick the right llama.cpp release asset for this platform/accelerator."""
    # llama.cpp asset naming: llama-b<NUMBER>-bin-<OS>-<ACCEL>-<ARCH>.zip
    # Examples:
    #   llama-b4400-bin-win-cuda-cu12.4-x64.zip
    #   llama-b4400-bin-win-vulkan-x64.zip
    #   llama-b4400-bin-win-cpu-x64.zip
    #   llama-b4400-bin-ubuntu-x64.zip
    #   llama-b4400-bin-macos-arm64.zip
    os_tag = {"windows": "win", "linux": "ubuntu", "macos": "macos"}.get(accel.os_family, "ubuntu")
    accel_tag = accel.kind  # cuda, vulkan, cpu, metal

    # macOS only has metal builds (no separate cpu)
    if accel.os_family == "macos":
        for a in assets:
            name = a.get("name", "")
            if "macos" in name and accel.arch in name and name.endswith(".zip"):
                return a.get("browser_download_url")
        return None

    # Score candidates
    candidates = []
    for a in assets:
        name = a.get("name", "").lower()
        if not name.endswith(".zip"):
            continue
        if os_tag not in name:
            continue
        score = 0
        if accel_tag in name:
            score += 100
        if accel.arch in name or "x64" in name:
            score += 10
        if accel.kind == "cuda" and "cu12" in name:
            score += 5
        if score > 0:
            candidates.append((score, a.get("browser_download_url")))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Fallback to CPU build for the OS
    for a in assets:
        name = a.get("name", "").lower()
        if name.endswith(".zip") and os_tag in name and "cpu" in name:
            return a.get("browser_download_url")
    return None


def _download(url: str, dest: Path, label: str = "download") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label}: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AitherADK-LocalOrchestrator/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            written = 0
            chunk = 256 * 1024
            with open(dest, "wb") as f:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    written += len(data)
                    if total:
                        pct = (written / total) * 100
                        print(f"\r    {pct:5.1f}%  ({written / 1e6:.1f} / {total / 1e6:.1f} MB)",
                              end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"  ERROR: download failed: {e}", file=sys.stderr)
        return False


def install_llamacpp(accel: AccelInfo, dry_run: bool = False) -> Optional[Path]:
    """Download + extract llama.cpp; return path to llama-server binary."""
    LLAMACPP_DIR.mkdir(parents=True, exist_ok=True)
    binary_name = "llama-server.exe" if accel.os_family == "windows" else "llama-server"
    existing = list(LLAMACPP_DIR.rglob(binary_name))
    if existing:
        print(f"  Found existing llama-server: {existing[0]}")
        return existing[0]

    if dry_run:
        print(f"  [DRY] Would download llama.cpp for {accel.os_family}/{accel.kind}")
        return LLAMACPP_DIR / binary_name

    print("  Querying llama.cpp latest release...")
    try:
        req = urllib.request.Request(LLAMACPP_RELEASES_API,
                                     headers={"User-Agent": "AitherADK/1.0",
                                              "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            release = json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: GitHub API failed: {e}", file=sys.stderr)
        return None

    asset_url = _pick_release_asset(release.get("assets", []), accel)
    if not asset_url:
        print(f"  ERROR: no matching llama.cpp build for {accel.os_family}/{accel.kind}/{accel.arch}",
              file=sys.stderr)
        return None

    zip_path = LLAMACPP_DIR / "llama-cpp.zip"
    if not _download(asset_url, zip_path, "llama.cpp"):
        return None

    print("  Extracting...")
    try:
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(LLAMACPP_DIR)
        zip_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"  ERROR: extract failed: {e}", file=sys.stderr)
        return None

    found = list(LLAMACPP_DIR.rglob(binary_name))
    if not found:
        print(f"  ERROR: llama-server binary not found after extract", file=sys.stderr)
        return None

    # Mark executable on POSIX
    if accel.os_family != "windows":
        try:
            os.chmod(found[0], 0o755)
        except Exception:
            pass

    print(f"  llama-server installed: {found[0]}")
    return found[0]


# ---------------------------------------------------------------------------
# GGUF model download
# ---------------------------------------------------------------------------

def install_model(repo: str, quant: str, dry_run: bool = False) -> Optional[Path]:
    """Download the requested quant GGUF from HuggingFace; return path to file."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Convention: <repo-base>-<quant>.gguf (matches bartowski/nvidia layout)
    # Try standard naming first
    repo_short = repo.split("/")[-1].replace("-GGUF", "")
    candidates = [
        f"{repo_short}-{quant}.gguf",
        f"{repo_short.lower()}-{quant}.gguf",
        f"{repo_short.replace('_', '-')}-{quant}.gguf",
    ]

    for filename in candidates:
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 100 * 1024 * 1024:  # >100MB sanity
            print(f"  Found existing model: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
            return dest

    if dry_run:
        print(f"  [DRY] Would download {repo} {quant}")
        return MODELS_DIR / candidates[0]

    # Try each candidate filename
    hf_token = os.environ.get("HF_TOKEN", "")
    last_err = None
    for filename in candidates:
        url = HF_RESOLVE_BASE.format(repo=repo, filename=filename)
        dest = MODELS_DIR / filename
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "AitherADK/1.0",
                **({"Authorization": f"Bearer {hf_token}"} if hf_token else {}),
            })
            # HEAD-style: try to download. urllib doesn't HEAD cleanly across mirrors, just GET.
            print(f"  Trying: {url}")
            if _download(url, dest, f"{quant} GGUF"):
                if dest.stat().st_size > 100 * 1024 * 1024:
                    return dest
                else:
                    dest.unlink(missing_ok=True)
                    last_err = "file too small (likely 404 HTML)"
            else:
                last_err = "download failed"
        except Exception as e:
            last_err = str(e)
            continue

    print(f"  ERROR: could not download {quant} from {repo}: {last_err}", file=sys.stderr)
    print(f"  Try manually: huggingface-cli download {repo} --include '*{quant}*' --local-dir {MODELS_DIR}")
    return None


# ---------------------------------------------------------------------------
# Service installation (systemd / launchd / Task Scheduler)
# ---------------------------------------------------------------------------

def _build_server_cmd(binary: Path, model: Path, port: int, accel: AccelInfo,
                      ctx: int = DEFAULT_CTX) -> list[str]:
    cmd = [
        str(binary),
        "-m", str(model),
        "-c", str(ctx),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--alias", DEFAULT_SERVED_NAME,
        "--jinja",  # use model's chat template
    ]
    # GPU layer offload — -1 = all layers
    if accel.kind in ("cuda", "vulkan", "metal"):
        cmd.extend(["-ngl", "99"])
    return cmd


def install_service(binary: Path, model: Path, port: int, accel: AccelInfo,
                    dry_run: bool = False) -> bool:
    """Install background service for this OS."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = _build_server_cmd(binary, model, port, accel)

    if accel.os_family == "linux":
        return _install_systemd_user(cmd, dry_run)
    elif accel.os_family == "macos":
        return _install_launchd(cmd, dry_run)
    elif accel.os_family == "windows":
        return _install_windows_task(cmd, dry_run)
    return False


def _install_systemd_user(cmd: list[str], dry_run: bool) -> bool:
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_path = unit_dir / "aither-orchestrator.service"
    exec_start = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    unit = f"""[Unit]
Description=AitherOS Local Orchestrator (llama.cpp + Nemotron-Orchestrator-8B)
After=network.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=5
StandardOutput=append:{LOG_DIR}/orchestrator.log
StandardError=append:{LOG_DIR}/orchestrator.err

[Install]
WantedBy=default.target
"""
    if dry_run:
        print(f"  [DRY] Would write systemd unit: {unit_path}")
        print(f"  [DRY] {exec_start}")
        return True
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit)
    print(f"  Wrote: {unit_path}")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", "aither-orchestrator.service"], check=False)
    print("  Service enabled + started (systemctl --user)")
    return True


def _install_launchd(cmd: list[str], dry_run: bool) -> bool:
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_path = plist_dir / "com.aitherium.orchestrator.plist"
    args_xml = "\n".join(f"        <string>{c}</string>" for c in cmd)
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aitherium.orchestrator</string>
    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{LOG_DIR}/orchestrator.log</string>
    <key>StandardErrorPath</key><string>{LOG_DIR}/orchestrator.err</string>
</dict>
</plist>
"""
    if dry_run:
        print(f"  [DRY] Would write launchd plist: {plist_path}")
        return True
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist)
    print(f"  Wrote: {plist_path}")
    subprocess.run(["launchctl", "unload", str(plist_path)], check=False,
                   capture_output=True)
    subprocess.run(["launchctl", "load", str(plist_path)], check=False)
    print("  Service loaded (launchctl)")
    return True


def _install_windows_task(cmd: list[str], dry_run: bool) -> bool:
    # Build a wrapper .cmd so we can capture logs cleanly
    wrapper = LLAMACPP_DIR / "aither-orchestrator.cmd"
    log_out = LOG_DIR / "orchestrator.log"
    cmd_line = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    wrapper_content = f"@echo off\r\ncd /d \"%~dp0\"\r\n{cmd_line} 1>>\"{log_out}\" 2>&1\r\n"
    if dry_run:
        print(f"  [DRY] Would write wrapper: {wrapper}")
        print(f"  [DRY] schtasks /create /tn AitherOrchestrator /sc onlogon ...")
        return True
    wrapper.write_text(wrapper_content)
    print(f"  Wrote wrapper: {wrapper}")
    # Create scheduled task — runs at user logon, restart on failure
    rc = subprocess.run([
        "schtasks", "/create", "/f",
        "/tn", "AitherOrchestrator",
        "/tr", str(wrapper),
        "/sc", "onlogon",
        "/rl", "limited",
    ], capture_output=True, text=True)
    if rc.returncode != 0:
        print(f"  WARN: schtasks failed: {rc.stderr}", file=sys.stderr)
        print(f"  Falling back to inline launch — service will not restart on reboot.")
        # Fire-and-forget start
        subprocess.Popen([str(wrapper)], creationflags=0x00000008)  # DETACHED_PROCESS
        return True
    print("  Task scheduled (runs at user logon)")
    # Start now too
    subprocess.run(["schtasks", "/run", "/tn", "AitherOrchestrator"],
                   capture_output=True, text=True)
    return True


# ---------------------------------------------------------------------------
# Config registration
# ---------------------------------------------------------------------------

def register_config(port: int, model_path: Path, quant: str, accel: AccelInfo) -> None:
    AITHER_HOME.mkdir(parents=True, exist_ok=True)
    cfg: dict = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except Exception:
            cfg = {}
    cfg["default_backend"] = "openai"
    cfg["setup_backend"] = "llamacpp"
    cfg["inference_url"] = f"http://localhost:{port}/v1"
    cfg["orchestrator_model"] = DEFAULT_SERVED_NAME
    cfg["llamacpp"] = {
        "port": port,
        "model_path": str(model_path),
        "quant": quant,
        "accel": accel.kind,
        "accel_name": accel.name,
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"  Config saved: {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Health + status
# ---------------------------------------------------------------------------

@dataclass
class StatusResult:
    running: bool
    port: int
    url: str
    model: str = ""
    error: str = ""


def status(port: int = DEFAULT_PORT) -> StatusResult:
    url = f"http://localhost:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m.get("id", "") for m in data.get("data", [])]
            return StatusResult(running=True, port=port, url=url,
                                model=models[0] if models else "")
    except Exception as e:
        return StatusResult(running=False, port=port, url=url, error=str(e))


def smoke_test(port: int = DEFAULT_PORT, timeout: int = 60) -> bool:
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": DEFAULT_SERVED_NAME,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 5,
    }).encode()
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  Smoke test: {content!r}")
            return bool(content)
    except Exception as e:
        print(f"  Smoke test failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

def uninstall(port: int = DEFAULT_PORT, purge: bool = False) -> bool:
    osf = "windows" if sys.platform == "win32" else ("macos" if sys.platform == "darwin" else "linux")
    if osf == "linux":
        subprocess.run(["systemctl", "--user", "disable", "--now",
                        "aither-orchestrator.service"], check=False)
        unit = Path.home() / ".config" / "systemd" / "user" / "aither-orchestrator.service"
        unit.unlink(missing_ok=True)
    elif osf == "macos":
        plist = Path.home() / "Library" / "LaunchAgents" / "com.aitherium.orchestrator.plist"
        if plist.exists():
            subprocess.run(["launchctl", "unload", str(plist)], check=False)
            plist.unlink(missing_ok=True)
    elif osf == "windows":
        subprocess.run(["schtasks", "/delete", "/tn", "AitherOrchestrator", "/f"],
                       check=False, capture_output=True)
    if purge:
        shutil.rmtree(LLAMACPP_DIR, ignore_errors=True)
        shutil.rmtree(MODELS_DIR, ignore_errors=True)
    print("  Uninstalled.")
    return True


# ---------------------------------------------------------------------------
# Main install flow
# ---------------------------------------------------------------------------

@dataclass
class InstallResult:
    success: bool
    port: int = DEFAULT_PORT
    binary: Optional[Path] = None
    model: Optional[Path] = None
    quant: str = ""
    accel: Optional[AccelInfo] = None
    service_installed: bool = False
    error: str = ""


def install(
    quant: Optional[str] = None,
    port: int = DEFAULT_PORT,
    model_repo: str = DEFAULT_MODEL_REPO,
    service: bool = True,
    dry_run: bool = False,
) -> InstallResult:
    """Install local orchestrator end-to-end. Returns InstallResult."""
    print()
    print("=" * 60)
    print("  AitherOS Local Orchestrator — llama.cpp + Nemotron-8B")
    print("=" * 60)

    # 1. Detect accelerator
    accel = detect_accel()
    print(f"\n  [1/5] Hardware: {accel.os_family}/{accel.arch}, accel={accel.kind}, "
          f"GPU={accel.name}, VRAM={accel.vram_gb:.1f}GB, RAM={accel.ram_gb:.1f}GB")
    for n in accel.notes:
        print(f"        {n}")

    # 2. Pick quant
    if not quant:
        quant = pick_quant(accel.vram_gb, accel.ram_gb, accel.kind)
    if quant not in QUANTS:
        return InstallResult(success=False, error=f"unknown quant {quant!r}")
    qmeta = QUANTS[quant]
    print(f"  [2/5] Quant: {quant} (~{qmeta['size_gb']:.1f} GB on disk, "
          f"~{qmeta['ram_gb']:.1f} GB working — {qmeta['quality']})")

    # 3. Install llama.cpp
    print(f"  [3/5] llama.cpp:")
    binary = install_llamacpp(accel, dry_run=dry_run)
    if not binary:
        return InstallResult(success=False, accel=accel,
                             error="llama.cpp install failed")

    # 4. Download model
    print(f"  [4/5] Model:")
    model = install_model(model_repo, quant, dry_run=dry_run)
    if not model:
        return InstallResult(success=False, accel=accel, binary=binary,
                             error="model download failed")

    # 5. Service + config
    print(f"  [5/5] Service:")
    svc_ok = False
    if service:
        svc_ok = install_service(binary, model, port, accel, dry_run=dry_run)
    else:
        print(f"  Skipping service install (--no-service)")
        cmd = _build_server_cmd(binary, model, port, accel)
        print(f"  Run manually: {' '.join(cmd)}")

    if not dry_run:
        register_config(port, model, quant, accel)

    print()
    print("=" * 60)
    print(f"  Endpoint: http://localhost:{port}/v1  (OpenAI-compatible)")
    print(f"  Model:    {DEFAULT_SERVED_NAME}  ({quant})")
    print(f"  Logs:     {LOG_DIR}")
    if accel.os_family == "linux":
        print(f"  Manage:   systemctl --user status aither-orchestrator")
    elif accel.os_family == "macos":
        print(f"  Manage:   launchctl list | grep aitherium")
    else:
        print(f"  Manage:   schtasks /query /tn AitherOrchestrator")
    print("=" * 60)
    print()
    return InstallResult(success=True, port=port, binary=binary, model=model,
                         quant=quant, accel=accel, service_installed=svc_ok)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="aither-local-orchestrator",
        description="Install + run a local Nemotron-Orchestrator-8B via llama.cpp.",
    )
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("install", help="Download + install + start the service")
    pi.add_argument("--quant", choices=list(QUANTS.keys()),
                    help="GGUF quant (auto-picked from hardware if omitted)")
    pi.add_argument("--port", type=int, default=DEFAULT_PORT)
    pi.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    pi.add_argument("--no-service", action="store_true",
                    help="Skip auto-start service install — print run command instead")
    pi.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="Check if local orchestrator is running")
    sub.add_parser("smoke", help="Run a chat completion to verify end-to-end")

    pu = sub.add_parser("uninstall", help="Remove service and (optionally) all files")
    pu.add_argument("--purge", action="store_true",
                    help="Also delete llama.cpp binary and GGUF model")

    args = p.parse_args(argv)
    cmd = args.command or "install"

    if cmd == "install":
        r = install(
            quant=getattr(args, "quant", None),
            port=getattr(args, "port", DEFAULT_PORT),
            model_repo=getattr(args, "model_repo", DEFAULT_MODEL_REPO),
            service=not getattr(args, "no_service", False),
            dry_run=getattr(args, "dry_run", False),
        )
        return 0 if r.success else 1
    elif cmd == "status":
        s = status()
        if s.running:
            print(f"  Running on {s.url}  (model: {s.model})")
            return 0
        print(f"  Not running. Last error: {s.error}")
        return 1
    elif cmd == "smoke":
        return 0 if smoke_test() else 1
    elif cmd == "uninstall":
        return 0 if uninstall(purge=getattr(args, "purge", False)) else 1
    else:
        p.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
