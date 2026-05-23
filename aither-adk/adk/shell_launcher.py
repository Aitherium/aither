"""AitherShell binary launcher — download and run the compiled CLI.

The binary is published as a GitHub Release asset on Aitherium/AitherOS
(tag: shell-cli-v*). `adk shell` downloads the platform-appropriate
binary on first use, caches it in ~/.aither/bin/, and launches it.
"""

from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
from pathlib import Path

_REPO = "Aitherium/aither-adk"
_CACHE_DIR = Path.home() / ".aither" / "bin"

_PLATFORM_BINARY = {
    ("Windows", "AMD64"): "aither-shell-win64.exe",
    ("Windows", "x86_64"): "aither-shell-win64.exe",
    ("Linux", "x86_64"): "aither-shell-linux-x64",
    ("Linux", "aarch64"): "aither-shell-linux-x64",  # fallback — no ARM build yet
    ("Darwin", "arm64"): "aither-shell-macos-arm64",
    ("Darwin", "x86_64"): "aither-shell-macos-arm64",
}


def _get_binary_name() -> str:
    key = (platform.system(), platform.machine())
    name = _PLATFORM_BINARY.get(key)
    if not name:
        print(f"Unsupported platform: {key[0]} {key[1]}")
        sys.exit(1)
    return name


def _get_binary_path() -> Path:
    name = _get_binary_name()
    return _CACHE_DIR / name


def _download_binary(version: str = "latest") -> Path:
    """Download the AitherShell binary from GitHub Releases."""
    import httpx

    name = _get_binary_name()
    dest = _CACHE_DIR / name
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching release info from {_REPO}...")
    try:
        if version != "latest":
            api_url = f"https://api.github.com/repos/{_REPO}/releases/tags/shell-cli-v{version}"
            resp = httpx.get(api_url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            release = resp.json()
        else:
            # Find latest shell-cli-v* release (repo may have other release tags)
            api_url = f"https://api.github.com/repos/{_REPO}/releases?per_page=20"
            resp = httpx.get(api_url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            release = None
            for r in resp.json():
                tag = r.get("tag_name", "")
                if tag.startswith("shell-cli-v") and not r.get("draft"):
                    release = r
                    break
            if not release:
                print("No shell-cli release found")
                sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Failed to fetch release: {e}")
        sys.exit(1)

    # Find the matching asset
    asset_url = None
    for asset in release.get("assets", []):
        if asset["name"] == name:
            asset_url = asset["browser_download_url"]
            break

    if not asset_url:
        print(f"Binary '{name}' not found in release {release.get('tag_name', '?')}")
        print(f"Available assets: {[a['name'] for a in release.get('assets', [])]}")
        sys.exit(1)

    print(f"Downloading {name} ({release.get('tag_name', '')})...")
    try:
        with httpx.stream("GET", asset_url, follow_redirects=True, timeout=120) as dl:
            dl.raise_for_status()
            total = int(dl.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in dl.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r  {pct}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)
            print()
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    # Make executable on Unix
    if platform.system() != "Windows":
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Installed: {dest}")
    return dest


def _preflight_check() -> tuple[bool, str]:
    """Verify at least one LLM backend is reachable before launching shell."""
    import urllib.request
    import urllib.error

    # Check vLLM ports
    for port in (8200, 8201, 8000):
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True, f"vLLM on :{port}"
        except (urllib.error.URLError, ConnectionError, OSError):
            pass

    # Check Ollama
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return True, "Ollama"
    except (urllib.error.URLError, ConnectionError, OSError):
        pass

    # Check Genesis
    try:
        req = urllib.request.Request("http://localhost:8001/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return True, "Genesis"
    except (urllib.error.URLError, ConnectionError, OSError):
        pass

    # Check for API keys in config
    try:
        from adk.config import load_saved_config
    except ImportError:
        load_saved_config = None
    if load_saved_config is not None:
        try:
            saved = load_saved_config()
            if saved.get("api_key") or saved.get("reasoning_api_key"):
                return True, "cloud API key"
        except (OSError, ValueError):
            pass

    if os.environ.get("AITHER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return True, "cloud API key"

    return False, ""


def cmd_shell(args) -> int:
    """Launch AitherShell interactive terminal."""
    binary = _get_binary_path()

    # Auto-download on first run (SHELL-2 fix)
    if not binary.exists():
        print("AitherShell not found — downloading...")
        try:
            _download_binary()
        except SystemExit:
            print("Download failed. Try: adk shell --install")
            return 1

    if getattr(args, "install", False):
        _download_binary()
        return 0

    if not binary.exists():
        print("AitherShell binary not found. Run: adk shell --install")
        return 1

    # Pre-flight: verify a backend is reachable (SHELL-3 fix)
    ok, backend = _preflight_check()
    if not ok:
        print("Warning: No LLM backend detected.")
        print("  Run 'adk setup' to configure local inference, or")
        print("  Run 'adk login' to connect to Aitherium cloud.")
        print()

    # Build command
    cmd = [str(binary)]

    # Pass full config via env vars (SHELL-4 fix)
    try:
        from adk.config import load_saved_config
    except ImportError:
        load_saved_config = None
    if load_saved_config is not None:
        try:
            saved = load_saved_config()
            _ENV_MAP = {
                "api_key": "AITHER_API_KEY",
                "tenant_id": "AITHER_TENANT_ID",
                "inference_url": "AITHER_INFERENCE_URL",
                "reasoning_url": "AITHER_REASONING_URL",
                "reasoning_backend": "AITHER_REASONING_BACKEND",
            }
            for config_key, env_key in _ENV_MAP.items():
                val = saved.get(config_key, "")
                if val and env_key not in os.environ:
                    os.environ[env_key] = val
        except (OSError, ValueError):
            pass

    # Set backend URL — AITHER_API_URL is canonical, AITHER_GENESIS_URL is legacy
    api_url = getattr(args, "api_url", None) or getattr(args, "genesis", None)
    if api_url:
        os.environ["AITHER_API_URL"] = api_url
    elif "AITHER_API_URL" not in os.environ and "AITHER_GENESIS_URL" not in os.environ:
        # Auto-detect: if ADK server is running locally, point shell at it
        try:
            from adk.config import Config
            cfg = Config.from_env()
            if cfg.server_port:
                os.environ["AITHER_API_URL"] = f"http://127.0.0.1:{cfg.server_port}"
        except ImportError:
            pass  # adk.config not available — shell uses its own defaults

    shell_args = getattr(args, "shell_args", [])
    if shell_args:
        cmd.extend(shell_args)

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError:
        print(f"Binary not executable or missing: {binary}")
        print("Try: adk shell --install")
        return 1
