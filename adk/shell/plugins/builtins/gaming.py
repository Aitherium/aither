"""
Gaming Mode Plugin for AitherShell
====================================

Toggle between AitherOS and gaming mode directly from AitherShell.

Usage:
    /gaming              — Enter gaming mode (free GPU + services)
    /gaming resume       — Resume AitherOS (bring everything back)
    /gaming status       — Check if gaming mode is active

Aliases: /game, /gpu
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any

from adk.shell.plugins import SlashCommand


def _find_gaming_script() -> Optional[str]:
    """Locate the Switch-GamingMode.ps1 script."""
    candidates = []

    root = os.environ.get("AITHEROS_ROOT")
    if root:
        candidates.append(Path(root) / ".." / "scripts" / "Switch-GamingMode.ps1")
        candidates.append(Path(root) / "scripts" / "Switch-GamingMode.ps1")

    here = Path(__file__).resolve()
    # aithershell/aithershell/plugins/builtins/ -> repo root
    repo_root = here.parent.parent.parent.parent.parent
    candidates.append(repo_root / "scripts" / "Switch-GamingMode.ps1")

    cwd = Path.cwd()
    candidates.extend([
        cwd / "scripts" / "Switch-GamingMode.ps1",
        cwd / ".." / "scripts" / "Switch-GamingMode.ps1",
    ])

    # Drive-root checkouts, DISCOVERED rather than hardcoded. This used to name
    # an absolute drive path explicitly — one developer's drive layout, in a public package.
    from adk.shell._repo_roots import candidate_repo_roots

    for root in candidate_repo_roots(include_cwd=False):
        candidates.append(root / "scripts" / "Switch-GamingMode.ps1")

    for c in candidates:
        resolved = c.resolve()
        if resolved.is_file():
            return str(resolved)
    return None


def _find_pwsh() -> str:
    """Find PowerShell 7 binary."""
    pwsh = shutil.which("pwsh")
    if pwsh:
        return pwsh
    ps = shutil.which("powershell")
    if ps:
        return ps
    return "pwsh"


def _is_docker_running() -> bool:
    """Quick check: is Docker daemon responding?"""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _gpu_status() -> Optional[str]:
    """Get GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            used = int(parts[0].strip())
            total = int(parts[1].strip())
            free = total - used
            return f"{used} MB used / {total} MB total ({free} MB free)"
    except Exception:
        pass
    return None


class GamingModePlugin(SlashCommand):
    name = "gaming"
    description = "Toggle gaming mode — free GPU or resume AitherOS"
    aliases = ["game", "gpu"]

    def __init__(self):
        super().__init__(
            name="gaming",
            description="Toggle gaming mode — free GPU or resume AitherOS",
            aliases=["game", "gpu"],
        )

    async def run(self, args: List[str], ctx: Dict[str, Any]) -> Optional[str]:
        if not args:
            return await self._enter_gaming_mode()

        subcmd = args[0].lower()
        rest = args[1:]

        if subcmd in ("resume", "back", "start", "up", "on"):
            stack = rest[0].capitalize() if rest else "Auto"
            return await self._resume(stack=stack)
        elif subcmd in ("stop", "off", "enter", "free"):
            return await self._enter_gaming_mode()
        elif subcmd == "status":
            return self._status()
        elif subcmd == "help":
            return self._help()
        else:
            return (
                f"Unknown sub-command: {subcmd}\n\n"
                + self._help()
            )

    async def _enter_gaming_mode(self) -> str:
        """Enter gaming mode — stop everything, free GPU."""
        script = _find_gaming_script()
        if not script:
            return (
                "❌ Switch-GamingMode.ps1 not found.\n"
                "Set AITHEROS_ROOT or run from the repo root."
            )

        pwsh = _find_pwsh()
        cmd = [pwsh, "-NoProfile", "-File", script]

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Let output stream directly to terminal
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return ""  # Output already printed to terminal
            return f"[Exit code: {result.returncode}]"
        except subprocess.TimeoutExpired:
            return "⚠️ Gaming mode script timed out after 5 minutes."
        except FileNotFoundError:
            return "❌ PowerShell 7 (pwsh) not found. Install: https://aka.ms/powershell"
        except Exception as e:
            return f"❌ Error: {e}"

    async def _resume(self, stack: str = "Auto") -> str:
        """Resume AitherOS from gaming mode."""
        script = _find_gaming_script()
        if not script:
            return (
                "❌ Switch-GamingMode.ps1 not found.\n"
                "Set AITHEROS_ROOT or run from the repo root."
            )

        pwsh = _find_pwsh()
        cmd = [pwsh, "-NoProfile", "-File", script, "-Resume"]
        if stack and stack != "Auto":
            cmd.extend(["-Stack", stack])

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Let output stream directly to terminal
                text=True,
                timeout=600,  # Resume takes longer (Docker start + health checks)
            )
            if result.returncode == 0:
                return ""  # Output already printed to terminal
            return f"[Exit code: {result.returncode}]"
        except subprocess.TimeoutExpired:
            return "⚠️ Resume timed out after 10 minutes. Docker may still be starting."
        except FileNotFoundError:
            return "❌ PowerShell 7 (pwsh) not found. Install: https://aka.ms/powershell"
        except Exception as e:
            return f"❌ Error: {e}"

    def _status(self) -> str:
        """Show current gaming mode status."""
        lines = []

        docker = _is_docker_running()
        gpu = _gpu_status()

        if docker:
            lines.append("🟢 Docker is running — AitherOS mode")
            # Check container count
            try:
                result = subprocess.run(
                    ["docker", "ps", "-q"],
                    capture_output=True, text=True, timeout=5,
                )
                count = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
                lines.append(f"   {count} container(s) running")
            except Exception:
                pass
        else:
            lines.append("🎮 Docker is stopped — Gaming mode active")

        if gpu:
            lines.append(f"   GPU: {gpu}")

        return "\n".join(lines)

    def _help(self) -> str:
        return (
            "Gaming Mode Commands:\n"
            "  /gaming                Enter gaming mode (free GPU + stop all services)\n"
            "  /gaming resume         Resume full AitherOS stack\n"
            "  /gaming resume full    Resume full stack (all profiles)\n"
            "  /gaming resume demo    Resume demo stack only\n"
            "  /gaming resume core    Resume core services only\n"
            "  /gaming status         Check current mode\n"
            "\n"
            "Aliases: /game, /gpu"
        )
