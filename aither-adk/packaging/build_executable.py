"""
Build standalone executables for AitherADK.

Produces a single-file executable (`aither` / `aither.exe`) that requires
no Python installation on the target machine.

Usage:
    python packaging/build_executable.py          # Build for current platform
    python packaging/build_executable.py --onedir  # Build as directory (faster startup)

Output:
    dist/aither        (Linux/macOS)
    dist/aither.exe    (Windows)

Requires: pip install pyinstaller
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(onedir: bool = False):
    """Build the executable."""
    entry_point = str(ROOT / "adk" / "cli.py")
    name = "aither"

    args = [
        sys.executable, "-m", "PyInstaller",
        entry_point,
        "--name", name,
        "--noconfirm",
        "--clean",
    ]

    if onedir:
        args.append("--onedir")
    else:
        args.append("--onefile")

    # Include data files
    data_sep = ";" if platform.system() == "Windows" else ":"

    # Include the docker-compose template
    compose = ROOT / "docker-compose.adk-vllm.yml"
    if compose.exists():
        args.extend(["--add-data", f"{compose}{data_sep}adk"])

    # Hidden imports that PyInstaller misses
    hidden = [
        "httpx",
        "httpx._transports",
        "httpx._transports.default",
        "yaml",
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "starlette",
        "anyio",
        "anyio._backends",
        "anyio._backends._asyncio",
    ]
    for h in hidden:
        args.extend(["--hidden-import", h])

    # Platform-specific
    if platform.system() == "Windows":
        args.extend(["--icon", "NONE"])  # TODO: add icon
    elif platform.system() == "Darwin":
        args.extend(["--target-arch", "universal2"])

    # Console app (not windowed)
    args.append("--console")

    print(f"Building {name} executable...")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Mode: {'onedir' if onedir else 'onefile'}")
    print(f"  Entry: {entry_point}")
    print()

    result = subprocess.run(args, cwd=str(ROOT))
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    dist_dir = ROOT / "dist"
    exe_name = f"{name}.exe" if platform.system() == "Windows" else name

    if onedir:
        exe_path = dist_dir / name / exe_name
    else:
        exe_path = dist_dir / exe_name

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print()
        print(f"Build complete: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
        print()
        print("Distribution:")
        print(f"  1. Upload to https://aitherium.com/download/{exe_name}")
        print(f"  2. Update install.sh/install.ps1 download URLs")
        print(f"  3. Update winget/brew/npm manifests with SHA256")
    else:
        print(f"Warning: expected output at {exe_path} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AitherADK executable")
    parser.add_argument("--onedir", action="store_true",
                        help="Build as directory instead of single file")
    args = parser.parse_args()
    build(onedir=args.onedir)
