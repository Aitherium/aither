"""
Saga Build System — Create distributable packages.

Supports three distribution modes:

    1. pip install (PyPI)
       - `pip install aither-saga` → installs saga + adk as dependency
       - Entry point: `saga` command → cli.py:main()
       - Pre-built UI assets bundled as package data

    2. PyInstaller (single binary)
       - `python build.py pyinstaller` → creates saga.exe / saga binary
       - Bundles Python, adk, saga_engine, UI, and Ollama model recommendation
       - ~50MB binary, no Python installation needed

    3. Tauri (native desktop app, v1.1)
       - `python build.py tauri` → creates native installer
       - Manages Ollama lifecycle, system tray, native file dialogs
       - ~5MB native shell wrapping the browser-based UI

Usage:
    python build.py pip          # Build pip package (sdist + wheel)
    python build.py pyinstaller  # Build standalone binary
    python build.py ui           # Build UI only (Vite)
    python build.py all          # Build everything
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
UI_DIR = PACKAGE_DIR / "ui"
DIST_DIR = PACKAGE_DIR / "dist"


def build_ui():
    """Build the Vite+React frontend."""
    if not UI_DIR.exists():
        print("UI directory not found — skipping UI build")
        print("Create the UI with: cd packages/saga/ui && npm create vite@latest . -- --template react-ts")
        return False

    print("Building UI...")
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(UI_DIR),
        shell=True,
    )
    if result.returncode != 0:
        print("UI build failed")
        return False

    print(f"UI built: {UI_DIR / 'dist'}")
    return True


def build_pip():
    """Build pip package (sdist + wheel)."""
    print("Building pip package...")

    # Ensure pyproject.toml exists
    pyproject = PACKAGE_DIR / "pyproject.toml"
    if not pyproject.exists():
        _create_pyproject()

    # Build UI first if available
    build_ui()

    result = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(DIST_DIR)],
        cwd=str(PACKAGE_DIR),
    )
    if result.returncode == 0:
        print(f"Package built: {DIST_DIR}")
    return result.returncode == 0


def build_pyinstaller():
    """Build standalone binary with PyInstaller."""
    print("Building standalone binary with PyInstaller...")

    # Build UI first
    build_ui()

    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-
import os
block_cipher = None

a = Analysis(
    ['{PACKAGE_DIR / "cli.py"}'],
    pathex=['{PACKAGE_DIR}', '{PACKAGE_DIR.parent.parent}'],
    binaries=[],
    datas=[
        ('{PACKAGE_DIR / "saga_engine"}', 'saga_engine'),
        ('{PACKAGE_DIR / "prompts"}', 'prompts'),
        ('{PACKAGE_DIR / "worlds"}', 'worlds'),
        ('{PACKAGE_DIR / "agent.yaml"}', '.'),
        ('{PACKAGE_DIR / "identity.yaml"}', '.'),
    ],
    hiddenimports=[
        'adk', 'adk.agent', 'adk.server', 'adk.memory', 'adk.graph_memory',
        'adk.faculties.embeddings', 'adk.setup', 'adk.shell_launcher',
        'adk.llm', 'adk.tools', 'adk.config', 'adk.identity', 'adk.nanogpt',
        'pydantic', 'uvicorn', 'fastapi', 'httpx', 'yaml',
        'saga_engine', 'saga_engine.graph', 'saga_engine.memory',
        'saga_engine.context', 'saga_engine.models',
        'saga_engine.simulation', 'saga_engine.nanogpt_gen',
        'saga_engine.sim_event_mapper', 'saga_engine.addons',
        'saga_engine.mcts', 'saga_engine.continuity',
        'saga_engine.embedding_recall', 'saga_engine.style',
        'saga_engine.effort_mapper', 'saga_engine.elysium',
        'saga_engine.memory_consolidation', 'saga_engine.persistent_memory',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'numpy.testing'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add UI dist if available
ui_dist = '{PACKAGE_DIR / "ui" / "dist"}'
if os.path.isdir(ui_dist):
    a.datas += Tree(ui_dist, prefix='ui/dist')

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='saga',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
"""

    spec_path = PACKAGE_DIR / "saga.spec"
    spec_path.write_text(spec_content)

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec_path),
         "--distpath", str(DIST_DIR), "--clean"],
        cwd=str(PACKAGE_DIR),
    )

    spec_path.unlink(missing_ok=True)
    if result.returncode == 0:
        print(f"Binary built: {DIST_DIR / 'saga'}")
    return result.returncode == 0


def _create_pyproject():
    """No-op — pyproject.toml is committed to the repo."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Saga Build System")
    parser.add_argument("target", choices=["pip", "pyinstaller", "ui", "all"],
                        help="What to build")
    args = parser.parse_args()

    DIST_DIR.mkdir(exist_ok=True)

    if args.target == "ui":
        build_ui()
    elif args.target == "pip":
        build_pip()
    elif args.target == "pyinstaller":
        build_pyinstaller()
    elif args.target == "all":
        build_ui()
        build_pip()
        build_pyinstaller()


if __name__ == "__main__":
    main()
