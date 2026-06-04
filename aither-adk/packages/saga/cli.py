"""Saga CLI — Entry point for the standalone app.

Integrates with AitherShell/AitherNode for LLM management and provides
the `saga` command that handles everything from first-run setup to
launching the full server + UI.

Usage:
    saga                    # Start server (runs setup if needed)
    saga setup              # Run/re-run first-time setup
    saga models             # List/manage LLM models via AitherShell
    saga models pull NAME   # Pull a model via Ollama
    saga projects           # List saved projects
    saga export NAME        # Export a project
    saga shell              # Launch AitherShell for advanced LLM management
    saga version            # Show version
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

SAGA_HOME = Path.home() / ".saga"
PACKAGE_DIR = Path(__file__).parent

logger = logging.getLogger("saga.cli")


def _setup_paths():
    """Add package dir to path for saga_engine imports."""
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))


def cmd_start(args):
    """Start the Saga server (default command)."""
    _setup_paths()

    # Run setup if not done
    from setup import is_setup_complete, run_setup
    if not is_setup_complete():
        print("First-time setup required.\n")
        run_setup()

    # Load config
    from setup import load_config
    config = load_config()

    port = args.port or int(config.get("port", 8080))
    host = args.host or "127.0.0.1"

    print(f"\nStarting Saga on http://{host}:{port}")
    print(f"Model: {config.get('model', 'auto-detect')}")
    print(f"Data: {SAGA_HOME}")

    # Open browser after short delay
    if not args.no_browser:
        import threading
        def _open():
            import time
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    # Start server
    import uvicorn
    from saga_server import create_saga_app
    app = create_saga_app(port=port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_setup(args):
    """Run first-time setup."""
    _setup_paths()
    from setup import run_setup
    run_setup()


def cmd_models(args):
    """Manage LLM models."""
    if args.model_action == "pull":
        if not args.model_name:
            print("Usage: saga models pull <model-name>")
            return
        print(f"Pulling {args.model_name}...")
        result = subprocess.run(["ollama", "pull", args.model_name])
        if result.returncode == 0:
            # Update config
            _setup_paths()
            from setup import load_config, save_config
            config = load_config()
            config["model"] = args.model_name
            save_config(config)
            print(f"Model set to: {args.model_name}")
        return

    if args.model_action == "list":
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Ollama not running. Start it with: ollama serve")
        return

    if args.model_action == "recommend":
        _setup_paths()
        from setup import MODEL_TIERS
        print("\nRecommended models for Saga:\n")
        for tier in MODEL_TIERS:
            print(f"  {tier['label']}: {tier['name']} ({tier['vram_gb']}GB VRAM, {tier['context']} context)")
        return

    # Default: list
    cmd_models_list(args)


def cmd_models_list(args):
    """List models."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Installed models:\n")
        print(result.stdout)
    else:
        print("Ollama not running. Install from: https://ollama.com/download")


def cmd_projects(args):
    """List saved projects."""
    projects_dir = SAGA_HOME / "projects"
    if not projects_dir.exists():
        print("No saved projects yet.")
        return

    print("\nSaved Projects:\n")
    for d in sorted(projects_dir.iterdir()):
        if d.is_dir():
            meta_path = d / "project.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                print(f"  {meta.get('name', d.name)}")
                print(f"    Story: {meta.get('story_name', 'Unknown')}")
                print(f"    Turn:  {meta.get('turn_number', 0)}")
                print(f"    Saved: {meta.get('saved_at', 'Unknown')}")
                print()
            else:
                print(f"  {d.name} (no metadata)")


def cmd_shell(args):
    """Launch AitherShell for advanced LLM management."""
    try:
        from adk.shell_launcher import launch
        launch(extra_args=args.shell_args)
    except ImportError:
        print("AitherShell integration requires aither-adk.")
        print("Install: pip install aither-adk")
    except Exception as e:
        print(f"Failed to launch AitherShell: {e}")
        print("You can manage models directly with: ollama list / ollama pull <model>")


def cmd_hardware(args):
    """Detect hardware and show capabilities."""
    _setup_paths()
    from adk.setup import AgentSetup
    setup = AgentSetup()
    info = asyncio.run(setup.detect_hardware())

    print(f"\nSystem: {info.os_name} {info.os_version} ({info.arch})")
    print(f"RAM: {info.ram_gb:.1f} GB")
    print(f"Python: {info.python_version}")

    if info.gpu.vendor != "none":
        print(f"\nGPU: {info.gpu.name}")
        print(f"VRAM: {info.gpu.vram_mb} MB")
        if info.gpu.cuda_version:
            print(f"CUDA: {info.gpu.cuda_version}")
        if info.gpu.count > 1:
            print(f"GPUs: {info.gpu.count} (total VRAM: {info.gpu.total_vram_mb} MB)")
    else:
        print("\nGPU: None detected (CPU-only mode)")

    print(f"\nOllama: {'installed' if info.ollama_installed else 'not found'}")
    if info.ollama_running:
        print(f"  Running with models: {', '.join(info.ollama_models) or 'none'}")
    print(f"Profile: {info.profile}")


def cmd_addons(args):
    """Manage addons."""
    _setup_paths()
    from saga_engine.addons import get_addon_registry

    registry = get_addon_registry()
    registry.discover()

    if args.addon_action == "list":
        addons = registry.list_addons()
        if not addons:
            print("No addons installed.")
            return
        print("\nInstalled Addons:\n")
        for a in addons:
            status = "active" if a["active"] else ("error" if a["error"] else "inactive")
            price = "free" if a["free"] else f"${a['price']:.0f}"
            print(f"  [{status}] {a['name']} v{a['version']} ({price})")
            print(f"         {a['description']}")
        print()

    elif args.addon_action == "available":
        catalog = registry.list_available()
        print("\nAvailable Addons:\n")
        for a in catalog:
            installed = " (installed)" if a["installed"] else ""
            price = "free" if a["free"] else f"${a['price']:.0f}"
            gpu = " [GPU required]" if a.get("requires_gpu") else ""
            print(f"  {a['name']} v{a['version']} — {price}{gpu}{installed}")
            print(f"    {a['description']}")
        print()
        print("Install addons with: pip install <addon-name>")

    elif args.addon_action == "info":
        if not args.addon_name:
            print("Usage: saga addons info <addon-name>")
            return
        addon = registry.get_addon(args.addon_name)
        if addon:
            m = addon.manifest
            print(f"\n{m.name} v{m.version}")
            print(f"  Category: {m.category.value}")
            print(f"  Description: {m.description}")
            print(f"  Author: {m.author}")
            print(f"  Active: {addon.active}")
            if addon.error:
                print(f"  Error: {addon.error}")
        else:
            print(f"Addon '{args.addon_name}' not found")


def cmd_version(args):
    """Show version."""
    print("Saga v1.0.0 (AitherADK)")
    print(f"Data: {SAGA_HOME}")
    _setup_paths()
    from setup import load_config
    config = load_config()
    if config.get("model"):
        print(f"Model: {config['model']}")


def main():
    parser = argparse.ArgumentParser(
        prog="saga",
        description="Saga — AI-Powered Interactive Storytelling Engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default (start)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", default="")
    parser.add_argument("--no-browser", action="store_true")

    # Setup
    subparsers.add_parser("setup", help="Run first-time setup")

    # Models
    models_p = subparsers.add_parser("models", help="Manage LLM models")
    models_p.add_argument("model_action", nargs="?", default="list",
                          choices=["list", "pull", "recommend"])
    models_p.add_argument("model_name", nargs="?", default="")

    # Projects
    subparsers.add_parser("projects", help="List saved projects")

    # Shell
    shell_p = subparsers.add_parser("shell", help="Launch AitherShell")
    shell_p.add_argument("shell_args", nargs="*", default=[])

    # Hardware
    subparsers.add_parser("hardware", help="Detect hardware capabilities")

    # Addons
    addons_p = subparsers.add_parser("addons", help="Manage addon plugins")
    addons_p.add_argument("addon_action", nargs="?", default="list",
                          choices=["list", "available", "info"])
    addons_p.add_argument("addon_name", nargs="?", default="")

    # Version
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    commands = {
        None: cmd_start,
        "setup": cmd_setup,
        "models": cmd_models,
        "projects": cmd_projects,
        "shell": cmd_shell,
        "hardware": cmd_hardware,
        "addons": cmd_addons,
        "version": cmd_version,
    }

    handler = commands.get(args.command, cmd_start)
    handler(args)


if __name__ == "__main__":
    main()
