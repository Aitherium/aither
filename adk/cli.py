"""CLI scaffolding — `aither init` and `aither run` commands.

Usage:
    aither init myproject          # Scaffold a new agent project
    aither run                     # Start the server (reads config.yaml)
    aither run --identity lyra -p 9000
"""

from __future__ import annotations
from adk._tls import tls_verify

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from adk.config import load_saved_config, save_saved_config


def _fix_ollama_host(raw: str) -> str:
    """Rewrite Ollama's bind address (0.0.0.0) to connectable localhost."""
    if not raw:
        return "http://localhost:11434"
    if "0.0.0.0" in raw:
        raw = raw.replace("0.0.0.0", "localhost")
    if not raw.startswith("http"):
        raw = "http://" + raw
    return raw

_AGENT_TEMPLATE = '''\
"""My AitherADK agent."""

from adk import AitherAgent, tool

agent = AitherAgent("{name}")


@agent.tool
def hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {{name}}!"


async def main():
    response = await agent.chat("Say hello to the world")
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''

_CONFIG_TEMPLATE = """\
# AitherADK agent configuration
# See https://github.com/Aitherium/aither-adk for docs

identity: {name}
port: 8080

# LLM backend: auto, ollama, openai, anthropic, gateway
llm_backend: auto

# Uncomment to set a specific model
# model: nemotron-orchestrator-8b

# Built-in tools (enabled by default)
builtin_tools: true

# Safety checks (enabled by default)
safety: true

# Required tool packs (auto-loaded at startup)
# required_packs:
#   - git-github
#   - codegraph
"""

_TOOLS_TEMPLATE = '''\
"""Custom tools for your agent."""

from adk import tool


@tool
def search_docs(query: str) -> str:
    """Search project documentation."""
    # Replace with your actual implementation
    return f"Found docs matching: {{query}}"


@tool
def get_status() -> str:
    """Get current project status."""
    return "All systems operational."
'''


def cmd_init(args):
    """Scaffold a new agent project directory."""
    name = args.name or "my-agent"
    target = Path(args.directory or name)

    if target.exists() and any(target.iterdir()):
        print(f"Error: {target} already exists and is not empty.")
        return 1

    target.mkdir(parents=True, exist_ok=True)

    (target / "agent.py").write_text(
        _AGENT_TEMPLATE.format(name=name), encoding="utf-8"
    )
    (target / "config.yaml").write_text(
        _CONFIG_TEMPLATE.format(name=name), encoding="utf-8"
    )
    (target / "tools.py").write_text(
        _TOOLS_TEMPLATE, encoding="utf-8"
    )

    print(f"Created AitherADK project at {target}/")
    print("  agent.py   — Your agent definition")
    print("  config.yaml — Configuration")
    print("  tools.py   — Custom tools")
    print()
    print("Next steps:")
    print(f"  cd {target}")
    print("  adk run              # Start the server")
    print("  python agent.py      # Run directly")
    print()
    print("Using an AI agent? Run `adk agent-prompt` for a copy-paste setup guide.")

    # OpenClaw detection — prompt integration if detected
    openclaw_dir = Path.home() / ".openclaw"
    if openclaw_dir.exists():
        oc_config = {}
        oc_config_path = openclaw_dir / "openclaw.json"
        if oc_config_path.exists():
            try:
                import json
                oc_config = json.loads(oc_config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

        aither_integrated = any(
            "aither" in k.lower()
            for k in oc_config.get("mcpServers", {})
        )
        if not aither_integrated:
            print()
            print("  OpenClaw detected! Connect it to AitherOS agents:")
            print("  aither integrate openclaw")

    return 0


def cmd_new(args):
    """Scaffold a complete project from a bundled template — e.g. `adk new deep-research`.

    Templates live in adk/templates/<name>/ and are copied verbatim. Unlike `adk init`
    (a minimal stub), these are full, runnable apps (a pack + a server + a web UI).
    """
    import shutil

    templates_dir = Path(__file__).resolve().parent / "templates"
    src = templates_dir / args.template
    if not src.is_dir() or not any(src.iterdir()):
        available = sorted(
            p.name for p in templates_dir.iterdir()
            if p.is_dir() and (p / "serve.py").exists()
        ) if templates_dir.is_dir() else []
        print(f"Unknown template '{args.template}'.")
        if available:
            print(f"Available templates: {', '.join(available)}")
        return 1

    target = Path(args.directory or args.template)
    if target.exists() and any(target.iterdir()):
        print(f"Error: {target} already exists and is not empty.")
        return 1

    shutil.copytree(src, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Created '{args.template}' at {target}/")
    print()
    print("Next steps:")
    print(f"  cd {target}")
    print("  pip install -r requirements.txt")
    print("  python serve.py        # opens a browser; paste your LLM key in the UI")
    return 0


def cmd_create_app(args):
    """Scaffold a full portal-kit workspace app using WorkspaceRuntime template."""
    import subprocess as _sp

    slug = args.subdomain or re.sub(r"[^a-z0-9-]", "-", args.name.lower().strip()).strip("-")[:30]
    output = args.output or f"./{slug}"

    # Locate scaffold.py — check several known paths
    scaffold_candidates = [
        # When installed inside a dev workspace with aitheros monorepo
        Path("/workspace/aitheros/AitherOS/apps/WorkspaceRuntime/scaffold.py"),
        # Relative to repo root on local dev
        Path(__file__).resolve().parents[2] / "AitherOS" / "apps" / "WorkspaceRuntime" / "scaffold.py",
        # Sibling directory (standalone workspace)
        Path.cwd() / "WorkspaceRuntime" / "scaffold.py",
    ]

    scaffold_path = None
    for candidate in scaffold_candidates:
        if candidate.exists():
            scaffold_path = candidate
            break

    if not scaffold_path:
        # Fallback: download scaffold.py from GitHub
        print("WorkspaceRuntime scaffold not found locally — downloading...")
        import urllib.request
        import tempfile
        dl_url = "https://raw.githubusercontent.com/Aitherium/AitherOS/develop/AitherOS/apps/WorkspaceRuntime/scaffold.py"
        try:
            tmp = Path(tempfile.mkdtemp()) / "scaffold.py"
            urllib.request.urlretrieve(dl_url, str(tmp))
            scaffold_path = tmp
            print(f"  Downloaded to {tmp}")
        except Exception as e:
            print(f"Error: Could not download scaffold: {e}")
            print()
            print("Expected locations:")
            for c in scaffold_candidates:
                print(f"  {c}")
            print()
            print("If you're in a dev workspace, make sure 'aitheros' is in your repos.")
            return 1

    # Build scaffold.py arguments
    cmd = [
        sys.executable, str(scaffold_path),
        "--name", args.name,
        "--output", output,
        "--subdomain", slug,
        "--llm-provider", args.llm_provider,
    ]
    if args.company:
        cmd += ["--company", args.company]
    if args.industry:
        cmd += ["--industry", args.industry]
    if args.description:
        cmd += ["--description", args.description]
    if args.color:
        cmd += ["--color", args.color]
    if args.force:
        cmd.append("--force")

    print(f"Scaffolding '{args.name}' -> {output}")
    print()
    result = _sp.run(cmd)
    if result.returncode == 0:
        print()
        print("Next: connect to AitherOS backend for inference:")
        print(f"  cd {output}")
        print("  .DEPLOYMENT/scripts/compose.sh aitheros -f docker-compose.yml up -d")
        print()
        print("Or standalone (local LLM):")
        print(f"  cd {output}")
        print("  docker compose up -d")
    return result.returncode


def cmd_workspace(args):
    """Manage dev workspaces on AitherOS tunnel."""
    import json as _json
    import urllib.request
    import urllib.error

    ws_cmd = getattr(args, "ws_command", None)
    if not ws_cmd:
        print("Usage: adk workspace [create|bundle|list|submit|scopes]")
        return 1

    # Load auth token
    cfg = load_saved_config()
    token = cfg.get("api_key") or cfg.get("access_token") or os.environ.get("AITHER_API_KEY", "")
    if not token and ws_cmd != "scopes":
        print("Error: Not authenticated. Run: adk login")
        return 1

    tunnel_url = getattr(args, "tunnel_url", "https://tunnel.aitherium.com")

    def _api(method, path, body=None):
        """Make authenticated request to tunnel API."""
        url = f"{tunnel_url}{path}"
        data = _json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                return _json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            try:
                err_body = _json.loads(e.read())
            except (ValueError, OSError):
                err_body = {"detail": str(e)}
            return err_body, e.code

    if ws_cmd == "scopes":
        scopes = {
            "fullstack": "Full monorepo + AitherZero (admin/developer)",
            "frontend": "AitherVeil + packages",
            "backend": "lib + services + config + tests",
            "gargbot": ".PRODUCTS/.GARGBOT + portal-kit",
            "chelle": ".PRODUCTS/.CHELLE + portal-kit",
            "veil": "AitherVeil + all packages",
            "portal": "AitherVeil + portal-kit + desktop-core",
            "node": "AitherNode (standalone + monorepo)",
            "connect": "AitherConnect",
            "shell": "AitherShell (standalone + monorepo)",
            "adk": "aither-adk (this package)",
            "desktop": "AitherDesktop + Veil + packages",
            "creative": "Canvas-Studio + creative services",
            "gpu": "VRAM-Sentinel + GPU services",
            "portal-kit-dev": "portal-kit + GargBot + Chelle (indie devs)",
        }
        print("Available workspace scopes:")
        print()
        for name, desc in scopes.items():
            print(f"  {name:20s} {desc}")
        print()
        print("Usage: adk workspace create --scope gargbot")
        return 0

    if ws_cmd == "create":
        scope = getattr(args, "scope", "fullstack")
        print(f"Creating cloud workspace (scope: {scope})...")
        data, status = _api("POST", "/tunnel/developer/workspace", {
            "scope_template": scope,
        })
        if status >= 400:
            print(f"Error ({status}): {data.get('detail', data.get('error', 'Unknown'))}")
            return 1
        print(f"  Container: {data.get('container_name', 'unknown')}")
        print(f"  Terminal:  {data.get('terminal_url', 'N/A')}")
        print(f"  VS Code:   {data.get('code_server_url', 'N/A')}")
        print(f"  Branch:    {data.get('branch', 'develop')}")
        print(f"  Scope:     {data.get('scope_template', scope)}")
        print()
        print("Connect via SSH:")
        print(f"  ssh dev@tunnel.aitherium.com -p {data.get('ssh_port', '22')}")
        print()
        print("Or open the terminal in browser:")
        print(f"  {data.get('terminal_url', tunnel_url)}")
        return 0

    if ws_cmd == "bundle":
        scope = getattr(args, "scope", "fullstack")
        output = getattr(args, "output", "aitheros-devws.zip")
        print(f"Downloading workspace bundle (scope: {scope})...")
        url = f"{tunnel_url}/tunnel/developer/workspace/bundle?scope={scope}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                with open(output, "wb") as f:
                    f.write(resp.read())
            print(f"  Saved: {output}")
            print()
            print("Next steps:")
            print(f"  unzip {output}")
            print("  cd aitheros-devws-*/")
            print("  docker compose up -d")
            return 0
        except urllib.error.HTTPError as e:
            print(f"Error ({e.code}): {e.reason}")
            return 1

    if ws_cmd == "list":
        data, status = _api("GET", "/tunnel/developer/workspaces")
        if status >= 400:
            print(f"Error ({status}): {data.get('detail', 'Unknown')}")
            return 1
        workspaces = data.get("workspaces", [])
        if not workspaces:
            print("No active workspaces.")
            return 0
        print(f"Active workspaces ({len(workspaces)}):")
        for ws in workspaces:
            name = ws.get("container_name", ws.get("name", "?"))
            scope = ws.get("scope_template", "?")
            status_str = ws.get("status", "?")
            print(f"  {name:40s} scope={scope:15s} status={status_str}")
        return 0

    if ws_cmd == "submit":
        message = getattr(args, "message", "")
        workspace = getattr(args, "workspace", "")
        if not workspace:
            # Auto-detect: check if we're inside a dev workspace container
            workspace = os.environ.get("DEV_SESSION_ID", "")
            if not workspace:
                # Try container hostname
                import socket
                hostname = socket.gethostname()
                if hostname.startswith("aitheros-devws-"):
                    workspace = hostname
        if not workspace:
            print("Error: --workspace required (or run from inside a dev workspace)")
            return 1
        print(f"Submitting changes from {workspace}...")
        data, status = _api("POST", "/tunnel/developer/workspace/submit-changes", {
            "workspace": workspace,
            "message": message,
        })
        if status >= 400:
            print(f"Error ({status}): {data.get('detail', data.get('error', 'Unknown'))}")
            return 1
        print(f"  Status: {data.get('status', '?')}")
        if data.get("pr_url"):
            print(f"  PR: {data['pr_url']}")
        for step in data.get("steps", []):
            icon = "+" if step["status"] == "success" else "x" if step["status"] == "failed" else "."
            print(f"  [{icon}] {step['name']}: {step.get('detail', '')}")
        return 0

    print(f"Unknown workspace command: {ws_cmd}")
    return 1


def cmd_run(args):
    """Start the agent server."""
    from adk.server import main as server_main
    # Re-inject args into sys.argv for server's argparse
    sys_args = ["aither-serve"]
    if args.identity:
        sys_args += ["--identity", args.identity]
    if args.port:
        sys_args += ["--port", str(args.port)]
    if args.host:
        sys_args += ["--host", args.host]
    if args.backend:
        sys_args += ["--backend", args.backend]
    if args.model:
        sys_args += ["--model", args.model]
    if args.fleet:
        sys_args += ["--fleet", args.fleet]
    if args.agents:
        sys_args += ["--agents", args.agents]

    sys.argv = sys_args
    server_main()


def _sync_entitled_packs_quiet() -> None:
    """Run a license-driven pack sync (best-effort) and print a one-line result.

    Shared by `adk up` startup. Resolves the portal + saved token like
    `adk pack sync`; a portal/auth absence is a no-op, never a hard error.
    """
    from adk.shell.plugins.builtins.packs import PacksPlugin

    plugin = PacksPlugin()
    portal = (
        os.environ.get("AITHER_PORTAL_URL")
        or os.environ.get("AITHER_ELYSIUM_URL")
        or "https://portal.aitherium.com"
    ).rstrip("/")
    plugin._base_url = portal
    cfg = load_saved_config()
    token = cfg.get("api_key") or cfg.get("access_token") or os.environ.get("AITHER_API_KEY", "")
    if not token:
        return  # not authenticated → nothing to converge
    plugin.auth.set_auth(token, cfg.get("tenant_id"))
    result = plugin._sync([])
    # Print only the summary line to keep startup output tight.
    first_line = (result or "").strip().splitlines()
    if first_line:
        print(f"  packs: {first_line[1] if len(first_line) > 1 else first_line[0]}")


def cmd_stack(args):
    """Start the consumer stack (Room + Ollama) as native processes.

    Supervises AitherRoom (:8350) and Ollama (:11434) as foreground processes
    with health checks, crash restarts, and graceful shutdown on Ctrl+C.

    (Formerly ``adk up``; that name now brings up a connected agent — see cmd_up.
    This behaviour lives under ``adk stack``.)

    Services:
    - default: Room + Ollama (default)
    - qdrant: Local Qdrant for graph memory (requires Docker)
    """
    # Handle "adk stack qdrant" for Qdrant provisioning
    service = getattr(args, "service", "default") or "default"
    if service == "qdrant":
        print()
        print("Provisioning local Qdrant for graph memory...")
        success, url, api_key = _provision_qdrant()
        if success:
            print(f"  [+] Qdrant URL: {url}")
            print(f"  [+] API key saved to ~/.aither/config.yaml")
            print()
            print("  Configuration:")
            print(f"    AITHER_FLEET_QDRANT_URL={url}")
            print(f"    AITHER_FLEET_QDRANT_API_KEY={api_key}")
            print()
            print("  Ready to use in graph_memory. Run 'adk status' to verify.")
            return 0
        # Provisioning failed — a non-zero exit so the caller/CI can detect it
        # (gate finding: never report a failed setup as success).
        print("  [!] Qdrant provisioning failed. Manual setup required.")
        return 1

    from adk.process_supervisor import ProcessSpec, ProcessSupervisor, resolve_room_launch_target

    # Converge entitled packs first so the agent comes up with everything the
    # customer has bought (a portal purchase → `adk up` → packs present). Opt-out
    # with --no-sync; never block startup if the portal is unreachable.
    if not getattr(args, "no_sync", False):
        try:
            _sync_entitled_packs_quiet()
        except Exception as e:  # noqa: BLE001
            print(f"  (pack sync skipped: {e})")

    supervisor = ProcessSupervisor()

    # Gather specs
    specs = []

    # Room service (if launch target resolvable)
    # Resolution order: env var → installed package → repo checkout → binary
    room_target = resolve_room_launch_target()
    if room_target:
        # Determine if target is a binary or a Python module
        target_path = Path(room_target)
        if target_path.exists() and target_path.is_file():
            # Binary: run directly
            room_spec = ProcessSpec(
                name="AitherRoom",
                argv=[str(target_path)],
                port=8350,
                health_url="http://127.0.0.1:8350/health",
                description="Company agent room service (binary)",
            )
        else:
            # Python module: run via uvicorn
            room_spec = ProcessSpec(
                name="AitherRoom",
                argv=[
                    sys.executable,
                    "-m",
                    "uvicorn",
                    room_target,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8350",
                ],
                port=8350,
                health_url="http://127.0.0.1:8350/health",
                description="Company agent room service (Python)",
            )
        specs.append(room_spec)
    else:
        print("Warning: AitherRoom not found — Room service will not start.")
        print("Tried: 1) cached in ~/.aither/bin/, 2) download from GitHub,")
        print("       3) bundled in adk/room_binaries/<platform>/.")
        print(
            "Ensure CI populates bundled binaries or set "
            "AITHER_ROOM_LAUNCH_TARGET to a dev module path."
        )

    # Ollama (if installed)
    if os.path.exists(os.path.expanduser("~/.ollama")):
        ollama_spec = ProcessSpec(
            name="Ollama",
            argv=["ollama", "serve"],
            port=11434,
            health_url="http://127.0.0.1:11434/api/tags",
            description="Local model serving",
        )
        specs.append(ollama_spec)

    if not specs:
        print("Error: No services available to start.")
        print("Ensure Ollama is installed and Room is available.")
        return 1

    # Start supervision
    print()
    print("Starting AitherOS consumer stack...")
    print()
    supervisor.supervise(specs, interval=10)


def cmd_up(args):
    """One command: run a persistent agent connected to your AitherOS fleet.

    Brings up ``aither-serve`` (single agent, identity ``aither`` by default),
    opens a secure Cloudflare quick-tunnel so the fleet can reach it, registers
    with the portal, and installs an autostart entry so it survives reboot.

    Two modes:
    - Interactive (a human at a terminal): minimal prompts, device-flow browser login.
    - Non-interactive (``--yes`` / not a TTY): ZERO prompts — everything from flags,
      env, and saved config; emits a machine-readable JSON status line. This is the
      path an autonomous agent uses to stand itself up unattended.

    By default the agent is DETACHED (the terminal is freed). ``--foreground`` blocks.
    Companions: ``adk status`` (state), ``adk down`` (stop + remove autostart).
    """
    import base64
    import getpass
    import re
    import secrets as _secrets
    import socket

    import httpx

    from adk import agent_daemon as daemon

    non_interactive = bool(getattr(args, "yes", False)) or not sys.stdin.isatty()
    offline = (os.environ.get("AITHER_OFFLINE", "").lower() in ("1", "true", "yes")
               or bool(getattr(args, "offline", False)))
    identity = getattr(args, "identity", None) or "aither"
    name = (getattr(args, "name", "") or "").strip() or re.sub(
        r"[^a-z0-9_-]", "-", f"{socket.gethostname()}-adk".lower())
    port = getattr(args, "port", None) or 8080
    foreground = bool(getattr(args, "foreground", False))
    persist = not getattr(args, "no_persist", False)
    force = bool(getattr(args, "force", False))
    dry_run = bool(getattr(args, "dry_run", False))
    require_register = bool(getattr(args, "require_register", False))
    provider = (getattr(args, "provider", "") or "").strip().lower()
    approval = getattr(args, "approve", None) or "file_write,shell_exec,shell"
    will_register = not getattr(args, "no_register", False) and not offline

    def _fail(code: int, error: str, hint: str = "", next_action: str = "") -> int:
        obj = {"ok": False, "error": error, "error_code": code}
        if hint:
            obj["hint"] = hint
        if next_action:
            obj["next_action"] = next_action
        if non_interactive:
            print(json.dumps(obj))
        else:
            print(f"  [x] {error}")
            if hint:
                print(f"      {hint}")
            if next_action:
                print(f"      -> {next_action}")
        return code

    # ── Idempotency: already running? ──
    existing = daemon.read_status()
    if existing and daemon.pid_alive(existing.get("server_pid")):
        if force:
            daemon.kill_pid(existing.get("tunnel_pid"))
            daemon.kill_pid(existing.get("server_pid"))
            daemon.clear_status()
        else:
            existing["ok"] = True
            existing["already_running"] = True
            if non_interactive:
                print(json.dumps(existing))
            else:
                print(f"  [+] Agent already running on :{existing.get('port')} "
                      f"(pid {existing.get('server_pid')}). "
                      "Use --force to restart, or 'adk down' to stop.")
            return 0

    # ── Portal token (used for BOTH the hosted brain and registration) ──
    portal = (getattr(args, "portal", "") or "https://veil.aitherium.com").rstrip("/")
    saved = load_saved_config()
    portal_token = (getattr(args, "token", "") or os.environ.get("AITHER_PORTAL_TOKEN", "")
                    or saved.get("api_key", "") or saved.get("access_token", ""))

    def _device_login(client: str) -> str:
        login_url = _resolve_identity_url(
            getattr(args, "login_url", "") or os.environ.get("AITHER_PORTAL_URL", "")
            or portal or _DEFAULT_IDENTITY_URL)
        print(f"  Signing in at {login_url} …")
        dev = _device_flow_login(login_url, client_name=client)
        tok = dev.get("access_token", "") or dev.get("token", "")
        if tok:
            try:
                save_saved_config({"api_key": tok})
            except (OSError, ValueError):
                pass
        return tok

    # ── Backend: local model  >  a provider key  >  the hosted Aither brain ──
    # The hosted-brain default means a normal human never needs an API key: if
    # there's no local model and no provider key, we sign in (device-flow) and
    # route inference through the gateway using the portal token.
    try:
        from adk.shell_launcher import _preflight_check
        have_backend, _backend_desc = _preflight_check()
    except (ImportError, RuntimeError, OSError):
        have_backend, _backend_desc = (False, "")

    use_gateway = False
    if not have_backend:
        cand = provider
        if not cand:
            _pk = _load_provider_keys()
            for p in ("deepseek", "openai", "anthropic"):
                if _pk.get(p) or os.environ.get(_KNOWN_PROVIDERS[p]["env"]):
                    cand = p
                    break
        if cand and cand not in _KNOWN_PROVIDERS:
            return _fail(2, f"unknown provider '{cand}'",
                         f"known: {', '.join(sorted(_KNOWN_PROVIDERS))}",
                         next_action="re-run with a supported --provider, or omit it")
        cand_key = ""
        if cand:
            cand_key = (_load_provider_keys().get(cand, "")
                        or os.environ.get(_KNOWN_PROVIDERS[cand]["env"], ""))

        if cand_key:
            provider = cand
            os.environ[_KNOWN_PROVIDERS[provider]["env"]] = cand_key
        elif provider:
            # Explicit --provider but no key for it.
            env_name = _KNOWN_PROVIDERS[provider]["env"]
            if non_interactive:
                return _fail(3, "no_backend",
                             f"provider '{provider}' selected but {env_name} is not set",
                             next_action=f"set {env_name}, or omit --provider to use the hosted brain")
            key = getpass.getpass(f"  {_KNOWN_PROVIDERS[provider]['label']} API key: ").strip()
            if not key:
                return _fail(3, "no_backend", "no key entered",
                             next_action="paste your API key, or omit --provider for the hosted brain")
            ok, msg = _test_provider_key(provider, key)
            print(f"  [{'+' if ok else 'x'}] {msg}")
            if not ok:
                return _fail(3, "invalid_key", "key failed validation (not saved)",
                             next_action="check the key and try again")
            _pk = _load_provider_keys()
            _pk[provider] = key
            _save_provider_keys(_pk)
            os.environ[env_name] = key
        else:
            # No local model + no provider key → default to the hosted Aither brain.
            if not portal_token and not offline:
                if non_interactive:
                    return _fail(3, "no_backend",
                                 "no local model, no API key, and not signed in",
                                 next_action="run 'adk login' (or set AITHER_PORTAL_TOKEN), "
                                             "or 'adk setup' for a local model")
                print("  No local model or API key found — using the hosted Aither brain "
                      "(no key needed).")
                try:
                    portal_token = _device_login(f"adk-up:{name}")
                except Exception as e:  # noqa: BLE001 — surface a clear next step
                    return _fail(3, f"sign-in failed: {e}", "",
                                 next_action="run 'adk login', pass --provider + key, "
                                             "or 'adk setup' for a local model")
            if portal_token:
                use_gateway = True
                provider = "gateway"
            else:
                return _fail(3, "no_backend",
                             "no local model and no hosted brain available",
                             next_action="run 'adk setup' for a local model, "
                                         "or drop --offline to use the hosted brain")

    # ── Portal token for REGISTRATION (reuse the one above, or acquire) ──
    if will_register and not portal_token:
        if non_interactive:
            if require_register:
                return _fail(5, "no_portal_token",
                             "unattended registration needs a token",
                             next_action="set AITHER_PORTAL_TOKEN or pass --token")
            will_register = False
        else:
            try:
                portal_token = _device_login(f"adk-up:{name}")
            except Exception as e:  # noqa: BLE001 — surface a clear next step
                return _fail(5, f"device-flow login failed: {e}", "",
                             next_action="pass --token, run 'adk login', or use --no-register")
            if not portal_token:
                will_register = False

    # ── cloudflared availability (auto-download if missing) ──
    cf = None
    if will_register:
        cf = _find_cloudflared()
        if not cf:
            if not non_interactive:
                print("  cloudflared not found — downloading it (one-time, ~50MB) …")
            cf = _ensure_cloudflared()
        if not cf:
            return _fail(4, "cloudflared_missing", _cloudflared_install_hint().strip(),
                         next_action="install cloudflared manually, or re-run with --no-register")

    # ── Callback bearer (the one secret the control plane presents back to us) ──
    auth_token = getattr(args, "auth_token", "") or os.environ.get("AITHER_SERVER_API_KEY", "")
    if not auth_token:
        auth_token = base64.urlsafe_b64encode(_secrets.token_bytes(24)).decode().rstrip("=")

    if dry_run:
        plan = {"ok": True, "dry_run": True, "identity": identity, "name": name,
                "port": port, "will_register": will_register, "offline": offline,
                "detached": not foreground, "persist": persist,
                "backend": "local" if have_backend else provider}
        print(json.dumps(plan) if non_interactive else f"  [dry-run] {json.dumps(plan)}")
        return 0

    # ── Launch aither-serve (detached, logs to file) ──
    daemon._ensure_dirs()
    agent_log = daemon.LOG_DIR / "agent.log"
    tunnel_log = daemon.LOG_DIR / "tunnel.log"
    child_env = dict(os.environ)
    child_env["AITHER_SERVER_API_KEY"] = auth_token
    child_env["AITHER_TOOL_APPROVAL"] = approval or ""
    if offline:
        child_env["AITHER_OFFLINE"] = "1"
    if use_gateway and portal_token:
        # Route inference through the hosted Aither brain with the portal token —
        # the router honours AITHER_LLM_BACKEND=gateway + AITHER_API_KEY (bearer).
        child_env["AITHER_API_KEY"] = portal_token
        child_env["AITHER_LLM_BACKEND"] = "gateway"
        _eps = _derive_cloud_endpoints(_resolve_identity_url(
            getattr(args, "login_url", "") or os.environ.get("AITHER_PORTAL_URL", "")
            or portal or _DEFAULT_IDENTITY_URL))
        if _eps and _eps.get("inference_url"):
            child_env["AITHER_INFERENCE_URL"] = _eps["inference_url"]
    serve_argv = [sys.executable, "-m", "adk.server",
                  "--port", str(port), "--identity", identity]
    if provider and provider != "gateway":
        serve_argv += ["--backend", provider]
    if not non_interactive:
        _brain = "hosted Aither brain" if use_gateway else (provider or "local")
        print(f"  Starting agent ({_brain}, :{port}) …")
    server_pid = daemon.spawn_detached(serve_argv, agent_log, env=child_env)

    if not daemon.wait_for_health(port, timeout=60):
        daemon.kill_pid(server_pid)
        return _fail(1, "agent did not become healthy in 60s",
                     f"see {agent_log}")
    if not non_interactive:
        print(f"  [+] Agent healthy on :{port}")

    # ── Tunnel + register ──
    public_url = None
    tunnel_pid = None
    registered = False
    if will_register and cf:
        tunnel_pid = daemon.spawn_detached(
            [cf, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            tunnel_log)
        public_url = daemon.tail_for_pattern(
            tunnel_log, r"https://[a-z0-9-]+\.trycloudflare\.com", timeout=45)
        if not public_url:
            daemon.kill_pid(tunnel_pid)
            tunnel_pid = None
            if require_register:
                daemon.kill_pid(server_pid)
                return _fail(4, "tunnel_failed", f"no trycloudflare URL; see {tunnel_log}")
            if not non_interactive:
                print("  [!] Tunnel did not come up — running local-only (will retry via heartbeat).")
        else:
            if not non_interactive:
                print(f"  [+] Tunnel live: {public_url}")
            reg = (getattr(args, "register_url", "")
                   or f"{portal}/api/genesis/v1/agent/fleet/register")
            body = {"name": name, "invoke_url": public_url, "reach": "tunnel",
                    "agent_type": "adk-agent", "token": auth_token,
                    "model": getattr(args, "model", None) or "", "provider_hint": provider}
            hdrs = {"Content-Type": "application/json"}
            if portal_token:
                hdrs["Authorization"] = f"Bearer {portal_token}"
            try:
                rr = httpx.post(reg, json=body, headers=hdrs, timeout=20)
                registered = rr.status_code < 300
                if not registered:
                    if require_register:
                        daemon.kill_pid(tunnel_pid)
                        daemon.kill_pid(server_pid)
                        return _fail(5, f"registration rejected ({rr.status_code})",
                                     rr.text[:200])
                    if not non_interactive:
                        print(f"  [!] Registration returned {rr.status_code} — "
                              "agent is up locally; heartbeat will retry.")
            except (httpx.HTTPError, OSError) as e:
                if require_register:
                    daemon.kill_pid(tunnel_pid)
                    daemon.kill_pid(server_pid)
                    return _fail(5, f"registration failed: {e}")
                if not non_interactive:
                    print(f"  [!] Registration failed ({e}) — heartbeat will retry.")

    # ── Persist (autostart) ──
    autostart = None
    if persist:
        up_argv = _autostart_up_argv(identity, port, provider, offline)
        autostart = daemon.install_autostart(up_argv)

    # ── Status file (single source of truth for status/down) ──
    # The agent serves its own streaming chat page at "/" — that's where a human
    # actually talks to it (live tokens, no 30s blank wait), reachable locally.
    chat_url = f"http://127.0.0.1:{port}/"
    status = {
        "ok": True, "identity": identity, "name": name, "port": port,
        "server_pid": server_pid, "tunnel_pid": tunnel_pid,
        "invoke_url": public_url, "registered": registered,
        "offline": offline, "detached": not foreground,
        "backend": "local" if have_backend else provider,
        "autostart": autostart, "portal": portal, "chat_url": chat_url,
        "log_path": str(agent_log),
    }
    daemon.write_status(status)

    # ── Report ──
    if non_interactive:
        print(json.dumps({**status, "health": "healthy"}))
    else:
        print()
        print("  [OK] Your agent is running.")
        if public_url:
            print(f"     Fleet: {portal}/portal/fleet")
            print(f"     URL:   {public_url}")
        print(f"     Chat:  {chat_url}")
        print(f"     Logs:  {agent_log}")
        print("     Stop:  adk down     Status: adk status")
        # Open the chat page so a non-technical user has somewhere to talk to it.
        # Interactive + detached only (never in --yes or blocking --foreground).
        if not foreground:
            try:
                import webbrowser
                # Pass the callback bearer in the URL fragment (#k=…) — the browser
                # never sends it to the server or logs, so the chat page can call the
                # gated /chat/stream without the user pasting a token.
                webbrowser.open(f"{chat_url}#k={auth_token}")
            except Exception:  # noqa: BLE001 — opening a browser is best-effort
                pass

    if foreground:
        if not non_interactive:
            print("\n  Foreground mode — Ctrl+C to stop.")
        try:
            while daemon.pid_alive(server_pid):
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n  Stopping …")
            daemon.kill_pid(tunnel_pid)
            daemon.kill_pid(server_pid)
            daemon.clear_status()
    return 0


def _autostart_up_argv(identity: str, port: int, provider: str, offline: bool) -> list[str]:
    """Build the command the autostart entry re-runs at logon (unattended, detached)."""
    argv = [sys.executable, "-m", "adk.cli", "up",
            "--identity", identity, "--port", str(port), "--yes"]
    # 'gateway' is the hosted-brain sentinel, not a real --provider — at reboot the
    # saved portal token is re-resolved and the hosted brain is chosen again.
    if provider and provider != "gateway":
        argv += ["--provider", provider]
    if offline:
        argv += ["--offline"]
    return argv


def cmd_down(args):
    """Stop the running agent + tunnel and remove its autostart entry."""
    from adk import agent_daemon as daemon

    import httpx

    st = daemon.read_status()
    if not st:
        print("  No agent is running (no status file).")
        return 0

    if daemon.kill_pid(st.get("tunnel_pid")):
        print("  [+] Tunnel stopped.")
    if daemon.kill_pid(st.get("server_pid")):
        print("  [+] Agent stopped.")

    # Best-effort deregister from the fleet.
    if st.get("registered") and st.get("name"):
        saved = load_saved_config()
        token = saved.get("api_key", "") or os.environ.get("AITHER_PORTAL_TOKEN", "")
        portal = (st.get("portal") or "https://veil.aitherium.com").rstrip("/")
        try:
            hdrs = {"Authorization": f"Bearer {token}"} if token else {}
            httpx.request("DELETE",
                          f"{portal}/api/genesis/v1/agent/agent-endpoints/{st['name']}",
                          headers=hdrs, timeout=10)
            print("  [+] Deregistered from the fleet.")
        except (httpx.HTTPError, OSError):
            pass

    if not getattr(args, "keep_autostart", False):
        if daemon.remove_autostart():
            print("  [+] Autostart removed.")

    daemon.clear_status()
    return 0


def _provision_qdrant() -> tuple[bool, str, str]:
    """Provision local Qdrant for adk stack.

    Generates/loads API key from config, starts container if needed.
    Returns (success, url, api_key).
    """
    import secrets
    import subprocess

    saved = load_saved_config()
    api_key = saved.get("qdrant_api_key", "").strip()

    # Generate if missing
    if not api_key:
        api_key = secrets.token_urlsafe(32)
        save_saved_config({"qdrant_api_key": api_key})
        print(f"  [+] Generated Qdrant API key")

    # Start the container, VERIFYING the start actually succeeded — a failed
    # `docker start` must not be reported as success (gate finding).
    try:
        subprocess.run(
            ["docker", "inspect", "aitheros-workspace-qdrant"],
            capture_output=True, timeout=5, check=False
        )
        start = subprocess.run(
            ["docker", "start", "aitheros-workspace-qdrant"],
            capture_output=True, timeout=5, check=False
        )
        if start.returncode != 0:
            print(f"  [!] Failed to start Qdrant: {start.stderr.decode(errors='replace').strip()}")
            print(f"      AITHER_FLEET_QDRANT_URL=http://localhost:6333")
            print(f"      AITHER_FLEET_QDRANT_API_KEY={api_key}")
            return False, "", api_key
        print("  [+] Qdrant container is running")
    except (OSError, subprocess.TimeoutExpired):
        # Docker not available; just return config for manual setup
        print("  [!] Docker not available; set env manually:")
        print(f"      AITHER_FLEET_QDRANT_URL=http://localhost:6333")
        print(f"      AITHER_FLEET_QDRANT_API_KEY={api_key}")
        return False, "", api_key

    # Set env for this process + child processes — but a customer's explicit
    # AITHER_FLEET_QDRANT_URL/API_KEY takes precedence (gate finding: don't
    # clobber a pre-existing setup).
    url = "http://localhost:6333"
    if not os.environ.get("AITHER_FLEET_QDRANT_URL"):
        os.environ["AITHER_FLEET_QDRANT_URL"] = url
    if not os.environ.get("AITHER_FLEET_QDRANT_API_KEY"):
        os.environ["AITHER_FLEET_QDRANT_API_KEY"] = api_key
    save_saved_config({
        "qdrant_url": url,
        "qdrant_api_key": api_key,
    })
    return True, url, api_key


def cmd_register(args):
    """Register a new Aitherium account."""
    import asyncio
    import getpass

    async def _register():
        from adk.elysium import Elysium

        email = args.email
        password = args.password

        # Interactive prompts when flags are omitted
        if not email:
            email = input("  Email: ").strip()
        if not password:
            password = getpass.getpass("  Password: ")

        if not email or not password:
            print("  Error: email and password are required.")
            return 1

        print()
        print(f"  Registering {email}...")

        ely = Elysium()
        try:
            result = await ely.register(email, password)
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            print(f"  Error: {exc}")
            return 1

        user_id = result.get("user_id", "")
        api_key = result.get("api_key", "")

        if api_key:
            save_saved_config({"api_key": api_key, "email": email})
            print("  API key saved to ~/.aither/config.json")

        print()
        print(f"  Account created (user_id: {user_id}).")
        print("  Check your email to verify, then run: aither connect")
        return 0

    return asyncio.run(_register())


# ---------------------------------------------------------------------------
# adk login / whoami / logout — device flow auth (RFC 8628)
# ---------------------------------------------------------------------------

_DEFAULT_IDENTITY_URL = "https://portal.aitherium.com"


def _derive_cloud_endpoints(identity_url: str) -> dict | None:
    """Map the identity/portal URL a user logged into → their workspace's
    data-plane endpoints (inference + MCP + identity).

    The Aitherium cloud serves both OpenAI-compatible inference (``/v1/*``)
    and the MCP protocol (``/mcp``) from ``mcp.aitherium.com`` — see
    AitherOS/config/tunnel-routes.yaml. We only auto-derive for the known
    aitherium.com topology; localhost and unknown hosts return None so we
    never clobber a local-dev or bespoke sovereign setup with a guess.
    """
    from urllib.parse import urlparse

    host = (urlparse(identity_url).hostname or "").lower()
    if not host or host in ("localhost", "127.0.0.1", "::1"):
        return None
    if host == "aitherium.com" or host.endswith(".aitherium.com"):
        return {
            "api_url": "https://mcp.aitherium.com",
            "mcp_url": "https://mcp.aitherium.com/mcp",
            "inference_url": "https://mcp.aitherium.com/v1",
            "identity_url": "https://idp.aitherium.com",
        }
    return None


def _persist_workspace_endpoints(identity_url: str) -> dict | None:
    """After login, write the workspace endpoints to ~/.aither/config.json AND
    ~/.aither/shell.yaml so the `aither` REPL connects to the cloud with no
    manual env vars. Returns the endpoints, or None if nothing was derived.
    """
    eps = _derive_cloud_endpoints(identity_url)
    if not eps:
        return None

    # 1) config.json — Python SDK + `adk whoami`.
    save_saved_config(eps)

    # 2) shell.yaml — the TS `aither` shell reads api_url/mcp_url/identity_url.
    try:
        cfg_dir = Path.home() / ".aither"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        shell_yaml = cfg_dir / "shell.yaml"
        existing: dict[str, str] = {}
        if shell_yaml.exists():
            for line in shell_yaml.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                existing[key.strip()] = value.strip()
        existing.update({
            "api_url": eps["api_url"],
            "mcp_url": eps["mcp_url"],
            "identity_url": eps["identity_url"],
        })
        shell_yaml.write_text(
            "\n".join(f"{k}: {v}" for k, v in existing.items()) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass  # config.json still carries the endpoints

    return eps


def _persist_shell_auth(identity_url: str, eps: dict | None, result: dict) -> bool:
    """Mirror the login into ~/.aither/auth.json so the `aither` (npm) shell
    picks up the token + cloud endpoint without a second login.

    `adk login` historically wrote only ~/.aither/config.json, but the
    TypeScript shell reads its token from auth.json — so one login left the
    REPL unauthenticated. Best-effort: never fails the login.
    """
    try:
        from adk.shell.auth import AuthStore, build_profile_from_response
    except Exception:
        return False
    genesis_url = (eps or {}).get("api_url") or identity_url
    try:
        profile = build_profile_from_response(identity_url, genesis_url, result)
        AuthStore.set_profile("cloud" if eps else "default", profile)
        return True
    except Exception:
        return False


def _resolve_identity_url(url: str) -> str:
    """Resolve the AitherIdentity API host from a portal/identity URL.

    The device-flow API (`/auth/device/code|token`) lives on AitherIdentity
    (prod: idp.aitherium.com), NOT the Veil/portal frontend (portal/veil
    .aitherium.com) — those 307-redirect API auth calls to `/login`. For the
    known aitherium.com topology, map to the idp host; otherwise (localhost /
    bespoke sovereign) use the URL as-is.
    """
    eps = _derive_cloud_endpoints(url)
    return (eps["identity_url"] if eps else url).rstrip("/")


def _post_json_resilient(
    url: str,
    payload: dict,
    headers: dict,
    *,
    attempt_timeout: int = 15,
    total_budget: float = 90.0,
) -> dict:
    """POST JSON with retry/backoff over TRANSIENT failures, returning parsed JSON.

    Why this exists: the first request to a just-restarted security-core (or
    through a cold Cloudflare tunnel) can be slow or briefly return 5xx
    (502/503/504/520-530) while it warms its vault/DB connections. A single-shot
    request dies in that window -- the #1 cause of "Cannot reach idp.aitherium.com"
    on `adk login`. We retry transient failures with exponential backoff up to
    `total_budget` seconds so login sails through the cold-start window. Genuine
    client errors (4xx other than 408/429) fail fast and are NOT masked.
    """
    import json as _json
    import time as _time
    import urllib.request as _ur
    import urllib.error as _ue

    _TRANSIENT = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530}  # noqa: N806 — deliberate constant-style set of transient HTTP codes
    body = _json.dumps(payload).encode()
    deadline = _time.time() + total_budget
    delay = 2.0
    attempt = 0
    printed = False
    last_exc = None
    while True:
        attempt += 1
        req = _ur.Request(url, data=body, headers=headers, method="POST")
        try:
            with _ur.urlopen(req, timeout=attempt_timeout) as resp:
                result = _json.loads(resp.read())
            if printed:
                print(" ok")
            return result
        except _ue.HTTPError as exc:
            if exc.code not in _TRANSIENT:
                if printed:
                    print()
                raise  # genuine error -- surface it, don't retry or mask
            last_exc = exc
        except (_ue.URLError, OSError) as exc:
            last_exc = exc
        remaining = deadline - _time.time()
        if remaining <= 0:
            if printed:
                print()
            raise RuntimeError(
                f"Cannot reach {url} after {attempt} attempts over "
                f"{int(total_budget)}s (last error: {last_exc}). The service may "
                f"be starting up -- wait a few seconds and run `adk login` again."
            )
        if not printed:
            print("  Service waking up (cold start) -- retrying", end="", flush=True)
            printed = True
        print(".", end="", flush=True)
        _time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 10.0)


def _device_flow_login(identity_url: str, client_name: str = "adk") -> dict:
    """Run RFC 8628 device code flow. Returns token response dict or raises."""
    import json as _json
    import time
    import urllib.request
    import urllib.error
    import webbrowser

    # A real User-Agent: idp.* sits behind Cloudflare, which 403s urllib's default
    # "Python-urllib/3.x" UA as a bot. Without this, device flow fails before it starts.
    try:
        from adk import __version__ as _v
    except Exception:  # noqa: BLE001
        _v = "0"
    _ua = f"aither-adk/{_v} ({client_name})"

    # Step 1: Request device code. Resilient: the first call after a fleet
    # restart can be slow or briefly 5xx while security-core warms its vault/DB,
    # so retry transient failures with backoff instead of dying on attempt 1.
    try:
        data = _post_json_resilient(
            f"{identity_url}/auth/device/code",
            {"client_name": client_name, "scopes": "full"},
            {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": _ua},
        )
    except RuntimeError:
        raise  # already a clear, budget-exhausted message
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Cannot reach {identity_url}: {exc}") from exc

    user_code = data["user_code"]
    device_code = data["device_code"]
    verification_uri = data.get("verification_uri_complete") or data.get("verification_uri", "")
    interval = max(2, int(data.get("interval", 5)))
    expires_in = int(data.get("expires_in", 900))

    # Step 2: Show code + open browser
    print()
    print(f"  Your code: {user_code}")
    print()
    print(f"  Opening browser to: {verification_uri}")
    print("  (If it doesn't open, visit the URL manually and enter the code)")
    print()

    try:
        webbrowser.open(verification_uri)
    except OSError:
        pass  # Browser open is best-effort

    print("  Waiting for approval", end="", flush=True)

    # Step 3: Poll for token
    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)
        poll_data = _json.dumps({"device_code": device_code}).encode()
        poll_req = urllib.request.Request(
            f"{identity_url}/auth/device/token",
            data=poll_data,
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": _ua},
            method="POST",
        )
        try:
            with urllib.request.urlopen(poll_req, timeout=10) as resp:
                result = _json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                try:
                    err_body = _json.loads(exc.read())
                    detail = err_body.get("detail", "")
                except (ValueError, OSError):
                    detail = ""
                if detail == "expired_token":
                    print()
                    raise RuntimeError("Device code expired. Run `adk login` again.")
                if detail == "invalid_device_code":
                    print()
                    raise RuntimeError("Invalid device code. Run `adk login` again.")
            print(".", end="", flush=True)
            continue
        except (urllib.error.URLError, OSError):
            print(".", end="", flush=True)
            continue

        status = result.get("status", "")
        if status == "authorization_pending":
            print(".", end="", flush=True)
            continue
        if result.get("access_token"):
            print(" approved!")
            return result
        # Unknown status — keep polling
        print(".", end="", flush=True)

    print()
    raise RuntimeError("Timed out waiting for approval. Run `adk login` again.")


def _save_account_license(result: dict) -> str:
    """Persist the account's signed license from a login result to
    ``~/.aither/license.json`` so LicenseManager gates packs on the user's real
    AitherIdentity/ACTA entitlements. Returns the tier (or "" if none/failed).

    The ``license_key`` from AitherIdentity is the base64 outer envelope
    (base64(json({payload, signature}))); LicenseManager's file path expects the
    decoded {payload, signature} envelope. Fail-soft: a bad/absent key just leaves
    the runtime on its prior tier (COMMUNITY) — never blocks login.
    """
    import base64 as _b64
    import json as _json
    from pathlib import Path as _Path

    license_key = (result.get("license_key") or "").strip()
    if not license_key:
        return ""
    try:
        envelope = _json.loads(_b64.b64decode(license_key).decode("utf-8"))
        if not (isinstance(envelope, dict) and "payload" in envelope and "signature" in envelope):
            return ""
        lic_path = _Path.home() / ".aither" / "license.json"
        lic_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = lic_path.with_suffix(".json.tmp")
        tmp.write_text(_json.dumps(envelope), encoding="utf-8")
        tmp.replace(lic_path)
        return str(result.get("tier") or "")
    except Exception as exc:  # never break login over license persistence
        import logging as _logging
        _logging.getLogger("adk.cli").debug("license persistence skipped: %s", exc)
        return ""


def cmd_login(args) -> int:
    """Authenticate with Aitherium — device flow, email/password, or API key."""
    import json as _json

    identity_url = (args.portal_url or os.getenv("AITHER_PORTAL_URL", _DEFAULT_IDENTITY_URL)).rstrip("/")

    # Path 1: Direct API key
    api_key = getattr(args, "api_key", None)
    if api_key:
        save_saved_config({"api_key": api_key})
        print("  API key saved to ~/.aither/config.json")
        eps = _persist_workspace_endpoints(identity_url)
        _persist_shell_auth(identity_url, eps, {"access_token": api_key, "token_type": "api_key", "user": {}})
        if eps:
            print(f"  Workspace endpoints configured → {eps['api_url']}")
        return 0

    # Path 2: Email/password
    email = getattr(args, "email", None)
    if email:
        import asyncio
        import getpass
        password = getattr(args, "password", None) or getpass.getpass("  Password: ")
        if not password:
            print("  Error: password required.")
            return 1

        async def _login():
            from adk.elysium import Elysium
            ely = Elysium(gateway_url=identity_url)
            try:
                result = await ely.login(email, password)
            except (ConnectionError, OSError, RuntimeError) as exc:
                print(f"  Error: {exc}")
                return 1
            token = result.get("token", "")
            if token:
                save_saved_config({"api_key": token, "email": email})
                print(f"  Logged in as {email}")
                print("  Token saved to ~/.aither/config.json")
                eps = _persist_workspace_endpoints(identity_url)
                _persist_shell_auth(identity_url, eps, result)
                if eps:
                    print(f"  Workspace endpoints configured → {eps['api_url']}")
                return 0
            print(f"  Login failed: {result}")
            return 1

        return asyncio.run(_login())

    # Path 3: Device flow (default — opens browser)
    print()
    print("  AitherOS Login")
    print("  ==============")
    try:
        # Device flow must hit AitherIdentity (idp.*), not the portal frontend.
        result = _device_flow_login(_resolve_identity_url(identity_url))
    except RuntimeError as exc:
        print(f"  Error: {exc}")
        return 1

    token = result.get("access_token", "")
    user = result.get("user", {})
    if not token:
        print("  Error: no token in response.")
        return 1

    # Save token
    config_update = {"api_key": token}
    if isinstance(user, dict):
        if user.get("tenant_id"):
            config_update["tenant_id"] = user["tenant_id"]
        if user.get("username"):
            config_update["username"] = user["username"]
    save_saved_config(config_update)

    # Persist the account's Ed25519 license so pack entitlements (can_use_formbridge,
    # can_use_untether, can_use_computer_use, ...) flow from the USER'S AitherIdentity
    # account into this runtime. AitherIdentity's device-flow already fetches it from
    # ACTA (/v1/billing/license) and returns it here — previously it was dropped, so
    # LicenseManager fell back to COMMUNITY (free) and every premium pack gated closed.
    # LicenseManager reads ~/.aither/license.json as the {payload, signature} envelope.
    license_tier = _save_account_license(result)

    eps = _persist_workspace_endpoints(identity_url)
    _persist_shell_auth(identity_url, eps, result)

    username = user.get("username", "") if isinstance(user, dict) else ""
    print()
    print(f"  Logged in{' as ' + username if username else ''}!")
    print("  Token saved to ~/.aither/config.json")
    if license_tier:
        print(f"  License saved → tier '{license_tier}' (pack entitlements active)")
    if eps:
        print(f"  Workspace endpoints configured → {eps['api_url']} (inference + MCP)")
    print()
    return 0


def cmd_whoami(args) -> int:
    """Show current auth status."""
    saved = load_saved_config()
    api_key = saved.get("api_key", "")
    username = saved.get("username", "")
    email = saved.get("email", "")
    tenant_id = saved.get("tenant_id", "")
    backend = saved.get("setup_backend", saved.get("default_backend", ""))
    inference_url = saved.get("inference_url", "")

    if not api_key and not backend:
        print("  Not logged in. Run: adk login")
        return 1

    print()
    print("  AitherOS Identity")
    print("  =================")
    if username:
        print(f"  User:      {username}")
    if email:
        print(f"  Email:     {email}")
    if api_key:
        # Mask the key
        if len(api_key) > 16:
            print(f"  API key:   {api_key[:12]}...{api_key[-4:]}")
        else:
            print("  API key:   (set)")
    if tenant_id:
        print(f"  Tenant:    {tenant_id}")
    if backend:
        print(f"  Backend:   {backend}")
    if inference_url:
        print(f"  Inference: {inference_url}")
    print()
    return 0


def cmd_logout(args) -> int:
    """Clear saved auth tokens."""
    import json as _json
    config_path = Path.home() / ".aither" / "config.json"
    if config_path.exists():
        try:
            config = _json.loads(config_path.read_text())
        except (OSError, ValueError):
            config = {}
        for key in ("api_key", "username", "email", "tenant_id"):
            config.pop(key, None)
        config_path.write_text(_json.dumps(config, indent=2))
        print("  Logged out. Auth tokens cleared from ~/.aither/config.json")
    else:
        print("  No config found — already logged out.")

    # Also clear auth.json active profile if it exists
    auth_path = Path.home() / ".aither" / "auth.json"
    if auth_path.exists():
        try:
            import json as _json
            auth = _json.loads(auth_path.read_text())
            if isinstance(auth, dict):
                auth["active_profile"] = ""
                auth_path.write_text(_json.dumps(auth, indent=2))
                print("  Cleared active profile in ~/.aither/auth.json")
        except Exception:
            pass
    return 0


def cmd_connect(args):
    """Connect to AitherOS — detect local LLMs, activate cloud, join mesh."""
    import asyncio
    import json as _json

    # ── Elysium desktop connect shortcut ──
    if getattr(args, "elysium", None):
        return _connect_elysium(args)

    async def _connect():
        from adk.elysium import Elysium

        print()
        print("  AitherOS Connect")
        print("  ================")
        print()

        # ── 1. Local inference ─────────────────────────────────────
        print("  LOCAL INFERENCE")
        print("  ───────────────")
        backends_found = []
        import httpx

        # vLLM (preferred — enables true concurrent/parallel agents)
        for port in [8000, 8100, 8101, 8102, 8120, 8200, 8201, 8202, 8203]:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"http://localhost:{port}/v1/models")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m["id"] for m in data.get("data", [])]
                        backends_found.append(("vllm", models))
                        print(f"  [OK] vLLM (:{port}) — {', '.join(models[:3])}")
            except Exception:
                pass

        # Ollama (fallback — serializes requests, no true parallelism)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    backends_found.append(("ollama", models))
                    print(f"  [OK] Ollama — {len(models)} model(s): {', '.join(models[:5])}")
        except Exception:
            if not backends_found:
                print("  [--] Ollama — not detected")

        if not backends_found:
            print("  [--] No local LLM backends found")
            print("       Run 'aither setup' to auto-configure vLLM (recommended)")
            print("       Or install Ollama as fallback: https://ollama.com")

        # ── 2. Cloud acceleration ──────────────────────────────────
        print()
        print("  CLOUD ACCELERATION (Elysium)")
        print("  ────────────────────────────")

        # Resolve API key: flag > env > saved config
        api_key = args.api_key or os.environ.get("AITHER_API_KEY", "")
        if not api_key:
            saved = load_saved_config()
            api_key = saved.get("api_key", "")

        gateway_ok = False
        models_available = []
        balance_info = {}

        if api_key:
            print(f"  [OK] API key: {api_key[:16]}...")

            # Test inference endpoint
            try:
                async with httpx.AsyncClient(timeout=5.0, headers={
                    "Authorization": f"Bearer {api_key}",
                }) as client:
                    resp = await client.get("https://mcp.aitherium.com/health")
                    if resp.status_code == 200:
                        print("  [OK] Inference gateway: mcp.aitherium.com")
            except Exception:
                print("  [!!] Inference gateway: unreachable")

            # Fetch models
            try:
                async with httpx.AsyncClient(timeout=5.0, headers={
                    "Authorization": f"Bearer {api_key}",
                }) as client:
                    resp = await client.get("https://mcp.aitherium.com/v1/models")
                    if resp.status_code == 200:
                        data = resp.json()
                        models_available = [m["id"] for m in data.get("data", []) if m.get("accessible", True)]
                        if models_available:
                            print(f"  [OK] Models: {', '.join(models_available[:5])}")
                            if len(models_available) > 5:
                                print(f"       + {len(models_available) - 5} more")
            except Exception:
                pass

            # Test gateway + balance
            try:
                async with httpx.AsyncClient(timeout=5.0, headers={
                    "Authorization": f"Bearer {api_key}",
                }) as client:
                    resp = await client.get("https://gateway.aitherium.com/health")
                    if resp.status_code == 200:
                        gateway_ok = True
                        print("  [OK] Gateway: gateway.aitherium.com")

                    resp = await client.get("https://gateway.aitherium.com/v1/billing/balance")
                    if resp.status_code == 200:
                        balance_info = resp.json()
                        plan = balance_info.get("plan", "free")
                        bal = balance_info.get("balance", 0)
                        print(f"  [OK] Plan: {plan} | Balance: {bal} tokens")
            except Exception:
                pass
        else:
            print("  [--] No API key found")
            print()
            print("  No account? Run: aither register")
            print()
            print("  Or set an existing key:")
            print("    aither connect --api-key aither_sk_live_...")
            print()
            print("  What you get with Elysium:")
            print("    - Cloud inference (no local GPU needed)")
            print("    - 100+ MCP tools (code search, memory, training)")
            print("    - AitherMesh — share compute with other nodes")
            print("    - Agent marketplace — discover and use community agents")

        # ── 2b. Tenant info ────────────────────────────────────────
        tenant_info = {}
        if api_key and gateway_ok:
            ely = Elysium(api_key=api_key)
            tenant_info = await ely.fetch_tenant_info()
            if tenant_info:
                tid = tenant_info.get("tenant_id", "unknown")
                tier = tenant_info.get("tier", tenant_info.get("plan", "unknown"))
                role = tenant_info.get("role", "member")
                print(f"  [OK] Tenant: {tid} | Tier: {tier} | Role: {role}")

        # ── 3. MCP tools ──────────────────────────────────────────
        print()
        print("  MCP TOOLS")
        print("  ─────────")

        # Local AitherNode
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get("http://localhost:8080/health")
                if resp.status_code == 200:
                    data = resp.json()
                    mode = data.get("mode", "unknown")
                    print(f"  [OK] AitherNode (local): port 8080, mode={mode}")
        except Exception:
            print("  [--] AitherNode (local): not running")

        # Cloud MCP
        if api_key and gateway_ok:
            print("  [OK] MCP Gateway (cloud): mcp.aitherium.com")
        elif api_key:
            print("  [--] MCP Gateway (cloud): gateway unreachable")
        else:
            print("  [--] MCP Gateway (cloud): needs API key")

        # ── 4. Mesh network ────────────────────────────────────────
        print()
        print("  MESH NETWORK (AitherNet)")
        print("  ────────────────────────")
        if api_key and gateway_ok:
            try:
                async with httpx.AsyncClient(timeout=5.0, headers={
                    "Authorization": f"Bearer {api_key}",
                }) as client:
                    resp = await client.get("https://gateway.aitherium.com/v1/mesh/status")
                    if resp.status_code == 200:
                        mesh = resp.json()
                        nodes = mesh.get("total_nodes", 0)
                        print(f"  [OK] Mesh active — {nodes} node(s) online")
                    else:
                        print("  [--] Mesh status unknown")
            except Exception:
                print("  [--] Mesh: not connected")
        else:
            print("  [--] Mesh: needs API key + gateway")
            print("       Join the mesh to share compute and accelerate inference")

        # ── 5. Save config ─────────────────────────────────────────
        if args.save:
            save_data = {
                "gateway_url": "https://gateway.aitherium.com",
                "inference_url": "https://mcp.aitherium.com/v1",
            }
            if api_key:
                save_data["api_key"] = api_key
            if backends_found:
                save_data["default_backend"] = backends_found[0][0]
            if tenant_info.get("tenant_id"):
                save_data["tenant_id"] = tenant_info["tenant_id"]

            config_path = save_saved_config(save_data)
            print(f"\n  Config saved to {config_path}")

        # ── Summary ───────────────────────────────────────────────
        print()
        print("  " + "=" * 48)
        local_count = sum(len(m) for _, m in backends_found)
        cloud_count = len(models_available)
        total_models = local_count + cloud_count

        if total_models > 0:
            parts = []
            if local_count:
                parts.append(f"{local_count} local")
            if cloud_count:
                parts.append(f"{cloud_count} cloud")
            print(f"  READY — {total_models} models ({', '.join(parts)})")
            print()
            print("  Next steps:")
            print("    aither init my-agent       # Create an agent")
            print("    cd my-agent && python agent.py")
            if not api_key:
                print()
                print("  Want more? Connect to Elysium for cloud acceleration:")
                print("    aither connect --api-key aither_sk_live_...")
        elif api_key:
            print("  CLOUD MODE — using Elysium for inference")
            print()
            print("  Next steps:")
            print("    aither init my-agent       # Create an agent")
            print("    cd my-agent && python agent.py")
        else:
            print("  NO BACKEND — install Ollama or connect to Elysium")
            print()
            print("  Option A (local):  Install Ollama at https://ollama.com")
            print("  Option B (cloud):  aither connect --api-key aither_sk_live_...")
            print("  No account?        aither register")

        # ── Tier comparison ───────────────────────────────────────
        if not api_key or (api_key and balance_info.get("plan") == "free"):
            print()
            print("  " + "-" * 48)
            print("  TIERS")
            print()
            print("  Free       Your GPU, your models, basic MCP tools")
            print("  Pro        + Cloud inference, 100+ MCP tools, mesh compute")
            print("  Enterprise + Sovereign deployment, full AitherOS, RBAC,")
            print("               tenant isolation, training pipelines")
            print()
            print("  https://aitherium.com/pricing")

        # ── OpenClaw detection ───────────────────────────────────
        from pathlib import Path as _Path
        openclaw_dir = _Path.home() / ".openclaw"
        if openclaw_dir.exists():
            import json as _oc_json
            oc_config = {}
            oc_config_path = openclaw_dir / "openclaw.json"
            if oc_config_path.exists():
                try:
                    oc_config = _oc_json.loads(oc_config_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            already = any("aither" in k.lower() for k in oc_config.get("mcpServers", {}))
            if not already:
                print()
                print("  " + "-" * 48)
                print("  OPENCLAW DETECTED")
                print()
                print("  Connect OpenClaw to AitherOS agent fleet:")
                print("    aither integrate openclaw")
                print()
                print("  This gives OpenClaw access to 29 agents, swarm coding,")
                print("  memory graph, and 100+ MCP tools.")

        print()
        return 0

    return asyncio.run(_connect())


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from LLM output."""
    # Closed tags (including <thinking>)
    text = re.sub(r'<think(?:ing)?>[\s\S]*?</think(?:ing)?>', '', text, flags=re.IGNORECASE)
    # Unclosed trailing tag
    text = re.sub(r'<think(?:ing)?>[^<]*$', '', text, flags=re.IGNORECASE)
    return text.strip()


def cmd_aeon(args):
    """Interactive multi-agent group chat."""
    import asyncio

    async def _aeon():
        from adk.aeon import AeonSession, AEON_PRESETS

        preset = args.preset or "balanced"
        custom_agents = args.agents.split(",") if args.agents else None
        rounds = args.rounds or 1
        synthesize = not args.no_synthesize

        participants = custom_agents
        if custom_agents:
            # Ensure orchestrator is present
            if "aither" not in custom_agents:
                custom_agents.append("aither")

        session = AeonSession(
            participants=participants,
            preset=preset,
            rounds=rounds,
            synthesize=synthesize,
        )

        # ANSI colors for agent names
        colors = [
            "\033[96m",   # cyan
            "\033[93m",   # yellow
            "\033[95m",   # magenta
            "\033[92m",   # green
            "\033[94m",   # blue
            "\033[91m",   # red
        ]
        reset = "\033[0m"
        bold = "\033[1m"

        agent_colors = {}
        for i, name in enumerate(session.participants):
            agent_colors[name] = colors[i % len(colors)]

        names = ", ".join(session.participants)
        print(f"\n  Aeon Group Chat — [{preset}] {names}")
        print(f"  Session: {session.session_id}")
        print(f"  Rounds: {rounds} | Synthesize: {synthesize}")
        print("  Type 'quit' to exit, 'reset' to start a new session.\n")

        while True:
            try:
                user_input = input(f"  {bold}you>{reset} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Bye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("  Bye!")
                break
            if user_input.lower() == "reset":
                session.reset()
                # Re-assign colors
                for i, name in enumerate(session.participants):
                    agent_colors[name] = colors[i % len(colors)]
                print(f"  New session: {session.session_id}\n")
                continue

            response = await session.chat(user_input)

            print()
            for msg in response.messages:
                color = agent_colors.get(msg.agent, "")
                content = _strip_think_tags(msg.content)
                print(f"  {color}[{msg.agent}]{reset} {content}")
                print()

            if response.synthesis:
                color = agent_colors.get(response.synthesis.agent, colors[0])
                content = _strip_think_tags(response.synthesis.content)
                print(f"  {color}{bold}[{response.synthesis.agent} - synthesis]{reset} {content}")
                print()

            print(f"  --- round {response.round_number} | {response.total_tokens} tokens | {response.total_latency_ms:.0f}ms ---\n")

        return 0

    return asyncio.run(_aeon())


def cmd_deploy(args):
    """Package and deploy an agent to AitherOS via the gateway."""
    import asyncio
    import json as _json
    import zipfile
    import tempfile

    async def _deploy():
        project_dir = Path(args.directory or ".").resolve()
        print(f"📦 Deploying agent from {project_dir}\n")

        # Validate project
        agent_file = project_dir / "agent.py"
        config_file = project_dir / "config.yaml"
        if not agent_file.exists():
            print("❌ No agent.py found. Run 'aither init' first.")
            return 1

        # Get API key
        api_key = args.api_key or os.environ.get("AITHER_API_KEY", "")
        if not api_key:
            # Try saved config
            config_path = Path.home() / ".aither" / "config.json"
            if config_path.exists():
                try:
                    saved = _json.loads(config_path.read_text())
                    api_key = saved.get("api_key", "")
                except Exception:
                    pass
        if not api_key:
            print("❌ No API key. Run 'aither connect --api-key <key>' first.")
            return 1

        # Read agent name from config or args
        agent_name = args.name
        if not agent_name and config_file.exists():
            try:
                import yaml
                cfg = yaml.safe_load(config_file.read_text())
                agent_name = cfg.get("identity", "my-agent")
            except Exception:
                agent_name = project_dir.name

        if not agent_name:
            agent_name = project_dir.name

        print(f"  Agent: {agent_name}")

        # Package the project into a zip
        print("  📁 Packaging project...")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in project_dir.rglob("*"):
                if f.is_file() and not any(
                    part.startswith(".") or part == "__pycache__"
                    for part in f.relative_to(project_dir).parts
                ):
                    zf.write(f, f.relative_to(project_dir))

        zip_size = os.path.getsize(tmp_path)
        print(f"  📦 Package size: {zip_size / 1024:.1f} KB")

        # Register agent with gateway
        print("  🚀 Registering with gateway...")
        try:
            import httpx
            gateway = args.gateway or "https://gateway.aitherium.com"
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Register agent metadata
                resp = await client.post(
                    f"{gateway}/v1/agents/register",
                    json={
                        "agent_name": agent_name,
                        "capabilities": args.capabilities.split(",") if args.capabilities else ["chat"],
                        "description": args.description or f"ADK agent: {agent_name}",
                        "version": args.version or "0.1.0",
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    agent_id = data.get("agent_id", "unknown")
                    print(f"  ✅ Registered: {agent_id}")
                else:
                    error = resp.json() if resp.headers.get(
                        "content-type", ""
                    ).startswith("application/json") else {"error": resp.text}
                    print(f"  ❌ Registration failed: {error}")
                    return 1

                # Upload package (deploy endpoint)
                print("  📤 Uploading package...")
                with open(tmp_path, "rb") as zf:
                    resp = await client.post(
                        f"{gateway}/v1/agents/{agent_id}/deploy",
                        content=zf.read(),
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/zip",
                            "X-Agent-Name": agent_name,
                        },
                    )
                    if resp.status_code in (200, 201):
                        print("  ✅ Deployed successfully!")
                    elif resp.status_code == 404:
                        print("  ⚠️  Deploy endpoint not yet available on gateway.")
                        print("     Agent registered but code deployment coming soon.")
                    else:
                        print(f"  ⚠️  Deploy returned {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return 1
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        print(f"\n✅ Agent '{agent_name}' deployed to AitherOS!")
        return 0

    return asyncio.run(_deploy())


def _onboard_agent(agent_name: str, tenant_slug: str, args) -> int:
    """Register a running agent with the portal fleet.

    Usage: adk onboard --agent gargbot --tenant garg-consulting

    Steps:
        1. Detect running agent on localhost (check /health)
        2. Read agent identity from ~/.aither/agents.json
        3. Register with portal via FederationLiteClient.upsert_agents()
        4. Configure inference (if not already done)
        5. Print fleet dashboard URL
    """
    import asyncio
    import json as _json

    async def _do_onboard():
        from adk.config import load_saved_config
        saved = load_saved_config()
        api_key = getattr(args, 'api_key', None) or saved.get("api_key", "") or os.environ.get("AITHER_API_KEY", "")

        if not tenant_slug:
            ts = saved.get("tenant_id", "") or saved.get("tenant_slug", "")
        else:
            ts = tenant_slug

        if not api_key:
            print("  No API key. Run 'adk login' first.")
            return 1

        print()
        print(f"  Onboarding agent: {agent_name}")
        print(f"  Tenant: {ts or '(not set)'}")
        print()

        # 1. Check if agent is running
        agent_url = "http://localhost:8080"
        from adk.agent_registry import get_local_agent
        local_entry = get_local_agent(agent_name)
        if local_entry:
            agent_url = local_entry.get("url", agent_url)
            print(f"  [OK] Found in local registry: {agent_url}")
        else:
            print("  [..] Not in local registry, checking localhost:8080...")

        import httpx
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{agent_url}/api/health")
                if resp.status_code < 300:
                    print(f"  [OK] Agent healthy at {agent_url}")
                else:
                    print(f"  [!!] Agent returned {resp.status_code}")
        except Exception:
            print(f"  [!!] Agent not reachable at {agent_url}")
            print(f"       Start it first: adk run --identity {agent_name}")
            return 1

        # 2. Read instance ID
        instance_id = ""
        if local_entry:
            instance_id = local_entry.get("instance_id", "")
        if not instance_id:
            home = Path.home()
            iid_file = home / ".aither" / "agents" / agent_name / ".aither" / "instance_id"
            if iid_file.exists():
                instance_id = iid_file.read_text(encoding="utf-8").strip()

        # 3. Register with portal
        portal_url = saved.get("portal_url", "") or os.environ.get(
            "AITHER_PORTAL_URL", "https://portal.aitherium.com"
        )
        invoke_url = os.environ.get("AITHER_INVOKE_URL", agent_url)

        try:
            from adk.federation_lite import FederationLiteClient
            fed = FederationLiteClient(
                hub_url=portal_url,
                api_key=api_key,
                node_id=instance_id or agent_name,
            )
            result = await fed.upsert_agents([{
                "name": agent_name,
                "invoke_url": invoke_url,
                "status": "online",
                "tenant_id": ts,
                "instance_id": instance_id,
            }])
            if result.get("error"):
                print(f"  [!!] Portal registration failed: {result}")
            else:
                print("  [OK] Registered with portal fleet")
        except Exception:
            # Fallback: direct HTTP
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    await client.post(
                        f"{portal_url}/v1/agents/upsert",
                        json={
                            "name": agent_name,
                            "scope": {"visibility": "workspace", "tenant_id": ts},
                            "invoke_url": invoke_url,
                            "instance_id": instance_id,
                            "status": "online",
                        },
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                print("  [OK] Registered with portal fleet (direct)")
            except Exception as e2:
                print(f"  [!!] Portal registration failed: {e2}")

        # 4. Print fleet URL
        print()
        print("  Fleet dashboard: https://portal.aitherium.com/portal/fleet")
        if instance_id:
            print(f"  Instance ID:     {instance_id}")
        print()
        return 0

    return asyncio.run(_do_onboard())


def cmd_onboard(args):
    """Interactive onboarding — detect products, configure, integrate."""
    import asyncio
    import json as _json

    agent_name = getattr(args, 'agent', None) or ''
    tenant_slug = getattr(args, 'tenant', None) or os.environ.get('AITHER_TENANT_SLUG', '')

    # ── Agent fleet registration mode ─────────────────────────────────
    if agent_name:
        return _onboard_agent(agent_name, tenant_slug, args)

    async def _onboard():
        # Inline ProductDetector (no AitherOS lib dependency)
        from pathlib import Path
        import shutil

        home = Path.home()
        aither_dir = home / ".aither"
        openclaw_dir = home / ".openclaw"

        # If tenant provided, write it to config immediately
        if tenant_slug:
            aither_dir.mkdir(parents=True, exist_ok=True)
            config_path = aither_dir / "config.json"
            existing = {}
            if config_path.exists():
                try:
                    existing = _json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            existing["tenant_slug"] = tenant_slug
            existing["tenant_id"] = f"tnt_{tenant_slug.replace('-', '_')}"
            config_path.write_text(_json.dumps(existing, indent=2), encoding="utf-8")

        print()
        print("  AitherOS Onboarding")
        print("  ===================")
        print()

        # ── 1. Detect products ────────────────────────────────
        print("  SCANNING ENVIRONMENT")
        print("  ────────────────────")

        products = []

        # ADK
        aither_bin = shutil.which("aither")
        if aither_bin:
            products.append("aither-adk")
            print("  [OK] AitherADK — installed")
        else:
            print("  [--] AitherADK — not found (you're running it though!)")

        # Config
        config = {}
        config_path = aither_dir / "config.json"
        if config_path.exists():
            try:
                config = _json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        api_key = args.api_key or os.environ.get("AITHER_API_KEY", "")
        if not api_key:
            api_key = config.get("api_key", "")

        if api_key:
            print(f"  [OK] API key: {api_key[:16]}...")
        else:
            print("  [--] No API key — run 'aither register' for cloud access")

        # Ollama
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            products.append("ollama")
            print("  [OK] Ollama — installed")

        # vLLM (check via import or docker)
        try:
            import importlib.util
            if importlib.util.find_spec("vllm"):
                products.append("vllm")
                print("  [OK] vLLM — installed (Python)")
        except (ImportError, ModuleNotFoundError):
            pass

        # OpenClaw
        openclaw_detected = openclaw_dir.exists()
        if openclaw_detected:
            products.append("openclaw")
            oc_config = {}
            oc_config_path = openclaw_dir / "openclaw.json"
            if oc_config_path.exists():
                try:
                    oc_config = _json.loads(oc_config_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            version = oc_config.get("version", "unknown")
            aither_integrated = any(
                "aither" in k.lower()
                for k in oc_config.get("mcpServers", {})
            )

            if aither_integrated:
                print(f"  [OK] OpenClaw v{version} — integrated with AitherOS")
            else:
                print(f"  [!!] OpenClaw v{version} — detected but NOT integrated")
                print("       Run 'aither integrate openclaw' to connect agent fleets")

        # GPU
        gpu_name = ""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
                if len(lines) > 1:
                    # Multi-GPU: show all, highlight best
                    best_vram = 0
                    total_vram = 0
                    for i, line in enumerate(lines):
                        parts = [p.strip() for p in line.split(",")]
                        g_name = parts[0] if parts else "GPU"
                        g_vram = float(parts[1]) / 1024 if len(parts) > 1 else 0
                        total_vram += g_vram
                        if g_vram > best_vram:
                            best_vram = g_vram
                            gpu_name = g_name
                        print(f"  [OK] GPU {i}: {g_name} ({g_vram:.0f}GB VRAM)")
                    print(f"  [OK] Total VRAM: {total_vram:.0f}GB across {len(lines)} GPUs")
                else:
                    parts = [p.strip() for p in lines[0].split(",")]
                    gpu_name = parts[0].strip()
                    vram = float(parts[1].strip()) / 1024 if len(parts) > 1 else 0
                    print(f"  [OK] GPU: {gpu_name} ({vram:.0f}GB VRAM)")
        except Exception:
            print("  [--] No NVIDIA GPU detected")

        # ── 2. Onboarding plan ────────────────────────────────
        print()
        print("  ONBOARDING PLAN")
        print("  ───────────────")

        step_num = 1

        if not api_key:
            print(f"  {step_num}. Register for Aitherium (free)")
            print("     -> aither register")
            step_num += 1

        if not ollama_bin and not gpu_name:
            print(f"  {step_num}. Set up inference backend")
            print("     -> Install Ollama: https://ollama.com")
            print("     -> Or use cloud: aither register")
            step_num += 1

        if openclaw_detected and not aither_integrated:
            print(f"  {step_num}. Connect OpenClaw to AitherOS agent fleet")
            print("     -> aither integrate openclaw")
            step_num += 1

        print(f"  {step_num}. Create your first agent")
        print("     -> aither init my-agent && cd my-agent && aither run")
        step_num += 1

        if api_key:
            print(f"  {step_num}. Publish to Elysium marketplace (optional)")
            print("     -> aither publish")
            step_num += 1

        # ── 3. Auto-configure IDE MCP servers ────────────────
        print()
        print("  CONFIGURING MCP SERVERS")
        print("  ���─────────────────────")

        mcp_url = "http://localhost:8080"
        mcp_configured = []

        # Claude Code — .mcp.json goes in PROJECT ROOT (CWD), not ~/.claude/
        # Claude Code reads MCP config from the working directory, not global.
        claude_dir = home / ".claude"
        mcp_json = {
            "mcpServers": {
                "aitheros": {
                    "command": "npx",
                    "args": ["-y", "aither-mcp-server"],
                    "disabled": False,
                }
            }
        }

        def _write_mcp(target: Path, label: str):
            """Write or merge MCP config into a .mcp.json file."""
            try:
                if target.exists():
                    existing = _json.loads(target.read_text(encoding="utf-8"))
                    servers = existing.get("mcpServers", {})
                    if "aitheros" not in servers:
                        servers["aitheros"] = mcp_json["mcpServers"]["aitheros"]
                        existing["mcpServers"] = servers
                        target.write_text(_json.dumps(existing, indent=2), encoding="utf-8")
                        print(f"  [OK] {label} — AitherOS MCP added to existing config")
                        return True
                    else:
                        print(f"  [OK] {label} — AitherOS MCP already configured")
                        return True
                else:
                    target.write_text(_json.dumps(mcp_json, indent=2), encoding="utf-8")
                    print(f"  [OK] {label} — MCP configured at {target}")
                    return True
            except Exception as e:
                print(f"  [!!] {label} — failed: {e}")
                return False

        # 1. Write to current project directory (primary — Claude Code reads from CWD)
        cwd_mcp = Path.cwd() / ".mcp.json"
        if _write_mcp(cwd_mcp, "Claude Code (project)"):
            mcp_configured.append("claude-code")

        # 2. Also write to ~/.claude/.mcp.json as global fallback
        if claude_dir.exists():
            _write_mcp(claude_dir / ".mcp.json", "Claude Code (global)")
        else:
            print("  [--] Claude Code global — ~/.claude/ not found (project-level config is sufficient)")

        # Cursor — write to ~/.cursor/mcp.json
        cursor_dir = home / ".cursor"
        if cursor_dir.exists():
            cursor_mcp = cursor_dir / "mcp.json"
            cursor_config = {
                "mcpServers": {
                    "aitheros": {
                        "url": f"{mcp_url}/sse",
                    }
                }
            }
            try:
                if cursor_mcp.exists():
                    existing = _json.loads(cursor_mcp.read_text(encoding="utf-8"))
                    if "aitheros" not in existing.get("mcpServers", {}):
                        existing.setdefault("mcpServers", {})["aitheros"] = cursor_config["mcpServers"]["aitheros"]
                        cursor_mcp.write_text(_json.dumps(existing, indent=2), encoding="utf-8")
                        print("  [OK] Cursor — AitherOS MCP added")
                        mcp_configured.append("cursor")
                    else:
                        print("  [OK] Cursor — AitherOS MCP already configured")
                        mcp_configured.append("cursor")
                else:
                    cursor_mcp.write_text(_json.dumps(cursor_config, indent=2), encoding="utf-8")
                    print(f"  [OK] Cursor — MCP configured at {cursor_mcp}")
                    mcp_configured.append("cursor")
            except Exception as e:
                print(f"  [!!] Cursor — failed to write config: {e}")
        else:
            print("  [--] Cursor — not detected (~/.cursor/ not found)")

        # OpenClaw — use aither integrate openclaw
        if openclaw_detected and not aither_integrated:
            try:
                oc_config_path = openclaw_dir / "openclaw.json"
                if oc_config_path.exists():
                    oc_config = _json.loads(oc_config_path.read_text(encoding="utf-8"))
                    oc_config.setdefault("mcpServers", {})["aither_mcp_configured"] = {
                        "command": "npx",
                        "args": ["-y", "aither-mcp-server"],
                        "disabled": False,
                    }
                    oc_config_path.write_text(_json.dumps(oc_config, indent=2), encoding="utf-8")
                    print("  [OK] OpenClaw — AitherOS MCP added")
                    mcp_configured.append("openclaw")
            except Exception as e:
                print(f"  [!!] OpenClaw — failed to integrate: {e}")
        elif openclaw_detected and aither_integrated:
            print("  [OK] OpenClaw — already integrated")
            mcp_configured.append("openclaw")

        # VS Code — write to .vscode/mcp.json in current dir
        vscode_dir = Path.cwd() / ".vscode"
        if vscode_dir.exists():
            vscode_mcp = vscode_dir / "mcp.json"
            if not vscode_mcp.exists():
                try:
                    vscode_mcp.write_text(_json.dumps({
                        "servers": {
                            "aitheros": {"url": f"{mcp_url}/sse"}
                        }
                    }, indent=2), encoding="utf-8")
                    print("  [OK] VS Code — MCP configured in .vscode/mcp.json")
                    mcp_configured.append("vscode")
                except Exception:
                    pass

        if not mcp_configured:
            print("  [--] No IDE detected — configure manually:")
            print(f"       MCP server URL: {mcp_url}/sse")

        # ── 4. Quick actions ──────────────────────────────────
        print()
        print("  QUICK ACTIONS")
        print("  ─────────────")
        print("  aither register        — Create account + get API key")
        print("  aither connect         — Detect backends + test cloud")
        print("  aither init <name>     — Scaffold new agent project")
        print("  aither integrate       — Connect external tools (OpenClaw, etc.)")
        print("  aither publish         — Submit agent to Elysium marketplace")
        print("  aither aeon            — Multi-agent group chat")
        print()

        # ── 4. Install other products ─────────────────────────
        print("  INSTALL OTHER PRODUCTS")
        print("  ──────────────────────")
        print("  pip install aither-adk          # CLI + SDK (this package)")
        print("  winget install Aitherium.Desktop # Native desktop app (Windows)")
        print("  brew install --cask aither-desktop  # Desktop (macOS)")
        print("  Chrome Web Store: AitherConnect  # Browser extension")
        print()

        if openclaw_detected and not aither_integrated:
            print("  " + "=" * 50)
            print("  OPENCLAW DETECTED — Integration available!")
            print("  Run 'aither integrate openclaw' to connect:")
            print("    - 29 specialized AI agents")
            print("    - 100+ MCP tools (code, memory, search)")
            print("    - Swarm coding (11 agents in parallel)")
            print("    - Memory graph + knowledge base")
            print("  " + "=" * 50)
            print()

        return 0

    return asyncio.run(_onboard())


def cmd_enroll(args) -> int:
    """Register this workstation with the control plane.

    Shows hardware info, available models, and enrollment status.
    Persists node_id to ~/.aither/node_auth.json for future heartbeats.
    """
    import asyncio

    try:
        portal_url = getattr(args, "portal", None) or os.environ.get("AITHER_PORTAL_URL", "https://portal.aitherium.com")
        genesis_url = getattr(args, "genesis", None) or os.environ.get("AITHER_GENESIS_URL", "http://localhost:8001")
        no_heartbeat = getattr(args, "no_heartbeat", False)
        force = getattr(args, "force", False)

        from adk.fleet_enroll import enroll_on_boot, _load_node_auth, _generate_node_id
        from adk.enrollment import build_registration

        # Check if already enrolled
        existing = _load_node_auth()
        if existing.get("node_id") and not force:
            node_id = existing["node_id"]
            print(f"✓ Already enrolled: {node_id}")
            print()
            print("Node details:")
            print(f"  Tenant: {existing.get('tenant_slug', 'unknown')}")
            print(f"  Hub: {existing.get('hub_url', 'unknown')}")
            print(f"  Mode: {existing.get('mode', 'legacy')}")
            print()
            print("To re-enroll, use: adk enroll --force")
            return 0

        # Perform enrollment
        result = asyncio.run(enroll_on_boot(
            genesis_url=genesis_url,
            portal_url=portal_url,
            enable_heartbeat=not no_heartbeat,
        ))

        if not result.get("enrolled"):
            error_detail = result.get("error", "unknown error")
            print(f"ERROR: Enrollment failed: {error_detail}")
            return 1

        # Build registration for display
        node_id = result.get("node_id")
        reg = build_registration(node_id)

        # Format success response
        print("✓ Enrollment successful")
        print()
        print("Node Information:")
        print(f"  ID: {result.get('node_id', 'unknown')}")
        print(f"  Hardware:")
        print(f"    CPU: {reg.get('cpu_count', 0)} cores")
        print(f"    RAM: {reg.get('ram_mb', 0)} MB")
        if reg.get("gpu_name"):
            print(f"    GPU: {reg['gpu_name']} ({reg.get('gpu_vram_mb', 0)} MB)")
        else:
            print(f"    GPU: none")
        models_str = ", ".join(reg.get("available_models", []))[:80]
        print(f"  Available Models: {models_str if models_str else 'none'}")
        print()
        print("Fleet Status:")
        print(f"  Workspace ID: {result.get('workspace_id', 'N/A')}")
        print(f"  Agents Upserted: {result.get('agents_upserted', 0)}")
        if not no_heartbeat:
            print(f"  Heartbeat: enabled (60s interval)")
        print()
        print(f"View in portal: {portal_url.rstrip('/')}/portal/workstation")
        print()
        return 0
    except Exception as e:
        print(f"ERROR: Enrollment failed: {e}")
        return 1


def _find_cloudflared() -> str | None:
    """Locate the cloudflared binary, checking PATH + common install dirs."""
    import shutil
    cf = shutil.which("cloudflared")
    if cf:
        return cf
    candidates = [
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
        "/usr/local/bin/cloudflared",
        "/opt/homebrew/bin/cloudflared",
        os.path.expanduser("~/.cloudflared/cloudflared"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _cloudflared_install_hint() -> str:
    if sys.platform == "win32":
        return "  winget install --id Cloudflare.cloudflared"
    if sys.platform == "darwin":
        return "  brew install cloudflared"
    return ("  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/"
            "download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && "
            "chmod +x /usr/local/bin/cloudflared")


def _cloudflared_asset() -> str | None:
    """Map this platform to the cloudflared GitHub release asset name."""
    import platform as _plat
    sysname = _plat.system()
    mach = _plat.machine().lower()
    if sysname == "Windows":
        if mach in ("x86", "i386", "i686"):
            return "cloudflared-windows-386.exe"
        return "cloudflared-windows-amd64.exe"
    if sysname == "Linux":
        if mach in ("aarch64", "arm64"):
            return "cloudflared-linux-arm64"
        if mach in ("armv7l", "armv6l", "arm"):
            return "cloudflared-linux-arm"
        return "cloudflared-linux-amd64"
    if sysname == "Darwin":
        if mach in ("arm64", "aarch64"):
            return "cloudflared-darwin-arm64.tgz"
        return "cloudflared-darwin-amd64.tgz"
    return None


def _parse_cloudflared_checksums(body: str) -> dict:
    """Parse 'cloudflared-...: <sha256>' pairs from a cloudflared release body.

    cloudflared publishes checksums as colon-separated lines inside a markdown code
    block (NOT the space-separated ``checksums.sha256`` format the shell binary uses),
    so this is deliberately its own parser.
    """
    out: dict[str, str] = {}
    hexset = set("0123456789abcdefABCDEF")
    for line in (body or "").splitlines():
        s = line.strip().strip("`").strip()
        if ":" not in s:
            continue
        name, _, digest = s.partition(":")
        name = name.strip()
        digest = digest.strip()
        if name.startswith("cloudflared-") and len(digest) == 64 and set(digest) <= hexset:
            out[name] = digest.lower()
    return out


def _ensure_cloudflared() -> str | None:
    """Return a path to cloudflared, downloading it to ~/.aither/bin if missing.

    Secure-by-default: verifies the published SHA256 and refuses to install without
    one unless AITHER_CLOUDFLARED_REQUIRE_CHECKSUM=0. Returns None on any failure.
    """
    existing = _find_cloudflared()
    if existing:
        return existing

    import hashlib
    import shutil as _shutil
    import stat as _stat

    import httpx

    asset = _cloudflared_asset()
    if not asset:
        import platform as _plat
        print(f"  Unsupported platform for cloudflared auto-download: "
              f"{_plat.system()} {_plat.machine()}")
        return None

    cache = Path.home() / ".aither" / "bin"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / ("cloudflared.exe" if sys.platform == "win32" else "cloudflared")
    require_ck = os.environ.get(
        "AITHER_CLOUDFLARED_REQUIRE_CHECKSUM", "1").lower() not in ("0", "false", "no")
    tmp = cache / (asset + ".part")

    try:
        rel = httpx.get(
            "https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
            follow_redirects=True, timeout=30).json()
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        url = assets.get(asset)
        if not url:
            print(f"  cloudflared asset '{asset}' not in the latest release.")
            return None
        checksums = _parse_cloudflared_checksums(rel.get("body", ""))

        with httpx.stream("GET", url, follow_redirects=True, timeout=180) as dl:
            dl.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in dl.iter_bytes(65536):
                    f.write(chunk)

        expected = checksums.get(asset)
        if expected:
            h = hashlib.sha256()
            with open(tmp, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            if h.hexdigest().lower() != expected:
                tmp.unlink(missing_ok=True)
                print("  cloudflared checksum verification FAILED — not installing.")
                return None
        elif require_ck:
            tmp.unlink(missing_ok=True)
            print("  No published checksum for cloudflared — refusing to install "
                  "(set AITHER_CLOUDFLARED_REQUIRE_CHECKSUM=0 to override).")
            return None

        if asset.endswith(".tgz"):
            import tarfile
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                with tarfile.open(tmp, "r:gz") as tf:
                    tf.extractall(td)  # noqa: S202 — trusted cloudflare release asset
                found = list(Path(td).rglob("cloudflared"))
                if not found:
                    tmp.unlink(missing_ok=True)
                    print("  cloudflared archive did not contain the binary.")
                    return None
                _shutil.copy2(found[0], dest)
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(dest)

        if sys.platform != "win32":
            dest.chmod(dest.stat().st_mode | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH)
        return str(dest)
    except (httpx.HTTPError, OSError, KeyError, ValueError) as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"  cloudflared download failed: {e}")
        return None


def cmd_host(args):
    """One command to host a self-hosted agent and connect it to your fleet.

    Prompts for your model key (stored locally in ~/.aither/provider_keys.json —
    NEVER sent to AitherOS), starts aither-serve on your machine with your chosen
    brain, opens a secure Cloudflare quick tunnel (no account, no inbound ports),
    and registers the agent with the AitherOS control plane. Ctrl+C stops the
    agent + tunnel and deregisters.

    You run the brain (your key), the body (this loop) and the hands (your MCP).
    AitherOS drives it over the tunnel — chat + live trace, tool approval,
    artifacts. The only secret AitherOS holds is the callback bearer it presents
    back to YOUR agent.
    """
    import base64
    import getpass
    import re
    import secrets as _secrets
    import subprocess
    import threading
    import time as _time
    from queue import Empty, Queue

    import httpx

    provider = (getattr(args, "provider", "") or "").strip().lower()
    port = getattr(args, "port", None) or 8080
    name = (getattr(args, "name", "") or "").strip()
    if not name:
        import socket
        name = re.sub(r"[^a-z0-9_-]", "-", f"{socket.gethostname()}-adk".lower())
    identity = getattr(args, "identity", None) or "aither"
    register_url = getattr(args, "register_url", "") or ""
    portal = (getattr(args, "portal", "") or "https://veil.aitherium.com").rstrip("/")
    dry_run = bool(getattr(args, "dry_run", False))

    print()
    print("  Host a self-hosted agent -> AitherOS fleet")
    print("  ==========================================")
    print()

    # ── 1. Provider + model key (local store, never an env var the user types) ──
    if not provider:
        try:
            provider = (input("  Model provider [deepseek/openai/anthropic]: ").strip().lower()
                        or "deepseek")
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
    if provider not in _KNOWN_PROVIDERS:
        print(f"  Unknown provider '{provider}'. Known: {', '.join(sorted(_KNOWN_PROVIDERS))}")
        return 1

    keys = _load_provider_keys()
    env_name = _KNOWN_PROVIDERS[provider]["env"]
    key = keys.get(provider, "") or os.environ.get(env_name, "")
    if key:
        print(f"  [+] {_KNOWN_PROVIDERS[provider]['label']} key found: {_mask_key(key)}")
    else:
        print(f"  Enter your {_KNOWN_PROVIDERS[provider]['label']} API key.")
        print("  It is stored locally (~/.aither/provider_keys.json) and NEVER sent to AitherOS.")
        try:
            key = getpass.getpass(f"  {_KNOWN_PROVIDERS[provider]['label']} API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if not key:
            print("  No key entered — aborting.")
            return 1
        ok, msg = _test_provider_key(provider, key)
        print(f"  [{'+' if ok else 'x'}] {msg}")
        if not ok:
            print("  Key failed validation — aborting (not saved).")
            return 1
        keys[provider] = key
        _save_provider_keys(keys)
        print(f"  Saved to {_keys_path()} (local only).")
    # Make the key available to the child server without the user ever exporting it.
    os.environ[env_name] = key

    _default_models = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-6",
        "gemini": "gemini-2.0-flash",
    }
    model = getattr(args, "model", None) or _default_models.get(provider, "")

    # ── 2. Callback bearer — the ONE secret the control plane presents back to us ──
    auth_token = getattr(args, "auth_token", "") or os.environ.get("AITHER_SERVER_API_KEY", "")
    minted = False
    if not auth_token:
        auth_token = base64.urlsafe_b64encode(_secrets.token_bytes(24)).decode().rstrip("=")
        minted = True

    # ── 3. Control-plane token (for registration) ──
    # Resolution order: explicit --token → env → saved login → DEVICE-FLOW LOGIN.
    # Device flow (RFC 8628) is the autonomous path: no token, no manual steps —
    # print a short code + URL, the human approves once in a browser (or an agent
    # approves it headlessly), and the token arrives. Anyone — including a non-tech
    # user or an autonomous agent on the internet — can self-onboard with ONE command.
    saved = load_saved_config()
    portal_token = (getattr(args, "token", "") or os.environ.get("AITHER_PORTAL_TOKEN", "")
                    or saved.get("api_key", "") or saved.get("access_token", ""))
    will_register = not getattr(args, "no_register", False)
    if will_register and not portal_token and not dry_run:
        login_url = _resolve_identity_url(
            getattr(args, "login_url", "") or os.environ.get("AITHER_PORTAL_URL", "")
            or portal or _DEFAULT_IDENTITY_URL
        )
        print()
        print(f"  Not signed in — starting device-flow login at {login_url} …")
        try:
            dev = _device_flow_login(login_url, client_name=f"adk-host:{name}")
            portal_token = dev.get("access_token", "") or dev.get("token", "")
        except Exception as e:  # noqa: BLE001 — surface a clear next step
            print(f"  Device-flow login failed: {e}")
            print("  Pass --token, run 'adk login', or use --no-register to run locally.")
            return 1
        if not portal_token:
            print("  No token returned from device-flow login. Aborting.")
            return 1
        try:
            save_saved_config({"api_key": portal_token})
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass
        print("  [+] Signed in (token saved to ~/.aither/config.json).")

    # ── 4. Approval policy ──
    approval = getattr(args, "approve", None)
    if approval is None:
        approval = "file_write,shell_exec,shell"  # sensible default: gate the dangerous ones
    print(f"  Tool approval gating: {approval or '(none)'}")

    # ── 5. cloudflared availability (skip if local-only) ──
    cf = _find_cloudflared()
    if will_register and not cf and not dry_run:
        print("  cloudflared is not installed (needed for the secure tunnel):")
        print(_cloudflared_install_hint())
        print("  Or re-run with --no-register to run locally without a tunnel.")
        return 1

    if dry_run:
        print()
        print("  [dry-run] would:")
        print(f"    1. aither-serve --backend {provider} --port {port} --identity {identity}")
        print(f"    2. cloudflared tunnel --url http://localhost:{port}")
        reg = register_url or f"{portal}/api/genesis/v1/agent/fleet/register"
        body = {"name": name, "invoke_url": "https://<tunnel>.trycloudflare.com",
                "reach": "tunnel", "agent_type": "adk-agent", "token": "<callback-bearer>",
                "model": model, "provider_hint": provider}
        print(f"    3. POST {reg}  {json.dumps(body)}")
        return 0

    procs: list[subprocess.Popen] = []

    def _cleanup():
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    try:
        # ── 6. Start aither-serve ──
        print()
        print(f"  Starting aither-serve ({provider}, :{port}) …")
        child_env = dict(os.environ)
        child_env["AITHER_SERVER_API_KEY"] = auth_token
        child_env["AITHER_TOOL_APPROVAL"] = approval or ""
        server = subprocess.Popen(
            [sys.executable, "-m", "adk.server", "--backend", provider,
             "--port", str(port), "--identity", identity],
            env=child_env,
        )
        procs.append(server)

        # Wait for /health
        healthy = False
        deadline = _time.time() + 60
        while _time.time() < deadline:
            if server.poll() is not None:
                print("  [x] aither-serve exited during startup.")
                return 1
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=3)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            _time.sleep(2)
        if not healthy:
            print("  [x] aither-serve did not become healthy in 60s.")
            _cleanup()
            return 1
        print(f"  [+] Agent healthy on :{port} (brain: {model or provider})")

        if not will_register:
            print()
            print(f"  Agent running locally at http://localhost:{port} (no tunnel, --no-register).")
            print("  Ctrl+C to stop.")
            server.wait()
            return 0

        # ── 7. Open the quick tunnel + capture the public URL ──
        print("  Opening a secure Cloudflare quick tunnel …")
        tunnel = subprocess.Popen(
            [cf, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        procs.append(tunnel)
        url_q: Queue = Queue()

        def _scan():
            pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
            for line in iter(tunnel.stdout.readline, ""):
                m = pat.search(line)
                if m:
                    url_q.put(m.group(0))
                    break

        threading.Thread(target=_scan, daemon=True).start()
        try:
            public_url = url_q.get(timeout=45)
        except Empty:
            print("  [x] Could not obtain a trycloudflare.com URL.")
            _cleanup()
            return 1
        print(f"  [+] Tunnel live: {public_url}")

        # ── 8. Register with the control plane ──
        # A stale/expired saved token must NOT dead-end onboarding: on 401/403 we
        # re-authenticate via device flow and retry once. This is what keeps the
        # one-command flow autonomous even when ~/.aither holds an old token.
        reg = register_url or f"{portal}/api/genesis/v1/agent/fleet/register"
        body = {"name": name, "invoke_url": public_url, "reach": "tunnel",
                "agent_type": "adk-agent", "token": auth_token,
                "model": model, "provider_hint": provider}

        def _do_register(tok: str):
            hdrs = {"Content-Type": "application/json"}
            if tok:
                hdrs["Authorization"] = f"Bearer {tok}"
            return httpx.post(reg, json=body, headers=hdrs, timeout=20)

        try:
            rr = _do_register(portal_token)
            if rr.status_code in (401, 403):
                login_url = _resolve_identity_url(
                    getattr(args, "login_url", "") or os.environ.get("AITHER_PORTAL_URL", "")
                    or portal or _DEFAULT_IDENTITY_URL
                )
                print(f"  Saved token rejected ({rr.status_code}) — re-authenticating via device flow…")
                dev = _device_flow_login(login_url, client_name=f"adk-host:{name}")
                portal_token = dev.get("access_token", "") or dev.get("token", "")
                if portal_token:
                    try:
                        save_saved_config({"api_key": portal_token})
                    except Exception:  # noqa: BLE001
                        pass
                    rr = _do_register(portal_token)
            if rr.status_code >= 300:
                print(f"  [x] Registration rejected ({rr.status_code}): {rr.text[:300]}")
                _cleanup()
                return 1
        except Exception as e:  # noqa: BLE001
            print(f"  [x] Registration failed: {e}")
            _cleanup()
            return 1
        print(f"  [+] Registered '{name}' with the control plane.")

        if minted:
            print()
            print("  [!] A callback bearer was minted for inbound control-plane calls.")
            print("     It is set on this agent for this run. To make it durable, start with:")
            print(f"       AITHER_SERVER_API_KEY={auth_token}")

        print()
        print("  [OK] Your self-hosted agent is LIVE in the fleet.")
        print(f"     Fleet:  {portal}/portal/fleet")
        print(f"     Chat:   {portal}/settings/agent?agent={name}")
        print("     Your model key never left this machine.")
        print()
        print("  Leave this running. Ctrl+C to stop + deregister.")
        server.wait()
        return 0

    except KeyboardInterrupt:
        print("\n  Stopping …")
        # Best-effort deregister
        if will_register and not getattr(args, "no_register", False) and (portal_token or register_url):
            try:
                reg_base = register_url.rsplit("/fleet/register", 1)[0] if register_url else f"{portal}/api/genesis/v1/agent"
                dh = {"Authorization": f"Bearer {portal_token}"} if portal_token else {}
                httpx.request("DELETE", f"{reg_base}/agent-endpoints/{name}", headers=dh, timeout=10)
                print("  [+] Deregistered from the fleet.")
            except Exception:
                pass
        return 0
    finally:
        _cleanup()


def cmd_integrate(args):
    """Integrate external tools with AitherOS."""
    import asyncio
    import json as _json

    target = args.target

    if target == "openclaw":
        return _integrate_openclaw(args)
    elif target == "list":
        print()
        print("  Available integrations:")
        print("  ───────────────────────")
        print("  openclaw    — Connect OpenClaw to AitherOS agent fleet")
        print("  (more coming: cursor, windsurf, continue, cline)")
        print()
        print("  Usage: aither integrate <target>")
        return 0
    else:
        print(f"  Unknown integration target: {target}")
        print("  Run 'aither integrate list' to see available integrations")
        return 1


def _integrate_openclaw(args):
    """Run OpenClaw <-> AitherOS integration."""
    import asyncio
    import json as _json
    from pathlib import Path

    async def _run():
        home = Path.home()
        openclaw_dir = home / ".openclaw"
        aither_dir = home / ".aither"

        print()
        print("  OpenClaw <-> AitherOS Integration")
        print("  ==================================")
        print()

        # 1. Detect OpenClaw
        if not openclaw_dir.exists():
            print("  [!!] OpenClaw not found at ~/.openclaw/")
            print()
            print("  Install OpenClaw first: https://openclaw.dev")
            print("  Then run this command again.")
            return 1

        print("  [OK] OpenClaw detected at ~/.openclaw/")

        # Parse config
        oc_config = {}
        oc_config_path = openclaw_dir / "openclaw.json"
        if oc_config_path.exists():
            try:
                oc_config = _json.loads(oc_config_path.read_text(encoding="utf-8"))
                version = oc_config.get("version", "unknown")
                print(f"  [OK] Version: {version}")
            except Exception:
                pass

        # Check workspace
        workspace = openclaw_dir / "workspace"
        if oc_config.get("agent", {}).get("workspace"):
            workspace = Path(oc_config["agent"]["workspace"]).expanduser()

        soul_files = []
        if workspace.exists():
            for f in ["SOUL.md", "IDENTITY.md", "AGENTS.md", "USER.md",
                       "TOOLS.md", "STYLE.md"]:
                if (workspace / f).exists():
                    soul_files.append(f)
            if soul_files:
                print(f"  [OK] Workspace soul files: {', '.join(soul_files)}")

        # Check agents
        agents_dir = openclaw_dir / "agents"
        if agents_dir.exists():
            agent_count = sum(1 for d in agents_dir.iterdir() if d.is_dir())
            if agent_count:
                print(f"  [OK] Agent sessions: {agent_count} agent(s)")

        # Already integrated?
        existing_mcp = oc_config.get("mcpServers", {})
        already = any("aither" in k.lower() for k in existing_mcp)
        if already:
            print()
            print("  [!!] AitherOS MCP servers already configured!")
            if not args.force:
                print("  Use --force to reconfigure")
                return 0

        # 2. Detect mode
        print()
        mode = args.mode or "auto"

        api_key = args.api_key or os.environ.get("AITHER_API_KEY", "")
        if not api_key:
            aither_config = aither_dir / "config.json"
            if aither_config.exists():
                try:
                    cfg = _json.loads(aither_config.read_text(encoding="utf-8"))
                    api_key = cfg.get("api_key", "")
                except Exception:
                    pass

        # Auto-detect local AitherOS
        local_running = False
        try:
            import httpx
            resp = httpx.get("http://localhost:8080/health", timeout=2.0)
            local_running = resp.status_code == 200
        except Exception:
            pass

        if mode == "auto":
            if local_running and api_key:
                mode = "hybrid"
            elif local_running:
                mode = "local"
            elif api_key:
                mode = "cloud"
            else:
                mode = "local"

        print(f"  Integration mode: {mode}")
        if local_running:
            print("  [OK] AitherOS Node running locally (port 8080)")
        if api_key:
            print(f"  [OK] API key: {api_key[:16]}...")

        # 3. Generate MCP config
        print()
        print("  CONFIGURING MCP SERVERS")
        print("  ───────────────────────")

        mcp_servers = {}

        if mode in ("local", "hybrid"):
            mcp_servers["aither-local"] = {
                "url": "http://localhost:8080/mcp/sse",
                "transport": "sse",
                "description": "AitherOS local — 29 agents, 100+ tools",
            }
            print("  [+] aither-local: localhost:8080/mcp/sse")

        if mode in ("cloud", "hybrid"):
            server_cfg = {
                "url": "https://mcp.aitherium.com/mcp/sse",
                "transport": "sse",
                "description": "AitherOS cloud — inference, agents, memory",
            }
            if api_key:
                server_cfg["env"] = {"AITHER_API_KEY": api_key}
                server_cfg["headers"] = {
                    "Authorization": f"Bearer {api_key}",
                }
            mcp_servers["aither-cloud"] = server_cfg
            print("  [+] aither-cloud: mcp.aitherium.com/mcp/sse")

        # A2A endpoint
        mcp_servers["aither-a2a"] = {
            "url": "http://localhost:8766",
            "transport": "a2a",
            "description": "AitherOS A2A — direct agent-to-agent dispatch",
        }
        print("  [+] aither-a2a: localhost:8766 (agent-to-agent)")

        if args.dry_run:
            print()
            print("  DRY RUN — would write:")
            print(_json.dumps({"mcpServers": mcp_servers}, indent=2))
            return 0

        # 4. Write config
        print()
        print("  WRITING CONFIGURATION")
        print("  ─────────────────────")

        existing_mcp.update(mcp_servers)
        oc_config["mcpServers"] = existing_mcp

        try:
            oc_config_path.write_text(
                _json.dumps(oc_config, indent=2), encoding="utf-8"
            )
            print(f"  [OK] Updated {oc_config_path}")
        except OSError as e:
            print(f"  [!!] Failed to write openclaw.json: {e}")
            return 1

        # 5. Write fleet config
        fleet_path = openclaw_dir / "aither-fleet.json"
        fleet_config = {
            "provider": "aitheros",
            "endpoint": "http://localhost:8080",
            "cloud_endpoint": "https://mcp.aitherium.com",
            "agents": [
                {"name": "demiurge", "role": "Code generation & refactoring", "tier": "pro"},
                {"name": "athena", "role": "Security analysis & threat modeling", "tier": "pro"},
                {"name": "hydra", "role": "Multi-perspective code review", "tier": "pro"},
                {"name": "apollo", "role": "Performance optimization", "tier": "pro"},
                {"name": "atlas", "role": "Service discovery & architecture", "tier": "free"},
                {"name": "viviane", "role": "Memory & knowledge recall", "tier": "free"},
                {"name": "scribe", "role": "Documentation generation", "tier": "pro"},
                {"name": "saga", "role": "Creative writing & content", "tier": "free"},
                {"name": "lyra", "role": "Research & web search", "tier": "pro"},
            ],
        }
        try:
            fleet_path.write_text(
                _json.dumps(fleet_config, indent=2), encoding="utf-8"
            )
            print(f"  [OK] Wrote {fleet_path}")
        except OSError as e:
            print(f"  [!!] Failed to write fleet config: {e}")

        # 6. Summary
        print()
        print("  " + "=" * 50)
        print("  INTEGRATION COMPLETE!")
        print()
        print("  Next steps:")
        print("  1. Restart OpenClaw to pick up the new MCP servers")
        print("  2. Try: 'use the aither agent fleet to review my code'")
        print("  3. Try: 'ask demiurge to refactor this function'")
        print("  4. Try: 'use aither swarm to implement feature X'")
        print()

        if not api_key:
            print("  Want cloud agents too?")
            print("    aither register     — Get free API key")
            print("    aither integrate openclaw --mode hybrid")
            print()

        agents_str = ", ".join(a["name"] for a in fleet_config["agents"])
        print(f"  Available agents: {agents_str}")
        print()

        return 0

    return asyncio.run(_run())


def cmd_publish(args):
    """Publish an agent to the Elysium marketplace."""
    import asyncio
    import json as _json

    async def _publish():
        project_dir = Path(args.directory or ".").resolve()

        print()
        print("  Elysium Marketplace Publisher")
        print("  =============================")
        print()

        # Check for agent.py
        if not (project_dir / "agent.py").exists():
            print("  [!!] No agent.py found in current directory.")
            print("  Run 'aither init my-agent' to create a project first.")
            return 1

        # Read config
        agent_name = args.name
        config_file = project_dir / "config.yaml"
        if not agent_name and config_file.exists():
            try:
                import yaml
                cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                agent_name = cfg.get("identity", project_dir.name)
            except Exception:
                agent_name = project_dir.name

        if not agent_name:
            agent_name = project_dir.name

        print(f"  Agent: {agent_name}")
        print(f"  Directory: {project_dir}")

        # Get API key
        api_key = args.api_key or os.environ.get("AITHER_API_KEY", "")
        if not api_key:
            saved = load_saved_config()
            api_key = saved.get("api_key", "")

        if not api_key:
            print()
            print("  [!!] No API key found.")
            print("  Run 'aither register' to create an account first.")
            return 1

        # Validate
        print()
        print("  VALIDATION")
        print("  ──────────")

        errors = []
        warnings = []

        if not (project_dir / "agent.py").exists():
            errors.append("Missing agent.py")
        if not config_file.exists():
            warnings.append("No config.yaml — using defaults")
        if not (project_dir / "README.md").exists():
            warnings.append("No README.md — recommended for discoverability")

        # Check for secrets
        for f in project_dir.rglob("*.py"):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                for pattern in ["sk-", "sk_live_", "PRIVATE_KEY"]:
                    if pattern in content:
                        warnings.append(f"Possible secret in {f.name}")
                        break
            except OSError:
                pass

        for e in errors:
            print(f"  [!!] {e}")
        for w in warnings:
            print(f"  [??] {w}")

        if errors:
            print()
            print("  Fix errors above and try again.")
            return 1

        if not errors:
            print("  [OK] Validation passed")

        if args.dry_run:
            print()
            print("  DRY RUN — would publish to Elysium marketplace")
            return 0

        # Package and submit
        print()
        print("  PUBLISHING")
        print("  ──────────")

        try:
            import httpx
            import tempfile
            import zipfile

            gateway = args.gateway or "https://gateway.aitherium.com"

            # Package
            print("  Packaging project...")
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_path = tmp.name

            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in project_dir.rglob("*"):
                    if f.is_file() and not any(
                        part.startswith(".") or part == "__pycache__"
                        for part in f.relative_to(project_dir).parts
                    ):
                        zf.write(f, f.relative_to(project_dir))

            zip_size = os.path.getsize(tmp_path)
            print(f"  Package size: {zip_size / 1024:.1f} KB")

            # Register
            print("  Registering with gateway...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{gateway}/v1/agents/register",
                    json={
                        "agent_name": agent_name,
                        "description": args.description or f"ADK agent: {agent_name}",
                        "capabilities": (
                            args.capabilities.split(",") if args.capabilities else ["chat"]
                        ),
                        "version": args.version or "0.1.0",
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )

                if resp.status_code not in (200, 201):
                    print(f"  [!!] Registration failed: {resp.text[:200]}")
                    return 1

                data = resp.json()
                agent_id = data.get("agent_id", "")
                print(f"  Registered: {agent_id}")

                # Submit listing
                print("  Submitting marketplace listing...")
                resp = await client.post(
                    f"{gateway}/v1/marketplace/listings",
                    json={
                        "agent_id": agent_id,
                        "name": agent_name,
                        "description": args.description or f"ADK agent: {agent_name}",
                        "version": args.version or "0.1.0",
                        "pricing": args.pricing or "free",
                        "tier": args.tier or "agent",
                        "category": args.category or "general",
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )

                if resp.status_code in (200, 201):
                    listing = resp.json()
                    print(f"  Listing created: {listing.get('listing_id', '')}")
                elif resp.status_code == 404:
                    print("  [??] Marketplace endpoint not yet available")
                    print("       Agent registered but listing pending.")

            os.unlink(tmp_path)

        except ImportError:
            print("  [!!] httpx not installed. Run: pip install httpx")
            return 1
        except Exception as e:
            print(f"  [!!] Error: {e}")
            return 1

        print()
        print("  " + "=" * 50)
        print(f"  PUBLISHED: {agent_name}")
        print(f"  Marketplace: https://aitherium.com/marketplace/{agent_name}")
        print("  Status: pending_review")
        print()
        print("  Your agent will be reviewed and listed within 24 hours.")
        print()

        return 0

    return asyncio.run(_publish())


def cmd_test(args):
    """Run agent tests."""
    import subprocess
    project_dir = args.directory or "."
    test_dir = os.path.join(project_dir, "tests")
    if not os.path.exists(test_dir):
        print(f"No tests/ directory in {project_dir}")
        print("Create tests/test_agent.py to get started.")
        return 1
    cmd = ["python", "-m", "pytest", test_dir, "-v"]
    if args.coverage:
        cmd.extend(["--cov", project_dir, "--cov-report", "term-missing"])
    result = subprocess.run(cmd)
    return result.returncode


def cmd_backend(args):
    """Manage LLM backends — list, set, test, switch, status."""
    import asyncio
    import time

    sub = getattr(args, "backend_command", None)

    if sub == "list":
        async def _list():
            from adk.llm import LLMRouter
            from adk.config import Config
            cfg = Config.from_env()
            router = LLMRouter(config=cfg)
            try:
                await router.get_provider()
            except ConnectionError:
                pass
            info = router.get_backends()
            print("LLM Backends")
            print("=" * 40)
            for k, v in info.items():
                print(f"  {k:20s} {v}")
            # Show available providers
            print()
            print("Available:")
            for name in ("ollama", "vllm", "openai", "anthropic", "deepseek",
                         "groq", "together", "gateway", "lmstudio", "picolm"):
                print(f"  - {name}")
        asyncio.run(_list())
        return 0

    elif sub == "set":
        provider = getattr(args, "provider", None)
        if not provider:
            print("Usage: adk backend set <provider> [--api-key KEY] [--base-url URL] [--model MODEL]")
            return 1
        data = {"default_backend": provider}
        api_key = getattr(args, "api_key", None)
        base_url = getattr(args, "base_url", None)
        model = getattr(args, "model", None)
        if api_key:
            if provider == "anthropic":
                data["anthropic_api_key"] = api_key
            elif provider == "deepseek":
                data["deepseek_api_key"] = api_key
            else:
                data["api_key"] = api_key
        if base_url:
            data["inference_url"] = base_url
        if model:
            data["default_model"] = model
        save_saved_config(data)
        print(f"Backend set to: {provider}")
        if base_url:
            print(f"  URL: {base_url}")
        if model:
            print(f"  Model: {model}")
        return 0

    elif sub == "set-reasoning":
        provider = getattr(args, "provider", None)
        if not provider:
            print("Usage: adk backend set-reasoning <provider> [--api-key KEY]")
            return 1
        data = {"reasoning_backend": provider}
        api_key = getattr(args, "api_key", None)
        model = getattr(args, "model", None)
        if api_key:
            data["reasoning_api_key"] = api_key
        if model:
            data["reasoning_model"] = model
        save_saved_config(data)
        print(f"Reasoning backend set to: {provider}")
        return 0

    elif sub == "status":
        """Show current backend configuration and connectivity."""
        cfg_dict = load_saved_config()
        current_backend = cfg_dict.get("setup_backend", "unknown")
        inference_url = cfg_dict.get("inference_url", "not configured")
        inference_model = cfg_dict.get(
            "inference_model",
            cfg_dict.get("ollama_model", "not configured"),
        )

        print("\n  Backend Status")
        print("  " + "=" * 50)
        print(f"  Backend:  {current_backend}")
        print(f"  Endpoint: {inference_url}")
        print(f"  Model:    {inference_model}")
        print()

        # Quick connectivity check
        async def _check():
            try:
                import httpx
                async with httpx.AsyncClient(timeout=3.0) as client:
                    # All three backends are OpenAI-compatible — /v1/models is
                    # the uniform liveness probe (Ollama/vLLM/llama.cpp serve it).
                    base = inference_url.rstrip("/")
                    probe = base + "/models" if base.endswith("/v1") else base + "/v1/models"
                    r = await client.get(probe)
                    if r.status_code == 200:
                        print("  Status:   UP (responding)")
                    else:
                        print(f"  Status:   HTTP {r.status_code}")
                    return 0
            except Exception as e:
                print(f"  Status:   DOWN ({type(e).__name__})")
                return 0

        asyncio.run(_check())
        print()
        return 0

    elif sub == "switch":
        """Switch to a different inference backend (ollama, llamacpp, vllm)."""
        target = getattr(args, "target_backend", None)
        if not target or target not in ("ollama", "llamacpp", "vllm"):
            print("Usage: adk backend switch <ollama|llamacpp|vllm>")
            return 1

        print()
        print("  Backend Switch")
        print("  " + "=" * 50)
        print()

        cfg_dict = load_saved_config()
        current_backend = cfg_dict.get("setup_backend", "unknown")

        if target == current_backend:
            print(f"  Already using {target}. No changes needed.")
            return 0

        # Install/ensure the target backend
        print(f"  [1/3] Preparing {target} backend...")

        if target == "ollama":
            from adk.ollama_setup import ensure_installed, ensure_running

            if not ensure_installed():
                print("  ERROR: Failed to install Ollama", file=sys.stderr)
                return 1
            if not ensure_running():
                print("  ERROR: Failed to start Ollama", file=sys.stderr)
                return 1
            endpoint = "http://localhost:11434/v1"
            # Preserve the model if it was ollama before, else use default
            model_name = cfg_dict.get("ollama_model", "gemma4:e2b")

        elif target == "llamacpp":
            from adk import llamacpp_setup

            result = llamacpp_setup.install(quant=None, port=8200, service=True)
            if not result.success:
                print(f"  ERROR: {result.error}", file=sys.stderr)
                return 1
            endpoint = f"http://localhost:{result.port}/v1"
            model_name = result.quant

        elif target == "vllm":
            from adk.setup_cli import cmd_setup

            class SetupArgs:
                shortcut = None
                tier = None
                mode = "auto"
                reasoning_api = None
                reasoning_model = ""
                dgx_spark = None
                stack = None
                dry_run = False
                non_interactive = True
                hf_token = ""
                api_key = ""
                output = "docker-compose.vllm.yml"

            setup_result = cmd_setup(SetupArgs())
            if setup_result != 0:
                print("  ERROR: vLLM setup failed", file=sys.stderr)
                return 1
            endpoint = "http://localhost:8200/v1"
            model_name = "auto"
        else:
            print(f"  ERROR: unknown backend {target}", file=sys.stderr)
            return 1

        # Step 2: Verify with smoke test
        print()
        print("  [2/3] Verifying endpoint...")
        time.sleep(2)  # Give service time to start

        if target in ("llamacpp", "vllm"):
            from adk import llamacpp_setup

            verify_port = 8200 if target == "vllm" else 8200
            for _ in range(30):
                if llamacpp_setup.status(port=verify_port).running:
                    break
                time.sleep(1)

            ok = llamacpp_setup.smoke_test(port=verify_port)
            if not ok:
                print(
                    "  ERROR: Smoke test failed - the endpoint did not return a valid "
                    "completion. Check logs and re-run.",
                    file=sys.stderr,
                )
                return 1
        elif target == "ollama":
            from adk.ollama_setup import smoke_test as ollama_smoke_test

            ok = ollama_smoke_test(model_name)
            if not ok:
                print(
                    "  ERROR: Smoke test failed - the endpoint did not respond. "
                    "Check logs and re-run.",
                    file=sys.stderr,
                )
                return 1

        print(f"  Endpoint ready: {endpoint}")

        # Step 3: Persist configuration
        print()
        print("  [3/3] Saving configuration...")
        save_saved_config({
            "setup_backend": target,
            "inference_url": endpoint,
            "inference_model": model_name,
        })
        if target == "ollama":
            save_saved_config({"ollama_model": model_name})

        print()
        print("=" * 60)
        print("  Backend Switched Successfully")
        print("=" * 60)
        print()
        print(f"  Backend:  {target}")
        print(f"  Endpoint: {endpoint}")
        print(f"  Model:    {model_name}")
        print()
        return 0

    elif sub == "test":
        async def _test():
            from adk.llm import LLMRouter
            from adk.config import Config
            cfg = Config.from_env()
            router = LLMRouter(config=cfg)
            try:
                await router.get_provider()
                print(f"Provider: {router.provider_name}")
                resp = await router.chat(
                    [{"role": "user", "content": "Say 'hello' in one word."}],
                    effort=3,
                )
                print(f"Model: {resp.model}")
                print(f"Response: {resp.content[:100]}")
                print(f"Tokens: {resp.tokens_used}")
                print("Status: OK")
            except Exception as e:
                print(f"FAILED: {e}")
                return 1
            return 0
        return asyncio.run(_test())

    print("Usage: adk backend [list|set|set-reasoning|test|switch|status]")
    return 1


def cmd_install(args) -> int:
    """Install ready-made agent packs into user workspace.

    Accepts `adk install` / `adk install list` (list packs), and
    `adk install pack:<name>` or `adk install <name>` (install a pack).
    """
    # Resolve the requested target across the CLI positional and the legacy
    # subcommand arg shape still used by tests.
    target = getattr(args, "target", None)
    if target is None:
        legacy = getattr(args, "install_command", None)
        if legacy == "pack":
            target = getattr(args, "pack_name", None)
        elif legacy:
            target = legacy

    # Get available packs from the bundled packs directory
    packs_dir = Path(__file__).parent / "packs"
    available_packs = {}
    if packs_dir.exists():
        for pack_dir in packs_dir.iterdir():
            if pack_dir.is_dir():
                agent_yaml = pack_dir / "agent.yaml"
                if agent_yaml.exists():
                    try:
                        import yaml
                        data = yaml.safe_load(agent_yaml.read_text(encoding="utf-8")) or {}
                        available_packs[pack_dir.name] = {
                            "path": pack_dir,
                            "name": data.get("name", pack_dir.name),
                            "description": data.get("description", ""),
                            "identity": data.get("identity", pack_dir.name),
                        }
                    except Exception:
                        pass

    if target is None or target == "list":
        """List available bundled packs."""
        if not available_packs:
            print("No agent packs available.")
            return 0

        print("\n  Available Agent Packs")
        print("  " + "=" * 50)
        print()
        for pack_id, info in sorted(available_packs.items()):
            print(f"  {pack_id:<20s} {info['name']}")
            if info["description"]:
                desc = info["description"].split("\n")[0]
                print(f"  {'':20s} {desc[:50]}")
            print()
        print(f"  Install with: adk install pack:<name>")
        print()
        return 0

    # Install mode — accept 'pack:<name>' or a bare '<name>'.
    pack_name = target[len("pack:"):] if target.startswith("pack:") else target
    pack_name = (pack_name or "").strip()
    if target is not None:
        if not pack_name:
            print("Usage: adk install pack:<name>")
            print()
            print("Available packs:")
            for pid in sorted(available_packs.keys()):
                print(f"  - {pid}")
            return 1

        if pack_name not in available_packs:
            print(f"ERROR: Unknown pack '{pack_name}'")
            print()
            print("Available packs:")
            for pid in sorted(available_packs.keys()):
                print(f"  - {pid}")
            return 1

        pack_info = available_packs[pack_name]
        pack_src = pack_info["path"]

        # Determine install location
        install_base = Path.home() / ".aither" / "agents"
        install_dst = install_base / pack_name

        print()
        print(f"  Installing {pack_name}...")
        print("  " + "=" * 50)
        print()

        # Create install directory
        try:
            install_base.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  ERROR: Could not create directory {install_base}: {e}")
            return 1

        # Copy pack files
        try:
            import shutil
            if install_dst.exists():
                print(f"  Removing existing installation at {install_dst}")
                shutil.rmtree(install_dst)
            shutil.copytree(pack_src, install_dst)
            print(f"  Installed to: {install_dst}")
        except Exception as e:
            print(f"  ERROR: Failed to install pack: {e}")
            return 1

        # Print instructions
        print()
        print("  Next steps:")
        print(f"    adk run --agents {pack_name}      # Run this agent")
        print(f"    adk serve --agents {pack_name}    # Serve as a web service")
        print()
        return 0

    print("Usage: adk install [list|pack:<name>]")
    return 1


def cmd_packs(args) -> int:
    """Alias for 'adk install list' — list available agent packs."""
    args.install_command = "list"
    return cmd_install(args)


# ── Phase 1: aither keys — API key management ─────────────────────────────

_KNOWN_PROVIDERS = {
    "openai": {"env": "OPENAI_API_KEY", "test_url": "https://api.openai.com/v1/models", "label": "OpenAI"},
    "anthropic": {"env": "ANTHROPIC_API_KEY", "test_url": "https://api.anthropic.com/v1/messages", "label": "Anthropic"},
    "deepseek": {"env": "DEEPSEEK_API_KEY", "test_url": "https://api.deepseek.com/v1/models", "label": "DeepSeek"},
    "google": {"env": "GOOGLE_API_KEY", "test_url": "https://generativelanguage.googleapis.com/v1/models", "label": "Google AI"},
    "openrouter": {"env": "OPENROUTER_API_KEY", "test_url": "https://openrouter.ai/api/v1/models", "label": "OpenRouter"},
    "groq": {"env": "GROQ_API_KEY", "test_url": "https://api.groq.com/openai/v1/models", "label": "Groq"},
    "together": {"env": "TOGETHER_API_KEY", "test_url": "https://api.together.xyz/v1/models", "label": "Together AI"},
}


def _keys_path() -> Path:
    """Path to local provider key store."""
    d = Path.home() / ".aither"
    d.mkdir(parents=True, exist_ok=True)
    return d / "provider_keys.json"


def _load_provider_keys() -> dict:
    p = _keys_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_provider_keys(data: dict) -> None:
    p = _keys_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    # Restrict permissions on Unix
    if sys.platform != "win32":
        os.chmod(p, 0o600)


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


def _test_provider_key(provider: str, key: str) -> tuple:
    """Test a provider key with a minimal API call. Returns (success, message)."""
    import urllib.request
    import urllib.error

    info = _KNOWN_PROVIDERS.get(provider)
    if not info:
        return False, f"Unknown provider: {provider}"

    test_url = info["test_url"]
    headers = {"Content-Type": "application/json"}

    if provider == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
        # Anthropic needs a POST to /messages with minimal payload to validate
        try:
            payload = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=15):
                return True, "Key valid"
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "Invalid API key"
            if e.code in (400, 429):
                return True, "Key valid (rate limited or minimal request rejected)"
            return False, f"HTTP {e.code}"
        except Exception as e:
            return False, str(e)
    else:
        headers["Authorization"] = f"Bearer {key}"
        try:
            req = urllib.request.Request(test_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15):
                return True, "Key valid"
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return False, "Invalid API key"
            if e.code == 403:
                return False, "Key forbidden (check permissions)"
            if e.code == 429:
                return True, "Key valid (rate limited)"
            return False, f"HTTP {e.code}"
        except Exception as e:
            return False, str(e)


def _push_key_to_vault(provider: str, key: str) -> bool:
    """Push key to AitherOS via the BYOK endpoint (tenant-scoped) or Secrets vault.

    Tries the proper tenant LLM keys endpoint first (scoped to tenant),
    falls back to raw AitherSecrets (platform-level).
    """
    import urllib.request
    import urllib.error

    # Try 1: Genesis BYOK endpoint (properly scoped to tenant)
    genesis_url = os.environ.get("AITHER_URL", "http://localhost:8001")
    try:
        payload = json.dumps({"api_key": key}).encode()
        headers = {"Content-Type": "application/json"}
        # Forward tenant context if available
        saved = load_saved_config()
        if saved.get("tenant_id"):
            headers["X-Tenant-ID"] = saved["tenant_id"]
        if saved.get("api_key"):
            headers["Authorization"] = f"Bearer {saved['api_key']}"
        req = urllib.request.Request(
            f"{genesis_url}/tenants/me/llm-keys/{provider}",
            data=payload, headers=headers, method="PUT",
        )
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        pass

    # Try 2: Direct AitherSecrets (platform-level fallback)
    info = _KNOWN_PROVIDERS.get(provider, {})
    env_name = info.get("env", f"{provider.upper()}_API_KEY")
    try:
        payload = json.dumps({
            "name": env_name, "value": key, "secret_type": "api_key",
        }).encode()
        req = urllib.request.Request(
            "http://localhost:8111/api/secrets",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def cmd_keys(args):
    """Manage cloud provider API keys."""
    import json as _json

    sub = getattr(args, "keys_command", None)

    if sub == "set":
        provider = getattr(args, "provider", "").lower()
        key = getattr(args, "key", "")
        if not provider or not key:
            print("Usage: adk keys set <provider> <key>")
            return 1
        if provider not in _KNOWN_PROVIDERS:
            print(f"Unknown provider: {provider}")
            print(f"Known: {', '.join(sorted(_KNOWN_PROVIDERS))}")
            return 1

        keys = _load_provider_keys()
        keys[provider] = key
        _save_provider_keys(keys)

        # Set env var for current session
        env_name = _KNOWN_PROVIDERS[provider]["env"]
        os.environ[env_name] = key
        print(f"  Saved {_KNOWN_PROVIDERS[provider]['label']} key: {_mask_key(key)}")

        # Try to push to vault
        if _push_key_to_vault(provider, key):
            print("  Synced to AitherSecrets vault")
        return 0

    elif sub == "list":
        keys = _load_provider_keys()
        print("\nCloud Provider API Keys")
        print("=" * 55)
        for pname, info in sorted(_KNOWN_PROVIDERS.items()):
            key = keys.get(pname, "") or os.environ.get(info["env"], "")
            if key:
                status = f"{_mask_key(key)}"
                print(f"  [+] {info['label']:15s} {status}")
            else:
                print(f"  [-] {info['label']:15s} not configured")
        configured = sum(1 for p in _KNOWN_PROVIDERS if keys.get(p) or os.environ.get(_KNOWN_PROVIDERS[p]["env"]))
        print(f"\n  {configured}/{len(_KNOWN_PROVIDERS)} providers configured")
        return 0

    elif sub == "test":
        provider = getattr(args, "provider", None)
        keys = _load_provider_keys()

        providers_to_test = [provider] if provider else list(_KNOWN_PROVIDERS.keys())
        print("\nTesting API Keys")
        print("=" * 50)

        for pname in providers_to_test:
            if pname not in _KNOWN_PROVIDERS:
                print(f"  [?] {pname}: unknown provider")
                continue
            info = _KNOWN_PROVIDERS[pname]
            key = keys.get(pname, "") or os.environ.get(info["env"], "")
            if not key:
                print(f"  [-] {info['label']:15s} no key configured")
                continue
            ok, msg = _test_provider_key(pname, key)
            icon = "+" if ok else "x"
            print(f"  [{icon}] {info['label']:15s} {msg}")
        return 0

    elif sub == "remove":
        provider = getattr(args, "provider", "").lower()
        if not provider:
            print("Usage: adk keys remove <provider>")
            return 1
        keys = _load_provider_keys()
        if provider in keys:
            del keys[provider]
            _save_provider_keys(keys)
            print(f"  Removed {provider} key")
        else:
            print(f"  No key stored for {provider}")
        return 0

    else:
        # Interactive mode — prompt for each provider
        print("\n  Cloud Provider API Key Setup")
        print("  " + "=" * 40)
        print("  Enter API keys for your cloud providers.")
        print("  Press Enter to skip a provider.\n")

        keys = _load_provider_keys()
        changed = False

        for pname, info in _KNOWN_PROVIDERS.items():
            existing = keys.get(pname, "") or os.environ.get(info["env"], "")
            if existing:
                status = f"(configured: {_mask_key(existing)})"
            else:
                status = "(not set)"

            try:
                val = input(f"  {info['label']} API key {status}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if val:
                keys[pname] = val
                os.environ[info["env"]] = val
                changed = True
                ok, msg = _test_provider_key(pname, val)
                icon = "+" if ok else "x"
                print(f"    [{icon}] {msg}")
                if _push_key_to_vault(pname, val):
                    print("    Synced to vault")

        if changed:
            _save_provider_keys(keys)
            print(f"\n  Keys saved to {_keys_path()}")

        configured = sum(1 for p in _KNOWN_PROVIDERS if keys.get(p) or os.environ.get(_KNOWN_PROVIDERS[p]["env"]))
        print(f"\n  {configured}/{len(_KNOWN_PROVIDERS)} providers ready")
        return 0


# ── Phase 3: aither routing — inference routing management ─────────────────

def _routing_config_path() -> Path:
    """Find inference_routing.yaml — local ~/.aither/ first, then AitherOS config."""
    local = Path.home() / ".aither" / "inference_routing.yaml"
    if local.exists():
        return local
    # Check AitherOS config
    aitheros = Path(os.environ.get("AITHER_ROOT", "")) / "config" / "inference_routing.yaml"
    if aitheros.exists():
        return aitheros
    # Check relative to this file (ADK might be inside repo)
    for candidate in [
        Path(__file__).resolve().parents[2] / "AitherOS" / "config" / "inference_routing.yaml",
        Path.cwd() / "AitherOS" / "config" / "inference_routing.yaml",
        Path.cwd() / "config" / "inference_routing.yaml",
    ]:
        if candidate.exists():
            return candidate
    return local  # default write location


def _load_routing_config() -> dict:
    import yaml
    p = _routing_config_path()
    if p.exists():
        try:
            with open(p) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {"enabled": False, "intent_routing": {}, "presets": {}}


def _save_routing_config(cfg: dict) -> None:
    import yaml
    p = _routing_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


# Map short intent aliases to config keys
_INTENT_ALIASES = {
    "code": "code_generation",
    "coding": "code_generation",
    "review": "code_review",
    "reasoning": "deep_analysis",
    "analysis": "deep_analysis",
    "planning": "complex_planning",
    "chat": "casual_chat",
    "question": "simple_question",
    "research": "research",
    "search": "web_search",
}

# Map short provider aliases
_PROVIDER_ALIASES = {
    "claude": "anthropic",
    "gpt": "openai",
    "ds": "deepseek",
    "local": "local",
}


def cmd_routing(args):
    """Manage per-intent model routing."""
    sub = getattr(args, "routing_command", None)

    if sub == "preset":
        preset_name = getattr(args, "preset_name", "")
        if not preset_name:
            print("Usage: adk routing preset <budget|balanced|quality>")
            return 1

        cfg = _load_routing_config()
        presets = cfg.get("presets", {})
        if preset_name not in presets:
            print(f"Unknown preset: {preset_name}")
            print(f"Available: {', '.join(presets.keys())}")
            return 1

        preset = presets[preset_name]
        desc = preset.get("description", "")
        print(f"\n  Applying preset: {preset_name}")
        if desc:
            print(f"  {desc}")

        # Apply preset to intent routing
        routing = cfg.get("intent_routing", {})
        dp = preset.get("default_provider", "local")
        df = preset.get("default_fallback", "")
        rp = preset.get("reasoning_provider", "")
        rm = preset.get("reasoning_model", "")

        # Update quick intents to use default provider
        for intent in ("casual_chat", "simple_question", "web_search"):
            if intent not in routing:
                routing[intent] = {}
            routing[intent]["provider"] = dp
            if df:
                routing[intent]["fallback_provider"] = df

        # Update reasoning intents
        for intent in ("deep_analysis", "complex_planning"):
            if rp:
                if intent not in routing:
                    routing[intent] = {}
                routing[intent]["provider"] = rp
                if rm:
                    routing[intent]["model"] = rm

        cfg["intent_routing"] = routing
        cfg["enabled"] = True
        _save_routing_config(cfg)

        # Also save to ADK config for backend selection
        save_data = {"routing_preset": preset_name}
        if rp and rm:
            save_data["reasoning_backend"] = rp
            save_data["reasoning_model"] = rm
        save_saved_config(save_data)

        print(f"  Routing preset '{preset_name}' applied and enabled")
        return 0

    elif sub == "set":
        intent_alias = getattr(args, "intent", "")
        provider_alias = getattr(args, "provider", "")
        model = getattr(args, "model", None)
        if not intent_alias or not provider_alias:
            print("Usage: adk routing set <intent> <provider> [--model MODEL]")
            return 1

        intent_key = _INTENT_ALIASES.get(intent_alias, intent_alias)
        provider_key = _PROVIDER_ALIASES.get(provider_alias, provider_alias)

        cfg = _load_routing_config()
        routing = cfg.get("intent_routing", {})
        if intent_key not in routing:
            routing[intent_key] = {}
        routing[intent_key]["provider"] = provider_key
        if model:
            routing[intent_key]["model"] = model

        cfg["intent_routing"] = routing
        cfg["enabled"] = True
        _save_routing_config(cfg)
        print(f"  {intent_key} -> {provider_key}" + (f" ({model})" if model else ""))
        return 0

    elif sub == "reset":
        cfg = _load_routing_config()
        cfg["enabled"] = False
        _save_routing_config(cfg)
        print("  Intent routing disabled — using effort-based defaults")
        return 0

    else:
        # Show current routing
        cfg = _load_routing_config()
        enabled = cfg.get("enabled", False)
        routing = cfg.get("intent_routing", {})

        print(f"\nInference Routing {'(ACTIVE)' if enabled else '(disabled — effort-based)'}")
        print("=" * 55)

        if not routing:
            print("  No intent overrides configured.")
            print("  Use 'adk routing preset balanced' to get started.")
        else:
            for intent, config in sorted(routing.items()):
                provider = config.get("provider", "?")
                model = config.get("model", "")
                fb = config.get("fallback_provider", "")
                line = f"  {intent:25s} -> {provider}"
                if model:
                    line += f" ({model})"
                if fb:
                    line += f"  [fallback: {fb}]"
                print(line)

        presets = cfg.get("presets", {})
        if presets:
            print(f"\n  Presets: {', '.join(presets.keys())}")
        print(f"\n  Config: {_routing_config_path()}")
        return 0


# ── Phase 3.5: adk grid — manage grid distributed infrastructure ──────────

_GRID_CONFIG_STRATA_PATH = "grid/config.json"


def cmd_grid(args) -> int:
    """Manage grid distributed inference nodes."""
    import asyncio

    sub = getattr(args, "grid_command", None)
    saved = load_saved_config()

    # Grid config lives under a "grid_nodes" key in ~/.aither/config.json
    grid_nodes = saved.get("grid_nodes", {})
    # grid_nodes = { "reasoning": {"host": "...", "port": 8121, "model": "..."},
    #                "cluster": [{"host": "...", "port": 8121, "model": "..."}] }

    if sub == "status" or sub is None:
        print()
        print("  Grid Topology")
        print("  " + "=" * 55)

        # Primary (local GPU)
        backend = saved.get("backend", "")
        base_url = saved.get("base_url", "")
        model = saved.get("model", "")
        if backend:
            print(f"  [GPU]     {base_url or 'auto-detect':30s}  {model or 'auto'}")
        else:
            print("  [GPU]     not configured (run: adk deploy grid)")

        # Reasoning node
        r_node = grid_nodes.get("reasoning")
        r_url = saved.get("reasoning_url", "")
        r_model = saved.get("reasoning_model", "")
        if r_node:
            r_display = f"{r_node['host']}:{r_node.get('port', 8121)}"
            print(f"  [reason]  {r_display:30s}  {r_node.get('model', r_model or 'auto')}")
        elif r_url:
            print(f"  [reason]  {r_url:30s}  {r_model or 'auto'}")
        else:
            print("  [reason]  not configured (run: adk grid add reasoning <ip>)")

        # Cluster nodes
        c_nodes = grid_nodes.get("cluster", [])
        c_url = saved.get("cluster_url", "")
        c_model = saved.get("cluster_model", "")
        if c_nodes:
            for i, node in enumerate(c_nodes):
                c_display = f"{node['host']}:{node.get('port', 8121)}"
                print(f"  [cpu.{i}]   {c_display:30s}  {node.get('model', c_model or 'auto')}")
        elif c_url:
            print(f"  [cpu.0]   {c_url:30s}  {c_model or 'auto'}")
        else:
            print("  [cpu]     not configured (run: adk grid add cluster <ip>)")

        # Auth status
        print()
        api_key = saved.get("api_key", "")
        tenant = saved.get("tenant_id", "")
        username = saved.get("username", "")
        if api_key:
            print(f"  Auth:     {username or 'logged in'} (tenant: {tenant or 'default'})")
            print("  Sync:     adk grid sync → portal.aitherium.com")
        else:
            print("  Account:  none (everything works locally without one)")
            print("  Optional: adk login → free account, enables config sync across machines")
            print("            https://portal.aitherium.com/signup")

        # Health check all nodes
        print()
        print("  Health")
        print("  " + "-" * 55)
        _grid_health_check(saved, grid_nodes)

        print()
        return 0

    elif sub == "add":
        role = args.role
        host = args.host
        port = getattr(args, "port", 8121) or 8121
        model_override = getattr(args, "model", None)

        node_entry = {"host": host, "port": port}
        if model_override:
            node_entry["model"] = model_override

        if role == "reasoning":
            grid_nodes["reasoning"] = node_entry
            # Also update the flat config for LLM router
            update = {
                "reasoning_backend": "openai",
                "reasoning_url": f"http://{host}:{port}/v1",
                "reasoning_model": model_override or "deepseek-r1-8b",
                "grid_nodes": grid_nodes,
            }
        else:  # cluster
            cluster_list = grid_nodes.get("cluster", [])
            # Deduplicate by host
            cluster_list = [n for n in cluster_list if n["host"] != host]
            cluster_list.append(node_entry)
            grid_nodes["cluster"] = cluster_list
            # Use the first cluster node for routing
            first = cluster_list[0]
            update = {
                "cluster_backend": "openai",
                "cluster_url": f"http://{first['host']}:{first.get('port', 8121)}/v1",
                "cluster_model": model_override or "qwen2.5-32b",
                "grid_nodes": grid_nodes,
            }

        save_saved_config(update)
        print(f"  Added {role} node: {host}:{port}")

        # Quick health check
        _grid_test_node(host, port)

        print("\n  Config saved to ~/.aither/config.json")
        print("  Sync to cloud: adk grid sync")
        return 0

    elif sub == "remove":
        host = args.host
        removed = False

        r_node = grid_nodes.get("reasoning")
        if r_node and r_node.get("host") == host:
            del grid_nodes["reasoning"]
            save_saved_config({
                "reasoning_backend": "",
                "reasoning_url": "",
                "reasoning_model": "",
                "grid_nodes": grid_nodes,
            })
            removed = True
            print(f"  Removed reasoning node: {host}")

        c_nodes = grid_nodes.get("cluster", [])
        new_cluster = [n for n in c_nodes if n.get("host") != host]
        if len(new_cluster) < len(c_nodes):
            grid_nodes["cluster"] = new_cluster
            update = {"grid_nodes": grid_nodes}
            if new_cluster:
                first = new_cluster[0]
                update["cluster_url"] = f"http://{first['host']}:{first.get('port', 8121)}/v1"
            else:
                update["cluster_backend"] = ""
                update["cluster_url"] = ""
                update["cluster_model"] = ""
            save_saved_config(update)
            removed = True
            print(f"  Removed cluster node: {host}")

        if not removed:
            print(f"  No node found with host: {host}")
            return 1
        return 0

    elif sub == "test":
        target = getattr(args, "host", None)
        print()
        print("  Grid Node Tests")
        print("  " + "=" * 50)
        _grid_health_check(saved, grid_nodes, target_host=target)
        print()
        return 0

    elif sub == "sync":
        api_key = saved.get("api_key", "")
        tenant = saved.get("tenant_id", "")
        if not api_key:
            print("  Not logged in. Run: adk login")
            print("  Grid sync requires authentication to store config in your workspace.")
            return 1

        grid_data = {
            "profile": saved.get("profile", ""),
            "backend": saved.get("backend", ""),
            "base_url": saved.get("base_url", ""),
            "model": saved.get("model", ""),
            "reasoning_backend": saved.get("reasoning_backend", ""),
            "reasoning_url": saved.get("reasoning_url", ""),
            "reasoning_model": saved.get("reasoning_model", ""),
            "cluster_backend": saved.get("cluster_backend", ""),
            "cluster_url": saved.get("cluster_url", ""),
            "cluster_model": saved.get("cluster_model", ""),
            "grid_nodes": grid_nodes,
        }

        async def _sync():
            from adk.strata import get_strata
            strata = get_strata()
            import json as _json
            ok = await strata.write(
                _GRID_CONFIG_STRATA_PATH,
                _json.dumps(grid_data, indent=2),
            )
            if ok:
                print(f"  Grid config synced to workspace (tenant: {tenant or 'default'})")
                print("  Pull on another machine: adk grid pull")
            else:
                # Fallback: try direct gateway API
                try:
                    import httpx
                    gateway = saved.get("gateway_url", "https://gateway.aitherium.com")
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.put(
                            f"{gateway}/api/v1/config/grid",
                            json=grid_data,
                            headers={"Authorization": f"Bearer {api_key}"},
                        )
                        if resp.status_code in (200, 201):
                            print("  Grid config synced via gateway")
                            return
                except Exception:
                    pass
                print("  Sync failed — Strata not available and gateway unreachable.")
                print("  Config is saved locally at ~/.aither/config.json")

        asyncio.run(_sync())
        return 0

    elif sub == "pull":
        api_key = saved.get("api_key", "")
        if not api_key:
            print("  Not logged in. Run: adk login")
            return 1

        async def _pull():
            import json as _json
            from adk.strata import get_strata
            strata = get_strata()
            data = await strata.read_text(_GRID_CONFIG_STRATA_PATH)
            if data:
                grid_data = _json.loads(data)
                save_saved_config(grid_data)
                print("  Grid config pulled from workspace and saved locally.")
                print("  Run: adk grid status")
                return

            # Fallback: try gateway API
            try:
                import httpx
                gateway = saved.get("gateway_url", "https://gateway.aitherium.com")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{gateway}/api/v1/config/grid",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code == 200:
                        grid_data = resp.json()
                        save_saved_config(grid_data)
                        print("  Grid config pulled from gateway and saved locally.")
                        print("  Run: adk grid status")
                        return
            except Exception:
                pass
            print("  No grid config found in workspace. Run: adk grid sync (from configured machine)")

        asyncio.run(_pull())
        return 0

    elif sub == "enroll":
        ttl = max(60, min(int(getattr(args, "ttl", 1.0) * 3600), 86400))
        body = {"ttl_seconds": ttl, "label": getattr(args, "label", ""), "tenant": getattr(args, "tenant", "")}
        ok, data = _grid_mesh_request("POST", "/nodes/enroll-token", body=body, need_key=True)
        if not ok:
            print(f"  {data}")
            return 1
        tok = data.get("enroll_token", "")
        install = data.get("install") or f'AITHER_NODE_TOKEN={tok} sh -c "$(curl -fsSL https://cluster.aitherium.com/install.sh)"'
        tslug = getattr(args, "tenant", "")
        print()
        print(f"  Enrollment token minted (expires in {ttl // 60} min, single-use)"
              + (f", tenant={tslug}" if tslug else ""))
        print("\n  Run this on the target node:")
        print(f"    {install}")
        print("\n  It registers through the tunnel, installs a reboot-safe service, and starts heartbeating.")
        print()
        return 0

    elif sub == "ls":
        ok, data = _grid_mesh_request("GET", "/nodes")
        if not ok:
            print(f"  {data}")
            return 1
        nodes = data.get("nodes", [])
        if not nodes:
            print("  No mesh nodes registered.")
            return 0
        print()
        print("  Mesh Nodes")
        print("  " + "=" * 55)
        for n in nodes:
            hw = n.get("hardware") or {}
            dot = "online " if n.get("status") == "online" else "offline"
            gpu = hw.get("gpu") or "—"
            mem = hw.get("memory_gb")
            mem_s = f"{round(mem)}GB" if isinstance(mem, (int, float)) else "—"
            ctr = len(n.get("services") or n.get("containers") or [])
            tenant = n.get("tenant") or (n.get("labels") or {}).get("tenant") or ""
            name = n.get("name") or n.get("node_id") or "?"
            line = f"  [{dot}] {name:18s} {str(gpu):14s} {mem_s:>6s}  {ctr} ctr"
            if tenant:
                line += f"  tenant={tenant}"
            print(line)
        print()
        return 0

    elif sub == "deregister":
        node_id = args.node_id
        ok, data = _grid_mesh_request("DELETE", f"/nodes/{node_id}", need_key=True)
        if not ok:
            print(f"  {data}")
            return 1
        print(f"  Removed {data.get('name') or node_id} ({data.get('remaining', '?')} remaining)")
        return 0

    else:
        print()
        print("  adk grid — Manage distributed inference nodes")
        print()
        print("  Commands:")
        print("    adk grid status              Show topology + health")
        print("    adk grid add reasoning <ip>  Add Mac/reasoning node")
        print("    adk grid add cluster <ip>    Add CPU cluster node")
        print("    adk grid remove <ip>         Remove a node")
        print("    adk grid test                Test all nodes")
        print("    adk grid test <ip>           Test specific node")
        print("    adk grid sync                Push config to your Aitherium workspace")
        print("    adk grid pull                Pull config from workspace (new machine)")
        print()
        print("  Mesh registry (enrolled nodes via the gateway tunnel):")
        print("    adk grid ls                  List enrolled mesh nodes")
        print("    adk grid enroll [--tenant]   Mint a token to onboard a remote node")
        print("    adk grid deregister <id>     Remove a node from the registry")
        print()
        print("  Setup:")
        print("    adk deploy grid              Deploy vLLM + configure grid")
        print("    adk login                    Auth for cloud sync")
        print()
        return 0


def _grid_mesh_request(method: str, path: str, body: dict | None = None, need_key: bool = False):
    """Call the AitherGateway node registry through the public Cloudflare tunnel.

    GET /nodes is open; minting/removing are internal-key gated (need_key=True,
    sourced from AITHER_INTERNAL_SECRET/AITHER_MASTER_KEY). Returns (ok, data|msg).
    Uses a browser UA — Cloudflare 403s the default python-urllib agent.
    """
    import json as _json
    import os as _os
    import ssl as _ssl
    import urllib.request as _ur

    gateway = (_os.environ.get("AITHER_NODE_GATEWAY_URL")
               or _os.environ.get("AITHER_PUBLIC_CLUSTER_URL")
               or "https://cluster.aitherium.com").rstrip("/")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AitherADK/1.0)",
        "Content-Type": "application/json",
    }
    if need_key:
        key = (_os.environ.get("AITHER_INTERNAL_SECRET") or _os.environ.get("AITHER_MASTER_KEY") or "").strip()
        if not key:
            return False, "Admin key required: set AITHER_INTERNAL_SECRET to mint or remove nodes."
        headers["X-API-Key"] = key
    data = _json.dumps(body).encode() if body is not None else None
    req = _ur.Request(f"{gateway}{path}", data=data, method=method, headers=headers)
    ctx = _ssl.create_default_context()
    if gateway.startswith("https://aitheros-") or "localhost" in gateway or "127.0.0.1" in gateway:
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
    try:
        with _ur.urlopen(req, context=ctx, timeout=15) as resp:
            return True, _json.loads(resp.read().decode() or "{}")
    except _ur.HTTPError as e:  # noqa: PERF203
        if e.code == 404:
            return False, f"Not found: {path.rsplit('/', 1)[-1]}"
        return False, f"Gateway returned {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, f"Gateway unreachable: {str(e)[:120]}"


def _grid_test_node(host: str, port: int) -> bool:
    """Test connectivity and API compatibility of a single grid node."""
    from urllib.request import Request, urlopen

    try:
        req = Request(
            f"http://{host}:{port}/health",
            headers={"User-Agent": "AitherADK/1.0"},
        )
        with urlopen(req, timeout=5):
            pass
    except Exception:
        print(f"  [x] {host}:{port} — unreachable")
        return False

    try:
        req = Request(
            f"http://{host}:{port}/v1/models",
            headers={"User-Agent": "AitherADK/1.0"},
        )
        with urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                import json as _json
                data = _json.loads(resp.read())
                models = [m.get("id", "") for m in data.get("data", [])]
                print(f"  [+] {host}:{port} — healthy, models: {', '.join(models[:3]) or 'default'}")
                return True
    except Exception:
        print(f"  [!] {host}:{port} — healthy but no /v1 API (missing --api-oai?)")
        return False

    return False


def _grid_health_check(saved: dict, grid_nodes: dict, target_host: str | None = None):
    """Run health checks on all or a specific grid node."""
    checked = False

    # Reasoning node
    r_node = grid_nodes.get("reasoning")
    if r_node and (target_host is None or target_host == r_node.get("host")):
        _grid_test_node(r_node["host"], r_node.get("port", 8121))
        checked = True

    # Cluster nodes
    for node in grid_nodes.get("cluster", []):
        if target_host is None or target_host == node.get("host"):
            _grid_test_node(node["host"], node.get("port", 8121))
            checked = True

    # Fallback: check flat config URLs if no grid_nodes
    if not checked and not target_host:
        r_url = saved.get("reasoning_url", "")
        if r_url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(r_url)
                host = parsed.hostname or ""
                port = parsed.port or 8121
                if host:
                    _grid_test_node(host, port)
            except Exception:
                pass

        c_url = saved.get("cluster_url", "")
        if c_url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(c_url)
                host = parsed.hostname or ""
                port = parsed.port or 8121
                if host:
                    _grid_test_node(host, port)
            except Exception:
                pass

    if not checked and target_host:
        print(f"  No node found with host: {target_host}")


# ── Phase 3.6: adk explore — marketplace browser ─────────────────────────


def cmd_explore(args) -> int:
    """Browse packs, agents, and skills in the Aitherium marketplace."""
    genesis_url = _get_genesis_url()
    category = getattr(args, "category", "all").lower()
    free_only = getattr(args, "free", False)

    catalog = _load_pack_catalog(genesis_url)

    if not catalog:
        print("\n  No catalog available. Install aither-adk or connect to Genesis.\n")
        return 1

    # Filter
    filtered = catalog
    if category == "agents":
        filtered = [p for p in catalog if p.get("type") == "agent_pack"]
    elif category == "tools":
        filtered = [p for p in catalog if p.get("type") == "tool_pack"]
    elif category == "skills":
        filtered = [p for p in catalog if p.get("type") == "skill_pack"]
    elif category == "grid":
        filtered = [p for p in catalog if "grid" in p.get("tags", []) or "distributed" in p.get("tags", [])]

    if free_only:
        filtered = [p for p in filtered if p.get("tier") == "free"]

    # Check installed
    packs_dir = Path.home() / ".aitheros" / "packs"
    installed_ids = set()
    if packs_dir.is_dir():
        for child in packs_dir.iterdir():
            if child.is_dir() and (child / ".toolpack.yaml").exists():
                installed_ids.add(child.name)

    # Group by type
    groups: dict[str, list] = {}
    for p in filtered:
        ptype = p.get("type", "other").replace("_pack", "").replace("_", " ").title()
        groups.setdefault(ptype, []).append(p)

    print()
    print("  Aitherium Marketplace")
    print("  " + "=" * 60)

    for gname in sorted(groups):
        packs = groups[gname]
        print(f"\n  {gname} Packs ({len(packs)})")
        print("  " + "-" * 55)
        for p in packs:
            pid = p.get("id", "?")
            name = p.get("name", pid)
            desc = p.get("description", "")[:70]
            tier = p.get("tier", "free")
            installed = pid in installed_ids
            pricing = p.get("pricing", {})

            icon = "[+]" if installed else "[ ]"
            tier_label = tier
            price = ""
            if pricing.get("subscription_cents"):
                price = f"${int(pricing['subscription_cents']) / 100:.0f}/mo"
            elif pricing.get("one_time_cents"):
                price = f"${int(pricing['one_time_cents']) / 100:.0f}"

            status = "installed" if installed else tier_label
            print(f"  {icon} {name}")
            if desc:
                print(f"      {desc}")
            parts = [status]
            if price:
                parts.append(price)
            if p.get("install_command"):
                parts.append(p["install_command"])
            print(f"      {' | '.join(parts)}")

    total = len(filtered)
    free_count = sum(1 for p in filtered if p.get("tier") == "free")
    inst_count = sum(1 for p in filtered if p.get("id") in installed_ids)

    print(f"\n  {total} packs shown ({free_count} free, {inst_count} installed)")
    print()
    print("  Quick actions:")
    print("    adk explore agents          Browse agent packs")
    print("    adk explore tools --free    Free tool packs only")
    print("    adk explore grid            Grid infrastructure")
    print("    adk pack install <id>       Install a pack")
    print("    adk upgrade <id>            Open checkout page")
    print()
    print("  Full catalog: https://portal.aitherium.com/marketplace")
    print()
    return 0


# ── Phase 3.7: adk upgrade — checkout shortcut ──────────────────────────


_UPGRADE_URLS: dict[str, tuple[str, str]] = {
    "managed": ("https://portal.aitherium.com/marketplace/grid?sku=grid_managed_monthly", "Grid Managed ($49/mo)"),
    "setup": ("https://portal.aitherium.com/marketplace/grid?sku=grid_setup_onetime", "Grid Setup Call ($199)"),
    "grid": ("https://portal.aitherium.com/marketplace/grid", "Grid Distributed Inference"),
    "demiurge": ("https://portal.aitherium.com/marketplace/agent.demiurge", "Demiurge — Code Architect"),
    "hydra": ("https://portal.aitherium.com/marketplace/agent.hydra", "Hydra — Code Guardian"),
    "athena": ("https://portal.aitherium.com/marketplace/agent.athena", "Athena — Security Oracle"),
    "lyra": ("https://portal.aitherium.com/marketplace/agent.lyra", "Lyra — Research Muse"),
    "pro": ("https://portal.aitherium.com/pricing", "Professional Plan"),
}


def cmd_upgrade(args) -> int:
    """Open upgrade/checkout page for a pack or plan."""
    target = getattr(args, "target", "").lower().strip()

    if not target:
        print("\n  Upgrade Options\n")
        for key, (url, label) in _UPGRADE_URLS.items():
            print(f"    {key:15s} {label}")
        print()
        print("  Usage: adk upgrade managed")
        print("         adk upgrade demiurge")
        print("         adk upgrade pro")
        print()
        return 0

    if target in _UPGRADE_URLS:
        url, label = _UPGRADE_URLS[target]
    else:
        url = f"https://portal.aitherium.com/marketplace/{target}"
        label = target

    print(f"\n  Opening: {label}")
    print(f"  {url}\n")

    import webbrowser
    try:
        webbrowser.open(url)
    except Exception:
        print("  (Could not open browser — copy the URL above)")

    return 0


# ── Version check (runs once per day, non-blocking) ─────────────────────


def _check_for_updates() -> None:
    """Check PyPI for newer aither-adk version. Runs at most once per day."""
    marker = Path.home() / ".aither" / ".last_update_check"
    now = time.time()

    # Check at most once per day
    if marker.exists():
        try:
            last = float(marker.read_text(encoding="utf-8").strip())
            if now - last < 86400:
                return
        except (ValueError, OSError):
            pass

    try:
        from urllib.request import urlopen
        resp = urlopen("https://pypi.org/pypi/aither-adk/json", timeout=3)
        data = json.loads(resp.read())
        latest = data.get("info", {}).get("version", "")

        # Read current version
        try:
            from importlib.metadata import version as pkg_version
            current = pkg_version("aither-adk")
        except Exception:
            current = ""

        if latest and current and latest != current:
            print(f"\n  Update available: aither-adk {current} -> {latest}")
            print("  Run: pip install --upgrade aither-adk\n")

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(now), encoding="utf-8")
    except Exception:
        pass  # Best effort, never crash


# ── Phase 4: aither costs — token economy visibility ──────────────────────

def cmd_costs(args):
    """Show cloud inference costs and savings."""
    sub = getattr(args, "costs_command", None)

    # Try Genesis API first
    def _genesis_get(path: str) -> dict | None:
        import urllib.request
        import urllib.error
        try:
            url = os.environ.get("AITHER_URL", "http://localhost:8001")
            req = urllib.request.Request(f"{url.rstrip('/')}{path}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    if sub == "budget":
        amount = getattr(args, "amount", 0)
        if amount <= 0:
            print("Usage: adk costs budget <amount_usd>")
            return 1
        result = _genesis_get(f"/costs/budget?monthly_usd={amount}")
        if result:
            print(f"  Monthly budget set to ${amount:.2f}")
        else:
            # Save locally
            budget_path = Path.home() / ".aither" / "cost_budget.json"
            budget_path.parent.mkdir(parents=True, exist_ok=True)
            budget_path.write_text(json.dumps({"monthly_budget_usd": amount}))
            print(f"  Monthly budget set to ${amount:.2f} (local — Genesis not running)")
        return 0

    elif sub == "compare":
        period = getattr(args, "period", "week")
        result = _genesis_get(f"/costs/compare?period={period}")
        if result:
            print(f"\n  Cost Comparison ({result.get('period', period)})")
            print("  " + "=" * 45)
            print(f"  Actual spend:          ${result.get('actual_cost_usd', 0):.4f}")
            print(f"  Est. raw API cost:     ${result.get('estimated_raw_api_cost_usd', 0):.4f}")
            print(f"  Savings:               ${result.get('savings_usd', 0):.4f} ({result.get('savings_percent', 0):.1f}%)")
            print(f"  Local requests:        {result.get('local_requests', 0)}")
            print(f"  Cloud requests:        {result.get('cloud_requests', 0)}")
        else:
            # Read local cost log
            _show_local_costs("compare", getattr(args, "period", "week"))
        return 0

    else:
        # Default: summary
        period = getattr(args, "period", "day")
        result = _genesis_get(f"/costs/summary?period={period}")
        if result:
            print(f"\n  Cost Summary ({result.get('period', period)})")
            print("  " + "=" * 45)
            print(f"  Total spend:     ${result.get('total_cost_usd', 0):.4f}")
            print(f"  Requests:        {result.get('total_requests', 0)} ({result.get('cloud_requests', 0)} cloud, {result.get('local_requests', 0)} local)")
            print(f"  Tokens:          {result.get('total_tokens', 0):,}")

            by_provider = result.get("by_provider", {})
            if by_provider:
                print("\n  By Provider:")
                for prov, cost in by_provider.items():
                    print(f"    {prov:15s} ${cost:.4f}")

            budget = result.get("monthly_budget_usd", 0)
            remaining = result.get("budget_remaining_usd")
            if budget:
                print(f"\n  Budget: ${budget:.2f}/mo  Remaining: ${remaining:.2f}")
        else:
            _show_local_costs("summary", period)
        return 0


def _show_local_costs(mode: str, period: str):
    """Fallback: read local cost JSONL when Genesis is not running."""
    from datetime import datetime, timedelta, timezone

    days_map = {"day": 1, "week": 7, "month": 30}
    days = days_map.get(period, 1)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Search for cost log
    candidates = [
        Path.home() / ".aither" / "cloud_costs.jsonl",
        Path.cwd() / "data" / "cloud_costs.jsonl",
        Path.cwd() / "AitherOS" / "data" / "cloud_costs.jsonl",
    ]
    log_path = None
    for c in candidates:
        if c.exists():
            log_path = c
            break

    if not log_path:
        print("\n  No cost data found (Genesis not running, no local cost log)")
        print("  Costs are tracked when cloud providers are used via AitherOS")
        return

    total = 0.0
    count = 0
    local_count = 0
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = entry.get("timestamp", "")
                    if ts:
                        entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if entry_time < cutoff:
                            continue
                    cost = entry.get("cost_usd", 0) or 0
                    total += cost
                    if entry.get("provider") == "local":
                        local_count += 1
                    else:
                        count += 1
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        pass

    print(f"\n  Cost Summary — {period} (local log)")
    print("  " + "=" * 40)
    print(f"  Total spend:  ${total:.4f}")
    print(f"  Cloud reqs:   {count}")
    print(f"  Local reqs:   {local_count}")


def cmd_tools(args):
    """List available tools (local + MCP)."""
    import asyncio

    async def _tools():
        from adk.tools import ToolRegistry
        from adk.builtin_tools import get_builtin_registry

        # Local built-in tools
        reg = get_builtin_registry()
        local_tools = reg.list_tools()

        print("Local Tools")
        print("=" * 50)
        for t in sorted(local_tools, key=lambda x: x.name):
            desc = (t.description or "")[:60]
            print(f"  {t.name:30s} {desc}")
        print(f"\n  Total: {len(local_tools)} local tools")

        # MCP tools (if connected)
        api_key = os.environ.get("AITHER_API_KEY", "")
        if not api_key:
            saved = load_saved_config()
            api_key = saved.get("api_key", "")

        if api_key:
            try:
                from adk.mcp import MCPBridge
                bridge = MCPBridge(api_key=api_key)
                mcp_tools = await bridge.list_tools()
                print("\nMCP Tools (cloud)")
                print("=" * 50)
                for t in sorted(mcp_tools, key=lambda x: x.get("name", ""))[:20]:
                    name = t.get("name", "?")
                    desc = t.get("description", "")[:50]
                    tier = t.get("tier", "")
                    marker = f" [{tier}]" if tier else ""
                    print(f"  {name:30s} {desc}{marker}")
                print(f"\n  Total: {len(mcp_tools)} MCP tools")
                if getattr(args, "upgrade", False):
                    print("\n  Upgrade at: https://portal.aitherium.com/pricing")
            except Exception as e:
                print(f"\n  MCP: not available ({e})")
        else:
            print("\n  MCP: no API key (run 'adk login' for cloud tools)")

    asyncio.run(_tools())
    return 0


def cmd_backup(args):
    """Backup all ~/.aither/ data."""
    import tarfile
    import time as _time

    data_dir = Path.home() / ".aither"
    if not data_dir.exists():
        print("Nothing to backup — ~/.aither/ does not exist")
        return 1

    ts = _time.strftime("%Y%m%d-%H%M%S")
    output = getattr(args, "output", None) or f"aither-backup-{ts}.tar.gz"
    output_path = Path(output)

    # Count files
    files = list(data_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())

    print(f"Backing up {file_count} files from ~/.aither/")

    with tarfile.open(str(output_path), "w:gz") as tar:
        tar.add(str(data_dir), arcname=".aither")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f}MB)")
    return 0


def cmd_ingest(args):
    """Ingest files into the agent's knowledge graph with optional brain sync."""
    import asyncio

    target = Path(args.path or ".")
    agent_name = getattr(args, "agent", "default")
    brain_sync = getattr(args, "brain", False)
    brain_url = getattr(args, "brain_url", "")
    classification = getattr(args, "classification", "internal")
    chunk_size = getattr(args, "chunk_size", 2000)
    chunk_overlap = getattr(args, "chunk_overlap", 200)
    workspace_id = getattr(args, "workspace", "default")
    skip_embeddings = getattr(args, "skip_embeddings", False)
    dry_run = getattr(args, "dry_run", False)

    async def _ingest():
        from adk.ingest import ingest_files

        if not target.exists():
            print(f"Error: path not found: {target}")
            return 1

        print(f"Ingesting files from {target}...")
        if brain_sync:
            print(f"  Brain sync: ENABLED (workspace: {workspace_id})")
        else:
            print(f"  Brain sync: disabled (local only)")
        print(f"  Classification: {classification}")
        print(f"  Chunk size: {chunk_size} bytes, overlap: {chunk_overlap} bytes")

        result = await ingest_files(
            path=target,
            classification=classification,
            brain_sync=brain_sync,
            brain_url=brain_url,
            workspace_id=workspace_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_embeddings=skip_embeddings,
            dry_run=dry_run,
            agent_name=agent_name,
        )

        # Print summary
        print()
        print("=" * 60)
        print(f"Ingest Summary")
        print("=" * 60)
        print(f"Total files scanned:      {result.files_total}")
        print(f"Files ingested:           {result.files_ingested}")
        print(f"Files skipped:            {result.files_skipped}")
        if result.skipped_files:
            for path, reason in result.skipped_files[:5]:
                print(f"  - {path}: {reason}")
            if len(result.skipped_files) > 5:
                print(f"  ... and {len(result.skipped_files) - 5} more")
        print()
        print(f"Chunks created:           {result.chunks_created}")
        print(f"Chunks embedded:          {result.chunks_embedded}")
        if result.embedding_degraded:
            print(f"  WARNING: Embedding degraded (fallback dimension)")
        print()
        if brain_sync:
            print(f"Brain sync status:        {'SUCCESS' if result.brain_synced else 'FAILED'}")
            print(f"Chunks synced to hub:     {result.chunks_synced}")
        else:
            print(f"Brain sync:               disabled")
        print()
        if result.errors:
            print("Errors encountered:")
            for error in result.errors:
                print(f"  ! {error}")
        print("=" * 60)

        # Return appropriate exit code
        if result.errors and result.files_ingested == 0:
            return 1
        return 0

    return asyncio.run(_ingest())


def cmd_sync(args):
    """AitherDrive — bidirectional file sync with AitherOS platform."""
    import asyncio

    action = getattr(args, "sync_action", None)
    if not action:
        # Default: show status
        action = "status"

    def _load_config():
        """Load auth config from ~/.aither/config.json."""
        cfg_path = Path.home() / ".aither" / "config.json"
        if not cfg_path.exists():
            print("Not logged in. Run `adk login` first.")
            sys.exit(1)
        import json
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    async def _run():
        from adk.sync import SyncManager, SyncManifest, MANIFEST_FILE
        from adk.client.services.strata import StrataClient
        from adk.client.services.data_plane import DataPlaneClient

        if action == "init":
            cfg = _load_config()
            tenant_id = cfg.get("tenant_id", "")
            if not tenant_id:
                print("No tenant_id in config. Run `adk login` first.")
                return 1

            sync_dir = Path(getattr(args, "directory", ".")).resolve()
            if not sync_dir.is_dir():
                print(f"Directory not found: {sync_dir}")
                return 1

            # Check if already initialized
            if (sync_dir / MANIFEST_FILE).exists():
                print(f"Already initialized: {sync_dir / MANIFEST_FILE}")
                return 0

            import httpx
            token = cfg.get("access_token", cfg.get("api_key", ""))
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            if tenant_id:
                headers["X-Tenant-ID"] = tenant_id
            async with httpx.AsyncClient(timeout=30.0, headers=headers) as http:
                async def get_client():
                    return http

                strata_url = cfg.get("strata_url", os.environ.get(
                    "AITHER_STRATA_URL", "http://localhost:8136"))
                dp_url = cfg.get("data_plane_url", os.environ.get(
                    "AITHER_DATAPLANE_URL", "http://localhost:8170"))
                strata = StrataClient(strata_url, get_client)
                data_plane = DataPlaneClient(dp_url, get_client)
                mgr = SyncManager(sync_dir, strata, data_plane, tenant_id)
                result = await mgr.init()

            if result.get("status") == "initialized":
                print(f"Sync root initialized at {sync_dir}")
                print(f"  Node ID:    {result['node_id']}")
                print(f"  Source ID:  {result.get('source_id', 'n/a')}")
                print(f"  Files:      {result['files_scanned']}")
                print()
                print("Run `adk sync push` to upload or `adk sync watch` to auto-sync.")
            else:
                print(f"Already initialized (node: {result.get('node_id', '?')})")
            return 0

        # All other actions require an existing manifest
        sync_dir = Path(".").resolve()
        manifest_path = sync_dir / MANIFEST_FILE
        if not manifest_path.exists():
            # Walk up to find manifest
            for parent in sync_dir.parents:
                if (parent / MANIFEST_FILE).exists():
                    sync_dir = parent
                    manifest_path = parent / MANIFEST_FILE
                    break
            else:
                print("Not a sync root. Run `adk sync init` first.")
                return 1

        manifest = SyncManifest(sync_dir)
        manifest.load()

        cfg = _load_config()
        token = cfg.get("access_token", cfg.get("api_key", ""))
        tenant_id = manifest.tenant_id or cfg.get("tenant_id", "")

        import httpx
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id

        async with httpx.AsyncClient(timeout=60.0, headers=headers) as http:
            async def get_client():
                return http

            strata_url = cfg.get("strata_url", os.environ.get(
                "AITHER_STRATA_URL", "http://localhost:8136"))
            dp_url = cfg.get("data_plane_url", os.environ.get(
                "AITHER_DATAPLANE_URL", "http://localhost:8170"))
            strata = StrataClient(strata_url, get_client)
            data_plane = DataPlaneClient(dp_url, get_client)
            mgr = SyncManager(sync_dir, strata, data_plane, tenant_id, manifest.node_id)
            mgr.manifest = manifest

            if action == "status":
                st = mgr.status()
                print(f"Sync root: {sync_dir}")
                print(f"Node:      {manifest.node_id}")
                print(f"Last sync: {manifest.last_sync_at or 'never'}")
                print(f"Status:    {st.summary()}")
                if st.new:
                    for f in st.new[:10]:
                        print(f"  + {f}")
                    if len(st.new) > 10:
                        print(f"  ... and {len(st.new) - 10} more")
                if st.changed:
                    for f in st.changed[:10]:
                        print(f"  ~ {f}")
                    if len(st.changed) > 10:
                        print(f"  ... and {len(st.changed) - 10} more")
                if st.deleted:
                    for f in st.deleted[:10]:
                        print(f"  - {f}")
                    if len(st.deleted) > 10:
                        print(f"  ... and {len(st.deleted) - 10} more")

            elif action == "push":
                print("Pushing local changes...")
                result = await mgr.push()
                print(f"Uploaded: {result['uploaded']}  Deleted: {result['deleted']}")
                if result["errors"]:
                    for e in result["errors"][:5]:
                        print(f"  Error: {e}")

            elif action == "pull":
                print("Pulling remote changes...")
                result = await mgr.pull()
                print(f"Downloaded: {result['downloaded']}")
                if result.get("errors"):
                    for e in result["errors"][:5]:
                        print(f"  Error: {e}")

            elif action == "watch":
                started = await mgr.watch()
                if not started:
                    print("watchdog not installed. Install with:")
                    print("  pip install aither-adk[sync]")
                    return 1
                print(f"Watching {sync_dir} for changes (Ctrl+C to stop)...")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    mgr.stop()
                    print("\nWatcher stopped.")

            elif action == "stop":
                mgr.stop()
                print("Watcher stopped.")

            elif action == "ignore":
                pattern = getattr(args, "pattern", "")
                if pattern:
                    mgr.add_ignore(pattern)
                    print(f"Added ignore pattern: {pattern}")

            elif action == "config":
                print(f"Sync root:     {sync_dir}")
                print(f"Node ID:       {manifest.node_id}")
                print(f"Tenant ID:     {manifest.tenant_id}")
                print(f"Source ID:     {manifest.source_id}")
                print(f"Strata prefix: {manifest.strata_prefix}")
                print(f"Conflict:      {manifest.conflict_strategy}")
                print(f"Settings sync: {manifest.settings_sync}")
                print(f"Max file size: {manifest.max_file_size // (1024*1024)}MB")
                print(f"Ignore:        {', '.join(manifest.ignore)}")
                print(f"Tracked files: {len(manifest.files)}")

        return 0

    return asyncio.run(_run())


def cmd_quickstart(args):
    """Unified first-run wizard — setup + auth + shell in one command."""

    cloud_mode = getattr(args, "cloud", False)

    print()
    print("  AitherADK Quickstart")
    print("  ====================")
    print()

    # Step 1: Check if already set up
    saved = load_saved_config()
    if saved.get("setup_backend"):
        print(f"  Already configured: {saved.get('setup_backend')} backend")
        print("  Run 'adk doctor' to check health, or 'adk start' to begin.")
        print()
        return 0

    # Step 1.5: Check for brain_pack.yaml in marketplace/local projects
    brain_pack_path = None
    if Path("brain_pack.yaml").exists():
        brain_pack_path = Path("brain_pack.yaml")
    elif Path("config/brain_pack.yaml").exists():
        brain_pack_path = Path("config/brain_pack.yaml")

    if brain_pack_path:
        print(f"  Found brain pack: {brain_pack_path}")
        try:
            import yaml
            with open(brain_pack_path) as f:
                brain = yaml.safe_load(f) or {}
            agent_name = brain.get("agent_name", "agent")
            model = brain.get("model", "")
            llm_backend = brain.get("llm_backend", "auto")
            print(f"    Agent: {agent_name}")
            print(f"    Model: {model or 'auto'}")
            print(f"    Backend: {llm_backend}")
            print()

            if model and llm_backend == "local":
                print(f"  Preparing model: {model}")
        except Exception as e:
            print(f"  Warning: could not parse brain pack: {e}")
            brain_pack_path = None
            print()

    if cloud_mode:
        # ── Cloud-only quickstart ──────────────────────────────────────
        print("  Cloud-Only Setup (no GPU required)")
        print("  " + "-" * 38)
        print()

        # Step 1: API key setup
        print("  Step 1: Connect Cloud Providers")
        print("  " + "-" * 38)
        print("  Enter API keys for at least one cloud provider.")
        print("  Press Enter to skip.\n")

        keys = _load_provider_keys()
        configured_providers = []

        for pname in ("openai", "anthropic", "deepseek"):
            info = _KNOWN_PROVIDERS[pname]
            existing = keys.get(pname, "") or os.environ.get(info["env"], "")
            if existing:
                print(f"  {info['label']}: already configured ({_mask_key(existing)})")
                configured_providers.append(pname)
                continue
            try:
                val = input(f"  {info['label']} API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if val:
                keys[pname] = val
                os.environ[info["env"]] = val
                ok, msg = _test_provider_key(pname, val)
                icon = "+" if ok else "x"
                print(f"    [{icon}] {msg}")
                if ok:
                    configured_providers.append(pname)
                _push_key_to_vault(pname, val)

        _save_provider_keys(keys)

        if not configured_providers:
            print("\n  No providers configured. At least one API key is required for cloud mode.")
            print("  Run: adk keys set openai sk-...")
            return 1

        # Step 2: Auto-select routing preset
        print()
        print("  Step 2: Routing Configuration")
        print("  " + "-" * 38)

        if set(configured_providers) >= {"openai", "anthropic", "deepseek"}:
            preset = "quality"
        elif "anthropic" in configured_providers and "deepseek" in configured_providers:
            preset = "balanced"
        elif "deepseek" in configured_providers:
            preset = "budget"
        else:
            preset = "balanced"

        print(f"  Auto-selected preset: {preset}")
        print("  (Change later with: adk routing preset <budget|balanced|quality>)")

        # Apply preset
        class RoutingArgs:
            routing_command = "preset"
            preset_name = preset
        cmd_routing(RoutingArgs())

        # Set cloud mode in ADK config
        save_saved_config({
            "setup_backend": "cloud",
            "cloud_mode": "cloud_first",
            "configured_providers": configured_providers,
        })

        # Step 2.5: Test cloud memory
        print()
        print("  Step 2.5: Cloud Memory")
        print("  " + "-" * 38)
        gateway_url = "https://gateway.aitherium.com"
        try:
            import httpx as _httpx
            _mem_resp = _httpx.post(
                f"{gateway_url}/v1/memory/teach",
                json={"content": "adk_quickstart_test", "category": "system"},
                timeout=5.0,
            )
            if _mem_resp.status_code in (200, 201):
                print("  Cloud memory: connected")
                save_saved_config({
                    "spirit_url": gateway_url,
                    "spirit_teach_path": "/v1/memory/teach",
                    "spirit_recall_path": "/v1/memory/recall",
                })
            else:
                print("  Cloud memory: not available (memories will be local-only)")
        except Exception:
            print("  Cloud memory: not available (memories will be local-only)")

        # Step 3: Cost estimate
        print()
        print("  Step 3: Ready!")
        print("  " + "-" * 38)
        print()
        print(f"  Configured providers: {', '.join(configured_providers)}")
        print(f"  Routing preset: {preset}")
        print()
        print("  Estimated costs per 1,000 requests:")
        print("    Budget preset:    ~$0.50 - $2.00")
        print("    Balanced preset:  ~$2.00 - $8.00")
        print("    Quality preset:   ~$5.00 - $20.00")
        print()
        print("  Set a budget:  adk costs budget 50")
        print("  View costs:    adk costs")
        print()
        print("  Next steps:")
        print("    adk start            Start chatting with your codebase")
        print("    adk shell            Interactive terminal with agents")
        print("    adk explore          Browse 47 packs (agents, tools, skills)")
        print("    adk deploy grid      Distributed inference (GPU + Mac + cluster)")
        print("    adk doctor           Check system health")
        print()

        if brain_pack_path:
            print("  Marketplace pack shortcuts:")
            print("    adk run              Launch your packaged agent")
            print("    docker compose up -d")
            print()
            print("  To register with fleet ($5/mo):")
            print("    adk deploy --register-fleet")
            print()
            print("  To add cloud MCP tools:")
            print("    adk mcp add mcp.aitherium.com --api-key <your-key>")
            print()

        # Offer local orchestrator
        try:
            answer = input("  Set up a local orchestrator to reduce costs further? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("y", "yes"):
            from adk.setup_cli import cmd_setup
            class SetupArgs:
                shortcut = None
                tier = None
                mode = "hybrid"
                reasoning_api = None
                reasoning_model = ""
                dgx_spark = None
                stack = None
                dry_run = False
                non_interactive = False
                hf_token = ""
                api_key = getattr(args, "api_key", "") or ""
                output = "docker-compose.vllm.yml"
                force = False
            cmd_setup(SetupArgs())

        return 0

    # ── Standard quickstart (GPU-based) ────────────────────────────────

    # Step 2: Run setup wizard
    print("  Step 1: GPU + Inference Setup")
    print("  " + "-" * 38)
    from adk.setup_cli import cmd_setup

    class SetupArgs:
        shortcut = None
        tier = None
        mode = "auto"
        reasoning_api = None
        reasoning_model = ""
        dgx_spark = None
        stack = None
        dry_run = False
        non_interactive = False
        hf_token = ""
        api_key = getattr(args, "api_key", "") or ""
        output = "docker-compose.vllm.yml"
        force = False

    setup_result = cmd_setup(SetupArgs())
    if setup_result != 0:
        print("  Setup had issues — but you may still be able to use ADK.")
        print()

    # Step 3: Auth (optional)
    print()
    print("  Step 2: Aitherium Account (optional)")
    print("  " + "-" * 38)

    try:
        answer = input("  Connect to Aitherium for cloud tools? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer in ("y", "yes"):
        class LoginArgs:
            email = None
            password = None
            api_key = None
            portal_url = ""
        cmd_login(LoginArgs())

    # Step 4: Shell
    print()
    print("  Step 3: Ready!")
    print("  " + "-" * 38)
    print()
    print("  Your agent system is configured. Next steps:")
    print("    adk start            Start chatting with your codebase")
    print("    adk shell            Launch AitherShell interactive terminal")
    print("    adk run              Start the agent server")
    print("    adk doctor           Check system health")
    print()

    if brain_pack_path:
        print("  Marketplace pack shortcuts:")
        print("    adk run              Launch your packaged agent")
        print("    docker compose up -d")
        print()
        print("  To register with fleet ($5/mo):")
        print("    adk deploy --register-fleet")
        print()
        print("  To add cloud MCP tools:")
        print("    adk mcp add mcp.aitherium.com --api-key <your-key>")
        print()

    return 0


def cmd_quickstart_local(args):
    """
    One-command local inference quickstart.

    Detects hardware → picks backend (llamacpp/ollama/vllm) →
    installs & verifies → launches agent.
    """
    from adk import llamacpp_setup
    from adk.local_backends import pick_backend, docker_available
    from adk.ollama_setup import (
        DEFAULT_OLLAMA_MODEL,
        ensure_running,
        pull,
        register_config as ollama_register_config,
        smoke_test as ollama_smoke_test,
    )

    print()
    print("  AitherADK Local Inference Quickstart")
    print("  ====================================")
    print()

    # Step 1: Detect accelerator
    print("  [1/5] Detecting hardware...")
    accel = llamacpp_setup.detect_accel()
    print(
        f"  Hardware: {accel.os_family}/{accel.arch}, "
        f"accel={accel.kind}, GPU={accel.name}, "
        f"VRAM={accel.vram_gb:.1f}GB, RAM={accel.ram_gb:.1f}GB"
    )

    # Step 2: Pick backend
    print()
    print("  [2/5] Selecting backend...")
    backend = pick_backend(
        accel,
        prefer=getattr(args, "backend", "auto"),
        docker_available_override=docker_available(),
    )
    print(f"  Backend: {backend}")
    if backend == "vllm":
        print("    (NVIDIA CUDA + Docker: optimal throughput)")
    elif backend == "ollama":
        print("    (Ollama installed: convenient & portable)")
    else:
        print("    (llama.cpp: pure stdlib, no dependencies)")

    # Step 3: Install backend
    print()
    print("  [3/5] Installing inference backend...")

    if backend == "llamacpp":
        result = llamacpp_setup.install(
            quant=None,  # auto-detect from hardware
            port=getattr(args, "port", llamacpp_setup.DEFAULT_PORT),
            service=True,
            dry_run=getattr(args, "dry_run", False),
        )
        if not result.success:
            print(
                f"  ERROR: {result.error}",
                file=sys.stderr,
            )
            return 1
        endpoint = f"http://localhost:{result.port}/v1"
        model_name = result.quant

    elif backend == "ollama":
        if not ensure_running():
            print(
                "  ERROR: Could not start Ollama. "
                "Install from https://ollama.ai",
                file=sys.stderr,
            )
            return 1

        # NB: args.model exists but defaults to None, so getattr's fallback never
        # fires — use `or` to apply the default when the flag was omitted.
        model = getattr(args, "model", None) or DEFAULT_OLLAMA_MODEL
        if getattr(args, "dry_run", False):
            print(f"  [DRY] Would pull model: {model}")
        else:
            if not pull(model):
                print(f"  ERROR: Failed to pull model {model}", file=sys.stderr)
                return 1
        ollama_register_config(model)
        endpoint = "http://localhost:11434/v1"
        model_name = model

    elif backend == "vllm":
        # Use setup_cli's cmd_setup for Docker-based vLLM
        from adk.setup_cli import cmd_setup

        class SetupArgs:
            shortcut = None
            tier = None
            mode = "auto"
            reasoning_api = None
            reasoning_model = ""
            dgx_spark = None
            stack = None
            dry_run = getattr(args, "dry_run", False)
            non_interactive = True
            hf_token = ""
            api_key = getattr(args, "api_key", "") or ""
            output = "docker-compose.vllm.yml"

        setup_result = cmd_setup(SetupArgs())
        if setup_result != 0:
            print("  ERROR: vLLM setup failed", file=sys.stderr)
            return 1
        endpoint = "http://localhost:8200/v1"
        model_name = "auto"
    else:
        print(f"  ERROR: unknown backend {backend}", file=sys.stderr)
        return 1

    # Step 4: Verify with smoke test. A failed verification FAILS the command —
    # proving inference actually works is the entire point of "quickstart".
    print()
    print("  [4/5] Verifying endpoint...")
    if getattr(args, "dry_run", False):
        print("  [DRY] Skipping smoke test")
    else:
        if backend in ("llamacpp", "vllm"):
            import time

            verify_port = getattr(args, "port", 8200) if backend == "llamacpp" else 8200
            # Wait for the freshly-started service to bind + load the model.
            for _ in range(60):
                if llamacpp_setup.status(port=verify_port).running:
                    break
                time.sleep(2)
            ok = llamacpp_setup.smoke_test(port=verify_port)
        else:  # ollama
            ok = ollama_smoke_test(model)
        if not ok:
            print(
                "  ERROR: Smoke test failed — the endpoint did not return a valid "
                "completion. The backend may have failed to start or load the model. "
                "Check logs and re-run.",
                file=sys.stderr,
            )
            return 1
    print(f"  Endpoint ready: {endpoint}")

    # Step 5: Persist config
    print()
    print("  [5/5] Saving configuration...")
    save_saved_config({
        "setup_backend": backend,
        "inference_url": endpoint,
        "inference_model": model_name,
    })
    print("  Config saved to ~/.aither/config.json")

    # Banner
    print()
    print("=" * 60)
    print("  Local Inference Backend Ready!")
    print("=" * 60)
    print()
    print(f"  Endpoint:  {endpoint}")
    print(f"  Model:     {model_name}")
    print(f"  Backend:   {backend}")
    print()
    print("  Next steps:")
    print("    adk start              Chat with your codebase")
    print("    adk shell              Interactive terminal")
    print("    adk doctor             Check system health")
    print()

    return 0


def cmd_status(args):
    """Show the running agent (adk up) plus backend and service status."""
    import asyncio

    from adk import agent_daemon as daemon

    # ── Local agent (from `adk up`) ──
    st = daemon.read_status()
    agent_state = None
    if st:
        alive = daemon.pid_alive(st.get("server_pid"))
        healthy = daemon.wait_for_health(st.get("port", 8080), timeout=3) if alive else False
        agent_state = {
            **st,
            "running": alive,
            "health": "healthy" if healthy else ("stale" if not alive else "unhealthy"),
            "tunnel_running": daemon.pid_alive(st.get("tunnel_pid")),
        }

    if getattr(args, "json", False):
        print(json.dumps(agent_state or {"running": False}, indent=2))
        return 0

    if agent_state:
        print("AitherADK Agent")
        print("=" * 50)
        icon = "+" if agent_state["running"] else "-"
        print(f"  [{icon}] {agent_state.get('identity', 'aither'):12s} "
              f":{agent_state.get('port')} ({agent_state['health']})")
        if agent_state.get("invoke_url"):
            print(f"      tunnel: {agent_state['invoke_url']} "
                  f"({'up' if agent_state['tunnel_running'] else 'down'})")
        print(f"      registered: {agent_state.get('registered')}   "
              f"autostart: {agent_state.get('autostart') or 'no'}")
        print()

    async def _status():
        import httpx
        from adk.config import load_saved_config

        checks = {
            "Genesis": os.environ.get("AITHER_URL", "http://localhost:8001"),
            "vLLM": os.environ.get("AITHER_VLLM_URL", os.environ.get("VLLM_URL", "http://localhost:8200")),
            "Ollama": _fix_ollama_host(os.environ.get("OLLAMA_HOST", "")),
            "AitherNode": "http://localhost:8090",
            "Gateway": os.environ.get("AITHER_GATEWAY_URL", "https://gateway.aitherium.com"),
        }

        # Add Qdrant if configured
        qdrant_url = os.environ.get("AITHER_FLEET_QDRANT_URL", "")
        if not qdrant_url:
            try:
                saved = load_saved_config()
                qdrant_url = saved.get("qdrant_url", "")
            except Exception:
                pass
        if qdrant_url:
            checks["Qdrant"] = qdrant_url

        print("AitherADK Backend Status")
        print("=" * 50)
        for name, url in checks.items():
            try:
                async with httpx.AsyncClient(timeout=3.0) as c:
                    hp = "/api/tags" if name == "Ollama" else "/health"
                    if name == "Qdrant":
                        hp = "/healthz"
                    r = await c.get(f"{url.rstrip('/')}{hp}")
                    status = "UP" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception:
                status = "DOWN"
            icon = "+" if status == "UP" else "-"
            print(f"  [{icon}] {name:12s} {url:45s} {status}")

        # Scan additional vLLM ports
        for extra_port in [8201, 8202, 8203, 8209]:
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    r = await c.get(f"http://localhost:{extra_port}/health")
                    if r.status_code == 200:
                        url = f"http://localhost:{extra_port}"
                        print(f"  [+] {'vLLM':12s} {url:45s} UP")
            except Exception:
                pass

        # API key check
        saved = {}
        try:
            saved = load_saved_config()
        except Exception:
            pass

        api_key = os.environ.get("AITHER_API_KEY", "")
        if api_key:
            print(f"\n  Portal API Key: {api_key[:16]}...{api_key[-4:]}")
        elif saved.get("api_key"):
            print(f"\n  Portal API Key (saved): {saved['api_key'][:16]}...")
        else:
            print("\n  Portal: No API key. Run: adk connect --api-key <key>")

        # Qdrant API key check
        qdrant_key = os.environ.get("AITHER_FLEET_QDRANT_API_KEY", "")
        if qdrant_key:
            print(f"  Qdrant API Key: {qdrant_key[:16]}...{qdrant_key[-4:]}")
        elif saved.get("qdrant_api_key"):
            print(f"  Qdrant API Key (saved): {saved['qdrant_api_key'][:16]}...")
        else:
            if qdrant_url:
                print("  Qdrant: No API key configured. Run: adk stack qdrant")

    asyncio.run(_status())
    return 0


def cmd_start(args):
    """Zero-config agent start — index, connect, chat. Works for anyone."""
    import asyncio
    import time as _time
    import shutil

    target = os.path.abspath(args.path or ".")
    project_name = os.path.basename(target)

    # ── Banner ──────────────────────────────────────────────────────
    print()
    print("  AitherADK")
    print("  =========")
    print()

    # ── Step 1: Detect project ──────────────────────────────────────
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv",
                 ".tox", "dist", "build", ".mypy_cache", "site-packages"}

    def _count_files(root, ext):
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            count += sum(1 for f in filenames if f.endswith(ext))
            if count > 5000:
                break  # Good enough
        return count

    py_count = _count_files(target, ".py")
    ts_count = _count_files(target, ".ts")
    js_count = _count_files(target, ".js")

    # Count all file types for a richer picture
    md_count = _count_files(target, ".md")
    txt_count = _count_files(target, ".txt")
    json_count = _count_files(target, ".json")
    yaml_count = _count_files(target, ".yaml") + _count_files(target, ".yml")
    total_files = py_count + ts_count + js_count + md_count + txt_count + json_count + yaml_count

    # Classify workspace type
    lang = None
    if py_count >= ts_count and py_count >= js_count and py_count > 5:
        lang = "Python"
    elif ts_count > 5:
        lang = "TypeScript"
    elif js_count > 5:
        lang = "JavaScript"

    workspace_parts = []
    if lang:
        workspace_parts.append(f"{lang} ({py_count or ts_count or js_count} files)")
    if md_count > 0:
        workspace_parts.append(f"{md_count} docs")
    if json_count + yaml_count > 0:
        workspace_parts.append(f"{json_count + yaml_count} configs")
    if txt_count > 0:
        workspace_parts.append(f"{txt_count} text files")

    if workspace_parts:
        print(f"  Workspace:  {project_name} -- {', '.join(workspace_parts)}")
    else:
        print(f"  Workspace:  {project_name} (empty or no recognized files)")
    print(f"  Directory:  {target}")

    # ── Step 2: Detect LLM backend ──────────────────────────────────
    llm_info = _detect_llm_backend()
    print(f"  LLM:        {llm_info['display']}")

    # ── Step 3: Index codebase (if applicable) ────────────────────
    code_graph = None
    memory_graph = None  # pre-existing REPL /stats,/memory refs were undefined (NameError)
    if lang == "Python" and py_count > 0:
        from adk.faculties.code_graph import CodeGraph
        code_graph = CodeGraph()
        print()
        print(f"  Indexing {py_count} Python files...", end="", flush=True)
        t0 = _time.perf_counter()
        stats = asyncio.run(code_graph.index_codebase(target))
        elapsed = _time.perf_counter() - t0
        print(f" {stats['total_chunks']:,} chunks in {elapsed:.1f}s")
    elif total_files > 0:
        print("  Code index: Skipped (no Python files -- code search works with Python)")
    else:
        print("  Code index: Skipped (empty directory)")

    # ── Step 4: Set up memory ───────────────────────────────────────
    # Suppress noisy warnings for casual use
    _logging = __import__("logging")
    _logging.getLogger("adk.graph_memory").setLevel(_logging.ERROR)
    _logging.getLogger("adk.identity").setLevel(_logging.ERROR)

    from adk.graph_memory import GraphMemory
    graph = GraphMemory(agent_name=project_name)
    mem_stats = asyncio.run(graph.get_stats())
    if mem_stats.get("nodes", 0) > 0:
        print(f"  Memory:     {mem_stats['nodes']} memories restored from previous sessions")
    else:
        print(f"  Memory:     New (will persist across sessions)")

    # ── Step 5: Build agent ─────────────────────────────────────────
    print()

    from adk.agent import AitherAgent
    from adk.llm import LLMRouter

    llm_kwargs = {}
    if llm_info.get("provider"):
        llm_kwargs["provider"] = llm_info["provider"]
    if llm_info.get("base_url"):
        llm_kwargs["base_url"] = llm_info["base_url"]
    if llm_info.get("model"):
        llm_kwargs["model"] = llm_info["model"]
    if llm_info.get("api_key"):
        llm_kwargs["api_key"] = llm_info["api_key"]

    llm = LLMRouter(**llm_kwargs) if llm_kwargs else None

    # Build system prompt based on what's available
    prompt_parts = [
        f"You are a helpful assistant for the '{project_name}' workspace.",
        f"The workspace is at: {target}",
    ]
    if code_graph:
        prompt_parts.append(
            "You have code_search and code_context tools — ALWAYS search before answering code questions."
        )
    prompt_parts.append(
        "You have remember/recall tools for persistent memory across sessions. "
        "Use them proactively to store important findings and user preferences."
    )
    prompt_parts.append(
        "You also have file tools (read_file, write_file, search_files, list_directory) "
        "for working with any files in the workspace."
    )
    prompt_parts.append(
        "Be direct and helpful. If you're unsure, search first, then answer."
    )

    agent = AitherAgent(
        name=project_name,
        llm=llm,
        system_prompt=" ".join(prompt_parts),
    )

    if code_graph:
        agent.set_code_graph(code_graph)
    # GraphMemory is already wired via agent._graph in __init__

    # ── Step 6: Interactive chat loop ───────────────────────────────
    print()
    capabilities = []
    if code_graph:
        capabilities.append("search your code")
    capabilities.append("read/write files")
    capabilities.append("remember things across sessions")
    print(f"  Ready! I can {', '.join(capabilities)}.")
    print("  Just ask a question. Type /help for commands, /quit to exit.")
    print()

    session_id = agent.new_session()

    async def _chat_loop():
        while True:
            try:
                user_input = input("  You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                break

            if user_input.lower() == "/help":
                print()
                print("  /quit     Exit")
                print("  /stats    Show index stats")
                print("  /memory   Show memory stats")
                print("  /forget   Clear session memory")
                print("  /reindex  Re-index the codebase")
                print()
                continue

            if user_input.lower() == "/stats":
                if code_graph:
                    print(f"  Code index: {len(code_graph.chunks):,} chunks, "
                          f"{code_graph.total_files} files, "
                          f"{code_graph.memory_usage_mb:.1f}MB")
                if memory_graph:
                    ms = memory_graph.get_stats()
                    print(f"  Memory:     {ms['nodes']} memories, {ms['edges']} connections")
                print(f"  Workspace:  {target}")
                continue

            if user_input.lower() == "/memory":
                if memory_graph:
                    ms = memory_graph.get_stats()
                    print(f"  Nodes: {ms['nodes']}, Edges: {ms['edges']}, "
                          f"Embeddings: {ms['embeddings_cached']}")
                else:
                    print("  (memory graph not initialized)")
                continue

            if user_input.lower() == "/forget":
                agent.new_session()
                print("  Session cleared.")
                continue

            if user_input.lower() == "/reindex":
                if code_graph:
                    print("  Re-indexing...", end="", flush=True)
                    stats = await code_graph.index_codebase(target)
                    print(f" {stats['total_chunks']:,} chunks")
                continue

            # Chat
            try:
                response = await agent.chat(
                    user_input,
                    session_id=session_id,
                    effort=5,
                )
                print()
                print(f"  {response.content}")
                print()
                if response.tool_calls_made:
                    tools_used = ", ".join(set(
                        t.split("[")[0] for t in response.tool_calls_made
                    ))
                    print(f"  [tools: {tools_used}]")
                    print()
            except Exception as e:
                print(f"\n  Error: {e}\n")

    asyncio.run(_chat_loop())

    # Save memory on exit (suppress noisy HMAC warnings)
    logging = __import__("logging")
    logging.getLogger("adk.faculties").setLevel(logging.ERROR)
    if memory_graph:
        memory_graph.save()
    print("  Memory saved. Goodbye!")
    return 0


def _detect_llm_backend():
    """Detect available LLM backend. Returns dict with provider info."""
    import shutil
    import subprocess

    # 1. Check for Ollama
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        try:
            import httpx
            resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m["name"] for m in models]
                # Pick best available model
                preferred = [
                    "llama3.2:latest", "llama3.2:3b", "llama3.1:8b",
                    "mistral:latest", "qwen2.5:7b",
                ]
                chosen = None
                for p in preferred:
                    if p in model_names:
                        chosen = p
                        break
                if not chosen and model_names:
                    chosen = model_names[0]
                if chosen:
                    return {
                        "provider": "ollama",
                        "model": chosen,
                        "display": f"Ollama ({chosen})",
                    }
                else:
                    return {
                        "provider": "ollama",
                        "display": "Ollama (no models pulled — run: ollama pull llama3.2)",
                    }
        except Exception:
            pass

    # 2. Check for vLLM
    try:
        import httpx
        for port in (8200, 8201, 8202, 8203, 8209, 8000):
            try:
                resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    model_id = data["data"][0]["id"] if data.get("data") else "unknown"
                    return {
                        "provider": "openai",
                        "base_url": f"http://localhost:{port}/v1",
                        "model": model_id,
                        "api_key": "not-needed",
                        "display": f"vLLM ({model_id})",
                    }
            except Exception:
                continue
    except ImportError:
        pass

    # 3. Check for Elysium API key
    api_key = os.environ.get("AITHER_API_KEY", "")
    if not api_key:
        config_path = Path.home() / ".aither" / "config.json"
        if config_path.exists():
            try:
                import json as _j
                cfg = _j.loads(config_path.read_text(encoding="utf-8"))
                api_key = cfg.get("api_key", "")
            except Exception:
                pass
    if api_key:
        return {
            "provider": "gateway",
            "base_url": "https://mcp.aitherium.com/v1",
            "api_key": api_key,
            "model": "aither-orchestrator",
            "display": "Elysium Cloud (aither-orchestrator)",
        }

    # 4. Check for OpenAI key
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": "gpt-4o-mini",
            "display": "OpenAI (gpt-4o-mini)",
        }

    # 5. Check for Anthropic key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "model": "claude-sonnet-4-20250514",
            "display": "Anthropic (claude-sonnet-4-20250514)",
        }

    return {
        "display": "None detected! Install Ollama (ollama.com) or set AITHER_API_KEY",
    }


def cmd_index(args):
    """Index a codebase for code search via CodeGraph."""
    import asyncio
    import time as _time

    target = os.path.abspath(args.path)
    if not os.path.isdir(target):
        print(f"Error: {target} is not a directory")
        return 1

    print(f"Indexing: {target}")
    print()

    from adk.faculties.code_graph import CodeGraph

    cg = CodeGraph()

    def on_progress(frac, msg):
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {frac*100:5.1f}%  {msg:<50}", end="", flush=True)

    t0 = _time.perf_counter()
    stats = asyncio.run(cg.index_codebase(target, on_progress=on_progress))
    elapsed = _time.perf_counter() - t0
    print()  # newline after progress bar
    print()
    print(f"  Files:      {stats['total_files']:,}")
    print(f"  Functions:  {stats['functions']:,}")
    print(f"  Methods:    {stats['methods']:,}")
    print(f"  Classes:    {stats['classes']:,}")
    print(f"  Total:      {stats['total_chunks']:,} chunks in {elapsed:.1f}s")

    if args.embed:
        print()
        print("Generating embeddings...")
        try:
            embed_stats = asyncio.run(cg.embed_chunks(on_progress=on_progress))
            print()
            print(f"  Embedded:   {embed_stats.get('new', 0)} new, {embed_stats.get('cached', 0)} cached")
            print(f"  Backend:    {embed_stats.get('model', 'unknown')}")
        except Exception as e:
            print(f"\n  Embedding failed: {e}")
            print("  (Install sentence-transformers for local embeddings, or set AITHER_API_KEY for cloud)")

    if args.stats:
        print()
        metrics = cg.get_python_metrics()
        print(f"  Total lines:      {metrics['total_py_lines']:,}")
        print(f"  Avg complexity:   {metrics['avg_complexity']}")
        print(f"  Test functions:   {metrics['test_functions']:,}")
        if metrics.get("top_complex_files"):
            print("  Most complex:")
            for name, cx in metrics["top_complex_files"][:5]:
                print(f"    {name}: {cx}")

    # Test a sample query
    print()
    sample_results = asyncio.run(cg.query("main", max_results=3))
    if sample_results:
        print("  Sample query 'main':")
        for r in sample_results:
            print(f"    {r.chunk_type.value}: {r.name} @ {Path(r.source_path).name}:{r.start_line}")

    print()
    print("Done! Use in your agent:")
    print()
    print("    from adk.faculties import CodeGraph")
    print("    cg = CodeGraph()")
    print(f"    await cg.index_codebase(\"{target}\")")
    print("    agent.set_code_graph(cg)")
    return 0


def _connect_elysium(args):
    """Connect to a desktop AitherOS instance via --elysium flag."""
    import asyncio

    async def _run():
        from adk.elysium_connect import connect_to_desktop

        url = args.elysium
        token = getattr(args, "token", None)

        print()
        print("  AitherOS Desktop Connect (Elysium)")
        print("  ===================================")
        print()
        print(f"  Desktop: {url}")

        result = await connect_to_desktop(url, token=token)

        if not result.get("ok"):
            print(f"  [!!] Connection failed: {result.get('error', 'unknown')}")
            return 1

        print("  [OK] Genesis reachable")

        if result.get("token"):
            print(f"  [OK] Node token: {result['token'][:16]}...")

        if result.get("mesh_joined"):
            print(f"  [OK] Mesh joined (node: {result.get('node_id', 'unknown')[:16]})")
        else:
            print("  [--] Mesh join: skipped or failed")

        if result.get("wireguard"):
            print("  [OK] WireGuard tunnel active")
        else:
            print("  [--] WireGuard: not configured (direct LAN is fine)")

        print(f"  [OK] Remote inference: {result.get('core_llm_url', 'N/A')}")
        print(f"  [OK] Config saved to {result.get('config_saved', '~/.aither/config.json')}")

        print()
        print("  Next steps:")
        print("    adk run              # Start agent server")
        print("    adk run --mesh       # Start with mesh hosting (share your tools)")
        print("    adk status           # Check backend status")
        print()

        return 0

    return asyncio.run(_run())


def cmd_admin(args):
    """Administration commands."""
    import asyncio

    admin_cmd = getattr(args, "admin_command", None)

    if admin_cmd == "create-token":
        return _admin_create_token(args)
    else:
        print("  Usage: adk admin create-token --name <name> --url <genesis-url>")
        return 1


def _admin_create_token(args):
    """Create a node token on the desktop for mesh enrollment."""
    import asyncio
    import platform as plat

    async def _run():
        import httpx

        url = args.url.rstrip("/")
        name = args.name or plat.node()

        print()
        print("  AitherOS Admin — Create Node Token")
        print("  ===================================")
        print()
        print(f"  Genesis: {url}")
        print(f"  Node name: {name}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{url}/admin/nodes/create",
                    json={
                        "node_name": name,
                        "capabilities": ["mcp", "inference"],
                    },
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    token = data.get("token") or data.get("node_token", "")
                    node_id = data.get("node_id", "")

                    print()
                    print("  [OK] Token created!")
                    print(f"  Node ID: {node_id}")
                    print(f"  Token:   {token}")
                    print()
                    print("  Use on the laptop:")
                    print(f"    adk connect --elysium {url} --token {token}")
                    print()
                    print("  Or set environment variables:")
                    print(f"    export AITHER_CORE_URL={url}")
                    print(f"    export AITHER_NODE_TOKEN={token}")
                    print()

                    # Save to local config
                    save_saved_config({
                        "admin_last_token": token,
                        "admin_last_node_id": node_id,
                    })

                    return 0
                else:
                    print(f"  [!!] Failed: HTTP {resp.status_code}")
                    print(f"       {resp.text[:200]}")
                    return 1
        except Exception as e:
            print(f"  [!!] Error: {e}")
            return 1

    return asyncio.run(_run())


def cmd_disconnect(args):
    """Disconnect from desktop AitherOS mesh."""
    import asyncio

    async def _run():
        from adk.elysium_connect import disconnect_from_desktop

        print()
        print("  Disconnecting from desktop mesh...")

        result = await disconnect_from_desktop()

        if result.get("mesh_left"):
            print("  [OK] Left mesh")
        if result.get("wireguard_down"):
            print("  [OK] WireGuard tunnel torn down")
        if result.get("config_cleared"):
            print("  [OK] Elysium config cleared")

        print("  Done.")
        print()
        return 0

    return asyncio.run(_run())


def cmd_jobs(args) -> int:
    """Manage background jobs.

    LOCAL by default (jobs run on THIS machine, tracked in ~/.aither/jobs.db);
    pass --remote to operate on cloud expeditions via genesis/portal instead.
    """
    import asyncio

    jobs_cmd = getattr(args, "jobs_command", None)
    remote = bool(getattr(args, "remote", False))

    # Local engine subcommands (no server required).
    if jobs_cmd == "run":
        return _jobs_run_local(args)
    elif jobs_cmd == "start":
        return _jobs_start_local(args)
    elif jobs_cmd == "cancel":
        return _jobs_cancel_local(args)
    elif jobs_cmd == "sync":
        return _jobs_sync_local(args)
    elif jobs_cmd == "_exec":
        from adk.local_jobs import _execute
        _execute(args.id)
        return 0

    # Read/steer: remote (cloud) when --remote, else the local store.
    if jobs_cmd == "list":
        return asyncio.run(_jobs_list(args)) if remote else _jobs_list_local(args)
    elif jobs_cmd == "status":
        return asyncio.run(_jobs_status(args)) if remote else _jobs_status_local(args)
    elif jobs_cmd == "steer":
        return asyncio.run(_jobs_steer(args)) if remote else _jobs_steer_local(args, "append")
    elif jobs_cmd == "hint":
        return asyncio.run(_jobs_hint(args)) if remote else _jobs_steer_local(args, "hint")
    elif jobs_cmd == "watch":
        return asyncio.run(_jobs_watch(args))  # SSE watch is cloud-only
    else:
        print("Usage: adk jobs [run|start|list|status|steer|hint|cancel|watch|sync]")
        print("  Local by default; add --remote to list/status/steer against the cloud.")
        return 1


# ── Local job engine handlers (~/.aither/jobs.db) ─────────────────────────────

def _jobs_run_local(args) -> int:
    """Run a job locally in the foreground and print its result."""
    from adk.local_jobs import run_foreground
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    print(f"  Running locally: {query[:70]}")
    row = run_foreground(query, agent=getattr(args, "agent", "aither"))
    status = row.get("status", "?")
    if status == "completed":
        print(f"\n  [done] job {row.get('id', '')[:8]}\n")
        print(row.get("result", "") or "  (no output)")
    else:
        print(f"  [{status}] {row.get('error', '') or ''}")
    return 0 if status == "completed" else 1


def _jobs_start_local(args) -> int:
    """Start a local job in the background (detached) and return its id."""
    from adk.local_jobs import spawn
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    jid = spawn(query, agent=getattr(args, "agent", "aither"))
    print(f"  Started local job {jid[:8]} in the background.")
    print(f"  Status: adk jobs status {jid[:8]}   ·   Cancel: adk jobs cancel {jid[:8]}")
    return 0


def _jobs_list_local(args) -> int:
    """List local jobs from ~/.aither/jobs.db."""
    from adk.local_jobs import get_store
    jobs = get_store().list()
    if not jobs:
        print("  No local jobs. Start one: adk jobs start \"<task>\"")
        return 0
    print("\n  Local Jobs")
    print("  " + "=" * 70)
    print(f"  {'ID':<10s} {'Status':<11s} {'Synced':<7s} {'Query':<40s}")
    print("  " + "-" * 70)
    for j in jobs:
        synced = "yes" if j.get("remote_id") else "-"
        print(f"  {j['id'][:8]:<10s} {j['status']:<11s} {synced:<7s} {(j['query'] or '')[:40]:<40s}")
    print()
    return 0


def _jobs_status_local(args) -> int:
    """Show a local job's status + result."""
    from adk.local_jobs import get_store
    job = get_store().get(args.id)
    if not job:
        print(f"  Local job not found: {args.id}")
        return 1
    print(f"\n  Job {job['id'][:8]}  [{job['status']}]")
    print(f"  Query: {job['query']}")
    if job.get("remote_id"):
        print(f"  Synced → {job.get('remote_url', '')}/expedition/{job['remote_id']}")
    if job.get("result"):
        print("\n" + job["result"])
    if job.get("error"):
        print(f"\n  Error: {job['error']}")
    return 0


def _jobs_steer_local(args, action: str) -> int:
    """Queue a steering/hint nudge for a running local job."""
    from adk.local_jobs import steer
    ok = steer(args.id, args.message, action)
    print(f"  {'Steered' if ok else 'Job not found:'} {args.id}" + ("" if ok else ""))
    return 0 if ok else 1


def _jobs_cancel_local(args) -> int:
    """Cancel a running local job."""
    from adk.local_jobs import cancel
    ok = cancel(args.id)
    print(f"  {'Cancelled' if ok else 'Not cancellable (missing or already finished):'} {args.id}")
    return 0 if ok else 1


def _jobs_sync_local(args) -> int:
    """Push a local job up to the portal, or pull its remote status back."""
    from adk.local_jobs import pull, push
    if args.direction == "push":
        res = push(args.id)
        if res.get("ok"):
            print(f"  Pushed local job {args.id} → portal expedition {res.get('remote_id', '')[:8]}")
            return 0
        print(f"  Push failed: {res.get('error')}")
        return 1
    else:
        res = pull(args.id)
        if res.get("ok"):
            print(f"  Pulled: local job {args.id} is now '{res.get('status')}'")
            return 0
        print(f"  Pull failed: {res.get('error')}")
        return 1


async def _jobs_list(args) -> int:
    """List all jobs and expeditions."""
    import httpx

    genesis_url = os.environ.get(
        "AITHER_GENESIS_URL", os.environ.get("AITHER_URL", "http://localhost:8001")
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{genesis_url.rstrip('/')}/expedition/list")
            if resp.status_code == 200:
                data = resp.json()
                expeditions = data.get("expeditions", [])

                if not expeditions:
                    print("  No jobs found.")
                    return 0

                print("\n  Expeditions / Jobs")
                print("  " + "=" * 70)
                print(
                    f"  {'ID':<20s} {'Status':<12s} {'Title':<36s}"
                )
                print("  " + "-" * 70)
                for exp in expeditions:
                    exp_id = exp.get("id", "unknown")[:20]
                    status = exp.get("status", "unknown")[:12]
                    title = exp.get("title", "")[:36]
                    print(f"  {exp_id:<20s} {status:<12s} {title:<36s}")
                print()
                return 0
            elif resp.status_code == 404:
                print("  Error: Genesis /expedition/list not found (404)")
                return 1
            else:
                print(
                    f"  Error: HTTP {resp.status_code} from {genesis_url}"
                )
                return 1
    except Exception as e:
        print(f"  Error: Could not reach Genesis at {genesis_url}")
        print(f"         {e}")
        return 1


async def _jobs_status(args) -> int:
    """Show status of a job."""
    import httpx

    genesis_url = os.environ.get(
        "AITHER_GENESIS_URL", os.environ.get("AITHER_URL", "http://localhost:8001")
    )
    exp_id = args.id

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get expedition status
            status_resp = await client.get(
                f"{genesis_url.rstrip('/')}/expedition/{exp_id}/status"
            )
            if status_resp.status_code == 404:
                print(f"  Error: Expedition '{exp_id}' not found")
                return 1
            elif status_resp.status_code != 200:
                print(
                    f"  Error: HTTP {status_resp.status_code} from {genesis_url}"
                )
                return 1

            status_data = status_resp.json()

            # Get expedition tasks
            tasks_resp = await client.get(
                f"{genesis_url.rstrip('/')}/expedition/{exp_id}/tasks"
            )
            tasks = []
            if tasks_resp.status_code == 200:
                tasks = tasks_resp.json().get("tasks", [])

            # Print status
            print()
            print(f"  Expedition: {exp_id}")
            print("  " + "=" * 70)
            for key, value in status_data.items():
                if key != "id":
                    print(f"  {key:.<20s} {value}")

            # Print tasks if available
            if tasks:
                print()
                print("  Tasks:")
                print("  " + "-" * 70)
                for task in tasks:
                    task_id = task.get("id", "unknown")
                    task_status = task.get("status", "unknown")
                    task_title = task.get("title", "")
                    print(f"    [{task_status:>8s}] {task_id:.<24s} {task_title}")
                    if task.get("result_summary"):
                        print(f"             Result: {task['result_summary'][:50]}")
                    if task.get("error"):
                        print(f"             Error: {task['error'][:50]}")
            print()
            return 0
    except Exception as e:
        print(f"  Error: Could not reach Genesis at {genesis_url}")
        print(f"         {e}")
        return 1


async def _jobs_steer(args) -> int:
    """Send a follow-up message to a job."""
    import httpx

    genesis_url = os.environ.get(
        "AITHER_GENESIS_URL", os.environ.get("AITHER_URL", "http://localhost:8001")
    )
    exp_id = args.id
    message = args.message

    if not message:
        print("  Error: message is required")
        return 1

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{genesis_url.rstrip('/')}/expedition/{exp_id}/steer",
                json={"message": message, "action": "append"},
            )
            if resp.status_code == 404:
                print(f"  Error: Expedition '{exp_id}' not found")
                return 1
            elif resp.status_code == 400:
                print("  Error: Invalid request (400)")
                return 1
            elif resp.status_code in (200, 201):
                data = resp.json()
                print(f"  [OK] Message sent to {exp_id}")
                if data.get("status"):
                    print(f"       Status: {data['status']}")
                return 0
            else:
                print(
                    f"  Error: HTTP {resp.status_code} from {genesis_url}"
                )
                return 1
    except Exception as e:
        print(f"  Error: Could not reach Genesis at {genesis_url}")
        print(f"         {e}")
        return 1


async def _jobs_hint(args) -> int:
    """Send an invisible hint to a job."""
    import httpx

    genesis_url = os.environ.get(
        "AITHER_GENESIS_URL", os.environ.get("AITHER_URL", "http://localhost:8001")
    )
    exp_id = args.id
    message = args.message

    if not message:
        print("  Error: message is required")
        return 1

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{genesis_url.rstrip('/')}/expedition/{exp_id}/steer",
                json={"message": message, "action": "hint"},
            )
            if resp.status_code == 404:
                print(f"  Error: Expedition '{exp_id}' not found")
                return 1
            elif resp.status_code == 400:
                print("  Error: Invalid request (400)")
                return 1
            elif resp.status_code in (200, 201):
                data = resp.json()
                print(f"  [OK] Hint sent to {exp_id}")
                if data.get("status"):
                    print(f"       Status: {data['status']}")
                return 0
            else:
                print(
                    f"  Error: HTTP {resp.status_code} from {genesis_url}"
                )
                return 1
    except Exception as e:
        print(f"  Error: Could not reach Genesis at {genesis_url}")
        print(f"         {e}")
        return 1


async def _jobs_watch(args) -> int:
    """Watch a job's progress in real-time via SSE."""
    import httpx

    genesis_url = os.environ.get(
        "AITHER_GENESIS_URL", os.environ.get("AITHER_URL", "http://localhost:8001")
    )
    exp_id = args.id

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET",
                f"{genesis_url.rstrip('/')}/expedition/{exp_id}/stream",
            ) as resp:
                if resp.status_code == 404:
                    print(f"  Error: Expedition '{exp_id}' not found")
                    return 1
                elif resp.status_code != 200:
                    print(
                        f"  Error: HTTP {resp.status_code} from {genesis_url}"
                    )
                    return 1

                print(f"\n  Watching expedition {exp_id}...")
                print("  " + "=" * 70)

                try:
                    async for line in resp.aiter_lines():
                        if not line or line.startswith(":"):
                            # Skip empty lines and comments
                            continue
                        if line.startswith("data: "):
                            # Parse SSE data line
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    import json as _json

                                    data = _json.loads(data_str)
                                    # Print event details
                                    event_type = data.get("type", "event")
                                    msg = data.get("message", "")
                                    if msg:
                                        print(f"  [{event_type}] {msg}")
                                except (ValueError, AttributeError):
                                    # Fallback for unparseable JSON
                                    print(f"  {data_str}")
                except KeyboardInterrupt:
                    print("\n  Watching stopped.")
                    return 0

                print("  " + "=" * 70)
                return 0
    except Exception as e:
        print(f"  Error: Could not watch expedition")
        print(f"         {e}")
        return 1


def _cmd_cron(args) -> int:
    """Handle `adk cron` subcommands."""
    sub = getattr(args, "cron_command", None)
    try:
        from adk.cron import CronScheduler
    except ImportError:
        print("Cron module not available.")
        return 1

    sched = CronScheduler()

    if sub == "list" or sub is None:
        jobs = sched.list_jobs()
        if not jobs:
            print("No cron jobs configured.")
            return 0
        print(f"{'Name':<30} {'Expression':<20} {'Enabled'}")
        print("-" * 60)
        for j in jobs:
            print(f"{j.name:<30} {j.expression:<20} {'yes' if j.enabled else 'no'}")
        return 0

    if sub == "add":
        expr = args.expression
        name = args.task_name
        # Register a placeholder -- actual task binding happens programmatically
        sched.add(expr, task=None, name=name)
        print(f"Added cron job '{name}' with schedule: {expr}")
        print("Note: bind a task function programmatically via CronScheduler.add()")
        return 0

    if sub == "remove":
        if sched.remove(args.name):
            print(f"Removed cron job '{args.name}'")
        else:
            print(f"Job '{args.name}' not found")
        return 0

    print("Usage: adk cron [list|add|remove]")
    return 1


def _cmd_addon(args) -> int:
    """Handle `adk addon` subcommands."""
    import asyncio

    sub = getattr(args, "addon_command", None)

    if sub == "list" or sub is None:
        from adk.addon_manager import load_all_manifests, AddonManager
        manifests = load_all_manifests()
        mgr = AddonManager()
        states = {s.get("addon_id"): s for s in [inst.to_dict() for inst in asyncio.run(mgr.status())]}

        if not manifests:
            print("No addon manifests found.")
            return 0

        print(f"{'Addon':<20} {'Type':<10} {'Port':<7} {'Plan':<12} {'Status':<10} {'Pack'}")
        print("-" * 80)
        for m in manifests:
            aid = m["id"]
            state = states.get(aid, {})
            status = state.get("status", "disabled")
            health = " (healthy)" if state.get("health_ok") else ""
            print(
                f"{aid:<20} {m.get('type', '?'):<10} {m.get('default_port', '?'):<7} "
                f"{m.get('requires_plan', 'free'):<12} {status + health:<10} "
                f"{m.get('pack_id', '-')}"
            )
        return 0

    elif sub == "enable":
        from adk.addon_manager import AddonManager
        addon_id = args.addon_id
        config = {}
        if getattr(args, "endpoint", None):
            config["endpoint"] = args.endpoint
        mgr = AddonManager()
        try:
            inst = asyncio.run(mgr.enable(addon_id, config=config))
            print(f"Addon {addon_id} enabled")
            print(f"  Status:   {inst.status}")
            print(f"  Endpoint: {inst.endpoint}")
            print(f"  Health:   {'OK' if inst.health_ok else 'checking...'}")
            if inst.error_message:
                print(f"  Error:    {inst.error_message}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Failed to enable {addon_id}: {e}")
            return 1
        return 0

    elif sub == "disable":
        from adk.addon_manager import AddonManager
        mgr = AddonManager()
        asyncio.run(mgr.disable(args.addon_id))
        print(f"Addon {args.addon_id} disabled")
        return 0

    elif sub == "status":
        from adk.addon_manager import AddonManager
        mgr = AddonManager()
        addon_id = getattr(args, "addon_id", None)
        instances = asyncio.run(mgr.status(addon_id))
        if not instances:
            print("No addons enabled." if not addon_id else f"Addon {addon_id} not found.")
            return 0
        for inst in instances:
            print(f"Addon: {inst.addon_id}")
            print(f"  Status:    {inst.status}")
            print(f"  Type:      {inst.addon_type}")
            print(f"  Endpoint:  {inst.endpoint}")
            print(f"  Health:    {'OK' if inst.health_ok else 'FAIL'}")
            if inst.container_id:
                print(f"  Container: {inst.container_id[:12]}")
            if inst.error_message:
                print(f"  Error:     {inst.error_message}")
            print()
        return 0

    elif sub == "logs":
        from adk.addon_manager import AddonManager, _load_state
        state = _load_state()
        addon_state = state.get(args.addon_id, {})
        cid = addon_state.get("container_id", "")
        if not cid:
            print(f"No container found for {args.addon_id}")
            return 1
        from adk.addon_docker import container_logs
        lines = getattr(args, "lines", 100)
        print(container_logs(cid, tail=lines))
        return 0

    elif sub == "update":
        from adk.addon_manager import AddonManager, _load_state, load_addon_manifest
        from adk.addon_docker import pull_image
        state = _load_state()
        if not state:
            print("No addons enabled.")
            return 0
        for addon_id, s in state.items():
            manifest = load_addon_manifest(addon_id)
            if not manifest or manifest.get("type") != "docker":
                continue
            image = manifest.get("image", "")
            if image:
                print(f"Pulling latest: {image}")
                try:
                    pull_image(image)
                    print(f"  Updated {addon_id}")
                except Exception as e:
                    print(f"  Failed: {e}")
        return 0

    else:
        print("Usage: adk addon [list|enable|disable|status|logs|update]")
        return 1


def _load_pack_catalog(genesis_url: str) -> list[dict]:
    """Load pack catalog: Genesis API first, bundled offline catalog as fallback.

    The bundled catalog at ``adk/data/packs_catalog.json`` is auto-generated
    from ``AitherOS/config/packs_catalog.yaml`` by the ADK sync workflow.
    This ensures ``adk pack list`` and ``adk pack search`` always work,
    even without internet or a running Genesis.
    """
    catalog: list[dict] = []

    # Try Genesis API
    try:
        import httpx
        with httpx.Client(base_url=genesis_url, timeout=5) as c:
            resp = c.get("/api/v1/catalog/packs")
            if resp.status_code == 200:
                catalog = resp.json().get("packs", [])
            else:
                resp = c.get("/v1/packs/catalog")
                if resp.status_code == 200:
                    catalog = resp.json().get("packs", [])
    except Exception:
        pass

    # Fallback: bundled offline catalog
    if not catalog:
        try:
            bundled = Path(__file__).parent / "data" / "packs_catalog.json"
            if bundled.exists():
                catalog = json.loads(bundled.read_text(encoding="utf-8")).get("packs", [])
        except Exception:
            pass

    return catalog


def _cmd_pack(args) -> int:
    """Handle `adk pack` subcommands."""
    sub = getattr(args, "pack_command", None)
    genesis_url = _get_genesis_url()

    if sub == "list" or sub is None:
        # Fetch catalog: Genesis API → bundled offline catalog fallback
        catalog = _load_pack_catalog(genesis_url)

        # Check local packs
        packs_dir = Path.home() / ".aitheros" / "packs"
        local_ids = set()
        if packs_dir.is_dir():
            for child in packs_dir.iterdir():
                if child.is_dir() and (child / ".toolpack.yaml").exists():
                    local_ids.add(child.name)

        # Merge local-only
        catalog_ids = {p["id"] for p in catalog}
        for lid in sorted(local_ids - catalog_ids):
            catalog.append({"id": lid, "name": lid, "version": "local", "tier": "free", "installed": True})

        for entry in catalog:
            if entry["id"] in local_ids:
                entry["installed"] = True

        if not catalog:
            print("No packs found.")
            return 0

        print(f"{'Pack':<25} {'Type':<12} {'Tier':<14} {'Status':<12} {'Version'}")
        print("-" * 75)
        for p in catalog:
            pid = p.get("id", "?")
            ver = p.get("version", "?")
            ptype = p.get("type", "?").replace("_pack", "").replace("_", " ")
            tier = p.get("tier", "free")
            installed = p.get("installed", False)
            status = "installed" if installed else "available"

            print(f"{pid:<25} {ptype:<12} {tier:<14} {status:<12} {ver}")
        print(f"\n{len(catalog)} pack(s) total")
        return 0

    if sub == "search":
        query = getattr(args, "query", "")
        if not query:
            print("Usage: adk pack search <query>")
            return 1

        catalog = _load_pack_catalog(genesis_url)

        q = query.lower()
        matches = [
            p for p in catalog
            if q in p.get("name", "").lower()
            or q in p.get("description", "").lower()
            or q in p.get("id", "").lower()
            or any(q in tag.lower() for tag in p.get("tags", []))
        ]

        if getattr(args, "json_output", False):
            print(json.dumps({"query": query, "count": len(matches), "packs": matches}, indent=2))
            return 0

        if not matches:
            print(f"No packs matching '{query}'")
            return 0

        print(f"{'Pack':<25} {'Version':<10} {'Status':<14} {'Price'}")
        print("-" * 65)

        # Check local packs
        packs_dir = Path.home() / ".aitheros" / "packs"
        local_ids = set()
        if packs_dir.is_dir():
            for child in packs_dir.iterdir():
                if child.is_dir() and (child / ".toolpack.yaml").exists():
                    local_ids.add(child.name)

        for p in matches:
            pid = p.get("id", "?")
            ver = p.get("version", "?")
            pricing = p.get("pricing", {})
            is_free = not pricing
            installed = p.get("installed", False) or pid in local_ids
            licensed = p.get("licensed", False)

            if installed and (is_free or licensed):
                status = "installed"
            elif licensed:
                status = "licensed"
            elif installed:
                status = "installed"
            else:
                status = "available"

            if is_free:
                price = "free"
            else:
                cents = pricing.get("subscription_cents", 0)
                price = f"${int(cents) / 100:.0f}/mo" if cents else "paid"

            print(f"{pid:<25} {ver:<10} {status:<14} {price}")

        print(f"\n{len(matches)} pack(s) matching '{query}'")
        return 0

    if sub == "sync":
        # License-driven convergence: install every entitled pack not present.
        # Delegates to the shared PacksPlugin sync (download → verify → extract).
        try:
            from adk.shell.plugins.builtins.packs import PacksPlugin
        except Exception as e:  # noqa: BLE001
            print(f"pack sync unavailable: {e}")
            return 1
        plugin = PacksPlugin()
        # Customer node talks to its portal/gateway; the v1 routes live there.
        portal = (
            os.environ.get("AITHER_PORTAL_URL")
            or os.environ.get("AITHER_ELYSIUM_URL")
            or genesis_url
            or "https://portal.aitherium.com"
        ).rstrip("/")
        plugin._base_url = portal
        cfg = load_saved_config()
        token = cfg.get("api_key") or cfg.get("access_token") or os.environ.get("AITHER_API_KEY", "")
        if token:
            plugin.auth.set_auth(token, cfg.get("tenant_id"))
        sync_args = ["--dry-run"] if getattr(args, "dry_run", False) else []
        print(plugin._sync(sync_args))
        return 0

    if sub in ("buy", "negotiate"):
        import httpx as _httpx
        portal = (
            os.environ.get("AITHER_PORTAL_URL")
            or os.environ.get("AITHER_ELYSIUM_URL")
            or genesis_url
            or "https://portal.aitherium.com"
        ).rstrip("/")
        cfg = load_saved_config()
        token = cfg.get("api_key") or cfg.get("access_token") or os.environ.get("AITHER_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-Aither-Api-Key"] = token
        try:
            if sub == "negotiate":
                body = {"listing_id": args.pack_id, "offer_credits": int(args.offer),
                        "rationale": getattr(args, "why", "")}
                with _httpx.Client(timeout=30.0, verify=tls_verify()) as c:
                    r = c.post(f"{portal}/v1/marketplace/negotiate", json=body, headers=headers)
                data = r.json() if r.content else {}
                print(json.dumps(data, indent=2))
                if data.get("decision") == "accept" and data.get("negotiation_token"):
                    print(f"\nAccepted at {data.get('agreed_credits')} credits. Buy it with:")
                    print(f"  adk pack buy {args.pack_id} --token {data['negotiation_token']} --install")
                return 0
            # buy
            body = {"listing_id": args.pack_id}
            if getattr(args, "token", ""):
                body["negotiation_token"] = args.token
            with _httpx.Client(timeout=60.0, verify=tls_verify()) as c:
                r = c.post(f"{portal}/v1/marketplace/purchase", json=body, headers=headers)
            data = r.json() if r.content else {}
            print(json.dumps(data, indent=2))
            if r.status_code == 200 and data.get("ok") and getattr(args, "install", False):
                from adk.shell.plugins.builtins.packs import PacksPlugin
                p = PacksPlugin()
                p._base_url = portal
                if token:
                    p.auth.set_auth(token, cfg.get("tenant_id"))
                print("\n" + p._sync([]))
            return 0 if r.status_code == 200 else 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {type(e).__name__}: {e}")
            return 1

    if sub == "install":
        pack_id = args.pack_id

        # Deploy-based packs (grid, sovereign) — redirect to deploy command
        deploy_packs = {
            "grid-distributed": ("grid", "adk deploy grid"),
            "grid": ("grid", "adk deploy grid"),
        }
        if pack_id in deploy_packs:
            component, hint = deploy_packs[pack_id]
            print(f"\n  {pack_id} is an infrastructure pack — installing via deploy.\n")
            print(f"  Running: {hint}")
            print()
            # Delegate to deploy
            args.component = component
            args.dry_run = getattr(args, "dry_run", False)
            from adk.deploy import cmd_deploy_component
            return cmd_deploy_component(args)

        import httpx
        import hashlib
        import tarfile
        import shutil
        from io import BytesIO

        packs_dir = Path.home() / ".aitheros" / "packs"
        target = packs_dir / pack_id

        try:
            with httpx.Client(base_url=genesis_url, timeout=60) as c:
                resp = c.get(f"/v1/packs/{pack_id}/download")

            if resp.status_code == 402:
                detail = resp.json().get("detail", "License required")
                if isinstance(detail, dict):
                    detail = detail.get("message", "License required")
                print(f"License required: {detail}")
                print("Purchase at: https://portal.aitherium.com/marketplace")
                return 2

            if resp.status_code == 404:
                print(f"Pack '{pack_id}' not found")
                return 1

            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"Download failed: {e}")
            return 1
        except Exception as e:
            print(f"Cannot reach Genesis: {e}")
            return 1

        # Verify SHA
        expected_sha = resp.headers.get("X-Pack-SHA256", "")
        actual_sha = hashlib.sha256(resp.content).hexdigest()
        if expected_sha and actual_sha != expected_sha:
            print(f"SHA256 mismatch: expected {expected_sha[:16]}..., got {actual_sha[:16]}...")
            return 1

        # Extract
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        buf = BytesIO(resp.content)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    print(f"Unsafe path in archive: {member.name}")
                    return 1
            tar.extractall(target, filter="data")

        version = resp.headers.get("X-Pack-Version", "unknown")

        # Trigger MCP reload so new tools appear without manual restart
        mcp_reloaded = False
        try:
            httpx.post(f"{genesis_url}/mcp/reload", timeout=5)
            mcp_reloaded = True
        except Exception:
            pass  # Non-fatal — user can restart manually

        print(f"Pack '{pack_id}' v{version} installed to {target}")
        if mcp_reloaded:
            print("MCP tools reloaded — new tools are available now.")
        else:
            print("Restart `adk mcp serve` to activate.")

        # Auto-register with portal if pack has agent.yaml
        agent_yaml = target / pack_id / "agent.yaml"
        if not agent_yaml.exists():
            agent_yaml = target / "agent.yaml"
        if agent_yaml.exists():
            try:
                import yaml
                spec = yaml.safe_load(agent_yaml.read_text(encoding="utf-8")) or {}
                if spec.get("portal"):
                    import asyncio
                    from adk.registration import register_with_portal
                    success = asyncio.run(register_with_portal(spec))
                    if success:
                        print(f"Registered '{spec.get('name', pack_id)}' with portal.")
                    else:
                        print("Portal registration skipped (portal unreachable).")
            except Exception as e:
                print(f"Portal registration skipped: {e}")

        return 0

    if sub == "remove":
        import shutil
        target = Path.home() / ".aitheros" / "packs" / args.pack_id
        if not target.is_dir():
            print(f"Pack '{args.pack_id}' is not installed")
            return 1
        shutil.rmtree(target)

        # Trigger MCP reload so removed tools disappear without manual restart
        mcp_reloaded = False
        try:
            import httpx
            httpx.post(f"{genesis_url}/mcp/reload", timeout=5)
            mcp_reloaded = True
        except Exception:
            pass  # Non-fatal — user can restart manually

        if mcp_reloaded:
            print(f"Pack '{args.pack_id}' removed. MCP tools reloaded.")
        else:
            print(f"Pack '{args.pack_id}' removed. Restart `adk mcp serve` to take effect.")
        return 0

    if sub == "info":
        import httpx
        pack_id = args.pack_id
        try:
            with httpx.Client(base_url=genesis_url, timeout=10) as c:
                # Try new catalog API first
                resp = c.get(f"/api/v1/catalog/packs/{pack_id}")
                if resp.status_code == 404:
                    # Fallback to old API
                    resp = c.get(f"/v1/packs/{pack_id}/manifest")
                if resp.status_code == 404:
                    print(f"Pack '{pack_id}' not found")
                    return 1
                resp.raise_for_status()
                data = resp.json()
                # New API wraps in {"pack": {...}}
                if "pack" in data:
                    data = data["pack"]
        except Exception as e:
            print(f"Cannot fetch pack info: {e}")
            return 1

        print(f"Pack:        {data.get('name', pack_id)}")
        print(f"ID:          {data.get('id', pack_id)}")
        print(f"Version:     {data.get('version', '?')}")
        print(f"Type:        {data.get('type', '?')}")
        print(f"Tier:        {data.get('tier', 'free')}")
        print(f"Category:    {data.get('category', '?')}")
        print(f"Description: {data.get('description', '')}")

        price = data.get("price_monthly_usd")
        if price is None:
            print("Price:       included in tier")
        elif price == 0:
            print("Price:       free")
        else:
            print(f"Price:       ${price}/mo add-on")

        deps = data.get("depends_on", [])
        if deps:
            print(f"Depends on:  {', '.join(deps)}")

        includes = data.get("includes", [])
        if includes:
            print(f"Includes:    {', '.join(includes)}")

        agents = data.get("agents", [])
        if agents:
            print(f"\nAgents ({len(agents)}):")
            for a in agents:
                identity = a.get("identity", "?").replace(".yaml", "")
                print(f"  - {identity} (effort cap: {a.get('effort_cap', '?')})")

        skills = data.get("skills", [])
        if skills:
            print(f"Skills:      {', '.join(skills)}")

        mcp_modules = data.get("mcp_modules", [])
        if mcp_modules:
            print(f"\nMCP Modules ({len(mcp_modules)}):")
            for m in mcp_modules:
                print(f"  - {m}")

        tool_count = data.get("tool_count")
        if tool_count:
            print(f"Tool count:  {tool_count}")

        skill_files = data.get("skill_files", [])
        if skill_files:
            print(f"\nSkill Files ({len(skill_files)}):")
            for sf in skill_files:
                print(f"  - {sf}")

        services = data.get("services", [])
        if services:
            print(f"\nServices ({len(services)}):")
            for s in services:
                print(f"  - {s.get('addon_id', '?')} (port {s.get('port', '?')})")

        # Fetch dependency tree
        try:
            with httpx.Client(base_url=genesis_url, timeout=10) as c:
                deps_resp = c.get(f"/api/v1/catalog/packs/{pack_id}/deps")
                if deps_resp.status_code == 200:
                    deps_data = deps_resp.json()
                    order = deps_data.get("install_order", [])
                    if len(order) > 1:
                        print(f"\nInstall order: {' -> '.join(order)}")
        except Exception:
            pass

        return 0

    if sub == "update":
        import httpx
        pack_id = getattr(args, "pack_id", None)
        packs_dir = Path.home() / ".aitheros" / "packs"

        if pack_id:
            # Update specific pack
            target = packs_dir / pack_id
            if not target.is_dir():
                print(f"Pack '{pack_id}' is not installed")
                return 1
            print(f"Updating {pack_id}...")
            args.pack_id = pack_id
            # Re-use install logic
            return _cmd_pack(type("Args", (), {"pack_command": "install", "pack_id": pack_id})())
        else:
            # Update all installed packs
            if not packs_dir.is_dir():
                print("No packs installed")
                return 0
            installed = [
                d.name for d in packs_dir.iterdir()
                if d.is_dir() and (d / ".toolpack.yaml").exists()
            ]
            if not installed:
                print("No packs installed")
                return 0
            print(f"Updating {len(installed)} pack(s)...")
            for pid in installed:
                print(f"  Updating {pid}...")
                _cmd_pack(type("Args", (), {"pack_command": "install", "pack_id": pid})())
            print("All packs updated.")
            return 0

    if sub == "export":
        pack_ids = getattr(args, "pack_ids", "").split(",")
        output = getattr(args, "output", ".")
        if not pack_ids or not pack_ids[0]:
            print("Usage: adk pack export <pack-ids> [-o output_dir]")
            return 1

        import httpx
        try:
            with httpx.Client(base_url=genesis_url, timeout=30) as c:
                resp = c.post("/api/v1/catalog/resolve", json={"packs": pack_ids})
                if resp.status_code != 200:
                    print(f"Failed to resolve packs: {resp.text}")
                    return 1
                resolved = resp.json()
        except Exception as e:
            print(f"Cannot reach Genesis: {e}")
            return 1

        order = resolved.get("install_order", [])
        print(f"Resolved {len(order)} packs: {', '.join(order)}")
        print(f"Export to: {output}")
        print("(Offline export bundles are an enterprise feature — contact sales@aitherium.com)")
        return 0

    if sub == "customize":
        import yaml
        from adk.pack_discovery import load_agent_spec

        pack_name = args.pack_id
        pack_dir = Path.home() / ".aither" / "agents" / pack_name

        if not pack_dir.exists():
            print(f"ERROR: Pack '{pack_name}' not installed.")
            print(f"Install it with: adk install pack:{pack_name}")
            return 1

        # Parse options
        system_prompt = None
        capabilities = None
        show_spec = getattr(args, "show", False)

        if getattr(args, "system_prompt_file", None):
            try:
                sp = Path(getattr(args, "system_prompt_file")).expanduser()
                system_prompt = sp.read_text(encoding="utf-8")
            except Exception as e:
                print(f"ERROR: Failed to read system prompt file: {e}")
                return 1

        if getattr(args, "system_prompt", None):
            system_prompt = args.system_prompt

        if getattr(args, "capabilities", None):
            capabilities = [c.strip() for c in args.capabilities.split(",")]

        # If --show and no overrides, just display current spec
        if show_spec and not system_prompt and not capabilities:
            spec = load_agent_spec(pack_dir / "agent.yaml")
            if not spec:
                print("No agent.yaml found in pack")
                return 1
            print(f"Pack '{pack_name}' current spec:")
            print()
            for key, value in sorted(spec.items()):
                if isinstance(value, (list, tuple)):
                    val_str = "[" + ", ".join(str(v) for v in value) + "]"
                elif isinstance(value, dict):
                    val_str = "{...}"
                elif isinstance(value, str) and len(value) > 80:
                    val_str = value[:80] + "..."
                else:
                    val_str = str(value)
                print(f"  {key}: {val_str}")
            return 0

        # Build overlay
        if not system_prompt and not capabilities:
            print("ERROR: No options provided. Use --system-prompt, --system-prompt-file, --capabilities, or --show")
            return 1

        overlay = {}
        if system_prompt:
            if len(system_prompt) > 8000:
                print("ERROR: system_prompt exceeds 8000 character limit")
                return 1
            overlay["system_prompt"] = system_prompt

        if capabilities:
            overlay["capabilities"] = capabilities

        # Write agent.yaml.local
        local_yaml = pack_dir / "agent.yaml.local"
        try:
            local_yaml.write_text(yaml.dump(overlay, default_flow_style=False), encoding="utf-8")
        except Exception as e:
            print(f"ERROR: Failed to write agent.yaml.local: {e}")
            return 1

        # Show confirmation
        print(f"✓ Customized pack '{pack_name}'")
        print()
        print(f"Overrides saved to: {local_yaml}")
        print()
        if system_prompt:
            prompt_preview = system_prompt[:100].replace("\n", " ")
            print(f"  system_prompt: {prompt_preview}...")
        if capabilities:
            print(f"  capabilities: {', '.join(capabilities)}")
        print()
        print(f"Apply with: adk run --agents {pack_name}")
        return 0

    print("Usage: adk pack [list|search|install|remove|info|update|export|customize]")
    return 1


def _cmd_skills(args) -> int:
    """Handle `adk skills` subcommands."""
    import json as _json

    sub = getattr(args, "skills_command", None)
    try:
        from adk.skills import SkillStore
    except ImportError:
        print("Skills module not available.")
        return 1

    store = SkillStore()

    if sub == "list" or sub is None:
        skills = store.list_all()
        if not skills:
            print("No skills learned yet.")
            return 0
        print(f"{'Name':<30} {'Uses':<8} {'Tags'}")
        print("-" * 60)
        for s in skills:
            tags = ", ".join(s.tags[:3]) if s.tags else ""
            print(f"{s.name:<30} {s.success_count:<8} {tags}")
        return 0

    if sub == "search":
        results = store.search(args.query)
        if not results:
            print(f"No skills matching '{args.query}'")
            return 0
        for s in results:
            print(f"  {s.name}: {s.description}")
        return 0

    if sub == "export":
        data = store.export_agentskills()
        print(_json.dumps(data, indent=2))
        return 0

    print("Usage: adk skills [list|search|export]")
    return 1


def _cmd_listen(args) -> int:
    """Handle `adk listen` subcommands — real-time audio intelligence."""
    sub = getattr(args, "listen_command", None)

    genesis_url = os.environ.get("AITHER_GENESIS_URL",
                                  os.environ.get("AITHER_URL", "http://localhost:8001"))

    if sub in ("audiobook", "meeting", "note"):
        try:
            import httpx
        except ImportError:
            print("httpx required: pip install httpx")
            return 1

        if sub == "audiobook":
            body = {
                "mode": "audiobook",
                "book_title": getattr(args, "title", ""),
                "author": getattr(args, "author", ""),
                "genre": getattr(args, "genre", "litrpg"),
                "capture_backend": getattr(args, "backend", "wasapi"),
                "audio_source": getattr(args, "audio_file", None),
                "workspace_id": getattr(args, "workspace", None),
            }
        elif sub == "meeting":
            mt = getattr(args, "meeting_type", "meeting")
            body = {
                "mode": "lecture" if mt == "lecture" else "meeting",
                "meeting_title": getattr(args, "title", ""),
                "meeting_type": mt,
                "participants": getattr(args, "participants", []),
                "capture_backend": getattr(args, "backend", "wasapi"),
                "workspace_id": getattr(args, "workspace", None),
            }
        else:  # note
            body = {
                "mode": "voice_note",
                "meeting_title": getattr(args, "title", "Voice Note"),
                "capture_backend": getattr(args, "backend", "wasapi"),
                "workspace_id": getattr(args, "workspace", None),
            }

        with httpx.Client(base_url=genesis_url, timeout=30) as c:
            resp = c.post("/audiobook/start", json=body)
            if resp.status_code != 200:
                print(f"Error: {resp.text}")
                return 1
            data = resp.json()

        sid = data.get("session_id", "")
        mode_label = {"audiobook": "Audiobook companion",
                      "meeting": "Meeting transcription",
                      "note": "Voice note"}[sub]
        print(f"{mode_label} started: {sid[:12]}")
        print(f"  Stop:   adk listen stop {sid[:12]}")
        print(f"  Export: adk listen export {sid[:12]}")
        return 0

    elif sub == "sessions":
        import httpx
        with httpx.Client(base_url=genesis_url, timeout=10) as c:
            resp = c.get("/audiobook/sessions")
            if resp.status_code != 200:
                print(f"Error: {resp.text}")
                return 1
            sessions = resp.json().get("sessions", [])

        if not sessions:
            print("No active sessions.")
        for s in sessions:
            sid = s.get("session_id", "")[:12]
            title = s.get("book_title", "Untitled")
            chunks = s.get("chunks_processed", 0)
            print(f"  {sid}  {title:<30}  {chunks} chunks")
        return 0

    elif sub == "stop":
        import httpx
        session_id = args.session_id
        # Partial ID matching
        if len(session_id) < 36:
            with httpx.Client(base_url=genesis_url, timeout=10) as c:
                resp = c.get("/audiobook/sessions")
                if resp.status_code == 200:
                    sessions = resp.json().get("sessions", [])
                    matches = [s for s in sessions
                               if s["session_id"].startswith(session_id)]
                    if len(matches) == 1:
                        session_id = matches[0]["session_id"]
                    elif len(matches) > 1:
                        print(f"Ambiguous ID '{session_id}' — {len(matches)} matches.")
                        return 1

        with httpx.Client(base_url=genesis_url, timeout=15) as c:
            resp = c.post(f"/audiobook/{session_id}/stop")
            if resp.status_code != 200:
                print(f"Error: {resp.text}")
                return 1
        print(f"Session stopped: {session_id[:12]}")
        return 0

    elif sub == "export":
        import httpx
        session_id = args.session_id
        fmt = getattr(args, "fmt", "notes")
        output = getattr(args, "output", None)

        with httpx.Client(base_url=genesis_url, timeout=15) as c:
            resp = c.get(f"/audiobook/{session_id}/export/{fmt}")
            if resp.status_code != 200:
                print(f"Error: {resp.text}")
                return 1
            data = resp.json()

        content = data.get("content") or data.get("transcript", "")
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Exported to {output}")
        else:
            print(content)
        return 0

    print("Usage: adk listen [audiobook|meeting|note|sessions|stop|export]")
    return 1


def _cmd_soul(args) -> int:
    """Handle `adk soul` subcommands."""
    sub = getattr(args, "soul_command", None)

    if sub == "import":
        from pathlib import Path
        from adk.identity import load_soul_md, Identity

        path = Path(args.path)
        if not path.exists():
            print(f"File not found: {path}")
            return 1

        config = load_soul_md(path)
        identity = Identity(**{k: v for k, v in config.items() if k != "knowledge"})
        print(f"Imported identity: {identity.name}")
        if identity.description:
            print(f"  Description: {identity.description}")
        if identity.skills:
            print(f"  Skills: {', '.join(identity.skills)}")
        if config.get("knowledge"):
            print(f"  Knowledge: {len(config['knowledge'])} chars")

        # Save as YAML
        import yaml
        out_path = Path("identities") / f"{identity.name}.yaml"
        out_path.parent.mkdir(exist_ok=True)
        data = {
            "name": identity.name,
            "role": identity.role,
            "description": identity.description,
            "skills": identity.skills,
        }
        if identity.system_prompt:
            data["system_prompt"] = identity.system_prompt
        out_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        print(f"  Saved to: {out_path}")
        return 0

    if sub == "export":
        from adk.identity import load_identity, export_soul_md

        identity = load_identity(args.name)
        print(export_soul_md(identity))
        return 0

    print("Usage: adk soul [import|export]")
    return 1


# ---------------------------------------------------------------------------
# adk train — training pipeline management
# ---------------------------------------------------------------------------

def _get_genesis_url() -> str:
    """Resolve the training API URL — Genesis, ADK server, or Aitherium cloud.

    Priority:
    1. AITHER_GENESIS_URL env var (explicit Genesis)
    2. elysium_url from saved config (remote connected desktop)
    3. Local Genesis on :8001 (if reachable)
    4. Local ADK server on configured port (if reachable)
    5. Aitherium cloud gateway (if API key available)
    6. Fallback to localhost:8001 (user gets a clear error)
    """
    import urllib.request
    import urllib.error

    cfg = load_saved_config()

    # Explicit env var
    explicit = os.environ.get("AITHER_GENESIS_URL", "")
    if explicit:
        return explicit.rstrip("/")

    # Remote connected instance
    elysium = cfg.get("elysium_url") or cfg.get("aither_gateway_url", "")
    if elysium:
        return elysium.rstrip("/")

    # Probe local Genesis
    try:
        req = urllib.request.Request("http://localhost:8001/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return "http://localhost:8001"
    except (urllib.error.URLError, ConnectionError, OSError):
        pass

    # Probe local ADK server
    server_port = cfg.get("server_port", 8080)
    try:
        req = urllib.request.Request(f"http://localhost:{server_port}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return f"http://localhost:{server_port}"
    except (urllib.error.URLError, ConnectionError, OSError):
        pass

    # Cloud gateway (for users without local Genesis)
    api_key = cfg.get("api_key") or os.environ.get("AITHER_API_KEY", "")
    if api_key:
        return "https://gateway.aitherium.com"

    # Fallback — will produce a clear error when the call fails
    return "http://localhost:8001"


def _train_headers() -> dict:
    """Build auth headers for Genesis training API calls."""
    cfg = load_saved_config()
    api_key = cfg.get("api_key") or os.environ.get("AITHER_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    tenant_id = cfg.get("tenant_id") or os.environ.get("AITHER_TENANT_ID", "")
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    return headers


def _cmd_train(args) -> int:
    import json as _json
    import urllib.request
    import urllib.error

    genesis = _get_genesis_url()
    headers = _train_headers()
    sub = getattr(args, "train_command", None)

    def _api(method: str, path: str, body: dict | None = None) -> dict:
        data = _json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            f"{genesis}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return _json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                err = _json.loads(exc.read())
                detail = err.get("detail", str(exc))
            except Exception:
                detail = str(exc)
            print(f"  Error ({exc.code}): {detail}")
            return {"error": detail}
        except (urllib.error.URLError, OSError) as exc:
            print(f"  Cannot reach Genesis at {genesis}: {exc}")
            return {"error": str(exc)}

    if sub == "status":
        print(f"  Connecting to {genesis}...")
        dashboard = _api("GET", "/training/dashboard")
        if "error" in dashboard:
            return 1
        readiness = _api("GET", "/training/pipeline/readiness")

        print()
        print("  Training Dashboard")
        print(f"  {'='*40}")
        print(f"  Total Runs:     {dashboard.get('total_runs', 0)}")
        print(f"  Active Runs:    {dashboard.get('active_runs', 0)}")
        print(f"  Successful:     {dashboard.get('successful_runs', 0)}")
        print(f"  Failed:         {dashboard.get('failed_runs', 0)}")
        print(f"  Total Cost:     ${dashboard.get('total_cost_usd', 0):.2f}")
        print(f"  GPU Hours:      {dashboard.get('total_gpu_hours', 0):.1f}")
        print()
        ready = readiness.get("ready", False)
        count = readiness.get("count", 0)
        threshold = readiness.get("threshold", 50)
        status_icon = "READY" if ready else "NOT READY"
        print(f"  Corpus: {status_icon} ({count}/{threshold} examples)")
        return 0

    elif sub == "launch":
        body = {
            "model_preset": args.preset,
            "target_gpu": args.gpu,
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "max_gpu_price": args.max_price,
            "auto_benchmark": not args.no_benchmark,
            "auto_deploy": args.auto_deploy,
        }
        if args.dataset:
            body["dataset_url"] = args.dataset

        print(f"  Launching training: {args.preset} on {args.gpu}...")
        result = _api("POST", "/training/orchestrate", body)
        if "error" in result:
            return 1
        run_id = result.get("run_id", "???")
        print(f"  Training run started: {run_id}")
        print(f"  Monitor: adk train logs {run_id[:12]}")
        return 0

    elif sub == "logs":
        run_id = args.run_id
        result = _api("GET", f"/training/runs/{run_id}/logs?lines={args.lines}")
        if "error" in result:
            return 1
        logs = result.get("logs", result.get("stdout", ""))
        if logs:
            print(logs)
        else:
            print("  No logs available yet.")
        return 0

    elif sub == "cancel":
        run_id = args.run_id
        result = _api("POST", f"/training/runs/{run_id}/cancel")
        if "error" in result:
            return 1
        print(f"  Training run {run_id} cancelled.")
        return 0

    elif sub == "runs":
        result = _api("GET", "/training/runs")
        if "error" in result:
            return 1
        runs = result.get("runs", result) if isinstance(result, dict) else result
        if not isinstance(runs, list):
            runs = []
        if not runs:
            print("  No training runs found.")
            return 0

        status_filter = getattr(args, "status", None)
        if status_filter:
            runs = [r for r in runs if r.get("status") == status_filter]

        print()
        print(f"  {'RUN ID':<20} {'STATUS':<15} {'MODEL':<25} {'GPU':<15} {'COST':>8}")
        print(f"  {'-'*20} {'-'*15} {'-'*25} {'-'*15} {'-'*8}")
        for run in runs[:20]:
            rid = (run.get("run_id", "")[:18] or "???")
            status = run.get("status", "?")
            model = (run.get("model_preset", "?")[:23] or "?")
            gpu = (run.get("gpu_name", "") or run.get("target_gpu", "?"))[:13]
            cost = run.get("cost_usd", 0)
            print(f"  {rid:<20} {status:<15} {model:<25} {gpu:<15} ${cost:>7.2f}")
        return 0

    elif sub == "register-gpu":
        host = args.host
        port = args.port
        gpu_model = getattr(args, "gpu_model", None) or ""
        vram = getattr(args, "vram", None) or 0

        # Auto-detect GPU if not specified
        if not gpu_model or not vram:
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    if not gpu_model:
                        gpu_model = props.name
                    if not vram:
                        vram = props.total_mem // (1024 ** 3)
                    print(f"  Detected: {gpu_model} ({vram}GB)")
                else:
                    print("  Warning: No CUDA GPU detected locally.")
            except ImportError:
                print("  Warning: PyTorch not installed, cannot auto-detect GPU.")

        body = {
            "name": f"adk-{host}-gpu",
            "host": host,
            "port": port,
            "capabilities": ["training", "inference"],
            "gpu_model": gpu_model,
            "gpu_vram_gb": vram,
            "node_type": "gpu_node",
            "location": "workstation",
        }
        result = _api("POST", "/compute/nodes/register", body)
        if "error" in result:
            # Try gateway mesh as fallback
            result = _api("POST", "/gateway/nodes/register", body)
            if "error" in result:
                print("  Failed to register GPU node.")
                return 1

        node_id = result.get("node_id", result.get("id", ""))
        print(f"  GPU registered: {gpu_model} ({vram}GB) at {host}:{port}")
        if node_id:
            print(f"  Node ID: {node_id}")
        print("  This GPU is now available for training via 'adk train launch --gpu customer'")
        return 0

    else:
        print("Usage: adk train [status|launch|logs|cancel|runs|register-gpu]")
        print()
        print("  status         Check training readiness and dashboard stats")
        print("  launch         Launch a new training run")
        print("  logs <run_id>  Stream training logs")
        print("  cancel <id>    Cancel an active run")
        print("  runs           List recent training runs")
        print("  register-gpu   Register your local GPU for remote training")
        return 1


# ---------------------------------------------------------------------------
# Slash-command manifest — auto-generates from argparse for AitherShell
# ---------------------------------------------------------------------------

def build_command_manifest() -> list[dict]:
    """Return a structured manifest of all CLI commands for AitherShell slash-command auto-discovery.

    AitherShell queries GET /slash-commands on the ADK server, gets this manifest,
    and registers each command as a /name slash command with tab-completion.

    Returns list of: {name, help, args: [{name, flags, required, type, default, help, choices}], subcommands: [...]}
    """
    # Build a temporary parser just for introspection (never calls parse_args)
    p = argparse.ArgumentParser(prog="adk")
    sub = p.add_subparsers(dest="command")

    # Import and call the registration block — we'll inline a lightweight
    # version that just registers the parser structure, not the dispatch.
    # This avoids import-time side effects.
    _register_commands(sub)

    commands = []
    for action in p._subparsers._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, subparser in action.choices.items():
            cmd = {"name": name, "help": "", "args": [], "subcommands": []}
            for ca in action._choices_actions:
                if ca.dest == name:
                    cmd["help"] = ca.help or ""
                    break
            for act in subparser._actions:
                if isinstance(act, argparse._HelpAction):
                    continue
                if isinstance(act, argparse._SubParsersAction):
                    for sn, sp in act.choices.items():
                        sc = {"name": sn, "help": "", "args": []}
                        for sca in act._choices_actions:
                            if sca.dest == sn:
                                sc["help"] = sca.help or ""
                                break
                        for sa in sp._actions:
                            if isinstance(sa, (argparse._HelpAction, argparse._SubParsersAction)):
                                continue
                            ai = _extract_arg(sa)
                            sc["args"].append(ai)
                        cmd["subcommands"].append(sc)
                    continue
                cmd["args"].append(_extract_arg(act))
            commands.append(cmd)
    return commands


def _extract_arg(act) -> dict:
    """Extract argument info from an argparse action."""
    ai = {
        "name": act.dest,
        "flags": act.option_strings or [],
        "required": getattr(act, "required", not bool(act.option_strings)),
        "type": act.type.__name__ if act.type else "str",
        "default": act.default if act.default != argparse.SUPPRESS else None,
        "help": act.help or "",
    }
    if act.choices:
        ai["choices"] = list(act.choices)
    return ai


_cached_parser: argparse.ArgumentParser | None = None


def get_parser() -> argparse.ArgumentParser:
    """Return the full CLI parser (cached). Used by manifest builder and main()."""
    global _cached_parser
    if _cached_parser is None:
        # Trigger main() to build it on first call — or build fresh
        _cached_parser = argparse.ArgumentParser(
            prog="adk",
            description="AitherADK — Build AI agent fleets with any LLM backend",
        )
        # We'll populate it in main() and cache it
    return _cached_parser


def _register_commands(sub):
    """Register all CLI subcommands on the given subparsers group.

    Shared between main() (for execution) and build_command_manifest() (for introspection).
    """
    # adk start — the main entry point for everyone
    start_p = sub.add_parser("start", help="Start chatting with your codebase (zero config)")
    start_p.add_argument("path", nargs="?", default=".", help="Project directory (default: current)")

    # aither init
    init_p = sub.add_parser("init", help="Scaffold a new agent project")
    init_p.add_argument("name", nargs="?", default="my-agent", help="Project/agent name")
    init_p.add_argument("-d", "--directory", help="Target directory (default: ./<name>)")

    # aither new <template> — scaffold a full template app (e.g. deep-research)
    new_p = sub.add_parser("new", help="Scaffold a full template app (e.g. deep-research)")
    new_p.add_argument("template", help="Template name, e.g. deep-research")
    new_p.add_argument("-d", "--directory", help="Target directory (default: ./<template>)")

    # aither run
    run_p = sub.add_parser("run", help="Start the agent server")
    run_p.add_argument("-i", "--identity", help="Agent identity")
    run_p.add_argument("-p", "--port", type=int, help="Server port")
    run_p.add_argument("--host", help="Server host")
    run_p.add_argument("-b", "--backend", help="LLM backend")
    run_p.add_argument("-m", "--model", help="Model name")
    run_p.add_argument("-f", "--fleet", help="Fleet YAML config")
    run_p.add_argument("-a", "--agents", help="Comma-separated agent identities")
    run_p.add_argument("--mesh", action="store_true",
                       help="Enable mesh hosting (advertise tools/inference to connected desktop)")

    # adk up — one command: run a persistent, fleet-connected agent
    up_p = sub.add_parser(
        "up", help="Run a persistent agent connected to your AitherOS fleet (one command)")
    up_p.add_argument("--identity", default="aither", help="Agent identity (default: aither)")
    up_p.add_argument("--name", help="Fleet label for this agent (default: <hostname>-adk)")
    up_p.add_argument("--provider", help="Cloud provider if no local LLM: deepseek/openai/anthropic")
    up_p.add_argument("--model", help="Model name (default: the provider's default)")
    up_p.add_argument("--port", type=int, default=8080, help="Local port for aither-serve (default: 8080)")
    up_p.add_argument("--yes", "--non-interactive", action="store_true", dest="yes",
                      help="Non-interactive: zero prompts, machine-readable JSON (for AI agents/CI)")
    up_p.add_argument("--foreground", action="store_true",
                      help="Block the terminal instead of detaching")
    up_p.add_argument("--no-persist", action="store_true",
                      help="Do not install a reboot-autostart entry")
    up_p.add_argument("--force", action="store_true", help="Restart even if an agent is already running")
    up_p.add_argument("--offline", action="store_true",
                      help="Sovereign/local-only: no tunnel, no portal (or set AITHER_OFFLINE=1)")
    up_p.add_argument("--no-register", action="store_true",
                      help="Run locally only — no tunnel, no fleet registration")
    up_p.add_argument("--require-register", action="store_true",
                      help="Fail (non-zero) if the fleet registration cannot complete")
    up_p.add_argument("--token", help="Portal token for registration (else 'adk login' / $AITHER_PORTAL_TOKEN)")
    up_p.add_argument("--auth-token", help="Callback bearer the control plane presents back (minted if omitted)")
    up_p.add_argument("--approve", help="Comma-list of tools that pause for approval (default: file_write,shell_exec,shell)")
    up_p.add_argument("--portal", default="https://veil.aitherium.com", help="Control-plane base URL")
    up_p.add_argument("--login-url", help="Device-flow login base URL")
    up_p.add_argument("--register-url", help="Full fleet-register URL (overrides --portal)")
    up_p.add_argument("--dry-run", action="store_true", help="Show what would happen without starting anything")

    # adk down — stop the agent started by `adk up`
    down_p = sub.add_parser("down", help="Stop the agent + tunnel and remove its autostart")
    down_p.add_argument("--keep-autostart", action="store_true",
                        help="Stop now but leave the reboot-autostart entry in place")

    # adk stack — supervise the consumer stack (Room + Ollama); was `adk up`
    stack_p = sub.add_parser("stack", help="Start the consumer stack (Room + Ollama) as native processes")
    stack_p.add_argument("service", nargs="?", default="default",
                         help="Service to start: default (Room+Ollama), qdrant (local Qdrant)")
    stack_p.add_argument("--interval", type=int, default=10,
                         help="Health check interval in seconds (default: 10)")
    stack_p.add_argument("--no-sync", action="store_true",
                         help="Skip the license-driven pack sync before starting")

    # adk wizard — first-run setup
    wizard_p = sub.add_parser("wizard", help="First-run wizard — hardware detection, setup recommendations, auth token")
    wizard_p.add_argument("--yes", action="store_true",
                         help="Non-interactive mode: accept defaults without prompts")

    # aither register
    register_p = sub.add_parser("register", help="Create a new Aitherium account")
    register_p.add_argument("--email", help="Account email (prompted if omitted)")
    register_p.add_argument("--password", help="Account password (prompted if omitted)")

    # adk login
    login_p = sub.add_parser("login", help="Authenticate with Aitherium (browser device flow)")
    login_p.add_argument("--email", help="Use email/password instead of browser flow")
    login_p.add_argument("--password", help="Password (prompted if --email given without it)")
    login_p.add_argument("--api-key", help="Save an API key directly (no login flow)")
    login_p.add_argument("--portal-url", default="",
                         help="Portal/Identity URL (default: portal.aitherium.com)")

    # adk whoami
    sub.add_parser("whoami", help="Show current auth status and config")

    # adk logout
    sub.add_parser("logout", help="Clear saved auth tokens")

    # adk agent-prompt
    ap_p = sub.add_parser("agent-prompt", help="Print the setup prompt for AI coding agents")
    ap_p.add_argument("--raw", action="store_true", help="Print raw prompt without footer")

    # aither connect
    connect_p = sub.add_parser("connect", help="Connect to AitherOS — detect LLMs, set up gateway, or join desktop mesh")
    connect_p.add_argument("--api-key", help="AITHER_API_KEY for cloud inference")
    connect_p.add_argument("--elysium", metavar="URL",
                           help="Connect to desktop AitherOS (e.g. http://192.168.1.10:8001)")
    connect_p.add_argument("--token", help="Node token for desktop mesh authentication")
    connect_p.add_argument("--save", action="store_true", default=True,
                           help="Save config to ~/.aither/config.json (default: true)")
    connect_p.add_argument("--no-save", action="store_false", dest="save",
                           help="Don't save config")

    # aither setup
    setup_p = sub.add_parser("setup", help="Interactive GPU setup wizard (vLLM/Ollama) + optional AitherOS stack")
    setup_p.add_argument("shortcut", nargs="?", default=None,
                         help="Quick setup: 'nemotron' (--tier lite), 'llamacpp' / 'local' / 'endpoint' (native local orchestrator, no Docker)")
    setup_p.add_argument("--mode", choices=["auto", "cloud", "hybrid"],
                         default="auto",
                         help="Setup mode: auto (detect GPU), cloud (cloud-only, no GPU), hybrid (local + cloud reasoning)")
    setup_p.add_argument("--tier", choices=["nano", "lite", "standard", "standard-tq4", "full", "hybrid", "hybrid-tq4", "ollama", "llamacpp"],
                         help="Force a specific tier (default: auto-detect from GPU). 'llamacpp' = native local Nemotron-Orchestrator-8B for endpoints (Windows/macOS/Linux, no Docker)")
    setup_p.add_argument("--backend", choices=["vllm", "ollama", "llamacpp"], default=None,
                         help="Backend engine override (default inferred from --tier)")
    setup_p.add_argument("--llamacpp-quant", default=None,
                         help="llama.cpp GGUF quant (e.g. Q4_K_M, Q5_K_M, Q8_0). Default: auto-pick from VRAM/RAM")
    setup_p.add_argument("--llamacpp-port", type=int, default=None,
                         help="llama.cpp server port (default: 8200)")
    setup_p.add_argument("--no-service", action="store_true",
                         help="llama.cpp: skip installing system service (systemd/launchd/scheduled task)")
    setup_p.add_argument("--reasoning-api", choices=["anthropic", "openai", "deepseek", "gateway"],
                         help="Cloud API for reasoning (effort 7+) — hybrid mode")
    setup_p.add_argument("--reasoning-model", default="",
                         help="Specific model for reasoning backend")
    setup_p.add_argument("--dgx-spark", metavar="URL",
                         help="DGX Spark / remote vLLM URL (e.g. http://192.168.0.33:8000)")
    setup_p.add_argument("--stack", choices=["minimal", "core", "full", "headless", "gpu", "agents"],
                         help="Also deploy AitherOS services via AitherZero")
    setup_p.add_argument("--dry-run", action="store_true",
                         help="Show what would happen without making changes")
    setup_p.add_argument("--non-interactive", action="store_true",
                         help="No prompts — auto-accept defaults (for CI/automation)")
    setup_p.add_argument("--hf-token", default="",
                         help="HuggingFace token for gated models")
    setup_p.add_argument("--api-key", help="AITHER_API_KEY for cloud + stack deployment")
    setup_p.add_argument("--output", default="docker-compose.vllm.yml",
                         help="Output compose file path (default: docker-compose.vllm.yml)")
    setup_p.add_argument("--force", action="store_true",
                         help="Start new containers even if inference is already running")

    # aither aeon
    aeon_p = sub.add_parser("aeon", help="Multi-agent group chat")
    aeon_p.add_argument("-p", "--preset", help="Preset: balanced, creative, technical, security, minimal, duo_code, research")
    aeon_p.add_argument("-a", "--agents", help="Comma-separated agent names (e.g. demiurge,athena)")
    aeon_p.add_argument("-r", "--rounds", type=int, default=1, help="Discussion rounds per message (default: 1)")
    aeon_p.add_argument("--no-synthesize", action="store_true", help="Skip orchestrator synthesis")

    # adk create-app — scaffold a full portal-kit workspace app
    ca_p = sub.add_parser("create-app",
                          help="Scaffold a portal-kit workspace app (like GargBot, Chelle)")
    ca_p.add_argument("name", help="App name (e.g. 'ACME Assistant')")
    ca_p.add_argument("-o", "--output", help="Output directory (default: ./<slug>)")
    ca_p.add_argument("--company", default="", help="Company name")
    ca_p.add_argument("--industry", default="general", help="Industry vertical")
    ca_p.add_argument("--description", default="", help="What this app does")
    ca_p.add_argument("--subdomain", default="", help="URL slug (auto-derived from name)")
    ca_p.add_argument("--color", default="#6366f1", help="Primary brand color")
    ca_p.add_argument("--template", default="default",
                      choices=["default", "gargbot", "chelle", "wildroot"],
                      help="Base template (default: default)")
    ca_p.add_argument("--llm-provider", dest="llm_provider", default="aitheros",
                      choices=["aitheros", "ollama", "portal", "deepseek", "openai", "anthropic", "gemini"],
                      help="LLM provider (default: aitheros)")
    ca_p.add_argument("--force", action="store_true", help="Overwrite existing directory")

    # aither deploy — component deployment OR agent deployment
    deploy_p = sub.add_parser("deploy", help="Deploy AitherOS components or agents")
    deploy_sub = deploy_p.add_subparsers(dest="component")

    # aither deploy ollama
    d_ollama = deploy_sub.add_parser("ollama", help="Install Ollama + pull models for your GPU")
    d_ollama.add_argument("--models", help="Comma-separated model list (default: auto-select by GPU)")
    d_ollama.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy vllm
    d_vllm = deploy_sub.add_parser("vllm", help="Deploy vLLM containers directly (use 'adk setup' for guided wizard)")
    d_vllm.add_argument("--tier", choices=["nano", "lite", "standard", "full"],
                        help="Force a specific tier (default: auto-detect)")
    d_vllm.add_argument("--hf-token", default="", help="HuggingFace token for gated models")
    d_vllm.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy node
    d_node = deploy_sub.add_parser(
        "node",
        help="Deploy agent node (default: ADK-native lightweight; --genesis for full stack)",
    )
    d_node.add_argument("--genesis", action="store_true",
                         help="Use full Genesis stack (14+ containers) instead of ADK-native (2 containers)")
    d_node.add_argument("--gpu", action="store_true", help="Enable GPU-accelerated vLLM")
    d_node.add_argument("--dashboard", action="store_true", help="Enable workspace dashboard (port 3000)")
    d_node.add_argument("--mesh", action="store_true", help="Enable mesh networking (Genesis mode only)")
    d_node.add_argument("--memory", action="store_true", help="Enable persistent vector memory (Spirit)")
    d_node.add_argument("--addons", help="Comma-separated addon IDs to co-deploy (e.g. qdrant,knowledge-rag)")
    d_node.add_argument("--tag", default="latest", help="Docker image tag (default: latest)")
    d_node.add_argument("--api-key", help="AITHER_API_KEY (or set env var)")
    d_node.add_argument("--dry-run", action="store_true", help="Show what would happen")
    d_node.add_argument("--sovereign", action="store_true",
                         help="Register with Aitherium hub after deployment (federation)")
    d_node.add_argument("--hub", default="https://portal.aitherium.com",
                         help="Hub URL for federation (default: portal.aitherium.com)")
    d_node.add_argument("--tenant", help="Tenant slug for federation registration")
    d_node.add_argument("--federate", action="store_true",
                         help="Also start AitherFederate (port 8094) — agent-fleet "
                              "registration, product-catalog sync, and knowledge "
                              "ingestion routing to portal.aitherium.com. Requires "
                              "--portal-token or AITHER_PORTAL_TOKEN.")
    d_node.add_argument("--portal-token", help="AITHER_PORTAL_TOKEN for --federate (or set env var)")

    # aither deploy core
    d_core = deploy_sub.add_parser("core", help="Core services (Node, Pulse, Watch, Genesis, Veil)")
    d_core.add_argument("--addons", help="Comma-separated addon IDs to co-deploy")
    d_core.add_argument("--tag", default="latest", help="Docker image tag (default: latest)")
    d_core.add_argument("--api-key", help="AITHER_API_KEY (or set env var)")
    d_core.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy full
    d_full = deploy_sub.add_parser("full", help="Full AitherOS stack (~31 containers)")
    d_full.add_argument("--profile", default="chat-agents",
                        choices=["chat-minimal", "chat-full", "chat-agents"],
                        help="Docker Compose profile (default: chat-agents)")
    d_full.add_argument("--addons", help="Comma-separated addon IDs to co-deploy")
    d_full.add_argument("--tag", default="latest", help="Docker image tag (default: latest)")
    d_full.add_argument("--api-key", help="AITHER_API_KEY (or set env var)")
    d_full.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy fleet-refresh
    d_fleet = deploy_sub.add_parser(
        "fleet-refresh",
        help="Rebuild all lib-baking Python images + safe rolling recreate of the running fleet")
    d_fleet.add_argument("--build-only", action="store_true", help="Rebuild images only")
    d_fleet.add_argument("--recreate-only", action="store_true", help="Recreate from existing images")
    d_fleet.add_argument("--dry-run", action="store_true", help="Print the target list, do nothing")

    # aither deploy addons
    d_addons = deploy_sub.add_parser("addons", help="Deploy self-hosted addon services")
    d_addons.add_argument("addon_ids", nargs="*", help="Addon IDs (default: all available)")
    d_addons.add_argument("--list", dest="list_addons", action="store_true",
                          help="List available addons without deploying")
    d_addons.add_argument("--tag", default="latest", help="Docker image tag (default: latest)")
    d_addons.add_argument("--api-key", help="AITHER_API_KEY (or set env var)")
    d_addons.add_argument("--dry-run", action="store_true", help="Show what would happen")
    d_addons.add_argument("--sovereign", action="store_true",
                          help="Register with federation hub after deployment")
    d_addons.add_argument("--hub", default="https://portal.aitherium.com",
                          help="Hub URL for federation")
    d_addons.add_argument("--tenant", help="Tenant slug for federation registration")

    # aither deploy connect
    d_connect = deploy_sub.add_parser("connect", help="AitherConnect browser extension")
    d_connect.add_argument("--api-key", help="AITHER_API_KEY (or set env var)")
    d_connect.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy desktop
    d_desktop = deploy_sub.add_parser("desktop", help="AitherDesktop native application")
    d_desktop.add_argument("--api-key", help="AITHER_API_KEY (or set env var)")
    d_desktop.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy gargbot
    d_gargbot = deploy_sub.add_parser("gargbot", help="Deploy GargBot sovereign package (setup + compose + health)")
    d_gargbot.add_argument("--tier", choices=["lite", "entry", "pro", "pro-reasoning", "full"],
                           help="Force a specific tier (default: auto-detect)")
    d_gargbot.add_argument("--no-pull", action="store_true", help="Skip image pulling")
    d_gargbot.add_argument("--start", action="store_true", default=True,
                           help="Start services after setup (default: true)")
    d_gargbot.add_argument("--no-start", action="store_true", help="Generate config only, don't start")
    d_gargbot.add_argument("--api-key", help="AITHER_API_KEY for portal federation")
    d_gargbot.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy grid
    d_grid = deploy_sub.add_parser(
        "grid",
        help="Deploy grid distributed stack (GPU + Mac + cluster)",
    )
    d_grid.add_argument("--mac-host", help="Mac Mini IP for Ollama reasoning")
    d_grid.add_argument("--cluster-nodes", help="JSON array of cluster node IPs")
    d_grid.add_argument("--hf-token", default="", help="HuggingFace token for gated models")
    d_grid.add_argument("--skip-health", action="store_true",
                           help="Skip remote node health checks")
    d_grid.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # aither deploy stop <component>
    d_stop = deploy_sub.add_parser("stop", help="Stop a running deployment")
    d_stop.add_argument("stop_target", nargs="?",
                        help="Component to stop: ollama, vllm, node, core, full, all")

    # aither deploy agent — download + configure + start a tenant agent, OR upload to gateway
    d_agent = deploy_sub.add_parser("agent", help="Deploy a tenant agent to this machine (or upload to gateway)")
    d_agent.add_argument("name", nargs="?", help="Agent name/slug (e.g. gargbot)")
    d_agent.add_argument("-d", "--directory", help="Project directory (default: .)")
    d_agent.add_argument("--api-key", help="AITHER_API_KEY")
    d_agent.add_argument("--gateway", help="Gateway URL (default: gateway.aitherium.com)")
    d_agent.add_argument("--capabilities", help="Comma-separated capabilities")
    d_agent.add_argument("--description", help="Agent description")
    d_agent.add_argument("--version", help="Agent version")
    d_agent.add_argument("--target", choices=["gateway", "docker", "kubernetes", "systemd", "cloud-gpu"],
                          default="gateway", help="Deploy target (default: gateway)")
    d_agent.add_argument("--strategy", choices=["rolling", "blue-green", "canary", "recreate"],
                          default="rolling", help="Deployment strategy (for container targets)")
    d_agent.add_argument("--tenant", help="Tenant slug — triggers download+run mode (e.g. garg-consulting)")
    d_agent.add_argument("--inference", choices=["local", "cloud", "hybrid"],
                          help="Inference mode for this endpoint")
    d_agent.add_argument("--from", dest="from_url", help="Direct download URL for the agent package")

    # adk workspace — manage dev workspaces on AitherOS tunnel
    ws_p = sub.add_parser("workspace", help="Manage dev workspaces on AitherOS tunnel")
    ws_sub = ws_p.add_subparsers(dest="ws_command")

    ws_create = ws_sub.add_parser("create", help="Create a cloud dev workspace")
    ws_create.add_argument("--scope", default="fullstack",
                           help="Scope template: fullstack, gargbot, chelle, veil, portal, frontend, backend, etc.")
    ws_create.add_argument("--tunnel-url", default="https://tunnel.aitherium.com",
                           help="Tunnel URL (default: tunnel.aitherium.com)")

    ws_bundle = ws_sub.add_parser("bundle", help="Download a dev workspace bundle (docker-compose + WireGuard)")
    ws_bundle.add_argument("--scope", default="fullstack",
                           help="Scope template")
    ws_bundle.add_argument("-o", "--output", default="aitheros-devws.zip",
                           help="Output zip file path")
    ws_bundle.add_argument("--tunnel-url", default="https://tunnel.aitherium.com",
                           help="Tunnel URL")

    ws_list = ws_sub.add_parser("list", help="List your active workspaces")
    ws_list.add_argument("--tunnel-url", default="https://tunnel.aitherium.com",
                         help="Tunnel URL")

    ws_submit = ws_sub.add_parser("submit", help="Submit changes from workspace (commit + PR)")
    ws_submit.add_argument("message", help="Commit message")
    ws_submit.add_argument("--workspace", help="Workspace container name (auto-detected if in one)")
    ws_submit.add_argument("--tunnel-url", default="https://tunnel.aitherium.com",
                           help="Tunnel URL")

    ws_sub.add_parser("scopes", help="List available scope templates")

    # aither onboard — interactive onboarding wizard
    onboard_p = sub.add_parser("onboard", help="Interactive onboarding — detect, configure, integrate")
    onboard_p.add_argument("--api-key", help="AITHER_API_KEY")
    onboard_p.add_argument("--tenant", help="Tenant slug to associate this node with")
    onboard_p.add_argument("--agent", help="Register a running agent with the portal fleet")
    onboard_p.add_argument("--non-interactive", action="store_true", help="Skip prompts, use defaults")

    # adk enroll — register workstation with control plane
    enroll_p = sub.add_parser("enroll", help="Register this workstation with the control plane")
    enroll_p.add_argument("--portal", help="Portal URL (default: portal.aitherium.com)")
    enroll_p.add_argument("--genesis", help="Genesis URL (default: localhost:8001)")
    enroll_p.add_argument("--no-heartbeat", action="store_true", help="Skip background heartbeat")
    enroll_p.add_argument("--force", action="store_true", help="Re-enroll even if already registered")

    # aither host — one command: serve a self-hosted agent + connect it to your fleet
    host_p = sub.add_parser(
        "host",
        help="Host a self-hosted agent (your model key) + connect it to your fleet — one command",
    )
    host_p.add_argument("--provider", help="Model provider: deepseek/openai/anthropic (prompted if omitted)")
    host_p.add_argument("--model", help="Model name (default: the provider's default)")
    host_p.add_argument("--name", help="Fleet label for this agent (default: <hostname>-adk)")
    host_p.add_argument("--identity", default="aither", help="Agent identity to load (default: aither)")
    host_p.add_argument("--port", type=int, default=8080, help="Local port for aither-serve (default: 8080)")
    host_p.add_argument("--approve", help="Comma-list of tools that pause for approval, or '*' (default: file_write,shell_exec,shell)")
    host_p.add_argument("--token", help="Control-plane token for registration (else 'adk login' / $AITHER_PORTAL_TOKEN)")
    host_p.add_argument("--auth-token", help="Callback bearer the control plane presents back to your agent (minted if omitted)")
    host_p.add_argument("--portal", default="https://veil.aitherium.com", help="Control-plane base URL")
    host_p.add_argument("--login-url", help="Device-flow login base URL (default: --portal, then portal.aitherium.com)")
    host_p.add_argument("--register-url", help="Full fleet-register URL (overrides --portal; e.g. http://localhost:8001/v1/agent/fleet/register)")
    host_p.add_argument("--no-register", action="store_true", help="Run locally only — no tunnel, no fleet registration")
    host_p.add_argument("--dry-run", action="store_true", help="Show what would happen without starting anything")

    # aither integrate — connect external tools
    integrate_p = sub.add_parser("integrate", help="Connect external tools (OpenClaw, etc.)")
    integrate_p.add_argument("target", nargs="?", default="list",
                             help="Integration target: openclaw, list")
    integrate_p.add_argument("--mode", choices=["local", "cloud", "hybrid", "auto"],
                             help="Integration mode (default: auto-detect)")
    integrate_p.add_argument("--api-key", help="AITHER_API_KEY for cloud mode")
    integrate_p.add_argument("--dry-run", action="store_true",
                             help="Show config without writing")
    integrate_p.add_argument("--force", action="store_true",
                             help="Overwrite existing integration config")

    # adk index — index a codebase for CodeGraph
    index_p = sub.add_parser("index", help="Index a codebase for code search (CodeGraph)")
    index_p.add_argument("path", nargs="?", default=".", help="Path to index (default: current directory)")
    index_p.add_argument("--embed", action="store_true", help="Also generate embeddings for semantic search")
    index_p.add_argument("--stats", action="store_true", help="Show Python metrics after indexing")

    # adk test
    test_p = sub.add_parser("test", help="Run agent tests")
    test_p.add_argument("-d", "--directory", help="Project directory (default: .)")
    test_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    test_p.add_argument("--coverage", action="store_true", help="Show coverage report")

    # adk status
    status_p = sub.add_parser("status", help="Show backend and service status")
    status_p.add_argument("--json", action="store_true",
                          help="Machine-readable JSON (agent state) for AI agents/CI")

    # adk publish — submit to Elysium marketplace
    publish_p = sub.add_parser("publish", help="Publish agent to Elysium marketplace")
    publish_p.add_argument("name", nargs="?", help="Agent name (default: from config.yaml)")
    publish_p.add_argument("-d", "--directory", help="Project directory (default: .)")
    publish_p.add_argument("--api-key", help="AITHER_API_KEY")
    publish_p.add_argument("--gateway", help="Gateway URL (default: gateway.aitherium.com)")
    publish_p.add_argument("--description", help="Agent description for marketplace")
    publish_p.add_argument("--capabilities", help="Comma-separated capabilities")
    publish_p.add_argument("--version", help="Agent version (default: 0.1.0)")
    publish_p.add_argument("--pricing", default="free",
                           help="Pricing model: free, per_request, flat_monthly")
    publish_p.add_argument("--tier", default="agent",
                           help="Agent tier: reflex, agent, reasoning, orchestrator")
    publish_p.add_argument("--category", default="general",
                           help="Category: general, engineering, content, research, security")
    publish_p.add_argument("--dry-run", action="store_true",
                           help="Validate without publishing")

    # adk admin — administration commands
    admin_p = sub.add_parser("admin", help="Administration commands")
    admin_sub = admin_p.add_subparsers(dest="admin_command")
    admin_token_p = admin_sub.add_parser("create-token",
                                          help="Create a node token on the desktop for mesh enrollment")
    admin_token_p.add_argument("--name", default="", help="Node name (default: hostname)")
    admin_token_p.add_argument("--url", default="http://localhost:8001",
                               help="Genesis URL (default: http://localhost:8001)")

    # adk disconnect — leave desktop mesh
    sub.add_parser("disconnect", help="Disconnect from desktop AitherOS mesh")

    # adk backend — manage LLM backends
    backend_p = sub.add_parser("backend", help="Manage LLM backends (list, set, test, switch, status)")
    backend_sub = backend_p.add_subparsers(dest="backend_command")
    backend_sub.add_parser("list", help="Show detected and configured backends")
    backend_set_p = backend_sub.add_parser("set", help="Set default backend")
    backend_set_p.add_argument("provider", help="Provider: ollama, vllm, openai, anthropic, deepseek, groq, together, gateway")
    backend_set_p.add_argument("--api-key", help="API key for the provider")
    backend_set_p.add_argument("--base-url", help="Custom base URL")
    backend_set_p.add_argument("--model", help="Default model")
    backend_reason_p = backend_sub.add_parser("set-reasoning", help="Set reasoning-only backend (effort 7+)")
    backend_reason_p.add_argument("provider", help="Provider for reasoning tasks")
    backend_reason_p.add_argument("--api-key", help="API key")
    backend_reason_p.add_argument("--model", help="Reasoning model")
    backend_sub.add_parser("test", help="Test current backend with a simple prompt")
    backend_switch_p = backend_sub.add_parser("switch", help="Switch to a different inference backend")
    backend_switch_p.add_argument("target_backend", choices=["ollama", "llamacpp", "vllm"],
                                  help="Target backend to switch to")
    backend_sub.add_parser("status", help="Show current backend configuration and connectivity")

    # adk keys — manage cloud provider API keys
    keys_p = sub.add_parser("keys", help="Manage cloud provider API keys (set, list, test, remove)")
    keys_sub = keys_p.add_subparsers(dest="keys_command")
    keys_set_p = keys_sub.add_parser("set", help="Set a provider API key")
    keys_set_p.add_argument("provider", help="Provider: openai, anthropic, deepseek, google, openrouter, groq, together")
    keys_set_p.add_argument("key", help="API key value")
    keys_sub.add_parser("list", help="Show configured provider keys and status")
    keys_test_p = keys_sub.add_parser("test", help="Test API keys (all or specific)")
    keys_test_p.add_argument("provider", nargs="?", help="Specific provider to test (default: all)")
    keys_rm_p = keys_sub.add_parser("remove", help="Remove a provider key")
    keys_rm_p.add_argument("provider", help="Provider to remove")

    # adk grid — manage grid distributed infrastructure
    grid_p = sub.add_parser("grid", help="Manage grid distributed nodes (add, remove, list, test, sync)")
    grid_sub = grid_p.add_subparsers(dest="grid_command")
    grid_sub.add_parser("status", help="Show grid topology and health of all nodes")
    grid_add_p = grid_sub.add_parser("add", help="Add a node to the grid")
    grid_add_p.add_argument("role", choices=["reasoning", "cluster"], help="Node role")
    grid_add_p.add_argument("host", help="Hostname or IP address")
    grid_add_p.add_argument("--port", type=int, default=8121, help="llama.cpp port (default: 8121)")
    grid_add_p.add_argument("--model", help="Model name override")
    grid_rm_p = grid_sub.add_parser("remove", help="Remove a node from the grid")
    grid_rm_p.add_argument("host", help="Hostname or IP to remove")
    grid_test_p = grid_sub.add_parser("test", help="Test connectivity to all or specific nodes")
    grid_test_p.add_argument("host", nargs="?", help="Specific host to test (default: all)")
    grid_sub.add_parser("sync", help="Sync grid config to your Aitherium workspace (requires login)")
    grid_sub.add_parser("pull", help="Pull grid config from your Aitherium workspace")
    # Mesh registry (the platform's enrolled nodes, via the AitherGateway tunnel)
    grid_enroll_p = grid_sub.add_parser("enroll", help="Mint a single-use token to onboard a remote machine as a mesh node")
    grid_enroll_p.add_argument("--ttl", type=float, default=1.0, help="Token lifetime in hours (default 1, max 24)")
    grid_enroll_p.add_argument("--tenant", default="", help="Attribute the node to a tenant slug")
    grid_enroll_p.add_argument("--label", default="", help="Human label for the node")
    grid_sub.add_parser("ls", help="List enrolled mesh nodes (GPU, memory, containers, status)")
    grid_rm_mesh_p = grid_sub.add_parser("deregister", help="Remove an enrolled mesh node from the registry")
    grid_rm_mesh_p.add_argument("node_id", help="Node id or name to deregister")

    # adk routing — per-intent model routing
    routing_p = sub.add_parser("routing", help="Manage per-intent model routing (which model handles which task)")
    routing_sub = routing_p.add_subparsers(dest="routing_command")
    routing_preset_p = routing_sub.add_parser("preset", help="Apply a routing preset (budget, balanced, quality)")
    routing_preset_p.add_argument("preset_name", help="Preset: budget, balanced, quality")
    routing_set_p = routing_sub.add_parser("set", help="Set model for an intent type")
    routing_set_p.add_argument("intent", help="Intent: code, reasoning, chat, research, review, planning, search")
    routing_set_p.add_argument("provider", help="Provider: openai, anthropic, deepseek, local")
    routing_set_p.add_argument("--model", help="Specific model name")
    routing_sub.add_parser("reset", help="Reset to effort-based routing (disable intent overrides)")

    # adk costs — token economy visibility
    costs_p = sub.add_parser("costs", help="Show cloud inference costs, savings, and budget")
    costs_sub = costs_p.add_subparsers(dest="costs_command")
    costs_sub.add_parser("summary", help="Show cost summary (default)")
    costs_compare_p = costs_sub.add_parser("compare", help="Compare AitherOS vs raw API costs")
    costs_compare_p.add_argument("--period", default="week", choices=["day", "week", "month"])
    costs_budget_p = costs_sub.add_parser("budget", help="Set monthly spending budget")
    costs_budget_p.add_argument("amount", type=float, help="Monthly budget in USD (0=unlimited)")
    costs_p.add_argument("--period", default="day", choices=["day", "week", "month"], help="Time period")

    # adk tools — list available tools
    tools_p = sub.add_parser("tools", help="List available tools (local + MCP)")
    tools_p.add_argument("--upgrade", action="store_true", help="Show what pro/enterprise unlocks")

    # adk quickstart — unified first-run wizard
    quickstart_p = sub.add_parser("quickstart", help="One-command setup: GPU + auth + shell")
    quickstart_p.add_argument("--api-key", help="AITHER_API_KEY")
    quickstart_p.add_argument("--cloud", action="store_true", help="Cloud-only setup (no GPU required)")

    # adk quickstart-local — local-only inference quickstart
    quickstart_local_p = sub.add_parser(
        "quickstart-local",
        help="Local inference quickstart (no cloud required)"
    )
    quickstart_local_p.add_argument(
        "--backend",
        choices=["auto", "llamacpp", "ollama", "vllm"],
        default="auto",
        help="Inference backend (auto = detect best fit)"
    )
    quickstart_local_p.add_argument(
        "--model",
        help="Model name/ID (for Ollama; others auto-detected)"
    )
    quickstart_local_p.add_argument(
        "--port",
        type=int,
        default=8200,
        help="Port for local inference endpoint (default: 8200)"
    )
    quickstart_local_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes"
    )
    quickstart_local_p.add_argument("--api-key", help="AITHER_API_KEY")

    # adk backup — export all ~/.aither/ data
    backup_p = sub.add_parser("backup", help="Backup all agent data (memory, graphs, config)")
    backup_p.add_argument("-o", "--output", help="Output file path (default: aither-backup-<timestamp>.tar.gz)")

    # adk ingest — manually ingest files into knowledge graph
    ingest_p = sub.add_parser("ingest", help="Ingest files into the agent's knowledge graph")
    ingest_p.add_argument("path", nargs="?", default=".", help="File or directory to ingest")
    ingest_p.add_argument("--agent", default="default", help="Agent name for the graph")
    ingest_p.add_argument("--brain", action="store_true",
                         help="Enable sync to CompanyBrain hub (default: local only)")
    ingest_p.add_argument("--brain-url", default="",
                         help="Override brain hub URL (default: from env/config)")
    ingest_p.add_argument("--classification", default="internal",
                         choices=["public", "internal", "confidential", "restricted"],
                         help="Classification level for ingested content (default: internal)")
    ingest_p.add_argument("--chunk-size", type=int, default=2000,
                         help="Bytes per chunk (default: 2000)")
    ingest_p.add_argument("--chunk-overlap", type=int, default=200,
                         help="Overlap bytes between chunks (default: 200)")
    ingest_p.add_argument("--workspace", default="default",
                         help="Workspace ID for brain sync (default: default)")
    ingest_p.add_argument("--skip-embeddings", action="store_true",
                         help="Skip embedding if brain unreachable")
    ingest_p.add_argument("--dry-run", action="store_true",
                         help="Print what would be ingested without persisting")

    # adk doctor — system health checks
    sub.add_parser("doctor", help="Check system health (Python, GPU, LLM backends, API keys)")

    # adk gateway — multi-channel agent gateway
    gateway_p = sub.add_parser("gateway", help="Run agent across messaging platforms")
    gateway_p.add_argument("-a", "--agent", default="assistant", help="Agent identity (default: assistant)")
    gateway_p.add_argument("--telegram", action="store_true", help="Enable Telegram (TELEGRAM_BOT_TOKEN)")
    gateway_p.add_argument("--discord", action="store_true", help="Enable Discord (DISCORD_BOT_TOKEN)")
    gateway_p.add_argument("--slack", action="store_true", help="Enable Slack (SLACK_BOT_TOKEN + SLACK_APP_TOKEN)")
    gateway_p.add_argument("--webhook", action="store_true", help="Enable webhook endpoint")
    gateway_p.add_argument("--webhook-port", type=int, default=9000, help="Webhook port (default: 9000)")

    # adk cron — cron scheduler
    cron_p = sub.add_parser("cron", help="Manage scheduled tasks")
    cron_sub = cron_p.add_subparsers(dest="cron_command")
    cron_sub.add_parser("list", help="List scheduled jobs")
    cron_add_p = cron_sub.add_parser("add", help="Add a cron job")
    cron_add_p.add_argument("expression", help="Cron expression (e.g. '0 9 * * *')")
    cron_add_p.add_argument("task_name", help="Task name / description")
    cron_rm_p = cron_sub.add_parser("remove", help="Remove a cron job")
    cron_rm_p.add_argument("name", help="Job name to remove")

    # adk skills — skill management
    skills_p = sub.add_parser("skills", help="Manage learned skills")
    skills_sub = skills_p.add_subparsers(dest="skills_command")
    skills_sub.add_parser("list", help="List all learned skills")
    skills_search_p = skills_sub.add_parser("search", help="Search skills")
    skills_search_p.add_argument("query", help="Search query")
    skills_sub.add_parser("export", help="Export skills in agentskills.io format")

    # adk addon — self-hosted addon management
    addon_p = sub.add_parser("addon", help="Manage self-hosted service addons (Qdrant, RAG, CodeGraph, etc.)")
    addon_sub = addon_p.add_subparsers(dest="addon_command")
    addon_sub.add_parser("list", help="Show available addons + status")
    addon_enable_p = addon_sub.add_parser("enable", help="Pull image, start container, register with portal")
    addon_enable_p.add_argument("addon_id", help="Addon ID to enable (e.g. qdrant, knowledge-rag)")
    addon_enable_p.add_argument("--endpoint", help="Endpoint URL (for external type addons)")
    addon_disable_p = addon_sub.add_parser("disable", help="Stop container, deregister")
    addon_disable_p.add_argument("addon_id", help="Addon ID to disable")
    addon_status_p = addon_sub.add_parser("status", help="Health + metrics for addons")
    addon_status_p.add_argument("addon_id", nargs="?", help="Specific addon (default: all)")
    addon_logs_p = addon_sub.add_parser("logs", help="Tail container logs")
    addon_logs_p.add_argument("addon_id", help="Addon ID")
    addon_logs_p.add_argument("--lines", type=int, default=100, help="Number of log lines (default: 100)")
    addon_sub.add_parser("update", help="Pull latest images for all enabled addons")

    # adk install — install agent packs and extensions
    install_p = sub.add_parser("install", help="Install an agent pack (e.g. adk install pack:openclaw)")
    install_p.add_argument(
        "target", nargs="?", default=None,
        help="'list', 'pack:<name>', or a pack name (openclaw, hermes, claude-code)",
    )

    # adk packs — alias for list
    sub.add_parser("packs", help="List available agent packs")

    # adk pack — tool pack management
    pack_p = sub.add_parser("pack", help="Manage ToolPack extensions (list, search, install, remove, info)")
    pack_sub = pack_p.add_subparsers(dest="pack_command")
    pack_sub.add_parser("list", help="List available and installed packs")
    pack_search_p = pack_sub.add_parser("search", help="Search packs by name, description, or tags")
    pack_search_p.add_argument("query", help="Search query")
    pack_search_p.add_argument("--json", dest="json_output", action="store_true", help="JSON output")
    pack_install_p = pack_sub.add_parser("install", help="Install a tool pack")
    pack_install_p.add_argument("pack_id", help="Pack ID to install")
    pack_sync_p = pack_sub.add_parser(
        "sync", help="Install every entitled pack not already present (license-driven)")
    pack_sync_p.add_argument("--dry-run", "-n", action="store_true",
                             help="Preview what would be installed without installing")
    pack_buy_p = pack_sub.add_parser(
        "buy", help="Autonomously buy a pack with Aitherium credits (no Stripe)")
    pack_buy_p.add_argument("pack_id", help="Listing id to buy")
    pack_buy_p.add_argument("--token", default="", help="Negotiation token (agreed price)")
    pack_buy_p.add_argument("--install", action="store_true", help="Run pack sync after buying")
    pack_neg_p = pack_sub.add_parser(
        "negotiate", help="Haggle with the seller Broker for a better price")
    pack_neg_p.add_argument("pack_id", help="Listing id to negotiate")
    pack_neg_p.add_argument("--offer", type=int, required=True, help="Your offer in Aitherium credits")
    pack_neg_p.add_argument("--why", default="", help="Optional rationale for the Broker")
    pack_remove_p = pack_sub.add_parser("remove", help="Remove an installed pack")
    pack_remove_p.add_argument("pack_id", help="Pack ID to remove")
    pack_update_p = pack_sub.add_parser("update", help="Update one or all installed packs")
    pack_update_p.add_argument("pack_id", nargs="?", help="Pack ID to update (omit for all)")
    pack_export_p = pack_sub.add_parser("export", help="Export offline bundle (.tar.gz)")
    pack_export_p.add_argument("pack_ids", help="Comma-separated pack IDs")
    pack_export_p.add_argument("-o", "--output", default=".", help="Output directory")
    pack_info_p = pack_sub.add_parser("info", help="Show pack details")
    pack_info_p.add_argument("pack_id", help="Pack ID to inspect")
    pack_customize_p = pack_sub.add_parser("customize", help="Customize installed pack (system_prompt, capabilities, domains)")
    pack_customize_p.add_argument("pack_id", help="Pack name to customize")
    pack_customize_p.add_argument("--system-prompt", help="Override system prompt")
    pack_customize_p.add_argument("--system-prompt-file", help="Load system prompt from file")
    pack_customize_p.add_argument("--capabilities", help="Override capabilities (comma-separated)")
    pack_customize_p.add_argument("--show", action="store_true", help="Show current spec")

    # adk support — help and community links
    sub.add_parser("support", help="Get help — Discord, GitHub, docs")

    # adk explore — browse marketplace catalog
    explore_p = sub.add_parser("explore", help="Browse packs, agents, and skills in the Aitherium marketplace")
    explore_p.add_argument("category", nargs="?", default="all",
                           help="Filter: agents, tools, skills, grid, all (default: all)")
    explore_p.add_argument("--free", action="store_true", help="Show only free packs")

    # adk upgrade — open checkout page
    upgrade_p = sub.add_parser("upgrade", help="Open upgrade/checkout page for a pack or plan")
    upgrade_p.add_argument("target", nargs="?", default="",
                           help="Pack ID or plan: managed, setup, grid, demiurge, pro")

    # adk soul — SOUL.md import/export
    soul_p = sub.add_parser("soul", help="Import/export SOUL.md identity files")
    soul_sub = soul_p.add_subparsers(dest="soul_command")
    soul_import_p = soul_sub.add_parser("import", help="Import a SOUL.md file")
    soul_import_p.add_argument("path", help="Path to SOUL.md file")
    soul_export_p = soul_sub.add_parser("export", help="Export identity as SOUL.md")
    soul_export_p.add_argument("name", help="Identity name to export")

    # adk doc — encrypted document storage
    doc_p = sub.add_parser("doc", help="Manage encrypted documents (upload, list, download, delete)")
    doc_sub = doc_p.add_subparsers(dest="doc_command")
    doc_upload_p = doc_sub.add_parser("upload", help="Upload and encrypt a document")
    doc_upload_p.add_argument("path", help="Path to document file")
    doc_upload_p.add_argument("--type", help="Document MIME type (optional, e.g. application/pdf)")
    doc_upload_p.add_argument("--gateway", help="Gateway/Genesis URL (default: http://localhost:8001)")
    doc_upload_p.add_argument("--api-key", help="API key (or set AITHER_API_KEY env var)")
    doc_list_p = doc_sub.add_parser("list", help="List your encrypted documents")
    doc_list_p.add_argument("--gateway", help="Gateway/Genesis URL (default: http://localhost:8001)")
    doc_list_p.add_argument("--api-key", help="API key (or set AITHER_API_KEY env var)")
    doc_get_p = doc_sub.add_parser("get", help="Download and decrypt a document")
    doc_get_p.add_argument("doc_id", help="Document UUID")
    doc_get_p.add_argument("-o", "--output", help="Output file path (required)")
    doc_get_p.add_argument("--gateway", help="Gateway/Genesis URL (default: http://localhost:8001)")
    doc_get_p.add_argument("--api-key", help="API key (or set AITHER_API_KEY env var)")
    doc_delete_p = doc_sub.add_parser("delete", help="Delete a document")
    doc_delete_p.add_argument("doc_id", help="Document UUID")
    doc_delete_p.add_argument("--gateway", help="Gateway/Genesis URL (default: http://localhost:8001)")
    doc_delete_p.add_argument("--api-key", help="API key (or set AITHER_API_KEY env var)")

    # adk mcp — MCP server (stdio for Claude Code, or config helper)
    mcp_p = sub.add_parser("mcp", help="MCP server, IDE setup, and cloud gateway connection")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command")
    mcp_serve_p = mcp_sub.add_parser("serve", help="Start stdio MCP server (for Claude Code)")
    mcp_serve_p.add_argument("-d", "--directory", default=".", help="Agent project directory")
    mcp_serve_p.add_argument("-p", "--port", type=int,
                             help="Print HTTP config for a running server instead of stdio")
    mcp_config_p = mcp_sub.add_parser("config", help="Print MCP client configuration")
    mcp_config_p.add_argument("-p", "--port", type=int, default=8080,
                              help="ADK server port (default: 8080)")
    mcp_config_p.add_argument("-m", "--mode", choices=["stdio", "http"], default="stdio",
                              help="Transport mode (default: stdio)")
    # adk mcp setup — generate IDE config for cloud or local MCP gateway
    mcp_setup_p = mcp_sub.add_parser("setup", help="Generate IDE config (.mcp.json) for MCP gateway")
    mcp_setup_p.add_argument("--mode", choices=["local", "remote"], default="local",
                             help="local = Docker gateway (8182), remote = mcp.aitherium.com")
    mcp_setup_p.add_argument("--ide", choices=["claude-code", "cursor", "windsurf", "vscode"],
                             default="claude-code", help="Target IDE")
    mcp_setup_p.add_argument("--project-dir", default=".", help="Project directory for config file")
    mcp_setup_p.add_argument("--bake-token", action="store_true",
                             help="Bake auth token into headers (fallback for IDEs without OAuth)")
    # adk mcp node — lightweight local MCP server
    mcp_node_p = mcp_sub.add_parser("node", help="Start lightweight local MCP server")
    mcp_node_p.add_argument("--mode", choices=["proxy", "standalone"], default="proxy",
                            help="proxy = forward to cloud, standalone = local tools only")
    mcp_node_p.add_argument("-p", "--port", type=int, default=8182, help="Port (default: 8182)")
    # adk mcp status — check gateway connectivity
    mcp_sub.add_parser("status", help="Check MCP gateway connectivity and tier")

    # adk shell — download/launch AitherShell interactive terminal
    shell_p = sub.add_parser("shell", help="Launch AitherShell interactive terminal")
    shell_p.add_argument("--install", action="store_true", help="Download/update the AitherShell binary")
    shell_p.add_argument("--api-url", dest="api_url", help="Backend URL (Genesis or ADK server)")
    shell_p.add_argument("--genesis", help="Legacy alias for --api-url")
    shell_p.add_argument("shell_args", nargs=argparse.REMAINDER, help="Arguments to pass to AitherShell")

    # adk platform — internal platform toolkit commands (merged from aither-platform)
    platform_p = sub.add_parser("platform", help="Internal platform toolkit (merged from aither-platform)")
    platform_p.add_argument("platform_args", nargs=argparse.REMAINDER, help="Platform subcommand args")

    # adk listen — real-time audio intelligence (audiobook, meeting, voice notes)
    listen_p = sub.add_parser("listen", help="Real-time audio intelligence — audiobook, meeting, voice notes")
    listen_sub = listen_p.add_subparsers(dest="listen_command")

    listen_audiobook_p = listen_sub.add_parser("audiobook", help="Audiobook companion — track characters, stats, spells")
    listen_audiobook_p.add_argument("title", nargs="?", default="", help="Book title")
    listen_audiobook_p.add_argument("--author", default="", help="Author name")
    listen_audiobook_p.add_argument("--genre", default="litrpg", choices=["litrpg", "fantasy", "scifi", "general"])
    listen_audiobook_p.add_argument("--backend", default="wasapi", choices=["wasapi", "pulse", "sounddevice", "file"])
    listen_audiobook_p.add_argument("--file", dest="audio_file", help="Audio file path (for file backend)")
    listen_audiobook_p.add_argument("--workspace", help="Auto-save to workspace ID")

    listen_meeting_p = listen_sub.add_parser("meeting", help="Meeting transcription — action items, decisions, key points")
    listen_meeting_p.add_argument("title", nargs="?", default="", help="Meeting title")
    listen_meeting_p.add_argument("--type", dest="meeting_type", default="meeting",
                                  choices=["meeting", "lecture", "interview", "brainstorm"])
    listen_meeting_p.add_argument("--participants", "-p", nargs="*", default=[], help="Participant names")
    listen_meeting_p.add_argument("--backend", default="wasapi", choices=["wasapi", "pulse", "sounddevice"])
    listen_meeting_p.add_argument("--workspace", help="Auto-save to workspace ID")

    listen_note_p = listen_sub.add_parser("note", help="Voice note — quick dictation with key point extraction")
    listen_note_p.add_argument("title", nargs="?", default="Voice Note", help="Note title")
    listen_note_p.add_argument("--backend", default="wasapi", choices=["wasapi", "pulse", "sounddevice"])
    listen_note_p.add_argument("--workspace", help="Auto-save to workspace ID")

    _ = listen_sub.add_parser("sessions", help="List active listening sessions")
    listen_stop_p = listen_sub.add_parser("stop", help="Stop a listening session")
    listen_stop_p.add_argument("session_id", help="Session ID (partial match supported)")

    listen_export_p = listen_sub.add_parser("export", help="Export session as markdown notes or transcript")
    listen_export_p.add_argument("session_id", help="Session ID")
    listen_export_p.add_argument("--format", dest="fmt", default="notes", choices=["notes", "transcript"])
    listen_export_p.add_argument("--output", "-o", help="Write to file instead of stdout")

    # adk sync — bidirectional file sync (AitherDrive)
    sync_p = sub.add_parser("sync", help="Sync local directory with AitherOS platform")
    sync_sub = sync_p.add_subparsers(dest="sync_action")
    sync_init_p = sync_sub.add_parser("init", help="Initialize sync root")
    sync_init_p.add_argument("directory", nargs="?", default=".", help="Directory to sync")
    sync_sub.add_parser("status", help="Show sync status (changed/new/deleted)")
    sync_sub.add_parser("push", help="Upload local changes to platform")
    sync_sub.add_parser("pull", help="Download remote changes")
    sync_sub.add_parser("watch", help="Auto-sync on file changes (requires watchdog)")
    sync_sub.add_parser("stop", help="Stop background watcher")
    sync_ignore_p = sync_sub.add_parser("ignore", help="Add ignore pattern")
    sync_ignore_p.add_argument("pattern", help="Glob pattern to ignore")
    sync_sub.add_parser("config", help="Show sync configuration")

    # adk train — training pipeline management
    train_p = sub.add_parser("train", help="Manage model training (launch, monitor, cancel)")
    train_sub = train_p.add_subparsers(dest="train_command")

    _ = train_sub.add_parser("status", help="Check training readiness and active runs")

    train_launch_p = train_sub.add_parser("launch", help="Launch a training run")
    train_launch_p.add_argument("--preset", "-p", default="nemotron-orchestrator-8b",
                                help="Model preset (default: nemotron-orchestrator-8b)")
    train_launch_p.add_argument("--gpu", "-g", default="auto",
                                choices=["auto", "local", "dgx", "vast.ai", "customer"],
                                help="GPU target (default: auto)")
    train_launch_p.add_argument("--epochs", type=int, default=2, help="Training epochs (default: 2)")
    train_launch_p.add_argument("--lora-r", type=int, default=32, help="LoRA rank (default: 32)")
    train_launch_p.add_argument("--max-price", type=float, default=0.50,
                                help="Max GPU price $/hr for cloud (default: 0.50)")
    train_launch_p.add_argument("--dataset", help="HuggingFace dataset URL or local path")
    train_launch_p.add_argument("--no-benchmark", action="store_true",
                                help="Skip auto-benchmarking after training")
    train_launch_p.add_argument("--auto-deploy", action="store_true",
                                help="Auto-deploy if benchmark passes")

    train_logs_p = train_sub.add_parser("logs", help="Stream training logs for a run")
    train_logs_p.add_argument("run_id", help="Training run ID (partial match supported)")
    train_logs_p.add_argument("--lines", type=int, default=100, help="Number of log lines")

    train_cancel_p = train_sub.add_parser("cancel", help="Cancel an active training run")
    train_cancel_p.add_argument("run_id", help="Training run ID to cancel")

    train_runs_p = train_sub.add_parser("runs", help="List recent training runs")
    train_runs_p.add_argument("--status", help="Filter by status (e.g. training, completed, failed)")

    train_register_gpu_p = train_sub.add_parser("register-gpu",
                                                help="Register your local GPU for training")
    train_register_gpu_p.add_argument("--host", default="localhost", help="SSH host (default: localhost)")
    train_register_gpu_p.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    train_register_gpu_p.add_argument("--gpu-model", help="GPU model name (auto-detected if omitted)")
    train_register_gpu_p.add_argument("--vram", type=int, help="GPU VRAM in GB (auto-detected if omitted)")

    # adk jobs — manage background jobs and expeditions
    jobs_p = sub.add_parser(
        "jobs",
        help="Manage background jobs — LOCAL by default, --remote for the portal/cloud",
    )
    jobs_sub = jobs_p.add_subparsers(dest="jobs_command")
    # `--remote` on read/steer commands routes to genesis/portal expeditions;
    # without it they operate on the LOCAL job store (~/.aither/jobs.db).
    jobs_list_p = jobs_sub.add_parser("list", help="List jobs (local by default)")
    jobs_list_p.add_argument("--remote", action="store_true", help="List cloud expeditions instead")
    jobs_status_p = jobs_sub.add_parser("status", help="Show status of a job")
    jobs_status_p.add_argument("id", help="Job ID (local) or Expedition ID (--remote)")
    jobs_status_p.add_argument("--remote", action="store_true", help="Query the cloud expedition")
    jobs_steer_p = jobs_sub.add_parser("steer", help="Send a follow-up message to a job")
    jobs_steer_p.add_argument("id", help="Job/Expedition ID")
    jobs_steer_p.add_argument("message", help="Follow-up message")
    jobs_steer_p.add_argument("--remote", action="store_true", help="Steer the cloud expedition")
    jobs_hint_p = jobs_sub.add_parser("hint", help="Send an invisible hint to a job")
    jobs_hint_p.add_argument("id", help="Job/Expedition ID")
    jobs_hint_p.add_argument("message", help="Hint message")
    jobs_hint_p.add_argument("--remote", action="store_true", help="Hint the cloud expedition")
    jobs_watch_p = jobs_sub.add_parser("watch", help="Watch a cloud job's progress in real-time")
    jobs_watch_p.add_argument("id", help="Expedition ID")
    # ── LOCAL job engine (runs on THIS machine; no server required) ──
    jobs_run_p = jobs_sub.add_parser("run", help="Run a job locally in the FOREGROUND")
    jobs_run_p.add_argument("query", nargs="+", help="The task to run")
    jobs_run_p.add_argument("--agent", default="aither", help="Local agent persona")
    jobs_start_p = jobs_sub.add_parser("start", help="Start a local job in the BACKGROUND (detached)")
    jobs_start_p.add_argument("query", nargs="+", help="The task to run")
    jobs_start_p.add_argument("--agent", default="aither", help="Local agent persona")
    jobs_cancel_p = jobs_sub.add_parser("cancel", help="Cancel a running local job")
    jobs_cancel_p.add_argument("id", help="Local job ID")
    jobs_sync_p = jobs_sub.add_parser("sync", help="Sync a local job to/from the portal (opt-in)")
    jobs_sync_p.add_argument("direction", choices=["push", "pull"], help="push=mirror up, pull=refresh")
    jobs_sync_p.add_argument("id", help="Local job ID")
    # Hidden executor used by `start` to run the detached job body.
    jobs_exec_p = jobs_sub.add_parser("_exec", help=argparse.SUPPRESS)
    jobs_exec_p.add_argument("id", help="Local job ID to execute")


# ── MCP subcommand handlers ───────────────────────────────────────────────


def _cmd_mcp_setup(args) -> int:
    """Generate IDE MCP config with OAuth-first auth."""
    from adk.mcp_setup import resolve_auth, resolve_gateway_url, generate_config, write_config, probe_gateway

    mode = getattr(args, "mode", "local")
    ide = getattr(args, "ide", "claude-code")
    project_dir = getattr(args, "project_dir", ".")
    bake_token = getattr(args, "bake_token", False)

    url = resolve_gateway_url(mode)
    token, source = resolve_auth()

    if bake_token:
        if token:
            print(f"  Auth: baking token from {source}")
        else:
            print("  Auth: no token found for --bake-token")
            print("  Run 'adk login' first, then re-run this command.")
            return 1
        config = generate_config(ide, url, token=token)
    else:
        print("  Auth: OAuth (IDE will handle via /authenticate)")
        config = generate_config(ide, url, token=None)

    out_path = write_config(config, ide, project_dir)
    print(f"  Config: {out_path}")
    print(f"  Gateway: {url}")

    if mode == "local":
        status = probe_gateway(url, token)
        if status["connected"]:
            print(f"  Status: connected ({status['status']})")
        else:
            print("  Status: not reachable")
            print("  Tip: Run 'adk mcp node' for a lightweight local server.")

        from adk.mcp_setup import ensure_local_ca_trust
        ca_result = ensure_local_ca_trust()
        if ca_result == "set":
            print("  TLS: AitherNet CA trusted")
        elif ca_result == "already":
            print("  TLS: AitherNet CA already trusted")
        elif ca_result:
            print(f"  TLS: {ca_result}")

    print()
    if not bake_token:
        print("  Restart your IDE, then use /authenticate to connect.")
    else:
        print("  Restart your IDE to apply.")
    return 0


def _cmd_mcp_node(args) -> int:
    """Start lightweight local MCP server."""
    from adk.node.server import run_node
    mode = getattr(args, "mode", "proxy")
    port = getattr(args, "port", 8182)
    run_node(mode=mode, port=port)
    return 0


def _cmd_mcp_status(args) -> int:
    """Check MCP gateway connectivity and tier."""
    from adk.mcp_setup import resolve_auth, probe_gateway, _GATEWAY_URLS

    token, source = resolve_auth()
    print(f"  Auth source: {source}")

    for mode_name, url in _GATEWAY_URLS.items():
        result = probe_gateway(url, token)
        icon = "[OK]" if result["connected"] else "[--]"
        print(f"  {icon} {mode_name:8s} {url}")
        if result["connected"]:
            if result.get("tier"):
                print(f"           tier={result['tier']}  tools={result['tool_count']}  "
                      f"balance={result['balance']}")
            if result.get("user"):
                print(f"           user={result['user']}")
        elif result.get("error"):
            print(f"           {result['error']}")
    return 0


def main():
    global _cached_parser
    # Windows consoles default to a legacy code page (cp1252) that cannot encode
    # the Unicode glyphs (arrows, box-drawing, emoji) the CLI prints — which
    # otherwise crashes commands like `adk login` with UnicodeEncodeError AFTER
    # they have already done their real work. Make all stdout/stderr writes
    # encoding-safe up front (errors="replace" degrades an unencodable glyph to
    # "?" instead of aborting). No-op where the stream can't be reconfigured.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        prog="adk",
        description="AitherADK — Build AI agent fleets with any LLM backend",
    )
    sub = parser.add_subparsers(dest="command")
    _register_commands(sub)
    _cached_parser = parser

    args = parser.parse_args()

    # Non-blocking update check (once per day)
    _check_for_updates()

    if args.command == "start":
        sys.exit(cmd_start(args))
    elif args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "new":
        sys.exit(cmd_new(args))
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "up":
        sys.exit(cmd_up(args))
    elif args.command == "down":
        sys.exit(cmd_down(args))
    elif args.command == "stack":
        sys.exit(cmd_stack(args))
    elif args.command == "wizard":
        from adk.setup_cli import cmd_wizard
        sys.exit(cmd_wizard(args))
    elif args.command == "register":
        sys.exit(cmd_register(args))
    elif args.command == "login":
        sys.exit(cmd_login(args))
    elif args.command == "whoami":
        sys.exit(cmd_whoami(args))
    elif args.command == "logout":
        sys.exit(cmd_logout(args))
    elif args.command == "agent-prompt":
        from adk.agent_prompt import cmd_agent_prompt
        sys.exit(cmd_agent_prompt(args))
    elif args.command == "connect":
        sys.exit(cmd_connect(args))
    elif args.command == "setup":
        from adk.setup_cli import cmd_setup
        sys.exit(cmd_setup(args))
    elif args.command == "aeon":
        sys.exit(cmd_aeon(args))
    elif args.command == "create-app":
        sys.exit(cmd_create_app(args))
    elif args.command == "deploy":
        component = getattr(args, "component", None)
        if component == "agent":
            # --tenant or --from triggers download+run mode (deploy TO this machine)
            if getattr(args, "tenant", None) or getattr(args, "from_url", None):
                from adk.deploy import cmd_deploy_tenant_agent
                sys.exit(cmd_deploy_tenant_agent(args))
            else:
                sys.exit(cmd_deploy(args))
        else:
            from adk.deploy import cmd_deploy_component
            sys.exit(cmd_deploy_component(args))
    elif args.command == "workspace":
        sys.exit(cmd_workspace(args))
    elif args.command == "onboard":
        sys.exit(cmd_onboard(args))
    elif args.command == "enroll":
        sys.exit(cmd_enroll(args))
    elif args.command == "host":
        sys.exit(cmd_host(args))
    elif args.command == "integrate":
        sys.exit(cmd_integrate(args))
    elif args.command == "publish":
        sys.exit(cmd_publish(args))
    elif args.command == "index":
        sys.exit(cmd_index(args))
    elif args.command == "test":
        sys.exit(cmd_test(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "admin":
        sys.exit(cmd_admin(args))
    elif args.command == "backend":
        sys.exit(cmd_backend(args))
    elif args.command == "keys":
        sys.exit(cmd_keys(args))
    elif args.command == "routing":
        sys.exit(cmd_routing(args))
    elif args.command == "costs":
        sys.exit(cmd_costs(args))
    elif args.command == "tools":
        sys.exit(cmd_tools(args))
    elif args.command == "quickstart":
        sys.exit(cmd_quickstart(args))
    elif args.command == "quickstart-local":
        sys.exit(cmd_quickstart_local(args))
    elif args.command == "backup":
        sys.exit(cmd_backup(args))
    elif args.command == "ingest":
        sys.exit(cmd_ingest(args))
    elif args.command == "disconnect":
        sys.exit(cmd_disconnect(args))
    elif args.command == "jobs":
        sys.exit(cmd_jobs(args))
    elif args.command == "doctor":
        from adk.doctor import cmd_doctor
        sys.exit(cmd_doctor(args))
    elif args.command == "gateway":
        from adk.gateway_process import cmd_gateway
        sys.exit(cmd_gateway(args))
    elif args.command == "cron":
        sys.exit(_cmd_cron(args))
    elif args.command == "addon":
        sys.exit(_cmd_addon(args))
    elif args.command == "install":
        sys.exit(cmd_install(args))
    elif args.command == "packs":
        sys.exit(cmd_packs(args))
    elif args.command == "pack":
        sys.exit(_cmd_pack(args))
    elif args.command == "skills":
        sys.exit(_cmd_skills(args))
    elif args.command == "soul":
        sys.exit(_cmd_soul(args))
    elif args.command == "doc":
        from adk.doc import cmd_doc
        sys.exit(cmd_doc(args))
    elif args.command == "mcp":
        mcp_cmd = getattr(args, "mcp_command", None)
        if mcp_cmd == "serve":
            from adk.mcp_stdio import cmd_mcp_serve
            sys.exit(cmd_mcp_serve(args))
        elif mcp_cmd == "config":
            from adk.mcp_stdio import cmd_mcp_config
            sys.exit(cmd_mcp_config(args))
        elif mcp_cmd == "setup":
            sys.exit(_cmd_mcp_setup(args))
        elif mcp_cmd == "node":
            sys.exit(_cmd_mcp_node(args))
        elif mcp_cmd == "status":
            sys.exit(_cmd_mcp_status(args))
        else:
            print("Usage: adk mcp [serve|config|setup|node|status]")
            print()
            print("  serve    Start stdio MCP server (pipe into Claude Code)")
            print("  config   Print MCP client configuration JSON")
            print("  setup    Generate IDE config for MCP gateway (OAuth-first)")
            print("  node     Start lightweight local MCP server")
            print("  status   Check MCP gateway connectivity and tier")
            sys.exit(1)
    elif args.command == "shell":
        from adk.shell_launcher import cmd_shell
        sys.exit(cmd_shell(args))
    elif args.command == "platform":
        # Delegate to the internal platform CLI (merged from aither_adk.cli)
        try:
            from adk.platform.cli import main as platform_main
            # Replace sys.argv so the platform CLI parses its own args
            sys.argv = ["adk-platform"] + (args.platform_args or [])
            platform_main()
        except ImportError:
            print("Platform toolkit not available. Install with: pip install aither-adk[platform]")
            sys.exit(1)
    elif args.command == "listen":
        sys.exit(_cmd_listen(args))
    elif args.command == "sync":
        sys.exit(cmd_sync(args))
    elif args.command == "grid":
        sys.exit(cmd_grid(args))
    elif args.command == "explore":
        sys.exit(cmd_explore(args))
    elif args.command == "upgrade":
        sys.exit(cmd_upgrade(args))
    elif args.command == "support":
        print("\n  AitherADK Support")
        print("  " + "=" * 40)
        print("  Docs:      https://github.com/Aitherium/aither-adk")
        print("  Discord:   https://discord.gg/aitherium")
        print("  Issues:    https://github.com/Aitherium/aither-adk/issues")
        print("  Portal:    https://portal.aitherium.com")
        print("  Email:     support@aitherium.com")
        print()
        sys.exit(0)
    elif args.command == "train":
        sys.exit(_cmd_train(args))
    elif args.command is None:
        # No command — default to start
        args.path = "."
        sys.exit(cmd_start(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
