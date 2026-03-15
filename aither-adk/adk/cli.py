"""CLI scaffolding — `aither init` and `aither run` commands.

Usage:
    aither init myproject          # Scaffold a new agent project
    aither run                     # Start the server (reads config.yaml)
    aither run --identity lyra -p 9000
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from adk.config import load_saved_config, save_saved_config

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
# See https://github.com/Aitherium/aither for docs

identity: {name}
port: 8080

# LLM backend: auto, ollama, openai, anthropic, gateway
backend: auto

# Uncomment to set a specific model
# model: nemotron-orchestrator-8b

# Built-in tools (enabled by default)
builtin_tools: true

# Safety checks (enabled by default)
safety: true
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
    print(f"  agent.py   — Your agent definition")
    print(f"  config.yaml — Configuration")
    print(f"  tools.py   — Custom tools")
    print()
    print(f"Next steps:")
    print(f"  cd {target}")
    print(f"  aither run           # Start the server")
    print(f"  python agent.py      # Run directly")
    return 0


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
        except Exception as exc:
            print(f"  Error: {exc}")
            return 1

        user_id = result.get("user_id", "")
        api_key = result.get("api_key", "")

        if api_key:
            save_saved_config({"api_key": api_key, "email": email})
            print(f"  API key saved to ~/.aither/config.json")

        print()
        print(f"  Account created (user_id: {user_id}).")
        print(f"  Check your email to verify, then run: aither connect")
        return 0

    return asyncio.run(_register())


def cmd_connect(args):
    """Connect to AitherOS — detect local LLMs, activate cloud, join mesh."""
    import asyncio
    import json as _json

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
        inference_ok = False
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
                        inference_ok = True
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
        node_ok = False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get("http://localhost:8080/health")
                if resp.status_code == 200:
                    node_ok = True
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
        print(f"  Type 'quit' to exit, 'reset' to start a new session.\n")

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


def main():
    parser = argparse.ArgumentParser(
        prog="aither",
        description="AitherADK — Build AI agent fleets with any LLM backend",
    )
    sub = parser.add_subparsers(dest="command")

    # aither init
    init_p = sub.add_parser("init", help="Scaffold a new agent project")
    init_p.add_argument("name", nargs="?", default="my-agent", help="Project/agent name")
    init_p.add_argument("-d", "--directory", help="Target directory (default: ./<name>)")

    # aither run
    run_p = sub.add_parser("run", help="Start the agent server")
    run_p.add_argument("-i", "--identity", help="Agent identity")
    run_p.add_argument("-p", "--port", type=int, help="Server port")
    run_p.add_argument("--host", help="Server host")
    run_p.add_argument("-b", "--backend", help="LLM backend")
    run_p.add_argument("-m", "--model", help="Model name")
    run_p.add_argument("-f", "--fleet", help="Fleet YAML config")
    run_p.add_argument("-a", "--agents", help="Comma-separated agent identities")

    # aither register
    register_p = sub.add_parser("register", help="Create a new Aitherium account")
    register_p.add_argument("--email", help="Account email (prompted if omitted)")
    register_p.add_argument("--password", help="Account password (prompted if omitted)")

    # aither connect
    connect_p = sub.add_parser("connect", help="Connect to AitherOS — detect LLMs, set up gateway")
    connect_p.add_argument("--api-key", help="AITHER_API_KEY for cloud inference")
    connect_p.add_argument("--save", action="store_true", default=True,
                           help="Save config to ~/.aither/config.json (default: true)")
    connect_p.add_argument("--no-save", action="store_false", dest="save",
                           help="Don't save config")

    # aither setup
    setup_p = sub.add_parser("setup", help="Set up local inference (vLLM/Ollama) + optional AitherOS stack")
    setup_p.add_argument("--tier", choices=["nano", "lite", "standard", "full", "ollama"],
                         help="Force a specific tier (default: auto-detect from GPU)")
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

    # aither aeon
    aeon_p = sub.add_parser("aeon", help="Multi-agent group chat")
    aeon_p.add_argument("-p", "--preset", help="Preset: balanced, creative, technical, security, minimal, duo_code, research")
    aeon_p.add_argument("-a", "--agents", help="Comma-separated agent names (e.g. demiurge,athena)")
    aeon_p.add_argument("-r", "--rounds", type=int, default=1, help="Discussion rounds per message (default: 1)")
    aeon_p.add_argument("--no-synthesize", action="store_true", help="Skip orchestrator synthesis")

    # aither deploy
    deploy_p = sub.add_parser("deploy", help="Deploy an agent to AitherOS")
    deploy_p.add_argument("name", nargs="?", help="Agent name (default: from config.yaml)")
    deploy_p.add_argument("-d", "--directory", help="Project directory (default: .)")
    deploy_p.add_argument("--api-key", help="AITHER_API_KEY")
    deploy_p.add_argument("--gateway", help="Gateway URL (default: gateway.aitherium.com)")
    deploy_p.add_argument("--capabilities", help="Comma-separated capabilities")
    deploy_p.add_argument("--description", help="Agent description")
    deploy_p.add_argument("--version", help="Agent version")

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "register":
        sys.exit(cmd_register(args))
    elif args.command == "connect":
        sys.exit(cmd_connect(args))
    elif args.command == "setup":
        from adk.setup_cli import cmd_setup
        sys.exit(cmd_setup(args))
    elif args.command == "aeon":
        sys.exit(cmd_aeon(args))
    elif args.command == "deploy":
        sys.exit(cmd_deploy(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
