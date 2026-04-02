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

    # OpenClaw detection — prompt integration if detected
    openclaw_dir = Path.home() / ".openclaw"
    if openclaw_dir.exists():
        oc_config = {}
        oc_config_path = openclaw_dir / "openclaw.json"
        if oc_config_path.exists():
            try:
                import json
                oc_config = json.loads(oc_config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        aither_integrated = any(
            "aither" in k.lower()
            for k in oc_config.get("mcpServers", {})
        )
        if not aither_integrated:
            print()
            print(f"  OpenClaw detected! Connect it to AitherOS agents:")
            print(f"  aither integrate openclaw")

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


def cmd_onboard(args):
    """Interactive onboarding — detect products, configure, integrate."""
    import asyncio
    import json as _json

    async def _onboard():
        # Inline ProductDetector (no AitherOS lib dependency)
        from pathlib import Path
        import shutil

        home = Path.home()
        aither_dir = home / ".aither"
        openclaw_dir = home / ".openclaw"

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
                parts = result.stdout.strip().split(",")
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
            print(f"     -> aither register")
            step_num += 1

        if not ollama_bin and not gpu_name:
            print(f"  {step_num}. Set up inference backend")
            print(f"     -> Install Ollama: https://ollama.com")
            print(f"     -> Or use cloud: aither register")
            step_num += 1

        if openclaw_detected and not aither_integrated:
            print(f"  {step_num}. Connect OpenClaw to AitherOS agent fleet")
            print(f"     -> aither integrate openclaw")
            step_num += 1

        print(f"  {step_num}. Create your first agent")
        print(f"     -> aither init my-agent && cd my-agent && aither run")
        step_num += 1

        if api_key:
            print(f"  {step_num}. Publish to Elysium marketplace (optional)")
            print(f"     -> aither publish")
            step_num += 1

        # ── 3. Quick actions ──────────────────────────────────
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
        print(f"  Run 'aither integrate list' to see available integrations")
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
        print(f"  Status: pending_review")
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


def cmd_status(args):
    """Show backend and service status."""
    import asyncio

    async def _status():
        import httpx
        checks = {
            "Genesis": os.environ.get("AITHER_URL", "http://localhost:8001"),
            "vLLM": os.environ.get("AITHER_VLLM_URL", os.environ.get("VLLM_URL", "http://localhost:8120")),
            "Ollama": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            "AitherNode": "http://localhost:8090",
            "Gateway": os.environ.get("AITHER_GATEWAY_URL", "https://gateway.aitherium.com"),
        }
        print("AitherADK Backend Status")
        print("=" * 50)
        for name, url in checks.items():
            try:
                async with httpx.AsyncClient(timeout=3.0) as c:
                    hp = "/api/tags" if name == "Ollama" else "/health"
                    r = await c.get(f"{url.rstrip('/')}{hp}")
                    status = "UP" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception:
                status = "DOWN"
            icon = "+" if status == "UP" else "-"
            print(f"  [{icon}] {name:12s} {url:45s} {status}")

        # API key check
        api_key = os.environ.get("AITHER_API_KEY", "")
        if api_key:
            print(f"\n  API Key: {api_key[:16]}...{api_key[-4:]}")
        else:
            saved = {}
            try:
                from adk.config import load_saved_config
                saved = load_saved_config()
            except Exception:
                pass
            if saved.get("api_key"):
                print(f"\n  API Key (saved): {saved['api_key'][:16]}...")
            else:
                print("\n  No API key. Run: adk connect --api-key <key>")

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
    print(f"  AitherADK")
    print(f"  =========")
    print()

    # ── Step 1: Detect project ──────────────────────────────────────
    _SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv",
             ".tox", "dist", "build", ".mypy_cache", "site-packages"}

    def _count_files(root, ext):
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP]
            count += sum(1 for f in filenames if f.endswith(ext))
            if count > 5000:
                break  # Good enough
        return count

    py_count = _count_files(target, ".py")
    ts_count = _count_files(target, ".ts")
    js_count = _count_files(target, ".js")

    # Count all file types for a richer picture
    all_count = sum(1 for _ in os.scandir(target) if _.is_file())
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
        print(f"  Code index: Skipped (no Python files -- code search works with Python)")
    else:
        print(f"  Code index: Skipped (empty directory)")

    # ── Step 4: Set up memory ───────────────────────────────────────
    # Suppress noisy warnings for casual use
    _logging = __import__("logging")
    _logging.getLogger("adk.faculties.base").setLevel(_logging.ERROR)
    _logging.getLogger("adk.faculties.memory_graph").setLevel(_logging.ERROR)
    _logging.getLogger("adk.identity").setLevel(_logging.ERROR)

    memory_dir = os.path.join(os.path.expanduser("~/.aither"), "memory", project_name)
    from adk.faculties.memory_graph import MemoryGraph
    memory_graph = MemoryGraph(data_dir=memory_dir)
    mem_stats = memory_graph.get_stats()
    if mem_stats["nodes"] > 0:
        print(f"  Memory:     {mem_stats['nodes']} memories restored from previous sessions")
    else:
        print(f"  Memory:     New (will persist to {memory_dir})")

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
    agent.set_memory_graph(memory_graph)

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
                ms = memory_graph.get_stats()
                print(f"  Memory:     {ms['nodes']} memories, {ms['edges']} connections")
                print(f"  Workspace:  {target}")
                continue

            if user_input.lower() == "/memory":
                ms = memory_graph.get_stats()
                print(f"  Nodes: {ms['nodes']}, Edges: {ms['edges']}, "
                      f"Embeddings: {ms['embeddings_cached']}")
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
        for port in (8200, 8120, 8116):
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
            print(f"  Most complex:")
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
    print(f"    cg = CodeGraph()")
    print(f"    await cg.index_codebase(\"{target}\")")
    print("    agent.set_code_graph(cg)")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="adk",
        description="AitherADK — Build AI agent fleets with any LLM backend",
    )
    sub = parser.add_subparsers(dest="command")

    # adk start — the main entry point for everyone
    start_p = sub.add_parser("start", help="Start chatting with your codebase (zero config)")
    start_p.add_argument("path", nargs="?", default=".", help="Project directory (default: current)")

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
    deploy_p.add_argument("--target", choices=["gateway", "docker", "kubernetes", "systemd", "cloud-gpu"],
                           default="gateway", help="Deploy target (default: gateway)")
    deploy_p.add_argument("--strategy", choices=["rolling", "blue-green", "canary", "recreate"],
                           default="rolling", help="Deployment strategy (for container targets)")

    # aither onboard — interactive onboarding wizard
    onboard_p = sub.add_parser("onboard", help="Interactive onboarding — detect, configure, integrate")
    onboard_p.add_argument("--api-key", help="AITHER_API_KEY")

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

    args = parser.parse_args()

    if args.command == "start":
        sys.exit(cmd_start(args))
    elif args.command == "init":
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
    elif args.command == "onboard":
        sys.exit(cmd_onboard(args))
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
    elif args.command is None:
        # No command — default to start
        args.path = "."
        sys.exit(cmd_start(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
