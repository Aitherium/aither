"""Sovereign agent scaffolding — `adk create` command.

Reads Jinja2 templates from ``adk/templates/<template>/`` and renders a
complete deployable agent project with backend, frontend, Docker, and
portal registration hooks.

Also provides ``interactive_create()`` — a Rich-powered interactive wizard
with three modes: Workspace App, Standalone Agent, Identity Only.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("adk.create")

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Available template types and their descriptions
TEMPLATES = {
    "sovereign": "Full agent app with UI, Docker, portal hooks (GargBot-equivalent)",
    "knowledge": "RAG + chat only — no frontend, no extraction",
    "assistant": "Chat-only, no documents or RAG",
    "minimal": "API-only, no frontend (same as `adk init`)",
    "platform": "Multi-service agent platform (AitherOS-equivalent)",
}

# Platform tier definitions
PLATFORM_TIERS = {
    "starter": "5-8 services, ~6 containers — structured agent app with proper patterns",
    "standard": "12-18 services, ~15 containers — multi-agent platform with memory, cognition, LLM queue",
    "full": "25-40 services, ~30 containers — complete platform with security, mesh, training",
}

DEFAULT_PLATFORM_PORT_BASE = 9001
DEFAULT_PLATFORM_AGENTS = ["orchestrator", "assistant"]

# Brain pack presets for workspace apps
BRAIN_PACK_PRESETS: Dict[str, Dict[str, Any]] = {
    "knowledge": {
        "description": "RAG + document ingestion (like GargBot)",
        "system_prompt": (
            "You are {name}, a knowledge assistant. You have access to internal "
            "documents uploaded by your team. Answer questions based on those documents "
            "and tell the user clearly when you don't have enough information."
        ),
        "features": {"documents": True, "chat": True, "social": False, "billing": False, "email": False},
        "doc_types": ["pdf", "txt", "md", "docx", "csv"],
    },
    "business": {
        "description": "CRM + social + billing (like Chelle)",
        "system_prompt": (
            "You are {name}, a business assistant. You help manage client relationships, "
            "handle communications, and track billing. You maintain a professional, "
            "helpful tone."
        ),
        "features": {"documents": True, "chat": True, "social": True, "billing": True, "email": True},
        "doc_types": ["pdf", "txt", "md"],
    },
    "generic": {
        "description": "Clean slate — configure everything yourself",
        "system_prompt": "You are {name}, an AI assistant. Be helpful, accurate, and concise.",
        "features": {"documents": False, "chat": True, "social": False, "billing": False, "email": False},
        "doc_types": [],
    },
}

# LLM provider presets
LLM_PROVIDERS = {
    "portal": {"label": "Portal (hosted, no setup)", "env_key": "AITHERIUM_API_KEY"},
    "ollama": {"label": "Ollama (local)", "env_key": "OLLAMA_HOST"},
    "openai": {"label": "OpenAI", "env_key": "OPENAI_API_KEY"},
    "anthropic": {"label": "Anthropic", "env_key": "ANTHROPIC_API_KEY"},
    "deepseek": {"label": "DeepSeek", "env_key": "DEEPSEEK_API_KEY"},
}

DEFAULT_PORT = 8900


def _slugify(name: str) -> str:
    """Convert a name to a safe slug for directories, env vars, etc."""
    return re.sub(r"[^a-z0-9_-]", "-", name.lower().strip()).strip("-")


def _env_prefix(name: str) -> str:
    """Convert a name to an env var prefix: MY_AGENT_"""
    slug = re.sub(r"[^a-z0-9]", "_", name.lower().strip()).strip("_").upper()
    return slug + "_"


def _title_case(name: str) -> str:
    """Convert a slug to title case: my-agent -> My Agent"""
    return " ".join(w.capitalize() for w in re.split(r"[-_ ]+", name))


def _load_brain_pack(pack_path: str) -> Dict[str, Any]:
    """Load a brain pack YAML file for use as template context."""
    import yaml

    path = Path(pack_path).expanduser()
    if not path.exists():
        log.warning("Brain pack not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _render_template(template_text: str, context: Dict[str, Any]) -> str:
    """Render a Jinja2 template string with the given context."""
    try:
        from jinja2 import Environment

        env = Environment(keep_trailing_newline=True)
        tmpl = env.from_string(template_text)
        return tmpl.render(**context)
    except ImportError:
        # Fallback: simple string substitution for basic cases
        result = template_text
        for key, value in context.items():
            if isinstance(value, str):
                result = result.replace("{{ " + key + " }}", value)
                result = result.replace("{{" + key + "}}", value)
        return result


def _generate_brain_pack_yaml(
    name: str,
    preset: str,
    display_name: str,
    description: str,
    features: Dict[str, bool],
    provider: str,
) -> str:
    """Generate brain_pack.yaml content from a preset and overrides."""
    import yaml

    preset_data = BRAIN_PACK_PRESETS.get(preset, BRAIN_PACK_PRESETS["generic"])
    slug = _slugify(name)

    pack = {
        "id": slug,
        "app_name": display_name,
        "company_name": "",
        "system_prompt": preset_data["system_prompt"].format(name=display_name),
        "welcome_message": f"Hi — I'm {display_name}. How can I help?",
        "doc_types": preset_data["doc_types"],
        "features": {**preset_data["features"], **features},
        "llm_provider": provider,
        "collection_name": f"{slug}_docs",
    }
    if description:
        pack["system_prompt"] = description

    return yaml.dump(pack, default_flow_style=False, sort_keys=False)


def build_context(
    name: str,
    template: str = "sovereign",
    pack_path: Optional[str] = None,
    port: int = DEFAULT_PORT,
    description: str = "",
) -> Dict[str, Any]:
    """Build the Jinja2 template context from user inputs."""
    slug = _slugify(name)
    env_pfx = _env_prefix(name)
    display_name = _title_case(name)

    # Load brain pack overrides if provided
    pack_data: Dict[str, Any] = {}
    if pack_path:
        pack_data = _load_brain_pack(pack_path)

    return {
        "agent_name": slug,
        "agent_display_name": pack_data.get("app_name", display_name),
        "agent_description": description or pack_data.get(
            "system_prompt", f"AI assistant: {display_name}"
        )[:200],
        "company_name": pack_data.get("company_name", ""),
        "env_prefix": env_pfx,
        "port": port,
        "template_type": template,
        "pack_id": pack_data.get("id", slug),
        "pack_path": pack_path or "",
        "welcome_message": pack_data.get(
            "welcome_message", f"Hi — I'm {display_name}. Upload documents and ask me anything."
        ),
        "system_prompt": pack_data.get("system_prompt", (
            f"You are {display_name}, an AI assistant. You have access to internal\n"
            f"documents uploaded by your team. Answer questions based on those documents\n"
            f"and tell the user clearly when you don't have enough information."
        )),
        "collection_name": f"{slug}_docs",
        "db_name": f"{slug}.db",
        "session_cookie": f"{slug}_session",
        "log_name": slug,
    }


def scaffold_project(
    name: str,
    template: str = "sovereign",
    pack_path: Optional[str] = None,
    port: int = DEFAULT_PORT,
    description: str = "",
    output_dir: Optional[str] = None,
) -> Path:
    """Scaffold a complete agent project from a template.

    Returns the path to the created project directory.
    """
    ctx = build_context(name, template, pack_path, port, description)
    slug = ctx["agent_name"]
    target = Path(output_dir or slug)

    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} already exists and is not empty")

    template_dir = _TEMPLATES_DIR / template
    if not template_dir.exists():
        raise ValueError(
            f"Template '{template}' not found. Available: {', '.join(TEMPLATES)}"
        )

    # Walk template directory and render all .j2 files
    for src_path in sorted(template_dir.rglob("*")):
        if src_path.is_dir():
            continue
        if "__pycache__" in str(src_path):
            continue

        # Compute relative path and strip .j2 extension
        rel = src_path.relative_to(template_dir)
        dest_name = str(rel)
        if dest_name.endswith(".j2"):
            dest_name = dest_name[:-3]

        dest_path = target / dest_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix == ".j2":
            # Render Jinja2 template
            template_text = src_path.read_text(encoding="utf-8")
            rendered = _render_template(template_text, ctx)
            dest_path.write_text(rendered, encoding="utf-8")
        else:
            # Copy verbatim (binary files, static assets)
            shutil.copy2(src_path, dest_path)

    # If a brain pack was provided, copy it into the project
    if pack_path:
        pack_src = Path(pack_path).expanduser()
        if pack_src.exists():
            packs_dir = target / "backend" / "app" / "packs"
            packs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pack_src, packs_dir / pack_src.name)

    # Register in local agent registry
    try:
        from adk.agent_registry import register_local_agent
        register_local_agent(
            name=slug,
            port=ctx["port"],
            capabilities=["chat"],
            metadata={"template": template, "project_dir": str(target.resolve())},
        )
    except Exception as e:
        log.debug("Local registry write skipped: %s", e)

    log.info("Scaffolded '%s' at %s (template=%s)", name, target, template)
    return target


def scaffold_workspace_app(
    name: str,
    display_name: str = "",
    description: str = "",
    preset: str = "generic",
    provider: str = "portal",
    features: Optional[Dict[str, bool]] = None,
    port: int = DEFAULT_PORT,
    output_dir: Optional[str] = None,
) -> Path:
    """Scaffold a full workspace app (backend + frontend + brain pack + Docker).

    Uses the 'sovereign' template as base, then generates a brain_pack.yaml
    from the selected preset and feature toggles.

    Returns the path to the created project directory.
    """
    slug = _slugify(name)
    display = display_name or _title_case(name)
    feat = features or BRAIN_PACK_PRESETS.get(preset, {}).get("features", {})

    # Generate brain pack YAML
    pack_yaml = _generate_brain_pack_yaml(
        name=name,
        preset=preset,
        display_name=display,
        description=description,
        features=feat,
        provider=provider,
    )

    # Write temporary brain pack file
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(pack_yaml)
        tmp_pack = f.name

    try:
        target = scaffold_project(
            name=name,
            template="sovereign",
            pack_path=tmp_pack,
            port=port,
            description=description,
            output_dir=output_dir,
        )
    finally:
        os.unlink(tmp_pack)

    # Also write the generated brain pack directly into the packs dir
    packs_dir = target / "backend" / "app" / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    (packs_dir / f"{slug}.yaml").write_text(pack_yaml, encoding="utf-8")

    # Generate .env with provider config
    provider_info = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["portal"])
    env_lines = [
        f"# {display} configuration",
        f"APP_NAME={display}",
        f"APP_ID={slug}",
        f"PORT={port}",
        f"LLM_PROVIDER={provider}",
        f"# {provider_info['env_key']}=your-key-here",
        f"BRAIN_PACK_PATH=app/packs/{slug}.yaml",
    ]
    env_path = target / ".env"
    if not env_path.exists():
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    log.info("Scaffolded workspace app '%s' at %s (preset=%s, provider=%s)",
             name, target, preset, provider)
    return target


def interactive_create() -> Dict[str, Any]:
    """Rich interactive wizard for `adk create`. Returns kwargs for scaffold_project
    or scaffold_workspace_app depending on mode selection."""

    # Try Rich for pretty prompts, fall back to plain input
    try:
        from rich.console import Console
        from rich.prompt import Prompt, IntPrompt, Confirm
        from rich.panel import Panel
        from rich.table import Table
        _HAS_RICH = True
        console = Console()
    except ImportError:
        _HAS_RICH = False

    if not _HAS_RICH:
        return _interactive_create_plain()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]AitherADK[/] — Create a new agent",
        border_style="cyan",
    ))
    console.print()

    # ── Step 1: Mode selection ──
    mode_table = Table(show_header=False, box=None, padding=(0, 2))
    mode_table.add_column(style="bold yellow", width=3)
    mode_table.add_column(style="bold white")
    mode_table.add_column(style="dim")
    mode_table.add_row("1", "Workspace App", "Full product with UI, RAG, Docker")
    mode_table.add_row("2", "Standalone Agent", "agent.yaml + tools — lightweight")
    mode_table.add_row("3", "Identity Only", "Deploy identity to running AitherOS")
    console.print(mode_table)
    console.print()

    mode_choice = Prompt.ask(
        "  What kind of agent?",
        choices=["1", "2", "3"],
        default="1",
    )

    mode_map = {"1": "workspace_app", "2": "standalone", "3": "identity"}
    mode = mode_map[mode_choice]

    # ── Step 2: Name & description ──
    console.print()
    name = Prompt.ask("  Agent name (slug)", default="my-agent")
    display_name = Prompt.ask("  Display name", default=_title_case(name))
    description = Prompt.ask("  Description", default="")

    if mode == "identity":
        # Identity only — generate YAML and register
        return {
            "_mode": "identity",
            "name": name,
            "display_name": display_name,
            "description": description,
        }

    if mode == "standalone":
        # Lightweight standalone agent
        console.print()
        console.print("  [dim]Available templates:[/]")
        for key, desc in TEMPLATES.items():
            marker = " [yellow](default)[/]" if key == "sovereign" else ""
            console.print(f"    [cyan]{key:12s}[/] — {desc}{marker}")
        template = Prompt.ask(
            "\n  Template",
            choices=list(TEMPLATES.keys()),
            default="sovereign",
        )
        port = IntPrompt.ask("  Port", default=DEFAULT_PORT)
        pack_path = Prompt.ask("  Brain pack YAML path (optional)", default="")
        return {
            "name": name,
            "template": template,
            "pack_path": pack_path or None,
            "port": port,
            "description": description,
        }

    # ── Workspace App mode ──
    console.print()
    console.print("  [bold]Brain Pack Template:[/]")
    preset_table = Table(show_header=False, box=None, padding=(0, 2))
    preset_table.add_column(style="bold yellow", width=3)
    preset_table.add_column(style="bold white")
    preset_table.add_column(style="dim")
    for i, (key, preset) in enumerate(BRAIN_PACK_PRESETS.items(), 1):
        preset_table.add_row(str(i), key.capitalize(), preset["description"])
    console.print(preset_table)
    console.print()

    preset_choice = Prompt.ask(
        "  Brain pack preset",
        choices=["1", "2", "3"],
        default="1",
    )
    preset_keys = list(BRAIN_PACK_PRESETS.keys())
    preset = preset_keys[int(preset_choice) - 1]

    # Provider selection
    console.print()
    console.print("  [bold]LLM Provider:[/]")
    provider_list = list(LLM_PROVIDERS.items())
    for i, (key, info) in enumerate(provider_list, 1):
        console.print(f"    [yellow]{i}[/]  {info['label']}")
    console.print()
    provider_choice = Prompt.ask(
        "  Provider",
        choices=[str(i) for i in range(1, len(provider_list) + 1)],
        default="1",
    )
    provider = provider_list[int(provider_choice) - 1][0]

    # Feature toggles
    console.print()
    console.print("  [bold]Features:[/]")
    default_features = BRAIN_PACK_PRESETS[preset]["features"]
    features: Dict[str, bool] = {}
    for feat_name, feat_default in default_features.items():
        features[feat_name] = Confirm.ask(
            f"    {feat_name.capitalize()}",
            default=feat_default,
        )

    port = IntPrompt.ask("\n  Port", default=DEFAULT_PORT)

    return {
        "_mode": "workspace_app",
        "name": name,
        "display_name": display_name,
        "description": description,
        "preset": preset,
        "provider": provider,
        "features": features,
        "port": port,
    }


# ---------------------------------------------------------------------------
# Platform template support
# ---------------------------------------------------------------------------

_AGENT_DEFAULTS = {
    "orchestrator": {
        "role": "orchestrator",
        "description": "System orchestrator — coordinates all agents",
        "work_types": ["management", "delegation", "coordination"],
        "effort_cap": 10,
        "hourly_budget": 60,
        "daily_budget": 500,
        "can_spawn": True,
        "can_delegate": True,
    },
    "assistant": {
        "role": "assistant",
        "description": "General-purpose assistant for user interactions",
        "work_types": ["chat", "analysis", "generation"],
        "effort_cap": 8,
        "hourly_budget": 30,
        "daily_budget": 200,
        "can_spawn": False,
        "can_delegate": False,
    },
}


def _pascal_case(name: str) -> str:
    """Convert a slug to PascalCase: acme-ai -> AcmeAi"""
    return "".join(w.capitalize() for w in re.split(r"[-_ ]+", name))


def _compute_services(tier: str, port_base: int) -> Dict[str, int]:
    """Compute a deterministic service→port map for a given tier."""
    services: Dict[str, int] = {
        "Genesis": port_base,
        "Node": port_base + 1,
        "Pulse": port_base + 2,
        "Secrets": port_base + 10,
        "Chronicle": port_base + 20,
    }
    if tier in ("standard", "full"):
        services.update({
            "Flux": port_base + 16,
            "MicroScheduler": port_base + 49,
            "Directory": port_base + 30,
            "Mind": port_base + 50,
            "MemoryCore": port_base + 60,
            "SecurityCore": port_base + 70,
            "Watch": port_base + 3,
        })
    if tier == "full":
        services.update({
            "Identity": port_base + 71,
            "CognitionCore": port_base + 51,
            "PerceptionCore": port_base + 80,
            "WorkingMemory": port_base + 61,
            "Scheduler": port_base + 90,
            "Council": port_base + 100,
            "Mesh": port_base + 110,
            "Strata": port_base + 35,
            "Trainer": port_base + 120,
        })

    # Validate uniqueness
    ports_seen: Dict[int, str] = {}
    for svc_name, svc_port in services.items():
        if svc_port in ports_seen:
            raise ValueError(
                f"Port collision: {svc_name} and {ports_seen[svc_port]} both use {svc_port}"
            )
        ports_seen[svc_port] = svc_name

    return services


class _ServiceMeta:
    """Lightweight holder for service metadata used in templates."""
    __slots__ = ("group", "layer", "boot_tier", "description")

    def __init__(self, group: str, layer: int, boot_tier: int, description: str):
        self.group = group
        self.layer = layer
        self.boot_tier = boot_tier
        self.description = description


def _build_services_meta(tier: str) -> Dict[str, _ServiceMeta]:
    """Build metadata for each service in a tier."""
    meta: Dict[str, _ServiceMeta] = {
        "Genesis": _ServiceMeta("core", 5, 1, "System orchestrator"),
        "Node": _ServiceMeta("core", 1, 0, "MCP tool server"),
        "Pulse": _ServiceMeta("core", 1, 0, "Health monitoring"),
        "Secrets": _ServiceMeta("infrastructure", 0, 0, "Secret vault"),
        "Chronicle": _ServiceMeta("infrastructure", 0, 0, "Structured logging"),
    }
    if tier in ("standard", "full"):
        meta.update({
            "Flux": _ServiceMeta("infrastructure", 0, 0, "Event bus"),
            "MicroScheduler": _ServiceMeta("core", 1, 1, "LLM queue"),
            "Directory": _ServiceMeta("infrastructure", 0, 0, "Service discovery"),
            "Mind": _ServiceMeta("cognition", 3, 2, "Cognitive core"),
            "MemoryCore": _ServiceMeta("memory", 4, 2, "Unified memory"),
            "SecurityCore": _ServiceMeta("security", 8, 1, "Auth and RBAC"),
            "Watch": _ServiceMeta("core", 1, 0, "Service supervisor"),
        })
    if tier == "full":
        meta.update({
            "Identity": _ServiceMeta("security", 8, 1, "Agent identity"),
            "CognitionCore": _ServiceMeta("cognition", 3, 2, "Advanced reasoning"),
            "PerceptionCore": _ServiceMeta("perception", 2, 2, "Input processing"),
            "WorkingMemory": _ServiceMeta("memory", 4, 2, "Session memory"),
            "Scheduler": _ServiceMeta("automation", 7, 2, "Job scheduling"),
            "Council": _ServiceMeta("agents", 5, 2, "Multi-agent coordination"),
            "Mesh": _ServiceMeta("mesh", 8, 3, "Cluster federation"),
            "Strata": _ServiceMeta("infrastructure", 0, 0, "Tiered data storage"),
            "Trainer": _ServiceMeta("training", 9, 3, "Model fine-tuning"),
        })
    return meta


def build_platform_context(
    name: str,
    tier: str = "standard",
    port_base: int = DEFAULT_PLATFORM_PORT_BASE,
    agents: Optional[List[str]] = None,
    org: str = "",
    domain: str = "",
    description: str = "",
) -> Dict[str, Any]:
    """Build the Jinja2 template context for a platform template."""
    slug = _slugify(name)
    env_pfx = _env_prefix(name)
    display_name = _title_case(name)
    platform_name = _pascal_case(name)
    initial_agents = agents or list(DEFAULT_PLATFORM_AGENTS)
    service_ports = _compute_services(tier, port_base)
    services_meta = _build_services_meta(tier)

    return {
        # Platform identity
        "platform_name": platform_name,
        "platform_slug": slug,
        "platform_display_name": display_name,
        "platform_description": description or f"{display_name} agent platform",
        "env_prefix": env_pfx,
        "org": org,
        "domain": domain,

        # Tier and services
        "tier": tier,
        "port_base": port_base,
        "service_ports": service_ports,
        "services_meta": services_meta,
        "service_count": len(service_ports),

        # Agents
        "initial_agents": initial_agents,

        # Template metadata
        "template_type": "platform",
    }


def scaffold_platform(
    name: str,
    tier: str = "standard",
    port_base: int = DEFAULT_PLATFORM_PORT_BASE,
    agents: Optional[List[str]] = None,
    org: str = "",
    domain: str = "",
    description: str = "",
    output_dir: Optional[str] = None,
) -> Path:
    """Scaffold a complete platform project.

    Returns the path to the created project directory.
    """
    ctx = build_platform_context(name, tier, port_base, agents, org, domain, description)
    slug = ctx["platform_slug"]
    target = Path(output_dir or slug)
    initial_agents = ctx["initial_agents"]

    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} already exists and is not empty")

    template_dir = _TEMPLATES_DIR / "platform"
    if not template_dir.exists():
        raise ValueError("Platform template not found in templates directory")

    # Walk template directory and render all .j2 files
    for src_path in sorted(template_dir.rglob("*")):
        if src_path.is_dir():
            continue
        if "__pycache__" in str(src_path):
            continue

        rel = src_path.relative_to(template_dir)
        rel_str = str(rel)

        # Skip the agent identity template — we render it per-agent below
        if "__agent__" in rel_str:
            continue

        dest_name = rel_str
        if dest_name.endswith(".j2"):
            dest_name = dest_name[:-3]

        dest_path = target / dest_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix == ".j2":
            template_text = src_path.read_text(encoding="utf-8")
            rendered = _render_template(template_text, ctx)
            dest_path.write_text(rendered, encoding="utf-8")
        else:
            shutil.copy2(src_path, dest_path)

    # Generate per-agent identity files
    agent_template_path = template_dir / "config" / "identities" / "__agent__.yaml.j2"
    if agent_template_path.exists():
        agent_template_text = agent_template_path.read_text(encoding="utf-8")
        for agent_name in initial_agents:
            defaults = _AGENT_DEFAULTS.get(agent_name, _AGENT_DEFAULTS["assistant"])
            agent_ctx = {
                **ctx,
                "agent_identity_name": agent_name,
                "agent_identity_role": defaults["role"],
                "agent_identity_description": defaults["description"],
                "agent_identity_work_types": defaults["work_types"],
                "agent_identity_effort_cap": defaults["effort_cap"],
                "agent_identity_hourly_budget": defaults["hourly_budget"],
                "agent_identity_daily_budget": defaults["daily_budget"],
                "agent_identity_can_spawn": defaults["can_spawn"],
                "agent_identity_can_delegate": defaults["can_delegate"],
            }
            rendered = _render_template(agent_template_text, agent_ctx)
            identity_path = target / "config" / "identities" / f"{agent_name}.yaml"
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            identity_path.write_text(rendered, encoding="utf-8")

    log.info("Scaffolded platform '%s' at %s (tier=%s, services=%d, agents=%s)",
             name, target, tier, ctx["service_count"], initial_agents)
    return target


def _interactive_create_plain() -> Dict[str, Any]:
    """Fallback interactive prompts when Rich is not installed."""
    print("\n  AitherADK — Create a new agent\n")

    print("  What kind of agent?")
    print("    1  Workspace App    (full product with UI, RAG, Docker)")
    print("    2  Standalone Agent (agent.yaml + tools — lightweight)")
    print("    3  Identity Only    (deploy to running AitherOS)")
    mode_choice = input("\n  Choice [1/2/3] (1): ").strip() or "1"

    name = input("  Agent name: ").strip() or "my-agent"
    display_name = input(f"  Display name ({_title_case(name)}): ").strip() or _title_case(name)
    description = input("  Description (optional): ").strip()

    if mode_choice == "3":
        return {
            "_mode": "identity",
            "name": name,
            "display_name": display_name,
            "description": description,
        }

    if mode_choice == "2":
        print(f"\n  Available templates:")
        for key, desc in TEMPLATES.items():
            marker = " (default)" if key == "sovereign" else ""
            print(f"    {key:12s} — {desc}{marker}")
        template = input(f"\n  Template (sovereign): ").strip() or "sovereign"
        if template not in TEMPLATES:
            template = "sovereign"
        port_str = input(f"  Port ({DEFAULT_PORT}): ").strip()
        port = int(port_str) if port_str.isdigit() else DEFAULT_PORT
        pack_path = input("  Brain pack YAML path (optional): ").strip() or None
        return {
            "name": name,
            "template": template,
            "pack_path": pack_path,
            "port": port,
            "description": description,
        }

    # Workspace App
    print("\n  Brain Pack Preset:")
    preset_keys = list(BRAIN_PACK_PRESETS.keys())
    for i, (key, preset) in enumerate(BRAIN_PACK_PRESETS.items(), 1):
        print(f"    {i}  {key.capitalize():12s} — {preset['description']}")
    preset_choice = input("\n  Preset (1): ").strip() or "1"
    preset = preset_keys[int(preset_choice) - 1] if preset_choice.isdigit() else "generic"

    print("\n  LLM Provider:")
    provider_list = list(LLM_PROVIDERS.items())
    for i, (key, info) in enumerate(provider_list, 1):
        print(f"    {i}  {info['label']}")
    provider_choice = input("\n  Provider (1): ").strip() or "1"
    provider = provider_list[int(provider_choice) - 1][0] if provider_choice.isdigit() else "portal"

    port_str = input(f"  Port ({DEFAULT_PORT}): ").strip()
    port = int(port_str) if port_str.isdigit() else DEFAULT_PORT

    return {
        "_mode": "workspace_app",
        "name": name,
        "display_name": display_name,
        "description": description,
        "preset": preset,
        "provider": provider,
        "features": BRAIN_PACK_PRESETS.get(preset, {}).get("features", {}),
        "port": port,
    }
