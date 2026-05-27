"""Brain pack + skill pack discovery for workspace mode.

Discovers installed packs via:
1. ``AGENT_BRAIN_PACK`` env var (explicit path)
2. ``brain_pack.yaml`` in current directory
3. Python entry points group ``aither.brain_packs``
4. ``~/.aither/packs/`` directory scan
5. Bundled ``adk/workspace-agent.yaml`` fallback

A "pack" is a pip-installable package that ships:
- ``brain_pack.yaml``  — persona, features, doc_types, UI labels, safety
- ``agent.yaml``       — capabilities, portal config, enabled_domains
- ``skills/``          — skill assets (.md files)
- ``packs/``           — tool pack manifests

Customers install: ``pip install aither-adk aither-pack-gargbot``
Then run: ``adk-workspace`` — auto-discovers the brain pack.

Entry point registration in the pack's pyproject.toml:
    [project.entry-points."aither.brain_packs"]
    gargbot = "aither_pack_gargbot:get_pack_dir"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("adk.pack_discovery")


def _find_entrypoint_packs() -> list[dict]:
    """Discover brain packs registered via Python entry points."""
    packs = []
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.10-3.11 returns dict
        if hasattr(eps, "select"):
            brain_eps = eps.select(group="aither.brain_packs")
        else:
            brain_eps = eps.get("aither.brain_packs", [])

        for ep in brain_eps:
            try:
                get_dir = ep.load()
                pack_dir = Path(get_dir())
                if pack_dir.exists():
                    packs.append({
                        "name": ep.name,
                        "dir": pack_dir,
                        "source": "entrypoint",
                    })
                    logger.info("Discovered brain pack via entry point: %s -> %s", ep.name, pack_dir)
            except Exception as e:
                logger.debug("Failed to load entry point %s: %s", ep.name, e)
    except Exception as e:
        logger.debug("Entry point discovery failed: %s", e)
    return packs


def _find_local_packs() -> list[dict]:
    """Scan ~/.aither/packs/ for brain packs."""
    packs = []
    packs_dir = Path(os.getenv("AITHER_PACKS_DIR", os.path.expanduser("~/.aither/packs")))
    if not packs_dir.is_dir():
        return packs

    for candidate in sorted(packs_dir.iterdir()):
        if candidate.is_dir():
            bp = candidate / "brain_pack.yaml"
            if bp.exists():
                packs.append({
                    "name": candidate.name,
                    "dir": candidate,
                    "source": "local",
                })
                logger.info("Found local brain pack: %s -> %s", candidate.name, candidate)
    return packs


def discover_brain_pack() -> Optional[Path]:
    """Find the best brain pack YAML to use.

    Returns the path to a brain_pack.yaml, or None if only defaults should be used.

    Priority:
    1. AGENT_BRAIN_PACK env var
    2. brain_pack.yaml in CWD
    3. Entry-point registered packs (first found)
    4. ~/.aither/packs/<name>/brain_pack.yaml (first found)
    5. None (use defaults)
    """
    # 1. Explicit env var
    env_path = os.getenv("AGENT_BRAIN_PACK")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            logger.info("Brain pack from AGENT_BRAIN_PACK: %s", p)
            return p
        logger.warning("AGENT_BRAIN_PACK=%s not found", env_path)

    # 2. CWD
    cwd_pack = Path.cwd() / "brain_pack.yaml"
    if cwd_pack.exists():
        logger.info("Brain pack from CWD: %s", cwd_pack)
        return cwd_pack

    # 3. Entry points
    ep_packs = _find_entrypoint_packs()
    if ep_packs:
        bp = ep_packs[0]["dir"] / "brain_pack.yaml"
        if bp.exists():
            return bp

    # 4. Local packs dir
    local_packs = _find_local_packs()
    if local_packs:
        bp = local_packs[0]["dir"] / "brain_pack.yaml"
        if bp.exists():
            return bp

    return None


def discover_agent_yaml() -> Optional[Path]:
    """Find the best agent.yaml to use.

    Same priority as discover_brain_pack but looks for agent.yaml.
    Falls back to the bundled workspace-agent.yaml in the ADK package.
    """
    # Explicit env
    env_path = os.getenv("AGENT_YAML")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    # CWD
    cwd = Path.cwd() / "agent.yaml"
    if cwd.exists():
        return cwd

    # Entry-point pack
    ep_packs = _find_entrypoint_packs()
    if ep_packs:
        ay = ep_packs[0]["dir"] / "agent.yaml"
        if ay.exists():
            return ay

    # Local packs
    local_packs = _find_local_packs()
    if local_packs:
        ay = local_packs[0]["dir"] / "agent.yaml"
        if ay.exists():
            return ay

    # Bundled fallback (shipped with ADK)
    bundled = Path(__file__).resolve().parent.parent / "agent.yaml"
    if bundled.exists():
        return bundled

    # Package data fallback
    pkg_bundled = Path(__file__).resolve().parent / "workspace-agent.yaml"
    if pkg_bundled.exists():
        return pkg_bundled

    return None


def discover_pack_dir() -> Optional[Path]:
    """Find the pack directory (contains skills/, packs/, brain_pack.yaml).

    Used by workspace mode to set AGENT_BRAIN_PACK and load skills/tool packs.
    """
    # Entry-point packs
    ep_packs = _find_entrypoint_packs()
    if ep_packs:
        return ep_packs[0]["dir"]

    # Local packs
    local_packs = _find_local_packs()
    if local_packs:
        return local_packs[0]["dir"]

    # CWD if it has a brain_pack.yaml
    if (Path.cwd() / "brain_pack.yaml").exists():
        return Path.cwd()

    return None


def list_available_packs() -> list[dict]:
    """List all discoverable packs (for CLI/UI display)."""
    packs = _find_entrypoint_packs() + _find_local_packs()

    # Deduplicate by name
    seen = set()
    unique = []
    for p in packs:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)
    return unique
