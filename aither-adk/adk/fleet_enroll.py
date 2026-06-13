"""Fleet enrollment for local AitherOS nodes — auto-register on boot.

Provides both Node and Agent fleet registration (portal + local Genesis).

Two-fleet model:
  - PORTAL fleet: external discovery via portal.aitherium.com (FederationLiteClient)
  - LOCAL fleet: Genesis/Node for in-network routing

On boot (after services are live), fleet_enroll.enroll_on_boot() will:
  1. Register the local node (via federation or direct Genesis API)
  2. Persist node_id + api_key to ~/.aither/node_auth.json
  3. Start a background heartbeat loop (every 60s)
  4. Scan ~/.aither/agents.json and upsert agents to the portal fleet

All operations are graceful: failures log but do not crash boot. Gated by
AITHER_FLEET_ENROLL env var (or config flag).

Usage:

    from adk.fleet_enroll import enroll_on_boot
    # After services are up:
    await enroll_on_boot()

"""

from __future__ import annotations

__all__ = [
    "enroll_on_boot",
]

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("adk.fleet_enroll")

_AITHER_DIR = Path.home() / ".aither"
_NODE_AUTH_FILE = _AITHER_DIR / "node_auth.json"
_AGENTS_FILE = _AITHER_DIR / "agents.json"
_AUTH_FILE = _AITHER_DIR / "auth.json"


def _should_enroll() -> bool:
    """Check if fleet enrollment is enabled.

    Checks (in order):
      1. AITHER_FLEET_ENROLL env var (true/1)
      2. Config entry in AITHER_CONFIG_FILE (fleet.enroll: true)
      3. Default: False (offline/sovereign by default)
    """
    if os.environ.get("AITHER_FLEET_ENROLL", "").lower() in ("true", "1"):
        return True

    # Check config file if it exists
    config_file = _AITHER_DIR / "config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
            if config.get("fleet", {}).get("enroll"):
                return True
        except Exception:
            pass

    return False


def _load_auth_config() -> Dict[str, Any]:
    """Load auth.json to get tenant info."""
    if not _AUTH_FILE.exists():
        return {}
    try:
        return json.loads(_AUTH_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read auth.json: %s", e)
        return {}


def _load_node_auth() -> Dict[str, Any]:
    """Load node_auth.json (persistent node identity)."""
    if not _NODE_AUTH_FILE.exists():
        return {}
    try:
        return json.loads(_NODE_AUTH_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read node_auth.json: %s", e)
        return {}


def _save_node_auth(data: Dict[str, Any]) -> None:
    """Persist node_auth.json."""
    _AITHER_DIR.mkdir(parents=True, exist_ok=True)
    data["enrolled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _NODE_AUTH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_agents_registry() -> Dict[str, Any]:
    """Load agents.json (local agent registry)."""
    if not _AGENTS_FILE.exists():
        return {"agents": {}}
    try:
        data = json.loads(_AGENTS_FILE.read_text(encoding="utf-8"))
        if "agents" not in data:
            data = {"agents": data}
        return data
    except Exception as e:
        log.warning("Failed to read agents.json: %s", e)
        return {"agents": {}}


async def _register_node_with_federation(
    hub_url: str,
    api_key: str,
    node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Register node with federation hub (portal or local Genesis).

    Args:
        hub_url: Portal or Genesis URL (e.g., https://portal.aitherium.com or http://localhost:8001)
        api_key: API key or token for authentication
        node_id: Optional pre-assigned node ID

    Returns:
        {"node_id": "...", "api_key": "...", ...} on success
        {"error": True, ...} on failure
    """
    try:
        from adk.federation_lite import FederationLiteClient

        tenant_slug = _extract_tenant_slug()
        client = FederationLiteClient(
            hub_url=hub_url,
            api_key=api_key,
            node_id=node_id,
        )
        result = await client.register(tenant_slug)
        if not result.get("error"):
            log.info("Successfully registered node with %s", hub_url)
            return {
                "node_id": result.get("node_id", client.node_id),
                "api_key": result.get("api_key", api_key),
                "hub_url": hub_url,
            }
        else:
            log.warning("Federation registration failed: %s", result.get("detail", "unknown"))
            return {"error": True, "detail": result.get("detail")}
    except Exception as e:
        log.warning("Federation registration error: %s", e)
        return {"error": True, "detail": str(e)}


async def _register_node_with_genesis(genesis_url: str, api_key: str) -> Dict[str, Any]:
    """Fallback: Register node directly with Genesis POST /federation/register.

    Args:
        genesis_url: Genesis service URL (e.g., http://localhost:8001)
        api_key: Bearer token

    Returns:
        {"node_id": "...", "api_key": "...", ...} on success
    """
    try:
        import httpx

        node_id = _generate_node_id()
        payload = {
            "tenant_slug": _extract_tenant_slug(),
            "node_id": node_id,
            "timestamp": int(time.time()),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{genesis_url.rstrip('/')}/federation/register",
                json=payload,
                headers=headers,
            )
            if resp.status_code >= 200 and resp.status_code < 300:
                data = resp.json()
                log.info("Successfully registered node with Genesis")
                return {
                    "node_id": data.get("node_id", node_id),
                    "api_key": data.get("api_key", api_key),
                    "hub_url": genesis_url,
                }
            else:
                log.warning("Genesis registration failed: %s", resp.text[:200])
                return {"error": True, "detail": resp.text[:200]}
    except Exception as e:
        log.warning("Genesis registration error: %s", e)
        return {"error": True, "detail": str(e)}


def _generate_node_id() -> str:
    """Generate a unique node ID (e.g., adk-a3f8b2c1-7x9k)."""
    import hashlib
    import secrets
    import socket
    import uuid

    # Fingerprint: hostname + MAC + home dir
    fingerprint = f"{socket.gethostname()}-{uuid.getnode()}-{Path.home()}"
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:8]
    rand = secrets.token_hex(4)
    return f"adk-{digest}-{rand}"


def _extract_tenant_slug() -> str:
    """Extract tenant_slug from auth.json or fall back to default.

    Returns a tenant slug suitable for federation (e.g., "personal", "acme-corp", etc.)
    """
    auth = _load_auth_config()
    slug = auth.get("tenant_slug") or auth.get("user", {}).get("tenant_slug", "")
    if slug:
        return slug

    # Fallback: use username or "personal"
    username = auth.get("user", {}).get("username", "")
    if username:
        return username.lower().replace(" ", "-")

    return "personal"


async def _upsert_agents_to_portal(hub_url: str, api_key: str, node_id: str) -> bool:
    """Upsert all local agents to the portal fleet.

    Reads ~/.aither/agents.json and sends to /v1/agents/upsert-batch on the hub.

    Returns True if all agents were sent (or no agents to send), False if error.
    """
    try:
        from adk.federation_lite import FederationLiteClient

        registry = _load_agents_registry()
        agents_dict = registry.get("agents", {})
        if not agents_dict:
            log.info("No local agents to upsert")
            return True

        # Convert registry entries to agent cards
        agents = []
        for name, entry in agents_dict.items():
            agents.append({
                "name": name,
                "url": entry.get("url", f"http://localhost:{entry.get('port', 8000)}"),
                "capabilities": entry.get("capabilities", []),
                "status": entry.get("status", "active"),
                "metadata": entry.get("metadata", {}),
            })

        client = FederationLiteClient(
            hub_url=hub_url,
            api_key=api_key,
            node_id=node_id,
        )
        result = await client.upsert_agents(agents)
        if result.get("error"):
            log.warning("Agent upsert failed: %s", result.get("detail"))
            return False
        log.info("Upserted %d agents to portal", len(agents))
        return True
    except Exception as e:
        log.warning("Agent upsert error: %s", e)
        return False


async def _heartbeat_loop(
    hub_url: str,
    api_key: str,
    node_id: str,
    interval: int = 60,
) -> None:
    """Background heartbeat loop — sends periodic status to the hub.

    Runs indefinitely (in background). Sends every `interval` seconds.
    Failures are logged but do not break the loop.
    """
    try:
        from adk.federation_lite import FederationLiteClient

        client = FederationLiteClient(
            hub_url=hub_url,
            api_key=api_key,
            node_id=node_id,
        )

        log.info("Starting heartbeat loop (interval=%ds)", interval)
        while True:
            await asyncio.sleep(interval)
            try:
                registry = _load_agents_registry()
                agents = list(registry.get("agents", {}).keys())
                metrics = {
                    "agents_active": len(agents),
                    "timestamp": int(time.time()),
                }
                result = await client.heartbeat(
                    status="online",
                    metrics=metrics,
                    agents=[{"name": a} for a in agents] if agents else None,
                )
                if not result.get("error"):
                    log.debug("Heartbeat sent (agents=%d)", len(agents))
                else:
                    log.debug("Heartbeat failed: %s", result.get("detail"))
            except asyncio.CancelledError:
                log.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                log.debug("Heartbeat error: %s", e)
    except Exception as e:
        log.warning("Heartbeat loop setup failed: %s", e)


async def enroll_on_boot(
    genesis_url: Optional[str] = None,
    portal_url: Optional[str] = None,
    enable_heartbeat: bool = True,
) -> Dict[str, Any]:
    """Enroll this node into both local and portal fleets.

    Called by process_supervisor after services are up. Orchestrates:
      1. Node registration (federation or Genesis fallback)
      2. Persist node_id + api_key
      3. Agent upsert to portal
      4. Start heartbeat loop (optional, background)

    Args:
        genesis_url: Local Genesis URL (default http://localhost:8001)
        portal_url: Portal hub URL (default https://portal.aitherium.com)
        enable_heartbeat: Start background heartbeat loop (default True)

    Returns:
        {
            "enrolled": bool,
            "node_id": str,
            "agents_upserted": int,
            "error": optional error detail
        }
    """
    if not _should_enroll():
        log.debug("Fleet enrollment disabled (AITHER_FLEET_ENROLL not set)")
        return {
            "enrolled": False,
            "reason": "enrollment disabled",
        }

    # Defaults
    genesis_url = genesis_url or os.environ.get("AITHER_GENESIS_URL", "http://localhost:8001")
    portal_url = portal_url or os.environ.get("AITHER_PORTAL_URL", "https://portal.aitherium.com")

    log.info("Enrolling node in fleet (genesis=%s, portal=%s)", genesis_url, portal_url)

    # Load or fetch API key
    auth = _load_auth_config()
    api_key = auth.get("access_token") or auth.get("user", {}).get("api_key") or "aither_root_local"

    # Check if already registered
    node_auth = _load_node_auth()
    if node_auth.get("node_id"):
        log.info("Node already enrolled: %s", node_auth["node_id"])
        # Still upsert agents and start heartbeat if enabled
        if enable_heartbeat and not node_auth.get("_heartbeat_started"):
            asyncio.create_task(
                _heartbeat_loop(
                    node_auth.get("hub_url", genesis_url),
                    node_auth.get("api_key", api_key),
                    node_auth["node_id"],
                )
            )
            node_auth["_heartbeat_started"] = True
            _save_node_auth(node_auth)
        return {
            "enrolled": True,
            "node_id": node_auth["node_id"],
            "already_registered": True,
        }

    # Try portal first
    node_id = _generate_node_id()
    result = await _register_node_with_federation(portal_url, api_key, node_id)
    if result.get("error"):
        # Fallback to Genesis
        log.info("Portal registration failed, falling back to Genesis")
        result = await _register_node_with_genesis(genesis_url, api_key)

    if result.get("error"):
        log.error("Node registration failed: %s", result.get("detail"))
        return {
            "enrolled": False,
            "error": result.get("detail", "unknown"),
        }

    # Persist
    node_id = result["node_id"]
    api_key = result.get("api_key", api_key)
    hub_url = result.get("hub_url", genesis_url)
    _save_node_auth({
        "node_id": node_id,
        "api_key": api_key,
        "hub_url": hub_url,
        "tenant_slug": _extract_tenant_slug(),
    })

    log.info("Node enrolled: %s", node_id)

    # Upsert agents
    agents_upserted = 0
    if await _upsert_agents_to_portal(hub_url, api_key, node_id):
        registry = _load_agents_registry()
        agents_upserted = len(registry.get("agents", {}))

    # Start heartbeat
    if enable_heartbeat:
        asyncio.create_task(
            _heartbeat_loop(hub_url, api_key, node_id)
        )

    return {
        "enrolled": True,
        "node_id": node_id,
        "agents_upserted": agents_upserted,
    }
