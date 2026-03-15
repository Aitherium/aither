"""Service bridge — auto-discover and connect to AitherOS services via AitherNode.

When AitherNode (localhost:8080) or the MCP gateway (mcp.aitherium.com) is
reachable, this module auto-discovers available services and registers their
tools on agents. When offline, agents fall back to built-in tools only.

Service tiers:
  Tier 0 (always): Built-in tools (file_io, shell, python, web, secrets)
  Tier 1 (local):  AitherNode stdio/HTTP MCP (449 tools when Genesis running)
  Tier 2 (cloud):  MCP Gateway at mcp.aitherium.com (filtered by tenant tier)

Usage:
    from adk.services import ServiceBridge, get_service_bridge

    bridge = get_service_bridge()
    await bridge.connect()  # auto-detect available services
    await bridge.register_on_agent(agent)  # inject discovered tools

    # Check what's available
    status = await bridge.status()
    # {"node_available": True, "genesis_available": True, "tools_count": 449, ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from adk.agent import AitherAgent

logger = logging.getLogger("adk.services")

_DEFAULT_NODE_URL = "http://localhost:8080"
_DEFAULT_GATEWAY_URL = "https://mcp.aitherium.com"


@dataclass
class ServiceStatus:
    """Status of connected AitherOS services."""
    node_available: bool = False
    genesis_available: bool = False
    gateway_available: bool = False
    mode: str = "standalone"  # standalone, local, cloud
    tools_count: int = 0
    services_discovered: list[str] = field(default_factory=list)
    node_url: str = ""
    gateway_url: str = ""
    last_check: float = 0.0


class ServiceBridge:
    """Auto-discovery bridge to AitherOS services.

    Probes AitherNode (local) and MCP Gateway (cloud) to determine
    what's available, then registers tools accordingly.
    """

    def __init__(
        self,
        node_url: str = "",
        gateway_url: str = "",
        api_key: str = "",
        prefer_local: bool = True,
    ):
        self.node_url = (
            node_url
            or os.getenv("AITHER_NODE_URL", _DEFAULT_NODE_URL)
        ).rstrip("/")
        self.gateway_url = (
            gateway_url
            or os.getenv("AITHER_GATEWAY_URL", "")
            or os.getenv("AITHER_MCP_URL", "")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("AITHER_API_KEY", "")
        self._prefer_local = prefer_local

        self._status = ServiceStatus()
        self._mcp_bridge = None  # Lazy: MCPBridge instance
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._status.mode

    async def connect(self) -> ServiceStatus:
        """Probe available services and determine operating mode.

        Checks in order:
        1. AitherNode local (localhost:8080/health)
        2. Genesis (localhost:8001/health) — indicates full AitherOS
        3. MCP Gateway (mcp.aitherium.com/health) — cloud fallback
        """
        import httpx

        self._status.last_check = time.time()

        # Check AitherNode local
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.node_url}/health")
                if resp.status_code == 200:
                    self._status.node_available = True
                    self._status.node_url = self.node_url
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    self._status.services_discovered = data.get("services", [])
                    logger.info("AitherNode available at %s", self.node_url)
        except Exception:
            self._status.node_available = False

        # Check Genesis (full AitherOS)
        if self._status.node_available:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get("http://localhost:8001/health")
                    if resp.status_code == 200:
                        self._status.genesis_available = True
                        logger.info("Genesis available — full AitherOS mode")
            except Exception:
                self._status.genesis_available = False

        # Check MCP Gateway (cloud)
        if self.gateway_url and self.api_key:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        f"{self.gateway_url}/health",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    if resp.status_code == 200:
                        self._status.gateway_available = True
                        self._status.gateway_url = self.gateway_url
                        logger.info("MCP Gateway available at %s", self.gateway_url)
            except Exception:
                self._status.gateway_available = False

        # Determine mode
        if self._status.node_available and self._prefer_local:
            self._status.mode = "local"
        elif self._status.gateway_available:
            self._status.mode = "cloud"
        else:
            self._status.mode = "standalone"

        self._connected = True
        logger.info("Service bridge mode: %s", self._status.mode)
        return self._status

    async def register_on_agent(self, agent: AitherAgent) -> int:
        """Register available service tools on an agent.

        Returns number of MCP tools registered (on top of built-in).
        """
        if not self._connected:
            await self.connect()

        if self._status.mode == "standalone":
            logger.info("Standalone mode — only built-in tools available")
            return 0

        # Get or create MCP bridge
        if self._status.mode == "cloud":
            from adk.mcp import MCPBridge, MCPAuth
            auth = MCPAuth(api_key=self.api_key, gateway_url=self.gateway_url)
            await auth.authenticate()
            self._mcp_bridge = MCPBridge(auth=auth)
        else:
            from adk.mcp import MCPBridge
            self._mcp_bridge = MCPBridge(mcp_url=self.node_url)

        try:
            count = await self._mcp_bridge.register_tools(agent)
            self._status.tools_count = count
            logger.info("Registered %d MCP tools from %s on agent %s",
                         count, self._status.mode, agent.name)
            return count
        except Exception as exc:
            logger.warning("Failed to register MCP tools: %s", exc)
            return 0

    async def call_service(self, tool_name: str, arguments: dict | None = None) -> str:
        """Call an AitherOS service tool directly (bypassing agent).

        Useful for one-off service calls without full agent setup.
        """
        if not self._connected:
            await self.connect()

        if self._mcp_bridge is None:
            if self._status.mode == "standalone":
                return json.dumps({"error": "No AitherOS services available (standalone mode)"})
            if self._status.mode == "cloud":
                from adk.mcp import MCPBridge, MCPAuth
                auth = MCPAuth(api_key=self.api_key, gateway_url=self.gateway_url)
                await auth.authenticate()
                self._mcp_bridge = MCPBridge(auth=auth)
            else:
                from adk.mcp import MCPBridge
                self._mcp_bridge = MCPBridge(mcp_url=self.node_url)

        return await self._mcp_bridge.call_tool(tool_name, arguments or {})

    async def status(self) -> dict:
        """Get current service bridge status."""
        if not self._connected:
            await self.connect()
        return {
            "mode": self._status.mode,
            "node_available": self._status.node_available,
            "genesis_available": self._status.genesis_available,
            "gateway_available": self._status.gateway_available,
            "tools_count": self._status.tools_count,
            "services_discovered": self._status.services_discovered,
            "node_url": self._status.node_url,
            "gateway_url": self._status.gateway_url,
            "last_check": self._status.last_check,
        }

    async def get_available_services(self) -> list[str]:
        """List service names available through the bridge.

        Returns tool categories when connected to AitherNode, or
        just built-in categories in standalone mode.
        """
        if self._status.mode == "standalone":
            return ["file_io", "shell", "python", "web", "secrets"]

        if self._mcp_bridge:
            tools = await self._mcp_bridge.list_tools()
            # Extract unique prefixes as "service" names
            prefixes = set()
            for t in tools:
                parts = t.name.split("_", 1)
                if len(parts) > 1:
                    prefixes.add(parts[0])
            return sorted(prefixes)

        return []


# ─────────────────────────────────────────────────────────────────────────────
# Module singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: ServiceBridge | None = None


def get_service_bridge() -> ServiceBridge:
    """Get or create the module-level ServiceBridge singleton."""
    global _instance
    if _instance is None:
        _instance = ServiceBridge()
    return _instance
