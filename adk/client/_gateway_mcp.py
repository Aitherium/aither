"""Gateway MCP Client for self-hosted agents.

Wires the agent's MCP client to the gateway (mcp.aitherium.com) authenticated as the owner.
This allows self-hosted agents to access platform tools and secrets.

Authentication hierarchy:
  1. Device-flow token (aither_pat_*) from AitherIdentity — preferred for self-hosted
  2. ACTA key (aither_sk_live_*) — cloud billing-backed
  3. Identity Bearer token — internal admin
  4. Local AitherSecrets self-mint (fallback only)

Fail-closed: no token => log error, MCP stays disabled (agent still runs locally).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("adk.client.gateway_mcp")


class GatewayMCPClient:
    """Connects self-hosted agent to gateway MCP endpoint.

    Usage:
        client = GatewayMCPClient(
            gateway_url="https://mcp.aitherium.com",
            api_key="aither_pat_xxxx"
        )
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("get_secret", {"name": "github_token"})
    """

    def __init__(
        self,
        gateway_url: str = "",
        api_key: str = "",
        timeout: float = 30.0,
    ):
        """Initialize gateway MCP client.

        Args:
            gateway_url: MCP gateway endpoint (default: https://mcp.aitherium.com)
            api_key: Bearer token for authentication (aither_pat_*, aither_sk_live_*, etc.)
            timeout: HTTP request timeout in seconds
        """
        self.gateway_url = (
            gateway_url or os.getenv("AITHER_GATEWAY_URL", "https://mcp.aitherium.com")
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("AITHER_API_KEY", "")
            or os.getenv("AITHER_MCP_KEY", "")
        )
        self._timeout = timeout
        self._connected = False
        self._request_id = 0

    def _next_id(self) -> int:
        """Generate next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def connect(self) -> bool:
        """Test connection to gateway.

        Returns True if gateway is reachable and responds to /health.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, headers=self._headers()
            ) as client:
                resp = await client.get(f"{self.gateway_url}/health")
            self._connected = resp.status_code == 200
            if self._connected:
                logger.info("Gateway MCP client connected: %s", self.gateway_url)
            else:
                logger.warning("Gateway returned %d", resp.status_code)
            return self._connected
        except httpx.ConnectError as exc:
            logger.warning("Gateway unreachable at %s: %s", self.gateway_url, exc)
            return False
        except Exception as exc:
            logger.warning("Gateway connection check failed: %s", exc)
            return False

    async def list_tools(self) -> list[dict]:
        """Discover available tools on gateway.

        Returns list of tool dicts with name, description, parameters.
        Filters by caller's tier and RBAC.
        """
        if not self.api_key:
            logger.error("Cannot list tools: no API key configured")
            return []

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, headers=self._headers()
            ) as client:
                resp = await client.post(
                    f"{self.gateway_url}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "id": self._next_id(),
                    },
                )

            if resp.status_code == 401:
                logger.error("Authorization failed: invalid or expired token")
                return []
            if resp.status_code == 402:
                logger.warning("Insufficient balance (402)")
                return []
            if resp.status_code >= 400:
                logger.warning("Gateway returned %d: %s", resp.status_code, resp.text[:200])
                return []

            data = resp.json()

            # Check for JSON-RPC error
            if "error" in data:
                err = data["error"]
                logger.error(
                    "MCP error %d: %s",
                    err.get("code", -1),
                    err.get("message", ""),
                )
                return []

            tools = []
            for t in data.get("result", {}).get("tools", []):
                tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {}),
                })
            return tools

        except httpx.ConnectError:
            logger.warning("Gateway unreachable")
            return []
        except Exception as exc:
            logger.warning("Tool discovery failed: %s", exc)
            return []

    async def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Call a tool on the gateway.

        Args:
            name: Tool name
            arguments: Tool arguments dict

        Returns:
            Result dict from MCP response (may contain error)
        """
        if not self.api_key:
            return {"error": "no_api_key", "message": "API key not configured"}

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, headers=self._headers()
            ) as client:
                resp = await client.post(
                    f"{self.gateway_url}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments or {}},
                        "id": self._next_id(),
                    },
                )

            if resp.status_code == 401:
                logger.error("Authorization failed")
                return {"error": "auth_failed", "message": "Invalid or expired token"}
            if resp.status_code == 402:
                logger.warning("Insufficient balance")
                return {"error": "insufficient_balance", "message": "No Aitherium tokens"}
            if resp.status_code == 403:
                logger.warning("Access denied to tool: %s", name)
                return {"error": "access_denied", "message": f"Tool '{name}' not allowed"}
            if resp.status_code >= 400:
                logger.warning("Gateway error %d: %s", resp.status_code, resp.text[:200])
                return {"error": "gateway_error", "status": resp.status_code}

            data = resp.json()

            # Check for JSON-RPC error
            if "error" in data:
                err = data["error"]
                return {
                    "error": "mcp_error",
                    "code": err.get("code", -1),
                    "message": err.get("message", ""),
                }

            result = data.get("result", {})
            content = result.get("content", [])

            # Extract text from content array
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)

            return {
                "success": True,
                "text": "\n".join(texts) if texts else "",
                "content": content,
                "raw": result,
            }

        except httpx.ConnectError:
            logger.warning("Gateway unreachable")
            return {"error": "gateway_unreachable"}
        except Exception as exc:
            logger.warning("Tool call failed: %s", exc)
            return {"error": "call_failed", "message": str(exc)}


async def create_gateway_mcp_client(
    gateway_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[GatewayMCPClient]:
    """Factory function to create and test gateway MCP client.

    Args:
        gateway_url: Override gateway URL
        api_key: Override API key

    Returns:
        Connected client, or None if no token available or connection failed.
        Never raises — fails soft.
    """
    # Resolve API key in order of preference
    key = (
        api_key
        or os.getenv("AITHER_API_KEY", "")
        or os.getenv("AITHER_MCP_KEY", "")
    )

    if not key:
        logger.debug("Gateway MCP: no API key configured (not an error)")
        return None

    # Resolve gateway URL
    url = (
        gateway_url
        or os.getenv("AITHER_GATEWAY_URL", "")
        or "https://mcp.aitherium.com"
    )

    try:
        client = GatewayMCPClient(gateway_url=url, api_key=key)
        if await client.connect():
            return client
        else:
            logger.warning("Gateway MCP connection failed (continuing offline)")
            return None
    except Exception as exc:  # noqa: BLE001 — must never crash startup
        logger.warning("Gateway MCP client creation failed: %s (continuing offline)", exc)
        return None
