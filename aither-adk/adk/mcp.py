"""MCP bridge — connect ADK agents to AitherOS tools via mcp.aitherium.com."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("adk.mcp")

_DEFAULT_MCP_URL = "https://mcp.aitherium.com"


@dataclass
class MCPTool:
    """A tool available via MCP."""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)


class MCPBridge:
    """Client for AitherOS MCP gateway at mcp.aitherium.com.

    Lets ADK agents use AitherOS tools (code search, memory, agent dispatch, etc.)
    without running the full stack locally.

    Usage:
        bridge = MCPBridge(api_key="your-key")
        tools = await bridge.list_tools()
        result = await bridge.call_tool("explore_code", {"query": "agent dispatch"})

        # Or register all MCP tools into an agent's tool registry
        agent = AitherAgent("atlas")
        await bridge.register_tools(agent)
    """

    def __init__(
        self,
        mcp_url: str = _DEFAULT_MCP_URL,
        api_key: str = "",
        timeout: float = 30.0,
    ):
        self.mcp_url = mcp_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout
        self._tools_cache: list[MCPTool] | None = None

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, headers=self._headers())

    async def list_tools(self, refresh: bool = False) -> list[MCPTool]:
        """List available tools from the MCP gateway."""
        if self._tools_cache and not refresh:
            return self._tools_cache

        async with self._client() as client:
            resp = await client.post(
                f"{self.mcp_url}/mcp",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            )
            resp.raise_for_status()
            data = resp.json()

        tools = []
        for t in data.get("result", {}).get("tools", []):
            tools.append(MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("inputSchema", {}),
            ))
        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call a tool on the MCP gateway and return the result as string."""
        async with self._client() as client:
            resp = await client.post(
                f"{self.mcp_url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments or {}},
                    "id": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        result = data.get("result", {})
        content_parts = result.get("content", [])

        # Concatenate text content
        texts = []
        for part in content_parts:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)

        return "\n".join(texts) if texts else json.dumps(result)

    async def register_tools(self, agent) -> int:
        """Register all MCP tools into an AitherAgent's tool registry.

        Returns the number of tools registered.
        """
        tools = await self.list_tools()
        count = 0

        for mcp_tool in tools:
            # Create a closure for each tool
            tool_name = mcp_tool.name

            async def _call(bridge=self, tn=tool_name, **kwargs) -> str:
                return await bridge.call_tool(tn, kwargs)

            _call.__name__ = tool_name
            _call.__doc__ = mcp_tool.description

            agent._tools.register(
                _call,
                name=tool_name,
                description=mcp_tool.description,
            )
            count += 1

        return count

    async def health(self) -> bool:
        """Check if the MCP gateway is reachable."""
        try:
            async with self._client() as client:
                resp = await client.get(f"{self.mcp_url}/health")
                return resp.status_code == 200
        except Exception:
            return False


async def connect_mcp(
    api_key: str = "",
    mcp_url: str = _DEFAULT_MCP_URL,
) -> MCPBridge:
    """Quick helper to create and verify an MCP bridge connection.

    Usage:
        bridge = await connect_mcp(api_key="your-key")
        tools = await bridge.list_tools()
    """
    bridge = MCPBridge(mcp_url=mcp_url, api_key=api_key)
    if await bridge.health():
        logger.info(f"Connected to MCP gateway at {mcp_url}")
    else:
        logger.warning(f"MCP gateway at {mcp_url} unreachable — tools will fail")
    return bridge
