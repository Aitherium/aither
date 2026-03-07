"""AitherOS Agent Development Kit — Build AI agents with any LLM backend."""

__version__ = "0.1.0a1"

from adk.agent import AitherAgent
from adk.tools import tool, ToolRegistry
from adk.llm import LLMRouter
from adk.config import Config

__all__ = ["AitherAgent", "tool", "ToolRegistry", "LLMRouter", "Config"]


def connect_mcp(api_key: str = "", mcp_url: str = "https://mcp.aitherium.com"):
    """Quick access to the MCP bridge. Returns a coroutine."""
    from adk.mcp import connect_mcp as _connect
    return _connect(api_key=api_key, mcp_url=mcp_url)


def connect_federation(host: str = "http://localhost", tenant: str = "public"):
    """Quick access to the federation client for connecting to AitherOS."""
    from adk.federation import FederationClient
    return FederationClient(host=host, tenant=tenant)


def auto_setup(**kwargs):
    """Quick access to agent self-setup. Returns a coroutine."""
    from adk.setup import auto_setup as _auto_setup
    return _auto_setup(**kwargs)
