"""AitherOS Alpha — Build AI agent fleets with any LLM backend."""

__version__ = "0.1.0a1"

from adk.agent import AitherAgent
from adk.tools import tool, ToolRegistry
from adk.llm import LLMRouter
from adk.config import Config

__all__ = [
    "AitherAgent",
    "tool",
    "ToolRegistry",
    "LLMRouter",
    "Config",
    "AgentRegistry",
    "AgentForge",
    "FleetConfig",
    "ConversationStore",
    # Extended capabilities
    "LoopGuard",
    "AitherSandbox",
    "AgentMeter",
    "SkillManifest",
    # Observability
    "ChronicleClient",
    "WatchReporter",
    "MetricsCollector",
    "PulseClient",
    # Core infrastructure
    "ServiceBridge",
    "EventEmitter",
    "IntakeGuard",
    "ContextManager",
    "register_builtin_tools",
]


def connect_mcp(api_key: str = "", mcp_url: str = "https://mcp.aitherium.com"):
    """Quick access to the MCP bridge. Returns a coroutine."""
    from adk.mcp import connect_mcp as _connect
    return _connect(api_key=api_key, mcp_url=mcp_url)


def connect_federation(host: str = "http://localhost", tenant: str = "public"):
    """Quick access to the federation client for connecting to Elysium."""
    from adk.federation import FederationClient
    return FederationClient(host=host, tenant=tenant)


def auto_setup(**kwargs):
    """Quick access to agent self-setup. Returns a coroutine."""
    from adk.setup import auto_setup as _auto_setup
    return _auto_setup(**kwargs)


# Lazy imports for heavier modules
def __getattr__(name):
    if name == "AgentRegistry":
        from adk.registry import AgentRegistry
        return AgentRegistry
    if name == "AgentForge":
        from adk.forge import AgentForge
        return AgentForge
    if name == "FleetConfig":
        from adk.fleet import FleetConfig
        return FleetConfig
    if name == "ConversationStore":
        from adk.conversations import ConversationStore
        return ConversationStore
    if name == "LoopGuard":
        from adk.loop_guard import LoopGuard
        return LoopGuard
    if name == "AitherSandbox":
        from adk.sandbox import AitherSandbox
        return AitherSandbox
    if name == "AgentMeter":
        from adk.metering import AgentMeter
        return AgentMeter
    if name == "SkillManifest":
        from adk.identity import SkillManifest
        return SkillManifest
    if name == "ChronicleClient":
        from adk.chronicle import ChronicleClient
        return ChronicleClient
    if name == "WatchReporter":
        from adk.watch import WatchReporter
        return WatchReporter
    if name == "MetricsCollector":
        from adk.metrics import MetricsCollector
        return MetricsCollector
    if name == "PulseClient":
        from adk.pulse import PulseClient
        return PulseClient
    if name == "ServiceBridge":
        from adk.services import ServiceBridge
        return ServiceBridge
    if name == "EventEmitter":
        from adk.events import EventEmitter
        return EventEmitter
    if name == "IntakeGuard":
        from adk.safety import IntakeGuard
        return IntakeGuard
    if name == "ContextManager":
        from adk.context import ContextManager
        return ContextManager
    if name == "register_builtin_tools":
        from adk.builtin_tools import register_builtin_tools
        return register_builtin_tools
    raise AttributeError(f"module 'adk' has no attribute {name!r}")
