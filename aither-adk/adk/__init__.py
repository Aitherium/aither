"""AitherOS Alpha — Build AI agent fleets with any LLM backend."""

__version__ = "0.15.0"

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
    "GraphMemory",
    "NanoGPT",
    "NeuronPool",
    "AutoNeuronFire",
    "DegenerationDetector",
    "strip_internal_tags",
    "CATEGORY_TOOLS",
    # Elysium cloud
    "Elysium",
    # Faculty graphs (local knowledge)
    "CodeGraph",
    "MemoryGraph",
    "EmbeddingProvider",
    # Mesh relay
    "AitherNetRelay",
    # Chat + Mail
    "ChatRelay",
    "MailRelay",
    # MCP client + server
    "MCPAuth",
    "MCPBridge",
    "MCPServer",
    "MCPError",
    "MCPAuthError",
    "MCPBalanceError",
    # Multi-agent group chat
    "AeonSession",
    "AeonResponse",
    "AeonMessage",
    "group_chat",
    "AEON_PRESETS",
    # A2A protocol
    "A2AServer",
    # Unified storage
    "Strata",
    "StrataBackend",
    "LocalBackend",
    # Cross-platform pairing
    "PairingManager",
    "PairingResult",
    "PlatformIdentity",
    # Voice
    "VoiceClient",
    "TranscriptionResult",
    "SynthesisResult",
    "EmotionResult",
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
    if name == "GraphMemory":
        from adk.graph_memory import GraphMemory
        return GraphMemory
    if name == "NanoGPT":
        from adk.nanogpt import NanoGPT
        return NanoGPT
    if name == "NeuronPool":
        from adk.neurons import NeuronPool
        return NeuronPool
    if name == "AutoNeuronFire":
        from adk.neurons import AutoNeuronFire
        return AutoNeuronFire
    if name == "DegenerationDetector":
        from adk.llm.base import DegenerationDetector
        return DegenerationDetector
    if name == "strip_internal_tags":
        from adk.llm.base import strip_internal_tags
        return strip_internal_tags
    if name == "CATEGORY_TOOLS":
        from adk.neurons import CATEGORY_TOOLS
        return CATEGORY_TOOLS
    if name == "CodeGraph":
        from adk.faculties.code_graph import CodeGraph
        return CodeGraph
    if name == "MemoryGraph":
        from adk.faculties.memory_graph import MemoryGraph
        return MemoryGraph
    if name == "EmbeddingProvider":
        from adk.faculties.embeddings import EmbeddingProvider
        return EmbeddingProvider
    if name == "Elysium":
        from adk.elysium import Elysium
        return Elysium
    if name == "AitherNetRelay":
        from adk.relay import AitherNetRelay
        return AitherNetRelay
    if name == "ChatRelay":
        from adk.chat import ChatRelay
        return ChatRelay
    if name == "MailRelay":
        from adk.smtp import MailRelay
        return MailRelay
    if name == "MCPServer":
        from adk.mcp_server import MCPServer
        return MCPServer
    if name == "MCPAuth":
        from adk.mcp import MCPAuth
        return MCPAuth
    if name == "MCPBridge":
        from adk.mcp import MCPBridge
        return MCPBridge
    if name == "MCPError":
        from adk.mcp import MCPError
        return MCPError
    if name == "MCPAuthError":
        from adk.mcp import MCPAuthError
        return MCPAuthError
    if name == "MCPBalanceError":
        from adk.mcp import MCPBalanceError
        return MCPBalanceError
    if name == "AeonSession":
        from adk.aeon import AeonSession
        return AeonSession
    if name == "AeonResponse":
        from adk.aeon import AeonResponse
        return AeonResponse
    if name == "AeonMessage":
        from adk.aeon import AeonMessage
        return AeonMessage
    if name == "group_chat":
        from adk.aeon import group_chat
        return group_chat
    if name == "AEON_PRESETS":
        from adk.aeon import AEON_PRESETS
        return AEON_PRESETS
    if name == "A2AServer":
        from adk.a2a import A2AServer
        return A2AServer
    if name == "Strata":
        from adk.strata import Strata
        return Strata
    if name == "StrataBackend":
        from adk.strata import StrataBackend
        return StrataBackend
    if name == "LocalBackend":
        from adk.strata import LocalBackend
        return LocalBackend
    if name == "PairingManager":
        from adk.pairing import PairingManager
        return PairingManager
    if name == "PairingResult":
        from adk.pairing import PairingResult
        return PairingResult
    if name == "PlatformIdentity":
        from adk.pairing import PlatformIdentity
        return PlatformIdentity
    if name == "VoiceClient":
        from adk.voice import VoiceClient
        return VoiceClient
    if name == "TranscriptionResult":
        from adk.voice import TranscriptionResult
        return TranscriptionResult
    if name == "SynthesisResult":
        from adk.voice import SynthesisResult
        return SynthesisResult
    if name == "EmotionResult":
        from adk.voice import EmotionResult
        return EmotionResult
    raise AttributeError(f"module 'adk' has no attribute {name!r}")
