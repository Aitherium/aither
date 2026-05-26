"""
Aither ADK - Communication Module
=================================

Inter-agent communication, messaging, and multi-agent coordination.

Includes:
- A2A Protocol: Google Agent-to-Agent protocol for agent interoperability
- Federation: Connect to external A2A-compatible agents
- Aeon/Demiurge clients: Direct access to core agents
- Group Chat: Multi-agent conversation management

Import directly from submodules:
    >>> from aither_adk.communication.mailbox import Mailbox
    >>> from aither_adk.communication.group_chat import GroupChatManager
    >>> from aither_adk.communication.aeon_client import AeonClient, aeon
    >>> from aither_adk.communication.demiurge_client import DemiurgeClient, demiurge
    >>> from aither_adk.communication.a2a_client import A2AClient, send_to_agent
    >>> from aither_adk.communication.a2a_federation import FederatedA2AClient, federate_task
"""

# Core agent clients - Aeon (formerly Council)
try:
    from .aeon_client import AeonClient, aeon, CouncilClient, council
except ImportError:
    AeonClient = None
    aeon = None
    CouncilClient = None
    council = None

try:
    from .demiurge_client import DemiurgeClient, demiurge
except ImportError:
    DemiurgeClient = None
    demiurge = None

# A2A Protocol clients
try:
    from .a2a_client import A2AClient, A2ATask, send_to_agent, discover_agents, ask_aeon, ask_council
except ImportError:
    A2AClient = None
    A2ATask = None
    send_to_agent = None
    discover_agents = None
    ask_aeon = None
    ask_council = None

try:
    from .a2a_federation import (
        FederatedA2AClient, ExternalAgentCard, FederationStatus,
        federate_task, discover_external_agent, quick_external_ask
    )
except ImportError:
    FederatedA2AClient = None
    ExternalAgentCard = None
    FederationStatus = None
    federate_task = None
    discover_external_agent = None
    quick_external_ask = None

__all__ = [
    # Core clients - Aeon (with backward compat aliases)
    "AeonClient",
    "aeon",
    "CouncilClient",  # Backward compat alias
    "council",  # Backward compat alias
    "DemiurgeClient",
    "demiurge",
    # A2A Protocol
    "A2AClient",
    "A2ATask",
    "send_to_agent",
    "discover_agents",
    "ask_aeon",
    "ask_council",  # Backward compat alias
    # A2A Federation
    "FederatedA2AClient",
    "ExternalAgentCard",
    "FederationStatus",
    "federate_task",
    "discover_external_agent",
    "quick_external_ask",
]
