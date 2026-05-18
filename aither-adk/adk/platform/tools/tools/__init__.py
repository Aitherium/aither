"""
AitherOS Common Agent Tools
============================

Shared tools for all Google ADK agents in the AitherOS ecosystem.

Tool Categories:
    - aither_tools: AitherZero automation integration
    - infrastructure_tools: System management
    - narrative_tools: Creative writing support
    - subagent_tools: Sub-agent spawning and research

Usage:
    from AitherOS.agents.common.tools import aither_tools, subagent_tools
    
    agent = Agent(
        name="MyAgent",
        model="gemini-2.5-flash",
        tools=aither_tools + subagent_tools,
    )
"""

# Aither automation tools
try:
    from .aither_tools import (
        aither_tools,
        execute_aither_script,
        list_automation_scripts,
        execute_aither_playbook,
    )
except ImportError:
    aither_tools = []

# Infrastructure tools
try:
    from .infrastructure_tools import infrastructure_tools
except ImportError:
    infrastructure_tools = []

# Narrative tools
try:
    from .narrative_tools import narrative_tools
except ImportError:
    narrative_tools = []

# Sub-agent/research tools
try:
    from .subagent_tools import (
        subagent_tools,
        spawn_scout,
        spawn_research_team,
        search_codebase,
        get_file_content,
        get_project_structure,
    )
except ImportError:
    subagent_tools = []

# MCP client tools
try:
    from .mcp_client import list_mcp_servers, list_mcp_tools, call_mcp_tool
except ImportError:
    pass

# Combined tool collections
all_aither_tools = aither_tools + infrastructure_tools + narrative_tools + subagent_tools

__all__ = [
    # Tool collections
    "aither_tools",
    "infrastructure_tools",
    "narrative_tools",
    "subagent_tools",
    "all_aither_tools",
    
    # Individual subagent tools
    "spawn_scout",
    "spawn_research_team",
    "search_codebase",
    "get_file_content",
    "get_project_structure",
    
    # Aither automation
    "execute_aither_script",
    "list_automation_scripts",
    "execute_aither_playbook",
    
    # MCP
    "list_mcp_servers",
    "list_mcp_tools",
    "call_mcp_tool",
]
