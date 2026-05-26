"""
Agent Tools - Internal Tools for Google ADK Agents

==============================================================================
                      ADK AGENT TOOLS (THIS MODULE)
==============================================================================

This module provides tools specifically designed for Google ADK-based agents
like Saga, Atlas, and GenesisAgent.

These tools are:
- Session-aware: Access to ToolContext for state and artifacts
- Persona-integrated: Character personality injection for roleplay
- Enhancement-enabled: Full prompt boosting for image generation
- ADK-native: Return dicts, use FunctionTool wrappers

Architecture:
    AitherOS/agents/common/
    +-- agent_tools/           # <- THIS MODULE (ADK Agent Tools)
    |   +-- __init__.py        # Module exports & tool factory
    |   +-- narrative.py       # Roleplay/dialogue tools
    |   +-- generation.py      # Image gen with persona/enhancements
    |   +-- state.py           # Session state & persona management
    |   +-- mcp_bridge.py      # Call MCP tools from agents
    |   +-- common.py          # Shared utilities
    |
    +-- tools/                 # LEGACY - being migrated here
        +-- aither_tools.py    # -> mcp_bridge.py
        +-- fal_tools.py       # -> generation.py
        +-- narrative_tools.py # -> narrative.py
        +-- state_manager.py   # -> state.py

==============================================================================
                        MCP TOOLS (SEPARATE MODULE)
==============================================================================

For universal MCP tools usable by ANY client (Copilot, Claude, etc.),
see the MCP tools module:

    AitherOS/AitherNode/aither_tools/

Key Differences:
+---------------------+-------------------------+-------------------------+
| Aspect              | Agent Tools (this)      | MCP Tools (aither_tools)|
+---------------------+-------------------------+-------------------------+
| Consumers           | ADK agents only         | Any MCP client          |
| State               | Session-aware           | Stateless               |
| Persona             | Character/persona aware | None                    |
| Prompt Enhancement  | Full NSFW/style boosts  | Minimal/safety-only     |
| Return Format       | Dict for ADK            | JSON strings            |
| Error Handling      | Raise exceptions        | Return error strings    |
| Tool Context        | ToolContext from ADK    | None                    |
| Wrapper             | FunctionTool            | @mcp.tool() decorator   |
+---------------------+-------------------------+-------------------------+

Usage:
    from agents.common.agent_tools import get_agent_tools, TOOL_SETS
    
    # Get all tools for a narrative agent
    tools = get_agent_tools(
        persona="Aither",
        tool_sets=["narrative", "generation", "memory"]
    )
    
    # Or use predefined agent profiles
    from agents.common.agent_tools import AGENT_PROFILES
    tools = get_agent_tools(profile="narrative_agent")

Author: Aitherium
Version: 2.0.0
"""

__version__ = "2.0.0"

from typing import List, Optional
from google.adk.tools import FunctionTool

# ============================================================================
# TOOL SET DEFINITIONS
# ============================================================================

TOOL_SETS = {
    "narrative": {
        "description": "Roleplay and narrative response tools",
        "module": "narrative",
        "tools": [
            "generate_narrative_response",
            "continue_scene",
            "describe_action",
            "express_emotion",
        ],
    },
    "generation": {
        "description": "Image generation with persona injection",
        "module": "generation", 
        "tools": [
            "generate_image",
            "refine_image",
            "generate_selfie",
            "generate_scene",
        ],
    },
    "state": {
        "description": "Session state and persona management",
        "module": "state",
        "tools": [
            "set_persona",
            "get_persona",
            "update_session_state",
            "get_session_state",
        ],
    },
    "mcp_bridge": {
        "description": "Bridge to call MCP tools from agents",
        "module": "mcp_bridge",
        "tools": [
            "call_mcp_tool",
            "remember",
            "recall",
            "run_script",
        ],
    },
}

# Predefined agent profiles with recommended tool sets
AGENT_PROFILES = {
    "narrative_agent": {
        "description": "Interactive roleplay and storytelling",
        "tool_sets": ["narrative", "generation", "state", "mcp_bridge"],
        "default_persona": "Aither",
    },
    "infrastructure_agent": {
        "description": "System automation and DevOps",
        "tool_sets": ["mcp_bridge"],
        "default_persona": None,
    },
    "genesis_agent": {
        "description": "Testing and validation",
        "tool_sets": ["mcp_bridge"],
        "default_persona": None,
    },
}


def get_agent_tools(
    tool_sets: Optional[List[str]] = None,
    profile: Optional[str] = None,
    persona: Optional[str] = None,
    include_mcp_bridge: bool = True,
) -> List[FunctionTool]:
    """
    Get ADK FunctionTool instances for an agent.
    
    Args:
        tool_sets: List of tool set names to include (e.g., ["narrative", "generation"])
        profile: Use a predefined agent profile (overrides tool_sets)
        persona: Persona name to configure for generation tools
        include_mcp_bridge: Always include MCP bridge tools (default True)
        
    Returns:
        List of FunctionTool instances ready for ADK agent
        
    Example:
        # Using tool sets
        tools = get_agent_tools(
            tool_sets=["narrative", "generation"],
            persona="Aither"
        )
        
        # Using profile
        tools = get_agent_tools(profile="narrative_agent")
    """
    tools = []
    
    # Resolve profile to tool sets
    if profile and profile in AGENT_PROFILES:
        profile_config = AGENT_PROFILES[profile]
        tool_sets = profile_config["tool_sets"]
        if persona is None:
            persona = profile_config.get("default_persona")
    
    # Default to all tool sets
    if tool_sets is None:
        tool_sets = list(TOOL_SETS.keys())
    
    # Ensure MCP bridge is included
    if include_mcp_bridge and "mcp_bridge" not in tool_sets:
        tool_sets.append("mcp_bridge")
    
    # Import and collect tools from each set
    for set_name in tool_sets:
        if set_name not in TOOL_SETS:
            continue
            
        set_config = TOOL_SETS[set_name]
        module_name = set_config["module"]
        
        try:
            # Dynamic import of tool module
            if module_name == "narrative":
                from . import narrative as tool_module
            elif module_name == "generation":
                from . import generation as tool_module
            elif module_name == "state":
                from . import state as tool_module
            elif module_name == "mcp_bridge":
                from . import mcp_bridge as tool_module
            else:
                continue
            
            # Get tools from module
            if hasattr(tool_module, "get_tools"):
                module_tools = tool_module.get_tools(persona=persona)
                tools.extend(module_tools)
                
        except ImportError as e:
            # Module not yet implemented - skip gracefully
            print(f"[agent_tools] Skipping {set_name}: {e}")
            continue
    
    return tools


def list_tool_sets() -> dict:
    """List available tool sets with descriptions."""
    return {
        name: config["description"] 
        for name, config in TOOL_SETS.items()
    }


def list_profiles() -> dict:
    """List available agent profiles."""
    return {
        name: config["description"]
        for name, config in AGENT_PROFILES.items()
    }


# ============================================================================
# TOOL DOCUMENTATION FOR AGENT INSTRUCTIONS
# ============================================================================

AGENT_TOOL_GUIDE = """
## Agent-Specific Tools

### [MASK] Narrative Tools
Tools for roleplay and character interaction:
- `generate_narrative_response(response)` - Generate in-character response
- `continue_scene(action, dialogue?, thoughts?)` - Continue roleplay scene
- `describe_action(action)` - Describe physical actions
- `express_emotion(emotion, intensity?)` - Express character emotions

###  Generation Tools (with Persona)
Image generation with automatic character injection:
- `generate_image(prompt, style?)` - Generate with persona awareness
- `generate_selfie(pose?, outfit?, location?)` - Generate character selfie
- `generate_scene(description, characters?)` - Generate scene with characters
- `refine_image(image_path, changes)` - Modify existing image

### [TARGET] State Tools
Session and persona management:
- `set_persona(name)` - Set active persona for generation
- `get_persona()` - Get current persona info
- `update_session_state(key, value)` - Store session data
- `get_session_state(key)` - Retrieve session data

###  MCP Bridge Tools  
Access universal MCP tools from within agent:
- `call_mcp_tool(tool_name, args)` - Call any MCP tool
- `remember(content, category?)` - Save to long-term memory (via MCP)
- `recall(query)` - Search memories (via MCP)
- `run_script(script_number)` - Run automation script (via MCP)

## Key Differences from MCP Tools

1. **Persona Injection**: Generation tools automatically inject character 
   description from the active persona configuration.

2. **Session State**: Tools can access and modify session state, enabling
   context-aware responses across conversation turns.

3. **Enhancement Pipeline**: Image generation goes through full prompt
   enhancement including style boosts, quality tags, and NSFW handling.

4. **ADK Integration**: Tools receive ToolContext and can save artifacts,
   access session state, and interact with the ADK runtime.
"""


def get_tool_guide() -> str:
    """Get the agent tool guide for system prompts."""
    return AGENT_TOOL_GUIDE
