"""
State Tools - Session State and Persona Management for ADK Agents

This module provides tools for managing session state, persona context,
and agent memory within a conversation. State is ephemeral (session-only)
while long-term persistence uses the memory MCP tools.

Architecture:
    Session State (this module)
    +-- Per-conversation context
    +-- Active persona configuration  
    +-- Conversation history tracking
    +-- Temporary artifacts
    
    Long-term Memory (via mcp_bridge)
    +-- Persistent across sessions
    +-- Semantic search/retrieval
    +-- Importance scoring

Usage:
    from agents.common.agent_tools.state import get_tools
    
    tools = get_tools(persona="Aither")

Author: Aitherium
Version: 2.0.0
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from google.adk.tools import FunctionTool

# Paths
PERSONA_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "AitherNode", "config", "personas"
)

# ============================================================================
# SESSION STATE STORAGE
# ============================================================================

# In-memory session state (reset on agent restart)
_session_state: Dict[str, Any] = {}
_persona_state: Dict[str, Any] = {
    "active_persona": None,
    "persona_config": None,
    "mood": "neutral",
    "energy": 0.7,
}
_conversation_state: Dict[str, Any] = {
    "turn_count": 0,
    "started_at": None,
    "topics": [],
    "relationship_level": "stranger",
}


def _load_persona(name: str) -> Optional[Dict[str, Any]]:
    """Load persona configuration by name."""
    persona_file = os.path.join(PERSONA_CONFIG_DIR, f"{name.lower()}.json")
    if os.path.exists(persona_file):
        with open(persona_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ============================================================================
# PERSONA MANAGEMENT
# ============================================================================

async def set_persona(
    name: str,
    mood: str = "neutral",
    energy: float = 0.7,
) -> Dict[str, Any]:
    """
    Set the active persona for the session.
    
    Loads the persona configuration and sets initial mood/energy state.
    This affects all persona-aware tools (generation, narrative, etc.)
    
    Args:
        name: Persona name (e.g., "Aither", "Nova")
        mood: Initial mood state
        energy: Initial energy level 0.0-1.0
        
    Returns:
        Dict with persona info or error
        
    Example:
        await set_persona("Aither", mood="cheerful", energy=0.9)
    """
    global _persona_state
    
    persona = _load_persona(name)
    if not persona:
        return {
            "success": False,
            "error": f"Persona '{name}' not found",
            "available": _list_available_personas(),
        }
    
    _persona_state = {
        "active_persona": name,
        "persona_config": persona,
        "mood": mood,
        "energy": energy,
    }
    
    # Reset conversation for new persona
    _conversation_state["turn_count"] = 0
    _conversation_state["started_at"] = datetime.now().isoformat()
    
    return {
        "success": True,
        "persona": name,
        "mood": mood,
        "energy": energy,
        "personality_traits": list(persona.get("personality_traits", {}).keys()),
        "greeting": persona.get("greeting", f"Hello! I'm {name}."),
    }


async def get_persona() -> Dict[str, Any]:
    """
    Get the current active persona and its state.
    
    Returns:
        Dict with persona name, config, mood, energy, and conversation context
    """
    if not _persona_state.get("active_persona"):
        return {
            "active": False,
            "message": "No persona is currently active",
        }
    
    return {
        "active": True,
        "persona": _persona_state["active_persona"],
        "mood": _persona_state["mood"],
        "energy": _persona_state["energy"],
        "turn_count": _conversation_state["turn_count"],
        "relationship": _conversation_state["relationship_level"],
        "topics_discussed": _conversation_state["topics"],
    }


async def update_persona_state(
    mood: Optional[str] = None,
    energy: Optional[float] = None,
    relationship: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update the persona's current emotional/social state.
    
    Used to track mood changes, energy levels, and relationship progression
    throughout the conversation.
    
    Args:
        mood: New mood state (happy, sad, curious, flirty, etc.)
        energy: New energy level 0.0-1.0
        relationship: Relationship level (stranger, acquaintance, friend, close_friend)
        
    Returns:
        Dict with updated state
    """
    if mood:
        _persona_state["mood"] = mood
    
    if energy is not None:
        _persona_state["energy"] = max(0.0, min(1.0, energy))
    
    if relationship:
        _conversation_state["relationship_level"] = relationship
    
    return {
        "persona": _persona_state.get("active_persona"),
        "mood": _persona_state["mood"],
        "energy": _persona_state["energy"],
        "relationship": _conversation_state["relationship_level"],
    }


def _list_available_personas() -> List[str]:
    """List available persona configurations."""
    if not os.path.exists(PERSONA_CONFIG_DIR):
        return []
    
    personas = []
    for file in os.listdir(PERSONA_CONFIG_DIR):
        if file.endswith(".json"):
            personas.append(file[:-5])  # Remove .json
    return personas


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

async def update_session_state(
    key: str,
    value: Any,
    merge: bool = False,
) -> Dict[str, Any]:
    """
    Store data in session state.
    
    Session state is ephemeral - it only lasts for the current conversation.
    For persistent storage, use the remember() function from mcp_bridge.
    
    Args:
        key: State key
        value: Value to store (any JSON-serializable type)
        merge: If True and value is dict, merge with existing dict
        
    Returns:
        Dict with success status and current value
        
    Example:
        await update_session_state("user_preferences", {"theme": "dark"})
    """
    global _session_state
    
    if merge and isinstance(value, dict) and key in _session_state:
        if isinstance(_session_state[key], dict):
            _session_state[key].update(value)
        else:
            _session_state[key] = value
    else:
        _session_state[key] = value
    
    return {
        "success": True,
        "key": key,
        "value": _session_state[key],
    }


async def get_session_state(
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve data from session state.
    
    Args:
        key: Specific key to retrieve (returns all if None)
        
    Returns:
        Dict with state value(s)
    """
    if key:
        if key in _session_state:
            return {
                "found": True,
                "key": key,
                "value": _session_state[key],
            }
        return {
            "found": False,
            "key": key,
            "message": f"Key '{key}' not found in session state",
        }
    
    return {
        "state": _session_state.copy(),
        "key_count": len(_session_state),
    }


async def clear_session_state(
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Clear session state.
    
    Args:
        keys: Specific keys to clear (clears all if None)
        
    Returns:
        Dict with cleared keys
    """
    global _session_state
    
    if keys:
        cleared = []
        for key in keys:
            if key in _session_state:
                del _session_state[key]
                cleared.append(key)
        return {
            "cleared": cleared,
            "remaining_keys": len(_session_state),
        }
    
    _session_state = {}
    return {
        "cleared": "all",
        "remaining_keys": 0,
    }


# ============================================================================
# CONVERSATION TRACKING
# ============================================================================

async def record_conversation_turn(
    topic: Optional[str] = None,
    sentiment: str = "neutral",
) -> Dict[str, Any]:
    """
    Record a conversation turn for context tracking.
    
    Called automatically by agents to track conversation progress.
    
    Args:
        topic: Optional topic being discussed
        sentiment: Sentiment of the turn (positive, neutral, negative)
        
    Returns:
        Dict with updated conversation state
    """
    global _conversation_state
    
    _conversation_state["turn_count"] += 1
    
    if topic and topic not in _conversation_state["topics"]:
        _conversation_state["topics"].append(topic)
        # Keep only last 10 topics
        _conversation_state["topics"] = _conversation_state["topics"][-10:]
    
    # Auto-update relationship based on turns
    turns = _conversation_state["turn_count"]
    if turns > 20 and _conversation_state["relationship_level"] == "acquaintance":
        _conversation_state["relationship_level"] = "friend"
    elif turns > 5 and _conversation_state["relationship_level"] == "stranger":
        _conversation_state["relationship_level"] = "acquaintance"
    
    return {
        "turn": _conversation_state["turn_count"],
        "relationship": _conversation_state["relationship_level"],
        "topics": _conversation_state["topics"],
    }


async def get_conversation_context() -> Dict[str, Any]:
    """
    Get full conversation context for decision making.
    
    Returns comprehensive context including persona state, session state,
    and conversation history summary.
    
    Returns:
        Dict with full context
    """
    return {
        "persona": {
            "name": _persona_state.get("active_persona"),
            "mood": _persona_state.get("mood"),
            "energy": _persona_state.get("energy"),
        },
        "conversation": {
            "turns": _conversation_state.get("turn_count", 0),
            "started": _conversation_state.get("started_at"),
            "relationship": _conversation_state.get("relationship_level"),
            "topics": _conversation_state.get("topics", []),
        },
        "session_keys": list(_session_state.keys()),
    }


# ============================================================================
# FUNCTION TOOL WRAPPERS
# ============================================================================

def get_tools(persona: Optional[str] = None) -> List[FunctionTool]:
    """
    Get state management tools as FunctionTool instances for ADK agents.
    
    Args:
        persona: Optional persona name to activate on initialization
        
    Returns:
        List of FunctionTool instances
    """
    # Note: We don't auto-set persona here since set_persona is async
    # The agent should call set_persona explicitly
    
    return [
        FunctionTool(set_persona),
        FunctionTool(get_persona),
        FunctionTool(update_persona_state),
        FunctionTool(update_session_state),
        FunctionTool(get_session_state),
        FunctionTool(clear_session_state),
        FunctionTool(record_conversation_turn),
        FunctionTool(get_conversation_context),
    ]


__all__ = [
    "set_persona",
    "get_persona",
    "update_persona_state",
    "update_session_state",
    "get_session_state",
    "clear_session_state",
    "record_conversation_turn",
    "get_conversation_context",
    "get_tools",
]
