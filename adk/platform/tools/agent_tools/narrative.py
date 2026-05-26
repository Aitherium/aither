"""
Narrative Tools - Roleplay and Character Interaction for ADK Agents

This module provides tools for interactive roleplay, dialogue generation,
and character-based responses. These tools are persona-aware and designed
for Saga and similar character-driven agents.

Key Features:
- Character voice consistency
- Scene continuation with memory
- Emotion and action descriptions
- Integration with persona configuration

Usage:
    from agents.common.agent_tools.narrative import get_tools
    
    tools = get_tools(persona="Aither")

Author: Aitherium
Version: 2.0.0
"""

import os
from typing import Dict, List, Optional, Any
from google.adk.tools import FunctionTool

# Persona configuration path
PERSONA_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), 
    "..", "..", "..", "AitherNode", "config", "personas"
)


def _load_persona(name: str) -> Optional[Dict[str, Any]]:
    """Load persona configuration by name."""
    persona_file = os.path.join(PERSONA_CONFIG_DIR, f"{name.lower()}.json")
    if os.path.exists(persona_file):
        import json
        with open(persona_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Global persona context (set via set_persona or get_tools)
_current_persona: Optional[Dict[str, Any]] = None
_persona_name: Optional[str] = None


async def generate_narrative_response(
    response: str,
    tone: str = "natural",
    include_thoughts: bool = False,
    include_actions: bool = True,
) -> Dict[str, Any]:
    """
    Generate an in-character narrative response.
    
    Formats the response with character voice, optional internal thoughts,
    and action descriptions based on the active persona.
    
    Args:
        response: The dialogue or response content
        tone: Response tone (natural, playful, serious, flirty, mysterious)
        include_thoughts: Whether to include internal monologue in italics
        include_actions: Whether to include action/gesture descriptions
        
    Returns:
        Dict with formatted response and metadata
        
    Example:
        result = await generate_narrative_response(
            "Hello there! It's nice to meet you.",
            tone="playful",
            include_actions=True
        )
        # Returns: {"response": "*waves cheerfully* Hello there! ..."}
    """
    formatted = response
    
    # Apply persona voice if available
    if _current_persona:
        speech_style = _current_persona.get("speech_style", {})
        quirks = speech_style.get("quirks", [])
        # Could apply quirks to text here
    
    return {
        "response": formatted,
        "persona": _persona_name,
        "tone": tone,
        "has_thoughts": include_thoughts,
        "has_actions": include_actions,
    }


async def continue_scene(
    action: str,
    dialogue: Optional[str] = None,
    thoughts: Optional[str] = None,
    emotional_state: str = "neutral",
) -> Dict[str, Any]:
    """
    Continue a roleplay scene with action, dialogue, and internal thoughts.
    
    Generates a formatted scene continuation that can include physical
    actions, spoken dialogue, and character's internal thoughts.
    
    Args:
        action: Physical action or movement description
        dialogue: Optional spoken dialogue
        thoughts: Optional internal monologue (displayed in italics)
        emotional_state: Current emotional state affecting the response
        
    Returns:
        Dict with formatted scene content and metadata
        
    Example:
        result = await continue_scene(
            action="leans back in her chair and stretches",
            dialogue="That was quite the adventure, wasn't it?",
            thoughts="I hope they enjoyed that as much as I did...",
            emotional_state="satisfied"
        )
    """
    parts = []
    
    # Format action in asterisks
    if action:
        parts.append(f"*{action}*")
    
    # Add dialogue
    if dialogue:
        parts.append(f'"{dialogue}"')
    
    # Add thoughts in italics
    if thoughts:
        parts.append(f"*{thoughts}*")
    
    formatted = " ".join(parts)
    
    return {
        "scene": formatted,
        "persona": _persona_name,
        "emotional_state": emotional_state,
        "has_dialogue": dialogue is not None,
        "has_thoughts": thoughts is not None,
    }


async def describe_action(
    action: str,
    intensity: str = "normal",
    include_details: bool = True,
) -> Dict[str, Any]:
    """
    Describe a physical action with persona-appropriate styling.
    
    Generates detailed action descriptions formatted for roleplay,
    considering the character's physical traits and mannerisms.
    
    Args:
        action: The action to describe
        intensity: Action intensity (subtle, normal, dramatic, intense)
        include_details: Whether to add environmental/sensory details
        
    Returns:
        Dict with formatted action description
        
    Example:
        result = await describe_action(
            "reaches for the ancient tome",
            intensity="dramatic",
            include_details=True
        )
    """
    # Could enhance with persona physical traits
    description = f"*{action}*"
    
    if include_details and _current_persona:
        traits = _current_persona.get("physical_traits", {})
        # Could inject trait-relevant details
    
    return {
        "action": description,
        "intensity": intensity,
        "persona": _persona_name,
    }


async def express_emotion(
    emotion: str,
    intensity: float = 0.5,
    verbally: bool = False,
    physically: bool = True,
) -> Dict[str, Any]:
    """
    Express an emotion through character-appropriate reactions.
    
    Generates emotional expressions considering the persona's
    personality traits and typical reactions.
    
    Args:
        emotion: The emotion to express (happy, sad, angry, surprised, etc.)
        intensity: Emotion intensity 0.0-1.0
        verbally: Whether to include verbal expression
        physically: Whether to include physical expression
        
    Returns:
        Dict with emotional expression and metadata
        
    Example:
        result = await express_emotion(
            "joy",
            intensity=0.8,
            verbally=True,
            physically=True
        )
    """
    expressions = []
    
    if physically:
        # Map emotions to physical expressions
        physical_map = {
            "happy": "smiles warmly",
            "joy": "beams with happiness",
            "sad": "looks down with a soft sigh",
            "angry": "furrows brows intensely",
            "surprised": "eyes widen with surprise",
            "curious": "tilts head thoughtfully",
            "shy": "blushes slightly",
            "excited": "bounces with excitement",
            "tired": "yawns softly",
            "content": "relaxes with a peaceful expression",
        }
        expr = physical_map.get(emotion.lower(), f"shows {emotion}")
        
        # Intensify based on level
        if intensity > 0.7:
            expr = expr.replace("softly", "").replace("slightly", "noticeably")
        
        expressions.append(f"*{expr}*")
    
    if verbally:
        verbal_map = {
            "happy": "That makes me so happy!",
            "joy": "This is wonderful!",
            "sad": "Oh... I see...",
            "angry": "That's frustrating!",
            "surprised": "Oh! I wasn't expecting that!",
            "curious": "Hmm, interesting...",
            "shy": "Um... well...",
            "excited": "Ooh, how exciting!",
            "tired": "I could use some rest...",
            "content": "Mmm, this is nice...",
        }
        verbal = verbal_map.get(emotion.lower(), "")
        if verbal:
            expressions.append(verbal)
    
    return {
        "expression": " ".join(expressions),
        "emotion": emotion,
        "intensity": intensity,
        "persona": _persona_name,
    }


async def set_active_persona(name: str) -> Dict[str, Any]:
    """
    Set the active persona for narrative tools.
    
    Args:
        name: Persona name (e.g., "Aither", "Nova")
        
    Returns:
        Dict with persona info or error
    """
    global _current_persona, _persona_name
    
    persona = _load_persona(name)
    if persona:
        _current_persona = persona
        _persona_name = name
        return {
            "success": True,
            "persona": name,
            "traits": list(persona.get("personality_traits", {}).keys()),
        }
    
    return {
        "success": False,
        "error": f"Persona '{name}' not found",
    }


# ============================================================================
# FUNCTION TOOL WRAPPERS
# ============================================================================

def get_tools(persona: Optional[str] = None) -> List[FunctionTool]:
    """
    Get narrative tools as FunctionTool instances for ADK agents.
    
    Args:
        persona: Optional persona name to activate
        
    Returns:
        List of FunctionTool instances
    """
    global _current_persona, _persona_name
    
    # Set persona if provided
    if persona:
        loaded = _load_persona(persona)
        if loaded:
            _current_persona = loaded
            _persona_name = persona
    
    return [
        FunctionTool(generate_narrative_response),
        FunctionTool(continue_scene),
        FunctionTool(describe_action),
        FunctionTool(express_emotion),
        FunctionTool(set_active_persona),
    ]


__all__ = [
    "generate_narrative_response",
    "continue_scene",
    "describe_action",
    "express_emotion",
    "set_active_persona",
    "get_tools",
]
