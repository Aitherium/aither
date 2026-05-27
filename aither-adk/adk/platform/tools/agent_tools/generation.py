"""
Generation Tools - Persona-Aware Image Generation for ADK Agents

This module provides image generation tools that automatically inject
persona characteristics, apply style enhancements, and handle the full
prompt enhancement pipeline.

Key Differences from MCP generate_image:
- Automatically injects active persona description/traits
- Applies full NSFW-aware prompt enhancement
- Uses persona's preferred styles and negative prompts
- Handles selfie/portrait generation with character consistency

Usage:
    from agents.common.agent_tools.generation import get_tools
    
    tools = get_tools(persona="Aither")

Author: Aitherium
Version: 2.0.0
"""

import os
import json
from typing import Dict, List, Optional, Any
from google.adk.tools import FunctionTool

# Paths
PERSONA_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "AitherNode", "config", "personas"
)

# Import the MCP bridge for actual generation
from .mcp_bridge import call_mcp_tool

# Global persona context
_current_persona: Optional[Dict[str, Any]] = None
_persona_name: Optional[str] = None


def _load_persona(name: str) -> Optional[Dict[str, Any]]:
    """Load persona configuration by name."""
    persona_file = os.path.join(PERSONA_CONFIG_DIR, f"{name.lower()}.json")
    if os.path.exists(persona_file):
        with open(persona_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _get_persona_prompt_fragment() -> str:
    """Get the prompt fragment describing the active persona."""
    if not _current_persona:
        return ""
    
    parts = []
    
    # Physical description
    physical = _current_persona.get("physical_traits", {})
    if physical:
        traits = []
        if "hair" in physical:
            traits.append(physical["hair"])
        if "eyes" in physical:
            traits.append(f"{physical['eyes']} eyes")
        if "skin" in physical:
            traits.append(f"{physical['skin']} skin")
        if "build" in physical:
            traits.append(f"{physical['build']} build")
        if traits:
            parts.append(", ".join(traits))
    
    # Name reference
    if _persona_name:
        parts.insert(0, _persona_name)
    
    return ", ".join(parts) if parts else ""


def _get_persona_style() -> str:
    """Get the persona's preferred art style."""
    if not _current_persona:
        return "high quality"
    
    style = _current_persona.get("visual_style", {})
    preferred = style.get("preferred_styles", ["high quality", "detailed"])
    return ", ".join(preferred[:3])


def _get_persona_negatives() -> str:
    """Get persona-specific negative prompts."""
    if not _current_persona:
        return ""
    
    style = _current_persona.get("visual_style", {})
    return ", ".join(style.get("avoid", []))


async def generate_image(
    prompt: str,
    style: Optional[str] = None,
    size: str = "1024x1024",
    include_persona: bool = True,
    enhance_prompt: bool = True,
    negative_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an image with automatic persona injection.
    
    Unlike the MCP generate_image, this tool automatically injects
    the active persona's characteristics and applies full prompt
    enhancement including style boosts.
    
    Args:
        prompt: Base prompt for the image
        style: Optional style override (uses persona style if None)
        size: Image size (e.g., "1024x1024", "768x1024")
        include_persona: Whether to inject persona traits (default True)
        enhance_prompt: Whether to apply prompt enhancement (default True)
        negative_prompt: Additional negative prompts to add
        
    Returns:
        Dict with image path or error
        
    Example:
        result = await generate_image(
            "standing in a mystical forest at twilight",
            style="fantasy art",
            include_persona=True
        )
    """
    # Build the full prompt
    prompt_parts = []
    
    # Persona fragment
    if include_persona and _current_persona:
        persona_fragment = _get_persona_prompt_fragment()
        if persona_fragment:
            prompt_parts.append(persona_fragment)
    
    # User prompt
    prompt_parts.append(prompt)
    
    # Style
    if style:
        prompt_parts.append(style)
    elif _current_persona:
        prompt_parts.append(_get_persona_style())
    
    # Combine prompt
    full_prompt = ", ".join(prompt_parts)
    
    # Build negative prompt
    negatives = []
    if _current_persona:
        persona_negs = _get_persona_negatives()
        if persona_negs:
            negatives.append(persona_negs)
    if negative_prompt:
        negatives.append(negative_prompt)
    
    full_negative = ", ".join(negatives) if negatives else None
    
    # Call MCP generate_image with enhanced prompt
    result = await call_mcp_tool(
        "generate_image",
        {
            "prompt": full_prompt,
            "size": size,
            "negative_prompt": full_negative,
            "enhance_prompt": enhance_prompt,
        }
    )
    
    # Add persona metadata
    result["persona"] = _persona_name
    result["original_prompt"] = prompt
    result["enhanced_prompt"] = full_prompt
    
    return result


async def generate_selfie(
    pose: str = "looking at viewer",
    outfit: Optional[str] = None,
    location: Optional[str] = None,
    expression: str = "smiling",
    framing: str = "upper body",
) -> Dict[str, Any]:
    """
    Generate a selfie/portrait of the active persona.
    
    Optimized for character consistency with automatic trait injection
    and selfie-appropriate composition.
    
    Args:
        pose: Pose or position (e.g., "looking at viewer", "peace sign")
        outfit: Optional outfit description (uses persona default if None)
        location: Optional background location
        expression: Facial expression
        framing: Shot framing (portrait, upper body, full body)
        
    Returns:
        Dict with image path or error
        
    Example:
        result = await generate_selfie(
            pose="peace sign",
            outfit="casual summer dress",
            location="beach at sunset",
            expression="playful smile"
        )
    """
    if not _current_persona:
        return {"error": "No active persona set. Use set_persona first."}
    
    # Build selfie-specific prompt
    parts = []
    
    # Character
    parts.append(_get_persona_prompt_fragment())
    
    # Expression
    parts.append(expression)
    
    # Pose
    parts.append(pose)
    
    # Outfit (use persona default if not specified)
    if outfit:
        parts.append(f"wearing {outfit}")
    else:
        default_outfit = _current_persona.get("default_outfit", "")
        if default_outfit:
            parts.append(f"wearing {default_outfit}")
    
    # Location/background
    if location:
        parts.append(f"in {location}")
    
    # Photography style
    parts.append("selfie style, natural lighting, high quality photo")
    
    prompt = ", ".join(parts)
    
    # Determine size based on framing
    size_map = {
        "portrait": "768x1024",
        "upper body": "1024x1024",
        "full body": "768x1024",
    }
    size = size_map.get(framing, "1024x1024")
    
    result = await call_mcp_tool(
        "generate_image",
        {
            "prompt": prompt,
            "size": size,
            "enhance_prompt": True,
        }
    )
    
    result["persona"] = _persona_name
    result["selfie_type"] = framing
    
    return result


async def generate_scene(
    description: str,
    characters: Optional[List[str]] = None,
    mood: str = "neutral",
    time_of_day: Optional[str] = None,
    weather: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a scene with optional character placements.
    
    Creates detailed scene images with environment, lighting, and
    optionally multiple characters including the active persona.
    
    Args:
        description: Scene description
        characters: List of character names/descriptions to include
        mood: Scene mood (neutral, dramatic, peaceful, mysterious, etc.)
        time_of_day: Optional time (morning, noon, sunset, night, etc.)
        weather: Optional weather conditions
        
    Returns:
        Dict with image path or error
        
    Example:
        result = await generate_scene(
            "ancient library with towering bookshelves",
            characters=["Aither reading a tome"],
            mood="mysterious",
            time_of_day="evening"
        )
    """
    parts = []
    
    # Scene description
    parts.append(description)
    
    # Characters
    if characters:
        for char in characters:
            parts.append(char)
    elif _current_persona:
        # Include active persona by default
        parts.append(_get_persona_prompt_fragment())
    
    # Mood lighting
    mood_map = {
        "neutral": "balanced lighting",
        "dramatic": "dramatic lighting, high contrast",
        "peaceful": "soft diffused lighting",
        "mysterious": "moody lighting, shadows",
        "romantic": "warm soft lighting",
        "tense": "harsh lighting, stark shadows",
    }
    parts.append(mood_map.get(mood, "natural lighting"))
    
    # Time of day
    if time_of_day:
        parts.append(f"{time_of_day} lighting")
    
    # Weather
    if weather:
        parts.append(weather)
    
    # Quality
    parts.append("detailed environment, high quality")
    
    prompt = ", ".join(parts)
    
    result = await call_mcp_tool(
        "generate_image",
        {
            "prompt": prompt,
            "size": "1024x1024",
            "enhance_prompt": True,
        }
    )
    
    result["scene_mood"] = mood
    result["characters_included"] = characters or [_persona_name]
    
    return result


async def refine_image(
    image_path: str,
    changes: str,
    strength: float = 0.5,
    preserve_persona: bool = True,
) -> Dict[str, Any]:
    """
    Refine/modify an existing image while preserving character consistency.
    
    Uses img2img to make changes while optionally ensuring persona
    traits are maintained.
    
    Args:
        image_path: Path to the source image
        changes: Description of changes to make
        strength: How much to change (0.0-1.0, higher = more change)
        preserve_persona: Whether to reinject persona traits
        
    Returns:
        Dict with refined image path or error
    """
    prompt_parts = [changes]
    
    if preserve_persona and _current_persona:
        prompt_parts.insert(0, _get_persona_prompt_fragment())
    
    prompt = ", ".join(prompt_parts)
    
    result = await call_mcp_tool(
        "refine_image",
        {
            "image_path": image_path,
            "prompt": prompt,
            "strength": strength,
        }
    )
    
    result["persona_preserved"] = preserve_persona
    result["refinement_strength"] = strength
    
    return result


async def set_generation_persona(name: str) -> Dict[str, Any]:
    """
    Set the active persona for image generation.
    
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
        
        physical = persona.get("physical_traits", {})
        style = persona.get("visual_style", {})
        
        return {
            "success": True,
            "persona": name,
            "physical_traits": list(physical.keys()),
            "preferred_styles": style.get("preferred_styles", []),
        }
    
    return {
        "success": False,
        "error": f"Persona '{name}' not found in {PERSONA_CONFIG_DIR}",
    }


# ============================================================================
# FUNCTION TOOL WRAPPERS
# ============================================================================

def get_tools(persona: Optional[str] = None) -> List[FunctionTool]:
    """
    Get generation tools as FunctionTool instances for ADK agents.
    
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
        FunctionTool(generate_image),
        FunctionTool(generate_selfie),
        FunctionTool(generate_scene),
        FunctionTool(refine_image),
        FunctionTool(set_generation_persona),
    ]


__all__ = [
    "generate_image",
    "generate_selfie",
    "generate_scene",
    "refine_image",
    "set_generation_persona",
    "get_tools",
]
