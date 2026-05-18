"""
AitherOS Safety Settings

This module provides Google GenAI safety settings for LLM interactions.
Uses the centralized AitherSafety module when available, with fallback defaults.

Usage:
    from aither_adk.ai.safety import get_safety_settings
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

# Try to import from services (when running in full AitherOS environment)
_AITHERNODE_PATH = Path(__file__).parent.parent.parent.parent.parent / "AitherNode"
if _AITHERNODE_PATH.exists() and str(_AITHERNODE_PATH) not in sys.path:
    sys.path.insert(0, str(_AITHERNODE_PATH))

# Also try services root for Docker containers
_SERVICES_PATH = Path(__file__).parent.parent.parent.parent.parent / "services"
if _SERVICES_PATH.exists() and str(_SERVICES_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_SERVICES_PATH.parent))

_has_centralized_safety = False
try:
    from services.cognition.AitherSafety import get_llm_safety_settings
    _has_centralized_safety = True
except ImportError:
    pass

# Try to import Google GenAI types
_has_genai = False
try:
    from google.genai import types as genai_types
    _has_genai = True
except ImportError:
    genai_types = None


def _get_default_safety_settings(safety_level: Optional[str] = None) -> List:
    """
    Fallback safety settings when AitherSafety module is not available.
    Returns Google GenAI compatible safety settings.
    """
    if not _has_genai:
        return []
    
    level = (safety_level or os.environ.get("AITHER_SAFETY_LEVEL", "MEDIUM")).upper()
    
    # Map level to HarmBlockThreshold
    threshold_map = {
        "HIGH": genai_types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        "MEDIUM": genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "LOW": genai_types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        "OFF": genai_types.HarmBlockThreshold.BLOCK_NONE,
    }
    threshold = threshold_map.get(level, genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
    
    categories = [
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    ]
    
    return [
        genai_types.SafetySetting(category=cat, threshold=threshold)
        for cat in categories
    ]


def get_safety_settings(safety_level: Optional[str] = None) -> List:
    """
    Returns Google GenAI safety settings for the specified level.
    
    Args:
        safety_level: One of "HIGH", "MEDIUM", "LOW", "OFF"
    
    Returns:
        List of SafetySetting objects for Google GenAI
    """
    if _has_centralized_safety:
        return get_llm_safety_settings(safety_level)
    else:
        return _get_default_safety_settings(safety_level)
