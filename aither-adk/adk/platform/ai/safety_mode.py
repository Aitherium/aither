"""
AitherOS Safety Mode System

This module provides safety level management for the AitherOS ecosystem.
All functionality is provided by the centralized AitherSafety module.

Safety Levels:
- PROFESSIONAL: Business-focused, cloud LLM (Gemini), content filtered
- CASUAL: Relaxed but filtered, cloud LLM
- UNRESTRICTED: Local LLM, no content filters

Override Prefixes (bypass safety for one turn):
- `::` - Direct scope access
- `~`  - Tilde prefix  
- `>>>` - Bypass redirect
- `[!]` - Force unrestricted

Usage:
    from aither_adk.ai.safety_mode import get_current_level, SafetyLevel, set_safety_level
"""

import sys
from pathlib import Path

# Add AitherNode to path for services imports
_AITHERNODE_PATH = Path(__file__).parent.parent.parent.parent.parent / "AitherNode"
if _AITHERNODE_PATH.exists() and str(_AITHERNODE_PATH) not in sys.path:
    sys.path.insert(0, str(_AITHERNODE_PATH))

# Re-export everything from centralized module
from services.cognition.AitherSafety import (
    # Enums and dataclasses
    SafetyLevel,
    SafetyConfig,
    SAFETY_CONFIGS,
    OVERRIDE_PREFIXES,
    
    # Manager access
    get_safety_manager,
    
    # Level management
    get_current_level,
    set_safety_level,
    get_safety_config,
    
    # Override detection
    check_message_override,
    get_effective_config,
    
    # Display helpers
    get_level_emoji,
    get_level_name,
)

# Convenience aliases
def check_message(message: str):
    """Process a message and return (config, cleaned_message, is_override)."""
    return get_effective_config(message)

def is_override(message: str) -> bool:
    """Check if message has an override prefix."""
    return check_message_override(message)[0]

def use_local_llm(message: str) -> bool:
    """Check if we should use local LLM for this message."""
    is_ovr, _ = check_message_override(message)
    if is_ovr:
        return True
    return not get_safety_config().use_cloud_llm
