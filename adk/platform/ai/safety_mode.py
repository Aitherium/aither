"""Safety-mode management for the ADK platform toolkit.

A self-contained safety-level system (no AitherOS internals).  Three levels
govern how an agent routes and filters content:

* PROFESSIONAL — business-focused, cloud LLM, strict filtering
* CASUAL       — relaxed but filtered, cloud LLM
* UNRESTRICTED — local LLM, no content filters

Override prefixes bypass safety for a single turn:
``::``  ``~``  ``>>>``  ``[!]``

Usage:
    from adk.platform.ai.safety_mode import get_current_level, SafetyLevel, set_safety_level
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Unified safety levels."""

    PROFESSIONAL = "professional"  # Business-safe, cloud LLM, strict filtering
    CASUAL = "casual"              # Relaxed but filtered, cloud LLM
    UNRESTRICTED = "unrestricted"  # Local LLM, no content filters

    @classmethod
    def from_string(cls, value: str) -> "SafetyLevel":
        value = (value or "").lower().strip()
        if value in ("high", "professional", "strict"):
            return cls.PROFESSIONAL
        if value in ("medium", "casual", "moderate"):
            return cls.CASUAL
        if value in ("low", "off", "unrestricted", "unsafe", "none"):
            return cls.UNRESTRICTED
        return cls.PROFESSIONAL

    def to_image_safety(self) -> str:
        return {
            SafetyLevel.PROFESSIONAL: "HIGH",
            SafetyLevel.CASUAL: "MEDIUM",
            SafetyLevel.UNRESTRICTED: "OFF",
        }[self]


@dataclass
class SafetyConfig:
    """Configuration for a safety level."""

    level: SafetyLevel
    use_cloud_llm: bool
    allow_explicit: bool
    content_filter: str
    llm_harm_threshold: str


SAFETY_CONFIGS: dict[SafetyLevel, SafetyConfig] = {
    SafetyLevel.PROFESSIONAL: SafetyConfig(
        level=SafetyLevel.PROFESSIONAL,
        use_cloud_llm=True,
        allow_explicit=False,
        content_filter="strict",
        llm_harm_threshold="BLOCK_LOW_AND_ABOVE",
    ),
    SafetyLevel.CASUAL: SafetyConfig(
        level=SafetyLevel.CASUAL,
        use_cloud_llm=True,
        allow_explicit=False,
        content_filter="moderate",
        llm_harm_threshold="BLOCK_MEDIUM_AND_ABOVE",
    ),
    SafetyLevel.UNRESTRICTED: SafetyConfig(
        level=SafetyLevel.UNRESTRICTED,
        use_cloud_llm=False,
        allow_explicit=True,
        content_filter="none",
        llm_harm_threshold="BLOCK_NONE",
    ),
}

OVERRIDE_PREFIXES: list[str] = ["::", "~", ">>>", "[!]"]


class SafetyManager:
    """Process-local safety-level state.

    Default level comes from ``AITHER_SAFETY_LEVEL`` (or PROFESSIONAL). The
    level is held in memory for the process; callers persist it as they see fit.
    """

    _instance: "SafetyManager | None" = None

    def __init__(self) -> None:
        self.current_level = SafetyLevel.from_string(os.getenv("AITHER_SAFETY_LEVEL", "professional"))

    @classmethod
    def get_instance(cls) -> "SafetyManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_level(self) -> SafetyLevel:
        return self.current_level

    def set_level(self, level: SafetyLevel) -> None:
        self.current_level = level

    def get_config(self) -> SafetyConfig:
        return SAFETY_CONFIGS[self.current_level]

    def check_override(self, message: str) -> tuple[bool, str]:
        stripped = (message or "").strip()
        for prefix in OVERRIDE_PREFIXES:
            if stripped.startswith(prefix):
                return True, stripped[len(prefix):].strip()
        return False, message

    def get_effective_config(self, message: str) -> tuple[SafetyConfig, str, bool]:
        is_override, cleaned = self.check_override(message)
        if is_override:
            return SAFETY_CONFIGS[SafetyLevel.UNRESTRICTED], cleaned, True
        return self.get_config(), message, False


# ── Public API (mirrors the names platform modules import) ──────────────────


def get_safety_manager() -> SafetyManager:
    return SafetyManager.get_instance()


def get_current_level() -> SafetyLevel:
    return get_safety_manager().get_level()


def get_safety_level() -> str:
    return get_safety_manager().get_level().value


def set_safety_level(level: str) -> None:
    get_safety_manager().set_level(SafetyLevel.from_string(level))


def get_safety_config() -> SafetyConfig:
    return get_safety_manager().get_config()


def check_message_override(message: str) -> tuple[bool, str]:
    return get_safety_manager().check_override(message)


def get_effective_config(message: str) -> tuple[SafetyConfig, str, bool]:
    return get_safety_manager().get_effective_config(message)


def get_level_emoji(level: SafetyLevel | None = None) -> str:
    level = level or get_current_level()
    return {"professional": "💼", "casual": "😎", "unrestricted": "🔓"}.get(level.value, "")


def get_level_name(level: SafetyLevel | None = None) -> str:
    level = level or get_current_level()
    return {
        "professional": "Professional",
        "casual": "Casual",
        "unrestricted": "Unrestricted",
    }.get(level.value, "Unknown")


# ── Convenience aliases ─────────────────────────────────────────────────────


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
