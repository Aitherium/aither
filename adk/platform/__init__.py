"""
ADK Platform — Internal AitherOS Runtime Library
=================================================

This sub-package contains the internal platform toolkit that was previously
the standalone ``aither-platform`` / ``aither_adk`` package. It is now
unified under the canonical ``adk`` package.

Modules:
  - ``adk.platform.ai``              — LLM providers, prompt engineering, safety
  - ``adk.platform.communication``   — IRC relay, messaging, notifications
  - ``adk.platform.infrastructure``  — Service management, auth, deployment
  - ``adk.platform.memory``          — Memory systems, game engine, storyboard
  - ``adk.platform.tools``           — Tool wrappers, function registration
  - ``adk.platform.ui``              — Console UI, toolbars, commands

Migration:
  Old imports like ``from adk.platform.ai.ollama import OllamaLlm``
  now resolve to ``from adk.platform.ai.ollama import OllamaLlm``.
  A compatibility shim in the ``aither_adk`` package re-exports everything,
  so existing code continues to work without changes.
"""

__version__ = "1.0.0"  # Unified with adk version

MODULES = [
    "ai",
    "communication",
    "infrastructure",
    "memory",
    "tools",
    "ui",
]


def get_info() -> dict:
    """Return platform package information."""
    return {
        "name": "adk.platform",
        "version": __version__,
        "modules": MODULES,
        "merged_from": "aither-platform (0.1.0)",
    }
