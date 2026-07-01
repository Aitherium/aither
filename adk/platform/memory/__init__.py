"""
Aither ADK - Memory Module  (LEGACY)
=====================================

.. deprecated:: 2.12.6
   ``adk.platform.memory`` is a legacy subsystem carried over from the
   merged ``aither-platform`` package.  For agent memory, use
   :class:`adk.memory.Memory` (short-term SQLite KV/conversation) and
   :class:`adk.graph_memory.GraphMemory` (long-term knowledge graph).
   These modules (MemoryManager, GameEngine, StoryboardEngine,
   UnifiedMemorySystem) remain importable for backward compatibility
   but are not part of the canonical ADK memory surface.

Import directly from submodules:
    >>> from adk.platform.memory.memory import MemoryManager
    >>> from adk.platform.memory.game_engine import GameEngine
"""

import warnings as _warnings
_warnings.warn(
    "adk.platform.memory is a legacy subsystem — use adk.memory.Memory "
    "and adk.graph_memory.GraphMemory for new code.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = []
