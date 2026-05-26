"""
Saga Engine — Standalone Story Memory & Context System
=======================================================

Vendored from AitherOS storygraph with zero platform dependencies.
Uses only Pydantic, standard library, and local JSON persistence.

Architecture:
    StoryGraph (nodes + edges)
      +-- ContextAssembler (6-stage glass-box pipeline)
      +-- MemoryManager (episodic/semantic/procedural/emotional)
      +-- NarrativeMCTS (tree search for story branching)
"""

from .models import (
    NodeType,
    EdgeType,
    MemoryType,
    StoryNode,
    StoryEdge,
    StoryMemory,
    WorldState,
    ContextAssembly,
    ContextStage,
    ActivatedNode,
    RecalledMemory,
    PrunedItem,
)
from .graph import StoryGraph
from .context import ContextAssembler
from .memory import MemoryManager
from .memory_consolidation import MemoryConsolidator

__all__ = [
    "NodeType", "EdgeType", "MemoryType",
    "StoryNode", "StoryEdge", "StoryMemory",
    "WorldState", "ContextAssembly", "ContextStage",
    "ActivatedNode", "RecalledMemory", "PrunedItem",
    "StoryGraph", "ContextAssembler", "MemoryManager",
    "MemoryConsolidator",
]
