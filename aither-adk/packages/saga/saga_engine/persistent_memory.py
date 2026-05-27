"""
Persistent Memory — Cross-Session Story Knowledge
===================================================

Wraps ADK's Memory (SQLite KV + conversation history) and GraphMemory
(embeddings + knowledge graph) into a unified story persistence layer.

Four tiers of memory for Saga:

    1. In-story memories (MemoryManager) — episodic/semantic/procedural/emotional
       per-project, JSON persistence, decay + pinning

    2. World graph (StoryGraph) — characters, locations, items, factions, edges
       per-project, JSON persistence

    3. Chat history (ADK Memory) — raw messages with session repair
       per-session, SQLite persistence

    4. Cross-session knowledge (ADK GraphMemory) — author preferences, patterns,
       entity relationships, embeddings
       global, SQLite + vector search

This module wires tiers 3 and 4 so they work alongside tiers 1 and 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("saga.persistent_memory")

_adk_memory = None
_graph_memory = None

SAGA_HOME = Path.home() / ".saga"


def get_adk_memory():
    """Get the ADK Memory instance for Saga (conversation history + KV store)."""
    global _adk_memory
    if _adk_memory is not None:
        return _adk_memory

    from adk.memory import Memory

    db_path = SAGA_HOME / "memory" / "saga.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _adk_memory = Memory(db_path=db_path, agent_name="saga")
    logger.info("ADK Memory initialized: %s", db_path)
    return _adk_memory


def get_graph_memory():
    """Get the GraphMemory instance for cross-session knowledge."""
    global _graph_memory
    if _graph_memory is not None:
        return _graph_memory

    from adk.graph_memory import GraphMemory

    db_path = SAGA_HOME / "knowledge" / "saga.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _graph_memory = GraphMemory(db_path=db_path, agent_name="saga")
    logger.info("GraphMemory initialized: %s", db_path)
    return _graph_memory


async def save_turn_to_history(
    session_id: str,
    user_input: str,
    saga_response: str,
    turn_number: int,
    metadata: Optional[dict] = None,
):
    """Persist a story turn to ADK Memory for session continuity."""
    mem = get_adk_memory()

    await mem.add_message(
        session_id=session_id,
        role="user",
        content=user_input,
        metadata={"turn_number": turn_number, **(metadata or {})},
    )
    await mem.add_message(
        session_id=session_id,
        role="assistant",
        content=saga_response,
        metadata={"turn_number": turn_number},
    )


async def get_session_history(session_id: str, limit: int = 50) -> list[dict]:
    """Retrieve conversation history for session continuity after restart."""
    mem = get_adk_memory()
    return await mem.get_history(session_id, limit=limit)


async def remember_preference(key: str, value: str, category: str = "preference"):
    """Store an author preference that persists across all sessions/projects."""
    mem = get_adk_memory()
    await mem.remember(key, value, category=category)


async def recall_preference(key: str) -> Optional[str]:
    """Recall a stored preference."""
    mem = get_adk_memory()
    return await mem.recall(key)


async def learn_from_story(
    user_input: str,
    saga_response: str,
    session_id: str = "",
):
    """Extract entities and relations from a story turn and store in GraphMemory.

    This builds a persistent knowledge graph of story patterns, character names,
    locations, and narrative structures that Saga can recall across sessions.
    """
    gm = get_graph_memory()
    await gm.ingest_conversation(
        session_id=session_id,
        messages=[
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": saga_response},
        ],
    )


async def search_knowledge(query: str, limit: int = 10) -> list:
    """Semantic search across all cross-session knowledge."""
    gm = get_graph_memory()
    return await gm.search(query, limit=limit)
