"""Memory Management Tool — Explore, recall, pin, forget across ALL tiers.

The 4 memory tiers:
    Tier 1: In-story memories (MemoryManager) — episodic/semantic/procedural/emotional
    Tier 2: World graph (StoryGraph) — characters, locations, items, factions, edges
    Tier 3: Chat history (ADK Memory) — raw messages with session repair
    Tier 4: Cross-session knowledge (ADK GraphMemory) — embeddings, entity relations

Plus the archive (append-only JSONL — nothing ever lost).

This tool provides unified access to ALL tiers so you can explore,
promote, demote, and search across the entire memory system.
"""

from __future__ import annotations

from typing import Optional

from adk.tools import tool


def _get_memory():
    from .story_turn import _get_engine
    _, memory, _ = _get_engine()
    return memory


def _get_graph():
    from .story_turn import _get_engine
    graph, _, _ = _get_engine()
    return graph


@tool(
    name="recall_memories",
    description=(
        "Search and recall relevant memories from ALL tiers: in-story memory store, "
        "world graph entities, and cross-session knowledge graph with semantic search."
    ),
)
def recall_memories(
    query: str = "",
    limit: int = 10,
    memory_type: Optional[str] = None,
    include_knowledge: bool = True,
    include_archive: bool = False,
) -> dict:
    """Recall memories by text search and relevance scoring across all tiers.

    Args:
        query: Search text (matches content and summaries)
        limit: Max memories to return per tier
        memory_type: Filter by type: 'episodic', 'semantic', 'procedural', 'emotional'
        include_knowledge: Also search cross-session knowledge graph (tier 4)
        include_archive: Also search the archive for old/consolidated memories
    """
    memory = _get_memory()
    graph = _get_graph()

    results = {"tiers": {}}

    # Tier 1: In-story memories
    tier1_results = memory.recall(
        query_text=query,
        limit=limit,
        current_turn=graph.world.turn_number,
    )
    if memory_type:
        tier1_results = [(m, r) for m, r in tier1_results if m.type.value == memory_type]

    results["tiers"]["in_story"] = {
        "count": len(tier1_results),
        "memories": [
            {
                "id": m.id,
                "tier": 1,
                "type": m.type.value,
                "summary": m.summary,
                "content": m.content[:200],
                "importance": m.effective_importance(graph.world.turn_number),
                "pinned": m.pinned,
                "turn": m.turn_number,
                "access_count": m.access_count,
                "reason": reason,
            }
            for m, reason in tier1_results
        ],
    }

    # Tier 2: World graph entities matching query
    if query:
        from saga_engine.models import NodeType
        matching_nodes = graph.search_nodes(query)
        results["tiers"]["world_graph"] = {
            "count": len(matching_nodes),
            "entities": [
                {
                    "id": n.id,
                    "tier": 2,
                    "type": n.type.value,
                    "name": n.name,
                    "description": n.short_description or n.description[:150],
                    "status": n.status.value,
                    "pinned": n.pinned,
                }
                for n in matching_nodes[:limit]
            ],
        }

    # Tier 4: Cross-session knowledge (semantic search)
    if include_knowledge:
        try:
            from saga_engine.persistent_memory import search_knowledge
            from .story_turn import _run_async
            knowledge_results = _run_async(search_knowledge(query, limit=limit))
            results["tiers"]["cross_session"] = {
                "count": len(knowledge_results),
                "knowledge": [
                    {
                        "tier": 4,
                        "label": k.label if hasattr(k, 'label') else str(k),
                        "content": k.content[:200] if hasattr(k, 'content') else str(k)[:200],
                        "type": k.node_type if hasattr(k, 'node_type') else "unknown",
                    }
                    for k in knowledge_results[:limit]
                ],
            }
        except Exception as e:
            results["tiers"]["cross_session"] = {"error": str(e), "count": 0}

    # Archive search (old/consolidated memories — nothing is ever lost)
    if include_archive:
        try:
            from saga_engine.memory_consolidation import MemoryConsolidator
            consolidator = MemoryConsolidator(memory, graph)
            archive_results = consolidator.search_archive(query, limit=limit)
            results["tiers"]["archive"] = {
                "count": len(archive_results),
                "archived": [
                    {
                        "tier": "archive",
                        "archived_at": a.get("archived_at", ""),
                        "reason": a.get("reason", ""),
                        "summary": a.get("memory", {}).get("summary", ""),
                        "content": a.get("memory", {}).get("content", "")[:200],
                        "type": a.get("memory", {}).get("type", ""),
                    }
                    for a in archive_results
                ],
            }
        except Exception as e:
            results["tiers"]["archive"] = {"error": str(e), "count": 0}

    # Summary
    total = sum(t.get("count", 0) for t in results["tiers"].values())
    results["total_across_tiers"] = total

    return results


@tool(
    name="pin_memory",
    description="Pin a memory to prevent it from decaying. Pinned memories are always available.",
)
def pin_memory(memory_id: str) -> dict:
    """Pin a memory by ID.

    Args:
        memory_id: The memory ID to pin
    """
    memory = _get_memory()
    mem = memory.get(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found"}

    memory.update(memory_id, pinned=True)
    memory.save()
    return {"id": memory_id, "pinned": True, "summary": mem.summary}


@tool(
    name="unpin_memory",
    description="Unpin a memory so it can naturally decay over time.",
)
def unpin_memory(memory_id: str) -> dict:
    """Unpin a memory by ID.

    Args:
        memory_id: The memory ID to unpin
    """
    memory = _get_memory()
    mem = memory.get(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found"}

    memory.update(memory_id, pinned=False)
    memory.save()
    return {"id": memory_id, "pinned": False, "summary": mem.summary}


@tool(
    name="promote_memory",
    description=(
        "Promote a tier-1 in-story memory to tier-4 cross-session knowledge. "
        "The memory stays in tier 1 AND is copied to tier 4 for permanence."
    ),
)
def promote_memory(memory_id: str) -> dict:
    """Promote a memory to cross-session knowledge.

    Args:
        memory_id: The memory ID to promote
    """
    memory = _get_memory()
    mem = memory.get(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found"}

    try:
        from saga_engine.persistent_memory import get_graph_memory
        from .story_turn import _run_async

        gm = get_graph_memory()
        _run_async(gm.add(
            label=mem.summary,
            content=mem.content,
            node_type=f"promoted_{mem.type.value}",
            tags=mem.tags + ["promoted"],
            importance=mem.importance,
            metadata={
                "source_memory_id": mem.id,
                "source_turn": mem.turn_number,
                "promoted_manually": True,
            },
        ))

        # Also pin it in tier 1
        memory.update(memory_id, pinned=True)
        memory.save()

        return {"promoted": memory_id, "summary": mem.summary, "to_tier": 4}
    except Exception as e:
        return {"error": f"Promotion failed: {e}"}


@tool(
    name="demote_memory",
    description=(
        "Demote a memory by reducing its importance. It won't be deleted — "
        "just ranked lower in recall results and eventually archived (not lost)."
    ),
)
def demote_memory(memory_id: str, new_importance: float = 0.1) -> dict:
    """Demote a memory by reducing importance.

    Args:
        memory_id: The memory ID to demote
        new_importance: New importance level (0-1, lower = less prominent)
    """
    memory = _get_memory()
    mem = memory.get(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found"}

    memory.update(memory_id, importance=max(0.01, new_importance), pinned=False)
    memory.save()
    return {"demoted": memory_id, "new_importance": new_importance, "summary": mem.summary}


@tool(
    name="consolidate_memories",
    description=(
        "Run memory consolidation: deduplicate near-duplicates, consolidate per-entity, "
        "promote important patterns, and archive decayed memories. Nothing is ever deleted — "
        "only compressed and promoted up through tiers."
    ),
)
def consolidate_memories() -> dict:
    """Run the consolidation pipeline manually."""
    from saga_engine.memory_consolidation import MemoryConsolidator

    memory = _get_memory()
    graph = _get_graph()

    consolidator = MemoryConsolidator(memory, graph)
    stats = consolidator.consolidate_all(graph.world.turn_number)
    stats["archive_total"] = consolidator.archive_count

    return stats


@tool(
    name="memory_stats",
    description="Get statistics about ALL memory tiers — in-story, world graph, knowledge, and archive.",
)
def memory_stats() -> dict:
    """Get comprehensive memory statistics across all tiers."""
    memory = _get_memory()
    graph = _get_graph()

    stats = {
        "tier_1_in_story": memory.stats(),
        "tier_2_world_graph": graph.stats(),
    }

    # Tier 4: Cross-session knowledge
    try:
        from saga_engine.persistent_memory import get_graph_memory
        gm = get_graph_memory()
        stats["tier_4_cross_session"] = {"status": "active", "db": str(gm._db_path)}
    except Exception:
        stats["tier_4_cross_session"] = {"status": "unavailable"}

    # Archive
    try:
        from saga_engine.memory_consolidation import MemoryConsolidator
        consolidator = MemoryConsolidator(memory, graph)
        stats["archive"] = {"count": consolidator.archive_count}
    except Exception:
        stats["archive"] = {"count": 0}

    return stats
