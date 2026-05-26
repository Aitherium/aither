"""Story Revision & Deletion Tool — Propagate changes through all memory layers.

When story elements are deleted or revised, the change MUST propagate to:
    1. StoryGraph (nodes, edges)
    2. MemoryManager (episodic/semantic/procedural/emotional memories)
    3. EmbeddingRecall (re-index or remove from vector store)
    4. Persistent Memory (ADK GraphMemory cross-session knowledge)
    5. Context assembler cache (invalidate last assembly)

This prevents ghost data — deleted characters appearing in context,
outdated facts contradicting the current narrative, etc.
"""

from __future__ import annotations

import logging
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.revise")


def _get_all_engines():
    """Get all engine instances for propagation."""
    from .story_turn import _get_engine, _get_embedding_recall, _run_async
    graph, memory, context = _get_engine()
    embed = _get_embedding_recall()
    return graph, memory, context, embed, _run_async


@tool(
    name="delete_story_element",
    description=(
        "Delete a story element (character, location, item, faction, etc.) from ALL layers: "
        "world graph, memories, embeddings, and cross-session knowledge. "
        "This is a PERMANENT deletion that propagates everywhere."
    ),
)
def delete_story_element(
    name: str,
    also_delete_memories: bool = True,
    reason: str = "",
) -> dict:
    """Delete a story element and propagate to all memory layers.

    Args:
        name: Name of the element to delete
        also_delete_memories: Also delete all memories mentioning this element
        reason: Why it's being deleted (stored as a revision record)
    """
    graph, memory, context, embed, run_async = _get_all_engines()

    node = graph.find_node_by_name(name)
    if not node:
        return {"error": f"Element '{name}' not found in world graph"}

    node_id = node.id
    node_type = node.type.value
    deleted = {"node": name, "type": node_type, "layers_updated": []}

    # 1. Delete from StoryGraph (also removes connected edges)
    graph.delete_node(node_id)
    graph.save()
    deleted["layers_updated"].append("world_graph")

    # 2. Delete related memories
    if also_delete_memories:
        related_mems = memory.get_by_node(node_id)
        # Also search by name in memory content
        name_matches = memory.search(name)
        all_to_delete = {m.id for m in related_mems}
        all_to_delete.update(m.id for m in name_matches)

        for mid in all_to_delete:
            memory.delete(mid)
        memory.save()
        deleted["memories_removed"] = len(all_to_delete)
        deleted["layers_updated"].append("memory_store")

    # 3. Remove from embedding index
    if embed:
        try:
            gm = embed.graph_mem
            # Search for and remove matching nodes in GraphMemory
            results = run_async(gm.search(name, limit=20))
            removed_count = 0
            for result in results:
                meta = result.metadata or {}
                if (meta.get("story_node_id") == node_id or
                        result.label.lower() == name.lower()):
                    run_async(gm.remove(result.id))
                    removed_count += 1
            if removed_count:
                deleted["layers_updated"].append("embedding_index")
                deleted["embeddings_removed"] = removed_count

            # Force re-index on next turn
            embed._indexed_node_ids.discard(node_id)
        except Exception as e:
            logger.warning("Embedding cleanup failed (non-fatal): %s", e)

    # 4. Store deletion as a procedural memory (so Saga knows it was deleted)
    from saga_engine.models import MemoryType
    memory.create(
        memory_type=MemoryType.PROCEDURAL,
        content=f"DELETED: {name} ({node_type}) was removed from the story. {reason}".strip(),
        summary=f"{name} no longer exists in this story",
        importance=0.8,
        turn_number=graph.world.turn_number,
        pinned=True,  # Pin so the deletion is never forgotten
    )
    memory.save()

    # 5. Invalidate context cache
    if context:
        context._last_assembly = None

    deleted["layers_updated"].append("revision_record")
    return deleted


@tool(
    name="revise_story_element",
    description=(
        "Revise/retcon a story element — update its description, properties, or status "
        "and propagate the change through all memory and embedding layers. "
        "Old versions are stored as revision history."
    ),
)
def revise_story_element(
    name: str,
    new_description: Optional[str] = None,
    new_properties: Optional[dict] = None,
    new_status: Optional[str] = None,
    revision_note: str = "",
) -> dict:
    """Revise a story element and propagate to all layers.

    Args:
        name: Name of the element to revise
        new_description: Updated description (None = keep current)
        new_properties: Dict of properties to update/add (None = keep current)
        new_status: New status: 'active', 'dormant', 'destroyed', 'hidden' (None = keep)
        revision_note: Why this revision was made
    """
    graph, memory, context, embed, run_async = _get_all_engines()

    node = graph.find_node_by_name(name)
    if not node:
        return {"error": f"Element '{name}' not found"}

    old_description = node.description
    old_status = node.status.value
    updates = {}

    if new_description is not None:
        updates["description"] = new_description
    if new_properties is not None:
        merged = dict(node.properties)
        merged.update(new_properties)
        updates["properties"] = merged
    if new_status is not None:
        from saga_engine.models import NodeStatus
        status_map = {
            "active": NodeStatus.ACTIVE,
            "dormant": NodeStatus.DORMANT,
            "destroyed": NodeStatus.DESTROYED,
            "hidden": NodeStatus.HIDDEN,
        }
        if new_status in status_map:
            updates["status"] = status_map[new_status]

    if not updates:
        return {"error": "No changes specified"}

    # 1. Update in StoryGraph
    graph.update_node(node.id, **updates)
    graph.save()

    # 2. Update related memories that reference old description
    revised_mems = 0
    if new_description and old_description:
        for mem in memory.search(old_description[:50]):
            new_content = mem.content.replace(old_description[:100], new_description[:100])
            if new_content != mem.content:
                memory.update(mem.id, content=new_content)
                revised_mems += 1
        memory.save()

    # 3. Re-index in embedding store
    if embed:
        try:
            # Remove old embedding, force re-index
            embed._indexed_node_ids.discard(node.id)
            run_async(embed.index_story_content(graph, memory))
        except Exception as e:
            logger.warning("Embedding re-index failed (non-fatal): %s", e)

    # 4. Store revision as procedural memory
    from saga_engine.models import MemoryType
    revision_content = f"REVISED: {name} was updated."
    if new_description:
        revision_content += f" Description changed."
    if new_status:
        revision_content += f" Status: {old_status} -> {new_status}."
    if revision_note:
        revision_content += f" Reason: {revision_note}"

    memory.create(
        memory_type=MemoryType.PROCEDURAL,
        content=revision_content,
        summary=f"{name} was revised (turn {graph.world.turn_number})",
        importance=0.6,
        turn_number=graph.world.turn_number,
    )
    memory.save()

    # 5. Invalidate context cache
    if context:
        context._last_assembly = None

    return {
        "revised": name,
        "fields_changed": list(updates.keys()),
        "memories_updated": revised_mems,
        "re_indexed": embed is not None,
    }


@tool(
    name="retcon_memory",
    description=(
        "Retcon (retroactively change) a specific memory — update its content, "
        "mark it as superseded, or delete it entirely. Propagates to embeddings."
    ),
)
def retcon_memory(
    memory_id: str,
    action: str = "revise",
    new_content: str = "",
    reason: str = "",
) -> dict:
    """Retcon a specific memory.

    Args:
        memory_id: The memory ID to modify
        action: 'revise' (update content), 'supersede' (mark as outdated), or 'delete'
        new_content: New content (for 'revise' action)
        reason: Why this retcon is happening
    """
    graph, memory, context, embed, run_async = _get_all_engines()

    mem = memory.get(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found"}

    old_summary = mem.summary

    if action == "delete":
        memory.delete(memory_id)
        memory.save()

        # Remove from embedding index
        if embed:
            try:
                embed._indexed_memory_ids.discard(memory_id)
            except Exception:
                pass

        return {"deleted": memory_id, "old_summary": old_summary}

    elif action == "supersede":
        memory.update(memory_id, importance=0.05, pinned=False)
        # Add supersession note
        from saga_engine.models import MemoryType
        memory.create(
            memory_type=MemoryType.PROCEDURAL,
            content=f"SUPERSEDED: '{old_summary}' is no longer accurate. {reason}",
            summary=f"Previous memory superseded: {old_summary[:60]}",
            importance=0.7,
            turn_number=graph.world.turn_number,
            pinned=True,
        )
        memory.save()
        return {"superseded": memory_id, "old_summary": old_summary}

    elif action == "revise":
        if not new_content:
            return {"error": "new_content required for revise action"}

        memory.update(memory_id, content=new_content)
        if not mem.summary or mem.summary == old_summary:
            from saga_engine.memory import MemoryManager
            memory.update(memory_id, summary=MemoryManager._auto_summary(new_content))
        memory.save()

        # Re-index
        if embed:
            embed._indexed_memory_ids.discard(memory_id)

        return {
            "revised": memory_id,
            "old_summary": old_summary,
            "new_summary": memory.get(memory_id).summary,
        }

    return {"error": f"Unknown action '{action}'. Use 'revise', 'supersede', or 'delete'."}
