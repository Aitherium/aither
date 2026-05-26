"""Story Turn Tool — Process a narrative turn through the full 6-pillar pipeline.

The 6 pillars of context assembly:
    1. EXTRACT  -- Parse user input for entity references + intents
    2. ACTIVATE -- Match extracted entities to graph nodes + scene state
    3. EXPAND   -- Walk edges 1-2 hops for related context
    4. RECALL   -- Semantic search via embeddings + keyword memory recall
    5. RANK     -- Score everything by relevance
    6. ASSEMBLE -- Build final context, log what's in and what's out

Additionally wires:
    - ADK GraphMemory for cross-session semantic search (embeddings)
    - ADK Memory for conversation persistence across restarts
    - Embedding indexing so every new story turn is searchable
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.story_turn")

# Lazy-loaded engine instances
_graph = None
_memory = None
_context = None
_embedding_recall = None
_data_dir = None


def _get_data_dir() -> Path:
    global _data_dir
    if _data_dir is None:
        _data_dir = Path.home() / ".saga" / "active_project"
        _data_dir.mkdir(parents=True, exist_ok=True)
    return _data_dir


def _get_engine():
    """Lazy-load the saga engine from the project directory."""
    global _graph, _memory, _context
    if _graph is not None:
        return _graph, _memory, _context

    from saga_engine.graph import StoryGraph
    from saga_engine.memory import MemoryManager
    from saga_engine.context import ContextAssembler

    data_dir = _get_data_dir()

    _graph = StoryGraph(data_dir=data_dir)
    _graph.load()
    _memory = MemoryManager(data_dir=data_dir)
    _memory.load()
    _context = ContextAssembler(_graph, _memory)

    return _graph, _memory, _context


def _get_embedding_recall():
    """Lazy-load the embedding recall system."""
    global _embedding_recall
    if _embedding_recall is not None:
        return _embedding_recall

    try:
        from saga_engine.embedding_recall import EmbeddingRecall
        _embedding_recall = EmbeddingRecall(data_dir=_get_data_dir())
        logger.info("Embedding recall initialized")
    except ImportError:
        logger.warning("EmbeddingRecall unavailable — falling back to keyword-only recall")
        _embedding_recall = None

    return _embedding_recall


def _run_async(coro):
    """Run an async function from sync context."""
    try:
        loop = asyncio.get_running_loop()
        # We're inside an event loop — schedule as task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


@tool(
    name="process_story_turn",
    description=(
        "Process a story turn through the full 6-pillar context pipeline: "
        "EXTRACT entities, ACTIVATE graph nodes, EXPAND via edge traversal, "
        "RECALL via semantic embeddings + memory, RANK by relevance, ASSEMBLE final context. "
        "Call this BEFORE generating your narrative to ensure continuity."
    ),
)
def process_story_turn(
    message: str,
    input_mode: str = "narrative",
    active_character: str = "",
    active_location: str = "",
    context_budget: int = 0,
) -> dict:
    """Process a story turn through the 6-pillar context pipeline.

    Args:
        message: The user's input text
        input_mode: One of 'narrative', 'dialogue', 'action'
        active_character: Name of the currently active character
        active_location: Name of the current location
        context_budget: Max tokens for context assembly (0 = auto from effort mapper)
    """
    graph, memory, context_asm = _get_engine()

    # Classify narrative complexity and map to effort level
    from saga_engine.effort_mapper import classify_effort
    effort_result = classify_effort(
        user_input=message,
        turn_number=graph.world.turn_number,
        active_plot_threads=len(graph.world.active_plot_threads),
        present_characters=len(graph.world.present_characters),
    )

    # Use effort-mapped budget if not explicitly provided
    if context_budget <= 0:
        context_budget = effort_result.context_budget

    # Index all current story content into embedding store (incremental)
    embed_recall = _get_embedding_recall()
    if embed_recall:
        try:
            _run_async(embed_recall.index_story_content(graph, memory))
        except Exception as e:
            logger.warning("Embedding indexing failed (non-fatal): %s", e)

    # Run the 6-stage context assembly pipeline
    assembly = context_asm.assemble(
        user_input=message,
        token_budget=context_budget,
    )

    # Supplement with semantic recall from embeddings (enhances stage 4)
    semantic_context = ""
    semantic_count = 0
    if embed_recall:
        try:
            activated_ids = [n.node_id for n in assembly.activated_nodes]
            sem_results = _run_async(
                embed_recall.semantic_recall(message, activated_ids, limit=8)
            )
            if sem_results:
                semantic_lines = []
                for r in sem_results:
                    if r["summary"] not in assembly.context_text:
                        semantic_lines.append(f"- {r['summary']}: {r['content'][:150]}")
                        semantic_count += 1
                if semantic_lines:
                    semantic_context = "\n[SEMANTIC RECALL]\n" + "\n".join(semantic_lines[:6])
        except Exception as e:
            logger.warning("Semantic recall failed (non-fatal): %s", e)

    # Combine base context + semantic recall
    full_context = assembly.context_text
    if semantic_context:
        full_context += "\n" + semantic_context

    # Advance turn
    graph.world.turn_number += 1

    # Auto-save graph state
    graph.save()

    # Persist to ADK Memory for restart resilience
    try:
        from saga_engine.persistent_memory import save_turn_to_history
        _run_async(save_turn_to_history(
            session_id=graph.world.story_name,
            user_input=message,
            saga_response="",  # Filled by after_story_turn
            turn_number=graph.world.turn_number,
        ))
    except Exception as e:
        logger.debug("Persistent memory save failed (non-fatal): %s", e)

    # Build style prompt if style system is active
    style_prompt = ""
    try:
        from saga_engine.style import get_active_style
        style = get_active_style()
        style_prompt = style.build_style_prompt()
    except Exception:
        pass

    return {
        "context": full_context,
        "style_context": style_prompt,
        "turn_number": graph.world.turn_number,
        "activated_nodes": len(assembly.activated_nodes),
        "recalled_memories": len(assembly.recalled_memories),
        "semantic_recall_count": semantic_count,
        "token_estimate": assembly.token_estimate + (len(semantic_context) // 4),
        "effort": {
            "level": effort_result.effort,
            "reasoning": effort_result.reasoning,
            "use_reasoning_tool": effort_result.use_reasoning_tool,
            "mcts_mode": effort_result.mcts_mode,
        },
        "world_state": {
            "story_name": graph.world.story_name,
            "current_location": graph.world.current_location,
            "mood": graph.world.mood,
            "time_of_day": graph.world.time_of_day,
        },
        "stages": [
            {"name": s.name, "items_in": s.items_in, "items_out": s.items_out,
             "duration_ms": s.duration_ms}
            for s in assembly.stages
        ],
        "embedding_stats": embed_recall.stats if embed_recall else {},
    }


@tool(
    name="after_story_turn",
    description=(
        "Call AFTER generating your narrative response. Stores the response as memory, "
        "indexes it for future semantic search, and persists to cross-session knowledge."
    ),
)
def after_story_turn(
    user_input: str,
    saga_response: str,
    memory_summary: str = "",
    importance: float = 0.5,
) -> dict:
    """Post-processing after a story turn. Stores memories and indexes for RAG.

    Args:
        user_input: The original user message
        saga_response: Your generated narrative response
        memory_summary: One-line summary of what happened (auto-generated if empty)
        importance: How important this turn was (0-1)
    """
    from saga_engine.models import MemoryType

    graph, memory, _ = _get_engine()

    # Store as episodic memory in story memory store
    mem = memory.create(
        memory_type=MemoryType.EPISODIC,
        content=f"Player: {user_input}\nNarrator: {saga_response[:500]}",
        summary=memory_summary,
        importance=importance,
        turn_number=graph.world.turn_number,
    )
    memory.save()

    # Index into embedding store for future semantic search
    embed_recall = _get_embedding_recall()
    if embed_recall:
        try:
            _run_async(embed_recall.ingest_story_turn(
                user_input=user_input,
                saga_response=saga_response,
                turn_number=graph.world.turn_number,
                session_id=graph.world.story_name,
            ))
        except Exception as e:
            logger.debug("Embedding ingestion failed (non-fatal): %s", e)

    # Store in cross-session knowledge graph
    try:
        from saga_engine.persistent_memory import learn_from_story
        _run_async(learn_from_story(user_input, saga_response, graph.world.story_name))
    except Exception as e:
        logger.debug("Cross-session learning failed (non-fatal): %s", e)

    return {
        "memory_id": mem.id,
        "memory_summary": mem.summary,
        "turn_number": graph.world.turn_number,
        "indexed": embed_recall is not None,
    }


@tool(
    name="store_story_memory",
    description="Store a new memory from the current story turn (episodic event, semantic fact, etc.)",
)
def store_story_memory(
    content: str,
    memory_type: str = "episodic",
    summary: str = "",
    importance: float = 0.5,
    pinned: bool = False,
) -> dict:
    """Store a new memory after a story turn.

    Args:
        content: Full memory text
        memory_type: One of 'episodic', 'semantic', 'procedural', 'emotional'
        summary: One-line summary (auto-generated if empty)
        importance: 0-1, how important
        pinned: If True, never decays
    """
    from saga_engine.models import MemoryType

    graph, memory, _ = _get_engine()

    type_map = {
        "episodic": MemoryType.EPISODIC,
        "semantic": MemoryType.SEMANTIC,
        "procedural": MemoryType.PROCEDURAL,
        "emotional": MemoryType.EMOTIONAL,
    }

    mem = memory.create(
        memory_type=type_map.get(memory_type, MemoryType.EPISODIC),
        content=content,
        summary=summary,
        importance=importance,
        turn_number=graph.world.turn_number,
        pinned=pinned,
    )
    memory.save()

    return {"id": mem.id, "summary": mem.summary, "type": mem.type.value}
