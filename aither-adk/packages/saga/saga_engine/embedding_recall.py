"""
Embedding-Powered Recall — Semantic Search for Story Context
=============================================================

Bridges ADK's EmbeddingProvider + GraphMemory into the ContextAssembler's
RECALL stage. This replaces dumb keyword matching with proper vector
similarity search over ALL story content.

Architecture:
    StoryGraph nodes/edges/memories
         |
         v
    EmbeddingRecall.index()   <-- embeds everything into GraphMemory (SQLite)
         |
         v
    EmbeddingRecall.recall()  <-- hybrid search (keyword + cosine similarity)
         |
         v
    ContextAssembler stage 4  <-- uses these results instead of keyword-only

The embedding model chain (from ADK's EmbeddingProvider):
    1. sentence-transformers (GPU/CPU, best quality)
    2. Ollama nomic-embed-text (local, no extra deps)
    3. Elysium gateway (cloud fallback)
    4. Feature hashing (zero deps, always works)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("saga.embedding_recall")

# Lazy-loaded singletons
_graph_memory = None
_embedding_provider = None


def _get_graph_memory(data_dir: Optional[Path] = None):
    """Get or create the GraphMemory instance for story content."""
    global _graph_memory
    if _graph_memory is not None:
        return _graph_memory

    from adk.graph_memory import GraphMemory

    db_path = (data_dir or Path.home() / ".saga" / "active_project") / "story_knowledge.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _graph_memory = GraphMemory(
        db_path=db_path,
        agent_name="saga",
    )
    logger.info("Story knowledge graph initialized: %s", db_path)
    return _graph_memory


def _get_embedding_provider():
    """Get or create the EmbeddingProvider."""
    global _embedding_provider
    if _embedding_provider is not None:
        return _embedding_provider

    from adk.faculties.embeddings import get_embedding_provider
    _embedding_provider = get_embedding_provider()
    return _embedding_provider


class EmbeddingRecall:
    """Semantic recall over story content using ADK's embedding infrastructure.

    Indexes all StoryGraph nodes and memories as embeddings in GraphMemory,
    then provides hybrid (keyword + vector) search for the ContextAssembler.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir
        self._indexed_node_ids: set = set()
        self._indexed_memory_ids: set = set()
        self._last_index_turn: int = -1

    @property
    def graph_mem(self):
        return _get_graph_memory(self.data_dir)

    @property
    def embedder(self):
        return _get_embedding_provider()

    async def index_story_content(self, story_graph, memory_manager, force: bool = False):
        """Index all story nodes and memories into the embedding store.

        Called once per turn (or on first load). Only indexes new/changed content.

        Args:
            story_graph: The StoryGraph instance
            memory_manager: The MemoryManager instance
            force: Re-index everything even if already indexed
        """
        if force:
            self._indexed_node_ids.clear()
            self._indexed_memory_ids.clear()

        indexed = 0

        # Index nodes
        for node in story_graph.get_all_nodes():
            if node.id in self._indexed_node_ids:
                continue

            content = f"{node.name}: {node.description}"
            if node.short_description:
                content = f"{node.name}: {node.short_description}. {node.description}"

            # Add properties as searchable content
            for key, val in node.properties.items():
                if isinstance(val, str) and len(val) > 5:
                    content += f" [{key}: {val}]"

            await self.graph_mem.add(
                label=node.name,
                content=content,
                node_type=f"story_{node.type.value}",
                tags=node.tags + [node.type.value],
                importance=0.8 if node.pinned else 0.5,
                metadata={"story_node_id": node.id, "node_type": node.type.value},
            )
            self._indexed_node_ids.add(node.id)
            indexed += 1

        # Index memories
        for mem in memory_manager.get_all():
            if mem.id in self._indexed_memory_ids:
                continue

            await self.graph_mem.add(
                label=mem.summary or mem.content[:80],
                content=mem.content,
                node_type=f"story_memory_{mem.type.value}",
                tags=mem.tags + [mem.type.value],
                importance=mem.importance,
                metadata={
                    "story_memory_id": mem.id,
                    "memory_type": mem.type.value,
                    "turn_number": mem.turn_number,
                    "pinned": mem.pinned,
                },
            )
            self._indexed_memory_ids.add(mem.id)
            indexed += 1

        # Index edges as relationship facts
        for edge in story_graph.get_all_edges():
            source = story_graph.get_node(edge.source_id)
            target = story_graph.get_node(edge.target_id)
            if source and target:
                await self.graph_mem.remember(
                    source.name, edge.type.value, target.name
                )

        if indexed > 0:
            logger.info("Indexed %d items into story knowledge graph", indexed)

    async def semantic_recall(
        self,
        query: str,
        activated_node_ids: List[str] = None,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """Semantic search over all indexed story content.

        Returns results ranked by hybrid score (keyword + cosine similarity).
        This is what the ContextAssembler's RECALL stage should use.

        Args:
            query: The user's input text
            activated_node_ids: Already-activated node IDs (for boosting)
            limit: Max results

        Returns:
            List of dicts with: content, score, reason, metadata
        """
        results = await self.graph_mem.search(query, limit=limit * 2)

        # Format for context assembler consumption
        recall_results = []
        activated = set(activated_node_ids or [])

        for node in results[:limit]:
            score = node.importance
            reason_parts = []

            # Boost if related to already-activated nodes
            meta = node.metadata or {}
            story_node_id = meta.get("story_node_id", "")
            if story_node_id and story_node_id in activated:
                score += 0.2
                reason_parts.append("related to active context")

            if meta.get("pinned"):
                score += 0.15
                reason_parts.append("pinned")

            reason_parts.append(f"semantic match (score={score:.2f})")

            recall_results.append({
                "content": node.content,
                "summary": node.label,
                "score": min(score, 1.0),
                "reason": "; ".join(reason_parts),
                "type": node.node_type,
                "metadata": meta,
            })

        recall_results.sort(key=lambda x: x["score"], reverse=True)
        return recall_results[:limit]

    async def get_related_context(self, entity_name: str, depth: int = 2) -> List[Dict]:
        """Multi-hop traversal — get everything related to an entity."""
        related = await self.graph_mem.get_related(entity_name, depth=depth)
        return [
            {
                "name": node.label,
                "content": node.content,
                "type": node.node_type,
                "importance": node.importance,
            }
            for node in related
        ]

    async def ingest_story_turn(
        self,
        user_input: str,
        saga_response: str,
        turn_number: int,
        session_id: str = "",
    ):
        """Ingest a completed story turn for future recall.

        Extracts entities and relations from the narrative and stores them
        as searchable graph nodes with embeddings.
        """
        # Combine for entity extraction
        full_text = f"Turn {turn_number}: Player: {user_input}\nNarrator: {saga_response}"

        await self.graph_mem.ingest_conversation(
            session_id=session_id or f"saga-turn-{turn_number}",
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": saga_response},
            ],
        )

    @property
    def stats(self) -> Dict:
        return {
            "indexed_nodes": len(self._indexed_node_ids),
            "indexed_memories": len(self._indexed_memory_ids),
            "embedding_backend": self.embedder.stats.get("backend", "pending"),
        }
