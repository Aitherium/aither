"""
Memory Manager — Story Memory CRUD + Recall
=============================================

Manages the story's memory store — episodic events, semantic facts,
procedural rules, and emotional sentiments.

Features:
    - Store new memories (with auto-summary)
    - Recall by related nodes, text similarity, tags
    - Importance scoring with time decay
    - Access tracking (frequently recalled memories stay important)
    - Pinning (user can pin memories to prevent decay)
    - JSON persistence
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import MemoryType, StoryMemory, _make_id, _now

logger = logging.getLogger("Saga.MemoryManager")


class MemoryManager:
    """
    Story memory store with recall scoring.

    Memories are scored by:
        1. Importance (user-set, 0-1)
        2. Relatedness (connected to activated nodes)
        3. Recency (time decay reduces importance)
        4. Access frequency (frequently recalled = more important)
        5. Text relevance (keyword overlap with query)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self._memories: Dict[str, StoryMemory] = {}
        self._data_dir = data_dir
        if data_dir:
            data_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # CRUD
    # ========================================================================

    def store(self, memory: StoryMemory) -> StoryMemory:
        """Store a new memory."""
        if not memory.summary and memory.content:
            # Auto-generate summary from first sentence
            memory.summary = self._auto_summary(memory.content)
        self._memories[memory.id] = memory
        logger.debug(f"Stored memory: [{memory.type.value}] {memory.summary}")
        return memory

    def create(
        self,
        memory_type: MemoryType,
        content: str,
        summary: str = "",
        importance: float = 0.5,
        related_nodes: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        story_time: Optional[str] = None,
        turn_number: int = 0,
        created_by: str = "system",
        pinned: bool = False,
    ) -> StoryMemory:
        """Create and store a new memory."""
        mem = StoryMemory(
            type=memory_type,
            content=content,
            summary=summary or self._auto_summary(content),
            importance=importance,
            related_nodes=related_nodes or [],
            tags=tags or [],
            story_time=story_time,
            turn_number=turn_number,
            created_by=created_by,
            pinned=pinned,
        )
        return self.store(mem)

    def get(self, memory_id: str) -> Optional[StoryMemory]:
        """Get a memory by ID."""
        return self._memories.get(memory_id)

    def update(self, memory_id: str, **updates) -> StoryMemory:
        """Update a memory's fields."""
        mem = self._memories.get(memory_id)
        if not mem:
            raise KeyError(f"Memory {memory_id} not found")
        for key, value in updates.items():
            if hasattr(mem, key):
                setattr(mem, key, value)
        return mem

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        return self._memories.pop(memory_id, None) is not None

    def get_all(self) -> List[StoryMemory]:
        """Return all memories."""
        return list(self._memories.values())

    def get_by_type(self, memory_type: MemoryType) -> List[StoryMemory]:
        """Get all memories of a given type."""
        return [m for m in self._memories.values() if m.type == memory_type]

    def get_by_node(self, node_id: str) -> List[StoryMemory]:
        """Get all memories related to a specific node."""
        return [m for m in self._memories.values() if node_id in m.related_nodes]

    def search(self, query: str) -> List[StoryMemory]:
        """Search memories by content/summary substring."""
        q = query.lower()
        return [
            m for m in self._memories.values()
            if q in m.content.lower() or q in m.summary.lower()
        ]

    @property
    def count(self) -> int:
        return len(self._memories)

    # ========================================================================
    # RECALL — The smart retrieval
    # ========================================================================

    def recall(
        self,
        related_node_ids: Optional[List[str]] = None,
        query_text: str = "",
        tags: Optional[List[str]] = None,
        limit: int = 10,
        current_turn: int = 0,
        min_importance: float = 0.05,
    ) -> List[Tuple[StoryMemory, str]]:
        """
        Recall relevant memories with scoring and reason attribution.

        Returns: [(memory, reason_string), ...] sorted by relevance.
        """
        related_ids = set(related_node_ids or [])
        tag_set = set(tags or [])
        query_words = set(re.findall(r'\b\w{3,}\b', query_text.lower())) if query_text else set()

        scored: List[Tuple[StoryMemory, float, str]] = []

        for mem in self._memories.values():
            score = 0.0
            reasons = []

            # 1. Base importance (with decay)
            eff_importance = mem.effective_importance(current_turn)
            if eff_importance < min_importance:
                continue
            score += eff_importance * 0.3
            reasons.append(f"importance={eff_importance:.2f}")

            # 2. Node relatedness
            if related_ids:
                overlap = related_ids & set(mem.related_nodes)
                if overlap:
                    node_score = len(overlap) / max(len(related_ids), 1)
                    score += node_score * 0.4
                    reasons.append(f"related to {len(overlap)} active node(s)")

            # 3. Tag match
            if tag_set:
                tag_overlap = tag_set & set(t.lower() for t in mem.tags)
                if tag_overlap:
                    score += 0.15
                    reasons.append(f"tag match: {', '.join(tag_overlap)}")

            # 4. Text relevance (keyword overlap)
            if query_words:
                mem_words = set(re.findall(
                    r'\b\w{3,}\b', (mem.content + " " + mem.summary).lower()
                ))
                word_overlap = query_words & mem_words
                if word_overlap:
                    text_score = len(word_overlap) / max(len(query_words), 1)
                    score += text_score * 0.15
                    reasons.append(f"text match: {len(word_overlap)} keyword(s)")

            # 5. Access frequency bonus (well-trodden memories are valuable)
            if mem.access_count > 0:
                freq_bonus = min(0.1, mem.access_count * 0.02)
                score += freq_bonus

            # 6. Pinned bonus
            if mem.pinned:
                score += 0.2
                reasons.append("pinned")

            if score > 0.05:
                reason = "; ".join(reasons)
                scored.append((mem, score, reason))

        # Sort by score, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(mem, reason) for mem, _, reason in scored[:limit]]

    def mark_accessed(self, memory_id: str):
        """Mark a memory as accessed (updates access count and timestamp)."""
        mem = self._memories.get(memory_id)
        if mem:
            mem.access_count += 1
            mem.last_accessed = _now()

    # ========================================================================
    # MAINTENANCE
    # ========================================================================

    def decay_all(self, current_turn: int, threshold: float = 0.02):
        """
        Check for memories that have decayed below threshold.
        Doesn't delete them, but logs them for potential cleanup.
        Returns list of near-forgotten memory IDs.
        """
        fading = []
        for mem in self._memories.values():
            if not mem.pinned:
                eff = mem.effective_importance(current_turn)
                if eff < threshold:
                    fading.append(mem.id)
        return fading

    def consolidate(self, node_id: str, current_turn: int) -> Optional[StoryMemory]:
        """
        If there are many low-importance memories about a node,
        consolidate them into one summary memory.
        Returns the new consolidated memory or None.
        """
        related = self.get_by_node(node_id)
        low_importance = [
            m for m in related
            if not m.pinned and m.effective_importance(current_turn) < 0.2
        ]

        if len(low_importance) < 5:
            return None

        # Merge into one
        summaries = [m.summary for m in low_importance]
        merged_content = "Consolidated memories: " + " | ".join(summaries)
        merged_summary = f"Various events involving {len(low_importance)} related occurrences"

        consolidated = self.create(
            memory_type=MemoryType.EPISODIC,
            content=merged_content,
            summary=merged_summary,
            importance=0.4,
            related_nodes=[node_id],
            turn_number=current_turn,
            created_by="system:consolidation",
        )

        # Remove the originals
        for m in low_importance:
            self.delete(m.id)

        logger.info(
            f"Consolidated {len(low_importance)} memories about node {node_id} → {consolidated.id}"
        )
        return consolidated

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _auto_summary(content: str) -> str:
        """Generate a one-line summary from content."""
        # Take first sentence or first 120 chars
        sentences = re.split(r'[.!?]\s', content)
        if sentences:
            first = sentences[0].strip()
            if len(first) <= 120:
                return first
            return first[:117] + "..."
        return content[:120]

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _memory_path(self) -> Path:
        assert self._data_dir, "No data_dir configured"
        return self._data_dir / "memories.json"

    def save(self):
        """Persist memories to disk."""
        if not self._data_dir:
            return
        data = {mid: m.model_dump() for mid, m in self._memories.items()}
        self._memory_path().write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Memories saved: {self.count} → {self._memory_path()}")

    def load(self) -> bool:
        """Load memories from disk."""
        if not self._data_dir:
            return False
        path = self._memory_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self._memories = {
                mid: StoryMemory(**mdata) for mid, mdata in data.items()
            }
            logger.info(f"Memories loaded: {self.count} ← {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Memory statistics."""
        type_counts = {}
        for mt in MemoryType:
            count = len(self.get_by_type(mt))
            if count > 0:
                type_counts[mt.value] = count

        pinned = len([m for m in self._memories.values() if m.pinned])
        return {
            "total": self.count,
            "by_type": type_counts,
            "pinned": pinned,
        }
