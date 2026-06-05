"""
Memory Consolidation — Deduplicate, Promote, Never Lose
=========================================================

Periodically processes the memory tiers to:

    1. DEDUPLICATE — Merge near-duplicate memories into consolidated entries
       (originals archived in tier 4, not deleted)

    2. PROMOTE — Move important recurring patterns from tier 1 (in-story) to
       tier 4 (cross-session knowledge graph) so they survive project switches

    3. CRYSTALLIZE — Convert scattered episodic memories about an entity into
       a single semantic fact (e.g., 10 memories about "Kael fought" → 1 fact
       "Kael is a skilled warrior")

    4. ARCHIVE — Low-importance memories decay but are never deleted. They get
       moved to an archive tier with minimal footprint. They can still be
       recalled if specifically searched for.

The golden rule: NOTHING IS EVER TRULY LOST.
Every piece of world detail, every narrative moment, every character trait
the LLM generated — it all stays in the graph somewhere. Consolidation
compresses it, but the raw data remains in the archive.

This runs:
    - After every 10 story turns (automatic)
    - On project save
    - On explicit user request
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import MemoryType, StoryMemory, _now

logger = logging.getLogger("saga.consolidation")

# Similarity threshold for deduplication (keyword overlap ratio)
DEDUP_THRESHOLD = 0.65
# Max memories per entity before consolidation triggers
CONSOLIDATION_TRIGGER = 8
# Min importance for promotion to cross-session
PROMOTION_THRESHOLD = 0.6
# Turns between auto-consolidation runs
AUTO_CONSOLIDATION_INTERVAL = 10


def _tokenize(text: str) -> Set[str]:
    """Extract meaningful words from text for similarity comparison."""
    words = set(re.findall(r'\b\w{3,}\b', text.lower()))
    # Remove very common narrative words
    stopwords = {"the", "and", "was", "were", "had", "has", "have", "been", "that",
                 "this", "with", "from", "they", "their", "into", "then", "said",
                 "could", "would", "will", "just", "like", "back", "over"}
    return words - stopwords


def _similarity(text_a: str, text_b: str) -> float:
    """Keyword overlap similarity between two texts."""
    words_a = _tokenize(text_a)
    words_b = _tokenize(text_b)
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    union = len(words_a | words_b)
    return overlap / union if union > 0 else 0.0


class MemoryConsolidator:
    """Processes memory tiers for deduplication, promotion, and archival."""

    def __init__(self, memory_manager, story_graph, data_dir: Optional[Path] = None):
        self.memory = memory_manager
        self.graph = story_graph
        self.data_dir = data_dir or Path.home() / ".saga" / "active_project"
        self._archive_path = self.data_dir / "memory_archive.jsonl"
        self._last_consolidation_turn = 0

    def should_auto_consolidate(self, current_turn: int) -> bool:
        """Check if automatic consolidation should run."""
        return (current_turn - self._last_consolidation_turn) >= AUTO_CONSOLIDATION_INTERVAL

    def consolidate_all(self, current_turn: int) -> Dict:
        """Run the full consolidation pipeline. Returns stats."""
        stats = {
            "deduplicated": 0,
            "consolidated": 0,
            "promoted": 0,
            "archived": 0,
            "turn": current_turn,
        }

        # 1. Deduplicate
        dedup_count = self._deduplicate(current_turn)
        stats["deduplicated"] = dedup_count

        # 2. Consolidate per-entity
        consol_count = self._consolidate_by_entity(current_turn)
        stats["consolidated"] = consol_count

        # 3. Promote important patterns
        promo_count = self._promote_to_knowledge(current_turn)
        stats["promoted"] = promo_count

        # 4. Archive decayed memories
        archive_count = self._archive_decayed(current_turn)
        stats["archived"] = archive_count

        self._last_consolidation_turn = current_turn
        self.memory.save()

        if any(v > 0 for k, v in stats.items() if k != "turn"):
            logger.info(
                "Consolidation complete: %d deduped, %d consolidated, %d promoted, %d archived",
                dedup_count, consol_count, promo_count, archive_count,
            )

        return stats

    # ========================================================================
    # STAGE 1: DEDUPLICATE
    # ========================================================================

    def _deduplicate(self, current_turn: int) -> int:
        """Find and merge near-duplicate memories."""
        all_mems = self.memory.get_all()
        if len(all_mems) < 3:
            return 0

        # Group by type for faster comparison
        by_type: Dict[str, List[StoryMemory]] = defaultdict(list)
        for mem in all_mems:
            by_type[mem.type.value].append(mem)

        merged_count = 0
        ids_to_remove: Set[str] = set()

        for mem_type, mems in by_type.items():
            # Compare pairs within each type
            for i, mem_a in enumerate(mems):
                if mem_a.id in ids_to_remove:
                    continue
                for mem_b in mems[i + 1:]:
                    if mem_b.id in ids_to_remove:
                        continue
                    if mem_a.pinned != mem_b.pinned:
                        continue  # Don't merge pinned with unpinned

                    sim = _similarity(mem_a.content, mem_b.content)
                    if sim >= DEDUP_THRESHOLD:
                        # Keep the one with higher importance (or newer if equal)
                        keeper, loser = (
                            (mem_a, mem_b)
                            if mem_a.importance >= mem_b.importance
                            else (mem_b, mem_a)
                        )

                        # Archive the loser (NEVER truly delete)
                        self._archive_memory(loser, reason=f"deduplicated into {keeper.id}")
                        ids_to_remove.add(loser.id)

                        # Boost keeper's importance slightly (it covers more ground)
                        new_importance = min(1.0, keeper.importance + 0.05)
                        self.memory.update(keeper.id, importance=new_importance)

                        # Merge related_nodes
                        merged_nodes = list(set(keeper.related_nodes + loser.related_nodes))
                        self.memory.update(keeper.id, related_nodes=merged_nodes)

                        merged_count += 1

        # Remove merged memories from active store
        for mid in ids_to_remove:
            self.memory.delete(mid)

        return merged_count

    # ========================================================================
    # STAGE 2: CONSOLIDATE BY ENTITY
    # ========================================================================

    def _consolidate_by_entity(self, current_turn: int) -> int:
        """Consolidate many low-importance memories about a single entity."""
        # Find entities with many memories
        node_memory_count: Dict[str, int] = defaultdict(int)
        for mem in self.memory.get_all():
            for nid in mem.related_nodes:
                node_memory_count[nid] += 1

        consolidated = 0
        for node_id, count in node_memory_count.items():
            if count >= CONSOLIDATION_TRIGGER:
                result = self.memory.consolidate(node_id, current_turn)
                if result:
                    consolidated += 1

        return consolidated

    # ========================================================================
    # STAGE 3: PROMOTE TO CROSS-SESSION KNOWLEDGE
    # ========================================================================

    def _promote_to_knowledge(self, current_turn: int) -> int:
        """Promote important/frequently accessed memories to cross-session graph."""
        try:
            from .persistent_memory import get_graph_memory
            gm = get_graph_memory()
        except ImportError:
            return 0

        promoted = 0
        for mem in self.memory.get_all():
            eff_importance = mem.effective_importance(current_turn)

            # Promote if: high importance, or frequently accessed, or pinned
            should_promote = (
                eff_importance >= PROMOTION_THRESHOLD
                or mem.access_count >= 5
                or mem.pinned
            )

            if not should_promote:
                continue

            # Check if already promoted (use hash of content as dedup key)
            content_hash = hashlib.md5(mem.content.encode()).hexdigest()[:12]
            existing = None
            try:
                import asyncio
                existing = asyncio.run(gm.search(mem.summary, limit=1))
            except Exception:
                pass

            if existing and _similarity(existing[0].content, mem.content) > 0.8:
                continue  # Already in knowledge graph

            # Promote
            try:
                import asyncio
                asyncio.run(gm.add(
                    label=mem.summary,
                    content=mem.content,
                    node_type=f"promoted_{mem.type.value}",
                    tags=mem.tags + ["promoted", mem.type.value],
                    importance=eff_importance,
                    metadata={
                        "source_memory_id": mem.id,
                        "source_turn": mem.turn_number,
                        "promoted_at_turn": current_turn,
                        "content_hash": content_hash,
                    },
                ))
                promoted += 1
            except Exception as e:
                logger.debug("Promotion failed for %s: %s", mem.id, e)

        return promoted

    # ========================================================================
    # STAGE 4: ARCHIVE DECAYED (NEVER DELETE)
    # ========================================================================

    def _archive_decayed(self, current_turn: int) -> int:
        """Move decayed memories to archive file. They're NOT deleted — just
        removed from the active hot store to keep context assembly fast.
        They remain searchable via the archive.
        """
        fading_ids = self.memory.decay_all(current_turn, threshold=0.03)
        archived = 0

        for mid in fading_ids:
            mem = self.memory.get(mid)
            if mem and not mem.pinned:
                self._archive_memory(mem, reason=f"decayed below threshold at turn {current_turn}")
                self.memory.delete(mid)
                archived += 1

        return archived

    # ========================================================================
    # ARCHIVE PERSISTENCE (append-only JSONL — never loses data)
    # ========================================================================

    def _archive_memory(self, mem: StoryMemory, reason: str = ""):
        """Append a memory to the archive file. This is the safety net —
        no memory is ever truly deleted, just moved here."""
        import json

        self._archive_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "archived_at": _now(),
            "reason": reason,
            "memory": mem.model_dump(),
        }

        with open(self._archive_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def search_archive(self, query: str, limit: int = 10) -> List[Dict]:
        """Search the archive for old/consolidated memories."""
        import json

        if not self._archive_path.exists():
            return []

        query_lower = query.lower()
        results = []

        with open(self._archive_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    mem_data = entry.get("memory", {})
                    content = mem_data.get("content", "") + " " + mem_data.get("summary", "")
                    if query_lower in content.lower():
                        results.append(entry)
                        if len(results) >= limit:
                            break
                except json.JSONDecodeError:
                    continue

        return results

    @property
    def archive_count(self) -> int:
        """Count of archived memories."""
        if not self._archive_path.exists():
            return 0
        with open(self._archive_path, "r") as f:
            return sum(1 for line in f if line.strip())
