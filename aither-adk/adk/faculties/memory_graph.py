#!/usr/bin/env python3
"""
MemoryGraph - Graph-Based Memory for ADK
==========================================

Graph structure over Memory objects: real relationship edges,
hybrid query (keyword + semantic + graph expansion), and associative
multi-hop recall.

Mirrors CodeGraph's architecture:
- MemoryNode wraps Memory (composition, not inheritance)
- Indexed by tag, type, agent for sublinear lookups
- Persistence via pickle (edges + embeddings separate)
- Thread-safe singleton via get_memorygraph()

Adapted from AitherOS MemoryGraph for the Aither ADK.
Version: 1.0.0
"""

import hashlib
import hmac as _hmac
import logging
import math
import os
import pickle
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from adk.faculties.base import BaseFacultyGraph, GraphSyncConfig

logger = logging.getLogger("adk.faculties.memory_graph")


# ============================================================================
# CONFIGURATION
# ============================================================================

_SPIRIT_DIR = Path(
    os.environ.get("AITHER_DATA_DIR", os.path.expanduser("~/.aither"))
) / "memory"

_GRAPH_PICKLE = _SPIRIT_DIR / "memory_graph.pkl"
_EMBEDDINGS_PICKLE = _SPIRIT_DIR / "memory_embeddings.pkl"


def _mg_pickle_hmac_key() -> bytes:
    """Get the HMAC key for pickle integrity validation."""
    secret = os.environ.get("AITHER_INTERNAL_SECRET", "")
    if not secret:
        logger.warning(
            "AITHER_INTERNAL_SECRET not set — using default pickle HMAC key. "
            "Set this env var in production!"
        )
        secret = "aither-pickle-hmac-default"
    return secret.encode()


def _mg_compute_file_hmac(filepath: str) -> str:
    """Compute HMAC-SHA256 of a file's contents."""
    with open(filepath, "rb") as f:
        data = f.read()
    return _hmac.new(_mg_pickle_hmac_key(), data, hashlib.sha256).hexdigest()


def _mg_verify_pickle_hmac(filepath: str) -> bool:
    """Verify HMAC sidecar for a pickle file. Returns True if valid or legacy."""
    hmac_path = filepath + ".hmac" if isinstance(filepath, str) else str(filepath) + ".hmac"
    filepath_str = str(filepath)
    if not os.path.exists(hmac_path):
        logger.warning(
            "[MemoryGraph] No HMAC sidecar for %s — refusing to load (rebuild required)",
            filepath_str,
        )
        return False
    try:
        with open(hmac_path, "r") as f:
            stored = f.read().strip()
        computed = _mg_compute_file_hmac(filepath_str)
        if not _hmac.compare_digest(stored, computed):
            logger.error(
                f"[MemoryGraph] HMAC mismatch for {filepath_str} — cache tampered, deleting"
            )
            os.unlink(filepath_str)
            os.unlink(hmac_path)
            return False
        return True
    except Exception as e:
        logger.error("[MemoryGraph] HMAC verification error for %s: %s — refusing to load", filepath_str, e)
        return False


def _mg_write_pickle_hmac(filepath: str) -> None:
    """Write HMAC sidecar after saving a pickle file."""
    filepath_str = str(filepath)
    try:
        hmac_val = _mg_compute_file_hmac(filepath_str)
        with open(filepath_str + ".hmac", "w") as f:
            f.write(hmac_val)
    except Exception as e:
        logger.warning(f"[MemoryGraph] Failed to write HMAC sidecar: {e}")


# Edge detection thresholds
_TAG_OVERLAP_MIN = 2             # Shared tags for TAG_SIBLING
_TEMPORAL_WINDOW_SECS = 300      # 5 minutes for temporal edges
_SIMILARITY_THRESHOLD = 0.7      # Embedding similarity for RELATED
_MAX_RELATED_PER_NODE = 5        # Cap auto-detected RELATED edges
_MAX_CANDIDATES_FOR_SIM = 50     # Cap candidates for embedding scan


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class EdgeType(str, Enum):
    """Types of relationships between memories."""
    DERIVED_FROM = "derived_from"       # B was created because of A
    SUPERSEDES = "supersedes"           # B replaces/updates A (upsert)
    RELATED = "related"                 # Embedding similarity > threshold
    TAG_SIBLING = "tag_sibling"         # Share 2+ tags
    SAME_AGENT = "same_agent"           # Same source_agent within temporal window
    SAME_SESSION = "same_session"       # Same source_session
    TEMPORAL = "temporal"               # Created within temporal window
    REINFORCED_BY = "reinforced_by"     # Co-accessed in same recall
    PART_OF = "part_of"                 # Memory is part of a procedure/composite
    ELABORATES = "elaborates"           # Memory expands on another


@dataclass
class MemoryEdge:
    """A directed edge between two memory nodes."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0         # 0.0-1.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEdge":
        data = dict(data)
        data["edge_type"] = EdgeType(data["edge_type"])
        return cls(**data)


@dataclass
class MemoryNode:
    """Wraps an existing Memory object with graph connectivity.

    The ``memory`` field accepts any object with ``id``, ``content``,
    ``title``, ``tags``, etc. attributes.  Using ``Any`` avoids import
    cycles and allows ADK users to pass arbitrary memory-like objects.
    """
    memory: Any
    outgoing_edges: List[str] = field(default_factory=list)   # Edge IDs
    incoming_edges: List[str] = field(default_factory=list)   # Edge IDs
    neighbors: List[str] = field(default_factory=list)        # Node IDs (pre-computed)

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def embedding(self) -> Optional[list]:
        return getattr(self.memory, "embedding", None)


# ============================================================================
# EMBEDDING + SIMILARITY UTILITIES
# ============================================================================

def _cosine_similarity(v1: Optional[list], v2: Optional[list]) -> float:
    """Cosine similarity. Uses NumPy BLAS when available (~10x faster)."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    try:
        import numpy as np
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        d = np.dot(a, b)
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(d / n) if n > 0 else 0.0
    except ImportError:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def _get_embedding(text: str) -> Optional[list]:
    """Get embedding via adk.faculties.embeddings provider."""
    try:
        from adk.faculties.embeddings import get_embedding_provider
        import asyncio
        provider = get_embedding_provider()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, provider.embed(text))
                result = future.result(timeout=10)
        else:
            result = asyncio.run(provider.embed(text))
        return result
    except Exception:
        return None


# ============================================================================
# QUERY CLASSIFIER (adapted for memories, not code)
# ============================================================================

# Pre-compiled patterns
_RE_IDENTITY = re.compile(r"\b(who|what)\s+(is|am|are)\b|my\s+name|about\s+me", re.I)
_RE_PROCEDURAL = re.compile(r"\b(how\s+(do|to|can)|steps?\s+to|procedure|workflow|process)\b", re.I)
_RE_SPECIFIC = re.compile(r'"[^"]+"|\b[A-Z][a-z]+[A-Z]\w+\b')  # Quoted or CamelCase (case-sensitive)
_RE_CONCEPTUAL = re.compile(r"\b(related\s+to|connections?\s+to|link|associated|concept)\b", re.I)
_RE_EXPLORATORY = re.compile(r"\b(what\s+do\s+I\s+know|everything\s+about|all\s+about|recall\s+about)\b", re.I)


def classify_query(query: str) -> Tuple[float, float, str]:
    """
    Classify a memory query and return (keyword_weight, semantic_weight, category).

    Categories:
        identity    -> kw=0.9, sem=0.1  ("what is my name", "who is")
        procedural  -> kw=0.6, sem=0.4  ("how do I", "steps to")
        specific    -> kw=0.8, sem=0.2  (quoted strings, exact titles)
        conceptual  -> kw=0.2, sem=0.8  ("related to", "connections to")
        exploratory -> kw=0.3, sem=0.7  ("what do I know about")
        balanced    -> kw=0.4, sem=0.6  (default)
    """
    q = query.strip()

    if _RE_IDENTITY.search(q):
        return 0.9, 0.1, "identity"
    if _RE_PROCEDURAL.search(q):
        return 0.6, 0.4, "procedural"
    if _RE_CONCEPTUAL.search(q):
        return 0.2, 0.8, "conceptual"
    if _RE_EXPLORATORY.search(q):
        return 0.3, 0.7, "exploratory"
    if _RE_SPECIFIC.search(q):
        return 0.8, 0.2, "specific"
    return 0.4, 0.6, "balanced"


# ============================================================================
# MEMORY GRAPH
# ============================================================================

class MemoryGraph(BaseFacultyGraph):
    """
    Graph structure over Memory objects.

    Three-layer index for sublinear lookups:
    - by_tag:   tag string  -> list of node IDs
    - by_type:  memory_type -> list of node IDs
    - by_agent: agent name  -> list of node IDs

    Edges are stored in a flat dict keyed by edge ID, with secondary
    indexes by source and target for fast traversal.
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__()
        if data_dir:
            self._data_dir = Path(os.path.expanduser(data_dir))
        else:
            self._data_dir = _SPIRIT_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._graph_pickle = self._data_dir / "memory_graph.pkl"
        self._embeddings_pickle = self._data_dir / "memory_embeddings.pkl"

        self._sync_config = GraphSyncConfig(
            enabled=True,
            domain="memory",
            source_graph="MemoryGraph",
            batch_size=20,
        )
        # Node storage
        self.nodes: Dict[str, MemoryNode] = {}

        # Indexes
        self.by_tag: Dict[str, List[str]] = defaultdict(list)
        self.by_type: Dict[str, List[str]] = defaultdict(list)
        self.by_agent: Dict[str, List[str]] = defaultdict(list)

        # -- Inverted keyword index (CodeGraph pattern) --
        # Maps keyword -> set of node_ids containing that keyword
        # in title, content, or tags. Updated on add_node.
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)

        # -- Query result cache (LRU, max 64) --
        self._query_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._query_cache_order: List[str] = []
        self._QUERY_CACHE_MAX = 64

        # Edge storage
        self.edges: Dict[str, MemoryEdge] = {}
        self.edges_by_source: Dict[str, List[str]] = defaultdict(list)
        self.edges_by_target: Dict[str, List[str]] = defaultdict(list)

        # Embedding cache (separate from Memory.embedding for persistence)
        self._embedding_cache: Dict[str, list] = {}

        # Stop words to skip in keyword indexing
        self._STOP_WORDS = frozenset({
            "a", "an", "the", "is", "it", "of", "to", "in", "for", "on",
            "and", "or", "not", "with", "this", "that", "from", "by", "at",
            "be", "as", "are", "was", "were", "been", "do", "does", "did",
            "i", "me", "my", "we", "our", "you", "your", "he", "she",
            "how", "what", "when", "where", "who", "which", "why",
        })

    # ------------------------------------------------------------------
    # NODE MANAGEMENT
    # ------------------------------------------------------------------

    def add_node(self, memory: Any, upsert: bool = False) -> MemoryNode:
        """
        Wrap a Memory into a MemoryNode, update indexes, auto-detect edges.

        Args:
            memory: Any object with at least an ``id`` attribute.
                    Typically also has ``content``, ``title``, ``tags``,
                    ``memory_type``, ``embedding``, etc.
            upsert: If True and ID already exists, add SUPERSEDES edge
        """
        mid = memory.id
        old_node = self.nodes.get(mid)

        if old_node and upsert:
            # Upsert: create new version, link SUPERSEDES
            # Remove old node from indexes (will re-add below)
            self._remove_from_indexes(mid)
            # Preserve old edge connectivity
            node = MemoryNode(
                memory=memory,
                outgoing_edges=list(old_node.outgoing_edges),
                incoming_edges=list(old_node.incoming_edges),
                neighbors=list(old_node.neighbors),
            )
            self.nodes[mid] = node
            # Add SUPERSEDES edge (new version supersedes old snapshot)
            self._add_edge(mid, mid, EdgeType.SUPERSEDES, weight=1.0,
                           metadata={"upserted_at": time.time()})
        elif old_node:
            # Already exists, not upsert -- update memory ref, keep edges
            old_node.memory = memory
            self._remove_from_indexes(mid)
            node = old_node
        else:
            node = MemoryNode(memory=memory)
            self.nodes[mid] = node

        # Update indexes
        self._add_to_indexes(node)

        # Cache embedding
        if getattr(memory, "embedding", None):
            self._embedding_cache[mid] = memory.embedding

        # Auto-detect edges
        self._auto_detect_edges(node)

        # Sync to knowledge graph (fire-and-forget)
        mem = memory
        self._queue_sync({
            "id": mid,
            "name": getattr(mem, "title", "") or getattr(mem, "key", mid),
            "type": getattr(mem, "memory_type", "memory"),
            "properties": {
                "tags": list(getattr(mem, "tags", []) or []),
                "source_agent": getattr(mem, "source_agent", ""),
                "importance": getattr(mem, "importance", 0.5),
            },
        })

        return node

    def remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]

        # Remove all edges involving this node
        edge_ids = set(node.outgoing_edges + node.incoming_edges)
        for eid in edge_ids:
            self._remove_edge(eid)

        # Remove from indexes
        self._remove_from_indexes(node_id)

        # Remove node
        del self.nodes[node_id]
        self._embedding_cache.pop(node_id, None)

    def _add_to_indexes(self, node: MemoryNode):
        """Add node to secondary indexes including keyword index."""
        mid = node.id
        mem = node.memory

        # Type index
        mt = mem.memory_type.value if hasattr(getattr(mem, "memory_type", None), "value") else str(getattr(mem, "memory_type", "memory"))
        if mid not in self.by_type[mt]:
            self.by_type[mt].append(mid)

        # Tag index
        for tag in (getattr(mem, "tags", None) or []):
            tag_lower = tag.lower()
            if mid not in self.by_tag[tag_lower]:
                self.by_tag[tag_lower].append(mid)

        # Agent index
        agent = getattr(mem, "source_agent", "unknown") or "unknown"
        if mid not in self.by_agent[agent]:
            self.by_agent[agent].append(mid)

        # -- Inverted keyword index --
        self._index_memory_keywords(mid, mem)

    def _remove_from_indexes(self, node_id: str):
        """Remove node from secondary indexes including keyword index."""
        if node_id not in self.nodes:
            return
        mem = self.nodes[node_id].memory

        mt = getattr(mem, "memory_type", None)
        mt = mt.value if hasattr(mt, "value") else str(mt) if mt else "memory"
        if node_id in self.by_type.get(mt, []):
            self.by_type[mt].remove(node_id)

        for tag in (getattr(mem, "tags", None) or []):
            tag_lower = tag.lower()
            if node_id in self.by_tag.get(tag_lower, []):
                self.by_tag[tag_lower].remove(node_id)

        agent = getattr(mem, "source_agent", "unknown") or "unknown"
        if node_id in self.by_agent.get(agent, []):
            self.by_agent[agent].remove(node_id)

        # -- Remove from keyword index --
        self._deindex_memory_keywords(node_id, mem)
        self._invalidate_query_cache()

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract indexable keywords from text."""
        if not text:
            return set()
        words = text.lower().replace("_", " ").replace("-", " ").split()
        return {w for w in words if len(w) > 1 and w not in self._STOP_WORDS}

    def _index_memory_keywords(self, node_id: str, mem: Any):
        """Add a memory's keywords to the inverted index."""
        keywords = set()
        keywords.update(self._extract_keywords(getattr(mem, "title", "") or ""))
        content = getattr(mem, "content", "") or ""
        keywords.update(self._extract_keywords(content[:500]))  # Cap content scan
        for tag in (getattr(mem, "tags", []) or []):
            keywords.update(self._extract_keywords(tag))
        for kw in keywords:
            self._keyword_index[kw].add(node_id)

    def _deindex_memory_keywords(self, node_id: str, mem: Any):
        """Remove a memory's keywords from the inverted index."""
        keywords = set()
        keywords.update(self._extract_keywords(getattr(mem, "title", "") or ""))
        content = getattr(mem, "content", "") or ""
        keywords.update(self._extract_keywords(content[:500]))
        for tag in (getattr(mem, "tags", []) or []):
            keywords.update(self._extract_keywords(tag))
        for kw in keywords:
            self._keyword_index[kw].discard(node_id)

    def _invalidate_query_cache(self):
        """Clear the query result cache."""
        self._query_cache.clear()
        self._query_cache_order.clear()

    # ------------------------------------------------------------------
    # EDGE MANAGEMENT
    # ------------------------------------------------------------------

    def _make_edge_id(self, source_id: str, target_id: str, edge_type: EdgeType) -> str:
        """Deterministic edge ID to prevent duplicates."""
        raw = f"{source_id}:{target_id}:{edge_type.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Optional[MemoryEdge]:
        """Add an edge. Returns None if duplicate or missing nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        eid = self._make_edge_id(source_id, target_id, edge_type)
        if eid in self.edges:
            # Update weight if higher
            if weight > self.edges[eid].weight:
                self.edges[eid].weight = weight
            return self.edges[eid]

        edge = MemoryEdge(
            id=eid,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self.edges[eid] = edge

        # Update node edge lists
        src = self.nodes[source_id]
        tgt = self.nodes[target_id]
        src.outgoing_edges.append(eid)
        tgt.incoming_edges.append(eid)

        # Update neighbor pre-computation
        if target_id not in src.neighbors:
            src.neighbors.append(target_id)
        if source_id not in tgt.neighbors:
            tgt.neighbors.append(source_id)

        # Secondary edge indexes
        self.edges_by_source[source_id].append(eid)
        self.edges_by_target[target_id].append(eid)

        return edge

    def _remove_edge(self, edge_id: str):
        """Remove an edge by ID."""
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return

        # Clean node edge lists
        src = self.nodes.get(edge.source_id)
        tgt = self.nodes.get(edge.target_id)
        if src:
            if edge_id in src.outgoing_edges:
                src.outgoing_edges.remove(edge_id)
        if tgt:
            if edge_id in tgt.incoming_edges:
                tgt.incoming_edges.remove(edge_id)

        # Clean secondary indexes
        if edge_id in self.edges_by_source.get(edge.source_id, []):
            self.edges_by_source[edge.source_id].remove(edge_id)
        if edge_id in self.edges_by_target.get(edge.target_id, []):
            self.edges_by_target[edge.target_id].remove(edge_id)

        # Rebuild neighbors for affected nodes
        if src:
            src.neighbors = self._compute_neighbors(edge.source_id)
        if tgt:
            tgt.neighbors = self._compute_neighbors(edge.target_id)

    def _compute_neighbors(self, node_id: str) -> List[str]:
        """Recompute neighbor list from edges."""
        nbrs: Set[str] = set()
        for eid in self.edges_by_source.get(node_id, []):
            e = self.edges.get(eid)
            if e:
                nbrs.add(e.target_id)
        for eid in self.edges_by_target.get(node_id, []):
            e = self.edges.get(eid)
            if e:
                nbrs.add(e.source_id)
        return list(nbrs)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Optional[MemoryEdge]:
        """Public edge creation (for explicit edges like DERIVED_FROM, PART_OF)."""
        return self._add_edge(source_id, target_id, edge_type, weight, metadata)

    # ------------------------------------------------------------------
    # AUTO-EDGE DETECTION (sublinear)
    # ------------------------------------------------------------------

    def _auto_detect_edges(self, node: MemoryNode):
        """
        Detect edges for a new/updated node WITHOUT O(n) scan.

        Strategy:
        1. Tag index -> TAG_SIBLING (shared tags)
        2. Agent temporal -> SAME_AGENT (recent same-agent memories)
        3. Session -> SAME_SESSION
        4. Embedding similarity against candidates from 1-2 + same type
        """
        mid = node.id
        mem = node.memory
        now = getattr(mem, "created_at", time.time())
        if isinstance(now, (int, float)):
            created = now
        else:
            created = time.time()

        candidates: Set[str] = set()

        # 1. TAG SIBLING: scan tag index for nodes sharing 2+ tags
        tag_counts: Dict[str, int] = defaultdict(int)
        for tag in (getattr(mem, "tags", None) or []):
            tag_lower = tag.lower()
            for other_id in self.by_tag.get(tag_lower, []):
                if other_id != mid:
                    tag_counts[other_id] += 1

        for other_id, count in tag_counts.items():
            if count >= _TAG_OVERLAP_MIN:
                self._add_edge(mid, other_id, EdgeType.TAG_SIBLING,
                               weight=min(1.0, count / 5.0),
                               metadata={"shared_tags": count})
                candidates.add(other_id)

        # 2. SAME AGENT temporal: recent memories from same agent
        agent = getattr(mem, "source_agent", "unknown") or "unknown"
        for other_id in self.by_agent.get(agent, []):
            if other_id == mid:
                continue
            other_mem = self.nodes[other_id].memory
            other_created = getattr(other_mem, "created_at", 0)
            if isinstance(other_created, (int, float)) and abs(created - other_created) <= _TEMPORAL_WINDOW_SECS:
                self._add_edge(mid, other_id, EdgeType.SAME_AGENT, weight=0.6)
                candidates.add(other_id)

        # 3. SAME SESSION
        session = getattr(mem, "source_session", "") or ""
        if session:
            # Quick scan of same-agent nodes (session is per-agent typically)
            for other_id in self.by_agent.get(agent, []):
                if other_id == mid:
                    continue
                other_session = getattr(self.nodes[other_id].memory, "source_session", "")
                if other_session == session:
                    self._add_edge(mid, other_id, EdgeType.SAME_SESSION, weight=0.5)
                    candidates.add(other_id)

        # 4. EMBEDDING SIMILARITY: only against candidates + same-type neighbors
        node_emb = self._embedding_cache.get(mid) or node.embedding
        if node_emb:
            # Add same-type nodes to candidate pool (capped)
            mt = getattr(mem, "memory_type", None)
            mt = mt.value if hasattr(mt, "value") else str(mt) if mt else "memory"
            type_ids = self.by_type.get(mt, [])
            for tid in type_ids[:_MAX_CANDIDATES_FOR_SIM]:
                if tid != mid:
                    candidates.add(tid)

            # Score and keep top-N
            scored: List[Tuple[float, str]] = []
            for cid in candidates:
                c_emb = self._embedding_cache.get(cid)
                if not c_emb:
                    c_node = self.nodes.get(cid)
                    c_emb = c_node.embedding if c_node else None
                if c_emb:
                    sim = _cosine_similarity(node_emb, c_emb)
                    if sim >= _SIMILARITY_THRESHOLD:
                        scored.append((sim, cid))

            scored.sort(key=lambda x: -x[0])
            for sim, cid in scored[:_MAX_RELATED_PER_NODE]:
                self._add_edge(mid, cid, EdgeType.RELATED,
                               weight=round(sim, 3),
                               metadata={"similarity": round(sim, 3)})

    # ------------------------------------------------------------------
    # REINFORCEMENT EDGES (co-access tracking)
    # ------------------------------------------------------------------

    def record_co_access(self, node_ids: List[str]):
        """
        Record that these nodes were co-accessed in a single recall.
        Creates REINFORCED_BY edges between all pairs.
        """
        ids = [nid for nid in node_ids if nid in self.nodes]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                self._add_edge(a, b, EdgeType.REINFORCED_BY, weight=0.3)

    # ------------------------------------------------------------------
    # HYBRID QUERY
    # ------------------------------------------------------------------

    def hybrid_query(
        self,
        query: str,
        max_results: int = 10,
        min_strength: float = 0.1,
        memory_types: Optional[list] = None,
        agent_id: Optional[str] = None,
        scope: str = "shared",
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Hybrid keyword + semantic search with graph expansion.

        Pipeline:
        1. Check LRU cache
        2. classify_query() -> weights
        3. keyword_search via inverted index (not O(N) scan)
        4. semantic_search (cosine similarity)
        5. Weighted merge
        6. Graph expansion: 1-hop BFS for top-5, add neighbors at 0.5x
        7. Strength decay weighting (optional -- skipped if memory objects
           lack ``calculate_current_strength``)
        8. Cache and return ranked results
        """
        # -- LRU cache check --
        type_str = ",".join(sorted(
            mt.value if hasattr(mt, "value") else str(mt) for mt in (memory_types or [])
        )) or "*"
        cache_key = f"{query}:{max_results}:{min_strength}:{type_str}:{agent_id}:{scope}"
        if cache_key in self._query_cache:
            try:
                self._query_cache_order.remove(cache_key)
            except ValueError:
                pass
            self._query_cache_order.append(cache_key)
            return [
                (self.nodes[nid], score)
                for nid, score in self._query_cache[cache_key]
                if nid in self.nodes
            ]

        kw_weight, sem_weight, category = classify_query(query)
        logger.debug(f"[MemoryGraph] Query classified as '{category}': kw={kw_weight}, sem={sem_weight}")

        # Filter eligible nodes
        eligible = self._get_eligible_nodes(min_strength, memory_types, agent_id, scope)
        if not eligible:
            return []

        # Keyword search
        kw_scores = self._keyword_search(query, eligible)

        # Semantic search
        sem_scores = self._semantic_search(query, eligible)

        # Merge
        all_ids = set(kw_scores.keys()) | set(sem_scores.keys())
        combined: Dict[str, float] = {}
        for nid in all_ids:
            kw = kw_scores.get(nid, 0.0)
            sem = sem_scores.get(nid, 0.0)
            combined[nid] = kw * kw_weight + sem * sem_weight

        # Strength decay weighting (optional for ADK memory objects)
        for nid in combined:
            node = self.nodes[nid]
            if hasattr(node.memory, "calculate_current_strength"):
                try:
                    strength = node.memory.calculate_current_strength()
                except Exception:
                    strength = 1.0
            else:
                strength = 1.0
            combined[nid] *= strength

        # Sort and get top-N for expansion
        ranked = sorted(combined.items(), key=lambda x: -x[1])

        # Graph expansion: for top-5, add 1-hop neighbors at 0.5x parent score
        expansion_seeds = ranked[:5]
        for nid, score in expansion_seeds:
            node = self.nodes.get(nid)
            if not node:
                continue
            for neighbor_id in node.neighbors:
                if neighbor_id not in combined and neighbor_id in eligible:
                    combined[neighbor_id] = score * 0.5

        # Re-rank with expansions
        ranked = sorted(combined.items(), key=lambda x: -x[1])

        # Build result
        results = []
        for nid, score in ranked[:max_results]:
            node = self.nodes.get(nid)
            if node and score > 0:
                results.append((node, round(score, 4)))

        # -- Update LRU cache --
        if len(self._query_cache) >= self._QUERY_CACHE_MAX:
            evict_key = self._query_cache_order.pop(0)
            self._query_cache.pop(evict_key, None)
        self._query_cache[cache_key] = [(n.id, s) for n, s in results]
        self._query_cache_order.append(cache_key)

        return results

    def _get_eligible_nodes(
        self,
        min_strength: float,
        memory_types: Optional[list],
        agent_id: Optional[str],
        scope: str,
    ) -> Set[str]:
        """Return set of node IDs passing filters.

        Uses by_type/by_agent indexes for sublinear filtering
        instead of scanning all nodes.
        """
        # Start with the smallest candidate pool from indexes
        if memory_types:
            type_values = set()
            for mt in memory_types:
                type_values.add(mt.value if hasattr(mt, "value") else str(mt))
            # Union of by_type lists -- avoids scanning non-matching types
            candidate_ids: Set[str] = set()
            for tv in type_values:
                candidate_ids.update(self.by_type.get(tv, []))
        elif agent_id:
            # Include nodes created by this agent AND all nodes (scope filter
            # handles visibility). Private-scoped memories may have a different
            # source_agent than the querying agent.
            candidate_ids = set(self.nodes.keys())
        else:
            candidate_ids = set(self.nodes.keys())

        eligible: Set[str] = set()
        for nid in candidate_ids:
            node = self.nodes.get(nid)
            if not node:
                continue
            mem = node.memory
            if getattr(mem, "archived", False):
                continue

            # Scope check
            mem_scope = getattr(mem, "scope", "shared")
            if mem_scope != "shared":
                if not agent_id or not mem_scope.startswith(f"private:{agent_id}"):
                    continue

            # Strength check (optional -- ADK Memory objects may not have this)
            if hasattr(mem, "calculate_current_strength"):
                try:
                    if mem.calculate_current_strength() < min_strength:
                        continue
                except Exception:
                    pass  # Skip strength check on error

            eligible.add(nid)
        return eligible

    def _keyword_search(self, query: str, eligible: Set[str]) -> Dict[str, float]:
        """
        Keyword scoring using inverted index.

        Instead of scanning ALL eligible nodes, use the keyword index to
        find only nodes containing query tokens, then score those.
        Returns dict of node_id -> score (0-1).
        """
        query_words = self._extract_keywords(query)
        if not query_words:
            return {}

        # Gather candidates from inverted index (O(k) where k = keywords)
        candidate_ids: Set[str] = set()
        for word in query_words:
            candidate_ids.update(self._keyword_index.get(word, set()))

        # Intersect with eligible set
        candidate_ids &= eligible
        if not candidate_ids:
            return {}

        scores: Dict[str, float] = {}
        n_query = max(len(query_words), 1)

        for nid in candidate_ids:
            node = self.nodes.get(nid)
            if not node:
                continue
            mem = node.memory
            title_lower = (getattr(mem, "title", "") or "").lower()
            content_lower = (getattr(mem, "content", "") or "").lower()
            tag_str = " ".join(getattr(mem, "tags", []) or []).lower()

            title_words = set(title_lower.split())
            content_words = set(content_lower.split())
            tag_words = set(tag_str.split())

            title_overlap = len(query_words & title_words) / n_query
            content_overlap = len(query_words & content_words) / n_query
            tag_overlap = len(query_words & tag_words) / n_query

            score = title_overlap * 0.5 + content_overlap * 0.3 + tag_overlap * 0.2
            if score > 0:
                scores[nid] = score

        # Normalize to 0-1 relative to max
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for nid in scores:
                    scores[nid] /= max_score

        return scores

    def _semantic_search(self, query: str, eligible: Set[str]) -> Dict[str, float]:
        """
        Semantic scoring: cosine similarity of query embedding vs node embeddings.
        Returns dict of node_id -> score (0-1).
        """
        query_emb = _get_embedding(query)
        if not query_emb:
            return {}

        scores: Dict[str, float] = {}
        for nid in eligible:
            emb = self._embedding_cache.get(nid)
            if not emb:
                node = self.nodes.get(nid)
                emb = node.embedding if node else None
            if emb:
                sim = _cosine_similarity(query_emb, emb)
                if sim > 0.1:
                    scores[nid] = sim

        # Normalize to 0-1 relative to max
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for nid in scores:
                    scores[nid] /= max_score

        return scores

    # ------------------------------------------------------------------
    # MULTI-HOP EXPANSION
    # ------------------------------------------------------------------

    def multi_hop_expand(
        self,
        seed_ids: List[str],
        query: str = "",
        max_depth: int = 2,
        max_chains: int = 3,
    ) -> List[Tuple[List[str], float]]:
        """
        BFS on edge graph following outgoing/incoming edges.

        Returns chains as (path, cumulative_score) tuples.
        Enables "trace the evolution of what I know about X".

        Args:
            seed_ids: Starting node IDs
            query: Optional query for relevance scoring
            max_depth: Max BFS depth
            max_chains: Max chains to return
        """
        query_emb = _get_embedding(query) if query else None
        chains: List[Tuple[List[str], float]] = []

        for seed in seed_ids:
            if seed not in self.nodes:
                continue
            self._bfs_chains(seed, query_emb, max_depth, chains, [seed], 1.0)

        # Sort by score descending
        chains.sort(key=lambda x: -x[1])
        return chains[:max_chains]

    def _bfs_chains(
        self,
        current: str,
        query_emb: Optional[list],
        depth_remaining: int,
        chains: List[Tuple[List[str], float]],
        path: List[str],
        cumulative_score: float,
    ):
        """Recursive BFS chain builder."""
        if depth_remaining <= 0:
            if len(path) > 1:
                chains.append((list(path), cumulative_score))
            return

        node = self.nodes.get(current)
        if not node:
            if len(path) > 1:
                chains.append((list(path), cumulative_score))
            return

        has_extension = False
        for neighbor_id in node.neighbors:
            if neighbor_id in path:
                continue  # No cycles

            # Score this hop
            hop_score = self._hop_relevance(current, neighbor_id, query_emb)
            if hop_score < 0.1:
                continue

            new_path = path + [neighbor_id]
            new_score = cumulative_score * hop_score
            has_extension = True

            self._bfs_chains(
                neighbor_id, query_emb, depth_remaining - 1,
                chains, new_path, new_score,
            )

        if not has_extension and len(path) > 1:
            chains.append((list(path), cumulative_score))

    def _hop_relevance(self, from_id: str, to_id: str, query_emb: Optional[list]) -> float:
        """Score a single hop: edge weight * optional query relevance."""
        # Find the best edge between these nodes
        best_weight = 0.0
        for eid in self.edges_by_source.get(from_id, []):
            edge = self.edges.get(eid)
            if edge and edge.target_id == to_id:
                best_weight = max(best_weight, edge.weight)
        for eid in self.edges_by_target.get(from_id, []):
            edge = self.edges.get(eid)
            if edge and edge.source_id == to_id:
                best_weight = max(best_weight, edge.weight)

        if best_weight == 0:
            best_weight = 0.3  # Default for neighbors without scored edges

        # Optionally weight by query relevance
        if query_emb:
            to_emb = self._embedding_cache.get(to_id)
            if to_emb:
                sim = _cosine_similarity(query_emb, to_emb)
                return best_weight * max(sim, 0.1)

        return best_weight

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self):
        """Persist graph edges + embedding cache to disk."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Save edges + node edge lists (not full Memory objects)
        graph_data = {
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
            "node_edges": {
                nid: {
                    "outgoing": list(n.outgoing_edges),
                    "incoming": list(n.incoming_edges),
                    "neighbors": list(n.neighbors),
                }
                for nid, n in self.nodes.items()
            },
        }

        # Atomic write: temp file -> rename
        tmp_graph = self._graph_pickle.with_suffix(".tmp")
        try:
            with open(tmp_graph, "wb") as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_graph.replace(self._graph_pickle)
            _mg_write_pickle_hmac(str(self._graph_pickle))
        except Exception as e:
            logger.error(f"[MemoryGraph] Failed to save graph: {e}")
            if tmp_graph.exists():
                tmp_graph.unlink()

        # Save embeddings separately (can be large)
        if self._embedding_cache:
            tmp_emb = self._embeddings_pickle.with_suffix(".tmp")
            try:
                with open(tmp_emb, "wb") as f:
                    pickle.dump(dict(self._embedding_cache), f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_emb.replace(self._embeddings_pickle)
                _mg_write_pickle_hmac(str(self._embeddings_pickle))
            except Exception as e:
                logger.error(f"[MemoryGraph] Failed to save embeddings: {e}")
                if tmp_emb.exists():
                    tmp_emb.unlink()

        logger.debug(f"[MemoryGraph] Saved {len(self.edges)} edges, {len(self._embedding_cache)} embeddings")

    def load(self, memories: Dict[str, Any]):
        """
        Load graph from disk, re-attaching to live Memory objects.

        Args:
            memories: Dict[id, Memory] -- mapping of memory IDs to memory
                      objects (any object with an ``id`` attribute).
        """
        # First, populate nodes from live memories (source of truth)
        for mid, mem in memories.items():
            if mid not in self.nodes:
                node = MemoryNode(memory=mem)
                self.nodes[mid] = node
                self._add_to_indexes(node)
            else:
                self.nodes[mid].memory = mem
            # Cache embedding
            if getattr(mem, "embedding", None):
                self._embedding_cache[mid] = mem.embedding

        # Load saved embeddings (supplement live embeddings)
        if self._embeddings_pickle.exists():
            if not _mg_verify_pickle_hmac(str(self._embeddings_pickle)):
                logger.warning("[MemoryGraph] Embedding cache HMAC invalid — skipping")
            else:
                try:
                    with open(self._embeddings_pickle, "rb") as f:
                        saved_emb = pickle.load(f)
                    for mid, emb in saved_emb.items():
                        if mid in self.nodes and mid not in self._embedding_cache:
                            self._embedding_cache[mid] = emb
                    logger.debug(f"[MemoryGraph] Loaded {len(saved_emb)} cached embeddings")
                except Exception as e:
                    logger.warning(f"[MemoryGraph] Failed to load embedding cache: {e}")

        # Fill missing embeddings in background thread (don't block boot)
        missing_emb = [mid for mid in self.nodes if mid not in self._embedding_cache]
        if missing_emb:
            import threading as _th
            def _warm_embeddings(graph_ref, node_ids):
                filled = 0
                for mid in node_ids:
                    node = graph_ref.nodes.get(mid)
                    if not node:
                        continue
                    content = getattr(node.memory, "content", "")
                    if content:
                        emb = _get_embedding(content)
                        if emb:
                            graph_ref._embedding_cache[mid] = emb
                            if hasattr(node.memory, "embedding"):
                                node.memory.embedding = emb
                            filled += 1
                if filled:
                    logger.info(f"[MemoryGraph] Background: computed {filled}/{len(node_ids)} embeddings")
                    # Re-detect RELATED edges now that embeddings exist
                    graph_ref._rebuild_edges()
                    graph_ref.save()
            t = _th.Thread(target=_warm_embeddings, args=(self, missing_emb), daemon=True)
            t.start()
            logger.info(f"[MemoryGraph] Warming {len(missing_emb)} embeddings in background")

        # Load saved edges
        if self._graph_pickle.exists():
            if not _mg_verify_pickle_hmac(str(self._graph_pickle)):
                logger.warning("[MemoryGraph] Graph cache HMAC invalid — skipping edge restore")
                self._rebuild_edges()
            else:
                try:
                    with open(self._graph_pickle, "rb") as f:
                        graph_data = pickle.load(f)

                    # Restore edges
                    for eid, edata in graph_data.get("edges", {}).items():
                        edge = MemoryEdge.from_dict(edata)
                        # Only restore edges where both nodes still exist
                        if edge.source_id in self.nodes and edge.target_id in self.nodes:
                            self.edges[eid] = edge
                            self.edges_by_source[edge.source_id].append(eid)
                            self.edges_by_target[edge.target_id].append(eid)

                    # Restore node edge lists
                    for nid, lists in graph_data.get("node_edges", {}).items():
                        if nid in self.nodes:
                            node = self.nodes[nid]
                            # Only keep edge refs that still exist
                            node.outgoing_edges = [e for e in lists.get("outgoing", []) if e in self.edges]
                            node.incoming_edges = [e for e in lists.get("incoming", []) if e in self.edges]
                            node.neighbors = [n for n in lists.get("neighbors", []) if n in self.nodes]

                    logger.info(f"[MemoryGraph] Loaded {len(self.edges)} edges for {len(self.nodes)} nodes")
                except Exception as e:
                    logger.warning(f"[MemoryGraph] Failed to load graph (will rebuild): {e}")
                    # Graph data corrupt -- auto-detect edges for all nodes
                    self._rebuild_edges()
        else:
            # No saved graph -- run initial edge detection
            if len(self.nodes) > 0:
                logger.info(f"[MemoryGraph] No saved graph, building edges for {len(self.nodes)} nodes...")
                self._rebuild_edges()

    def _rebuild_edges(self):
        """Rebuild all edges from scratch via auto-detection."""
        self.edges.clear()
        self.edges_by_source.clear()
        self.edges_by_target.clear()
        for node in self.nodes.values():
            node.outgoing_edges.clear()
            node.incoming_edges.clear()
            node.neighbors.clear()

        for node in list(self.nodes.values()):
            self._auto_detect_edges(node)

        logger.info(f"[MemoryGraph] Rebuilt {len(self.edges)} edges")

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the graph."""
        edge_type_counts: Dict[str, int] = defaultdict(int)
        for e in self.edges.values():
            edge_type_counts[e.edge_type.value] += 1

        avg_neighbors = 0.0
        if self.nodes:
            avg_neighbors = sum(len(n.neighbors) for n in self.nodes.values()) / len(self.nodes)

        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "edge_types": dict(edge_type_counts),
            "tags_indexed": len(self.by_tag),
            "types_indexed": len(self.by_type),
            "agents_indexed": len(self.by_agent),
            "embeddings_cached": len(self._embedding_cache),
            "avg_neighbors": round(avg_neighbors, 2),
        }

    # ------------------------------------------------------------------
    # BATCH INGESTION
    # ------------------------------------------------------------------

    def batch_remember(
        self,
        entries: List[Dict[str, Any]],
        source: str = "batch_backfill",
    ) -> int:
        """
        Batch-ingest memory entries from training data or conversation export.

        Each entry dict should have at minimum:
            - "content" or "instruction"+"output": the memory text
            - "id" (optional): unique ID, auto-generated if missing

        Returns count of entries added.
        """
        from types import SimpleNamespace

        added = 0
        for entry in entries:
            try:
                content = entry.get("content", "")
                if not content:
                    # Build from instruction/output format
                    instruction = entry.get("instruction", "")
                    output = entry.get("output", "")
                    if instruction and output:
                        content = f"Q: {instruction}\nA: {output}"
                if not content:
                    continue

                mid = entry.get("id") or hashlib.md5(
                    content[:200].encode()
                ).hexdigest()

                mem = SimpleNamespace(
                    id=mid,
                    title=content[:80],
                    content=content,
                    memory_type=entry.get("type", "episodic"),
                    tags=entry.get("tags", [source]),
                    source_agent=entry.get("agent", "system"),
                    importance=entry.get("importance", 0.4),
                    embedding=None,
                    created_at=entry.get("timestamp", time.time()),
                )

                self.add_node(mem, upsert=True)
                added += 1
            except Exception as e:
                logger.debug(f"[MemoryGraph] Batch skip: {e}")
                continue

        if added > 0:
            self.save()
            logger.info(
                f"[MemoryGraph] Batch ingested {added}/{len(entries)} "
                f"entries from {source}"
            )
        return added


# ============================================================================
# SINGLETON
# ============================================================================

_memorygraph_instance: Optional[MemoryGraph] = None
_memorygraph_lock = threading.Lock()


def get_memorygraph() -> MemoryGraph:
    """Get or create the MemoryGraph singleton (thread-safe)."""
    global _memorygraph_instance
    if _memorygraph_instance is not None:
        return _memorygraph_instance

    with _memorygraph_lock:
        if _memorygraph_instance is not None:
            return _memorygraph_instance
        _memorygraph_instance = MemoryGraph()
        return _memorygraph_instance
