"""Local knowledge graph — entity extraction, embeddings, hybrid search.

Zero external dependencies beyond httpx (already required). Uses SQLite for
storage, Ollama nomic-embed-text for embeddings (falls back to feature hashing).

This is a lightweight port of AitherOS MemoryGraph. Key differences:
  - SQLite instead of pickle (single file, no HMAC needed)
  - Feature hashing fallback instead of sentence-transformers
  - Simpler entity extraction (regex, no spacy)
  - Same hybrid search pipeline (keyword + semantic)

Usage:
    from adk.graph_memory import GraphMemory

    graph = GraphMemory(agent_name="atlas")
    await graph.remember("AitherOS", "uses", "SQLite")
    await graph.remember("AitherOS", "has", "97 microservices")

    results = await graph.search("What database does AitherOS use?")
    # [GraphNode(label="AitherOS", ...), GraphNode(label="SQLite", ...)]

    # Auto-ingest from conversations
    await graph.ingest_conversation("session1", [
        {"role": "user", "content": "How does AitherOS handle memory?"},
        {"role": "assistant", "content": "AitherOS uses MemoryGraph with embeddings."},
    ])

    # Multi-hop traversal
    related = await graph.get_related("AitherOS", depth=2)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("adk.graph_memory")

_EMBED_DIM = 384  # Match nomic-embed-text small dimension
_OLLAMA_URL = "http://localhost:11434"
_EMBED_MODEL = "nomic-embed-text"

# Stopwords for keyword extraction
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for with on at "
    "by from as into through during before after above below between "
    "and or but not no nor so yet both either neither each every all any "
    "few more most other some such this that these those it its he she "
    "they them their we our you your i me my what which who whom how "
    "when where why if then than too very just about also back only "
    "even still already again further once here there up down out off".split()
)


class EdgeType(str, Enum):
    RELATED = "related"           # Embedding similarity > threshold
    DERIVED_FROM = "derived_from" # B created because of A
    TAG_SIBLING = "tag_sibling"   # Share 2+ tags
    SAME_SESSION = "same_session" # Same conversation session
    TEMPORAL = "temporal"         # Created within time window
    MENTIONS = "mentions"         # Entity mentions entity
    IS_A = "is_a"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    CONNECTS_TO = "connects_to"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    label: str
    node_type: str = "entity"   # entity, concept, session, fact, relation
    content: str = ""
    tags: list[str] = field(default_factory=list)
    source_agent: str = ""
    source_session: str = ""
    importance: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge connecting two nodes."""
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction patterns
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_PATTERNS = [
    # Service/class names (CamelCase with known suffixes)
    (re.compile(r'\b([A-Z][a-zA-Z]+(?:Service|Manager|Client|Engine|Controller|Provider|Graph|Store|Bridge|Guard|Pipeline))\b'), "service"),
    # Capitalized multi-word phrases (2-4 words)
    (re.compile(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b'), "entity"),
    # Single capitalized words (3+ chars, not at sentence start — rough heuristic)
    (re.compile(r'(?<=[.!?]\s)\b([A-Z][a-z]{2,})\b|(?<=\s)\b([A-Z][a-z]{2,})\b'), "entity"),
    # File paths
    (re.compile(r'\b([a-zA-Z0-9_/\\]+\.[a-z]{1,5})\b'), "file"),
    # Code identifiers (snake_case with 2+ segments)
    (re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b'), "code"),
]

_RELATION_PATTERNS = [
    (re.compile(r'(\w+)\s+(?:is|are)\s+(?:a|an|the)\s+(\w+)', re.I), EdgeType.IS_A),
    (re.compile(r'(\w+)\s+(?:uses?|using|utilizes?)\s+(\w+)', re.I), EdgeType.USES),
    (re.compile(r'(\w+)\s+(?:depends?\s+on|requires?)\s+(\w+)', re.I), EdgeType.DEPENDS_ON),
    (re.compile(r'(\w+)\s+(?:contains?|includes?|has)\s+(\w+)', re.I), EdgeType.CONTAINS),
    (re.compile(r'(\w+)\s+(?:connects?\s+to|communicates?\s+with)\s+(\w+)', re.I), EdgeType.CONNECTS_TO),
]


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract (entity_label, entity_type) from text."""
    entities = []
    seen = set()
    for pattern, etype in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            label = match.group(1) or match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
            if label and label.lower() not in _STOPWORDS and len(label) >= 3:
                key = label.lower()
                if key not in seen:
                    seen.add(key)
                    entities.append((label, etype))
    return entities[:30]  # Cap


def extract_relations(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) triples from text."""
    relations = []
    for pattern, rel_type in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            subj, obj = match.group(1), match.group(2)
            if subj.lower() not in _STOPWORDS and obj.lower() not in _STOPWORDS:
                relations.append((subj, rel_type.value, obj))
    return relations[:20]


def extract_keywords(text: str) -> list[str]:
    """Extract keywords from text (stopword-filtered, lowercased)."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in _STOPWORDS][:50]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed_to_blob(embedding: list[float]) -> bytes:
    """Pack float list into compact binary for SQLite BLOB storage."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def _blob_to_embed(blob: bytes) -> list[float]:
    """Unpack BLOB back to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. No numpy needed."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _fallback_embed(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """Feature-hashing embedding. Works offline, no model needed."""
    vector = [0.0] * dim
    words = text.lower().split()
    for word in words:
        idx = hash(word) % dim
        vector[idx] += 1.0
        # Bigrams for basic context
        if len(word) > 3:
            idx2 = hash(word[:3]) % dim
            vector[idx2] += 0.5
    # L2 normalize
    magnitude = sum(x * x for x in vector) ** 0.5
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    return vector


# ─────────────────────────────────────────────────────────────────────────────
# Query classification (simplified from AitherOS 6-category system)
# ─────────────────────────────────────────────────────────────────────────────

_RE_IDENTITY = re.compile(r'(?:what|who)\s+(?:is|am|are)\s+(?:my|your|the)\s', re.I)
_RE_PROCEDURAL = re.compile(r'(?:how\s+(?:do|to|can)|steps?\s+to|procedure)', re.I)
_RE_SPECIFIC = re.compile(r'"[^"]+"|\b[A-Z][a-zA-Z]+(?:Service|Graph|Engine)\b', re.I)
_RE_CONCEPTUAL = re.compile(r'(?:related\s+to|connections?\s+(?:to|between)|associated)', re.I)
_RE_EXPLORATORY = re.compile(r'(?:what\s+do\s+(?:I|you|we)\s+know|tell\s+me\s+about)', re.I)


def _classify_query(query: str) -> tuple[float, float]:
    """Returns (keyword_weight, semantic_weight)."""
    if _RE_IDENTITY.search(query):
        return (0.9, 0.1)
    if _RE_PROCEDURAL.search(query):
        return (0.6, 0.4)
    if _RE_SPECIFIC.search(query):
        return (0.8, 0.2)
    if _RE_CONCEPTUAL.search(query):
        return (0.2, 0.8)
    if _RE_EXPLORATORY.search(query):
        return (0.3, 0.7)
    return (0.4, 0.6)  # balanced default


# ─────────────────────────────────────────────────────────────────────────────
# GraphMemory
# ─────────────────────────────────────────────────────────────────────────────

_SIMILARITY_THRESHOLD = 0.65
_TEMPORAL_WINDOW_SECS = 300
_MAX_RELATED_PER_NODE = 5
_MAX_CANDIDATES_FOR_SIM = 50


class GraphMemory:
    """Local knowledge graph with embedding-based search.

    SQLite-backed, Ollama-optional, zero external dependencies.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        agent_name: str = "default",
        embed_model: str = _EMBED_MODEL,
        ollama_url: str = "",
    ):
        if db_path is None:
            data_dir = Path(
                os.getenv("AITHER_DATA_DIR", os.path.expanduser("~/.aither"))
            ) / "graph"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / f"{agent_name}.db"

        self._db_path = str(db_path)
        self._agent = agent_name
        self._embed_model = embed_model
        self._ollama_url = (
            ollama_url
            or os.getenv("OLLAMA_HOST", _OLLAMA_URL)
        ).rstrip("/")
        self._ollama_available: bool | None = None  # Lazy detect
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    node_type TEXT DEFAULT 'entity',
                    content TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    source_agent TEXT DEFAULT '',
                    source_session TEXT DEFAULT '',
                    importance REAL DEFAULT 0.5,
                    embedding BLOB,
                    created_at REAL,
                    updated_at REAL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at REAL,
                    metadata TEXT DEFAULT '{}',
                    PRIMARY KEY (source_id, target_id, relation),
                    FOREIGN KEY (source_id) REFERENCES nodes(id),
                    FOREIGN KEY (target_id) REFERENCES nodes(id)
                );
                CREATE TABLE IF NOT EXISTS keywords (
                    keyword TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    PRIMARY KEY (keyword, node_id),
                    FOREIGN KEY (node_id) REFERENCES nodes(id)
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_nodes_agent ON nodes(source_agent);
                CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(source_session);
                CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
                CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
                CREATE INDEX IF NOT EXISTS idx_keywords ON keywords(keyword);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ─── Core CRUD ──────────────────────────────────────────────────────

    async def add_node(
        self,
        label: str,
        node_type: str = "entity",
        content: str = "",
        tags: list[str] | None = None,
        source_session: str = "",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> GraphNode:
        """Add a node to the graph. Auto-detects edges."""
        node_id = hashlib.md5(f"{label}:{node_type}".encode()).hexdigest()[:12]
        now = time.time()
        tags = tags or []
        embedding = await self._embed(f"{label} {content}")

        with self._connect() as conn:
            # Upsert node
            existing = conn.execute(
                "SELECT id, access_count FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE nodes SET content = ?, tags = ?, updated_at = ?, "
                    "access_count = ?, embedding = ?, importance = ?, metadata = ? WHERE id = ?",
                    (content, json.dumps(tags), now, (existing[1] or 0) + 1,
                     _embed_to_blob(embedding) if embedding else None,
                     importance, json.dumps(metadata or {}), node_id),
                )
            else:
                conn.execute(
                    "INSERT INTO nodes (id, label, node_type, content, tags, source_agent, "
                    "source_session, importance, embedding, created_at, updated_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (node_id, label, node_type, content, json.dumps(tags),
                     self._agent, source_session, importance,
                     _embed_to_blob(embedding) if embedding else None,
                     now, now, json.dumps(metadata or {})),
                )

            # Update keyword index
            conn.execute("DELETE FROM keywords WHERE node_id = ?", (node_id,))
            keywords = extract_keywords(f"{label} {content}")
            for kw in keywords:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO keywords (keyword, node_id) VALUES (?, ?)",
                        (kw, node_id),
                    )
                except sqlite3.IntegrityError:
                    pass

        # Auto-detect edges (non-blocking)
        try:
            await self._auto_detect_edges(node_id, label, tags, embedding, source_session)
        except Exception as exc:
            logger.debug("Auto-edge detection failed (non-fatal): %s", exc)

        node = GraphNode(
            id=node_id, label=label, node_type=node_type, content=content,
            tags=tags, source_agent=self._agent, source_session=source_session,
            importance=importance, created_at=now, updated_at=now,
            metadata=metadata or {},
        )
        return node

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, label, node_type, content, tags, source_agent, "
                "source_session, importance, created_at, updated_at, access_count, metadata "
                "FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
        if not row:
            return None
        return GraphNode(
            id=row[0], label=row[1], node_type=row[2], content=row[3],
            tags=json.loads(row[4] or "[]"), source_agent=row[5] or "",
            source_session=row[6] or "", importance=row[7] or 0.5,
            created_at=row[8] or 0, updated_at=row[9] or 0,
            access_count=row[10] or 0, metadata=json.loads(row[11] or "{}"),
        )

    async def remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        with self._connect() as conn:
            conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
            conn.execute("DELETE FROM keywords WHERE node_id = ?", (node_id,))
            conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> GraphEdge:
        """Add an edge between two nodes."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source_id, target_id, relation, weight, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, target_id, relation, weight, now, json.dumps(metadata or {})),
            )
        return GraphEdge(
            source_id=source_id, target_id=target_id, relation=relation,
            weight=weight, created_at=now, metadata=metadata or {},
        )

    async def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "both",
    ) -> list[GraphNode]:
        """Get neighboring nodes via edges."""
        node_ids = set()
        with self._connect() as conn:
            if direction in ("out", "both"):
                q = "SELECT target_id FROM edges WHERE source_id = ?"
                params: list = [node_id]
                if relation:
                    q += " AND relation = ?"
                    params.append(relation)
                for row in conn.execute(q, params).fetchall():
                    node_ids.add(row[0])
            if direction in ("in", "both"):
                q = "SELECT source_id FROM edges WHERE target_id = ?"
                params = [node_id]
                if relation:
                    q += " AND relation = ?"
                    params.append(relation)
                for row in conn.execute(q, params).fetchall():
                    node_ids.add(row[0])

        nodes = []
        for nid in node_ids:
            node = await self.get_node(nid)
            if node:
                nodes.append(node)
        return nodes

    # ─── Convenience API ────────────────────────────────────────────────

    async def remember(
        self,
        subject: str,
        relation: str,
        object_: str,
        metadata: dict | None = None,
    ):
        """Store a knowledge triple (subject, relation, object)."""
        subj_node = await self.add_node(
            label=subject, node_type="entity",
            content=f"{subject} {relation} {object_}",
            metadata=metadata,
        )
        obj_node = await self.add_node(
            label=object_, node_type="entity",
            content=f"{object_} (related to {subject})",
        )
        await self.add_edge(subj_node.id, obj_node.id, relation, weight=0.8)

    async def recall(self, subject: str, relation: str | None = None) -> list[dict]:
        """Query triples by subject. Returns [{relation, object, weight}]."""
        node_id = hashlib.md5(f"{subject}:entity".encode()).hexdigest()[:12]
        results = []
        with self._connect() as conn:
            q = "SELECT e.relation, n.label, e.weight FROM edges e JOIN nodes n ON e.target_id = n.id WHERE e.source_id = ?"
            params: list = [node_id]
            if relation:
                q += " AND e.relation = ?"
                params.append(relation)
            for row in conn.execute(q, params).fetchall():
                results.append({"relation": row[0], "object": row[1], "weight": row[2]})
        return results

    # ─── Hybrid Search ──────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: int = 10,
        node_type: str | None = None,
    ) -> list[GraphNode]:
        """Hybrid search: keyword + semantic, weighted by query type."""
        if not query.strip():
            return []

        kw_weight, sem_weight = _classify_query(query)
        keywords = extract_keywords(query)
        query_embedding = await self._embed(query)

        scores: dict[str, float] = {}

        with self._connect() as conn:
            # Stage 1: Keyword search via inverted index
            if keywords:
                placeholders = ",".join("?" * len(keywords))
                rows = conn.execute(
                    f"SELECT node_id, COUNT(*) as hits FROM keywords "
                    f"WHERE keyword IN ({placeholders}) GROUP BY node_id "
                    f"ORDER BY hits DESC LIMIT ?",
                    (*keywords, limit * 3),
                ).fetchall()
                max_hits = max((r[1] for r in rows), default=1)
                for nid, hits in rows:
                    scores[nid] = (hits / max_hits) * kw_weight

            # Stage 2: Semantic search via embedding similarity
            if query_embedding:
                type_filter = ""
                params: list = []
                if node_type:
                    type_filter = "WHERE node_type = ?"
                    params.append(node_type)
                rows = conn.execute(
                    f"SELECT id, embedding FROM nodes {type_filter} "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (*params, _MAX_CANDIDATES_FOR_SIM * 5),
                ).fetchall()
                for nid, emb_blob in rows:
                    if emb_blob:
                        emb = _blob_to_embed(emb_blob)
                        sim = cosine_similarity(query_embedding, emb)
                        if sim > 0.1:
                            scores[nid] = scores.get(nid, 0) + sim * sem_weight

        # Stage 3: Rank and fetch
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        nodes = []
        for nid, score in ranked:
            node = await self.get_node(nid)
            if node:
                # Increment access count
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE nodes SET access_count = access_count + 1 WHERE id = ?",
                        (nid,),
                    )
                nodes.append(node)
        return nodes

    async def query(self, question: str, limit: int = 5) -> list[GraphNode]:
        """Natural language query — alias for search()."""
        return await self.search(question, limit=limit)

    # ─── Conversation Ingestion ─────────────────────────────────────────

    async def ingest_conversation(
        self,
        session_id: str,
        messages: list[dict],
        extract_triples: bool = True,
    ) -> int:
        """Auto-ingest entities and relationships from conversation messages.

        Returns number of nodes created/updated.
        """
        count = 0
        all_text = " ".join(m.get("content", "") for m in messages)

        # Extract and store entities
        entities = extract_entities(all_text)
        for label, etype in entities:
            try:
                await self.add_node(
                    label=label, node_type=etype,
                    content=all_text[:500],
                    source_session=session_id,
                    tags=[etype, "auto_extracted"],
                )
                count += 1
            except Exception:
                pass

        # Extract and store relations as edges
        if extract_triples:
            relations = extract_relations(all_text)
            for subj, rel, obj in relations:
                try:
                    await self.remember(subj, rel, obj)
                    count += 2  # subject + object nodes
                except Exception:
                    pass

        # Store conversation as a session node
        try:
            user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
            summary = "; ".join(user_msgs)[:300]
            await self.add_node(
                label=f"session:{session_id[:8]}",
                node_type="session",
                content=summary,
                source_session=session_id,
                tags=["conversation", "auto_ingested"],
            )
            count += 1
        except Exception:
            pass

        if count:
            logger.debug("Ingested %d nodes from session %s", count, session_id[:8])
        return count

    async def ingest_text(self, text: str, source: str = "unknown") -> int:
        """Ingest entities from arbitrary text."""
        return await self.ingest_conversation(
            session_id=f"text:{source}",
            messages=[{"role": "user", "content": text}],
        )

    # ─── Graph Traversal ────────────────────────────────────────────────

    async def get_related(self, entity: str, depth: int = 2) -> dict:
        """BFS expansion from entity. Returns subgraph as adjacency dict."""
        node_id = hashlib.md5(f"{entity}:entity".encode()).hexdigest()[:12]
        visited = set()
        subgraph: dict[str, list[dict]] = {}
        queue = [(node_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)

            node = await self.get_node(current_id)
            if not node:
                continue

            neighbors = await self.get_neighbors(current_id)
            subgraph[node.label] = []
            for n in neighbors:
                # Get edge details
                with self._connect() as conn:
                    edge = conn.execute(
                        "SELECT relation, weight FROM edges "
                        "WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
                        (current_id, n.id, n.id, current_id),
                    ).fetchone()
                rel = edge[0] if edge else "related"
                weight = edge[1] if edge else 0.5
                subgraph[node.label].append({
                    "entity": n.label, "relation": rel, "weight": weight,
                })
                if n.id not in visited and current_depth + 1 <= depth:
                    queue.append((n.id, current_depth + 1))

        return subgraph

    # ─── Stats ──────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """Get graph statistics."""
        with self._connect() as conn:
            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            keyword_count = conn.execute("SELECT COUNT(DISTINCT keyword) FROM keywords").fetchone()[0]
            types = conn.execute(
                "SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type"
            ).fetchall()
            relations = conn.execute(
                "SELECT relation, COUNT(*) FROM edges GROUP BY relation"
            ).fetchall()
            embedded = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {
            "nodes": node_count,
            "edges": edge_count,
            "keywords": keyword_count,
            "embedded": embedded,
            "node_types": dict(types),
            "edge_relations": dict(relations),
            "agent": self._agent,
            "db_path": self._db_path,
            "ollama_available": self._ollama_available,
        }

    # ─── Embeddings ─────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        """Get embedding for text. Tries Ollama, falls back to feature hashing."""
        if self._ollama_available is False:
            return _fallback_embed(text)

        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self._ollama_url}/api/embeddings",
                    json={"model": self._embed_model, "prompt": text},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = data.get("embedding", [])
                    if embedding:
                        self._ollama_available = True
                        return embedding
        except Exception:
            pass

        if self._ollama_available is None:
            self._ollama_available = False
            logger.info("Ollama embeddings unavailable — using feature hashing fallback")
        return _fallback_embed(text)

    # ─── Auto-Edge Detection ────────────────────────────────────────────

    async def _auto_detect_edges(
        self,
        node_id: str,
        label: str,
        tags: list[str],
        embedding: list[float] | None,
        source_session: str,
    ):
        """Detect and create edges for a new/updated node."""
        with self._connect() as conn:
            # 1. TAG_SIBLING: shared tags (via keyword index on tag values)
            if tags:
                for tag in tags:
                    tag_lower = tag.lower()
                    rows = conn.execute(
                        "SELECT id, tags FROM nodes WHERE id != ? AND tags LIKE ?",
                        (node_id, f'%"{tag_lower}"%'),
                    ).fetchall()
                    for other_id, other_tags_json in rows:
                        other_tags = json.loads(other_tags_json or "[]")
                        shared = set(t.lower() for t in tags) & set(t.lower() for t in other_tags)
                        if len(shared) >= 2:
                            weight = min(1.0, len(shared) / 5.0)
                            try:
                                conn.execute(
                                    "INSERT OR IGNORE INTO edges (source_id, target_id, relation, weight, created_at) "
                                    "VALUES (?, ?, ?, ?, ?)",
                                    (node_id, other_id, EdgeType.TAG_SIBLING.value, weight, time.time()),
                                )
                            except sqlite3.IntegrityError:
                                pass

            # 2. SAME_SESSION: nodes from same conversation
            if source_session:
                rows = conn.execute(
                    "SELECT id FROM nodes WHERE id != ? AND source_session = ? LIMIT 10",
                    (node_id, source_session),
                ).fetchall()
                for (other_id,) in rows:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO edges (source_id, target_id, relation, weight, created_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (node_id, other_id, EdgeType.SAME_SESSION.value, 0.5, time.time()),
                        )
                    except sqlite3.IntegrityError:
                        pass

            # 3. RELATED: embedding similarity
            if embedding:
                rows = conn.execute(
                    "SELECT id, embedding FROM nodes WHERE id != ? AND embedding IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT ?",
                    (node_id, _MAX_CANDIDATES_FOR_SIM),
                ).fetchall()
                related_count = 0
                for other_id, emb_blob in rows:
                    if related_count >= _MAX_RELATED_PER_NODE:
                        break
                    other_emb = _blob_to_embed(emb_blob)
                    sim = cosine_similarity(embedding, other_emb)
                    if sim >= _SIMILARITY_THRESHOLD:
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO edges (source_id, target_id, relation, weight, created_at) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (node_id, other_id, EdgeType.RELATED.value, round(sim, 3), time.time()),
                            )
                            related_count += 1
                        except sqlite3.IntegrityError:
                            pass


# ─────────────────────────────────────────────────────────────────────────────
# Module singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: GraphMemory | None = None


def get_graph_memory(agent_name: str = "default") -> GraphMemory:
    """Get or create the module-level GraphMemory singleton."""
    global _instance
    if _instance is None:
        _instance = GraphMemory(agent_name=agent_name)
    return _instance
