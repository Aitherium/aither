"""Vector memory — semantic recall via pluggable embedding backends.

Complements :mod:`adk.memory` (SQLite KV) with embedding-based search.
The embedding backend defaults to :class:`adk.faculties.embeddings.EmbeddingProvider`
which already auto-selects sentence-transformers / Ollama / Elysium /
feature-hash.

Layout
------
* :class:`VectorHit`             — search result (id, score, payload)
* :class:`VectorStore` Protocol  — backend interface
* :class:`InMemoryVectorStore`   — zero-dep cosine scan, fine to ~10k records
* :class:`VectorMemory`          — high-level API used by agents

Usage
-----
::

    from adk.vector_memory import VectorMemory

    vmem = VectorMemory()  # uses default embedding provider + in-memory store
    await vmem.add("note-1", "Atlas owns the agent registry.")
    await vmem.add("note-2", "Prom keeps the host metrics.")
    hits = await vmem.search("who runs the registry?", k=1)
    assert hits[0].id == "note-1"
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Protocol, Sequence, runtime_checkable

logger = logging.getLogger("adk.vector_memory")

EmbedFn = Callable[[str], Awaitable[Sequence[float]]]


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VectorHit:
    """Search result returned by :meth:`VectorStore.search`."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Math
# ─────────────────────────────────────────────────────────────────────────────


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# ─────────────────────────────────────────────────────────────────────────────
# Embedder adapters
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into a vector asynchronously."""

    async def embed(self, text: str) -> Sequence[float]:
        ...


class _CallableEmbedder:
    """Wraps a plain ``async def embed(text)`` callable."""

    def __init__(self, fn: EmbedFn) -> None:
        self._fn = fn

    async def embed(self, text: str) -> Sequence[float]:
        return await self._fn(text)


def _as_embedder(obj: Any) -> Embedder:
    if isinstance(obj, Embedder):
        return obj
    if callable(obj):
        return _CallableEmbedder(obj)
    raise TypeError(f"unsupported embedder: {type(obj).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Store protocol + in-memory impl
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class VectorStore(Protocol):
    """Synchronous backing store. Embeddings are pushed in by the caller."""

    def add(self, vector_id: str, vector: Sequence[float], payload: dict[str, Any]) -> None: ...

    def search(self, query: Sequence[float], k: int = 5) -> list[VectorHit]: ...

    def get(self, vector_id: str) -> VectorHit | None: ...

    def delete(self, vector_id: str) -> bool: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


@dataclass
class _Entry:
    vector: list[float]
    payload: dict[str, Any]


class InMemoryVectorStore:
    """Pure-Python cosine-scan store. O(N·D) per query."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._dim: int | None = None

    @property
    def dim(self) -> int | None:
        return self._dim

    def add(self, vector_id: str, vector: Sequence[float], payload: dict[str, Any]) -> None:
        vec = list(vector)
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            raise ValueError(f"vector dim mismatch: {len(vec)} vs {self._dim}")
        self._entries[vector_id] = _Entry(vector=vec, payload=dict(payload))

    def search(self, query: Sequence[float], k: int = 5) -> list[VectorHit]:
        if not self._entries:
            return []
        q = list(query)
        if self._dim is not None and len(q) != self._dim:
            raise ValueError(f"query dim mismatch: {len(q)} vs {self._dim}")
        scored = [
            VectorHit(id=vid, score=_cosine(q, e.vector), payload=dict(e.payload))
            for vid, e in self._entries.items()
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(0, k)]

    def get(self, vector_id: str) -> VectorHit | None:
        entry = self._entries.get(vector_id)
        if entry is None:
            return None
        return VectorHit(id=vector_id, score=1.0, payload=dict(entry.payload))

    def delete(self, vector_id: str) -> bool:
        return self._entries.pop(vector_id, None) is not None

    def clear(self) -> None:
        self._entries.clear()
        self._dim = None

    def all_ids(self) -> list[str]:
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)


# ─────────────────────────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VectorRecord:
    """Result returned by :meth:`VectorMemory.search` — includes content."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created: float = 0.0


def _default_embedder() -> Embedder:
    """Build the default embedder using ``adk.faculties.embeddings``."""
    from adk.faculties.embeddings import get_embedding_provider  # local import to avoid cycles

    provider = get_embedding_provider()

    class _ProviderEmbedder:
        async def embed(self, text: str) -> Sequence[float]:
            return await provider.embed(text)

    return _ProviderEmbedder()


class VectorMemory:
    """Embedding-backed memory layer for agents.

    Wraps an :class:`Embedder` + :class:`VectorStore`. Use the default
    constructor for the common case (provider auto-detect, in-memory store).
    """

    def __init__(
        self,
        *,
        embedder: Embedder | EmbedFn | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self._embedder: Embedder | None = _as_embedder(embedder) if embedder is not None else None
        self._store: VectorStore = store if store is not None else InMemoryVectorStore()
        self._order: list[str] = []

    async def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = _default_embedder()
        return self._embedder

    async def add(
        self,
        content: str,
        *,
        vector_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Embed ``content`` and store it. Returns the assigned ``vector_id``."""
        emb = await self._ensure_embedder()
        vec = await emb.embed(content)
        vid = vector_id or f"vmem-{uuid.uuid4().hex[:12]}"
        payload = {
            "content": content,
            "metadata": dict(metadata or {}),
            "created": time.time(),
        }
        self._store.add(vid, vec, payload)
        if vid not in self._order:
            self._order.append(vid)
        return vid

    async def search(self, query: str, *, k: int = 5) -> list[VectorRecord]:
        """Return the top-``k`` most semantically similar records."""
        emb = await self._ensure_embedder()
        qvec = await emb.embed(query)
        hits = self._store.search(qvec, k=k)
        return [self._hit_to_record(h) for h in hits]

    async def get(self, vector_id: str) -> VectorRecord | None:
        hit = self._store.get(vector_id)
        if hit is None:
            return None
        return self._hit_to_record(hit)

    async def delete(self, vector_id: str) -> bool:
        removed = self._store.delete(vector_id)
        if removed:
            try:
                self._order.remove(vector_id)
            except ValueError:
                pass
        return removed

    async def clear(self) -> None:
        self._store.clear()
        self._order.clear()

    def ids(self) -> list[str]:
        return list(self._order)

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def _hit_to_record(hit: VectorHit) -> VectorRecord:
        p = hit.payload
        return VectorRecord(
            id=hit.id,
            content=str(p.get("content", "")),
            score=hit.score,
            metadata=dict(p.get("metadata") or {}),
            created=float(p.get("created", 0.0) or 0.0),
        )


__all__ = [
    "Embedder",
    "EmbedFn",
    "InMemoryVectorStore",
    "VectorHit",
    "VectorMemory",
    "VectorRecord",
    "VectorStore",
]
