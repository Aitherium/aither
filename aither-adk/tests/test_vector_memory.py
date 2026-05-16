"""Tests for adk.vector_memory."""

from __future__ import annotations

import math

import pytest

from adk.vector_memory import (
    InMemoryVectorStore,
    VectorHit,
    VectorMemory,
    VectorRecord,
    _cosine,
)


# ─── Fake embedder ───────────────────────────────────────────────────────────


class _StaticEmbedder:
    """Maps a fixed set of phrases to fixed 3-d vectors; unknowns get hash."""

    def __init__(self) -> None:
        self.table: dict[str, list[float]] = {
            "atlas owns the registry": [1.0, 0.0, 0.0],
            "prom runs the metrics": [0.0, 1.0, 0.0],
            "muse paints the dashboard": [0.0, 0.0, 1.0],
            "who runs the registry?": [0.95, 0.05, 0.0],
            "who handles metrics?": [0.05, 0.95, 0.0],
        }

    async def embed(self, text: str) -> list[float]:
        if text in self.table:
            return list(self.table[text])
        # Stable fallback so the test is deterministic for new strings.
        h = abs(hash(text)) % 1000
        v = [(h % 7) / 7.0, (h % 11) / 11.0, (h % 13) / 13.0]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


# ─── _cosine ────────────────────────────────────────────────────────────────


class TestCosine:
    def test_identical(self):
        assert _cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0, 0, 0], [1, 1, 1]) == 0.0

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            _cosine([1, 0], [1, 0, 0])


# ─── InMemoryVectorStore ────────────────────────────────────────────────────


class TestInMemoryStore:
    def test_add_and_get(self):
        s = InMemoryVectorStore()
        s.add("a", [1, 0, 0], {"k": "v"})
        hit = s.get("a")
        assert hit is not None
        assert hit.id == "a"
        assert hit.score == 1.0
        assert hit.payload == {"k": "v"}

    def test_get_missing(self):
        assert InMemoryVectorStore().get("missing") is None

    def test_search_orders_by_similarity(self):
        s = InMemoryVectorStore()
        s.add("near", [1.0, 0.0, 0.0], {})
        s.add("ortho", [0.0, 1.0, 0.0], {})
        s.add("opp", [-1.0, 0.0, 0.0], {})
        hits = s.search([0.95, 0.05, 0.0], k=3)
        assert [h.id for h in hits] == ["near", "ortho", "opp"]
        assert hits[0].score > hits[1].score > hits[2].score

    def test_search_k_limits(self):
        s = InMemoryVectorStore()
        s.add("a", [1, 0, 0], {})
        s.add("b", [0, 1, 0], {})
        assert len(s.search([1, 0, 0], k=1)) == 1

    def test_search_empty(self):
        assert InMemoryVectorStore().search([1, 0, 0], k=5) == []

    def test_dim_mismatch_on_add(self):
        s = InMemoryVectorStore()
        s.add("a", [1, 0, 0], {})
        with pytest.raises(ValueError):
            s.add("b", [1, 0], {})

    def test_dim_mismatch_on_search(self):
        s = InMemoryVectorStore()
        s.add("a", [1, 0, 0], {})
        with pytest.raises(ValueError):
            s.search([1, 0], k=1)

    def test_delete(self):
        s = InMemoryVectorStore()
        s.add("a", [1, 0, 0], {})
        assert s.delete("a") is True
        assert s.delete("a") is False
        assert len(s) == 0

    def test_clear_resets_dim(self):
        s = InMemoryVectorStore()
        s.add("a", [1, 0, 0], {})
        s.clear()
        assert s.dim is None
        # Now a different-dim vector is accepted.
        s.add("b", [1, 0], {})
        assert s.dim == 2

    def test_all_ids(self):
        s = InMemoryVectorStore()
        s.add("a", [1, 0], {})
        s.add("b", [0, 1], {})
        assert set(s.all_ids()) == {"a", "b"}


# ─── VectorMemory ───────────────────────────────────────────────────────────


class TestVectorMemory:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        vmem = VectorMemory(embedder=_StaticEmbedder())
        await vmem.add("atlas owns the registry", vector_id="atlas")
        await vmem.add("prom runs the metrics", vector_id="prom")
        await vmem.add("muse paints the dashboard", vector_id="muse")

        hits = await vmem.search("who runs the registry?", k=2)
        assert isinstance(hits[0], VectorRecord)
        assert hits[0].id == "atlas"
        assert hits[0].score > hits[1].score

    @pytest.mark.asyncio
    async def test_metric_query(self):
        vmem = VectorMemory(embedder=_StaticEmbedder())
        await vmem.add("atlas owns the registry", vector_id="atlas")
        await vmem.add("prom runs the metrics", vector_id="prom")
        hits = await vmem.search("who handles metrics?", k=1)
        assert hits[0].id == "prom"

    @pytest.mark.asyncio
    async def test_metadata_round_trip(self):
        vmem = VectorMemory(embedder=_StaticEmbedder())
        await vmem.add(
            "atlas owns the registry",
            vector_id="atlas",
            metadata={"owner": "ops"},
        )
        rec = await vmem.get("atlas")
        assert rec is not None
        assert rec.metadata == {"owner": "ops"}
        assert rec.content == "atlas owns the registry"

    @pytest.mark.asyncio
    async def test_auto_id(self):
        vmem = VectorMemory(embedder=_StaticEmbedder())
        vid = await vmem.add("atlas owns the registry")
        assert vid.startswith("vmem-")
        assert vid in vmem.ids()

    @pytest.mark.asyncio
    async def test_delete(self):
        vmem = VectorMemory(embedder=_StaticEmbedder())
        await vmem.add("atlas owns the registry", vector_id="atlas")
        assert await vmem.delete("atlas") is True
        assert await vmem.get("atlas") is None
        assert "atlas" not in vmem.ids()

    @pytest.mark.asyncio
    async def test_clear(self):
        vmem = VectorMemory(embedder=_StaticEmbedder())
        await vmem.add("atlas owns the registry")
        await vmem.add("prom runs the metrics")
        await vmem.clear()
        assert len(vmem) == 0
        assert vmem.ids() == []

    @pytest.mark.asyncio
    async def test_callable_embedder(self):
        async def emb(text: str) -> list[float]:
            return [1.0, 0.0] if "a" in text else [0.0, 1.0]

        vmem = VectorMemory(embedder=emb)
        await vmem.add("apple", vector_id="apple")
        await vmem.add("zoo", vector_id="zoo")
        hits = await vmem.search("apricot", k=1)
        assert hits[0].id == "apple"

    @pytest.mark.asyncio
    async def test_hit_to_record_score_preserved(self):
        vmem = VectorMemory(embedder=_StaticEmbedder())
        await vmem.add("atlas owns the registry", vector_id="atlas")
        hits = await vmem.search("atlas owns the registry", k=1)
        assert hits[0].score == pytest.approx(1.0)
