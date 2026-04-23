"""Tests for InMemoryStore (protocol-level behavior)."""

from __future__ import annotations

from lsm.core.embeddings import HashEmbedder
from lsm.core.memory import InMemoryStore, MemoryStore


def test_put_query_delete() -> None:
    store: MemoryStore = InMemoryStore(embedder=HashEmbedder(dim=32))
    a = store.put("the quick brown fox", source="a")
    store.put("lazy dog sleeps in the sun", source="b")
    store.put("kangaroos hop across meadows", source="c")
    assert store.count() == 3

    hits = store.query("fox", top_k=2)
    assert len(hits) == 2
    assert all(h.score >= -1.0 and h.score <= 1.0 for h in hits)
    # The sha-based hash embedder is not semantic, but cosine self-similarity
    # should still surface the exact-text item highest.
    exact = store.query("the quick brown fox", top_k=1)
    assert exact[0].entry.id == a.id

    n = store.delete([a.id])
    assert n == 1
    assert store.count() == 2


def test_delete_by_source() -> None:
    store = InMemoryStore(embedder=HashEmbedder(dim=16))
    store.put("x", source="s1")
    store.put("y", source="s1")
    store.put("z", source="s2")
    store.delete_by_source("s1")
    assert store.count() == 1
    rest = store.query("anything", top_k=10)
    assert all(e.entry.source == "s2" for e in rest)


def test_custom_combiner_is_used() -> None:
    store = InMemoryStore(embedder=HashEmbedder(dim=16))
    store.put("A", source="s")
    store.put("B", source="s")

    def neg(base: float, entry, context):  # type: ignore[no-untyped-def]
        return -base, {"cosine": base, "neg": 1.0}

    hits = store.query("A", top_k=2, combiner=neg)
    assert all("neg" in h.components for h in hits)
    # Scores should be negated; top-ranked after sort reversed is the least-similar
    assert hits[0].score >= hits[-1].score
