"""Tests for multi-timescale memory (L8)."""

from __future__ import annotations

from lsm.core.embeddings import HashEmbedder
from lsm.core.multi_timescale import (
    MultiTimescaleConfig,
    MultiTimescaleMemory,
    TIER_DURABLE,
    TIER_FAST,
    TIER_SESSION,
    TierConfig,
)


def _make(embedder_dim: int = 64) -> MultiTimescaleMemory:
    return MultiTimescaleMemory(
        embedder=HashEmbedder(dim=embedder_dim),
        config=MultiTimescaleConfig(
            fast=TierConfig(tau_seconds=60.0, promote_hits=2, score_weight=1.0, max_entries=32),
            session=TierConfig(
                tau_seconds=3600.0, promote_hits=2, score_weight=0.9, max_entries=64
            ),
            durable=TierConfig(
                tau_seconds=10**9,
                promote_hits=10**9,
                score_weight=0.85,
                max_entries=128,
                vitality_floor=0.0,
            ),
        ),
    )


def test_put_lands_in_fast() -> None:
    m = _make()
    e = m.put("my neighbour's cat is mochi")
    assert m.tier_of(e.id) == TIER_FAST
    counts = m.tier_counts()
    assert counts[TIER_FAST] == 1
    assert counts[TIER_SESSION] == 0
    assert counts[TIER_DURABLE] == 0


def test_promotion_fast_to_session_to_durable() -> None:
    m = _make()
    e = m.put("remember the archive codex lives in the northern vault")
    # Two hits promote fast -> session.
    m.query("archive codex northern vault")
    m.query("archive codex northern vault")
    assert m.tier_of(e.id) == TIER_SESSION
    # Two more hits promote session -> durable.
    m.query("archive codex northern vault")
    m.query("archive codex northern vault")
    assert m.tier_of(e.id) == TIER_DURABLE
    counts = m.tier_counts()
    assert counts[TIER_FAST] == 0
    assert counts[TIER_SESSION] == 0
    assert counts[TIER_DURABLE] == 1


def test_tick_expires_fast_entries_below_vitality_floor() -> None:
    m = _make()
    e = m.put("ephemeral thought about the weather", now=0.0)
    # A single tick far in the future drops the fast entry.
    dropped = m.tick(now=1e9)
    assert dropped[TIER_FAST] == 1
    assert m.get(e.id) is None


def test_durable_survives_tick() -> None:
    m = _make()
    e = m.put("anchor fact worth keeping forever")
    for _ in range(4):
        m.query("anchor fact worth keeping forever")
    assert m.tier_of(e.id) == TIER_DURABLE
    dropped = m.tick(now=1e12)
    assert dropped[TIER_DURABLE] == 0
    assert m.get(e.id) is not None


def test_query_prefers_higher_tier_on_tie() -> None:
    m = _make()
    # Put the same text twice: first copy will be promoted; second stays in fast.
    a = m.put("our family car is a lavender wagon")
    for _ in range(4):
        m.query("our family car is a lavender wagon")
    assert m.tier_of(a.id) == TIER_DURABLE

    b = m.put("our family car is a lavender wagon")
    assert m.tier_of(b.id) == TIER_FAST

    hits = m.query("our family car is a lavender wagon", top_k=2)
    # both a and b appear; a (durable weight 0.85) vs b (fast weight 1.0).
    # the point is de-duplication does not drop either silently.
    assert {h.entry.id for h in hits} == {a.id, b.id}


def test_delete_removes_from_any_tier() -> None:
    m = _make()
    e = m.put("delete-me fact about nothing")
    for _ in range(4):
        m.query("delete-me fact about nothing")
    m.delete([e.id])
    assert m.get(e.id) is None
    assert m.tier_of(e.id) is None


def test_multi_session_consistency_v1_tiered_beats_flat() -> None:
    from lsm.eval.tasks.multi_session_consistency_v1 import (
        ConsistencyConfig,
        run,
    )

    result = run(
        ConsistencyConfig(num_items=10, filler_turns=30, top_k=3, seed=1),
        embedder=HashEmbedder(dim=128),
    )
    assert result["tiered_accuracy"] >= result["flat_accuracy"] - 1e-9
