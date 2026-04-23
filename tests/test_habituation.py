"""Tests for habituation and the retrieval-layer benchmark."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lsm.core.embeddings import HashEmbedder
from lsm.core.habituation import (
    ExpDecayHabituation,
    HabituationConfig,
    NullHabituation,
)
from lsm.core.memory import MemoryEntry


def _entry(eid: str = "k1") -> MemoryEntry:
    return MemoryEntry(id=eid, text="x")


def test_null_habituation_is_passthrough() -> None:
    n = NullHabituation()
    score, comps = n.combiner(0.9, _entry(), [])
    assert score == 0.9
    assert comps["habituation"] == 0.0
    n.observe("k1")
    assert n.value("k1") == 0.0


def test_exp_decay_grows_on_hit_and_damps_score() -> None:
    h = ExpDecayHabituation(HabituationConfig(tau_seconds=100.0, eta=1.0, damping_strength=0.5))
    h.observe("k1", now=0.0)
    h.observe("k1", now=1.0)
    h.observe("k1", now=2.0)
    v = h.value("k1", now=2.0)
    assert v > 1.5

    score, comps = h.combiner(0.8, _entry("k1"), [], now=2.0)
    assert score < 0.8
    assert comps["habituation"] > 0
    # untouched key is not damped
    fresh_score, _ = h.combiner(0.8, _entry("unseen"), [], now=2.0)
    assert fresh_score == 0.8


def test_exp_decay_decays_over_time() -> None:
    h = ExpDecayHabituation(HabituationConfig(tau_seconds=10.0, eta=1.0))
    h.observe("k1", now=0.0)
    v_0 = h.value("k1", now=0.0)
    v_later = h.value("k1", now=30.0)
    assert v_later < v_0
    assert v_later == pytest.approx(math.exp(-3.0), rel=1e-6)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = HabituationConfig(tau_seconds=123.0, eta=0.7, damping_strength=0.42)
    h = ExpDecayHabituation(cfg)
    h.observe("a", now=1.0)
    h.observe("b", now=2.0)
    h.observe("a", now=3.0)

    p = tmp_path / "h.json"
    h.save(p)

    h2 = ExpDecayHabituation(HabituationConfig())
    h2.load(p)
    assert h2.config.tau_seconds == 123.0
    assert h2.config.eta == 0.7
    assert h2.config.damping_strength == 0.42
    assert h2.value("a", now=3.0) == h.value("a", now=3.0)


def test_habituation_signal_v1_ablation_improves_mrr() -> None:
    from lsm.eval.tasks.habituation_signal_v1 import HabSignalConfig, compare_ablation

    cfg = HabSignalConfig(num_items=12, n_filler_ids=6, filler_hits=4, seed=1)
    result = compare_ablation(cfg, embedder=HashEmbedder(dim=64))
    # We don't require a huge delta for this synthetic retrieval task with a
    # hash embedder, but habituation-on should never *lose* to habituation-off,
    # because NullHabituation is strictly less informative than ExpDecay.
    on = result["with_habituation"]["mrr_mean"]
    off = result["without_habituation"]["mrr_mean"]
    assert on >= off - 1e-9
