"""Tests for Hebbian association and the association_signal_v1 benchmark."""

from __future__ import annotations

import math
from pathlib import Path

from lsm.core.association import (
    AssociationConfig,
    HebbianAssociation,
    NullAssociation,
    chain_combiners,
)
from lsm.core.embeddings import HashEmbedder
from lsm.core.habituation import ExpDecayHabituation, HabituationConfig
from lsm.core.memory import MemoryEntry


def _entry(eid: str = "x1") -> MemoryEntry:
    return MemoryEntry(id=eid, text="x")


def test_null_association_is_passthrough() -> None:
    n = NullAssociation()
    score, comps = n.combiner(0.9, _entry(), ["a", "b"])
    assert score == 0.9
    assert comps["association"] == 0.0
    n.co_hit(["a", "b"])
    assert n.weight("a", "b") == 0.0


def test_hebbian_co_hit_is_symmetric() -> None:
    a = HebbianAssociation(AssociationConfig(eta=0.5, lambda_per_second=0.0))
    a.co_hit(["k1", "k2"], now=0.0)
    assert a.weight("k1", "k2", now=0.0) == 0.5
    assert a.weight("k2", "k1", now=0.0) == 0.5
    assert a.weight("k1", "k3", now=0.0) == 0.0


def test_hebbian_co_hit_accumulates_and_decays() -> None:
    a = HebbianAssociation(AssociationConfig(eta=1.0, lambda_per_second=0.1))
    a.co_hit(["k1", "k2"], now=0.0)
    a.co_hit(["k1", "k2"], now=0.0)
    raw = a.weight("k1", "k2", now=0.0)
    assert raw == 2.0
    later = a.weight("k1", "k2", now=10.0)
    assert later < raw
    assert later == 2.0 * math.exp(-1.0) or abs(later - 2.0 * math.exp(-1.0)) < 1e-6


def test_hebbian_combiner_boosts_associated_entries() -> None:
    a = HebbianAssociation(
        AssociationConfig(eta=1.0, lambda_per_second=0.0, boost_strength=0.5)
    )
    a.co_hit(["target", "partner"], now=0.0)
    a.co_hit(["target", "partner"], now=0.0)

    score_assoc, comps_assoc = a.combiner(0.6, _entry("target"), ["partner"])
    assert score_assoc > 0.6
    assert comps_assoc["association"] > 0.0

    score_unrelated, comps_unrelated = a.combiner(0.6, _entry("stranger"), ["partner"])
    assert score_unrelated == 0.6
    assert comps_unrelated["association"] == 0.0


def test_chain_combiners_composes_habituation_and_association() -> None:
    hab = ExpDecayHabituation(
        HabituationConfig(tau_seconds=100.0, eta=1.0, damping_strength=0.4)
    )
    hab.observe("target", now=0.0)
    hab.observe("target", now=0.0)
    a = HebbianAssociation(
        AssociationConfig(eta=1.0, lambda_per_second=0.0, boost_strength=0.4)
    )
    a.co_hit(["target", "partner"], now=0.0)
    a.co_hit(["target", "partner"], now=0.0)

    combined = chain_combiners(hab.combiner, a.combiner)
    score, comps = combined(0.7, _entry("target"), ["partner"])
    assert "habituation" in comps
    assert "association" in comps
    assert comps["association"] > 0.0
    # habituation damps, association boosts — final is the signed sum
    expected = 0.7 + comps["habituation"] + comps["association"]
    assert abs(score - expected) < 1e-6


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    a = HebbianAssociation(
        AssociationConfig(eta=0.9, lambda_per_second=1e-4, boost_strength=0.7)
    )
    a.co_hit(["alpha", "beta"], now=1.0)
    a.co_hit(["alpha", "gamma"], now=2.0)

    p = tmp_path / "assoc.json"
    a.save(p)

    a2 = HebbianAssociation(AssociationConfig())
    a2.load(p)
    assert a2.config.eta == 0.9
    assert a2.config.lambda_per_second == 1e-4
    assert a2.config.boost_strength == 0.7
    assert a2.weight("alpha", "beta", now=2.0) > 0.0
    assert a2.weight("alpha", "gamma", now=2.0) > 0.0


def test_association_signal_v1_on_beats_off() -> None:
    from lsm.eval.tasks.association_signal_v1 import (
        AssocSignalConfig,
        compare_ablation,
    )

    cfg = AssocSignalConfig(num_items=12, co_hits=12, seed=1)
    result = compare_ablation(cfg, embedder=HashEmbedder(dim=64))
    on = result["with_association"]["mrr_mean"]
    off = result["without_association"]["mrr_mean"]
    # association can only add signal via a positive boost on target-partner
    # edges, so it must not *lose* to the ablation.
    assert on >= off - 1e-9
