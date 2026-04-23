"""Tests for meta-memory / plasticity scheduling (L10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lsm.core.memory import MemoryEntry
from lsm.core.plasticity import (
    ExpDecayPlasticity,
    NullPlasticity,
    PlasticityConfig,
)


def _entry(eid: str) -> MemoryEntry:
    return MemoryEntry(id=eid, text="x")


def test_null_plasticity_is_passthrough() -> None:
    p = NullPlasticity()
    p.observe(["a"], target=1.0)
    assert p.trust("a") == 0.5
    score, comps = p.combiner(0.7, _entry("a"), [])
    assert score == 0.7
    assert comps["trust"] == 0.5


def test_exp_decay_trust_moves_toward_target() -> None:
    p = ExpDecayPlasticity(PlasticityConfig(eta_init=0.5, rho=0.0))
    p.observe(["k"], target=1.0)
    t1 = p.trust("k")
    assert t1 == pytest.approx(0.75)  # (1-0.5)*0.5 + 0.5*1.0
    p.observe(["k"], target=1.0)
    t2 = p.trust("k")
    # eta=0.5, new_trust = 0.5*0.75 + 0.5*1 = 0.875
    assert t2 == pytest.approx(0.875)


def test_exp_decay_plasticity_shrinks_with_consistent_evidence() -> None:
    p = ExpDecayPlasticity(PlasticityConfig(eta_init=0.4, rho=0.5, eta_min=0.01))
    st0 = p.state_of("k").plasticity
    # Feed consistent evidence: every observation pulls trust toward 0.9.
    for _ in range(20):
        p.observe(["k"], target=0.9)
    st1 = p.state_of("k").plasticity
    assert st1 < st0  # consistent evidence -> less plastic


def test_exp_decay_plasticity_grows_with_volatile_evidence() -> None:
    p = ExpDecayPlasticity(PlasticityConfig(eta_init=0.1, rho=0.5, eta_max=0.9))
    for i in range(20):
        target = 0.95 if i % 2 == 0 else 0.05
        p.observe(["k"], target=target)
    st = p.state_of("k").plasticity
    assert st > 0.3  # volatility pushes plasticity up


def test_combiner_boosts_trusted_entries() -> None:
    p = ExpDecayPlasticity(PlasticityConfig(eta_init=0.5, rho=0.0, trust_boost=0.4))
    # Drive trust up for "a".
    for _ in range(10):
        p.observe(["a"], target=1.0)
    high, _ = p.combiner(0.6, _entry("a"), [])
    neutral, _ = p.combiner(0.6, _entry("fresh"), [])
    assert high > neutral
    # Drive trust down for "b".
    for _ in range(10):
        p.observe(["b"], target=0.0)
    low, _ = p.combiner(0.6, _entry("b"), [])
    assert low < neutral


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    p = ExpDecayPlasticity(PlasticityConfig(eta_init=0.4, rho=0.2))
    p.observe(["a", "b"], target=0.9)
    path = tmp_path / "p.json"
    p.save(path)

    p2 = ExpDecayPlasticity()
    p2.load(path)
    assert p2.trust("a") == pytest.approx(p.trust("a"))
    assert p2.state_of("b").plasticity == pytest.approx(p.state_of("b").plasticity)
