"""Unit tests for V.1 verifier."""

from __future__ import annotations

from pathlib import Path

import pytest
from lsm.core.facts import (
    CAT_USER_MEDIA,
    CAT_USER_PREFERENCES,
    Fact,
    FactStore,
)
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryEntry
from lsm.core.verifier import (
    NullVerifier,
    RuleBasedVerifier,
    VerifierContext,
)


@pytest.fixture
def empty_ctx() -> VerifierContext:
    return VerifierContext()


@pytest.fixture
def facts(tmp_path: Path) -> FactStore:
    store = FactStore(tmp_path)
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_PREFERENCES,
            subject="restaurants",
            value="yakiniku",
        )
    )
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_PREFERENCES,
            subject="restaurants",
            value="korean bbq",
        )
    )
    return store


# --------------------------------------------------------------------------- #
# NullVerifier
# --------------------------------------------------------------------------- #


def test_null_verifier_always_accepts(empty_ctx: VerifierContext) -> None:
    v = NullVerifier()
    out = v.score("whatever", empty_ctx)
    assert out.decision == "accept"
    assert out.energy == 0.0
    assert out.violations == ()


# --------------------------------------------------------------------------- #
# Fact contradiction
# --------------------------------------------------------------------------- #


def test_fact_contradiction_fires_on_genre_mismatch(facts: FactStore) -> None:
    v = RuleBasedVerifier()
    out = v.score(
        "Crimson Desert is a strategy game with many factions.",
        VerifierContext(facts=facts),
    )
    kinds = [vi.kind for vi in out.violations]
    assert "fact_contradiction" in kinds
    assert out.energy >= 1.5


def test_fact_contradiction_does_not_fire_when_value_matches(facts: FactStore) -> None:
    v = RuleBasedVerifier()
    out = v.score(
        "Crimson Desert is an action RPG; I've heard good things.",
        VerifierContext(facts=facts),
    )
    kinds = [vi.kind for vi in out.violations]
    assert "fact_contradiction" not in kinds


def test_fact_contradiction_no_assertion_means_no_violation(facts: FactStore) -> None:
    """Mentioning the subject without asserting a value is not a contradiction."""
    v = RuleBasedVerifier()
    out = v.score(
        "Have you played Crimson Desert recently?",
        VerifierContext(facts=facts),
    )
    kinds = [vi.kind for vi in out.violations]
    assert "fact_contradiction" not in kinds


# --------------------------------------------------------------------------- #
# Identity drift
# --------------------------------------------------------------------------- #


def test_identity_drift_fires_on_i_am_assistant_name() -> None:
    v = RuleBasedVerifier()
    ident = IdentityProfile(name="Elsum", description="x")
    out = v.score("i am Elsum and i like coffee", VerifierContext(identity=ident))
    kinds = [vi.kind for vi in out.violations]
    assert "identity_drift" in kinds


def test_identity_drift_does_not_fire_when_name_used_neutrally() -> None:
    v = RuleBasedVerifier()
    ident = IdentityProfile(name="Elsum", description="x")
    out = v.score("Hi Elsum, can you help me?", VerifierContext(identity=ident))
    kinds = [vi.kind for vi in out.violations]
    assert "identity_drift" not in kinds


# --------------------------------------------------------------------------- #
# Correction violation
# --------------------------------------------------------------------------- #


def test_correction_violation_fires_when_candidate_restates_stale() -> None:
    v = RuleBasedVerifier()
    correction = MemoryEntry(
        text="Crimson Desert is not a strategy game, it is an action RPG",
        tags=["correction"],
    )
    out = v.score(
        "I recommend strategy game tactics for Crimson Desert.",
        VerifierContext(recent_corrections=[correction]),
    )
    kinds = [vi.kind for vi in out.violations]
    assert "correction_violation" in kinds


def test_correction_violation_does_not_fire_when_canon_present() -> None:
    v = RuleBasedVerifier()
    correction = MemoryEntry(
        text="Crimson Desert is not a strategy game, it is an action RPG",
        tags=["correction"],
    )
    out = v.score(
        "Crimson Desert is an action RPG with strategy game elements maybe.",
        VerifierContext(recent_corrections=[correction]),
    )
    kinds = [vi.kind for vi in out.violations]
    assert "correction_violation" not in kinds


# --------------------------------------------------------------------------- #
# List merge
# --------------------------------------------------------------------------- #


def test_list_merge_fires_when_paired_item_dropped() -> None:
    v = RuleBasedVerifier()
    out = v.score(
        "You enjoy yakiniku restaurants.",
        VerifierContext(recent_user_turns=["i like yakiniku and korean bbq"]),
    )
    kinds = [vi.kind for vi in out.violations]
    assert "list_merge" in kinds


def test_list_merge_does_not_fire_when_both_items_present() -> None:
    v = RuleBasedVerifier()
    out = v.score(
        "You enjoy yakiniku and korean bbq.",
        VerifierContext(recent_user_turns=["i like yakiniku and korean bbq"]),
    )
    kinds = [vi.kind for vi in out.violations]
    assert "list_merge" not in kinds


# --------------------------------------------------------------------------- #
# Length overflow
# --------------------------------------------------------------------------- #


def test_length_overflow_fires_on_long_slot_value() -> None:
    v = RuleBasedVerifier()
    long_value = "x" * 200
    candidate = f"preferences/likes: {long_value}"
    out = v.score(candidate, VerifierContext())
    kinds = [vi.kind for vi in out.violations]
    assert "length_overflow" in kinds


def test_length_overflow_does_not_fire_on_short_value() -> None:
    v = RuleBasedVerifier()
    out = v.score("preferences/likes: tea", VerifierContext())
    kinds = [vi.kind for vi in out.violations]
    assert "length_overflow" not in kinds


# --------------------------------------------------------------------------- #
# Thresholds + decision logic
# --------------------------------------------------------------------------- #


def test_below_reprompt_threshold_accepts() -> None:
    v = RuleBasedVerifier(threshold_reprompt=1.0, threshold_reject=2.5)
    # Only a length_overflow violation -> weight 0.2 -> accept.
    candidate = "x/y: " + ("a" * 200)
    out = v.score(candidate, VerifierContext())
    assert out.decision == "accept"


def test_between_thresholds_reprompts(facts: FactStore) -> None:
    v = RuleBasedVerifier(threshold_reprompt=1.0, threshold_reject=2.5)
    # One contradiction (weight 1.5) -> falls in reprompt band.
    out = v.score(
        "Crimson Desert is a strategy game.",
        VerifierContext(facts=facts),
    )
    assert out.decision == "reprompt"


def test_above_reject_threshold_rejects(facts: FactStore) -> None:
    v = RuleBasedVerifier(threshold_reprompt=1.0, threshold_reject=2.5)
    ident = IdentityProfile(name="Elsum", description="x")
    correction = MemoryEntry(
        text="Crimson Desert is not a strategy game, it is an action RPG",
    )
    # contradiction (1.5) + correction_violation (1.5) + identity_drift (1.5)
    # = 4.5 -> reject.
    candidate = (
        "Crimson Desert is a strategy game. i am Elsum so let's discuss strategy game tactics."
    )
    out = v.score(
        candidate,
        VerifierContext(
            facts=facts,
            identity=ident,
            recent_corrections=[correction],
        ),
    )
    assert out.decision == "reject"


# --------------------------------------------------------------------------- #
# Weights override
# --------------------------------------------------------------------------- #


def test_weights_override_flips_borderline() -> None:
    """Bumping length_overflow weight pushes a length-only candidate over the bar."""
    v = RuleBasedVerifier(
        threshold_reprompt=1.0,
        weights={"length_overflow": 2.0},
    )
    candidate = "slot: " + ("a" * 200)
    out = v.score(candidate, VerifierContext())
    assert out.decision == "reprompt"


# --------------------------------------------------------------------------- #
# Determinism + empty input
# --------------------------------------------------------------------------- #


def test_score_is_deterministic(facts: FactStore) -> None:
    v = RuleBasedVerifier()
    ctx = VerifierContext(facts=facts)
    candidate = "Crimson Desert is a strategy game."
    a = v.score(candidate, ctx)
    b = v.score(candidate, ctx)
    assert a == b


def test_empty_candidate_accepts() -> None:
    v = RuleBasedVerifier()
    out = v.score("", VerifierContext())
    assert out.decision == "accept"
    assert out.violations == ()


def test_fix_hints_match_violations(facts: FactStore) -> None:
    v = RuleBasedVerifier()
    out = v.score(
        "Crimson Desert is a strategy game.",
        VerifierContext(facts=facts),
    )
    assert len(out.fix_hints) == len(out.violations) >= 1
    # Each hint is non-empty and labels its kind.
    for hint, vio in zip(out.fix_hints, out.violations, strict=True):
        assert vio.kind in hint
        assert vio.detail in hint
