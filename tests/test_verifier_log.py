"""Tests for V.1 verifier telemetry sqlite log."""

from __future__ import annotations

from pathlib import Path

from lsm.core.facts import Fact, FactStore
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryEntry
from lsm.core.verifier import Verdict, VerifierContext, Violation
from lsm.core.verifier_log import VerifierLog, build_context_record


def test_verifier_log_append_and_recent(tmp_path: Path) -> None:
    log = VerifierLog(tmp_path / "verifier_log.sqlite")
    verdict = Verdict(
        energy=1.5,
        violations=(
            Violation(
                kind="list_merge",
                detail="dropped item",
                weight=1.5,
                evidence="x",
            ),
        ),
        decision="reprompt",
        fix_hints=("fix it",),
    )
    eid = log.append(
        source="reply",
        candidate="Crimson Desert is a strategy game",
        verdict=verdict,
        reprompt_used=True,
    )
    assert eid
    rows = log.recent(limit=5)
    assert len(rows) == 1
    assert rows[0].source == "reply"
    assert rows[0].decision == "reprompt"
    assert rows[0].reprompt_used is True
    assert rows[0].energy == 1.5
    assert rows[0].violations[0]["kind"] == "list_merge"
    assert log.count() == 1
    log.close()


def test_verifier_log_persists_context_snapshot(tmp_path: Path) -> None:
    log = VerifierLog(tmp_path / "verifier_log.sqlite")
    facts = FactStore(tmp_path / "home")
    facts.upsert(Fact(owner="user", category="user.preferences", subject="drink", value="matcha"))
    ctx = VerifierContext(
        facts=facts,
        identity=IdentityProfile(name="Elsum", description="assistant"),
        recent_corrections=(
            MemoryEntry(text="it's not strategy game, it's action RPG", source="conversation"),
        ),
        recent_user_turns=("i like tea and coffee",),
    )
    verdict = Verdict(energy=0.0, decision="accept")
    eid = "event-1"
    rec = build_context_record(eid, ctx)
    log.append(source="writer", candidate="ok", verdict=verdict, context=rec, event_id=eid)
    loaded = log.context_for(eid)
    assert loaded is not None
    assert loaded.event_id == eid
    assert loaded.identity_snapshot is not None
    assert loaded.identity_snapshot["name"] == "Elsum"
    assert loaded.recent_user_turns == ("i like tea and coffee",)
    assert loaded.facts_snapshot
    assert loaded.facts_snapshot[0]["value"] == "matcha"
    assert len(loaded.ctx_hash) == 64
    log.close()
