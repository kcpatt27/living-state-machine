from __future__ import annotations

from pathlib import Path

from lsm.core.episodic import EpisodicLog, TurnRecord
from lsm.core.verifier import Verdict, VerifierContext
from lsm.core.verifier_log import VerifierEvent, VerifierLog, build_context_record
from lsm.verifier_data.correction_miner import mine_correction_examples
from lsm.verifier_data.export import export_verifier_dataset
from lsm.verifier_data.perturb import perturb_from_events


def test_correction_miner_finds_corrected_assistant_turn(tmp_path: Path) -> None:
    home = tmp_path / "home"
    episodic = EpisodicLog(home / "episodes.sqlite")
    try:
        a_turn = TurnRecord(
            session_id="s1", role="assistant", content="Crimson Desert is strategy game."
        )
        episodic.append(a_turn)
        episodic.append(
            TurnRecord(
                session_id="s1",
                role="user",
                content="no, that's wrong. it's action RPG.",
            )
        )
    finally:
        episodic.close()

    vlog = VerifierLog(home / "verifier_log.sqlite")
    try:
        ctx = VerifierContext()
        rec = build_context_record("ev1", ctx)
        vlog.append(
            source="reply",
            candidate=a_turn.content,
            verdict=Verdict(energy=1.2, decision="reprompt"),
            context=rec,
            event_id="ev1",
        )
    finally:
        vlog.close()

    mined = mine_correction_examples(home, limit=10)
    assert mined
    assert mined[0].candidate == a_turn.content
    assert mined[0].label == "reject"


def test_perturb_from_events_produces_negative_variant() -> None:
    event = VerifierEvent(
        id="ev1",
        ts=0.0,
        source="reply",
        candidate="I like apples and oranges.",
        energy=0.0,
        decision="accept",
        violations=(),
        reprompt_used=False,
    )
    out = perturb_from_events([(event, None)], limit=2, seed=0)
    assert out
    assert out[0].label == "reject"
    assert out[0].candidate != event.candidate


def test_export_verifier_dataset_writes_distillation_rows(tmp_path: Path) -> None:
    home = tmp_path / "home"
    vlog = VerifierLog(home / "verifier_log.sqlite")
    try:
        rec = build_context_record("ev1", VerifierContext())
        vlog.append(
            source="reply",
            candidate="hello",
            verdict=Verdict(energy=0.0, decision="accept"),
            context=rec,
            event_id="ev1",
        )
    finally:
        vlog.close()

    out = tmp_path / "dataset.jsonl"
    rows = export_verifier_dataset(
        home,
        out_path=out,
        include_mined=False,
        include_perturb=False,
        limit=10,
    )
    assert out.exists()
    assert rows
    assert rows[0].label_source == "rule_distill"
