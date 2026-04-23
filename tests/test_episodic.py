"""Tests for the episodic log."""

from __future__ import annotations

from pathlib import Path

from lsm.core.episodic import EpisodicLog, TurnRecord


def test_append_and_read(tmp_path: Path) -> None:
    log = EpisodicLog(tmp_path / "e.sqlite")
    tid = log.append(TurnRecord(session_id="s1", role="user", content="hi"))
    log.append(TurnRecord(session_id="s1", role="assistant", content="hello"))
    log.append(TurnRecord(session_id="s2", role="user", content="other"))

    rows = log.session("s1")
    assert len(rows) == 2
    assert rows[0].role == "user"
    assert rows[0].content == "hi"
    assert rows[0].turn_id == tid
    assert rows[1].role == "assistant"
    assert log.count() == 3
    log.close()


def test_meta_roundtrip(tmp_path: Path) -> None:
    log = EpisodicLog(tmp_path / "e.sqlite")
    log.append(
        TurnRecord(
            session_id="s1",
            role="assistant",
            content="x",
            meta={"model": "qwen", "score": 0.9},
        )
    )
    [row] = log.session("s1")
    assert row.meta == {"model": "qwen", "score": 0.9}
    log.close()
