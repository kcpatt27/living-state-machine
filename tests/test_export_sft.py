"""Tests for SFT export from episodic log."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.export_sft_from_lsm_home import build_sft_rows, export_sft


def _seed_episodes(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE turns (turn_id TEXT PRIMARY KEY, session_id TEXT, role TEXT, "
            "content TEXT, ts REAL, meta TEXT)"
        )
        conn.executemany(
            "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("u1", "s1", "user", "I prefer negroni.", 1.0, "{}"),
                ("a1", "s1", "assistant", "Got it.", 2.0, '{"in_reply_to":"u1"}'),
                ("u2", "s1", "user", "thanks", 3.0, "{}"),
                ("a2", "s1", "assistant", "You're welcome.", 4.0, '{"in_reply_to":"u2"}'),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _seed_verifier_reject(path: Path, candidate: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE verifier_events (id TEXT PRIMARY KEY, candidate TEXT, decision TEXT)"
        )
        conn.execute(
            "INSERT INTO verifier_events VALUES (?, ?, ?)",
            ("e1", candidate, "reject"),
        )
        conn.commit()
    finally:
        conn.close()


def test_build_sft_rows_pairs_user_assistant() -> None:
    turns = [
        {"session_id": "s1", "role": "user", "content": "hello", "meta": {}},
        {"session_id": "s1", "role": "assistant", "content": "hi there", "meta": {}},
    ]
    rows = build_sft_rows(turns, rejected=set())
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "hello"
    assert rows[0]["messages"][1]["content"] == "hi there"


def test_export_sft_skips_verifier_rejects(tmp_path: Path) -> None:
    epis = tmp_path / "episodes.sqlite"
    verifier = tmp_path / "verifier_log.sqlite"
    out = tmp_path / "sft.jsonl"
    _seed_episodes(epis)
    _seed_verifier_reject(verifier, "Got it.")

    stats = export_sft(episodes_db=epis, verifier_db=verifier, out_path=out)
    assert stats["pairs_written"] == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(lines[0])
    assert row["messages"][1]["content"] == "You're welcome."
