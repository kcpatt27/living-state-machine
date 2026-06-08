"""Tests for recall_pref bootstrap SFT."""

from __future__ import annotations

from scripts.bootstrap_sft_recall_pref import build_bootstrap_rows


def test_bootstrap_row_count_and_shape() -> None:
    rows = build_bootstrap_rows(count=20, seed=0)
    assert len(rows) == 20
    for row in rows:
        messages = row["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
