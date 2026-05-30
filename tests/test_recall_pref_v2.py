"""Tests for recall_pref_v2 benchmark generator."""

from __future__ import annotations

from lsm.eval.tasks.recall_pref_v1 import RecallConfig
from lsm.eval.tasks.recall_pref_v2 import TASK_NAME, generate


def test_recall_pref_v2_deterministic() -> None:
    a = generate(RecallConfig(num_items=5, distractor_turns=12, seed=7))
    b = generate(RecallConfig(num_items=5, distractor_turns=12, seed=7))
    assert [x.item_id for x in a] == [x.item_id for x in b]
    assert all(x.item_id.startswith("recall_pref_v2-") for x in a)


def test_recall_pref_v2_longer_than_v1_defaults() -> None:
    items = generate(RecallConfig(num_items=3, distractor_turns=30, seed=0))
    assert len(items) == 3
    assert len(items[0].dialogue) == 2 + 2 * 30


def test_recall_pref_v2_includes_decoys_and_repeats() -> None:
    items = generate(RecallConfig(num_items=20, distractor_turns=40, seed=1))
    joined = "\n".join(u for role, u in items[0].dialogue if role == "user")
    assert "coworker" in joined or "sister" in joined or "celebrity" in joined
    user_lines = [u for role, u in items[0].dialogue if role == "user"]
    assert len(user_lines) != len(set(user_lines))


def test_task_name() -> None:
    assert TASK_NAME == "recall_pref_v2"
