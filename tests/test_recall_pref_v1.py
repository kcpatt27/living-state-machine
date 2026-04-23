"""Tests for the recall_pref_v1 benchmark generator + grader."""

from __future__ import annotations

from lsm.eval.metrics import any_keyword_match
from lsm.eval.tasks.recall_pref_v1 import RecallConfig, generate


def test_deterministic_generation() -> None:
    a = generate(RecallConfig(num_items=10, distractor_turns=10, seed=42))
    b = generate(RecallConfig(num_items=10, distractor_turns=10, seed=42))
    assert [x.correct for x in a] == [y.correct for y in b]
    assert [x.dialogue for x in a] == [y.dialogue for y in b]


def test_distractor_density() -> None:
    items = generate(RecallConfig(num_items=3, distractor_turns=20, seed=0))
    for it in items:
        # 2 seed turns + 2 * distractor_turns
        assert len(it.dialogue) == 2 + 2 * 20
        assert it.dialogue[0][0] == "user"
        assert it.dialogue[1][0] == "assistant"
        assert it.correct in it.dialogue[0][1]
        assert it.topic in it.question


def test_grading_matches_keyword() -> None:
    it = generate(RecallConfig(num_items=1, distractor_turns=5, seed=7))[0]
    assert any_keyword_match(f"it's {it.correct} of course", [it.correct])
    assert not any_keyword_match("no idea", [it.correct])
