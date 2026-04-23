"""Tests for outcome scoring (L9)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from lsm.core.outcome import (
    HeuristicOutcomeConfig,
    HeuristicOutcomeScorer,
    LLMSelfCriticScorer,
    NullOutcomeScorer,
    OutcomeLog,
    OutcomeRecord,
)
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _StubBackend:
    """Backend that returns the same canned text for every call."""

    name = "stub"
    model = "stub-model"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> str:
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        yield self._reply

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_null_scorer_returns_neutral() -> None:
    rec = await NullOutcomeScorer().score(user_msg="hi", assistant_reply="hello!")
    assert rec.source == "null"
    assert rec.score == 0.5
    assert rec.trust_delta == 0.0


@pytest.mark.asyncio
async def test_heuristic_penalises_empty_reply() -> None:
    rec = await HeuristicOutcomeScorer().score(user_msg="what is the date", assistant_reply="")
    assert rec.score == 0.0
    assert "empty_reply" in rec.reasons


@pytest.mark.asyncio
async def test_heuristic_penalises_echo() -> None:
    echoing = await HeuristicOutcomeScorer().score(
        user_msg="what is the password", assistant_reply="what is the password"
    )
    normal = await HeuristicOutcomeScorer().score(
        user_msg="what is the password",
        assistant_reply="the password is stored in your keychain",
    )
    assert echoing.score < normal.score
    assert "echoes_user" in echoing.reasons


@pytest.mark.asyncio
async def test_heuristic_rewards_grounded_reply() -> None:
    retrieved = ["our family car is a lavender wagon"]
    grounded = await HeuristicOutcomeScorer().score(
        user_msg="what is our family car",
        assistant_reply="your family car is a lavender wagon, as noted in our prior chat",
        retrieved=retrieved,
    )
    ungrounded = await HeuristicOutcomeScorer().score(
        user_msg="what is our family car",
        assistant_reply="I do not have that information on hand today",
        retrieved=retrieved,
    )
    assert grounded.score > ungrounded.score


@pytest.mark.asyncio
async def test_heuristic_penalises_filler_reply() -> None:
    rec = await HeuristicOutcomeScorer().score(
        user_msg="please remember my preference",
        assistant_reply="ok",
    )
    assert rec.score < 0.4


@pytest.mark.asyncio
async def test_self_critic_parses_json() -> None:
    backend = _StubBackend('{"score": 0.82, "reason": "clear and well-grounded"}')
    scorer = LLMSelfCriticScorer(backend=backend)
    rec = await scorer.score(user_msg="hi", assistant_reply="hello there")
    assert rec.source == "self_critic"
    assert rec.score == pytest.approx(0.82)
    assert rec.reasons == ["clear and well-grounded"]


@pytest.mark.asyncio
async def test_self_critic_falls_back_on_garbage() -> None:
    backend = _StubBackend("sure, looks fine to me.")
    scorer = LLMSelfCriticScorer(backend=backend)
    rec = await scorer.score(user_msg="hi", assistant_reply="hello!")
    # Fallback is heuristic, so we should get a reasonable numeric score.
    assert 0.0 <= rec.score <= 1.0
    assert rec.source == "heuristic"


def test_outcome_log_roundtrip(tmp_path: Path) -> None:
    log = OutcomeLog(tmp_path / "o.sqlite")
    log.append(
        OutcomeRecord(
            turn_id="t1",
            session_id="s",
            score=0.8,
            source="self_critic",
            reasons=["grounded"],
            features={"length_tokens": 12.0},
        )
    )
    log.append(
        OutcomeRecord(
            turn_id="t2", session_id="s", score=0.2, source="heuristic", reasons=["too_short"]
        )
    )
    records = log.session("s")
    assert len(records) == 2
    mean = log.mean_score("s")
    assert mean == pytest.approx(0.5)
    log.close()


def test_heuristic_config_thresholds() -> None:
    scorer = HeuristicOutcomeScorer(
        HeuristicOutcomeConfig(min_reply_chars=2, min_reply_tokens=1)
    )
    # does not error on tiny reply; still scores
    import asyncio

    rec = asyncio.run(scorer.score(user_msg="hi", assistant_reply="ok!"))
    assert 0.0 < rec.score < 1.0
