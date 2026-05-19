"""Integration: pipeline updates plasticity trust on outcome scoring."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import cast

import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.memory import InMemoryStore, MemoryEntry
from lsm.core.outcome import HeuristicOutcomeScorer
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.plasticity import (
    ExpDecayPlasticity,
    NullPlasticity,
    PlasticityConfig,
    PlasticityState,
)
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _FixedBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        return self._reply

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        yield self._reply

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_pipeline_plasticity_updates_on_good_outcome(tmp_path: Path) -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    seed = memory.put("our family car is a lavender wagon")
    episodic = EpisodicLog(tmp_path / "e.sqlite")

    plasticity = ExpDecayPlasticity(PlasticityConfig(eta_init=0.5, rho=0.0))
    pipe = Pipeline(
        backend=_FixedBackend("your family car is a lavender wagon, as we noted together earlier"),
        episodic=episodic,
        config=PipelineConfig(
            session_id="s",
            backend_name="fake",
            model="fake-model",
            retrieve_top_k=3,
            retrieve_min_score=-1.0,
        ),
        memory=memory,
        outcome_scorer=HeuristicOutcomeScorer(),
        plasticity=plasticity,
    )
    await pipe.generate(
        [ChatMessage(role="user", content="what kind of family car do we have")],
        GenerateOptions(),
    )
    trust_after = plasticity.trust(seed.id)
    assert trust_after > 0.5
    episodic.close()


class _SpyPlasticity(NullPlasticity):
    """Records ``observe`` calls so we can assert pipeline wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def observe(
        self,
        entry_ids: Sequence[str],
        target: float,
        now: float | None = None,
        *,
        surprise: float = 1.0,
    ) -> None:
        self.calls.append(
            {
                "entry_ids": list(entry_ids),
                "target": float(target),
                "surprise": float(surprise),
            }
        )

    def state_of(self, entry_id: str) -> PlasticityState:
        return PlasticityState(trust=0.5, plasticity=self.config.eta_init)


def _seeded_pipeline(
    tmp_path: Path,
    *,
    reply: str,
    correction_dynamics_enabled: bool = True,
) -> tuple[Pipeline, _SpyPlasticity, EpisodicLog, MemoryEntry]:
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    seed = memory.put("crimson desert is a strategy game")
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    spy = _SpyPlasticity()
    pipe = Pipeline(
        backend=_FixedBackend(reply),
        episodic=episodic,
        config=PipelineConfig(
            session_id="s",
            backend_name="fake",
            model="fake-model",
            retrieve_top_k=3,
            retrieve_min_score=-1.0,
            correction_dynamics_enabled=correction_dynamics_enabled,
            correction_surprise=2.0,
            correction_target=0.2,
        ),
        memory=memory,
        outcome_scorer=HeuristicOutcomeScorer(),
        plasticity=spy,
    )
    return pipe, spy, episodic, seed


@pytest.mark.asyncio
async def test_pipeline_correction_turn_triggers_surprise(tmp_path: Path) -> None:
    pipe, spy, episodic, seed = _seeded_pipeline(
        tmp_path, reply="acknowledged, crimson desert is a strategy game"
    )
    try:
        await pipe.generate(
            [
                ChatMessage(
                    role="user",
                    content="actually, Crimson Desert is an action RPG",
                )
            ],
            GenerateOptions(),
        )
    finally:
        episodic.close()

    assert spy.calls, "plasticity.observe was never called"
    last = spy.calls[-1]
    ids = cast(list[str], last["entry_ids"])
    surprise = cast(float, last["surprise"])
    target = cast(float, last["target"])
    assert seed.id in ids
    assert surprise == pytest.approx(2.0)
    assert target <= 0.2 + 1e-9


@pytest.mark.asyncio
async def test_pipeline_normal_turn_uses_default_surprise(tmp_path: Path) -> None:
    pipe, spy, episodic, seed = _seeded_pipeline(
        tmp_path, reply="sure, here's some info about strategy games"
    )
    try:
        await pipe.generate(
            [ChatMessage(role="user", content="tell me about strategy games")],
            GenerateOptions(),
        )
    finally:
        episodic.close()

    assert spy.calls, "plasticity.observe was never called"
    last = spy.calls[-1]
    ids = cast(list[str], last["entry_ids"])
    surprise = cast(float, last["surprise"])
    assert seed.id in ids
    assert surprise == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_pipeline_correction_disabled_flag(tmp_path: Path) -> None:
    pipe, spy, episodic, _seed = _seeded_pipeline(
        tmp_path,
        reply="ok",
        correction_dynamics_enabled=False,
    )
    try:
        await pipe.generate(
            [
                ChatMessage(
                    role="user",
                    content="actually, Crimson Desert is an action RPG",
                )
            ],
            GenerateOptions(),
        )
    finally:
        episodic.close()

    assert spy.calls, "plasticity.observe was never called"
    last = spy.calls[-1]
    surprise = cast(float, last["surprise"])
    # Even on a correction-shaped message, the master switch keeps the
    # baseline surprise + unbiased target.
    assert surprise == pytest.approx(1.0)
