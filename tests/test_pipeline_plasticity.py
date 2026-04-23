"""Integration: pipeline updates plasticity trust on outcome scoring."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.memory import InMemoryStore
from lsm.core.outcome import HeuristicOutcomeScorer
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.plasticity import ExpDecayPlasticity, PlasticityConfig
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _FixedBackend:
    name = "fake"
    model = "fake-model"

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
async def test_pipeline_plasticity_updates_on_good_outcome(tmp_path: Path) -> None:
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    seed = memory.put("our family car is a lavender wagon")
    episodic = EpisodicLog(tmp_path / "e.sqlite")

    plasticity = ExpDecayPlasticity(PlasticityConfig(eta_init=0.5, rho=0.0))
    pipe = Pipeline(
        backend=_FixedBackend(
            "your family car is a lavender wagon, as we noted together earlier"
        ),
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
