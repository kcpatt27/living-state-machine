"""Integration: pipeline runs outcome scoring after generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.memory import InMemoryStore
from lsm.core.outcome import HeuristicOutcomeScorer, OutcomeLog
from lsm.core.pipeline import Pipeline, PipelineConfig
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
async def test_pipeline_stores_outcome_after_generate(tmp_path: Path) -> None:
    episodic = EpisodicLog(tmp_path / "e.sqlite")
    outcome_log = OutcomeLog(tmp_path / "o.sqlite")
    pipe = Pipeline(
        backend=_FixedBackend(
            "your family car is a lavender wagon, as we discussed earlier this week"
        ),
        episodic=episodic,
        config=PipelineConfig(
            session_id="s", backend_name="fake", model="fake-model"
        ),
        outcome_scorer=HeuristicOutcomeScorer(),
        outcome_log=outcome_log,
    )
    await pipe.generate(
        [ChatMessage(role="user", content="what is our family car")],
        GenerateOptions(),
    )
    assert pipe.last_outcome is not None
    assert pipe.last_outcome.source == "heuristic"
    assert 0.0 <= pipe.last_outcome.score <= 1.0
    persisted = outcome_log.session("s")
    assert persisted and persisted[0].turn_id == pipe.last_outcome.turn_id
    episodic.close()
    outcome_log.close()


@pytest.mark.asyncio
async def test_outcome_feeds_writer_self_critic(tmp_path: Path) -> None:
    from lsm.core.writer import RuleBasedWriter, RuleBasedWriterConfig

    episodic = EpisodicLog(tmp_path / "e.sqlite")
    memory = InMemoryStore(embedder=HashEmbedder(dim=64))
    pipe = Pipeline(
        backend=_FixedBackend("ok"),
        episodic=episodic,
        config=PipelineConfig(
            session_id="s",
            backend_name="fake",
            model="fake-model",
            write_assistant_to_memory=True,
        ),
        memory=memory,
        writer=RuleBasedWriter(RuleBasedWriterConfig(assistant_requires_durable_hit=False)),
        outcome_scorer=HeuristicOutcomeScorer(),
    )
    await pipe.generate(
        [ChatMessage(role="user", content="please remember my preference is green tea")],
        GenerateOptions(),
    )
    # The "ok" reply should have a low outcome score and therefore the writer
    # saw a low self_critic_score (though the user message still passes the
    # durable-hit pattern and is kept).
    user_written = any(
        d.write and d.features.get("is_user", 0.0) >= 1.0
        for d in pipe.last_write_decisions
    )
    assert user_written
    assert pipe.last_outcome is not None
    assert pipe.last_outcome.score < 0.5
    episodic.close()
