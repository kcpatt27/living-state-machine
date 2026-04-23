"""Integration: pipeline wires habituation and association into retrieval."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from lsm.core.association import AssociationConfig, HebbianAssociation
from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.habituation import ExpDecayHabituation, HabituationConfig
from lsm.core.memory import InMemoryStore
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _NoopBackend:
    name = "fake"
    model = "fake-model"

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> str:
        return "ok"

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        yield "ok"

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_pipeline_calls_association_on_retrieval(tmp_path: Path) -> None:
    embedder = HashEmbedder(dim=64)
    memory = InMemoryStore(embedder=embedder)
    target = memory.put("the northern vault holds the rare codex", source="test")
    partner = memory.put("moonstone compass for the north wing", source="test")
    memory.put("the north wing is sunny", source="test")

    assoc = HebbianAssociation(
        AssociationConfig(eta=1.0, lambda_per_second=0.0, boost_strength=0.4)
    )
    hab = ExpDecayHabituation(HabituationConfig(tau_seconds=1e9, eta=0.0))

    episodic = EpisodicLog(tmp_path / "e.sqlite")
    pipe = Pipeline(
        backend=_NoopBackend(),
        episodic=episodic,
        config=PipelineConfig(
            session_id="s",
            backend_name="fake",
            model="fake-model",
            retrieve_top_k=5,
            retrieve_min_score=-1.0,
            recent_context_window=4,
        ),
        memory=memory,
        habituation=hab,
        association=assoc,
    )
    pipe._recent_ids.append(partner.id)
    assoc.co_hit([target.id, partner.id])
    assoc.co_hit([target.id, partner.id])
    assoc.co_hit([target.id, partner.id])

    await pipe.generate(
        [ChatMessage(role="user", content="what is in the northern vault")],
        GenerateOptions(),
    )
    retrieved_ids = [e.entry.id for e in pipe.last_retrieved]
    assert target.id in retrieved_ids
    # partner co-hit with target happens via pipeline on this turn, so the
    # association weight should have grown further
    assert assoc.weight(target.id, partner.id) > 3.0

    episodic.close()
