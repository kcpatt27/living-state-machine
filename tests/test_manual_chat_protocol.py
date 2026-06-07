"""Automated manual chat protocol (docs/manual_chat_protocol.md).

Uses extractable phrasing the current rule extractor accepts (``I prefer …``),
not ``my favorite X is Y`` (no dedicated regex — learning > hardcode).
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.facts import CAT_USER_PREFERENCES, FactStore
from lsm.core.memory import InMemoryStore
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.projection import DeterministicProjection
from lsm.core.state import StateStore
from lsm.core.verifier import NullVerifier
from lsm.core.verifier_log import VerifierLog
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions
from lsm.runtime.state_updater_factory import DEFAULT_HYBRID_CKPT, build_state_updater

has_torch = importlib.util.find_spec("torch") is not None
_hybrid_ckpt = Path(DEFAULT_HYBRID_CKPT)


class _EchoBackend:
    name = "scripted"
    model = "scripted"

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        _ = messages, options
        return "acknowledged"

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        _ = messages, options
        yield "acknowledged"

    async def close(self) -> None:
        return None


def _build_pipe(
    tmp_path: Path,
    *,
    state_updater_kind: str = "facts_backed",
    hybrid_device: str = "cpu",
) -> tuple[Pipeline, FactStore, InMemoryStore, StateStore]:
    embedder = HashEmbedder(dim=128)
    facts = FactStore(tmp_path / "home")
    memory = InMemoryStore(embedder)
    store = StateStore(tmp_path / "home")
    projection = DeterministicProjection()
    updater = build_state_updater(
        state_updater_kind,
        backend=_EchoBackend(),
        embedder=embedder,
        memory=memory,
        fact_store=facts,
        projection=projection,
        hybrid_cfg={
            "ckpt_path": str(_hybrid_ckpt),
            "device": hybrid_device,
            "summary_tokens": 8,
        },
    )
    cfg = PipelineConfig(
        session_id="protocol",
        backend_name="scripted",
        model="scripted",
        state_updater=state_updater_kind,  # type: ignore[arg-type]
        write_user_to_memory=True,
        write_assistant_to_memory=True,
        retrieve_facts_enabled=True,
    )
    pipe = Pipeline(
        backend=_EchoBackend(),
        episodic=EpisodicLog(tmp_path / "episodes.sqlite"),
        config=cfg,
        state_store=store,
        state_updater=updater,
        memory=memory,
        fact_store=facts,
        embedder=embedder,
        verifier=NullVerifier(),
        verifier_log=VerifierLog(tmp_path / "verifier_log.sqlite"),
    )
    return pipe, facts, memory, store


async def _turn(pipe: Pipeline, user_msg: str) -> None:
    await pipe.generate([ChatMessage(role="user", content=user_msg)], GenerateOptions())


@pytest.mark.asyncio
async def test_manual_chat_protocol_facts_backed(tmp_path: Path) -> None:
    pipe, facts, memory, store = _build_pipe(tmp_path)
    try:
        await _turn(pipe, "just for context: I prefer negroni. please remember that.")
        pref_facts = [f for f in facts.all_facts() if f.category == CAT_USER_PREFERENCES]
        assert any("negroni" in f.value.lower() for f in pref_facts)

        state = store.load()
        assert "negroni" in state.text.lower()

        await _turn(pipe, "what do you think about the weather today?")
        await _turn(pipe, "can you tell me a short joke?")
        await _turn(
            pipe,
            "Someone at work loves hibiscus tea, not that it matters.",
        )

        pref_facts = [f for f in facts.all_facts() if f.category == CAT_USER_PREFERENCES]
        assert any("negroni" in f.value.lower() for f in pref_facts)
        assert not any("hibiscus" in f.value.lower() for f in pref_facts)
        assert "hibiscus" not in store.load().text.lower()

        stored = [e.text for e in memory._entries.values()]
        assert any("negroni" in t.lower() for t in stored)
        hits = memory.query("negroni", top_k=3)
        assert hits
    finally:
        memory.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not has_torch, reason="torch not installed")
@pytest.mark.skipif(not _hybrid_ckpt.exists(), reason="hybrid checkpoint missing")
async def test_manual_chat_protocol_hybrid_updater(tmp_path: Path) -> None:
    pipe, facts, memory, store = _build_pipe(tmp_path, state_updater_kind="hybrid")
    try:
        await _turn(pipe, "just for context: I prefer negroni. please remember that.")
        assert any("negroni" in f.value.lower() for f in facts.all_facts())

        state = store.load()
        assert state.revision >= 1
        assert state.text  # hybrid projection non-empty

        await _turn(pipe, "Someone at work loves hibiscus tea, not that it matters.")
        assert not any("hibiscus" in f.value.lower() for f in facts.all_facts())

        hits = memory.query("negroni", top_k=3)
        assert hits
    finally:
        memory.close()
