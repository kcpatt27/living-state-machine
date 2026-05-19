"""Integration tests for V.1 verifier wiring into the per-turn pipeline.

Covers:
- Reply gate: ``Pipeline.generate`` re-prompts on contradicting candidate.
- Reply gate: rejected reply is excluded from the durable writer path.
- Writer gate: a contradicting candidate is tagged ``verifier_reject`` and
  not stored.
- Master switch: ``verifier_enabled=False`` uses ``NullVerifier`` and the
  pipeline matches the pre-V.1 behavior (no extra LLM calls).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from lsm.core.embeddings import HashEmbedder
from lsm.core.episodic import EpisodicLog
from lsm.core.facts import (
    CAT_USER_MEDIA,
    Fact,
    FactStore,
)
from lsm.core.identity import IdentityProfile
from lsm.core.memory import InMemoryStore
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.projection import DeterministicProjection
from lsm.core.state import FactsBackedUpdater, LLMSummarizerUpdater
from lsm.core.verifier import NullVerifier, RuleBasedVerifier
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions


class _ScriptedBackend:
    """LLMBackend stub returning a queued list of replies in order.

    ``generate`` pops the next reply; ``stream`` yields the next reply as a
    single chunk. Tracks how many ``generate`` calls were made.
    """

    name = "scripted"
    model = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies: list[str] = list(replies)
        self.generate_calls = 0
        self.stream_calls = 0

    def _next(self) -> str:
        if not self._replies:
            return ""
        return self._replies.pop(0)

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        _ = messages, options
        self.generate_calls += 1
        return self._next()

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        _ = messages, options
        self.stream_calls += 1
        yield self._next()

    async def close(self) -> None:
        return None


def _seed_facts(tmp_path: Path) -> FactStore:
    store = FactStore(tmp_path / "home")
    store.upsert(
        Fact(
            owner="user",
            category=CAT_USER_MEDIA,
            subject="Crimson Desert",
            value="action RPG",
        )
    )
    return store


def _pipeline(
    tmp_path: Path,
    backend: _ScriptedBackend,
    *,
    verifier_enabled: bool = True,
    verifier_reprompt_enabled: bool = True,
) -> tuple[Pipeline, InMemoryStore, FactStore]:
    embedder = HashEmbedder(dim=64)
    memory = InMemoryStore(embedder=embedder)
    fact_store = _seed_facts(tmp_path)
    identity = IdentityProfile(name="Elsum", description="Test assistant.")
    projection = DeterministicProjection()
    updater = FactsBackedUpdater(
        embedder,
        projection,
        identity=identity,
        memory=memory,
        fact_store=fact_store,
    )
    cfg = PipelineConfig(
        session_id="s",
        backend_name="scripted",
        model="scripted",
        retrieve_top_k=3,
        retrieve_min_score=-1.0,
        verifier_enabled=verifier_enabled,
        verifier_reprompt_enabled=verifier_reprompt_enabled,
    )
    pipe = Pipeline(
        backend=backend,
        episodic=EpisodicLog(tmp_path / "e.sqlite"),
        config=cfg,
        memory=memory,
        state_updater=updater,
        fact_store=fact_store,
        identity=identity,
    )
    return pipe, memory, fact_store


@pytest.mark.asyncio
async def test_generate_reprompts_on_contradicting_reply(tmp_path: Path) -> None:
    """First draft contradicts facts; second draft is clean and gets used."""
    backend = _ScriptedBackend(
        [
            "Crimson Desert is a strategy game.",  # contradicts facts.
            "Crimson Desert is an action RPG.",  # clean retry.
        ]
    )
    pipe, _memory, _facts = _pipeline(tmp_path, backend)

    text = await pipe.generate(
        [ChatMessage(role="user", content="tell me about crimson desert")],
        GenerateOptions(),
    )
    assert "action rpg" in text.lower()
    assert backend.generate_calls == 2
    assert pipe.last_verifier_reprompted is True
    assert pipe.last_verifier_verdict is not None
    assert pipe.last_verifier_verdict.decision == "accept"


@pytest.mark.asyncio
async def test_generate_keeps_best_draft_when_retry_worse(tmp_path: Path) -> None:
    """Retry is even worse -> keep first draft (still gated by writer)."""
    backend = _ScriptedBackend(
        [
            "Crimson Desert is a strategy game.",
            "Crimson Desert is a strategy game and i am Elsum.",
        ]
    )
    pipe, _memory, _facts = _pipeline(tmp_path, backend)
    text = await pipe.generate(
        [ChatMessage(role="user", content="tell me about crimson desert")],
        GenerateOptions(),
    )
    assert backend.generate_calls == 2
    # We kept the first draft (energy strictly lower than the second).
    assert "strategy" in text.lower()
    assert "elsum" not in text.lower()
    # Writer gate (in this default config) doesn't durably store assistant
    # text -- write_assistant_to_memory is False -- so the assertion here is
    # simply that the pipeline's verdict reflects rejection.
    assert pipe.last_verifier_verdict is not None
    assert pipe.last_verifier_verdict.decision != "accept"


@pytest.mark.asyncio
async def test_writer_gate_skips_high_energy_assistant_text(tmp_path: Path) -> None:
    """Two violations push energy past the reject threshold -> no durable write."""
    # fact_contradiction (1.5) + identity_drift (1.5) = 3.0 -> reject.
    bad = "Crimson Desert is a strategy game. i am Elsum and i love strategy game tactics."
    backend = _ScriptedBackend([bad])
    pipe, memory, _facts = _pipeline(tmp_path, backend, verifier_reprompt_enabled=False)
    pipe.config.write_assistant_to_memory = True

    await pipe.generate(
        [ChatMessage(role="user", content="tell me about crimson desert")],
        GenerateOptions(),
    )
    assistant_rows = memory.query_by_tag("assistant", top_k=10)
    assert assistant_rows == []  # writer gate refused
    assert backend.generate_calls == 1  # no reprompt since reprompt disabled


@pytest.mark.asyncio
async def test_writer_gate_tags_low_confidence_when_reprompt_band(tmp_path: Path) -> None:
    """Single contradiction (energy 1.5) lands in the reprompt band -> stored
    but tagged ``verifier_low_confidence`` so retrieval can deprioritize."""
    backend = _ScriptedBackend(["Crimson Desert is a strategy game."])
    pipe, memory, _facts = _pipeline(tmp_path, backend, verifier_reprompt_enabled=False)
    pipe.config.write_assistant_to_memory = True

    await pipe.generate(
        [ChatMessage(role="user", content="tell me about crimson desert")],
        GenerateOptions(),
    )
    assistant_rows = memory.query_by_tag("assistant", top_k=10)
    assert assistant_rows  # row landed
    assert any("verifier_low_confidence" in r.tags for r in assistant_rows)


@pytest.mark.asyncio
async def test_master_switch_off_uses_null_verifier(tmp_path: Path) -> None:
    """``verifier_enabled=False`` -> NullVerifier, no extra LLM calls, no rejects."""
    backend = _ScriptedBackend(["Crimson Desert is a strategy game."])
    pipe, memory, _facts = _pipeline(tmp_path, backend, verifier_enabled=False)
    pipe.config.write_assistant_to_memory = True
    assert isinstance(pipe.verifier, NullVerifier)

    await pipe.generate(
        [ChatMessage(role="user", content="tell me about crimson desert")],
        GenerateOptions(),
    )
    assert backend.generate_calls == 1
    # NullVerifier accepts everything -> the contradicting text IS stored.
    assistant_rows = memory.query_by_tag("assistant", top_k=10)
    assert assistant_rows  # at least one assistant row landed


@pytest.mark.asyncio
async def test_stream_records_verdict_without_blocking_ux(tmp_path: Path) -> None:
    """Streaming path doesn't re-prompt; verdict is recorded for downstream use."""
    backend = _ScriptedBackend(["Crimson Desert is a strategy game."])
    pipe, _memory, _facts = _pipeline(tmp_path, backend)

    pieces: list[str] = []
    async for chunk in pipe.stream(
        [ChatMessage(role="user", content="hi")],
        GenerateOptions(),
    ):
        pieces.append(chunk)
    full = "".join(pieces)
    assert "strategy" in full.lower()  # user saw the streamed text verbatim
    assert pipe.last_verifier_verdict is not None
    assert pipe.last_verifier_verdict.decision != "accept"
    assert pipe.last_verifier_reprompted is False
    assert backend.generate_calls == 0  # streamed path only


@pytest.mark.asyncio
async def test_summarizer_gate_reprompts_on_contradicting_s_t(tmp_path: Path) -> None:
    """Legacy ablation path: ``LLMSummarizerUpdater`` re-prompts once on
    a contradicting candidate ``s_t`` and commits the clean retry."""
    backend = _ScriptedBackend(
        [
            # First call: the reply itself (we don't gate this in this test;
            # the reply gate sees an empty FactStore relative to this turn).
            "ok.",
            # Second call: the summarizer's first draft -- contradicts facts.
            "preferences/likes: tea\ngames/media: Crimson Desert (strategy game)",
            # Third call: the summarizer's retry -- clean.
            "preferences/likes: tea\ngames/media: Crimson Desert (action RPG)",
        ]
    )
    embedder = HashEmbedder(dim=64)
    memory = InMemoryStore(embedder=embedder)
    fact_store = _seed_facts(tmp_path)
    identity = IdentityProfile(name="Elsum", description="x")
    verifier = RuleBasedVerifier()
    updater = LLMSummarizerUpdater(
        backend,
        embedder,
        identity=identity,
        memory=memory,
        fact_store=fact_store,
        verifier=verifier,
        verifier_reprompt_enabled=True,
    )
    cfg = PipelineConfig(
        session_id="s",
        backend_name="scripted",
        model="scripted",
        state_updater="llm_summarizer",
        # Disable the reply gate so we isolate the summarizer-gate behavior.
        verifier_reprompt_enabled=False,
    )
    pipe = Pipeline(
        backend=backend,
        episodic=EpisodicLog(tmp_path / "e.sqlite"),
        config=cfg,
        memory=memory,
        state_updater=updater,
        fact_store=fact_store,
        identity=identity,
        verifier=verifier,
    )
    await pipe.generate(
        [ChatMessage(role="user", content="tell me about crimson desert")],
        GenerateOptions(),
    )
    # 3 backend calls: assistant reply (1) + summarizer first + retry (2).
    assert backend.generate_calls == 3
    # Final s_t reflects the corrected retry.
    assert "action rpg" in pipe.state.text.lower()
    assert "strategy game" not in pipe.state.text.lower()


@pytest.mark.asyncio
async def test_summarizer_gate_reject_keeps_prior_s_t(tmp_path: Path) -> None:
    """Both summarizer drafts are still bad -> prior ``s_t`` is retained."""
    prior_text = "games/media: Crimson Desert (action RPG)"
    backend = _ScriptedBackend(
        [
            "ok.",  # reply
            # First summarizer draft: 2 violations -> reject.
            "games/media: Crimson Desert (strategy game). i am Elsum and i love strategy game tactics.",
            # Retry: still bad (different wording, same kinds).
            "games/media: Crimson Desert (strategy game). i'm Elsum so let's play strategy game.",
        ]
    )
    embedder = HashEmbedder(dim=64)
    memory = InMemoryStore(embedder=embedder)
    fact_store = _seed_facts(tmp_path)
    identity = IdentityProfile(name="Elsum", description="x")
    verifier = RuleBasedVerifier()
    updater = LLMSummarizerUpdater(
        backend,
        embedder,
        identity=identity,
        memory=memory,
        fact_store=fact_store,
        verifier=verifier,
        verifier_reprompt_enabled=True,
    )
    cfg = PipelineConfig(
        session_id="s",
        backend_name="scripted",
        model="scripted",
        state_updater="llm_summarizer",
        verifier_reprompt_enabled=False,  # isolate summarizer gate
    )
    # Seed prior state via the pipeline's state attribute.
    pipe = Pipeline(
        backend=backend,
        episodic=EpisodicLog(tmp_path / "e.sqlite"),
        config=cfg,
        memory=memory,
        state_updater=updater,
        fact_store=fact_store,
        identity=identity,
        verifier=verifier,
    )
    from lsm.core.state import State

    pipe._state = State(text=prior_text, vector=embedder.embed(prior_text), revision=7)

    await pipe.generate(
        [ChatMessage(role="user", content="tell me about crimson desert")],
        GenerateOptions(),
    )
    # Prior s_t preserved verbatim; revision unchanged.
    assert pipe.state.text == prior_text
    assert pipe.state.revision == 7
    # Both drafts were attempted (3 backend calls: reply + 2 summarizer).
    assert backend.generate_calls == 3


@pytest.mark.asyncio
async def test_explicit_verifier_kwarg_overrides_config(tmp_path: Path) -> None:
    """Passing a verifier instance wins over the config switch."""
    backend = _ScriptedBackend(["whatever"])
    embedder = HashEmbedder(dim=64)
    memory = InMemoryStore(embedder=embedder)
    fact_store = _seed_facts(tmp_path)
    identity = IdentityProfile(name="Elsum", description="x")
    updater = FactsBackedUpdater(
        embedder,
        DeterministicProjection(),
        identity=identity,
        memory=memory,
        fact_store=fact_store,
    )
    cfg = PipelineConfig(
        session_id="s",
        backend_name="scripted",
        model="scripted",
        verifier_enabled=False,  # config says off
    )
    explicit = RuleBasedVerifier(threshold_reprompt=999.0, threshold_reject=999.0)
    pipe = Pipeline(
        backend=backend,
        episodic=EpisodicLog(tmp_path / "e.sqlite"),
        config=cfg,
        memory=memory,
        state_updater=updater,
        fact_store=fact_store,
        identity=identity,
        verifier=explicit,  # overrides
    )
    assert pipe.verifier is explicit
