"""Construct a fully-wired :class:`Pipeline` for CLI and server use."""

from __future__ import annotations

from pathlib import Path

from lsm.config import Settings, get_settings
from lsm.core.association import AssociationConfig, HebbianAssociation
from lsm.core.embeddings import get_default_embedder
from lsm.core.episodic import EpisodicLog
from lsm.core.habituation import ExpDecayHabituation, HabituationConfig
from lsm.core.memory import MemoryStore, build_store
from lsm.core.outcome import HeuristicOutcomeScorer, OutcomeLog
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.plasticity import ExpDecayPlasticity
from lsm.core.state import LLMSummarizerUpdater, StateStore
from lsm.core.writer import RuleBasedWriter, RuleBasedWriterConfig
from lsm.runtime.llm_backend import LLMBackend, build_backend


def episodic_path(settings: Settings) -> Path:
    return settings.memory_dir / "episodes.sqlite"


def build_default_pipeline(
    settings: Settings | None = None,
    *,
    session_id: str = "default",
    model: str | None = None,
) -> tuple[Pipeline, LLMBackend, EpisodicLog, StateStore, MemoryStore, OutcomeLog]:
    """Wire the Phase-1 stack: state, memory, habituation, association, writer, outcomes."""
    s = settings or get_settings()
    effective_model = model or s.model
    backend = build_backend(s.backend, model=effective_model, ollama_host=s.ollama_host)
    episodic = EpisodicLog(episodic_path(s))
    store = StateStore(s.home)
    embedder = get_default_embedder(s.embed_model)
    updater = LLMSummarizerUpdater(backend, embedder)
    memory = build_store(
        "lancedb",
        root=s.memory_dir / "durable.lance",
        embedder=embedder,
    )
    outcome_log = OutcomeLog(s.home / "outcomes.sqlite")
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(
            session_id=session_id,
            backend_name=s.backend,
            model=effective_model,
        ),
        state_store=store,
        state_updater=updater,
        memory=memory,
        habituation=ExpDecayHabituation(HabituationConfig()),
        association=HebbianAssociation(AssociationConfig()),
        plasticity=ExpDecayPlasticity(),
        outcome_scorer=HeuristicOutcomeScorer(),
        outcome_log=outcome_log,
        writer=RuleBasedWriter(RuleBasedWriterConfig()),
    )
    return pipe, backend, episodic, store, memory, outcome_log
