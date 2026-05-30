"""Construct a fully-wired :class:`Pipeline` for CLI and server use."""

from __future__ import annotations

from pathlib import Path

from lsm.config import Settings, get_settings
from lsm.core.association import AssociationConfig, HebbianAssociation
from lsm.core.embeddings import get_default_embedder
from lsm.core.episodic import EpisodicLog
from lsm.core.facts import FactStore
from lsm.core.habituation import ExpDecayHabituation, HabituationConfig
from lsm.core.identity import IdentityStore
from lsm.core.memory import MemoryStore, build_store
from lsm.core.normalize import EmbeddingTermNormalizer
from lsm.core.outcome import HeuristicOutcomeScorer, OutcomeLog
from lsm.core.pipeline import Pipeline, PipelineConfig
from lsm.core.plasticity import ExpDecayPlasticity
from lsm.core.projection import DeterministicProjection, LLMPolishedProjection
from lsm.core.state import FactsBackedUpdater, LLMSummarizerUpdater, StateStore
from lsm.core.verifier import NullVerifier, RuleBasedVerifier, Verifier
from lsm.core.verifier_log import VerifierLog
from lsm.core.writer import RuleBasedWriter, RuleBasedWriterConfig
from lsm.runtime.llm_backend import GenerateOptions, LLMBackend, build_backend


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
    memory = build_store(
        "lancedb",
        root=s.memory_dir / "durable.lance",
        embedder=embedder,
    )
    identity = IdentityStore(s.home).load()
    fact_store = FactStore(s.home)
    fact_store.sync_assistant_identity(identity)
    term_normalizer = EmbeddingTermNormalizer(embedder, s.home)
    verifier_log = VerifierLog(s.home / "verifier_log.sqlite")
    cfg = PipelineConfig(
        session_id=session_id,
        backend_name=s.backend,
        model=effective_model,
    )
    verifier: Verifier = (
        RuleBasedVerifier(
            threshold_reprompt=cfg.verifier_threshold_reprompt,
            threshold_reject=cfg.verifier_threshold_reject,
            weights=cfg.verifier_weights,  # type: ignore[arg-type]
        )
        if cfg.verifier_enabled
        else NullVerifier()
    )
    if cfg.state_updater == "llm_summarizer":
        updater = LLMSummarizerUpdater(
            backend,
            embedder,
            identity=identity,
            memory=memory,
            fact_store=fact_store,
            verifier=verifier,
            verifier_reprompt_enabled=cfg.verifier_reprompt_enabled,
            verifier_log=verifier_log,
        )
    else:
        projection: DeterministicProjection | LLMPolishedProjection = DeterministicProjection()
        if cfg.state_projection_polish:
            projection = LLMPolishedProjection(
                base=projection,
                backend=backend,
                enabled=True,
                options=GenerateOptions(
                    temperature=0.0,
                    max_tokens=cfg.state_projection_polish_max_tokens,
                ),
            )
        updater = FactsBackedUpdater(
            embedder,
            projection,
            identity=identity,
            memory=memory,
            fact_store=fact_store,
        )
    outcome_log = OutcomeLog(s.home / "outcomes.sqlite")
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=cfg,
        state_store=store,
        state_updater=updater,
        memory=memory,
        habituation=ExpDecayHabituation(HabituationConfig()),
        association=HebbianAssociation(AssociationConfig()),
        plasticity=ExpDecayPlasticity(),
        outcome_scorer=HeuristicOutcomeScorer(),
        outcome_log=outcome_log,
        writer=RuleBasedWriter(RuleBasedWriterConfig()),
        fact_store=fact_store,
        term_normalizer=term_normalizer,
        verifier=verifier,
        identity=identity,
        verifier_log=verifier_log,
        embedder=embedder,
    )
    return pipe, backend, episodic, store, memory, outcome_log
