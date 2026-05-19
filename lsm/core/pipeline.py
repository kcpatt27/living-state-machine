"""Per-turn LSM pipeline.

The control loop the whole project revolves around. At L2 this only:

1. Logs the incoming user messages.
2. Composes a prompt (base system only).
3. Calls the backend.
4. Logs the assistant response.

Subsequent levels plug additional steps in, in the order laid out in
``docs/architecture.md``. Each new step is a method on this class so it can
be overridden in tests and disabled by config.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

from lsm.core.association import Association, chain_combiners
from lsm.core.episodic import EpisodicLog, TurnRecord
from lsm.core.facts import FactStore, apply_extracted_facts
from lsm.core.habituation import Habituation
from lsm.core.identity import IdentityProfile
from lsm.core.memory import MemoryStore, ScoreCombiner, ScoredEntry
from lsm.core.normalize import TermNormalizer
from lsm.core.outcome import (
    NullOutcomeScorer,
    OutcomeLog,
    OutcomeRecord,
    OutcomeScorer,
)
from lsm.core.plasticity import Plasticity
from lsm.core.state import (
    State,
    StateStore,
    StateUpdater,
    _looks_like_additive,
    _looks_like_correction,
)
from lsm.core.verifier import NullVerifier, RuleBasedVerifier, Verifier, VerifierContext
from lsm.core.writer import AlwaysWriter, MemoryWriter, WriteCandidate, WriteDecision
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend
from lsm.runtime.prompt_compose import PromptContext, compose

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    """Runtime knobs for :class:`Pipeline`."""

    session_id: str
    backend_name: str
    model: str
    retrieve_top_k: int = 4
    retrieve_min_score: float = 0.15
    write_user_to_memory: bool = True
    write_assistant_to_memory: bool = False
    recent_context_window: int = 8
    # --- Surprise-modulated plasticity (correction dynamics) ---
    # When ``_looks_like_correction(user_msg)`` fires and the master switch
    # below is on, the per-turn plasticity update on retrieved entries is
    # amplified (``correction_surprise``) and the trust target is capped
    # downward (``correction_target``). This converts the existing
    # correction tag into a real RPE-style learning signal at the L10
    # plasticity layer.
    correction_dynamics_enabled: bool = True
    correction_surprise: float = 2.0
    correction_target: float = 0.2
    # F.2 updater selection / optional polish wrapper.
    state_updater: Literal["facts_backed", "llm_summarizer"] = "facts_backed"
    state_projection_polish: bool = False
    state_projection_polish_max_tokens: int = 192
    # V.1 verifier (scoring-not-authoring gate). Master switch defaults on;
    # NullVerifier is used when ``verifier_enabled=False``. The re-prompt
    # path applies to the summarizer + assistant-reply gates; the writer
    # gate has no re-prompt and instead downgrades / skips.
    verifier_enabled: bool = True
    verifier_reprompt_enabled: bool = True
    verifier_threshold_reprompt: float = 1.0
    verifier_threshold_reject: float = 2.5
    verifier_weights: dict[str, float] | None = None
    # Number of recent user turns to feed into the verifier's list-merge
    # check (kept small to avoid stale paired-and noise).
    verifier_recent_user_turns: int = 3


class Pipeline:
    """The per-turn orchestrator.

    Level hooks (names reserved even when unimplemented):

    - ``observe`` (L2): write user turn to episodic log.
    - ``update_state`` (L3): update the hidden state ``s_t``.
    - ``retrieve`` (L4): fetch memory items.
    - ``apply_habituation`` (L5): damp repeated-key retrieval scores.
    - ``apply_association`` (L6): boost co-occurring keys.
    - ``compose_prompt`` (all levels): assemble the final prompt.
    - ``call_llm`` (L2+): invoke the backend.
    - ``score_outcome`` (L9): run the self-critic.
    - ``write_memory`` (L7): gated writer commits memories.
    - ``update_meta`` (L10): adjust per-entry plasticity.
    """

    def __init__(
        self,
        backend: LLMBackend,
        episodic: EpisodicLog,
        config: PipelineConfig,
        *,
        state_store: StateStore | None = None,
        state_updater: StateUpdater | None = None,
        memory: MemoryStore | None = None,
        score_combiner: ScoreCombiner | None = None,
        habituation: Habituation | None = None,
        association: Association | None = None,
        writer: MemoryWriter | None = None,
        outcome_scorer: OutcomeScorer | None = None,
        outcome_log: OutcomeLog | None = None,
        plasticity: Plasticity | None = None,
        fact_store: FactStore | None = None,
        term_normalizer: TermNormalizer | None = None,
        verifier: Verifier | None = None,
        identity: IdentityProfile | None = None,
    ) -> None:
        self.backend = backend
        self.episodic = episodic
        self.config = config
        self.state_store = state_store
        self.state_updater = state_updater
        self.memory = memory
        self.fact_store = fact_store
        self.term_normalizer = term_normalizer
        self.identity = identity
        # V.1 verifier: if caller passed one, use it; otherwise honor the
        # config switch. ``NullVerifier`` is the explicit ablation baseline.
        if verifier is not None:
            self.verifier: Verifier = verifier
        elif config.verifier_enabled:
            self.verifier = RuleBasedVerifier(
                threshold_reprompt=config.verifier_threshold_reprompt,
                threshold_reject=config.verifier_threshold_reject,
                weights=config.verifier_weights,  # type: ignore[arg-type]
            )
        else:
            self.verifier = NullVerifier()
        self.habituation = habituation
        self.association = association
        self.plasticity = plasticity
        self.writer: MemoryWriter = writer or AlwaysWriter()
        self.outcome_scorer: OutcomeScorer = outcome_scorer or NullOutcomeScorer()
        self.outcome_log = outcome_log
        self.last_outcome: OutcomeRecord | None = None
        self.last_write_decisions: list[WriteDecision] = []
        self._recent_ids: deque[str] = deque(maxlen=config.recent_context_window)
        # Last N user turns for the verifier's list-merge check.
        self._recent_user_turns: deque[str] = deque(
            maxlen=max(1, config.verifier_recent_user_turns)
        )
        # Bookkeeping for the reply gate so callers / tests can see what
        # the verifier said about the most recent assistant turn.
        from lsm.core.verifier import Verdict  # local: avoid cycle in stubs

        self.last_verifier_verdict: Verdict | None = None
        self.last_verifier_reprompted: bool = False

        if score_combiner is not None:
            self.score_combiner: ScoreCombiner | None = score_combiner
        else:
            parts: list[ScoreCombiner] = []
            if habituation is not None:
                parts.append(habituation.combiner)
            if association is not None:
                parts.append(association.combiner)
            if plasticity is not None:
                parts.append(plasticity.combiner)
            self.score_combiner = chain_combiners(*parts) if parts else None

        self._state: State = state_store.load() if state_store is not None else State()
        self._last_retrieved: list[ScoredEntry] = []

    def _log_user(self, messages: Sequence[ChatMessage]) -> str:
        turn_id = str(uuid.uuid4())
        user = next((m for m in reversed(messages) if m.role == "user"), None)
        if user is None:
            return turn_id
        self.episodic.append(
            TurnRecord(
                turn_id=turn_id,
                session_id=self.config.session_id,
                role="user",
                content=user.content,
                meta={"backend": self.config.backend_name, "model": self.config.model},
            )
        )
        return turn_id

    def _log_assistant(self, text: str, *, in_reply_to: str) -> str:
        record = TurnRecord(
            session_id=self.config.session_id,
            role="assistant",
            content=text,
            meta={
                "backend": self.config.backend_name,
                "model": self.config.model,
                "in_reply_to": in_reply_to,
            },
        )
        return self.episodic.append(record)

    def _retrieve(self, query_text: str) -> list[ScoredEntry]:
        if self.memory is None or not query_text:
            return []
        # association's combiner reads self._recent_ids via closure-less path:
        # we temporarily set context_keys for the store query to the recent deque.
        hits = self.memory.query(
            query_text,
            top_k=self.config.retrieve_top_k,
            context_keys=list(self._recent_ids),
            combiner=self.score_combiner,
        )
        filtered = [h for h in hits if h.score >= self.config.retrieve_min_score]
        hit_ids = [h.entry.id for h in filtered]
        if self.habituation is not None:
            for hid in hit_ids:
                self.habituation.observe(hid)
        if self.association is not None and hit_ids:
            self.association.co_hit(list(self._recent_ids) + hit_ids)
        for hid in hit_ids:
            self._recent_ids.append(hid)
        return filtered

    def _context(self, user_msg: str) -> PromptContext:
        """Compose the prompt context: state projection + retrieved memories."""
        self._last_retrieved = self._retrieve(user_msg)
        retrieved_texts = [h.entry.text for h in self._last_retrieved]
        return PromptContext(
            state_projection=self._state.project(),
            retrieved=retrieved_texts,
        )

    @property
    def last_retrieved(self) -> list[ScoredEntry]:
        return self._last_retrieved

    @property
    def state(self) -> State:
        return self._state

    async def _update_state(self, user_msg: str, assistant_reply: str) -> None:
        if self.state_updater is None:
            return
        try:
            self._state = await self.state_updater.update(self._state, user_msg, assistant_reply)
        except Exception:
            log.exception("state update failed; keeping prior state")
            return
        if self.state_store is not None:
            try:
                self.state_store.save(self._state)
            except Exception:
                log.exception("state persist failed")

    @staticmethod
    def _last_user(messages: Sequence[ChatMessage]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return ""

    def _novelty(self, text: str) -> float | None:
        if self.memory is None or not text:
            return None
        try:
            hits = self.memory.query(text, top_k=1)
        except Exception:
            log.exception("novelty probe failed")
            return None
        if not hits:
            return 1.0
        top = hits[0].components.get("cosine", hits[0].score)
        return max(0.0, min(1.0, 1.0 - float(top)))

    def _verifier_context(self) -> VerifierContext:
        """Build a :class:`VerifierContext` from the live pipeline state."""
        corrections: list = []
        if self.memory is not None:
            try:
                corrections = self.memory.query_by_tag("correction", top_k=5)
            except Exception:
                log.debug("verifier corrections fetch failed", exc_info=True)
                corrections = []
        return VerifierContext(
            facts=self.fact_store,
            identity=self.identity,
            recent_corrections=tuple(corrections),
            recent_user_turns=tuple(self._recent_user_turns),
            normalizer=self.term_normalizer,
        )

    def _maybe_write(self, text: str, role: str) -> None:
        if self.memory is None or not text:
            return
        critic: float | None = None
        if self.last_outcome is not None:
            critic = self.last_outcome.score
        tags = [role, self.config.session_id]
        # Tag user-side corrections so we can later (Phase D) prefer
        # merge/overwrite over append. No behavior change here beyond the tag.
        if role == "user" and _looks_like_correction(text):
            tags.append("correction")
        if role == "user" and _looks_like_additive(text):
            tags.append("additive")
        candidate = WriteCandidate(
            text=text,
            role=role,
            source="conversation",
            tags=tags,
            novelty=self._novelty(text),
            self_critic_score=critic,
        )
        decision = self.writer.decide(candidate)
        self.last_write_decisions.append(decision)
        if not decision.write:
            log.debug("writer skipped %s turn: %s", role, decision.reason)
            return

        # V.1 writer gate: don't durably store assistant text that contradicts
        # facts or restates a corrected claim. User turns are skipped here --
        # the structured FactStore is the authoritative path for user content
        # and we don't want the verifier rejecting raw user corrections.
        verifier_meta: dict[str, object] = {}
        if role == "assistant":
            verdict = self.verifier.score(text, self._verifier_context())
            verifier_meta = {
                "verifier_energy": float(verdict.energy),
                "verifier_decision": verdict.decision,
                "verifier_violations": [v.kind for v in verdict.violations],
            }
            if verdict.decision == "reject":
                log.info(
                    "writer gate rejected assistant turn (energy=%.2f, %s)",
                    verdict.energy,
                    [v.kind for v in verdict.violations],
                )
                tags.append("verifier_reject")
                # Demote to no-store. We *do* keep the verdict on
                # ``last_write_decisions`` via the existing decision object
                # so callers can inspect it; the WriteDecision itself is
                # unmodified to keep the writer's contract simple.
                return
            if verdict.decision == "reprompt":
                # No re-prompt loop on the writer path; downgrade to a
                # tagged write so retrieval can deprioritize later.
                tags.append("verifier_low_confidence")

        try:
            self.memory.put(
                text,
                source="conversation",
                tags=candidate.tags,
                meta={
                    "writer": self.writer.name,
                    "writer_reason": decision.reason,
                    "writer_score": decision.score,
                    **verifier_meta,
                },
            )
        except Exception:
            log.exception("memory write (%s) failed", role)

    def _maybe_extract_facts(self, user_msg: str) -> None:
        if self.fact_store is None or not user_msg:
            return
        try:
            apply_extracted_facts(self.fact_store, user_msg, normalizer=self.term_normalizer)
        except Exception:
            log.exception("fact extraction failed")

    def _write_turn_memories(self, user_msg: str, assistant_reply: str) -> None:
        if self.memory is None and self.fact_store is None:
            return
        self.last_write_decisions = []
        if self.config.write_user_to_memory and user_msg:
            self._maybe_extract_facts(user_msg)
            if self.memory is not None:
                self._maybe_write(user_msg, "user")
        if self.memory is not None and self.config.write_assistant_to_memory:
            self._maybe_write(assistant_reply, "assistant")

    async def _score_outcome(
        self,
        *,
        turn_id: str,
        user_msg: str,
        assistant_reply: str,
    ) -> None:
        if isinstance(self.outcome_scorer, NullOutcomeScorer):
            self.last_outcome = None
            return
        retrieved_texts = [h.entry.text for h in self._last_retrieved]
        try:
            record = await self.outcome_scorer.score(
                user_msg=user_msg,
                assistant_reply=assistant_reply,
                retrieved=retrieved_texts,
                state_projection=self._state.project(),
            )
        except Exception:
            log.exception("outcome scorer failed")
            self.last_outcome = None
            return
        record.turn_id = turn_id
        record.session_id = self.config.session_id
        self.last_outcome = record
        if self.outcome_log is not None:
            try:
                self.outcome_log.append(record)
            except Exception:
                log.exception("outcome log append failed")
        if self.plasticity is not None and self._last_retrieved:
            ids = [h.entry.id for h in self._last_retrieved]
            is_correction = self.config.correction_dynamics_enabled and _looks_like_correction(
                user_msg
            )
            if is_correction:
                target = min(record.score, self.config.correction_target)
                surprise = self.config.correction_surprise
            else:
                target = record.score
                surprise = 1.0
            try:
                self.plasticity.observe(ids, target, surprise=surprise)
            except Exception:
                log.exception("plasticity update failed")

    def _track_user_turn(self, user_msg: str) -> None:
        if user_msg:
            self._recent_user_turns.append(user_msg)

    def _fix_hint_message(self, hints: Sequence[str]) -> ChatMessage:
        bullets = "\n".join(f"- {h}" for h in hints)
        content = (
            "Your previous draft failed verification. Fix these issues "
            "before re-emitting:\n"
            f"{bullets}"
        )
        return ChatMessage(role="system", content=content)

    async def generate(self, messages: Sequence[ChatMessage], options: GenerateOptions) -> str:
        turn_id = self._log_user(messages)
        user_msg = self._last_user(messages)
        self._track_user_turn(user_msg)
        composed = compose(self._context(user_msg), messages)
        text = await self.backend.generate(composed, options)

        # V.1.1 reply gate (non-streaming path): up to one re-prompt with
        # fix hints; reject -> keep best draft for episodic logging but tag
        # the durable write as verifier_reject (handled in _maybe_write).
        verdict = self.verifier.score(text, self._verifier_context())
        self.last_verifier_verdict = verdict
        self.last_verifier_reprompted = False
        if (
            verdict.decision != "accept"
            and self.config.verifier_reprompt_enabled
            and verdict.fix_hints
        ):
            self.last_verifier_reprompted = True
            retry_messages = [*composed, self._fix_hint_message(verdict.fix_hints)]
            try:
                retry_text = await self.backend.generate(retry_messages, options)
            except Exception:
                log.exception("verifier re-prompt failed; keeping original draft")
                retry_text = ""
            if retry_text:
                retry_verdict = self.verifier.score(retry_text, self._verifier_context())
                if retry_verdict.energy < verdict.energy:
                    text = retry_text
                    self.last_verifier_verdict = retry_verdict

        assistant_turn_id = self._log_assistant(text, in_reply_to=turn_id)
        await self._score_outcome(
            turn_id=assistant_turn_id, user_msg=user_msg, assistant_reply=text
        )
        self._write_turn_memories(user_msg, text)
        await self._update_state(user_msg, text)
        return text

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> AsyncIterator[str]:
        turn_id = self._log_user(messages)
        user_msg = self._last_user(messages)
        self._track_user_turn(user_msg)
        composed = compose(self._context(user_msg), messages)
        chunks: list[str] = []
        async for piece in self.backend.stream(composed, options):
            chunks.append(piece)
            yield piece
        full = "".join(chunks)

        # V.1.1 reply gate (streaming path): we don't break the UX by
        # rewinding what the user already saw. We score post-stream and let
        # _maybe_write apply the writer gate; the verdict is exposed on the
        # pipeline so callers / next-turn extractors can react.
        self.last_verifier_verdict = self.verifier.score(full, self._verifier_context())
        self.last_verifier_reprompted = False

        assistant_turn_id = self._log_assistant(full, in_reply_to=turn_id)
        await self._score_outcome(
            turn_id=assistant_turn_id, user_msg=user_msg, assistant_reply=full
        )
        self._write_turn_memories(user_msg, full)
        await self._update_state(user_msg, full)
