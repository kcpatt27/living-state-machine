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

from lsm.core.association import Association, chain_combiners
from lsm.core.episodic import EpisodicLog, TurnRecord
from lsm.core.habituation import Habituation
from lsm.core.memory import MemoryStore, ScoreCombiner, ScoredEntry
from lsm.core.outcome import (
    NullOutcomeScorer,
    OutcomeLog,
    OutcomeRecord,
    OutcomeScorer,
)
from lsm.core.plasticity import Plasticity
from lsm.core.state import State, StateStore, StateUpdater
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
    ) -> None:
        self.backend = backend
        self.episodic = episodic
        self.config = config
        self.state_store = state_store
        self.state_updater = state_updater
        self.memory = memory
        self.habituation = habituation
        self.association = association
        self.plasticity = plasticity
        self.writer: MemoryWriter = writer or AlwaysWriter()
        self.outcome_scorer: OutcomeScorer = outcome_scorer or NullOutcomeScorer()
        self.outcome_log = outcome_log
        self.last_outcome: OutcomeRecord | None = None
        self.last_write_decisions: list[WriteDecision] = []
        self._recent_ids: deque[str] = deque(maxlen=config.recent_context_window)

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
            self._state = await self.state_updater.update(
                self._state, user_msg, assistant_reply
            )
        except Exception:  # noqa: BLE001
            log.exception("state update failed; keeping prior state")
            return
        if self.state_store is not None:
            try:
                self.state_store.save(self._state)
            except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            log.exception("novelty probe failed")
            return None
        if not hits:
            return 1.0
        top = hits[0].components.get("cosine", hits[0].score)
        return max(0.0, min(1.0, 1.0 - float(top)))

    def _maybe_write(self, text: str, role: str) -> None:
        if self.memory is None or not text:
            return
        critic: float | None = None
        if self.last_outcome is not None:
            critic = self.last_outcome.score
        candidate = WriteCandidate(
            text=text,
            role=role,
            source="conversation",
            tags=[role, self.config.session_id],
            novelty=self._novelty(text),
            self_critic_score=critic,
        )
        decision = self.writer.decide(candidate)
        self.last_write_decisions.append(decision)
        if not decision.write:
            log.debug("writer skipped %s turn: %s", role, decision.reason)
            return
        try:
            self.memory.put(
                text,
                source="conversation",
                tags=candidate.tags,
                meta={
                    "writer": self.writer.name,
                    "writer_reason": decision.reason,
                    "writer_score": decision.score,
                },
            )
        except Exception:  # noqa: BLE001
            log.exception("memory write (%s) failed", role)

    def _write_turn_memories(self, user_msg: str, assistant_reply: str) -> None:
        if self.memory is None:
            return
        self.last_write_decisions = []
        if self.config.write_user_to_memory:
            self._maybe_write(user_msg, "user")
        if self.config.write_assistant_to_memory:
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
        except Exception:  # noqa: BLE001
            log.exception("outcome scorer failed")
            self.last_outcome = None
            return
        record.turn_id = turn_id
        record.session_id = self.config.session_id
        self.last_outcome = record
        if self.outcome_log is not None:
            try:
                self.outcome_log.append(record)
            except Exception:  # noqa: BLE001
                log.exception("outcome log append failed")
        if self.plasticity is not None and self._last_retrieved:
            try:
                self.plasticity.observe(
                    [h.entry.id for h in self._last_retrieved], record.score
                )
            except Exception:  # noqa: BLE001
                log.exception("plasticity update failed")

    async def generate(
        self, messages: Sequence[ChatMessage], options: GenerateOptions
    ) -> str:
        turn_id = self._log_user(messages)
        user_msg = self._last_user(messages)
        composed = compose(self._context(user_msg), messages)
        text = await self.backend.generate(composed, options)
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
        composed = compose(self._context(user_msg), messages)
        chunks: list[str] = []
        async for piece in self.backend.stream(composed, options):
            chunks.append(piece)
            yield piece
        full = "".join(chunks)
        assistant_turn_id = self._log_assistant(full, in_reply_to=turn_id)
        await self._score_outcome(
            turn_id=assistant_turn_id, user_msg=user_msg, assistant_reply=full
        )
        self._write_turn_memories(user_msg, full)
        await self._update_state(user_msg, full)
