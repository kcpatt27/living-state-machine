"""The LSM system under test.

Plays a benchmark dialogue through a full :class:`Pipeline` (with state and
memory), then poses the final question and returns the answer. The dialogue
is replayed as if it were real: each "user" turn drives a pipeline generate
call; the assistant's canned reply is recorded into memory as if the LSM had
produced it (so the test compares retrieval/state quality, not LLM sampling
noise).

Ablation is achieved by wiring or omitting components at construction time.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from lsm.core.association import Association, NullAssociation, chain_combiners
from lsm.core.embeddings import Embedder
from lsm.core.fact_retrieval import fact_lines_for_prompt
from lsm.core.facts import FactStore, apply_extracted_facts
from lsm.core.habituation import Habituation, NullHabituation
from lsm.core.memory import MemoryStore, ScoreCombiner
from lsm.core.state import State, StateStore, StateUpdater
from lsm.core.verifier import (
    EncoderLearnedVerifier,
    HybridVerifier,
    LinearLearnedVerifier,
    NullVerifier,
    RuleBasedVerifier,
    Verifier,
    VerifierContext,
)
from lsm.core.writer import AlwaysWriter, MemoryWriter, WriteCandidate
from lsm.eval.tasks.recall_pref_v1 import RecallItem
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend
from lsm.runtime.prompt_compose import PromptContext, compose

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LSMSystemConfig:
    top_k: int = 4
    min_score: float = 0.15
    answer_max_tokens: int = 64
    state_enabled: bool = True
    memory_enabled: bool = True
    habituation_enabled: bool = True
    association_enabled: bool = True
    writer_enabled: bool = True
    recent_context_window: int = 8
    verifier_enabled: bool = False
    verifier_reprompt_enabled: bool = True
    verifier_kind: str = "rule"
    verifier_threshold_reprompt: float = 1.0
    verifier_threshold_reject: float = 2.5
    verifier_linear_model_path: str | None = None
    verifier_encoder_model_path: str | None = None
    verifier_encoder_tokenizer_path: str | None = None


class LSMSystem:
    """Pipeline-equivalent system used for eval runs."""

    name = "lsm"

    def __init__(
        self,
        backend: LLMBackend,
        embedder: Embedder,
        memory: MemoryStore | None,
        state_store: StateStore | None,
        state_updater: StateUpdater | None,
        config: LSMSystemConfig | None = None,
        habituation: Habituation | None = None,
        association: Association | None = None,
        writer: MemoryWriter | None = None,
        fact_store: FactStore | None = None,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.memory = memory
        self.state_store = state_store
        self.state_updater = state_updater
        self.fact_store = fact_store
        self.config = config or LSMSystemConfig()
        self.habituation: Habituation = habituation or NullHabituation()
        self.association: Association = association or NullAssociation()
        self.writer: MemoryWriter = writer or AlwaysWriter()
        self.verifier: Verifier = self._build_verifier()
        self._state: State = state_store.load() if state_store is not None else State()
        self._recent_ids: deque[str] = deque(maxlen=self.config.recent_context_window)

    async def _update_state(self, user_msg: str, reply: str) -> None:
        if not self.config.state_enabled or self.state_updater is None:
            return
        try:
            self._state = await self.state_updater.update(self._state, user_msg, reply)
        except Exception:
            log.exception("state updater failed")

    def _build_verifier(self) -> Verifier:
        if not self.config.verifier_enabled:
            return NullVerifier()
        rule = RuleBasedVerifier(
            threshold_reprompt=self.config.verifier_threshold_reprompt,
            threshold_reject=self.config.verifier_threshold_reject,
        )
        kind = (self.config.verifier_kind or "rule").lower()
        if kind in {"rule", "default"}:
            return rule
        if kind == "null":
            return NullVerifier()
        if kind == "linear":
            if not self.config.verifier_linear_model_path:
                raise ValueError("verifier_linear_model_path required for verifier_kind=linear")
            return LinearLearnedVerifier(self.config.verifier_linear_model_path, fallback=rule)
        if kind == "encoder":
            if (
                not self.config.verifier_encoder_model_path
                or not self.config.verifier_encoder_tokenizer_path
            ):
                raise ValueError(
                    "encoder verifier requires verifier_encoder_model_path + "
                    "verifier_encoder_tokenizer_path"
                )
            return EncoderLearnedVerifier(
                self.config.verifier_encoder_model_path,
                tokenizer_path=self.config.verifier_encoder_tokenizer_path,
                fallback=rule,
            )
        if kind == "hybrid":
            if self.config.verifier_linear_model_path:
                learned: Verifier = LinearLearnedVerifier(
                    self.config.verifier_linear_model_path,
                    fallback=rule,
                )
            elif (
                self.config.verifier_encoder_model_path
                and self.config.verifier_encoder_tokenizer_path
            ):
                learned = EncoderLearnedVerifier(
                    self.config.verifier_encoder_model_path,
                    tokenizer_path=self.config.verifier_encoder_tokenizer_path,
                    fallback=rule,
                )
            else:
                raise ValueError(
                    "hybrid verifier requires a linear or encoder learned head artifact"
                )
            return HybridVerifier(learned=learned, rule=rule)
        raise ValueError(f"unknown verifier_kind: {self.config.verifier_kind!r}")

    def _write_memory(self, text: str, tag: str) -> None:
        if not self.config.memory_enabled or self.memory is None or not text:
            return
        candidate = WriteCandidate(text=text, role=tag, source="eval", tags=[tag])
        if self.config.writer_enabled:
            decision = self.writer.decide(candidate)
            if not decision.write:
                return
        try:
            self.memory.put(text, source="eval", tags=[tag])
        except Exception:
            log.exception("memory put failed")

    def _retrieve(self, query: str) -> list[str]:
        if not self.config.memory_enabled or self.memory is None:
            return []
        parts: list[ScoreCombiner] = []
        if self.config.habituation_enabled:
            parts.append(self.habituation.combiner)
        if self.config.association_enabled:
            parts.append(self.association.combiner)
        combiner: ScoreCombiner | None = chain_combiners(*parts) if parts else None
        hits = self.memory.query(
            query,
            top_k=self.config.top_k,
            context_keys=list(self._recent_ids),
            combiner=combiner,
        )
        filtered = [h for h in hits if h.score >= self.config.min_score]
        hit_ids = [h.entry.id for h in filtered]
        if self.config.habituation_enabled:
            for hid in hit_ids:
                self.habituation.observe(hid)
        if self.config.association_enabled and hit_ids:
            self.association.co_hit(list(self._recent_ids) + hit_ids)
        for hid in hit_ids:
            self._recent_ids.append(hid)
        return [h.entry.text for h in filtered]

    async def answer(self, item: RecallItem) -> str:
        # Reset per-item state/memory so items are independent.
        self._state = State()
        self._recent_ids.clear()
        if self.memory is not None:
            self.memory.delete_by_source("eval")
        if self.fact_store is not None:
            self.fact_store.clear_all()

        dialogue = item.dialogue
        recent_user_turns: deque[str] = deque(maxlen=3)
        for i in range(0, len(dialogue), 2):
            user_msg = dialogue[i][1] if dialogue[i][0] == "user" else ""
            reply_idx = i + 1
            reply = (
                dialogue[reply_idx][1]
                if reply_idx < len(dialogue) and dialogue[reply_idx][0] == "assistant"
                else ""
            )
            if user_msg:
                recent_user_turns.append(user_msg)
                if self.fact_store is not None:
                    apply_extracted_facts(self.fact_store, user_msg)
                self._write_memory(user_msg, "user")
            if reply:
                self._write_memory(reply, "assistant")
            if user_msg or reply:
                await self._update_state(user_msg, reply)

        retrieved = self._retrieve(item.question)
        fact_lines: list[str] = []
        if self.fact_store is not None:
            fact_lines = fact_lines_for_prompt(
                self.fact_store,
                item.question,
                self.embedder,
                top_k=self.config.top_k,
                min_score=self.config.min_score,
            )
        ctx = PromptContext(
            state_projection=self._state.project(),
            structured_facts=fact_lines,
            retrieved=retrieved,
        )
        final_messages: Sequence[ChatMessage] = [ChatMessage(role="user", content=item.question)]
        composed = compose(ctx, final_messages)
        options = GenerateOptions(temperature=0.0, max_tokens=self.config.answer_max_tokens)
        answer = await self.backend.generate(composed, options)
        if isinstance(self.verifier, NullVerifier):
            return answer
        vctx = VerifierContext(
            recent_corrections=(
                tuple(self.memory.query_by_tag("correction", top_k=5))
                if self.memory is not None
                else ()
            ),
            recent_user_turns=tuple(recent_user_turns),
        )
        verdict = self.verifier.score(answer, vctx)
        if (
            verdict.decision != "accept"
            and self.config.verifier_reprompt_enabled
            and verdict.fix_hints
        ):
            hint = "\n".join(f"- {h}" for h in verdict.fix_hints)
            retry_messages = [
                *composed,
                ChatMessage(
                    role="system",
                    content=(
                        "Your previous answer failed verification. "
                        "Fix these issues before re-emitting:\n"
                        f"{hint}"
                    ),
                ),
            ]
            retry = await self.backend.generate(retry_messages, options)
            retry_verdict = self.verifier.score(retry, vctx)
            if retry_verdict.energy < verdict.energy:
                answer = retry
        return answer
