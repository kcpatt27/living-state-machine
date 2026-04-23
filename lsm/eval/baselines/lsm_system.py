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
from lsm.core.habituation import Habituation, NullHabituation
from lsm.core.memory import MemoryStore, ScoreCombiner
from lsm.core.state import State, StateStore, StateUpdater
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
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.memory = memory
        self.state_store = state_store
        self.state_updater = state_updater
        self.config = config or LSMSystemConfig()
        self.habituation: Habituation = habituation or NullHabituation()
        self.association: Association = association or NullAssociation()
        self.writer: MemoryWriter = writer or AlwaysWriter()
        self._state: State = state_store.load() if state_store is not None else State()
        self._recent_ids: deque[str] = deque(maxlen=self.config.recent_context_window)

    async def _update_state(self, user_msg: str, reply: str) -> None:
        if not self.config.state_enabled or self.state_updater is None:
            return
        try:
            self._state = await self.state_updater.update(self._state, user_msg, reply)
        except Exception:  # noqa: BLE001
            log.exception("state updater failed")

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
        except Exception:  # noqa: BLE001
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

        dialogue = item.dialogue
        for i in range(0, len(dialogue), 2):
            user_msg = dialogue[i][1] if dialogue[i][0] == "user" else ""
            reply_idx = i + 1
            reply = (
                dialogue[reply_idx][1]
                if reply_idx < len(dialogue) and dialogue[reply_idx][0] == "assistant"
                else ""
            )
            if user_msg:
                self._write_memory(user_msg, "user")
            if reply:
                self._write_memory(reply, "assistant")
            if user_msg or reply:
                await self._update_state(user_msg, reply)

        retrieved = self._retrieve(item.question)
        ctx = PromptContext(
            state_projection=self._state.project(),
            retrieved=retrieved,
        )
        final_messages: Sequence[ChatMessage] = [
            ChatMessage(role="user", content=item.question)
        ]
        composed = compose(ctx, final_messages)
        options = GenerateOptions(temperature=0.0, max_tokens=self.config.answer_max_tokens)
        return await self.backend.generate(composed, options)
