"""Hidden state ``s_t``.

At L3 ``s_t`` is a dual representation:

- **Text projection** — a compact natural-language summary the LLM can read in
  the system prompt. It is the primary, canonical form.
- **Vector** — the fastembed-encoded version of the text projection. Used as a
  retrieval-conditioning signal in L4+, and as the input to the Phase-2
  custom recurrent core (L14).

Both are persisted atomically: write to ``*.tmp``, then ``os.replace``. Reads
tolerate missing files (first run) and fall back to empty state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from lsm.core.embeddings import Embedder
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, LLMBackend

log = logging.getLogger(__name__)


@dataclass(slots=True)
class State:
    """Hidden state ``s_t`` (text + vector)."""

    text: str = ""
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    updated_at: float = 0.0
    revision: int = 0
    notes: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.text and self.vector.size == 0

    def project(self, max_chars: int = 1200) -> str:
        """Return a prompt-friendly textual projection."""
        if not self.text:
            return ""
        return self.text if len(self.text) <= max_chars else self.text[: max_chars - 1] + "…"


class StateStore:
    """Filesystem-backed store for one :class:`State` instance."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.json_path = self.root / "s_t.json"
        self.npy_path = self.root / "s_t.npy"

    def load(self) -> State:
        if not self.json_path.exists():
            return State()
        meta = json.loads(self.json_path.read_text(encoding="utf-8"))
        vec = (
            np.load(self.npy_path).astype(np.float32)
            if self.npy_path.exists()
            else np.zeros(0, dtype=np.float32)
        )
        return State(
            text=meta.get("text", ""),
            vector=vec,
            updated_at=float(meta.get("updated_at", 0.0)),
            revision=int(meta.get("revision", 0)),
            notes=dict(meta.get("notes", {})),
        )

    def save(self, state: State) -> None:
        tmp_json = self.json_path.with_suffix(".json.tmp")
        tmp_npy = self.npy_path.with_suffix(".npy.tmp")

        meta = {
            "text": state.text,
            "updated_at": state.updated_at,
            "revision": state.revision,
            "notes": state.notes,
            "vector_dim": int(state.vector.size),
        }
        tmp_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Use a file handle so numpy writes to this exact path.
        # np.save(path, ...) auto-appends ".npy" if the path doesn't end in it,
        # which would produce "s_t.npy.tmp.npy" and break the os.replace below.
        with open(tmp_npy, "wb") as f:
            np.save(f, state.vector, allow_pickle=False)
        os.replace(tmp_json, self.json_path)
        os.replace(tmp_npy, self.npy_path)

    def snapshot(self, tag: str) -> Path:
        """Snapshot the current state directory under ``snapshots/<tag>/``."""
        if not tag:
            raise ValueError("tag must be non-empty")
        dest = self.root / "snapshots" / tag
        if dest.exists():
            raise FileExistsError(f"snapshot {tag!r} already exists at {dest}")
        dest.mkdir(parents=True)
        for f in (self.json_path, self.npy_path):
            if f.exists():
                shutil.copy2(f, dest / f.name)
        return dest

    def restore(self, tag: str) -> None:
        src = self.root / "snapshots" / tag
        if not src.exists():
            raise FileNotFoundError(f"no snapshot {tag!r}")
        for f in ("s_t.json", "s_t.npy"):
            s = src / f
            if s.exists():
                shutil.copy2(s, self.root / f)

    def list_snapshots(self) -> list[str]:
        d = self.root / "snapshots"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())


@runtime_checkable
class StateUpdater(Protocol):
    """Produces the next :class:`State` from the previous one plus the turn."""

    async def update(
        self, prev: State, user_msg: str, assistant_reply: str
    ) -> State: ...


class NullStateUpdater:
    """Ablation: never updates state. ``s_t`` stays at its initial empty value."""

    async def update(
        self, prev: State, user_msg: str, assistant_reply: str
    ) -> State:
        return prev


class PassthroughStateUpdater:
    """Cheap baseline: concatenate turn into the text and re-embed.

    Keeps at most ``max_chars`` of the most-recent content. Useful as a cost
    baseline against the LLM-written summarizer.
    """

    def __init__(self, embedder: Embedder, max_chars: int = 1200) -> None:
        self._embedder = embedder
        self._max_chars = max_chars

    async def update(
        self, prev: State, user_msg: str, assistant_reply: str
    ) -> State:
        joined = f"{prev.text}\nuser: {user_msg}\nassistant: {assistant_reply}".strip()
        if len(joined) > self._max_chars:
            joined = joined[-self._max_chars :]
        vec = self._embedder.embed(joined) if joined else np.zeros(self._embedder.dim, dtype=np.float32)
        return State(
            text=joined,
            vector=vec,
            updated_at=time.time(),
            revision=prev.revision + 1,
            notes=prev.notes,
        )


SUMMARIZER_SYSTEM = (
    "You maintain a compact running summary of a conversation between a user "
    "and an AI assistant. Merge the latest exchange into the prior summary.\n\n"
    "Keep ≤ 120 words. Prioritise stable facts about the user (preferences, "
    "ongoing tasks, decisions, names), then unresolved questions, then key "
    "context. Drop small talk. Use short declarative sentences. Never invent "
    "details. Output only the summary, no preamble."
)


class LLMSummarizerUpdater:
    """L3 default: ask the LLM itself to merge the new turn into ``s_t``.

    The same backend is reused (a fresh non-streaming call per turn). An
    embedder projects the text into the vector slot.
    """

    def __init__(
        self,
        backend: LLMBackend,
        embedder: Embedder,
        *,
        max_words: int = 120,
        options: GenerateOptions | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._max_words = max_words
        self._options = options or GenerateOptions(temperature=0.2, max_tokens=256)

    async def update(
        self, prev: State, user_msg: str, assistant_reply: str
    ) -> State:
        prompt_messages: Sequence[ChatMessage] = [
            ChatMessage(role="system", content=SUMMARIZER_SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    "Prior summary:\n"
                    f"{prev.text or '(empty)'}\n\n"
                    "Latest exchange:\n"
                    f"user: {user_msg}\n"
                    f"assistant: {assistant_reply}\n\n"
                    f"Produce the new summary (≤ {self._max_words} words)."
                ),
            ),
        ]
        try:
            new_text = (await self._backend.generate(prompt_messages, self._options)).strip()
        except Exception:  # noqa: BLE001 — degrade, don't crash the turn
            log.warning("state updater failed; keeping prior state", exc_info=True)
            return prev

        if not new_text:
            return prev

        vec = self._embedder.embed(new_text)
        return State(
            text=new_text,
            vector=vec,
            updated_at=time.time(),
            revision=prev.revision + 1,
            notes=prev.notes,
        )


def state_to_dict(s: State) -> dict[str, object]:
    """JSON-friendly view of a state (omits the raw vector)."""
    d = asdict(s)
    d.pop("vector", None)
    d["vector_dim"] = int(s.vector.size)
    return d
