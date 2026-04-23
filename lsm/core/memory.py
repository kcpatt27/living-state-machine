"""Memory: entries, scoring, stores.

L4 introduces a single ``DurableMemory`` tier backed by LanceDB. L8 splits
memory into fast / session / durable tiers with promotion rules; the tier
classes live here too.

All stores implement :class:`MemoryStore`. A fast :class:`InMemoryStore` is
used in tests and in small experiments; :class:`LanceDBStore` is the default
for durable persistence.

Retrieval is decomposed so later levels can inject habituation damping and
association boosts without changing the store itself:

    base_score = cosine(query, entry)
    score = combiner(base_score, entry, context_keys)

Combiner defaults to identity at L4. L5 swaps in habituation damping; L6 adds
association boost. See ``lsm/core/habituation.py`` and
``lsm/core/association.py``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from lsm.core.embeddings import Embedder

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryEntry:
    """A single memory."""

    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    trust: float = 0.5
    habituation: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredEntry:
    """A memory entry paired with its final retrieval score.

    Components let callers see how the final score was composed (useful for
    debugging, ablations, and UI).
    """

    entry: MemoryEntry
    score: float
    components: dict[str, float] = field(default_factory=dict)


ScoreCombiner = Callable[[float, MemoryEntry, Sequence[str]], tuple[float, dict[str, float]]]


def identity_combiner(
    base: float, entry: MemoryEntry, context: Sequence[str]
) -> tuple[float, dict[str, float]]:
    """Default combiner: passthrough of cosine similarity."""
    _ = entry
    _ = context
    return base, {"cosine": base}


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for a vector-addressable memory store."""

    embedder: Embedder

    def put(
        self,
        text: str,
        *,
        source: str = "conversation",
        tags: Iterable[str] = (),
        meta: dict[str, Any] | None = None,
    ) -> MemoryEntry: ...

    def put_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """Insert a pre-built entry, preserving its id.

        Used by the multi-timescale memory to promote entries between tiers
        without losing the id (which association/habituation track).
        """
        ...

    def put_many(self, texts: Sequence[str], **kw: Any) -> list[MemoryEntry]: ...

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        context_keys: Sequence[str] = (),
        combiner: ScoreCombiner | None = None,
    ) -> list[ScoredEntry]: ...

    def get(self, entry_id: str) -> MemoryEntry | None: ...

    def delete(self, ids: Sequence[str]) -> int: ...

    def delete_by_source(self, source: str) -> int: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class InMemoryStore:
    """Simple numpy-backed store. Fast, deterministic, perfect for tests."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._entries: dict[str, MemoryEntry] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def put(
        self,
        text: str,
        *,
        source: str = "conversation",
        tags: Iterable[str] = (),
        meta: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(text=text, source=source, tags=list(tags), meta=dict(meta or {}))
        self._entries[entry.id] = entry
        self._vectors[entry.id] = self.embedder.embed(text)
        return entry

    def put_entry(self, entry: MemoryEntry) -> MemoryEntry:
        self._entries[entry.id] = entry
        self._vectors[entry.id] = self.embedder.embed(entry.text)
        return entry

    def put_many(self, texts: Sequence[str], **kw: Any) -> list[MemoryEntry]:
        return [self.put(t, **kw) for t in texts]

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        context_keys: Sequence[str] = (),
        combiner: ScoreCombiner | None = None,
    ) -> list[ScoredEntry]:
        if not self._entries:
            return []
        cmb = combiner or identity_combiner
        q = self.embedder.embed(text)
        scored: list[ScoredEntry] = []
        for eid, entry in self._entries.items():
            base = _cosine(q, self._vectors[eid])
            final, comps = cmb(base, entry, context_keys)
            scored.append(ScoredEntry(entry=entry, score=final, components=comps))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def delete(self, ids: Sequence[str]) -> int:
        n = 0
        for i in ids:
            if i in self._entries:
                del self._entries[i]
                del self._vectors[i]
                n += 1
        return n

    def delete_by_source(self, source: str) -> int:
        targets = [i for i, e in self._entries.items() if e.source == source]
        return self.delete(targets)

    def count(self) -> int:
        return len(self._entries)

    def close(self) -> None:
        return None


class LanceDBStore:
    """Persistent LanceDB-backed store. Default for durable memory."""

    TABLE = "memory"

    def __init__(self, root: Path, embedder: Embedder) -> None:
        self.embedder = embedder
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            import lancedb
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "lancedb is required for LanceDBStore. Run: uv sync"
            ) from e
        self._lancedb = lancedb
        self._db = lancedb.connect(str(self._root))
        self._table = self._open_or_create()

    def _open_or_create(self) -> Any:
        import pyarrow as pa

        names = self._db.table_names()
        if self.TABLE in names:
            return self._db.open_table(self.TABLE)
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("text", pa.string()),
                ("vector", pa.list_(pa.float32(), self.embedder.dim)),
                ("source", pa.string()),
                ("tags", pa.list_(pa.string())),
                ("created_at", pa.float64()),
                ("updated_at", pa.float64()),
                ("trust", pa.float32()),
                ("habituation", pa.float32()),
                ("meta_json", pa.string()),
            ]
        )
        return self._db.create_table(self.TABLE, schema=schema, mode="create")

    @staticmethod
    def _row_to_entry(row: dict[str, Any]) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            text=row["text"],
            source=row["source"],
            tags=list(row.get("tags") or []),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            trust=float(row["trust"]),
            habituation=float(row["habituation"]),
            meta=json.loads(row.get("meta_json") or "{}"),
        )

    def put(
        self,
        text: str,
        *,
        source: str = "conversation",
        tags: Iterable[str] = (),
        meta: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(text=text, source=source, tags=list(tags), meta=dict(meta or {}))
        vec = self.embedder.embed(text)
        self._table.add(
            [
                {
                    "id": entry.id,
                    "text": entry.text,
                    "vector": vec.tolist(),
                    "source": entry.source,
                    "tags": entry.tags,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "trust": entry.trust,
                    "habituation": entry.habituation,
                    "meta_json": json.dumps(entry.meta),
                }
            ]
        )
        return entry

    def put_entry(self, entry: MemoryEntry) -> MemoryEntry:
        vec = self.embedder.embed(entry.text)
        self._table.add(
            [
                {
                    "id": entry.id,
                    "text": entry.text,
                    "vector": vec.tolist(),
                    "source": entry.source,
                    "tags": entry.tags,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "trust": entry.trust,
                    "habituation": entry.habituation,
                    "meta_json": json.dumps(entry.meta),
                }
            ]
        )
        return entry

    def put_many(self, texts: Sequence[str], **kw: Any) -> list[MemoryEntry]:
        return [self.put(t, **kw) for t in texts]

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        context_keys: Sequence[str] = (),
        combiner: ScoreCombiner | None = None,
    ) -> list[ScoredEntry]:
        q = self.embedder.embed(text).astype(np.float32)
        # LanceDB returns ``_distance`` (L2) by default. Convert to cosine via
        # pre-normalised embeddings when possible; else compute from vectors.
        results = (
            self._table.search(q.tolist())
            .metric("cosine")
            .limit(max(top_k * 4, top_k))
            .to_list()
        )
        cmb = combiner or identity_combiner
        scored: list[ScoredEntry] = []
        for row in results:
            base = 1.0 - float(row.get("_distance", 1.0))
            entry = self._row_to_entry(row)
            final, comps = cmb(base, entry, context_keys)
            scored.append(ScoredEntry(entry=entry, score=final, components=comps))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def get(self, entry_id: str) -> MemoryEntry | None:
        rows = self._table.search().where(f"id = '{entry_id}'").limit(1).to_list()
        if not rows:
            return None
        return self._row_to_entry(rows[0])

    def delete(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        quoted = ",".join(f"'{i}'" for i in ids)
        self._table.delete(f"id IN ({quoted})")
        return len(ids)

    def delete_by_source(self, source: str) -> int:
        escaped = source.replace("'", "''")
        self._table.delete(f"source = '{escaped}'")
        return -1  # LanceDB doesn't return affected row count; -1 = unknown

    def count(self) -> int:
        return int(self._table.count_rows())

    def close(self) -> None:
        return None


def build_store(
    kind: str, *, root: Path, embedder: Embedder
) -> MemoryStore:
    """Factory for memory stores."""
    if kind == "lancedb":
        return LanceDBStore(root=root, embedder=embedder)
    if kind == "memory":
        return InMemoryStore(embedder=embedder)
    raise ValueError(f"unknown memory store kind: {kind!r}")
