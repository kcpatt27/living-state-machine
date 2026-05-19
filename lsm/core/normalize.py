"""Term normalization (F.1.1).

General, infrastructure-shaped alias resolution. Used by the fact extractor
so canonical surface forms (`korean bbq` vs `Korean BBQ` vs `kbbq`,
`action RPG` vs `arpg` vs `action role-playing`) collapse to a single
value per scope without hardcoding domain tables.

Three pieces:

- :class:`TermNormalizer` Protocol — ``canonicalize(scope, raw) -> str`` plus
  an explicit ``merge(scope, a, b)`` for user-driven alias creation.
- :class:`NullTermNormalizer` — lowercase + whitespace squash, no learning.
  CI default and a stable baseline.
- :class:`EmbeddingTermNormalizer` — backed by the existing
  :class:`~lsm.core.embeddings.Embedder` protocol. Resolves novel surface
  forms by cosine similarity against previously seen canonicals in the
  same scope; persists every observed alias to ``$LSM_HOME/aliases.json``
  so the system "learns" canonical forms from what it actually sees.

Scopes are free-form strings like ``"user.preferences:restaurants"`` or
``"user.media:genre"``. Aliases never cross scopes — registering ``"java"``
under drinks does not affect ``user.media:genre``.

The seed file ``seed_aliases.json`` (optional, repo-shipped) is loaded only
when the live ``aliases.json`` is empty. It contains a tiny cold-start set
so the first session is usable; everything else is learned at runtime.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from lsm.core.embeddings import Embedder

_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\"'.,;:!?()\[\]{}]+|[\s\"'.,;:!?()\[\]{}]+$")


def clean_term(raw: str) -> str:
    """Lowercase, strip edge punctuation/quotes, squash internal whitespace."""
    if not raw:
        return ""
    s = raw.strip().lower()
    s = _EDGE_PUNCT.sub("", s)
    s = _WS.sub(" ", s)
    return s.strip()


@runtime_checkable
class TermNormalizer(Protocol):
    """Resolve a raw surface form to its canonical form within a scope."""

    def canonicalize(self, scope: str, raw: str) -> str: ...

    def merge(self, scope: str, alias: str, canonical: str) -> str: ...


class NullTermNormalizer:
    """Identity normalizer: clean only, no learning.

    Used in CI and as the ablation baseline. Two surface forms that
    ``clean_term`` reduces to the same string still collapse; everything
    else is left alone.
    """

    def canonicalize(self, scope: str, raw: str) -> str:
        return clean_term(raw)

    def merge(self, scope: str, alias: str, canonical: str) -> str:
        return clean_term(canonical)


class EmbeddingTermNormalizer:
    """Embedding-backed normalizer with persisted per-scope alias tables.

    Resolution order on ``canonicalize(scope, raw)``:

    1. Clean ``raw``. If the cleaned form is empty, return ``""``.
    2. If cleaned form is already a canonical in this scope, return it.
    3. If cleaned form is an alias in this scope, return its canonical.
    4. Embed cleaned form, compute cosine vs every canonical in scope.
       If max similarity >= ``threshold``, register cleaned form as an
       alias of that canonical and return the canonical.
    5. Otherwise register cleaned form as a new canonical and return it.

    Steps 4 and 5 are how the system "learns" — every novel form either
    joins an existing canonical or becomes one. The alias table is
    inspectable in ``$LSM_HOME/aliases.json``.
    """

    FILENAME = "aliases.json"
    SEED_FILENAME = "seed_aliases.json"

    def __init__(
        self,
        embedder: Embedder,
        home: Path,
        *,
        threshold: float = 0.86,
        seed_path: Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.home = home
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / self.FILENAME
        self.threshold = float(threshold)
        self._scopes: dict[str, dict[str, object]] = {}
        # Lazy per-scope cache of (canonical -> embedding vector).
        self._vec_cache: dict[str, dict[str, np.ndarray]] = {}
        self._load(seed_path=seed_path)

    def _load(self, *, seed_path: Path | None) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._scopes = {
                str(scope): {
                    "canonicals": list(payload.get("canonicals") or []),
                    "aliases": dict(payload.get("aliases") or {}),
                }
                for scope, payload in raw.items()
            }
            return
        if seed_path is not None and seed_path.exists():
            raw = json.loads(seed_path.read_text(encoding="utf-8"))
            self._scopes = {
                str(scope): {
                    "canonicals": list(payload.get("canonicals") or []),
                    "aliases": dict(payload.get("aliases") or {}),
                }
                for scope, payload in raw.items()
            }
            self._save()
            return
        self._scopes = {}
        self._save()

    def _save(self) -> None:
        payload = {
            scope: {
                "canonicals": list(state["canonicals"]),  # type: ignore[arg-type]
                "aliases": dict(state["aliases"]),  # type: ignore[arg-type]
            }
            for scope, state in self._scopes.items()
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        # Windows: transient file locks can race os.replace in tests/CI.
        for attempt in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def _ensure_scope(self, scope: str) -> dict[str, object]:
        if scope not in self._scopes:
            self._scopes[scope] = {"canonicals": [], "aliases": {}}
            self._vec_cache.setdefault(scope, {})
        return self._scopes[scope]

    def _cache_for(self, scope: str) -> dict[str, np.ndarray]:
        cache = self._vec_cache.setdefault(scope, {})
        canonicals: list[str] = self._scopes[scope]["canonicals"]  # type: ignore[assignment]
        missing = [c for c in canonicals if c not in cache]
        if missing:
            vecs = self.embedder.embed_batch(missing)
            for term, vec in zip(missing, vecs, strict=False):
                cache[term] = _l2_normalize(np.asarray(vec, dtype=np.float32))
        return cache

    def canonicalize(self, scope: str, raw: str) -> str:
        cleaned = clean_term(raw)
        if not cleaned:
            return ""
        state = self._ensure_scope(scope)
        canonicals: list[str] = state["canonicals"]  # type: ignore[assignment]
        aliases: dict[str, str] = state["aliases"]  # type: ignore[assignment]

        if cleaned in canonicals:
            return cleaned
        if cleaned in aliases:
            return aliases[cleaned]

        # Step 4: compare embeddings against known canonicals.
        if canonicals:
            cache = self._cache_for(scope)
            query = _l2_normalize(np.asarray(self.embedder.embed(cleaned), dtype=np.float32))
            best_term = ""
            best_sim = -1.0
            for term in canonicals:
                sim = float(np.dot(query, cache[term]))
                if sim > best_sim:
                    best_sim = sim
                    best_term = term
            if best_sim >= self.threshold:
                aliases[cleaned] = best_term
                self._save()
                return best_term

        # Step 5: register as a new canonical.
        canonicals.append(cleaned)
        self._save()
        # Invalidate cache so next embed picks the new canonical up.
        self._vec_cache.setdefault(scope, {})
        return cleaned

    def merge(self, scope: str, alias: str, canonical: str) -> str:
        """Force ``alias`` to resolve to ``canonical`` in ``scope``.

        Registers ``canonical`` as a canonical if it is not already, then
        writes the alias entry. Returns the cleaned canonical.
        """
        c_alias = clean_term(alias)
        c_can = clean_term(canonical)
        if not c_can:
            return ""
        state = self._ensure_scope(scope)
        canonicals: list[str] = state["canonicals"]  # type: ignore[assignment]
        aliases: dict[str, str] = state["aliases"]  # type: ignore[assignment]
        if c_can not in canonicals:
            canonicals.append(c_can)
            self._vec_cache.setdefault(scope, {})
        if c_alias and c_alias != c_can:
            aliases[c_alias] = c_can
        self._save()
        return c_can

    # Debug / inspection helpers (no behavior change).
    def canonicals(self, scope: str) -> list[str]:
        if scope not in self._scopes:
            return []
        return list(self._scopes[scope]["canonicals"])  # type: ignore[arg-type]

    def aliases(self, scope: str) -> dict[str, str]:
        if scope not in self._scopes:
            return {}
        return dict(self._scopes[scope]["aliases"])  # type: ignore[arg-type]


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v
    return v / n
