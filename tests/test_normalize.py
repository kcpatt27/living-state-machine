"""Tests for the F.1.1 term normalizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lsm.core.embeddings import HashEmbedder
from lsm.core.normalize import (
    EmbeddingTermNormalizer,
    NullTermNormalizer,
    clean_term,
)

SCOPE_RESTAURANTS = "user.preferences:restaurants"
SCOPE_GENRE = "user.media:genre"
SCOPE_DRINKS = "user.preferences:drinks"


def test_clean_term_strips_punct_and_squashes_whitespace() -> None:
    assert clean_term("  Korean BBQ ! ") == "korean bbq"
    assert clean_term('"yakiniku"') == "yakiniku"
    assert clean_term("action   RPG") == "action rpg"
    assert clean_term("") == ""


def test_null_normalizer_is_identity_on_clean(tmp_path: Path) -> None:
    n = NullTermNormalizer()
    assert n.canonicalize(SCOPE_RESTAURANTS, "yakiniku") == "yakiniku"
    assert n.canonicalize(SCOPE_RESTAURANTS, "Korean BBQ") == "korean bbq"
    assert n.merge(SCOPE_RESTAURANTS, "kbbq", "korean bbq") == "korean bbq"


def test_first_observation_registers_canonical(tmp_path: Path) -> None:
    n = EmbeddingTermNormalizer(HashEmbedder(), tmp_path)
    out = n.canonicalize(SCOPE_RESTAURANTS, "yakiniku")
    assert out == "yakiniku"
    assert n.canonicals(SCOPE_RESTAURANTS) == ["yakiniku"]


def test_case_variant_resolves_via_clean_path(tmp_path: Path) -> None:
    n = EmbeddingTermNormalizer(HashEmbedder(), tmp_path)
    n.canonicalize(SCOPE_RESTAURANTS, "korean bbq")
    out = n.canonicalize(SCOPE_RESTAURANTS, "Korean BBQ")
    assert out == "korean bbq"
    assert n.canonicals(SCOPE_RESTAURANTS) == ["korean bbq"]


def test_explicit_merge_writes_alias(tmp_path: Path) -> None:
    n = EmbeddingTermNormalizer(HashEmbedder(), tmp_path)
    n.merge(SCOPE_RESTAURANTS, "kbbq", "korean bbq")
    assert n.canonicalize(SCOPE_RESTAURANTS, "kbbq") == "korean bbq"
    assert n.canonicalize(SCOPE_RESTAURANTS, "KBBQ") == "korean bbq"
    assert "kbbq" in n.aliases(SCOPE_RESTAURANTS)


def test_persistence_roundtrip(tmp_path: Path) -> None:
    n = EmbeddingTermNormalizer(HashEmbedder(), tmp_path)
    n.canonicalize(SCOPE_RESTAURANTS, "yakiniku")
    n.merge(SCOPE_RESTAURANTS, "kbbq", "korean bbq")

    n2 = EmbeddingTermNormalizer(HashEmbedder(), tmp_path)
    assert sorted(n2.canonicals(SCOPE_RESTAURANTS)) == ["korean bbq", "yakiniku"]
    assert n2.canonicalize(SCOPE_RESTAURANTS, "kbbq") == "korean bbq"


def test_scope_isolation(tmp_path: Path) -> None:
    n = EmbeddingTermNormalizer(HashEmbedder(), tmp_path)
    n.canonicalize(SCOPE_DRINKS, "java")
    n.canonicalize(SCOPE_GENRE, "java")
    assert n.canonicals(SCOPE_DRINKS) == ["java"]
    assert n.canonicals(SCOPE_GENRE) == ["java"]
    n.merge(SCOPE_DRINKS, "coffee", "java")
    assert n.canonicalize(SCOPE_DRINKS, "coffee") == "java"
    assert n.canonicalize(SCOPE_GENRE, "coffee") == "coffee"


class _SimEmbedder:
    """Embedder that returns a controllable vector per text.

    Used to gate the threshold path deterministically: terms in ``close``
    share a vector (cosine 1.0); everything else gets an orthogonal-ish
    sha-style vector.
    """

    dim = 8

    def __init__(self, close: list[set[str]]) -> None:
        self.close = close

    def _vec_for(self, text: str) -> np.ndarray:
        for i, group in enumerate(self.close):
            if text in group:
                v = np.zeros(self.dim, dtype=np.float32)
                v[i % self.dim] = 1.0
                return v
        # Unique vector per text (orthogonal to all `close` groups).
        h = abs(hash(text)) % 1000
        v = np.zeros(self.dim, dtype=np.float32)
        v[3] = float((h % 7) + 1)
        v[5] = float((h % 11) + 1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed(self, text: str) -> np.ndarray:
        return self._vec_for(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec_for(t) for t in texts], axis=0)


def test_threshold_gate_collapses_close_terms(tmp_path: Path) -> None:
    emb = _SimEmbedder(close=[{"korean bbq", "kbbq"}])
    n = EmbeddingTermNormalizer(emb, tmp_path, threshold=0.9)
    n.canonicalize(SCOPE_RESTAURANTS, "korean bbq")
    out = n.canonicalize(SCOPE_RESTAURANTS, "kbbq")
    assert out == "korean bbq"
    assert n.aliases(SCOPE_RESTAURANTS).get("kbbq") == "korean bbq"


def test_threshold_gate_keeps_distant_terms_separate(tmp_path: Path) -> None:
    emb = _SimEmbedder(close=[{"yakiniku"}])
    n = EmbeddingTermNormalizer(emb, tmp_path, threshold=0.9)
    n.canonicalize(SCOPE_RESTAURANTS, "yakiniku")
    out = n.canonicalize(SCOPE_RESTAURANTS, "ramen")
    assert out == "ramen"
    assert sorted(n.canonicals(SCOPE_RESTAURANTS)) == ["ramen", "yakiniku"]


def test_seed_file_loaded_when_live_missing(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed_aliases.json"
    seed_path.write_text(
        '{"user.media:genre": {"canonicals": ["action rpg"], '
        '"aliases": {"arpg": "action rpg"}}}',
        encoding="utf-8",
    )
    home = tmp_path / "home"
    n = EmbeddingTermNormalizer(HashEmbedder(), home, seed_path=seed_path)
    assert n.canonicals(SCOPE_GENRE) == ["action rpg"]
    assert n.canonicalize(SCOPE_GENRE, "ARPG") == "action rpg"


def test_empty_input_returns_empty(tmp_path: Path) -> None:
    n = EmbeddingTermNormalizer(HashEmbedder(), tmp_path)
    assert n.canonicalize(SCOPE_RESTAURANTS, "") == ""
    assert n.canonicalize(SCOPE_RESTAURANTS, "   ") == ""
