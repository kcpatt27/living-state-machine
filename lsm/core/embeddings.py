"""Embedding backends.

Canonical production embedder is :class:`FastEmbedder` (ONNX on CPU, 384-d).
Tests use :class:`HashEmbedder`, a deterministic fake that requires no model
download.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Text -> dense vector."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, zero-dependency embedder for tests.

    Uses sha256 bytes as vector entries, normalized. Not meaningful for
    semantic retrieval, but stable and fast.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        raw = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (raw * ((self.dim // len(raw)) + 1))[: self.dim]
        arr = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
        arr = (arr - 127.5) / 127.5
        n = np.linalg.norm(arr)
        if n > 0:
            arr = arr / n
        return arr

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts], axis=0)


class FastEmbedder:
    """CPU ONNX embedder via ``fastembed``. Lazy-loaded."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: object | None = None
        self.dim = 384  # bge-small

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "fastembed is required for FastEmbedder. "
                "Run: uv sync"
            ) from e
        log.info("loading fastembed model: %s", self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        [vec] = list(self._model.embed([text]))  # type: ignore[attr-defined]
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        vecs = list(self._model.embed(texts))  # type: ignore[attr-defined]
        return np.asarray(vecs, dtype=np.float32)


def get_default_embedder(model_name: str | None = None) -> Embedder:
    """Build a :class:`FastEmbedder` using the provided or default model name."""
    return FastEmbedder(model_name=model_name or "BAAI/bge-small-en-v1.5")
