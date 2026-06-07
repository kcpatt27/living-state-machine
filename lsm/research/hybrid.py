"""Hybrid state bridge — L14.

The wrapped 14B LLM consumes a textual state projection (``s_t``) in the
prompt. The custom recurrent core produces a *vector* state. L14 closes
the loop: after each turn the custom core advances its state on the
conversation tokens; the final state is summarised into a short text
block and fed into the LSM prompt as ``s_t``.

The summariser runs in one of three modes:

- ``topk_tokens`` (default, no extra training): greedy decode the
  custom core for K tokens conditioned on a "summarise:" prefix and use
  that as the text projection.
- ``feature_report``: describe the state vector numerically — top-k
  dimensions by magnitude with learned labels (requires a label
  dictionary; not trained yet).
- ``nearest_neighbours``: use the core's decoded summary as a semantic
  query into the LSM memory store and project the top-k nearest durable
  memories. A learned ``projection`` (state-vector -> embedder space) is a
  future enhancement; until it is trained we route through text, which is
  what the LLM consumes anyway.

Only ``topk_tokens`` and ``nearest_neighbours`` ship with Phase 2's
opening; ``feature_report`` (numeric top-k state dims with learned labels)
is scaffolding for future research.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lsm.core.embeddings import Embedder
from lsm.core.memory import MemoryStore
from lsm.core.state import State, StateUpdater
from lsm.research.small_core.minimal_ssm import _require_torch

log = logging.getLogger(__name__)


@dataclass(slots=True)
class HybridConfig:
    """Knobs for :class:`HybridStateUpdater`."""

    mode: str = "topk_tokens"  # "topk_tokens" | "nearest_neighbours"
    summary_tokens: int = 48
    top_k_neighbours: int = 4
    device: str = "cuda"


class HybridStateUpdater:
    """Custom-core state updater that produces a text projection for the LLM."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: HybridConfig | None = None,
        memory: MemoryStore | None = None,
        embedder: Embedder | None = None,
        projection: Any | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or HybridConfig()
        self.memory = memory
        self.embedder = embedder
        # Optional learned projection from state-vector to embedder-vector.
        self.projection = projection
        self._states: list[Any] | None = None

        # Move model to device
        torch = _require_torch()
        self.model.to(torch.device(self.config.device))

    async def update(
        self,
        prev: State,
        user_msg: str,
        assistant_reply: str,
    ) -> State:
        """Advance the custom core and write a fresh text projection."""
        torch = _require_torch()
        device = torch.device(self.config.device)
        vocab_size = self.model.token_emb.num_embeddings
        raw_ids = self.tokenizer.encode(
            f"User: {user_msg}\nAssistant: {assistant_reply}\n",
        )
        filtered_ids = [tid for tid in raw_ids if tid < vocab_size]
        if not filtered_ids:
            filtered_ids = [0]
        ids = torch.tensor([filtered_ids], device=device)

        with torch.no_grad():
            if self._states is None:
                self._states = self.model.init_state(1, device, torch.float32)
            _, self._states = self.model(ids, states=self._states, return_states=True)

        summary = await self._summarise_topk(
            seed_prompt=f"Summarise the user's latest turn: {user_msg}"
        )
        if self.config.mode == "topk_tokens":
            projection = summary
        elif self.config.mode == "nearest_neighbours":
            # Use the core's decoded "thought" as a semantic query into
            # memory; fall back to the raw user turn when the undertrained
            # core emits an empty/degenerate summary.
            projection = self._summarise_neighbours(query_text=summary or user_msg)
        else:
            raise ValueError(f"unknown hybrid mode: {self.config.mode!r}")

        if self.embedder is not None:
            vec = self.embedder.embed(projection)
        else:
            vec = self._flatten_last_state()

        return State(
            text=projection,
            vector=vec,
            updated_at=time.time(),
            revision=prev.revision + 1,
            notes=prev.notes | {"hybrid_mode": self.config.mode},
        )

    async def _summarise_topk(self, *, seed_prompt: str) -> str:
        torch = _require_torch()
        device = torch.device(self.config.device)
        vocab_size = self.model.token_emb.num_embeddings
        raw_ids = self.tokenizer.encode(seed_prompt)
        filtered_ids = [tid for tid in raw_ids if tid < vocab_size]
        if not filtered_ids:
            filtered_ids = [0]
        input_ids = torch.tensor([filtered_ids], device=device)

        out_ids: list[int] = []
        with torch.no_grad():
            logits, states = self.model(input_ids, states=self._states, return_states=True)
            next_id = int(torch.argmax(logits[:, -1], dim=-1).item())
            out_ids.append(next_id)
            for _ in range(self.config.summary_tokens - 1):
                one = torch.tensor([[next_id]], device=device)
                logits, states = self.model(one, states=states, return_states=True)
                next_id = int(torch.argmax(logits[:, -1], dim=-1).item())
                out_ids.append(next_id)
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    def _summarise_neighbours(self, *, query_text: str) -> str:
        """Render a text projection from the memory neighbours of ``query_text``.

        The custom core's decoded summary is used as the semantic query so the
        recurrent state stays in the loop; the resulting projection is the
        top-k nearest durable memories. A learned ``projection`` (state-vector
        -> embedder space) is a future enhancement once it is trained — until
        then we route through text, which is what the LLM consumes anyway.
        """
        if self.memory is None:
            log.warning("nearest_neighbours mode needs a memory store")
            return ""
        if not query_text.strip():
            return ""
        try:
            hits = self.memory.query(query_text, top_k=self.config.top_k_neighbours)
        except Exception:
            log.exception("neighbour query failed")
            return ""
        lines = [f"- {hit.entry.text}" for hit in hits if hit.entry.text.strip()]
        return "\n".join(lines)

    def _flatten_last_state(self) -> Any:
        if self._states is None:
            return None
        torch = _require_torch()
        pieces: list[Any] = []
        for s in self._states:
            if isinstance(s, dict):
                pieces.append(s["h"].flatten())
            else:
                pieces.append(s.flatten())
        full = torch.cat(pieces).detach().cpu().numpy()
        return full


def hybrid_from_checkpoint(
    ckpt_path: Path,
    *,
    tokenizer: Any,
    config: HybridConfig | None = None,
    memory: MemoryStore | None = None,
    embedder: Embedder | None = None,
) -> HybridStateUpdater:
    """Load a trained custom core from ``ckpt_path`` and return a ready updater."""
    torch = _require_torch()
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    kind = cfg.get("model_kind", "minimal")
    if kind == "minimal":
        from lsm.research.small_core.minimal_ssm import MinimalSSMConfig, build_model

        model_cfg = MinimalSSMConfig(**cfg.get("model", {}))
        model = build_model(model_cfg)
    else:
        from lsm.research.small_core.stentor_ssm import StentorSSMConfig, build_model

        model_cfg = StentorSSMConfig(**cfg.get("model", {}))
        model = build_model(model_cfg)
    model.load_state_dict(payload["model"])
    return HybridStateUpdater(
        model=model,
        tokenizer=tokenizer,
        config=config or HybridConfig(),
        memory=memory,
        embedder=embedder,
    )


def _adheres_to_state_updater(u: HybridStateUpdater) -> StateUpdater:
    """Type-check helper: HybridStateUpdater satisfies :class:`StateUpdater`."""
    return u


__all__ = ["HybridConfig", "HybridStateUpdater", "hybrid_from_checkpoint"]
