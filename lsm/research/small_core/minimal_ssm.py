"""Minimal SSM core — L11.

A hand-rolled, dependency-light state-space block written so reishi can
train a ~30-50M-parameter language model on the 5060 Ti without
depending on ``mamba-ssm`` (which has finicky Windows wheels). The block
is intentionally simple; correctness and legibility matter more than
peak throughput at this level.

Each token update is::

    h_t = sigmoid(A) * h_{t-1} + B * x_t
    y_t = C * h_t + D * x_t

where ``A``, ``B``, ``C`` are learned matrices and the recurrence uses
an element-wise gated update to stay stable across long sequences. The
block exposes the internal state ``h`` so downstream code (L12, L14)
can read or mutate it.

Usage
-----

The model is PyTorch-only. Import it lazily — this file is
``torch``-free at import time; ``build_model`` raises a clear error if
the dependency is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_TORCH_ERR = (
    "torch is required for the minimal SSM core. "
    "Install with: uv sync --extra research"
)


def _require_torch() -> Any:
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_TORCH_ERR) from e


@dataclass(slots=True)
class MinimalSSMConfig:
    """Hyperparameters for the minimal SSM language model.

    Defaults target ~30-50M parameters with ``d_model=512, n_layers=12,
    d_state=64`` and a 32k BPE tokenizer.
    """

    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 12
    d_state: int = 64
    d_inner: int = 1024  # expansion before the recurrence (like MLP in GPT)
    dropout: float = 0.0
    tie_embeddings: bool = True

    def parameter_count_estimate(self) -> int:
        """Back-of-envelope parameter count; useful for size planning."""
        emb = self.vocab_size * self.d_model
        per_layer = (
            2 * self.d_model * self.d_inner  # input + output projections
            + self.d_inner * self.d_state * 2  # B, C
            + self.d_state  # diag A
            + self.d_inner  # D
            + 2 * self.d_model  # two layer norms
        )
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return emb + self.n_layers * per_layer + lm_head


def build_model(config: MinimalSSMConfig) -> Any:
    """Construct the PyTorch model. Import torch lazily."""
    torch = _require_torch()
    from torch import nn

    class MinimalSSMBlock(nn.Module):
        """A single SSM block: input proj -> diagonal SSM -> output proj."""

        def __init__(self, cfg: MinimalSSMConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.in_proj = nn.Linear(cfg.d_model, cfg.d_inner, bias=False)
            self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)
            # log of 1 - sigmoid parameter; kept negative so ``A_stable`` sits in (0, 1).
            self.log_A = nn.Parameter(torch.randn(cfg.d_state) * 0.01 - 2.0)
            self.B = nn.Linear(cfg.d_inner, cfg.d_state, bias=False)
            self.C = nn.Linear(cfg.d_state, cfg.d_inner, bias=False)
            self.D = nn.Parameter(torch.zeros(cfg.d_inner))
            self.norm = nn.LayerNorm(cfg.d_model)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            x: Any,
            state: Any | None = None,
        ) -> tuple[Any, Any]:
            """x: (B, T, d_model); returns (y, final_state) with final state (B, d_state)."""
            b, t, _ = x.shape
            x_n = self.norm(x)
            u = self.in_proj(x_n)
            A = torch.sigmoid(self.log_A)

            if state is None:
                h = torch.zeros(b, self.cfg.d_state, device=x.device, dtype=x.dtype)
            else:
                h = state

            outs = []
            for step in range(t):
                u_t = u[:, step]
                b_t = self.B(u_t)
                h = A * h + b_t
                y_t = self.C(h) + self.D * u_t
                outs.append(y_t)
            y = torch.stack(outs, dim=1)
            y = self.dropout(self.out_proj(y))
            return x + y, h

    class MinimalSSM(nn.Module):
        """Stacked minimal SSM language model."""

        def __init__(self, cfg: MinimalSSMConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = nn.ModuleList([MinimalSSMBlock(cfg) for _ in range(cfg.n_layers)])
            self.final_norm = nn.LayerNorm(cfg.d_model)
            if cfg.tie_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.apply(_init_weights)

        def forward(
            self,
            input_ids: Any,
            states: list[Any] | None = None,
            return_states: bool = False,
        ) -> Any:
            x = self.token_emb(input_ids)
            new_states: list[Any] = []
            for i, block in enumerate(self.blocks):
                s = None if states is None else states[i]
                x, s_new = block(x, s)
                new_states.append(s_new)
            x = self.final_norm(x)
            if self.lm_head is None:
                logits = x @ self.token_emb.weight.T
            else:
                logits = self.lm_head(x)
            if return_states:
                return logits, new_states
            return logits

        def init_state(self, batch_size: int, device: Any, dtype: Any) -> list[Any]:
            return [
                torch.zeros(batch_size, self.cfg.d_state, device=device, dtype=dtype)
                for _ in range(self.cfg.n_layers)
            ]

    def _init_weights(module: Any) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=1.0 / math.sqrt(config.d_model))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=1.0 / math.sqrt(config.d_model))

    return MinimalSSM(config)


__all__ = ["MinimalSSMConfig", "build_model"]
