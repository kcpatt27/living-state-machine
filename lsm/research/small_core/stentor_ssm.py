"""Stentor-augmented SSM — L12.

Adds two primitives directly inside the recurrence, rather than in the
Python wrapper:

**Habituation gate (``g_hab``).** A learned per-token sensitivity score
is driven toward zero when a token has been seen recently. The gate
scales the input contribution to the state update, so repeated filler
tokens stop moving the state while novel tokens still do.

**Association trace (``M``).** A low-rank outer-product memory that
accumulates token/state correlations Hebbian-style and contributes an
additive bias to the next state. This is the in-network cousin of the
Python ``HebbianAssociation`` used in Phase 1.

The full recurrence becomes::

    g_hab_t   = sigmoid(W_h x_t - gamma * hab_count_t)
    h_tilde_t = A * h_{t-1} + B * (g_hab_t * x_t)
    M_t       = (1 - lam) * M_{t-1} + eta * outer(phi(x_t), psi(h_tilde_t))
    h_t       = h_tilde_t + beta * M_t @ phi(x_t)
    y_t       = C * h_t + D * x_t

Everything is differentiable; the training loop (L11) optimises the full
model end-to-end on language modelling loss.

Ablations
---------

- ``use_habituation=False`` reverts the gate to 1.0 everywhere.
- ``use_association=False`` freezes ``M`` at zero.

The Phase 1 findings (MRR on ``habituation_signal_v1`` / ``association_signal_v1``)
set the expectation: the Stentor-augmented core should beat the plain
minimal SSM on synthetic memory-retrieval sub-tasks with the same
parameter budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lsm.research.small_core.minimal_ssm import (
    MinimalSSMConfig,
    _require_torch,
    diagonal_ssm_scan,
)


@dataclass(slots=True)
class StentorSSMConfig(MinimalSSMConfig):
    """Hyperparameters with two extra ablation knobs."""

    use_habituation: bool = True
    use_association: bool = True
    hab_gamma: float = 0.5
    assoc_eta: float = 0.05
    assoc_lambda: float = 0.01
    assoc_rank: int = 16
    assoc_beta: float = 0.1


def build_model(config: StentorSSMConfig) -> Any:
    """Build the Stentor-augmented language model."""
    torch = _require_torch()
    from torch import nn

    class StentorBlock(nn.Module):
        def __init__(self, cfg: StentorSSMConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.in_proj = nn.Linear(cfg.d_model, cfg.d_inner, bias=False)
            self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)
            self.norm = nn.LayerNorm(cfg.d_model)
            self.log_A = nn.Parameter(torch.randn(cfg.d_state) * 0.01 - 2.0)
            self.B = nn.Linear(cfg.d_inner, cfg.d_state, bias=False)
            self.C = nn.Linear(cfg.d_state, cfg.d_inner, bias=False)
            self.D = nn.Parameter(torch.zeros(cfg.d_inner))
            self.dropout = nn.Dropout(cfg.dropout)

            # Habituation gate params.
            self.hab_gate = nn.Linear(cfg.d_inner, cfg.d_inner, bias=True)
            self.hab_ema_decay = nn.Parameter(torch.tensor(0.02))

            # Association trace params (low-rank).
            self.phi = nn.Linear(cfg.d_inner, cfg.assoc_rank, bias=False)
            self.psi = nn.Linear(cfg.d_state, cfg.assoc_rank, bias=False)

        def forward(
            self,
            x: Any,
            block_state: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            b, t, _ = x.shape
            u = self.in_proj(self.norm(x))
            A = torch.sigmoid(self.log_A)
            cfg = self.cfg

            if block_state is None:
                h = torch.zeros(b, cfg.d_state, device=x.device, dtype=x.dtype)
                M = torch.zeros(b, cfg.assoc_rank, cfg.assoc_rank, device=x.device, dtype=x.dtype)
                hab_ema = torch.zeros_like(u[:, 0])
            else:
                h = block_state["h"]
                M = block_state["M"]
                hab_ema = block_state["hab_ema"]

            decay = torch.sigmoid(self.hab_ema_decay)
            u_gated = u
            if cfg.use_habituation:
                gated = []
                for step in range(t):
                    u_t = u[:, step]
                    g_hab = torch.sigmoid(self.hab_gate(u_t) - cfg.hab_gamma * hab_ema)
                    u_t_g = g_hab * u_t
                    hab_ema = (1 - decay) * hab_ema + decay * u_t_g.abs()
                    gated.append(u_t_g)
                u_gated = torch.stack(gated, dim=1)
            elif t > 1:
                u_gated = u

            h_seq = diagonal_ssm_scan(A, self.B(u_gated))

            if cfg.use_association:
                outs = []
                for step in range(t):
                    u_t = u_gated[:, step]
                    h = h_seq[:, step]
                    phi_t = self.phi(u_t)
                    psi_t = self.psi(h)
                    M = (1 - cfg.assoc_lambda) * M + cfg.assoc_eta * torch.einsum(
                        "bi,bj->bij", phi_t, psi_t
                    )
                    assoc_bias = torch.einsum("bij,bi->bj", M, phi_t)
                    assoc_h = torch.einsum("rd,br->bd", self.psi.weight, assoc_bias)
                    h = h + cfg.assoc_beta * assoc_h
                    outs.append(self.C(h) + self.D * u_t)
                y = torch.stack(outs, dim=1)
            else:
                y = self.C(h_seq) + self.D.unsqueeze(0).unsqueeze(0) * u_gated
            y = self.dropout(self.out_proj(y))
            return x + y, {"h": h_seq[:, -1], "M": M, "hab_ema": hab_ema}

    class StentorSSM(nn.Module):
        def __init__(self, cfg: StentorSSMConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = nn.ModuleList([StentorBlock(cfg) for _ in range(cfg.n_layers)])
            self.final_norm = nn.LayerNorm(cfg.d_model)
            if cfg.tie_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        def forward(
            self,
            input_ids: Any,
            states: list[dict[str, Any]] | None = None,
            return_states: bool = False,
        ) -> Any:
            x = self.token_emb(input_ids)
            new_states: list[dict[str, Any]] = []
            for i, block in enumerate(self.blocks):
                s = None if states is None else states[i]
                x, s_new = block(x, s)
                new_states.append(s_new)
            x = self.final_norm(x)
            logits = x @ self.token_emb.weight.T if self.lm_head is None else self.lm_head(x)
            if return_states:
                return logits, new_states
            return logits

        def init_state(self, batch_size: int, device: Any, dtype: Any) -> list[dict[str, Any]]:
            states: list[dict[str, Any]] = []
            for _ in range(self.cfg.n_layers):
                states.append(
                    {
                        "h": torch.zeros(batch_size, self.cfg.d_state, device=device, dtype=dtype),
                        "M": torch.zeros(
                            batch_size,
                            self.cfg.assoc_rank,
                            self.cfg.assoc_rank,
                            device=device,
                            dtype=dtype,
                        ),
                        "hab_ema": torch.zeros(
                            batch_size, self.cfg.d_inner, device=device, dtype=dtype
                        ),
                    }
                )
            return states

    return StentorSSM(config)


__all__ = ["StentorSSMConfig", "build_model"]
