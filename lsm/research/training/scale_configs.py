"""Pre-tuned TrainerConfig bundles for the 5060 Ti — L13.

Defines the canonical size-tiers we iterate on::

    tiny      ~12M   (debug / CI)
    small     ~30M   (first real run; L11 target)
    medium    ~60M   (ablation matrix)
    large     ~120M  (L13 scale-up; local-first)
    xl        ~300M  (cloud optional; noted in docs)

Each entry returns a :class:`TrainerConfig` sized so a forward + backward
with ``bf16`` fits within the ~12 GB of VRAM budgeted for the AI on the
5060 Ti. Sequence length and batch/grad-accum are chosen per-tier.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from lsm.research.small_core.minimal_ssm import MinimalSSMConfig
from lsm.research.small_core.stentor_ssm import StentorSSMConfig
from lsm.research.training.trainer import TrainerConfig


def _promote(base: MinimalSSMConfig) -> StentorSSMConfig:
    """Copy a :class:`MinimalSSMConfig` into a :class:`StentorSSMConfig`."""
    return StentorSSMConfig(**asdict(base))


def _base(name: str) -> TrainerConfig:
    return TrainerConfig(
        data_path=Path(f"data/tokens_{name}.bin"),
        ckpt_dir=Path(f"runs/{name}/ckpts"),
        wandb_run_name=name,
    )


def tiny(*, stentor: bool = False) -> TrainerConfig:
    cfg = _base("tiny")
    base = MinimalSSMConfig(
        vocab_size=16_000, d_model=256, n_layers=6, d_state=32, d_inner=512
    )
    model = _promote(base) if stentor else base
    return replace(
        cfg,
        model_kind="stentor" if stentor else "minimal",
        model=model,
        vocab_size=base.vocab_size,
        seq_len=512,
        batch_size=8,
        grad_accum=2,
        steps=2_000,
        lr=5e-4,
    )


def small(*, stentor: bool = False) -> TrainerConfig:
    cfg = _base("small")
    base = MinimalSSMConfig(
        vocab_size=32_000, d_model=512, n_layers=12, d_state=64, d_inner=1024
    )
    model = _promote(base) if stentor else base
    return replace(
        cfg,
        model_kind="stentor" if stentor else "minimal",
        model=model,
        vocab_size=base.vocab_size,
        seq_len=1024,
        batch_size=4,
        grad_accum=4,
        steps=20_000,
    )


def medium(*, stentor: bool = False) -> TrainerConfig:
    cfg = _base("medium")
    base = MinimalSSMConfig(
        vocab_size=32_000, d_model=768, n_layers=16, d_state=96, d_inner=1536
    )
    model = _promote(base) if stentor else base
    return replace(
        cfg,
        model_kind="stentor" if stentor else "minimal",
        model=model,
        vocab_size=base.vocab_size,
        seq_len=1024,
        batch_size=2,
        grad_accum=8,
        steps=40_000,
        lr=2.5e-4,
    )


def large(*, stentor: bool = False) -> TrainerConfig:
    cfg = _base("large")
    base = MinimalSSMConfig(
        vocab_size=32_000, d_model=1024, n_layers=20, d_state=128, d_inner=2048
    )
    model = _promote(base) if stentor else base
    return replace(
        cfg,
        model_kind="stentor" if stentor else "minimal",
        model=model,
        vocab_size=base.vocab_size,
        seq_len=1024,
        batch_size=1,
        grad_accum=16,
        steps=80_000,
        lr=2e-4,
    )


PROFILES = {
    "tiny": tiny,
    "small": small,
    "medium": medium,
    "large": large,
}


def profile(name: str, *, stentor: bool = False) -> TrainerConfig:
    fn = PROFILES.get(name)
    if fn is None:
        raise ValueError(f"unknown profile: {name!r}; choose from {list(PROFILES)}")
    return fn(stentor=stentor)


__all__ = ["profile", "PROFILES", "tiny", "small", "medium", "large"]
