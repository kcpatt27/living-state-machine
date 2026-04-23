"""Sanity tests for the Phase 2 research scaffolding.

These run without torch installed (they only exercise pure-Python parts
like config construction and parameter estimation). Actual model
construction is tested under a ``torch`` skip-marker so it can execute
in full research environments but is a no-op in the Phase 1 CI.
"""

from __future__ import annotations

import importlib.util

import pytest

from lsm.research.small_core.minimal_ssm import MinimalSSMConfig
from lsm.research.small_core.stentor_ssm import StentorSSMConfig
from lsm.research.training.scale_configs import PROFILES, profile
from lsm.research.training.trainer import TrainerConfig, cosine_lr

has_torch = importlib.util.find_spec("torch") is not None


def test_minimal_ssm_param_estimate_scales_with_layers() -> None:
    small = MinimalSSMConfig(n_layers=6)
    large = MinimalSSMConfig(n_layers=24)
    assert large.parameter_count_estimate() > small.parameter_count_estimate()


def test_stentor_inherits_minimal_defaults() -> None:
    s = StentorSSMConfig()
    m = MinimalSSMConfig()
    for k in ("vocab_size", "d_model", "n_layers", "d_state"):
        assert getattr(s, k) == getattr(m, k)
    assert s.use_habituation is True
    assert s.use_association is True


def test_scale_profiles_all_valid() -> None:
    for name in PROFILES:
        cfg = profile(name)
        assert isinstance(cfg, TrainerConfig)
        assert cfg.model.parameter_count_estimate() > 0
        st = profile(name, stentor=True)
        assert st.model_kind == "stentor"


def test_cosine_schedule_warms_then_decays() -> None:
    cfg = TrainerConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, steps=100)
    assert cosine_lr(0, cfg=cfg) == 0.0
    assert cosine_lr(10, cfg=cfg) == pytest.approx(cfg.lr)
    mid = cosine_lr(55, cfg=cfg)
    end = cosine_lr(100, cfg=cfg)
    assert end == pytest.approx(cfg.min_lr)
    assert cfg.min_lr < mid < cfg.lr


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_minimal_ssm_forward_pass_runs() -> None:
    import torch

    from lsm.research.small_core.minimal_ssm import build_model

    cfg = MinimalSSMConfig(vocab_size=128, d_model=32, n_layers=2, d_state=8, d_inner=32)
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_stentor_ssm_forward_pass_runs() -> None:
    import torch

    from lsm.research.small_core.stentor_ssm import build_model

    cfg = StentorSSMConfig(
        vocab_size=128, d_model=32, n_layers=2, d_state=8, d_inner=32, assoc_rank=4
    )
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_stentor_ssm_ablations_toggle() -> None:
    import torch

    from lsm.research.small_core.stentor_ssm import build_model

    for use_hab, use_assoc in [(False, False), (True, False), (False, True), (True, True)]:
        cfg = StentorSSMConfig(
            vocab_size=64,
            d_model=16,
            n_layers=1,
            d_state=4,
            d_inner=16,
            assoc_rank=4,
            use_habituation=use_hab,
            use_association=use_assoc,
        )
        model = build_model(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model(x)
        assert out.shape == (1, 4, cfg.vocab_size)
