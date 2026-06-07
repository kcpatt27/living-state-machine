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


def test_small_profile_minimal_branch() -> None:
    from lsm.research.small_core.minimal_ssm import MinimalSSMConfig
    from lsm.research.small_core.stentor_ssm import StentorSSMConfig
    from lsm.research.training.scale_configs import small

    minimal = small(stentor=False)
    assert minimal.model_kind == "minimal"
    assert isinstance(minimal.model, MinimalSSMConfig)
    stentor = small(stentor=True)
    assert stentor.model_kind == "stentor"
    assert isinstance(stentor.model, StentorSSMConfig)


def test_cosine_schedule_warms_then_decays() -> None:
    cfg = TrainerConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, steps=100)
    assert cosine_lr(0, cfg=cfg) == 0.0
    assert cosine_lr(10, cfg=cfg) == pytest.approx(cfg.lr)
    mid = cosine_lr(55, cfg=cfg)
    end = cosine_lr(100, cfg=cfg)
    assert end == pytest.approx(cfg.min_lr)
    assert cfg.min_lr < mid < cfg.lr


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_diagonal_ssm_scan_matches_python_loop() -> None:
    import torch
    from lsm.research.small_core.minimal_ssm import diagonal_ssm_scan

    torch.manual_seed(0)
    b, t_len, s = 2, 32, 8
    a = torch.sigmoid(torch.randn(s))
    b_seq = torch.randn(b, t_len, s)
    h_scan = diagonal_ssm_scan(a, b_seq)
    h_loop = torch.zeros(b, s)
    rows = []
    for t in range(t_len):
        h_loop = a * h_loop + b_seq[:, t]
        rows.append(h_loop)
    h_ref = torch.stack(rows, dim=1)
    assert torch.allclose(h_scan, h_ref, atol=1e-4, rtol=1e-4)


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


class _FakeTokenizer:
    """Minimal GPT-2-like tokenizer for CPU unit tests.

    ``encode`` maps characters to small ids (some intentionally above the
    model vocab so the out-of-vocab filter is exercised); ``decode`` is a
    cheap inverse good enough to produce non-empty text.
    """

    def encode(self, text: str) -> list[int]:
        ids = [(ord(c) % 200) + 1 for c in text[:24]]
        return ids or [1]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr((int(i) % 26) + 97) for i in ids)


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_hybrid_updater_topk_tokens_produces_state_text() -> None:
    import asyncio

    from lsm.core.embeddings import HashEmbedder
    from lsm.core.state import State
    from lsm.research.hybrid import HybridConfig, HybridStateUpdater
    from lsm.research.small_core.minimal_ssm import MinimalSSMConfig, build_model

    cfg = MinimalSSMConfig(vocab_size=128, d_model=32, n_layers=2, d_state=8, d_inner=32)
    model = build_model(cfg)
    embedder = HashEmbedder(dim=64)
    updater = HybridStateUpdater(
        model=model,
        tokenizer=_FakeTokenizer(),
        config=HybridConfig(mode="topk_tokens", summary_tokens=6, device="cpu"),
        embedder=embedder,
    )

    new_state = asyncio.run(
        updater.update(State(), "my favourite city is lisbon", "Got it, lisbon.")
    )
    assert isinstance(new_state, State)
    assert new_state.text  # non-empty decoded projection
    assert new_state.vector.shape == (64,)
    assert new_state.revision == 1
    assert new_state.notes["hybrid_mode"] == "topk_tokens"


@pytest.mark.skipif(not has_torch, reason="torch not installed")
def test_hybrid_updater_nearest_neighbours_returns_memory_lines() -> None:
    import asyncio

    from lsm.core.embeddings import HashEmbedder
    from lsm.core.memory import InMemoryStore
    from lsm.core.state import State
    from lsm.research.hybrid import HybridConfig, HybridStateUpdater
    from lsm.research.small_core.minimal_ssm import MinimalSSMConfig, build_model

    cfg = MinimalSSMConfig(vocab_size=128, d_model=32, n_layers=2, d_state=8, d_inner=32)
    model = build_model(cfg)
    embedder = HashEmbedder(dim=64)
    memory = InMemoryStore(embedder)
    memory.put("user lives in lisbon", source="test")
    memory.put("user enjoys matcha", source="test")

    updater = HybridStateUpdater(
        model=model,
        tokenizer=_FakeTokenizer(),
        config=HybridConfig(mode="nearest_neighbours", summary_tokens=6, device="cpu"),
        memory=memory,
        embedder=embedder,
    )

    new_state = asyncio.run(updater.update(State(), "where do i live?", "You said lisbon."))
    assert new_state.notes["hybrid_mode"] == "nearest_neighbours"
    # Projection should be drawn from durable memory entries.
    assert "- " in new_state.text
    assert "lisbon" in new_state.text or "matcha" in new_state.text
