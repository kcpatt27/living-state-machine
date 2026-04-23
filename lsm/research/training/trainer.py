"""Trainer for the custom recurrent core — L11 (scale to L13).

Minimal, readable training loop over a tokenised corpus on a single
GPU. Designed to stay on the 5060 Ti (16 GB VRAM) at
``d_model=512, n_layers=12`` with ``seq_len=1024, batch=4,
grad_accum=4, mixed=bf16``.

Design points
-------------

- All hyperparameters live in :class:`TrainerConfig` (serialisable via
  ``asdict`` so every run can archive its config).
- Logs go to ``stdout`` and an optional ``wandb`` run (opt-in via
  ``cfg.wandb_project``).
- Checkpoints roll over ``keep_last_n`` files.
- Supports both the plain :mod:`.small_core.minimal_ssm` core and the
  :mod:`.small_core.stentor_ssm` augmented core; pick via
  ``cfg.model_kind``.

The dataset interface is intentionally tiny: give it an iterable of
``torch.LongTensor`` chunks of shape ``(seq_len + 1,)`` (input+target
pair). A helper :func:`packed_bin_iterator` reads from a flat binary
file of token ids (the standard "tokens.bin" format).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lsm.research.small_core.minimal_ssm import (
    MinimalSSMConfig,
    _require_torch,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainerConfig:
    model_kind: str = "minimal"  # "minimal" | "stentor"
    model: MinimalSSMConfig = field(default_factory=MinimalSSMConfig)

    data_path: Path = Path("data/tokens.bin")
    vocab_size: int = 32_000
    seq_len: int = 1024
    batch_size: int = 4
    grad_accum: int = 4

    steps: int = 10_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: str = "bf16"  # "fp32" | "bf16" | "fp16"

    device: str = "cuda"
    seed: int = 0

    ckpt_dir: Path = Path("runs/ssm_default/ckpts")
    log_every: int = 20
    eval_every: int = 500
    save_every: int = 1_000
    keep_last_n: int = 3

    wandb_project: str | None = None
    wandb_run_name: str | None = None


def cosine_lr(step: int, *, cfg: TrainerConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


def packed_bin_iterator(
    path: Path, *, seq_len: int, batch_size: int, device: str, seed: int
) -> Iterator[Any]:
    """Yield batches of shape ``(batch_size, seq_len + 1)`` from a uint16 bin."""
    torch = _require_torch()
    import numpy as np

    arr = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    max_start = n - (seq_len + 1)
    if max_start <= 0:
        raise RuntimeError(f"token file too small: {n} tokens, need > {seq_len + 1}")
    while True:
        starts = rng.integers(0, max_start, size=batch_size)
        batch = [arr[s : s + seq_len + 1].astype(np.int64) for s in starts]
        x = torch.tensor(np.stack(batch), device=device)
        yield x


def build_for_cfg(cfg: TrainerConfig) -> Any:
    if cfg.model_kind == "minimal":
        from lsm.research.small_core.minimal_ssm import build_model

        return build_model(cfg.model)
    if cfg.model_kind == "stentor":
        from lsm.research.small_core.stentor_ssm import (
            StentorSSMConfig,
            build_model,
        )

        if not isinstance(cfg.model, StentorSSMConfig):
            sc = StentorSSMConfig(**asdict(cfg.model))
        else:
            sc = cfg.model
        return build_model(sc)
    raise ValueError(f"unknown model_kind: {cfg.model_kind!r}")


def train(cfg: TrainerConfig, data_iter: Iterable[Any] | None = None) -> dict[str, Any]:
    """Run training to completion; return final metrics + ckpt paths."""
    torch = _require_torch()

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = build_for_cfg(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("model ready; params=%.2fM", param_count / 1e6)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.precision]

    if data_iter is None:
        data_iter = packed_bin_iterator(
            cfg.data_path,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            device=str(device),
            seed=cfg.seed,
        )

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    (cfg.ckpt_dir.parent / "trainer_config.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2),
        encoding="utf-8",
    )

    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={k: str(v) for k, v in asdict(cfg).items()},
            )
        except Exception:  # noqa: BLE001
            log.exception("wandb init failed, continuing without it")
            wandb_run = None

    ckpts: list[Path] = []
    t0 = time.time()
    running_loss = 0.0
    running_steps = 0

    data_iter_exhausted = iter(data_iter)

    for step in range(1, cfg.steps + 1):
        lr_now = cosine_lr(step, cfg=cfg)
        for g in opt.param_groups:
            g["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            batch = next(data_iter_exhausted)
            x = batch[:, :-1].long()
            y = batch[:, 1:].long()
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
                loss = loss / cfg.grad_accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        running_loss += float(loss.detach().item()) * cfg.grad_accum
        running_steps += 1

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            avg = running_loss / max(running_steps, 1)
            log.info(
                "step=%d loss=%.4f lr=%.5f elapsed=%ds",
                step,
                avg,
                lr_now,
                int(elapsed),
            )
            if wandb_run is not None:
                wandb_run.log({"train/loss": avg, "train/lr": lr_now, "step": step})
            running_loss = 0.0
            running_steps = 0

        if step % cfg.save_every == 0 or step == cfg.steps:
            ckpt_path = cfg.ckpt_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
            log.info("saved checkpoint: %s", ckpt_path)
            ckpts.append(ckpt_path)
            while len(ckpts) > cfg.keep_last_n:
                old = ckpts.pop(0)
                try:
                    old.unlink()
                except OSError:
                    pass

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "steps": cfg.steps,
        "param_count": param_count,
        "ckpts": [str(p) for p in ckpts],
        "final_loss": running_loss / max(running_steps, 1) if running_steps else None,
    }


__all__ = [
    "TrainerConfig",
    "train",
    "packed_bin_iterator",
    "cosine_lr",
    "build_for_cfg",
]
