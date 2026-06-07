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

import contextlib
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
    seq_len: int = 512
    batch_size: int = 8
    grad_accum: int = 2

    steps: int = 2_500
    compile_model: bool = False
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


def _setup_cuda_runtime(device: Any) -> None:
    """TF32 + cuDNN autotune for matmul-heavy paths (scan still benefits)."""
    if device.type != "cuda":
        return
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


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
    win = seq_len + 1
    use_cuda = device.startswith("cuda")
    pin = use_cuda
    while True:
        starts = rng.integers(0, max_start, size=batch_size)
        batch_np = np.empty((batch_size, win), dtype=np.int64)
        for i, s in enumerate(starts):
            batch_np[i] = np.asarray(arr[s : s + win], dtype=np.int64)
        batch = torch.from_numpy(batch_np)
        if pin:
            batch = batch.pin_memory()
        yield batch.to(device, non_blocking=pin)


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
    _setup_cuda_runtime(device)
    model = build_for_cfg(cfg).to(device)
    # Stentor still has a sequential association loop; compile often slows first runs on Windows.
    if cfg.compile_model and device.type == "cuda" and cfg.model_kind == "minimal":
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            log.info("torch.compile enabled (minimal)")
        except Exception:
            log.warning("torch.compile unavailable, using eager mode", exc_info=True)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("model ready; params=%.2fM", param_count / 1e6)
    tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum

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
        except Exception:
            log.exception("wandb init failed, continuing without it")
            wandb_run = None

    ckpts: list[Path] = []
    t0 = time.time()
    step_t0 = time.time()
    running_loss = 0.0
    running_steps = 0
    last_avg_loss: float | None = None

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
            step_dt = max(time.time() - step_t0, 1e-6)
            step_t0 = time.time()
            avg = running_loss / max(running_steps, 1)
            last_avg_loss = avg
            tok_s = tokens_per_step / step_dt
            log.info(
                "step=%d loss=%.4f lr=%.5f elapsed=%ds step_time=%.2fs tok/s=%.0f",
                step,
                avg,
                lr_now,
                int(elapsed),
                step_dt,
                tok_s,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": avg,
                        "train/lr": lr_now,
                        "train/tok_per_sec": tok_s,
                        "step": step,
                    }
                )
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
                with contextlib.suppress(OSError):
                    old.unlink()

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "steps": cfg.steps,
        "param_count": param_count,
        "ckpts": [str(p) for p in ckpts],
        "final_loss": last_avg_loss,
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
    }


__all__ = [
    "TrainerConfig",
    "train",
    "packed_bin_iterator",
    "cosine_lr",
    "build_for_cfg",
]
