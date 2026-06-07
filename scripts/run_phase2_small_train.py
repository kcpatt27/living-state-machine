"""Phase 2 ``small`` train: minimal vs Stentor SSM on GPU (or CPU fallback).

Default profile: ``scale_configs.small()`` (~30M params, **2500** steps, seq_len **512**).
Runs **minimal** then **stentor** unless ``--minimal-only`` / ``--stentor-only``.

**Windows:** disable sleep/hibernate during training; screen-off is OK if the machine
stays awake. Old runs were ~60s/step due to a Python loop over every token (fixed).

Token data
----------
- **Real run:** prepare once, then train::

      uv run python scripts/prepare_tokens_small.py
      uv run python scripts/run_phase2_small_train.py --data-path data/tokens_small_tinystories.bin

  Source: TinyStories (HF) + GPT-2 tokenizer → ``data/tokens_small.bin``.
  See ``scripts/prepare_tokens_small.py``.
- **Pipeline test:** ``--synthetic`` writes random ids (loss not meaningful).

Prereqs::

    uv sync --extra research
    .\\scripts\\install_research_torch_cuda.ps1   # Windows CUDA torch

Examples::

    # Full small train (both variants, real data you prepared):
    python scripts/run_phase2_small_train.py --data-path data/tokens_small_tinystories.bin

    # Shorter dry-run on GPU:
    python scripts/run_phase2_small_train.py --synthetic --steps 500

Outputs::

    runs/small/<minimal|stentor>/ckpts/step_*.pt   (gitignored)
    runs/small/summary.json
    runs/small/<variant>/trainer_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "tokens_small.bin"
DEFAULT_OUT = REPO_ROOT / "runs" / "small" / "summary.json"
# ~80M tokens — enough random windows for 20k steps at batch=4, seq=1024.
SYNTHETIC_NUM_TOKENS = 80_000_000


def ensure_token_bin(path: Path, *, vocab_size: int, num_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    need_bytes = num_tokens * 2
    if path.exists() and path.stat().st_size >= need_bytes:
        log.info("reuse token file %s (%d bytes)", path, path.stat().st_size)
        return
    log.info("writing synthetic tokens %s (%d tokens)", path, num_tokens)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    ids.tofile(path)


def pick_device(requested: str) -> str:
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            log.error("torch not installed; run: uv sync --extra research")
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        log.info("using GPU: %s (capability %s)", name, cap)
        return "cuda"
    if requested == "cuda":
        ver = getattr(torch.version, "cuda", None)
        log.warning(
            "cuda requested but unavailable (torch=%s, cuda_build=%s). "
            "Windows PyPI torch is often CPU-only; run: "
            ".\\scripts\\install_research_torch_cuda.ps1",
            torch.__version__,
            ver,
        )
    return "cpu"


def run_variant(
    *,
    stentor: bool,
    steps: int | None,
    seq_len: int | None,
    device: str,
    data_path: Path,
    synthetic: bool,
    run_root: Path,
) -> dict[str, object]:
    from lsm.research.training.scale_configs import small
    from lsm.research.training.trainer import train

    cfg = small(stentor=stentor)
    variant = "stentor" if stentor else "minimal"
    if synthetic:
        ensure_token_bin(data_path, vocab_size=cfg.vocab_size, num_tokens=SYNTHETIC_NUM_TOKENS)
    elif not data_path.is_file():
        raise FileNotFoundError(
            f"token file not found: {data_path}\n"
            "Prepare uint16 token ids or pass --synthetic for a pipeline test."
        )

    n_steps = steps if steps is not None else cfg.steps
    if seq_len is not None:
        cfg = replace(cfg, seq_len=seq_len)
    ckpt_dir = run_root / variant / "ckpts"
    cfg = replace(
        cfg,
        data_path=data_path,
        ckpt_dir=ckpt_dir,
        steps=n_steps,
        log_every=max(20, n_steps // 100),
        save_every=max(500, n_steps // 20),
        eval_every=n_steps + 1,
        device=device,
        precision="bf16" if device == "cuda" else "fp32",
        batch_size=2 if device == "cpu" else cfg.batch_size,
        grad_accum=8 if device == "cpu" else cfg.grad_accum,
        seq_len=512 if device == "cpu" else cfg.seq_len,
        wandb_run_name=f"small_{variant}",
    )
    log.info(
        "training %s on %s for %d steps (~%.1fM params)",
        cfg.model_kind,
        device,
        n_steps,
        cfg.model.parameter_count_estimate() / 1e6,
    )
    metrics = train(cfg)
    metrics["model_kind"] = cfg.model_kind
    metrics["device"] = device
    metrics["steps"] = n_steps
    metrics["data_path"] = str(data_path)
    metrics["synthetic"] = synthetic
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA,
        help="uint16 token .bin (default: data/tokens_small.bin)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override step count (default: scale_configs.small = 2500)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override context length (default: 512)",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate random tokens at --data-path (pipeline test only)",
    )
    parser.add_argument("--minimal-only", action="store_true")
    parser.add_argument("--stentor-only", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next variant if one variant fails",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Aggregate metrics JSON",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "runs" / "small",
        help="Checkpoints under <run-root>/<minimal|stentor>/ckpts/",
    )
    args = parser.parse_args()
    device = pick_device(args.device)
    data_path = args.data_path if args.data_path.is_absolute() else REPO_ROOT / args.data_path

    variants: list[tuple[str, bool]] = []
    if not args.stentor_only:
        variants.append(("minimal", False))
    if not args.minimal_only:
        variants.append(("stentor", True))
    if not variants:
        parser.error("choose at least one of minimal or stentor")

    results: dict[str, Any] = {
        "profile": "small",
        "device": device,
        "steps": args.steps,
        "synthetic": args.synthetic,
        "data_path": str(data_path),
        "variants": {},
    }
    for name, stentor in variants:
        try:
            results["variants"][name] = run_variant(
                stentor=stentor,
                steps=args.steps,
                seq_len=args.seq_len,
                device=device,
                data_path=data_path,
                synthetic=args.synthetic,
                run_root=args.run_root,
            )
        except Exception as exc:
            log.exception("%s train failed", name)
            results["variants"][name] = {"error": str(exc)}
            if not args.keep_going:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    failed = [k for k, v in results["variants"].items() if isinstance(v, dict) and "error" in v]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
