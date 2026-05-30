"""Phase 2 tiny smoke: synthetic tokens + short minimal/stentor train on GPU or CPU."""

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


def ensure_token_bin(path: Path, *, vocab_size: int, num_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= num_tokens * 2:
        log.info("reuse token file %s (%d bytes)", path, path.stat().st_size)
        return
    rng = np.random.default_rng(0)
    ids = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    ids.tofile(path)
    log.info("wrote %s (%d tokens)", path, num_tokens)


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
            "Windows PyPI torch is often CPU-only; run: .\\scripts\\install_research_torch_cuda.ps1",
            torch.__version__,
            ver,
        )
    return "cpu"


def run_smoke(*, stentor: bool, steps: int, device: str) -> dict[str, object]:
    from lsm.research.training.scale_configs import tiny
    from lsm.research.training.trainer import train

    cfg = tiny(stentor=stentor)
    data_path = Path("data/tokens_tiny.bin")
    ensure_token_bin(data_path, vocab_size=cfg.vocab_size, num_tokens=500_000)
    cfg = replace(
        cfg,
        data_path=data_path,
        ckpt_dir=Path("runs/tiny_smoke") / ("stentor" if stentor else "minimal"),
        steps=steps,
        log_every=max(1, steps // 10),
        save_every=max(1, steps),
        eval_every=steps + 1,
        device=device,
        precision="bf16" if device == "cuda" else "fp32",
        batch_size=4 if device == "cpu" else cfg.batch_size,
        seq_len=256 if device == "cpu" else cfg.seq_len,
    )
    log.info("training %s on %s for %d steps", cfg.model_kind, device, steps)
    metrics = train(cfg)
    metrics["model_kind"] = cfg.model_kind
    metrics["device"] = device
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128, help="Training steps per variant.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--minimal-only", action="store_true")
    parser.add_argument("--stentor-only", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/tiny_smoke/summary.json"),
    )
    args = parser.parse_args()
    device = pick_device(args.device)
    results: dict[str, Any] = {"device": device, "steps": args.steps, "variants": {}}
    variants: list[tuple[str, bool]] = []
    if not args.stentor_only:
        variants.append(("minimal", False))
    if not args.minimal_only:
        variants.append(("stentor", True))
    if not variants:
        parser.error("choose at least one of minimal or stentor")
    for name, stentor in variants:
        try:
            results["variants"][name] = run_smoke(stentor=stentor, steps=args.steps, device=device)
        except Exception as exc:
            log.exception("%s smoke failed", name)
            results["variants"][name] = {"error": str(exc)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    failed = [k for k, v in results["variants"].items() if isinstance(v, dict) and "error" in v]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
