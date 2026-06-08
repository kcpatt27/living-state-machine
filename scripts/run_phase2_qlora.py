"""Phase 2 L15 — QLoRA train runner (local or RunPod).

Smoke (50 steps, 32-row slice)::

    uv run python scripts/run_phase2_qlora_smoke.py

Full train (default 2000 steps)::

    uv run python scripts/run_phase2_qlora.py --dataset data/sft_v0.jsonl

See ``docs/cloud_burst.md`` for RunPod setup.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "sft_v0.jsonl"
BOOTSTRAP_DATASET = REPO_ROOT / "data" / "sft_bootstrap.jsonl"


def _require_research_deps() -> None:
    try:
        import bitsandbytes  # noqa: F401
        import peft  # noqa: F401
        import torch
        import transformers  # noqa: F401
    except ImportError as exc:
        log.error(
            "research extras missing (%s). Run: uv sync --extra research",
            exc,
        )
        sys.exit(1)
    if not torch.cuda.is_available():
        log.error(
            "CUDA required for QLoRA. On RunPod use a GPU pod; "
            "locally run install_research_torch_cuda.ps1 first."
        )
        sys.exit(1)


def _resolve_dataset(path: Path, *, bootstrap: bool, bootstrap_count: int) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    if bootstrap:
        from scripts.bootstrap_sft_recall_pref import build_bootstrap_rows

        log.warning("dataset %s missing — writing bootstrap rows", path)
        rows = build_bootstrap_rows(count=bootstrap_count, seed=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
    log.error(
        "dataset not found: %s. Export with export_sft_from_lsm_home.py or pass --bootstrap",
        path,
    )
    sys.exit(1)


def _gpu_info() -> dict[str, str]:
    import torch

    return {
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": str(getattr(torch.version, "cuda", "")),
    }


def run_qlora(
    *,
    smoke: bool,
    dataset: Path,
    output_dir: Path | None,
    steps: int | None,
    max_rows: int | None,
    grad_accum: int | None,
    batch_size: int | None,
    bootstrap: bool,
    bootstrap_count: int,
    seed: int,
    l40s: bool = False,
) -> dict[str, Any]:
    _require_research_deps()
    from lsm.research.training.qlora import QLoRAConfig, train_qlora

    dataset = _resolve_dataset(dataset, bootstrap=bootstrap, bootstrap_count=bootstrap_count)
    if smoke:
        steps = steps or 50
        max_rows = max_rows if max_rows is not None else 32
        output_dir = output_dir or REPO_ROOT / "runs" / "qlora_smoke"
        grad_accum = grad_accum or 4
        batch_size = batch_size or 1
    else:
        steps = steps or 2_000
        output_dir = output_dir or REPO_ROOT / "runs" / "qlora_qwen_7b"
        grad_accum = grad_accum or 16
        batch_size = batch_size or 1

    cfg = QLoRAConfig(
        dataset_path=dataset,
        steps=steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        output_dir=output_dir,
        seed=seed,
        max_rows=max_rows,
        save_every=min(500, steps) if not smoke else steps,
    )
    if smoke:
        cfg = replace(cfg, seq_len=512)
    elif l40s:
        cfg = replace(cfg, seq_len=1024)

    log.info(
        "QLoRA %s: steps=%d batch=%d grad_accum=%d max_rows=%s dataset=%s -> %s",
        "smoke" if smoke else "train",
        cfg.steps,
        cfg.batch_size,
        cfg.grad_accum,
        max_rows,
        dataset,
        output_dir,
    )
    metrics = train_qlora(cfg)
    metrics.update(_gpu_info())
    metrics["smoke"] = smoke
    metrics["dataset_path"] = str(dataset)
    metrics["max_rows"] = max_rows
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="50-step smoke on 32-row slice.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--l40s",
        action="store_true",
        help="L40S preset: batch=4, grad_accum=4, seq_len=1024 (full train only).",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="If dataset missing, write recall_pref bootstrap JSONL first.",
    )
    parser.add_argument("--bootstrap-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/metrics.json",
    )
    args = parser.parse_args(argv)

    batch_size = args.batch_size
    grad_accum = args.grad_accum
    steps = args.steps
    if args.l40s and not args.smoke:
        batch_size = batch_size or 4
        grad_accum = grad_accum or 4
        steps = steps or 2_000

    metrics = run_qlora(
        smoke=args.smoke,
        dataset=args.dataset,
        output_dir=args.output_dir,
        steps=steps,
        max_rows=args.max_rows,
        grad_accum=grad_accum,
        batch_size=batch_size,
        bootstrap=args.bootstrap,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
        l40s=args.l40s,
    )
    out_dir = Path(metrics["output_dir"])
    metrics_path = args.metrics_out or out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
