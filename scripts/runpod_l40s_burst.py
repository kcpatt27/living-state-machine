"""RunPod L40S burst — smoke + QLoRA + minimal SSM in one session.

Tuned for **L40S 48GB** (~$0.86/hr): QLoRA ``batch=4``, ``grad_accum=4``;
minimal SSM ``batch=16``, ``grad_accum=1``.

Usage on pod (after ``bash scripts/runpod_l15_setup.sh``)::

    export HF_HOME=/workspace/hf_cache
    uv run python scripts/runpod_l40s_burst.py --bootstrap

Skip SSM or shorten for a faster bill::

    uv run python scripts/runpod_l40s_burst.py --bootstrap --skip-ssm
    uv run python scripts/runpod_l40s_burst.py --bootstrap --ssm-steps 5000

Three QLoRA seeds (pick best adapter locally by eval later)::

    uv run python scripts/runpod_l40s_burst.py --bootstrap --qlora-seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BURST_ROOT = REPO_ROOT / "runs" / "runpod_l40s_burst"
DEFAULT_DATASET = REPO_ROOT / "data" / "sft_v0.jsonl"
DEFAULT_TOKENS = REPO_ROOT / "data" / "tokens_small.bin"

# L40S QLoRA: effective batch 16 (same as 5060 Ti default), fewer micro-steps.
L40S_QLORA_BATCH = 4
L40S_QLORA_GRAD_ACCUM = 4
L40S_QLORA_STEPS = 2_000
L40S_QLORA_SEQ = 1024

# L40S SSM: more micro-batch, less grad accumulation.
L40S_SSM_BATCH = 16
L40S_SSM_GRAD_ACCUM = 1
L40S_SSM_STEPS = 10_000


def _gpu_info() -> dict[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": str(getattr(torch.version, "cuda", "")),
            }
    except ImportError:
        pass
    return {}


def _run_prepare_tokens() -> dict[str, Any]:
    if DEFAULT_TOKENS.exists() and DEFAULT_TOKENS.stat().st_size > 0:
        log.info("reuse token file %s", DEFAULT_TOKENS)
        return {"skipped": True, "path": str(DEFAULT_TOKENS)}
    log.info("preparing TinyStories tokens (first run may take 15-30 min)...")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "prepare_tokens_small.py")],
        cwd=REPO_ROOT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("prepare_tokens_small.py failed")
    return {"skipped": False, "path": str(DEFAULT_TOKENS)}


def _run_ssm(*, steps: int, run_root: Path) -> dict[str, Any]:
    from dataclasses import replace

    from lsm.research.training.scale_configs import small
    from lsm.research.training.trainer import train

    if not DEFAULT_TOKENS.is_file():
        raise FileNotFoundError(f"token file missing: {DEFAULT_TOKENS}")

    cfg = small(stentor=False)
    cfg = replace(
        cfg,
        data_path=DEFAULT_TOKENS,
        ckpt_dir=run_root / "minimal" / "ckpts",
        steps=steps,
        batch_size=L40S_SSM_BATCH,
        grad_accum=L40S_SSM_GRAD_ACCUM,
        seq_len=512,
        log_every=max(20, steps // 100),
        save_every=max(500, steps // 20),
        eval_every=steps + 1,
        device="cuda",
        precision="bf16",
        wandb_run_name="runpod_l40s_minimal",
    )
    log.info(
        "SSM minimal: steps=%d batch=%d grad_accum=%d -> %s",
        steps,
        L40S_SSM_BATCH,
        L40S_SSM_GRAD_ACCUM,
        cfg.ckpt_dir,
    )
    metrics = train(cfg)
    metrics["model_kind"] = "minimal"
    metrics["steps"] = steps
    metrics["batch_size"] = L40S_SSM_BATCH
    metrics["grad_accum"] = L40S_SSM_GRAD_ACCUM
    return metrics


def _run_qlora_l40s(
    *,
    smoke: bool,
    dataset: Path,
    output_dir: Path,
    steps: int,
    max_rows: int | None,
    bootstrap: bool,
    bootstrap_count: int,
    seed: int,
) -> dict[str, Any]:
    from lsm.research.training.qlora import QLoRAConfig, train_qlora
    from scripts.run_phase2_qlora import _gpu_info, _require_research_deps, _resolve_dataset

    _require_research_deps()
    dataset = _resolve_dataset(dataset, bootstrap=bootstrap, bootstrap_count=bootstrap_count)
    grad_accum = 4 if smoke else L40S_QLORA_GRAD_ACCUM
    seq_len = 512 if smoke else L40S_QLORA_SEQ

    cfg = QLoRAConfig(
        dataset_path=dataset,
        steps=steps,
        batch_size=L40S_QLORA_BATCH if not smoke else 1,
        grad_accum=grad_accum,
        seq_len=seq_len,
        output_dir=output_dir,
        seed=seed,
        max_rows=max_rows if smoke else None,
        save_every=min(500, steps) if not smoke else steps,
    )
    log.info(
        "QLoRA L40S smoke=%s steps=%d batch=%d grad_accum=%d -> %s",
        smoke,
        cfg.steps,
        cfg.batch_size,
        cfg.grad_accum,
        output_dir,
    )
    metrics = train_qlora(cfg)
    metrics.update(_gpu_info())
    metrics["smoke"] = smoke
    metrics["dataset_path"] = str(dataset)
    metrics["max_rows"] = max_rows
    metrics["l40s"] = True
    out = output_dir / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_burst(
    *,
    dataset: Path,
    bootstrap: bool,
    bootstrap_count: int,
    qlora_steps: int,
    qlora_seeds: list[int],
    skip_smoke: bool,
    skip_ssm: bool,
    ssm_steps: int,
) -> dict[str, Any]:
    t0 = time.time()
    summary: dict[str, Any] = {
        "profile": "runpod_l40s_burst",
        "l40s_qlora": {
            "batch_size": L40S_QLORA_BATCH,
            "grad_accum": L40S_QLORA_GRAD_ACCUM,
            "seq_len": L40S_QLORA_SEQ,
            "steps": qlora_steps,
            "seeds": qlora_seeds,
        },
        "l40s_ssm": {
            "batch_size": L40S_SSM_BATCH,
            "grad_accum": L40S_SSM_GRAD_ACCUM,
            "steps": ssm_steps,
            "minimal_only": True,
        },
        "phases": {},
    }
    summary.update(_gpu_info())

    if not skip_smoke:
        log.info("=== phase 1: QLoRA smoke ===")
        summary["phases"]["qlora_smoke"] = _run_qlora_l40s(
            smoke=True,
            dataset=dataset,
            output_dir=BURST_ROOT / "qlora_smoke",
            steps=50,
            max_rows=32,
            bootstrap=bootstrap,
            bootstrap_count=bootstrap_count,
            seed=0,
        )

    qlora_runs: dict[str, Any] = {}
    for seed in qlora_seeds:
        log.info("=== QLoRA train seed=%d ===", seed)
        out = BURST_ROOT / "qlora_qwen_7b" / f"seed_{seed}"
        qlora_runs[str(seed)] = _run_qlora_l40s(
            smoke=False,
            dataset=dataset,
            output_dir=out,
            steps=qlora_steps,
            max_rows=None,
            bootstrap=bootstrap,
            bootstrap_count=bootstrap_count,
            seed=seed,
        )
    summary["phases"]["qlora_train"] = qlora_runs

    if not skip_ssm:
        log.info("=== token prep ===")
        summary["phases"]["tokens"] = _run_prepare_tokens()
        log.info("=== SSM minimal (%d steps) ===", ssm_steps)
        summary["phases"]["ssm_minimal"] = _run_ssm(
            steps=ssm_steps,
            run_root=BURST_ROOT / "ssm",
        )

    summary["elapsed_sec"] = int(time.time() - t0)
    summary["estimated_cost_usd_086_hr"] = round(summary["elapsed_sec"] / 3600 * 0.86, 2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-count", type=int, default=500)
    parser.add_argument("--qlora-steps", type=int, default=L40S_QLORA_STEPS)
    parser.add_argument("--qlora-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--ssm-steps", type=int, default=L40S_SSM_STEPS)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-ssm", action="store_true")
    args = parser.parse_args()

    BURST_ROOT.mkdir(parents=True, exist_ok=True)
    summary = run_burst(
        dataset=args.dataset,
        bootstrap=args.bootstrap,
        bootstrap_count=args.bootstrap_count,
        qlora_steps=args.qlora_steps,
        qlora_seeds=args.qlora_seeds,
        skip_smoke=args.skip_smoke,
        skip_ssm=args.skip_ssm,
        ssm_steps=args.ssm_steps,
    )
    summary_path = BURST_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "burst complete in %ds (~$%.2f @ $0.86/hr)",
        summary["elapsed_sec"],
        summary["estimated_cost_usd_086_hr"],
    )
    log.info("summary -> %s", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
