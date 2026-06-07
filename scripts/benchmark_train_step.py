"""Measure optimizer-step wall time for minimal vs stentor (sanity check after perf fixes)."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def bench(*, stentor: bool, steps: int, device: str) -> None:
    import numpy as np
    from lsm.research.training.scale_configs import small
    from lsm.research.training.trainer import train

    cfg = small(stentor=stentor)
    cfg = replace(
        cfg,
        steps=steps,
        device=device,
        compile_model=False,
        log_every=max(1, steps // 5),
        save_every=steps + 1,
        data_path=__import__("pathlib").Path("data/tokens_tiny.bin"),
    )
    path = cfg.data_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size < 500_000:
        rng = np.random.default_rng(0)
        rng.integers(0, cfg.vocab_size, size=250_000, dtype=np.uint16).tofile(path)

    t0 = time.perf_counter()
    metrics = train(cfg)
    elapsed = time.perf_counter() - t0
    per_step = elapsed / steps
    log.info(
        "%s: %.2fs total, %.2fs/step, final_loss=%s",
        "stentor" if stentor else "minimal",
        elapsed,
        per_step,
        metrics.get("final_loss"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--minimal-only", action="store_true")
    p.add_argument("--stentor-only", action="store_true")
    args = p.parse_args()
    if not args.stentor_only:
        bench(stentor=False, steps=args.steps, device=args.device)
    if not args.minimal_only:
        bench(stentor=True, steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
