"""L15 QLoRA smoke — 50 steps on a 32-row slice → ``runs/qlora_smoke/metrics.json``."""

from __future__ import annotations

from scripts.run_phase2_qlora import main

if __name__ == "__main__":
    main(["--smoke", "--bootstrap"])
