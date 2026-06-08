#!/usr/bin/env bash
# Full L40S burst: setup + train. Run from repo root on a RunPod L40S pod.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
bash scripts/runpod_l15_setup.sh

echo "Starting L40S burst (smoke + QLoRA + SSM)..."
uv run python scripts/runpod_l40s_burst.py --bootstrap "$@"

echo "Done. Download runs/runpod_l40s_burst/ before terminating the pod."
