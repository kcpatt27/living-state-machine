#!/usr/bin/env bash
# RunPod one-shot setup for L15 QLoRA. Run from repo root after clone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME" data runs

echo "Syncing research extras (CUDA torch)..."
uv sync --extra research

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "Setup done. L40S burst:"
echo "  bash scripts/runpod_l40s_burst.sh"
echo "Or manual:"
echo "  uv run python scripts/runpod_l40s_burst.py --bootstrap"
