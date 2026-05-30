# Phase 2 small train (minimal + stentor). Requires CUDA torch — see install_research_torch_cuda.ps1.
#
# Real corpus:
#   uv run python scripts/run_phase2_small_train.py --data-path data/tokens_small.bin
#
# GPU pipeline test (random tokens, shorter):
#   uv run python scripts/run_phase2_small_train.py --synthetic --steps 500

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..

if ($args.Count -eq 0) {
    Write-Host "No args — running --synthetic --steps 500 (dry-run). Pass args for full train."
    uv run python scripts/run_phase2_small_train.py --synthetic --steps 500
} else {
    uv run python scripts/run_phase2_small_train.py @args
}
