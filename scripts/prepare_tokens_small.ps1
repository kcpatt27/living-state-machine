# Download TinyStories + write data/tokens_small.bin (uint16, vocab 32k clip).
# If tokens_small.bin is locked by a training run, output goes to
# data/tokens_small_tinystories.bin automatically.
#
# Full ~80M tokens (~160 MB):
#   uv run python scripts/prepare_tokens_small.py
#
# Faster subset (~25M tokens):
#   uv run python scripts/prepare_tokens_small.py --max-tokens 25000000
#
# Full train (use tinystories file if default path is synthetic/locked):
#   uv run python scripts/run_phase2_small_train.py --data-path data/tokens_small_tinystories.bin

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..

uv run python scripts/prepare_tokens_small.py @args
