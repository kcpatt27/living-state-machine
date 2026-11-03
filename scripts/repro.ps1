# One-command reproduction check.
#
#   .\scripts\repro.ps1          # verify Tables 1-2 from committed artifacts
#   .\scripts\repro.ps1 -Rerun   # print the commands that re-run the evals
#
# The default path needs no model, GPU, or network. The -Rerun path needs
# Ollama with qwen2.5:7b-instruct-q8_0 and a GPU; it is not automated here
# because it takes hours and is hardware-specific.

param([switch]$Rerun)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if ($Rerun) {
    Write-Host "Full re-run (requires Ollama + qwen2.5:7b-instruct-q8_0):"
    Write-Host "  scripts/run_phase1_suite.ps1"
    Write-Host "  scripts/run_recall_pref_v2_suite.ps1"
    Write-Host "  scripts/run_deferred_ablations.ps1"
    Write-Host ""
    Write-Host "Then re-run this script without -Rerun to check the tables."
    exit 0
}

python scripts/verify_tables.py
exit $LASTEXITCODE
