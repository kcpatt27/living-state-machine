# L15 regression: Transformers 7B base vs QLoRA adapter on recall_pref_v1.
# Requires: uv sync --extra research, CUDA torch, adapter weights on disk.
# Optional hybrid smoke: pass -Hybrid

param(
    [switch]$Hybrid,
    [switch]$SkipQlora
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot\..

$configs = @(
    "experiments/sample_recall_pref_v1_lsm_transformers_base.yaml"
)
if (-not $SkipQlora) {
    $configs += "experiments/sample_recall_pref_v1_lsm_transformers_qlora.yaml"
}
if ($Hybrid) {
    $configs += "experiments/sample_recall_pref_v1_lsm_hybrid_updater.yaml"
}

foreach ($cfg in $configs) {
    Write-Host "`n=== $cfg ===" -ForegroundColor Cyan
    uv run lsm eval run --config $cfg
}

Write-Host "`nDone. Compare accuracy_mean in experiments/*-$(Split-Path $cfg -LeafBase)/metrics.json" -ForegroundColor Green
Pop-Location
