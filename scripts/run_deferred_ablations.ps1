# Run the four deferred recall_pref_v1 LSM ablations (requires Ollama + qwen2.5:7b).
param(
    [switch]$KeepGoing
)

$ErrorActionPreference = if ($KeepGoing) { 'Continue' } else { 'Stop' }

$configs = @(
    'experiments/sample_recall_pref_v1_lsm_no_habituation.yaml',
    'experiments/sample_recall_pref_v1_lsm_no_writer.yaml',
    'experiments/sample_recall_pref_v1_lsm_flat_memory.yaml',
    'experiments/sample_recall_pref_v1_lsm_multi_memory.yaml'
)

foreach ($c in $configs) {
    Write-Host ">>> uv run lsm eval run --config $c"
    try {
        uv run lsm eval run --config $c
    } catch {
        Write-Warning "failed: $c -> $_"
        if (-not $KeepGoing) { throw }
    }
}

Write-Host "deferred ablations complete. Fill docs/phase1_report.md section 5.1 from experiments/*/metrics.json"
