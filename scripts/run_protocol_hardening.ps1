# Re-run headline recall_pref_v1 configs with seeds [0, 1, 2] (requires Ollama).
param(
    [switch]$KeepGoing,
    [switch]$DeferredOnly
)

$ErrorActionPreference = if ($KeepGoing) { 'Continue' } else { 'Stop' }

$headline = @(
    'experiments/sample_recall_pref_v1_raw_llm.yaml',
    'experiments/sample_recall_pref_v1_naive_rag.yaml',
    'experiments/sample_recall_pref_v1_lsm.yaml',
    'experiments/sample_recall_pref_v1_lsm_no_assoc.yaml'
)

$deferred = @(
    'experiments/sample_recall_pref_v1_lsm_no_habituation.yaml',
    'experiments/sample_recall_pref_v1_lsm_no_writer.yaml',
    'experiments/sample_recall_pref_v1_lsm_flat_memory.yaml',
    'experiments/sample_recall_pref_v1_lsm_multi_memory.yaml'
)

$configs = if ($DeferredOnly) { $deferred } else { $headline + $deferred }

foreach ($c in $configs) {
    Write-Host ">>> protocol seeds [0,1,2]: $c"
    try {
        $tmp = (uv run python scripts/patch_eval_seeds.py $c --seeds 0 1 2).Trim()
        if (-not $tmp) { throw "patch_eval_seeds produced no path" }
        uv run lsm eval run --config $tmp
    } catch {
        Write-Warning "failed: $c -> $_"
        if (-not $KeepGoing) { throw }
    }
}

Write-Host "protocol hardening complete."
