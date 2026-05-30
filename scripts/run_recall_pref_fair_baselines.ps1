# Matched-context raw_llm baselines (run.max_context_turns = 4, ~LSM top_k budget).
# Full-dialogue arms: sample_recall_pref_*_raw_llm.yaml (max_context_turns: 0).

$ErrorActionPreference = 'Continue'

$configs = @(
    'experiments/sample_recall_pref_v1_raw_llm_matched.yaml',
    'experiments/sample_recall_pref_v2_raw_llm_matched.yaml'
)

foreach ($c in $configs) {
    Write-Host ">>> uv run lsm eval run --config $c"
    try {
        uv run lsm eval run --config $c
    } catch {
        Write-Warning "failed: $c -> $_"
    }
}

Write-Host "Fair baselines complete. Fill docs/phase1_report.md and recall_pref_v2_findings.md."
