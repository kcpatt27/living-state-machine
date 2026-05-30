# Shared recall_pref_v2 task settings (harder than v1).
# num_items=15, distractor_turns=45, decoys + repeated filler, seeds [0,1,2].
# Production path: FactsBackedUpdater (state_updater: facts_backed).

$ErrorActionPreference = 'Continue'

$configs = @(
    'experiments/sample_recall_pref_v2_raw_llm.yaml',
    'experiments/sample_recall_pref_v2_naive_rag.yaml',
    'experiments/sample_recall_pref_v2_lsm.yaml',
    'experiments/sample_recall_pref_v2_lsm_no_assoc.yaml',
    'experiments/sample_recall_pref_v2_lsm_no_habituation.yaml',
    'experiments/sample_recall_pref_v2_lsm_no_writer.yaml',
    'experiments/sample_recall_pref_v2_lsm_flat_memory.yaml',
    'experiments/sample_recall_pref_v2_lsm_multi_memory.yaml',
    'experiments/sample_recall_pref_v2_lsm_hybrid_verifier.yaml'
)

foreach ($c in $configs) {
    Write-Host ">>> uv run lsm eval run --config $c"
    try {
        uv run lsm eval run --config $c
    } catch {
        Write-Warning "failed: $c -> $_"
    }
}

Write-Host "recall_pref_v2 suite complete. Update docs/phase1_report.md section 5.2"
