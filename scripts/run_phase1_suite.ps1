# Phase 1 full experiment suite.
#
# Runs every headline experiment (see docs/phase1_report.md) plus the
# retrieval-layer benchmarks that do not need an LLM call. Intended to
# be run after ``scripts/setup.ps1`` + ``uv sync``. Every experiment
# writes into ``experiments/<run_id>/`` with metrics.json + raw_outputs.

param(
    [switch]$SkipEndToEnd,  # skip LLM-driven tasks (faster; no Ollama required)
    [switch]$KeepGoing       # continue on per-experiment failure
)

$ErrorActionPreference = if ($KeepGoing) { 'Continue' } else { 'Stop' }

function RunEval($configPath) {
    Write-Host ">>> uv run lsm eval run --config $configPath"
    uv run lsm eval run --config $configPath
}

function RunPython($snippet, $label) {
    Write-Host ">>> python: $label"
    uv run python -c $snippet
}

# ---------- retrieval-layer benchmarks (no LLM) ------------------------------

RunPython @"
from pathlib import Path
from lsm.core.embeddings import HashEmbedder
from lsm.eval.tasks.habituation_signal_v1 import HabSignalConfig, compare_ablation
out = Path('experiments') / 'habituation_signal_v1' / 'result.json'
print(compare_ablation(HabSignalConfig(num_items=40, n_filler_ids=10, filler_hits=6, seed=0), embedder=HashEmbedder(dim=128), out_path=out))
print('->', out)
"@ 'habituation_signal_v1'

RunPython @"
from pathlib import Path
from lsm.core.embeddings import HashEmbedder
from lsm.eval.tasks.association_signal_v1 import AssocSignalConfig, compare_ablation
out = Path('experiments') / 'association_signal_v1' / 'result.json'
print(compare_ablation(AssocSignalConfig(num_items=40, co_hits=16, seed=0), embedder=HashEmbedder(dim=128), out_path=out))
print('->', out)
"@ 'association_signal_v1'

RunPython @"
from pathlib import Path
from lsm.eval.tasks.memory_write_precision_v1 import WritePrecisionConfig, compare
out = Path('experiments') / 'memory_write_precision_v1' / 'result.json'
print(compare(WritePrecisionConfig(n_signal=60, n_filler=60, seed=0), out_path=out))
print('->', out)
"@ 'memory_write_precision_v1'

RunPython @"
from pathlib import Path
from lsm.core.embeddings import HashEmbedder
from lsm.eval.tasks.multi_session_consistency_v1 import ConsistencyConfig, compare
out = Path('experiments') / 'multi_session_consistency_v1' / 'result.json'
print(compare(ConsistencyConfig(num_items=40, filler_turns=60, top_k=5, seed=0), embedder=HashEmbedder(dim=128), out_path=out))
print('->', out)
"@ 'multi_session_consistency_v1'

# ---------- end-to-end recall_pref_v1 (requires Ollama running) --------------

if (-not $SkipEndToEnd) {
    $configs = @(
        'experiments/sample_recall_pref_v1_raw_llm.yaml',
        'experiments/sample_recall_pref_v1_naive_rag.yaml',
        'experiments/sample_recall_pref_v1_lsm.yaml',
        'experiments/sample_recall_pref_v1_lsm_no_assoc.yaml',
        'experiments/sample_recall_pref_v1_lsm_rules_writer.yaml'
    )
    foreach ($c in $configs) {
        try { RunEval $c } catch {
            Write-Warning "experiment failed: $c -> $_"
            if (-not $KeepGoing) { throw }
        }
    }
}

Write-Host "phase 1 suite complete. See experiments/ for outputs."
