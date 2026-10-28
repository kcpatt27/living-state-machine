# Reproducing the paper's tables

Verified 2026-10-27 against the run artifacts committed under `experiments/`.
Each directory holds `config.yaml`, `metrics.json`, `raw_outputs.jsonl`, and a
`seed_*` folder. `accuracy_mean` / `accuracy_std` in `metrics.json` are the
values quoted below.

## Table 1 — preference recall (seeds 0–2)

| System | v1 | Artifact | v2 | Artifact |
| --- | --- | --- | --- | --- |
| Naive RAG | 0.100 ± 0.100 | `20260529-174501-sample_recall_pref_v1_naive_rag` | 0.044 ± 0.038 | `20260530-072347-sample_recall_pref_v2_naive_rag` |
| Raw LLM, matched budget | 0.000 ± 0.000 | `20260530-081245-sample_recall_pref_v1_raw_llm_matched` | 0.000 ± 0.000 | `20260530-081310-sample_recall_pref_v2_raw_llm_matched` |
| Raw LLM, full transcript | 1.000 ± 0.000 | `20260529-174440-sample_recall_pref_v1_raw_llm` | 0.911 ± 0.038 | `20260530-072304-sample_recall_pref_v2_raw_llm` |
| LSM full | 1.000 ± 0.000 | `20260529-174517-sample_recall_pref_v1_lsm` | 0.089 ± 0.038 | `20260530-072409-sample_recall_pref_v2_lsm` |

## Table 2 — ablations on `recall_pref_v1`

| Ablation | Paper | Artifact |
| --- | --- | --- |
| LSM full | 1.000 ± 0.000 | `20260529-174517-sample_recall_pref_v1_lsm` |
| No habituation | 0.967 ± 0.058 | `20260529-182227-sample_recall_pref_v1_lsm_no_habituation` |
| No writer gate | 0.933 ± 0.115 | `20260529-184120-sample_recall_pref_v1_lsm_no_writer` |
| Flat memory | 0.933 ± 0.058 | `20260529-190018-sample_recall_pref_v1_lsm_flat_memory` |
| No association | 1.000 ± 0.000 | `20260529-180413-sample_recall_pref_v1_lsm_no_assoc` |
| Multi-memory | 1.000 ± 0.000 | `20260529-191925-sample_recall_pref_v1_lsm_multi_memory` |

## Table 3 — component evidence

The writer-precision and verifier-calibration figures come from the component
benches (`recall_pref_v2` no-writer run above shows the collapse to zero) and
the phase reports under `docs/`. Check those against `docs/phase2_report.md`
before the submission pass; the table was not re-derived in this check.

## Note on `runs/`

`runs/` holds training runs (QLoRA burst, minimal/stentor cores), not the
evaluation suite. The evaluation artifacts live in `experiments/`; an earlier
review checkout only had `runs/`, which is why the draft said raw run
directories were missing.
