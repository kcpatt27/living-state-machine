# Phase 2 Report — L14 hybrid + L15 QLoRA

**Status:** draft (2026-06-10). Complements `docs/phase1_report.md`.

Hardware for L15 train: RunPod **NVIDIA L40S** (~$8.04, ~9.3 h wall). Local eval:
**RTX 5060 Ti 16GB** (Transformers 4-bit 7B + optional LoRA).

---

## 1. What Phase 2 adds

| Track | Artifact | Claim |
| --- | --- | --- |
| **L14** | Minimal SSM hybrid state updater | Recurrent core can project `s_t` without regressing `recall_pref_v1` |
| **L15** | QLoRA on `Qwen2.5-7B-Instruct` | Low-rank adapter on memory-adjacent SFT does not wreck the wrapper path |

Phase 1 frozen Qwen (Ollama) remains the **daily-driver** inference path. Phase 2
research paths use **Transformers** (adapter eval) and optional **hybrid** `s_t`.

---

## 2. RunPod L40S burst (2026-06-09)

Source: `runs/runpod_l40s_burst/summary.json` · Provenance:
`runs/runpod_l40s_burst/DATA_PROVENANCE.md`.

| Phase | Steps | Wall (s) | Notes |
| --- | --- | --- | --- |
| QLoRA smoke | 50 | 91 | 32-row slice |
| QLoRA train | 2000 | 28,496 | 500 bootstrap rows, seed 0 |
| SSM minimal | 10,000 | (in total) | `final_loss` ≈ 2.35 |
| **Total** | — | **33,642** | Est. **$8.04** @ $0.86/hr |

**SFT:** synthetic `recall_pref` bootstrap (seed 0, no personal data).
**Adapter checkpoint:** `qlora_qwen_7b/seed_0/step_002000/`
**SSM checkpoint (production default):** `ssm/minimal/ckpts/step_010000.pt`

---

## 3. L14 hybrid updater — `recall_pref_v1`

Config: `experiments/sample_recall_pref_v1_lsm_hybrid_updater.yaml`
Embedder: **fastembed** (hash embedder is not meaningful for structured-fact retrieval).

| Run id | `state_updater` | ckpt | accuracy (seed 0) |
| --- | --- | --- | --- |
| `20260606-205810-…_hybrid_updater` | hybrid | `runs/small/…/step_002500.pt` | **1.000** |
| `20260610-175541-…_hybrid_updater` | hybrid | `step_010000.pt` | **1.000** |

Go/no-go: hybrid must match `facts_backed` on v1 (≥ 1.000 − 0.05).

---

## 4. L15 QLoRA — adapter regression (`recall_pref_v1`)

Configs:

- Base: `experiments/sample_recall_pref_v1_lsm_transformers_base.yaml`
- + LoRA: `experiments/sample_recall_pref_v1_lsm_transformers_qlora.yaml`

Both use `backend: transformers`, `Qwen/Qwen2.5-7B-Instruct`, 4-bit, hash embedder
(same retrieval substrate as Phase 1 headline `lsm` YAML).

**Pre-L15 Ollama baseline (frozen):** `20260522-061302-sample_recall_pref_v1_lsm` →
**1.000** (seed 0, `qwen2.5:7b-instruct-q8_0`, Ollama).

| Arm | Run id | accuracy_mean | Δ vs pre-L15 |
| --- | --- | --- | --- |
| Transformers 7B base | `20260610-174849-…` | **0.100** | −0.900 |
| Transformers 7B + LoRA | `20260610-175429-…` | **0.200** | −0.800 (gate **not met**) |

Reproduce:

```powershell
uv sync --extra research
.\scripts\run_l15_regression.ps1
.\scripts\run_l15_regression.ps1 -Hybrid   # optional hybrid re-smoke
```

---

## 5. Results log (2026-06-10, local RTX 5060 Ti)

| Config | Run directory | accuracy_mean | Notes |
| --- | --- | --- | --- |
| `transformers_base` | `experiments/20260610-174849-sample_recall_pref_v1_lsm_transformers_base` | **0.100** | 4-bit HF; ignores facts/memories often |
| `transformers_qlora` | `experiments/20260610-175429-sample_recall_pref_v1_lsm_transformers_qlora` | **0.200** | +0.10 vs base; SFT phrasing, cross-item bleed |
| `hybrid_updater` (step_010000) | `experiments/20260610-175541-sample_recall_pref_v1_lsm_hybrid_updater` | **1.000** | Ollama + fastembed; L14 gate **pass** |

**Interpretation:** The L15 exit rule compares adapter accuracy to the **Ollama** pre-L15 baseline
(1.000). Transformers 4-bit inference is not parity with Ollama q8 on this harness — base
already scores 0.100. LoRA improves the Transformers arm (+0.10) but does **not** close the
gap to 0.950. Next investigative levers: Transformers prompt/context limits, fastembed on the
Transformers YAMLs, or a gate reframed as adapter-vs-Transformers-base only.

---

## 6. L15 exit checklist

| Criterion | Status |
| --- | --- |
| SFT v0 documented | ✅ `DATA_PROVENANCE.md` |
| QLoRA trained + archived | ✅ RunPod burst |
| Adapter loads in Transformers eval | ✅ `adapter_path` in `build_backend` |
| `recall_pref_v1` regression ≥ baseline − 0.05 | ❌ **0.200** vs 1.000 Ollama baseline |
| Hybrid `step_010000` smoke (L14) | ✅ **1.000** |
| `DATA_PROVENANCE` + reproduction script | ✅ |

---

## 7. Explicit non-claims

- LoRA does **not** replace Ollama `qwen3.5:9b` daily chat (no Ollama adapter hook yet).
- Bootstrap SFT is **in-domain** with `recall_pref`; not general assistant tuning.
- SSM LM loss on TinyStories does not imply Stentor beats minimal (see `docs/negative_results.md`).
