# Phase 2 L15 gate — joint QLoRA fine-tune

**Status:** train **complete** (2026-06-10); eval regression **run** (2026-06-10) — **L15 exit
not met** (adapter 0.200 vs Ollama baseline 1.000). L14 go/no-go **closed** including
`step_010000` hybrid re-smoke — see `docs/phase2_report.md` §5.

L15 is the first step that **touches the wrapped LLM's weights** (via low-rank
adapters). Everything through L14 kept Qwen frozen behind Ollama; the custom SSM
only projects `s_t`.

---

## What L15 is (and is not)

| In scope | Out of scope |
| --- | --- |
| 4-bit QLoRA on **Qwen2.5-7B-Instruct** via Transformers (`lsm/research/training/qlora.py`) | Retraining or replacing the Ollama-served 14B default |
| Adapter checkpoints under `runs/qlora_*` | Full-weight fine-tune of 7B/14B |
| SFT on conversation / memory-adjacent JSONL derived from `$LSM_HOME` logs | End-to-end Robin / multi-agent discovery loop (north star — separate repo) |
| Eval regression on `recall_pref_v1` after adapter load | Making hybrid `s_t` the default in `lsm chat` (daily-drive gate) |

Scaffold lives in `lsm/research/training/qlora.py`. Runners + RunPod playbook:

- `scripts/run_phase2_qlora_smoke.py` — 50-step smoke → `runs/qlora_smoke/metrics.json`
- `scripts/run_phase2_qlora.py` — full train (default 2000 steps)
- `docs/cloud_burst.md` — RunPod pod spec, upload/download, privacy checklist
- `scripts/runpod_l40s_burst.py` — L40S one-shot: smoke + QLoRA + SSM (see `docs/cloud_burst.md`)

---

## Entry criteria (all must pass before a real L15 train)

- [x] L14 go/no-go closed (minimal core, hybrid eval 1.000 on v1, chat protocol test).
- [ ] **Daily-drive hybrid smoke:** reishi runs `LSM_STATE_UPDATER=hybrid` + `lsm chat`
  for one real session; notes in `docs/manual_chat_protocol.md` or a dated log.
- [x] **SFT dataset v0:** 500 bootstrap rows on-pod (`recall_pref` mirror, seed 0).
  Provenance: `runs/runpod_l40s_burst/DATA_PROVENANCE.md` §1.
- [ ] **VRAM budget check:** 5060 Ti can hold 4-bit 7B + LoRA + hybrid SSM without OOM
  (one forward pass smoke, no full train).
- [ ] **Baseline frozen:** document pre-L15 `recall_pref_v1` + manual chat numbers so
  adapter regressions are measurable.

---

## Suggested first tasks (order)

1. **`scripts/export_sft_from_lsm_home.py`** — read episodic log + facts, emit JSONL
   (reject verifier-rejected assistant rows). **Shipped** — run after `lsm chat` sessions.
2. **`scripts/run_phase2_qlora_smoke.py`** — 50-step QLoRA on 32-row slice; write
   `runs/qlora_smoke/metrics.json`. **Shipped** — run on RunPod per `docs/cloud_burst.md`.
3. **`scripts/run_phase2_qlora.py`** — full QLoRA train. **Shipped**.
4. **Eval hook** — optional `system.backend: transformers` + adapter path in eval YAML
   (keep Ollama default for daily chat). **Done** — `build_backend(adapter_path=…)`,
   `tests/test_transformers_adapter.py`.
5. **Regression matrix** — one `recall_pref_v1` run with adapter vs without. **Done**
   (2026-06-10): base **0.100**, +LoRA **0.200**; see `docs/phase2_report.md` §5.

---

## Success looks like (L15 exit)

- Adapter loads in Transformers backend without breaking `lsm chat` or eval runner. **✅**
- `recall_pref_v1` accuracy **≥ pre-L15 baseline − 0.05** (adapter must not wreck memory path).
  **❌ 0.200** (need ≥ 0.950 vs Ollama 1.000).
- Checkpoint + config archived under `runs/qlora_qwen_7b/` with reproduction command in
  `runs/runpod_l40s_burst/DATA_PROVENANCE.md`. **✅**

**Open follow-ups:** diagnose Transformers-vs-Ollama harness gap; consider fastembed on
Transformers YAMLs; retrain or more bootstrap rows if adapter-vs-base delta stalls.

---

## North star link

L15 nudges the **LLM brain** on domain text. L14's hybrid core nudges **state dynamics**.
The discovery-OS track (`docs/north_star_science_in_simulation.md`) still waits on the
**robotics sim episode API** — L15 does not block that parallel work.

When both L15 smoke and robotics sim API exist, draft the 90-day discovery-OS plan in a
**new repo**.
