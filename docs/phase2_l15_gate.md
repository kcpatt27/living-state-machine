# Phase 2 L15 gate — joint QLoRA fine-tune

**Status:** open (2026-06-06). L14 go/no-go is **closed** — see appendix in
`docs/phase2_post_train_tasks.md`.

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

Scaffold lives in `lsm/research/training/qlora.py`. No runner script or dataset
pipeline is committed yet.

---

## Entry criteria (all must pass before a real L15 train)

- [x] L14 go/no-go closed (minimal core, hybrid eval 1.000 on v1, chat protocol test).
- [ ] **Daily-drive hybrid smoke:** reishi runs `LSM_STATE_UPDATER=hybrid` + `lsm chat`
  for one real session; notes in `docs/manual_chat_protocol.md` or a dated log.
- [ ] **SFT dataset v0:** ≥500 rows JSONL at `data/sft_v0.jsonl` with fields
  `{messages: [{role, content}, ...]}` mined from verifier-safe turns in
  `$LSM_HOME` (or synthetic mirror of `recall_pref` dialogues for bootstrap).
- [ ] **VRAM budget check:** 5060 Ti can hold 4-bit 7B + LoRA + hybrid SSM without OOM
  (one forward pass smoke, no full train).
- [ ] **Baseline frozen:** document pre-L15 `recall_pref_v1` + manual chat numbers so
  adapter regressions are measurable.

---

## Suggested first tasks (order)

1. **`scripts/export_sft_from_lsm_home.py`** — read episodic log + facts, emit JSONL
   (reject verifier-rejected assistant rows).
2. **`scripts/run_phase2_qlora_smoke.py`** — 50-step QLoRA on 32-row slice; write
   `runs/qlora_smoke/metrics.json`.
3. **Eval hook** — optional `system.backend: transformers` + adapter path in eval YAML
   (keep Ollama default for daily chat).
4. **Regression matrix** — one `recall_pref_v1` run with adapter vs without.

---

## Success looks like (L15 exit)

- Adapter loads in Transformers backend without breaking `lsm chat` or eval runner.
- `recall_pref_v1` accuracy **≥ pre-L15 baseline − 0.05** (adapter must not wreck memory path).
- Checkpoint + config archived under `runs/qlora_qwen_7b/` with reproduction command in
  `runs/qlora_qwen_7b/DATA_PROVENANCE.md`.

---

## North star link

L15 nudges the **LLM brain** on domain text. L14's hybrid core nudges **state dynamics**.
The discovery-OS track (`docs/north_star_science_in_simulation.md`) still waits on the
**robotics sim episode API** — L15 does not block that parallel work.

When both L15 smoke and robotics sim API exist, draft the 90-day discovery-OS plan in a
**new repo**.
