# After Phase 2 `small` GPU train — task list

Use this after `scripts/run_phase2_small_train.py` completes and
`runs/small/summary.json` exists. Addresses open product/research questions
1–5 from the Phase 1 → Phase 2 handoff (robotics separation and git hygiene
are already settled).

---

## 1. Clarify what Phase 1 claims (and does not)

**Goal:** One paragraph in the paper trail that cannot be misread as “LSM beat
full-context LLM on v2.”

| Task | Done when |
| --- | --- |
| Add a **Claims** subsection to `docs/phase1_exit.md`: supported (v1 matched budget, CPU ablations, verifier shipped), stress-only (v2), not claimed (beat full `raw_llm` on long scripts). | Section merged |
| Point §5.2 readers to `docs/recall_pref_v2_findings.md` + matched baseline table. | Links in report |
| Optional: one-slide / README bullet for external readers. | If sharing repo |

---

## 2. Map where “learning” lives (post-train inventory)

**Goal:** No ambiguity between regex extraction, wrapper dynamics, and trained weights.

| Task | Done when |
| --- | --- |
| Document **three layers** in `memory-bank/systemPatterns.md`: (A) rule extractor + categories = bootstrap only; (B) wrapper dynamics + normalizer + verifier = Phase 1 learning; (C) SSM weights = Phase 2 learning. | Doc updated |
| After `small` train: log whether **stentor** beat **minimal** on `final_loss` in `runs/small/summary.json`; if not, `docs/negative_results.md` entry. | Result recorded |
| List what is **not** trained yet: frozen Qwen, fact regexes, LanceDB contents. | Explicit list |

---

## 3. Real token data (move off `--synthetic`)

**Goal:** H12 is testable; random tokens only validated the trainer.

| Task | Done when |
| --- | --- |
| Pick corpus (TinyStories, OpenWebText slice, or domain text) and tokenizer → uint16 `.bin`. | `data/tokens_small.bin` from real text |
| Re-run `run_phase2_small_train.py` **without** `--synthetic` for full 20k steps. | New `summary.json` |
| Archive `trainer_config.json` + data provenance note (source URL, tokenizer, byte size). | Under `runs/small/` or `docs/` |

---

## 4. Chat vs benchmark (product truth)

**Goal:** Synthetic `recall_pref` does not become the optimization target.

| Task | Done when |
| --- | --- |
| Define a **manual chat protocol** (5–10 min): opening preference, distraction, recall question — mirror v2 shape but in `lsm chat`. | Script in `docs/` or memory-bank |
| Run once on facts-backed pipeline; capture `lsm state show`, `lsm fact list`, `lsm memory query`. | Notes in `docs/` or dated log |
| Compare qualitatively to v2 metrics (do not expect numeric parity). | Short write-up |

---

## 5. Criteria to touch the frozen 7B (hybrid gate)

**Goal:** Know when L14–L15 is allowed to start.

| Task | Done when |
| --- | --- |
| Write **go/no-go** checklist in `docs/phase2_post_train_tasks.md` (this file) appendix or `memory-bank/activeContext.md`: e.g. stentor ≤ minimal loss on real tokens; `hybrid.py` smoke updates `s_t` without regression on v1 recall. | Checklist written |
| Implement `lsm/research/hybrid.py` wire into eval **one** `recall_pref_v1` config (not v2 headline). | YAML + one run |
| Only then plan QLoRA / 7B joint fine-tune (L15). | Deferred until go |

---

## Suggested order

1. Run `small` train (real or synthetic dry-run).
2. Task **2** (inventory + negative result if needed).
3. Task **1** (claims — quick doc pass).
4. Task **3** (real tokens + re-train).
5. Task **4** (chat protocol).
6. Task **5** (hybrid gate — depends on 2–3).

---

## Appendix: Go/No-Go Checklist for Phase 2 recurrent integration (L14-L15)

Before we allow ourselves to wire recurrent SSM weights directly into the production `s_t` state-update path or plan joint QLoRA fine-tuning (L15), we must satisfy the following criteria:

- [x] **Real Token Training Success:** Both `minimal` and `stentor` models successfully trained on real TinyStories tokens (`tokens_small_tinystories.bin`).
- [ ] **Stentor Convergence or Mitigation:** Stentor SSM achieves a final loss comparable to or better than Minimal SSM on real tokens, OR we explicitly decide to proceed with Minimal SSM as our recurrent core due to the parallel scan performance advantage. (Currently open — `minimal` 2.95 < `stentor` 5.21; see `docs/negative_results.md`.)
- [x] **Hybrid State Updater Smoke Test:** `lsm/research/hybrid.py` is implemented and successfully updates `s_t` from the trained `minimal` checkpoint without crashing. Wired into the eval runner via `state_updater: hybrid` (see `experiments/sample_recall_pref_v1_lsm_hybrid_updater.yaml`).
- [x] **No Regression on recall_pref_v1:** The hybrid state updater scores **1.000** on `recall_pref_v1` (seed 0, `fastembed` retrieval), matching the production `FactsBackedUpdater`. Run: `experiments/20260606-205810-sample_recall_pref_v1_lsm_hybrid_updater/`. Note: requires a real (`fastembed`) embedder — the `hash` embedder scores 0.100 because structured-fact retrieval is meaningless under hashed vectors.
- [ ] **Manual Chat Protocol Pass:** The hybrid state updater passes the 5-10 minute manual chat protocol without corrupting state or failing to retrieve asserted preferences.
