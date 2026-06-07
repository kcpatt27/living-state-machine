# Phase 1 exit checklist

**Status:** closed for implementation and paper-trail purposes as of 2026-05-29.
**Scope:** L0–L10 wrapper LSM + V.2 learned verifier (linear/hybrid). Phase 2 (L11+) is explicitly out of scope here.

## What we set out to prove

> Compact Stentor-inspired dynamics (habituation, association, gated memory, multi-timescale tiers, plasticity) wrapped around a frozen local LLM improve long-horizon preference recall and memory behavior versus naive baselines, with each component measurable via ablation.

## Exit criteria (from `memory-bank/projectbrief.md`)

| Criterion | Status | Evidence |
| --- | --- | --- |
| `recall_pref` ≥ naive-RAG + 0.20 at matched budget | **Met** | v1 protocol `[0,1,2]`: LSM 1.000 vs naive_rag 0.100 |
| Each Stentor component ablatable with logged configs | **Met** | 8-config matrix + deferred rows; v2 harder bench added |
| Reproducible from clean checkout | **Met** | `scripts/setup.ps1`, `run_phase1_suite.ps1`, YAML configs |
| Multi-seed protocol | **Met** | `run_protocol_hardening.ps1` (2026-05-29) |
| Negative results logged | **Met** | `docs/negative_results.md` (multi-memory CPU task, encoder verifier) |

## Claims (What we proved and didn't prove)

### Supported Claims (What we proved)
1. **LSM beats Naive RAG at Matched Budget:** On `recall_pref_v1`, LSM achieves 1.000 accuracy vs Naive RAG's 0.100 under matched context/retrieval constraints.
2. **Ablatability of Stentor Components:** Every key component (habituation, association, gated memory, meta-memory plasticity) is ablatable, showing measurable degradation when removed.
3. **Verifier Calibration:** The V.2 linear learned verifier head improves over the rule-based verifier baseline (macro-F1 0.771 vs 0.556).

### Unsupported/Stress-Only Claims (What we did not claim)
1. **Beating Full-Context LLM on Long Scripts:** We do *not* claim that LSM beats `raw_llm` when the latter has access to the full, uncompressed dialogue history (e.g. on `recall_pref_v2` where `raw_llm` gets 90+ turns in context).
2. **Encoder-based Verification:** The V.2 encoder verifier underperformed the linear verifier on our small dataset (collapsed to reject-all), which we logged as a negative result.

## Headline numbers (where to look)

| Report section | Content |
| --- | --- |
| §5.1 | `recall_pref_v1`, seeds `[0,1,2]`, legacy summarizer path |
| §5.2 | `recall_pref_v2` (harder), seeds `[0,1,2]`, **production `FactsBackedUpdater`** |
| §5.3–§5.4 | CPU retrieval/writer benchmarks |
| §5.5 | V.2 verifier calibration (`linear` / `hybrid` shipped; `encoder` negative) |

## V.2 verifier — shipped vs deferred

| Piece | Status |
| --- | --- |
| Context logging + `lsm verify export` | Shipped |
| `verifier_calibration_v1` task | Shipped |
| **Linear learned head** | Shipped — production path for eval hybrid |
| **Hybrid** (rule floor + linear) | Shipped — no E2E regression on v1/v2 smoke |
| Encoder cross-encoder | **Deferred** — quality negative at n=13; revisit after more labels |
| Live `lsm chat` default hybrid | **Not wired** — intentional; enable when daily-driving chat |

## Known limitations (honest)

1. **Easy v1 bench** — many ablations tied at 1.0 until protocol hardening; see §5.1.
2. **Hard v2 bench** — `recall_pref_v2` does **not** support the same headline claim as
   v1 (LSM ≪ `raw_llm` on full-context baseline). Documented in
   **`docs/recall_pref_v2_findings.md`** — stress test, not exit blocker.
3. **Eval path** — v1 E2E used `LLMSummarizerUpdater`; v2 uses `FactsBackedUpdater`
   (matches `lsm chat` default).
4. **Association / multi-memory** — tied on v2 prefs; CPU negative on
   `multi_session_consistency_v1` logged.
5. **Verifier dataset** — n=13 local rows; calibration is directional, not paper-grade ML.
6. **Encryption at rest** — listed in brief; not fully audited in this exit pass.

## Reproduction commands

```powershell
uv sync
uv run pytest -q
.\scripts\run_phase1_suite.ps1 -KeepGoing
.\scripts\run_protocol_hardening.ps1 -KeepGoing      # v1, seeds [0,1,2]
.\scripts\run_recall_pref_v2_suite.ps1               # v2, FactsBackedUpdater
uv run lsm eval run --config experiments/verifier_calibration_v1.yaml
```

## What comes next (Phase 2 — not part of this exit)

- L11 `small` train: `scripts/run_phase2_small_train.py` (tiny smoke:
  `scripts/run_phase2_tiny_smoke.py` + `runs/tiny_smoke/summary.json`).
- Post-train checklist: `docs/phase2_post_train_tasks.md`.
- L12–L14: bake Stentor dynamics into recurrent core; hybrid `s_t` updater.
- See `docs/north_star_science_in_simulation.md` for the long-term discovery goal (separate track).
