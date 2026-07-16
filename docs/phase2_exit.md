# Phase 2 exit checklist

**Status:** closed for research / paper-trail purposes as of 2026-07-15.
**Scope:** L11–L15 — small recurrent core, hybrid `s_t` bridge, QLoRA smoke on
wrapped Qwen. Product daily-drive (home-agent, Ollama chat UX) is out of scope.

Companion docs: `docs/phase2_report.md`, `docs/phase1_exit.md`,
`docs/paper_abstract.md`, `docs/negative_results.md`.

---

## What we set out to prove

Two questions, deliberately separated:

1. **Integration:** Can a trained small SSM project into the wrapper’s `s_t`
   path, and can a LoRA adapter load on the Transformers eval path, without
   wrecking `recall_pref_v1`?
2. **Inductive bias (ambitious):** Does a Stentor-augmented recurrence beat a
   matched minimal SSM on language-modeling loss at equal params?

Only (1) is an exit criterion for this close-out. (2) was tested and **failed**
on TinyStories — logged, not hidden.

---

## Exit criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Real-token small train (minimal + stentor) | **Met** | `runs/small/`, TinyStories provenance |
| Hybrid updater preserves `recall_pref_v1` | **Met** | `step_010000.pt` → accuracy **1.000** (fastembed) |
| QLoRA train + provenance | **Met** | RunPod L40S ~$8; `runs/runpod_l40s_burst/DATA_PROVENANCE.md` |
| Adapter eval hook + regression ≥ baseline − 0.05 | **Met** | base & +LoRA **1.000** (fastembed; seed 0) |
| Negative results logged | **Met** | Stentor LM loss worse than minimal; hash-harness false failure documented |
| L13 100–150M scale | **Deferred** | Scaffold only; not required for integration claims |
| Stentor SSM ≥ minimal on LM loss | **Not met** | See `docs/negative_results.md` (2026-05-31) |

---

## Claims

### Supported

1. **Hybrid bridge is safe on v1:** Minimal SSM checkpoint used as
   `state_updater=hybrid` scores **1.000** on `recall_pref_v1` (seed 0,
   fastembed), matching the facts-backed path’s ceiling on this bench.
2. **QLoRA does not wreck the wrapper path:** Transformers 4-bit 7B base and
   +LoRA both score **1.000** on the same task under a working retrieval
   harness (`docs/phase2_report.md` §5).
3. **Harness honesty:** Hash embedder + `min_score: 0.15` yields empty
   retrieval on this task; fastembed is required for a meaningful score.
   Documented in §5.1 of the Phase 2 report.

### Not claimed / failed

1. **Stentor recurrence beats minimal SSM** on TinyStories LM loss — **failed**
   (minimal loss ~2.95 vs stentor ~5.21 on unequal step budgets; still a clear
   negative for the inductive-bias headline).
2. **LoRA improves daily Ollama chat** — not wired; daily driver remains
   frozen Ollama weights.
3. **Bootstrap SFT generalizes** beyond synthetic `recall_pref`-shaped rows —
   in-domain only.
4. **L13 scale** or equal-compute paper-grade Stentor-vs-minimal table on a
   long-horizon suite — not run.

---

## Headline numbers (where to look)

| Source | Content |
| --- | --- |
| `docs/phase2_report.md` §2 | RunPod wall / $ / steps |
| §3 | Hybrid v1 = 1.000 |
| §4–§5 | Transformers base / LoRA = 1.000 (fastembed) |
| `docs/negative_results.md` | Stentor vs minimal LM loss |
| `runs/runpod_l40s_burst/DATA_PROVENANCE.md` | SFT + ckpt provenance |

**Seed note:** Phase 2 integration regressions are **seed 0**. Multi-seed
protocol (`[0,1,2]`) remains a Phase 1 claim for the wrapper suite.

---

## Reproduction

```powershell
uv sync --extra research
.\scripts\run_l15_regression.ps1
.\scripts\run_l15_regression.ps1 -Hybrid
# Optional: small train numbers already in runs/small/summary.json
```

Requires local adapter weights under `runs/runpod_l40s_burst/` (gitignored)
and CUDA research extras for Transformers 4-bit.

---

## Known limitations

1. Easy v1 ceiling (1.000) — good for “no regression,” weak for ranking adapters.
2. Single-seed Phase 2 eval.
3. Verifier calibration still n≈13 (Phase 1 limitation carries forward).
4. Encryption-at-rest not audited (project brief goal, not Phase 2 exit).
5. Stentor-in-recurrence needs more tuning / parallelization before another LM race.

---

## What is closed vs what is product work

| Closed (research) | Open (improvements / product) |
| --- | --- |
| Phase 1 wrapper claims | home-agent orchestration |
| Phase 2 hybrid + QLoRA integration gates | Wire LoRA into Ollama daily chat |
| Negative Stentor LM result | Fact regex for `favourite X is Y` |
| Paper trail + abstract | L13 scale, more SFT data, discovery-OS |

Further work is **improvement**, not unfinished research exit criteria.
