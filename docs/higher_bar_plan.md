# Higher bar — optional peer-review / conference track

**Status:** parked plan (2026-07-15). Research exit is already closed
(`docs/phase2_exit.md`). This doc is for if reishi wants a citable preprint or
workshop/conference submission later. It is **not** unfinished exit work.

Companion: `docs/paper_abstract.md` (current claims + QC checklist).

---

## Goal ladder

| Tier | Output | Peer reviewed? | Effort (rough) |
| --- | --- | --- | --- |
| **A — Portfolio (done)** | GitHub + abstract + exit docs | No | — |
| **B — Preprint** | 6–10 page PDF on arXiv (cs.AI / cs.CL) | Moderated only | 1–2 weekends |
| **C — Workshop** | Short paper + poster/talk | Light | +1–2 weeks on B |
| **D — Main conference** | Full paper, strong related work, multi-seed Phase 2 | Hard | months; may need new results |

Default recommendation: stop at **A** for resume; climb **B** only if you want a
citable link. Climb **C/D** only with a clear novelty angle (see §Framing).

---

## Framing (pick one story)

Reviewers punish “we built a lot of stuff.” Pick a single claim:

1. **Systems (strongest with current data):** Matched-budget memory wrapper with
   ablatable Stentor dynamics beats naive RAG on preference recall; negatives
   and stress benches documented.
2. **Negative inductive bias:** Baking Stentor gates into a small SSM does
   *not* beat a minimal SSM on TinyStories — useful for the “compact dynamics”
   hypothesis literature.
3. **Integration:** Hybrid SSM→`s_t` + LoRA load without regression (weaker as
   a sole claim; good as a systems appendix).

Do **not** lead with “beats full-context LLM” or “Stentor learns like biology.”

---

## Gap list vs current artifact

### Must fix before Tier B (arXiv)

| Gap | Why | Concrete action |
| --- | --- | --- |
| No PDF manuscript | arXiv needs a paper | Expand abstract → intro / method / results / limitations (~6–8 pp) |
| Related work thin | Reviewers / moderators expect it | 1 page: MemGPT, RAG, SSM/Mamba, habitiation/associative memory papers |
| Phase 2 single-seed | Easy attack | Re-run hybrid + Transformers base/LoRA on seeds `[0,1,2]`; report mean±std |
| Headline figures | Skimmability | 1 system diagram + 1 ablation table figure from Phase 1 |
| Author/ethics blurb | Norm | Disclose AI-assisted engineering; all numbers from committed runs |

### Should fix before Tier C (workshop)

| Gap | Action |
| --- | --- |
| Easy-bench ceiling (v1 = 1.0) | Add one harder *matched-budget* preference task (not full-context raw_llm) |
| Verifier n≈13 | Collect ≥50–100 labeled rows *or* drop verifier from the paper claim |
| Hash vs fastembed story | One paragraph in methods: eval embedder = production path |
| Fair compute note | Wall time / VRAM table for Phase 1 suite + RunPod $8 burst |

### Required before Tier D (main track)

| Gap | Action |
| --- | --- |
| Novelty vs prior memory agents | Position clearly; may need a new mechanism or stronger theory |
| Stentor-in-SSM | Either fix training (parallel assoc scan, hyperparam sweep) and re-test H12, or keep as explicit negative and don’t claim inductive bias wins |
| L13 / equal-params long-horizon suite | Only if chasing the original Phase 2 brief success criterion |
| External baselines | Compare to at least one published local memory stack under matched budget |
| Multi-seed + significance | Seeds `[0,1,2]` minimum; prefer 5 where cheap |
| Blind spots | Human eval protocol; encryption/privacy claim only if audited |

---

## Suggested work order (when you reopen)

1. **Freeze claims** — copy §Claims from both exit docs into the draft intro;
   no new experiments until the story is one sentence.
2. **Figures** — architecture diagram + Phase 1 ablation table.
3. **Multi-seed Phase 2** — `run_l15_regression.ps1` × seeds `[0,1,2]` (or
   extend YAML); update `phase2_report.md`.
4. **Related work pass** — 8–12 citations, honest diffs.
5. **Write PDF** — LaTeX or Typst; dump numbers only from exit/report docs.
6. **Decide venue** — arXiv first; then scan workshop CFPs (systems, agents,
   reproducibility, negative results).
7. **Only then** reopen Stentor SSM or L13 if the venue needs that claim.

---

## Venue notes (practical)

- **arXiv:** needs endorsement if new; use cs.AI or cs.CL; upload PDF +
  optionally ancillary code link to this repo.
- **Workshops:** look for “agents,” “memory,” “reproducibility,” “negative
  results,” “systems for ML.” Deadlines matter more than prestige here.
- **Main conferences:** treat as a *new* project with a sharper hypothesis —
  don’t force the current repo into NeurIPS shape without new evidence.

---

## Explicit non-goals of this plan

- Rewriting home-agent into the paper
- Retraining QLoRA “just because”
- Claiming peer review before a venue actually reviews
- Hiding the Stentor LM-loss negative

When this plan is active, update `memory-bank/activeContext.md` and open a
dated subsection under `docs/hypotheses.md` for any *new* claim.
