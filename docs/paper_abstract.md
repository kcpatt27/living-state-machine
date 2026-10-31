# Living State Machine: Gated, Ablatable Memory for Long-Horizon Preference Recall Around a Frozen Language Model

**Repo:** https://github.com/kcpatt27/living-state-machine
**Author:** Kalen Patterson
**Status:** research closed (Phase 1 exit 2026-05-29; Phase 2 exit 2026-07-15). Manuscript: `paper/main.pdf` on branch `paper/arxiv` (CC BY 4.0); code MIT.
**Assist:** AI pair-programming (Cursor); every claim is backed by committed configs, scripts, and logged negatives.

---

## Abstract

Large language models are stateless between turns. Practical “memory” is usually
prompt stuffing or naive retrieval. This project asks whether a small set of
Stentor-inspired dynamical rules — habituation, association, gated rewrite,
multi-timescale memory, plasticity — can improve long-horizon preference
consistency when wrapped around a frozen local LLM, and whether those rules can
later be baked into a tiny recurrent core.

**Phase 1 (systems).** We built a full local wrapper: OpenAI-compatible server,
persistent hidden state `s_t`, LanceDB memory, ablatable Stentor components, and
a rule/linear verifier path. On a matched-budget preference-recall bench
(`recall_pref_v1`, seeds `[0,1,2]`), LSM accuracy is **1.000** versus naive RAG
**0.100**. Component-level ablations and a harder stress bench
(`recall_pref_v2`) are documented; the gated writer reached **F1 0.98** and was
necessary under decoy load, where always-store collapsed to zero. We do **not**
claim beating a full-context raw LLM on long scripts.

**Phase 2 (integration + negative inductive-bias test).** We trained a minimal
diagonal SSM and a Stentor-augmented variant on TinyStories, then bridged the
minimal checkpoint into the wrapper as an optional hybrid state updater. We also
ran a cheap QLoRA pass on Qwen2.5-7B (~$8 RunPod). Hybrid and adapter arms both
preserve **1.000** on `recall_pref_v1` under a corrected fastembed harness. The
ambitious claim - Stentor recurrence beats matched minimal SSM on LM loss -
**failed** and is logged. Three negatives are recorded: multi-timescale memory,
the encoder verifier (linear + hybrid shipped instead), and the Stentor-style
core. A fourth finding is an evaluation-harness fault that initially looked like
a model regression.

The research track is closed. Remaining work is polish, not unfinished exit criteria.

---

## One-line resume version

Built a local LLM memory wrapper with Stentor-style dynamics; matched-budget
preference recall **1.0 vs 0.1** naive RAG with full ablations; trained a small
SSM + QLoRA (~$8) and verified hybrid/adapter paths do not regress the bench;
documented a negative result that Stentor-in-SSM did not beat a minimal SSM on
TinyStories LM loss.

---

## Pointers

| Doc | Role |
| --- | --- |
| `docs/phase1_exit.md` / `docs/phase1_report.md` | Wrapper claims + numbers |
| `docs/phase2_exit.md` / `docs/phase2_report.md` | Integration claims + numbers |
| `docs/negative_results.md` | Failed hypotheses |
| `docs/recall_pref_v2_findings.md` | Stress-test honesty |
| `docs/hypotheses.md` | Pre-registered claims |
| `paper/main.pdf` / `paper/REPRODUCE.md` (branch `paper/arxiv`) | Manuscript; table-to-run mapping |

---

## Quality-control checklist (AI-assisted research)

Use this when claiming the work is “done,” not when iterating product.

| Check | How we did it |
| --- | --- |
| Hypothesis before code (post-L4) | `docs/hypotheses.md` |
| Ablation / baseline | YAML under `experiments/`; Null\* swaps |
| Multi-seed where claimed | Phase 1 protocol `[0,1,2]`; Phase 2 seed 0 called out |
| Negative results | `docs/negative_results.md` |
| Provenance | `DATA_PROVENANCE.md`, run dirs, reproduction scripts |
| Reproducible checkout | `scripts/setup.ps1`, suite drivers |
| Harness sanity | Hash vs fastembed failure diagnosed before blaming the model |
| Explicit non-claims | Exit docs §Claims |
| Types / tests | pyright + pytest as level “done” gates |
| Separate product from paper | home-agent / LoRA-in-Ollama marked out of scope |

**Preprint:** arXiv `cs.LG`, pending endorsement; the steps are in
[`PUBLICATION.md`](../PUBLICATION.md). The manuscript compiles to 7 pages with
Tectonic. A workshop or conference is the next step for peer review - see the
parked checklist in [`docs/higher_bar_plan.md`](higher_bar_plan.md). Neither is
required to call this research track done.
