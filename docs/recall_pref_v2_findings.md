# `recall_pref_v2` — stress-test findings (reference)

**Purpose.** Capture what the harder preference-recall suite showed, without
treating it as a Phase 1 success metric. Revisit when improving long-horizon
recall, eval fairness, or Phase 2 dynamics — not as a mandate to add more
hand-written preference regexes.

**Suite.** `lsm/eval/tasks/recall_pref_v2.py`; driver
`scripts/run_recall_pref_v2_suite.ps1`; configs `experiments/sample_recall_pref_v2_*.yaml`.
Completed **2026-05-30**, seeds `[0, 1, 2]`. Headline numbers in
`docs/phase1_report.md` §5.2.

---

## Headline numbers (mean ± std over seeds)

| System | Accuracy |
| --- | --- |
| `raw_llm` | 0.911 ± 0.038 |
| `naive_rag` | 0.044 ± 0.038 |
| `lsm_full` (+ most LSM ablations) | 0.089 ± 0.038 |
| `lsm_no_writer` (`writer.kind: always`) | 0.000 ± 0.000 |

Run folders: `experiments/20260530-072304-*` through `…-072606-*`.

---

## What this is *not*

- **Not proof the wrapper is useless.** v1 protocol (§5.1) still shows LSM ≫
  naive RAG at matched retrieval budget on the easier bench.
- **Not a random eval bug.** Runs completed cleanly; failures are qualitative
  (wrong decoy value, “don’t know”, hedging) not crashes.
- **Not solved by “add another extractor pattern for `my favourite X is Y`.”**
  That would hard-code one surface form. The intended direction is **systems
  that learn what to bind to which slot** (preferences, goals, entities) from
  experience and dynamics — see §“Architectural direction” below.

---

## What the pattern likely means

### 1. Unmatched baseline (`raw_llm`)

`RawLLMBaseline` feeds the **entire** scripted dialogue (~90+ turns) into the
model, then asks the question. LSM at answer time sees only:

- `state_projection` (from `FactsBackedUpdater`, capped ~600 chars), and
- **top-k = 4** retrieved memory lines (`min_score` 0.15).

So v2 compares **full transcript in context** vs **compressed state + 4 snippets**.
That alone can explain much of the `raw_llm` vs LSM gap without invoking a
mysterious regression.

**Matched baseline (eval hygiene):** `experiments/sample_recall_pref_v2_raw_llm_matched.yaml`
sets `run.max_context_turns: 4` (last four user+assistant pairs + question).
Driver: `scripts/run_recall_pref_fair_baselines.ps1`. v1 twin:
`sample_recall_pref_v1_raw_llm_matched.yaml`. Fill results in §“Matched baseline
results” below after running.

### 2. Opening lines vs conservative fact extraction

v1/v2 openings use phrasing like:

`just for context: my favourite city is medellín. please remember that.`

Today’s rule extractor is intentionally conservative (`like` / `love` /
`prefer`, corrections, paired terms, media, etc.). That opening shape does
**not** currently produce structured facts (verified on representative strings).

On v2, LSM therefore leans on **episodic retrieval + projection from whatever
facts exist**, not on a rich preference slot filled at turn 0. v1 hid this
because the dialogue was shorter and less adversarial (everything tied at 1.0).

This is a **coverage / learning problem**, not a reason to bolt on a one-off
“favourite X is Y” regex. The product answer is: **better association of
utterances to slots** via normalizer + dynamics + (Phase 2) recurrent core,
not a longer pattern list.

### 3. Decoy pollution (memory + retrieval)

Decoys are same-topic, wrong-value lines (`my coworker's favourite city is …`).
They usually **do not** enter the fact store (no first-person `like` anchor),
but they **are** written to durable memory (when the gated writer allows) and
compete in vector retrieval. `naive_rag` at ~0.04 shows that **snippet
retrieval alone** loses under this noise — consistent with the stress-test
design.

### 4. Writer ablation naming

`lsm_no_writer` in the v2 matrix uses **`writer.kind: always`** (store every
turn), not “no memory.” Flooding the store with filler and decoys → **0.000**
accuracy. That **validates** the gated writer hypothesis under stress, even
when other ablations tie.

### 5. Tied LSM ablations at ~0.09

Habituation, association, flat vs multi-memory, and hybrid verifier all cluster
with `lsm_full` on v2. Interpretation: **the bottleneck is upstream** (what
gets into state/facts vs what retrieval returns), not the marginal effect of
those components on this specific synthetic task.

### 6. Eval vs `lsm chat` parity gap (documented, low priority)

Production `Pipeline._context()` can attach **`structured_facts`** (vector
retrieval over the fact store). `LSMSystem` in eval currently composes only
`state_projection` + `retrieved` memories. Even after richer learning, eval
should mirror chat wiring — but this is **not** the main v2 story while
openings still yield few facts.

---

## Architectural direction (Kalen Patterson’s intent)

**Goal:** patterns and systems **inside** LSM so it learns to associate
surface forms with the right **ideas** (preferences, goals, nouns, roles) —
not a growing list of “when X then preference” rules.

**Already aligned with that goal (Phase 1):**

- `EmbeddingTermNormalizer` — learns canonical forms from exposure, not
  hardcoded alias tables.
- `FactStore` + categories/slots — ontology for *what* to store, not every
  English phrasing.
- Habituation, association, plasticity, gated writer — dynamics that shape
  what persists and what resurfaces.
- V.2 **learned verifier** — scoring from context features, not new regexes.

**Where harder recall likely moves (Phase 2+, not Phase 1 exit blockers):**

- Recurrent / Stentor core (L12–L14) binding salient turns to state.
- Richer training signal from corrections, outcomes, and verifier feedback.
- Attribution (user vs third-party) via **verifier + reconcile**, not phrase lists.
- Optional eval baselines with **matched context budget**.

Explicitly **deferred:** expanding `extract_facts_from_user_turn` with
preference-specific templates for benchmark phrasing.

---

## Disposition

| Item | Disposition |
| --- | --- |
| v2 as Phase 1 success metric | **Rejected** — stress test only (`docs/phase1_exit.md`) |
| v1 protocol + CPU signals | **Keep** as primary Phase 1 evidence |
| Matched `raw_llm` baseline | **Shipped** — YAMLs + `run_recall_pref_fair_baselines.ps1` |
| Eval `structured_facts` parity | **Shipped** in `LSMSystem.answer` (fact retrieval at question time) |
| More preference regexes | **Rejected** per product direction above |
| Phase 2 train / dynamics | **Next** engineering track |

---

## Matched baseline results

| Task | `raw_llm` (full) | `raw_llm` matched (`max_context_turns: 4`) | `lsm_full` (v2) |
| --- | --- | --- | --- |
| `recall_pref_v1` | 1.000 ± 0.000 (§5.1) | **0.000 ± 0.000** | — |
| `recall_pref_v2` | 0.911 ± 0.038 | **0.000 ± 0.000** | 0.089 ± 0.038 |

_Runs 2026-05-30:_ `20260530-081245-sample_recall_pref_v1_raw_llm_matched`,
`20260530-081310-sample_recall_pref_v2_raw_llm_matched`.

With only the **last four** dialogue pairs in context, the opening preference is
never visible — `raw_llm` fails entirely. LSM still ~0.09 because retrieval/state
can surface older material. The v2 gap vs full `raw_llm` (~0.91) is therefore
**dominated by full-transcript access**, not proof that the wrapper adds nothing.

## Reproduction

```powershell
.\scripts\run_recall_pref_v2_suite.ps1
.\scripts\run_recall_pref_fair_baselines.ps1
```

Inspect per-run `metrics.json` under `experiments/20260530-*`.
