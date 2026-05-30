# Phase 1 Report — Living State Machine (Wrapper)

**Scope.** Levels L0 – L10 of the build ladder: a fully local, wrapped-LLM
personal AI with an explicit hidden state, episodic memory, durable
vector memory, habituation, association, gated writing, multi-timescale
memory, outcome scoring, and per-entry plasticity. All components are
decomposed behind `Protocol` interfaces so each can be ablated
independently.

Hardware: single RTX 5060 Ti (16 GB VRAM; ~12 GB for the AI), 32 GB RAM,
Windows 11. No cloud resources were used.

> **Run log (reishi's machine).** **2026-05-22 ~06:06 local** — CPU
> retrieval/writer benchmarks (`.\scripts\run_phase1_suite.ps1
> -SkipEndToEnd`, `HashEmbedder`, seed 0). **~06:32** — full suite including
> end-to-end `recall_pref_v1` (`.\scripts\run_phase1_suite.ps1 -KeepGoing`,
> Ollama `qwen2.5:7b-instruct-q8_0`, seed `[0]`). Unit suite at fill time:
> **241 passed**, 3 skipped.

---

## 1. System under test

The **LSM wrapper** runs as an OpenAI-compatible FastAPI server in front
of a frozen pretrained LLM (Ollama primary: `qwen2.5:14b-instruct-q4_K_M`;
Transformers path optional for research). Every turn traverses the
pipeline in this order:

1. `observe`      — append user turn to the SQLite episodic log
2. `retrieve`     — vector search over durable memory, scored by
   `cosine * habituation_damping * association_boost * trust_boost`
3. `compose`      — assemble the prompt: base system + `s_t` projection +
   top-k retrieved memories + most recent user turns
4. `call_llm`     — run the frozen model
5. `score_outcome`— heuristic or self-critic rates the reply in `[0, 1]`
6. `update_state` — LLM-summarized next `s_t`; persisted to disk
7. `write_memory` — gated writer decides keep/skip
8. `update_plasticity` — trust moves toward outcome score on retrieved
   entries; plasticity shrinks with consistent evidence

Each stage is a class behind a protocol with a corresponding `Null*`
implementation; ablation flags in the experiment YAML pick one or the
other.

## 2. Components and interfaces

| Level | Module | Protocol | Prod impl. | Ablation (`Null*`) |
| --- | --- | --- | --- | --- |
| L2 | `runtime/llm_backend.py` | `LLMBackend` | `OllamaBackend`, `TransformersBackend` | — |
| L3 | `core/state.py` | `StateUpdater` | `LLMSummarizerUpdater` | `NullStateUpdater` / `PassthroughStateUpdater` |
| L4 | `core/memory.py` | `MemoryStore` | `LanceDBStore` | `InMemoryStore` |
| L5 | `core/habituation.py` | `Habituation` | `ExpDecayHabituation` | `NullHabituation` |
| L6 | `core/association.py` | `Association` | `HebbianAssociation` | `NullAssociation` |
| L7 | `core/writer.py` | `MemoryWriter` | `RuleBasedWriter`, `LearnedWriter` | `AlwaysWriter`, `NeverWriter` |
| L8 | `core/multi_timescale.py` | `MemoryStore` (fan-out) | `MultiTimescaleMemory` | flat `InMemoryStore` / `LanceDBStore` |
| L9 | `core/outcome.py` | `OutcomeScorer` | `HeuristicOutcomeScorer`, `LLMSelfCriticScorer` | `NullOutcomeScorer` |
| L10 | `core/plasticity.py` | `Plasticity` | `ExpDecayPlasticity` | `NullPlasticity` |

Score composition is exposed through `chain_combiners`, so
habituation × association × plasticity signals each contribute a
component to the final cosine-weighted score and are separately logged
for post-hoc analysis.

## 3. Benchmark suite

All benchmarks live in `lsm/eval/tasks/`. Each is deterministic given its
seed, returns per-item outputs, and ships an ablation comparison entry
point.

| Benchmark | Layer | Metric | Primary purpose |
| --- | --- | --- | --- |
| `recall_pref_v1` | end-to-end | accuracy | Can the system answer a preference question buried ≥ N turns deep? |
| `habituation_signal_v1` | retrieval | mean reciprocal rank | Does habituation surface a rare signal over repeated filler? |
| `association_signal_v1` | retrieval | mean reciprocal rank | Does a Hebbian co-occurrence edge move co-activated items up? |
| `memory_write_precision_v1` | writer | precision / recall / F1 | Does the writer keep durable facts and reject filler? |
| `multi_session_consistency_v1` | memory tiers | top-k accuracy | Do facts survive a cross-session gap under tiered decay? |

## 4. Experimental protocol

- Every run is driven by a YAML config committed to
  `experiments/<run_id>/config.yaml`.
- Seeds: `[0, 1, 2]` for end-to-end tasks; seeded deterministic generation
  for retrieval-only tasks.
- Metrics reported as `mean ± std` across seeds; raw per-item outputs
  persisted to `raw_outputs.jsonl`.
- Each ablation variant gets its own config YAML and its own run folder.
- Compute budget (wall time, peak VRAM) is logged; wrapper-layer runs are
  CPU-for-embedding, GPU-for-LLM.

### 4.1 Ablation matrix

All ablation variables are boolean flags under the `ablation:` block:

```
state_enabled
memory_enabled
habituation_enabled
association_enabled
writer_enabled
```

Plus variable-level selectors in `system:`:

```
memory.kind:     memory | lancedb | multi
writer.kind:     always | never | rules | learned
outcome.kind:    null   | heuristic | self_critic
plasticity.kind: null   | exp_decay
```

The full matrix is `2^5 × {memory.kind} × {writer.kind}` = sparse in
practice. The headline runs we prioritise:

| Row | Cfg |
| --- | --- |
| `raw_llm` | no state, no memory, no Stentor primitives |
| `naive_rag` | memory only; `writer.kind: always`, `habituation: off`, `association: off`, `plasticity: off` |
| `lsm_full` | everything on |
| `lsm_no_habituation` | `habituation_enabled: false` |
| `lsm_no_association` | `association_enabled: false` |
| `lsm_no_writer` | `writer.kind: always` |
| `lsm_no_state` | `state_enabled: false` |
| `lsm_no_plasticity` | `plasticity.kind: null` |
| `lsm_flat_memory` | `memory.kind: memory` (single tier) |
| `lsm_multi_memory` | `memory.kind: multi` |

## 5. Results

Source files: `experiments/<task>/result.json` (CPU benchmarks);
`experiments/<timestamp>-sample_recall_pref_v1_*/metrics.json` (end-to-end;
gitignored run dirs, see §5.1 footnote).

### 5.1 `recall_pref_v1` (end-to-end accuracy; higher is better)

| System | Accuracy (mean ± std) |
| --- | --- |
| `raw_llm` | 1.000 ± 0.000 |
| `naive_rag` | 0.100 ± 0.100 |
| `lsm_full` | 1.000 ± 0.000 |
| `lsm_no_association` | 1.000 ± 0.000 |
| `lsm_no_habituation` | 0.967 ± 0.058 |
| `lsm_no_writer` | 0.933 ± 0.115 |
| `lsm_flat_memory` | 0.933 ± 0.058 |
| `lsm_multi_memory` | 1.000 ± 0.000 |

_End-to-end footnote:_ configs under `experiments/sample_recall_pref_v1_*.yaml`;
`task.num_items=10`, `distractor_turns=20`, seeds **`[0, 1, 2]`** (protocol
hardening pass **2026-05-29**, `.\scripts\run_protocol_hardening.ps1`), model
`qwen2.5:7b-instruct-q8_0`. Run ids:
`20260529-174440-…-raw_llm` through `20260529-191925-…-multi_memory`.
Writer and flat-memory ablations now show modest drop vs `lsm_full`; habituation
ablation borderline; association and multi-memory still tie full at 1.0 on this
bench. First-pass seed-0-only numbers (2026-05-22 / 20260529-165529 deferred
batch) superseded for headline rows above.
Run folders (2026-05-22 headline pass):
`20260522-061246-sample_recall_pref_v1_raw_llm`,
`20260522-061257-sample_recall_pref_v1_naive_rag`,
`20260522-061302-sample_recall_pref_v1_lsm`,
`20260522-061936-sample_recall_pref_v1_lsm_no_assoc`,
`20260522-062553-sample_recall_pref_v1_lsm_rules_writer` (supplementary:
`writer.kind: rules`, **0.900 ± 0.000**), and
`20260522-072258-sample_recall_pref_v1_lsm_hybrid_verifier` (supplementary:
`verifier.kind: hybrid` with linear head, **1.000 ± 0.000**, no regression vs
`lsm_full` on this seed-0 pass).
Eval uses `LLMSummarizerUpdater` (legacy summarizer path), not production
`FactsBackedUpdater` — interpret as wrapper retrieval/state ablation, not
daily `lsm chat` defaults.

### 5.2 `recall_pref_v2` (harder end-to-end; production path)

Longer gaps (45 distractor turns), decoy preferences on the same topic, repeated
filler prompts, 15 items/seed, seeds **`[0, 1, 2]`**. LSM configs use
**`FactsBackedUpdater`** + fact extraction (`state_updater: facts_backed`) —
matches production `lsm chat`. Configs:
`experiments/sample_recall_pref_v2_*.yaml`; driver:
`.\scripts\run_recall_pref_v2_suite.ps1`.

| System | Accuracy (mean ± std) |
| --- | --- |
| `raw_llm` | 0.911 ± 0.038 |
| `naive_rag` | 0.044 ± 0.038 |
| `lsm_full` | 0.089 ± 0.038 |
| `lsm_no_association` | 0.089 ± 0.038 |
| `lsm_no_habituation` | 0.089 ± 0.038 |
| `lsm_no_writer` | 0.000 ± 0.000 |
| `lsm_flat_memory` | 0.089 ± 0.038 |
| `lsm_multi_memory` | 0.089 ± 0.038 |
| `lsm_hybrid_verifier` | 0.089 ± 0.038 |

_Suite completed 2026-05-30_ (`20260530-072304-…-raw_llm` through
`…-072606-hybrid_verifier`). v2 **does** separate ablations (writer gate
critical at 0.000), but headline LSM configs cluster at ~0.09 while
`raw_llm` stays high — likely interaction of long scripted dialogues,
decoy pollution, and retrieval/top-k limits vs full-context baseline.
Treat as **stress-test**, not a Phase 1 success claim. Full write-up (baseline
mismatch, retrieval/decoys, learning-vs-regex direction):
**`docs/recall_pref_v2_findings.md`**. Exit framing: `docs/phase1_exit.md`.

### 5.3 Retrieval-layer benchmarks (MRR; higher is better)

Headline field per arm: **`mrr_mean`** in `experiments/<task>/result.json`
(`with_*` = component on, `without_*` = `Null*` ablation).

| Task | Ablation-off (`Null*`) | Ablation-on (prod impl.) | Δ MRR |
| --- | --- | --- | --- |
| `habituation_signal_v1` | 0.314 | 0.382 (`ExpDecayHabituation`) | +0.068 |
| `association_signal_v1` | 0.454 | 0.772 (`HebbianAssociation`) | +0.317 |

Config: 40 items, seed 0, `HashEmbedder` dim 128 (suite driver).

### 5.4 Writer precision (`memory_write_precision_v1`)

Synthetic 60 signal / 60 filler turns, seed 0. Headline fields:
`precision`, `recall`, `f1`, `false_positive_rate` per `writer` in
`results[]`.

|  Writer   | Precision | Recall |   F1   |  FPR   |
| --------- | --------- | ------ | ------ | ------ |
| `always`  |    0.50   |  1.00  |  0.67  |  1.00  |
| `never`   |    0.00   |  0.00  |  0.00  |  0.00  |
| `rules`   |    1.00   |  0.97  |  0.98  |  0.00  |
| `learned` |    1.00   |  1.00  |  1.00  |  0.00  |

### 5.5 Multi-session consistency (`multi_session_consistency_v1`)

Headline fields: **`flat_accuracy`**, **`tiered_accuracy`** (Δ =
`tiered_accuracy − flat_accuracy`).

| Store | Top-k accuracy |
| --- | --- |
| `flat` (`InMemoryStore`) | 0.25 |
| `multi-timescale` (`MultiTimescaleMemory`) | 0.20 |

Δ tiered − flat: **−0.05** (tiered underperformed flat on this synthetic
gap task at seed 0; logged as a candidate negative result).

### 5.6 Verifier calibration (`verifier_calibration_v1`)

Dataset run id: `20260522-075336-verifier_calibration_v1`, source
`data/verifier_dataset_v1.jsonl` (n=13 rows; rule-distill + synthetic
perturbations from local verifier telemetry). Encoder artifact:
`models/verifier_encoder_v1/` (HF + ONNX, 1 epoch on n=13).

| Verifier variant | Accuracy | Macro-F1 | Brier (accept) | ROC-AUC (accept vs not) |
| --- | --- | --- | --- | --- |
| `null` | 0.462 | 0.211 | 0.538 | 0.500 |
| `rule` | 0.538 | 0.556 | 0.489 | 0.429 |
| `linear` | 0.692 | 0.771 | 0.187 | 0.810 |
| `hybrid` (`rule` safety floor + linear head) | 0.692 | 0.771 | 0.187 | 0.810 |
| `encoder` (MiniLM cross-encoder, ONNX) | 0.462 | 0.211 | 0.265 | 0.476 |

Interpretation: **checkpoint B** — linear head improves over the rule
baseline on this first local dataset. **checkpoint C** — encoder did not
beat linear (Δ macro-F1 = −0.56); the model collapsed to predicting
`reject` for all rows (0% accept/recall). Logged as a quality negative in
`docs/negative_results.md`; **ship linear + hybrid** for V.2 until more
training data and/or a better base checkpoint.

**V.2 closeout:** ship **`linear` + `hybrid`** for eval and future live wiring;
encoder deferred per `docs/negative_results.md`. See `docs/phase1_exit.md`.

## 6. Reproduction

From a clean checkout on Windows:

```powershell
.\scripts\setup.ps1          # installs Scoop-managed tools, verifies Ollama/Docker, pulls models
uv sync                       # installs the Python stack
uv run pytest -q              # runs the unit + integration suite
.\scripts\run_phase1_suite.ps1  # runs every headline experiment
```

Each experiment writes `experiments/<run_id>/` containing:

- `config.yaml` — exact config used
- `metrics.json` — aggregate + per-seed metrics
- `raw_outputs.jsonl` — per-item answers and grading
- `log.txt` — runner log

## 7. Negative results

Tracked in `docs/negative_results.md`. Current entries:

- **2026-05-22 — multi-timescale memory underperforms flat on synthetic gap
  task** (Δ = −0.05 on `multi_session_consistency_v1`, seed 0; kept and
  flagged for re-tune, not abandoned).
- **2026-05-22 — encoder verifier underperforms linear on calibration**
  (macro-F1 0.211 vs 0.771 at n=13; reject-all collapse; see
  `docs/negative_results.md`).

Pre-registered hypotheses live in `docs/hypotheses.md` and will be checked
off against the filled tables above.

## 8. Limitations

- No cloud compute; ceilings on batch size and long-context runs.
- Heuristic outcome scorer is a stand-in for human labels; the
  self-critic variant inherits the base LLM's biases.
- The learned writer is trained on the synthetic benchmark distribution
  and has not been stress-tested on personal corpora.
- LanceDB on Windows occasionally holds file locks across rapid
  open/close cycles; the test suite uses `InMemoryStore` by default.

## 9. Appendix — file map

```
lsm/core/
  state.py              # L3
  memory.py             # L4
  habituation.py        # L5
  association.py        # L6
  writer.py             # L7
  multi_timescale.py    # L8
  outcome.py            # L9
  plasticity.py         # L10
  pipeline.py           # orchestrates all of the above
lsm/runtime/            # L2 — LLM backends + prompt composer
lsm/serve/              # L2 — FastAPI server
lsm/cli/                # L2 — typer CLI
lsm/eval/tasks/         # benchmarks
lsm/eval/baselines/     # raw_llm, naive_rag, lsm_system
lsm/eval/runner.py      # YAML -> experiment run
experiments/            # per-run folders
docs/                   # hypotheses, architecture, privacy, this report
memory-bank/            # project documentation
```
