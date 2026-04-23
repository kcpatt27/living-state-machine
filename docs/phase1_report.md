# Phase 1 Report — Living State Machine (Wrapper)

**Scope.** Levels L0 – L10 of the build ladder: a fully local, wrapped-LLM
personal AI with an explicit hidden state, episodic memory, durable
vector memory, habituation, association, gated writing, multi-timescale
memory, outcome scoring, and per-entry plasticity. All components are
decomposed behind `Protocol` interfaces so each can be ablated
independently.

Hardware: single RTX 5060 Ti (16 GB VRAM; ~12 GB for the AI), 32 GB RAM,
Windows 11. No cloud resources were used.

> **Note on numbers.** This report fixes the protocol, the code paths, the
> reproduction scripts, and the ablation matrix. Tables marked
> *to-be-filled* require running the scripts below on reishi's machine
> (the scaffolding environment did not have Python/uv installed at the
> time of writing). Every script is deterministic given its seed.

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

## 5. Expected results (to be filled after first run)

### 5.1 `recall_pref_v1` (end-to-end accuracy; higher is better)

| System | Accuracy (mean ± std) |
| --- | --- |
| `raw_llm` | _tbd_ |
| `naive_rag` | _tbd_ |
| `lsm_full` | _tbd_ |
| `lsm_no_association` | _tbd_ |
| `lsm_no_habituation` | _tbd_ |
| `lsm_no_writer` | _tbd_ |
| `lsm_flat_memory` | _tbd_ |
| `lsm_multi_memory` | _tbd_ |

### 5.2 Retrieval-layer benchmarks (MRR; higher is better)

| Task | Ablation-off | Ablation-on | Δ MRR |
| --- | --- | --- | --- |
| `habituation_signal_v1` | _tbd_ | _tbd_ | _tbd_ |
| `association_signal_v1` | _tbd_ | _tbd_ | _tbd_ |

### 5.3 Writer precision (`memory_write_precision_v1`)

| Writer | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- |
| `always` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `never` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `rules` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| `learned` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

### 5.4 Multi-session consistency (`multi_session_consistency_v1`)

| Store | Top-k accuracy |
| --- | --- |
| `flat` | _tbd_ |
| `multi-timescale` | _tbd_ |

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

Tracked in `docs/negative_results.md`. Phase 1 has not yet logged a
headline negative result; pre-registered hypotheses live in
`docs/hypotheses.md` and will be checked off against the filled tables
above.

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
