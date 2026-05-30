# Hypotheses log

Every Stentor-augmented level registers its hypothesis here before code is
written. Format:

```
### H{level}: <one-line claim>
- Metric: <name, range, direction>
- Baseline: <exact config>
- Expected delta: <number + justification>
- Falsification: <what result would disprove it>
- Outcome (2026-05-29): <result or pending>
```

---

### H4: explicit long-term memory + retrieval beats naive long-prompt

- Metric: `recall_pref_v1` accuracy on 50-turn dialogues (0..1, up).
- Baseline: raw Qwen2.5-14B with a sliding-window context of 4k tokens and
  no retrieval.
- Expected delta: +0.20 absolute at matched token budget.
- Falsification: delta < +0.05 across 3 seeds.
- **Outcome:** **Supported** on v1 protocol hardening — LSM 1.000 vs naive_rag
  0.100 (Δ +0.90). v2 harder bench pending (`recall_pref_v2`).

### H5: habituation improves retrieval quality by suppressing repeated filler

- Metric: `recall_pref_v1` with distractor density doubled; same metric.
- Baseline: L4 system without habituation.
- Expected delta: +0.05–0.10 absolute.
- Falsification: no improvement or regression across 3 seeds.
- **Outcome:** **Partial** — v1 protocol: `lsm_no_habituation` 0.967 vs full
  1.000 (Δ −0.033). CPU `habituation_signal_v1` MRR +0.068. v2 pending.

### H6: association traces improve multi-key retrieval

- Metric: `association_signal_v1` hit-rate (0..1, up).
- Baseline: L5 system without association.
- Expected delta: +0.10 absolute at matched top-k.
- Falsification: no improvement across 3 seeds.
- **Outcome:** **Supported** on CPU signal task (Δ MRR +0.317). E2E v1
  `lsm_no_association` still 1.000 — bench too easy; v2 pending.

### H7: gated writer reduces memory bloat without hurting recall

- Metrics: durable-memory entry count (down, soft target: <= 1.5x naive) AND
  `multi_session_consistency_v1` accuracy (up, hard constraint: >= baseline).
- Baseline: L6 system with "always store" writer.
- Expected: entries reduced by ~40%, accuracy unchanged or improved.
- Falsification: accuracy drop > 0.03.
- **Outcome:** **Supported** on writer precision (rules F1 0.98). E2E v1:
  `lsm_no_writer` 0.933 vs full 1.000 (Δ −0.067 > falsification threshold on
  easy bench — writer matters when stressed).

### H8: multi-timescale memory beats single-tier at long horizons

- Metric: `multi_session_consistency_v1` across 5 sessions (up).
- Baseline: L7 system with single durable tier.
- Expected delta: +0.08 absolute.
- Falsification: no improvement across 3 seeds.
- **Outcome:** **Not supported** on CPU gap task (Δ −0.05); logged in
  `docs/negative_results.md`. E2E v1 multi-memory tied full at 1.0.

### H9: outcome-driven state update improves task-specific adaptation

- Metric: repeated-task improvement rate (slope of accuracy vs repeat count).
- Baseline: L8 system with no outcome scorer.
- Expected: positive slope > baseline flat slope.
- Falsification: no positive slope or slope <= baseline.
- **Outcome:** **Not evaluated** in Phase 1 headline suite (deferred benchmark).

### H10: meta-memory plasticity scheduling reduces catastrophic state drift

- Metric: state-integrity score on adversarial inputs.
- Baseline: L9 system with flat eta across all entries.
- Expected: fewer integrity regressions; matched or improved task accuracy.
- Falsification: integrity equal or worse.
- **Outcome:** **Not evaluated** in Phase 1 headline suite (unit/integration
  tests only).

### H12: Stentor-augmented recurrence outperforms matched vanilla SSM

- Metric: perplexity on curated long-context eval + recall on our suite.
- Baseline: same-size SSM without the habituation/association gate.
- Expected: lower perplexity at equal params and compute.
- Falsification: equal or worse perplexity at matched compute.
- **Outcome:** **Phase 2** — tiny GPU smoke passed; `small` train pending.
