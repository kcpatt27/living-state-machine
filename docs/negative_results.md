# Negative results log

If a hypothesis fails, record it here with the run id, config, and brief
analysis. Reviewers respect transparent negative results; do not hide them.

Template:

```
## <date> — <short title>
- Hypothesis tested: H{level}
- Run id(s): <experiments/<run_id>>
- Config: <path to YAML>
- Observed: <metric values + direction>
- Expected: <from docs/hypotheses.md>
- Analysis: <1–3 paragraphs: likely cause, what we tried, next steps>
- Disposition: <kept / revised / abandoned>
```

---

## 2026-05-22 — Multi-timescale memory underperforms flat on synthetic gap task

- Hypothesis tested: H8 (tiered, decay-aware memory recalls cross-session
  facts better than a flat store).
- Run id(s): `experiments/multi_session_consistency_v1/` (seed 0).
- Config: built-in defaults inside `lsm/eval/tasks/multi_session_consistency_v1.py`
  — 40 items, 60 filler turns, `top_k=5`, `gap_seconds=21600`, `HashEmbedder`.
- Observed: `flat_accuracy = 0.25`, `tiered_accuracy = 0.20`, **Δ = −0.05**.
  Tiered store was *worse* than the flat `InMemoryStore` baseline.
- Expected: tiered ≥ flat, with positive Δ once gap exceeds the fast-tier
  τ. Hypotheses doc projected a small positive Δ on this scale.
- Analysis: most likely causes, in order of suspicion: (1) default
  `TierConfig` thresholds (`tau_seconds`, `promote_hits`, `vitality_floor`)
  are tuned for realistic conversation cadence, not a 60-turn synthetic
  filler block — fast-tier entries decay below `vitality_floor` before
  the gap, never reach session/durable, and so are missed; (2) `HashEmbedder`
  at dim 384 gives noisy neighbours, which hurts the tiered store more
  because it composes scores across tiers; (3) only one seed run so far —
  the absolute numbers (0.20 vs 0.25) are within plausible seed-to-seed
  noise on a 40-item task. The task itself is synthetic and may not be
  the right shape for the tiered store's strengths.
- Disposition: **kept, flagged for re-tune.** Do not remove
  `MultiTimescaleMemory` from the matrix. Next steps before declaring the
  hypothesis falsified: (a) re-run with seeds `[0, 1, 2]` for variance;
  (b) sweep `TierConfig.tau_seconds` and `vitality_floor` on this task;
  (c) add a more realistic conversation-cadence variant of the task; (d)
  re-test with `FastEmbedder` to rule out embedding noise.

## 2026-05-22 — Encoder verifier underperforms linear on calibration (checkpoint C)

- Hypothesis tested: V.2 checkpoint C (`encoder` should beat `linear` by ≥0.02
  macro-F1 on `verifier_calibration_v1`).
- Run id(s): `experiments/20260522-075336-verifier_calibration_v1/` (full matrix
  including `encoder`); train:
  `uv sync --extra dev --extra research` then
  `uv run python scripts/train_verifier_encoder.py --dataset data/verifier_dataset_v1.jsonl --out-dir models/verifier_encoder_v1 --epochs 1 --batch-size 4 --export-onnx`.
- Config: `experiments/verifier_calibration_v1.yaml` with
  `encoder_model_path: models/verifier_encoder_v1/model.onnx`,
  `encoder_tokenizer_path: models/verifier_encoder_v1/tokenizer`.
- Observed: `encoder` accuracy **0.462**, macro-F1 **0.211** (ties `null`);
  `linear` / `hybrid` remain **0.692** / **0.771**. Per-class: encoder got
  100% reject recall but **0% accept recall** (reject-all collapse on n=13).
- Expected: encoder ≥ linear + 0.02 macro-F1.
- Analysis: with only **13** distilled/perturbed rows and **1 epoch**, a
  freshly initialized classification head on `all-MiniLM-L6-v2` is
  under-specified; the cross-encoder is not a quality win yet. Infrastructure
  is unblocked (research extras, ONNX export, calibration wiring).
- Disposition: **do not promote encoder** for production V.2; keep **linear +
  hybrid**. Revisit after ≥50–100 labeled rows, more epochs, or a
  seq-classification base (not sentence-transformers encoder-only weights).

### Resolved — encoder blocked on research extras (2026-05-22, earlier)

- Was environment-only (`ModuleNotFoundError: torch`). Fixed via
  `uv sync --extra research`; superseded by the quality result above.

## 2026-05-31 — Stentor SSM underperforms Minimal SSM on language modeling loss (Phase 2 small train)

- Hypothesis tested: H12 (Stentor-augmented recurrence outperforms matched vanilla SSM on language modeling loss).
- Run id(s): `runs/small/summary.json` (real TinyStories train).
- Config: `lsm/research/training/scale_configs.py` (`small` profile).
- Observed: `minimal` final loss **2.95** (2,500 steps, seq_len 512) vs `stentor` final loss **5.21** (1,200 steps, seq_len 256).
- Expected: Stentor-augmented SSM should achieve equal or better perplexity/loss compared to minimal SSM at matched compute.
- Analysis: Stentor SSM adds habituation gates (`g_hab`) and association traces (`M`) directly inside the recurrence. On a sequential story dataset like TinyStories, these additional recurrent dynamics introduce extra complexity and noise during early training phases. Furthermore, because the association trace update and habituation EMA decay are currently computed sequentially over the sequence length `T` (rather than fully parallelized via a scan like the diagonal SSM path), the training dynamics are slower and gradients may be more difficult to propagate efficiently.
- Disposition: **Kept, flagged for optimization.** Stentor SSM's sequential gates and association traces require further tuning of hyper-parameters (e.g. `assoc_eta`, `assoc_lambda`, `hab_gamma`) or a parallelized scan implementation for the association trace before we can expect it to outperform a clean, parallelized minimal SSM on general language modeling.
