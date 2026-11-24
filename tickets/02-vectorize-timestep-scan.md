# 02 - Vectorize the per-timestep loop in the Stentor core

Status: open
Source: mining S8-D0097, shard-8.md.
Repo: `C:\Users\kcpat\projects\living-state-machine`.

## Evidence

The Stentor block runs two Python loops over the sequence length, both carrying
recurrent state:

- `lsm/research/small_core/stentor_ssm.py:113-119` habituation EMA:
  `g_hab` depends on `hab_ema`, and `hab_ema` is updated from the gated input, so
  the loop cannot be removed as-is.
- `lsm/research/small_core/stentor_ssm.py:127-139` association trace: `M` is
  updated each step and feeds the bias for the next step.

`lsm/research/training/trainer.py:153-156` already records the constraint
("Stentor still has a sequential association loop") and keeps `torch.compile`
opt-in (`compile_model: bool = False`, only enabled for CUDA + minimal). So the
"compile is always on" part of the original report is stale; the loop cost is the
real item.

## Action

1. Replace the two Python loops with a chunked scan: a parallel scan for the
   linear decay parts and a blocked step for the gated terms, or a fused kernel.
2. Keep numerical equivalence: a test compares eager vs scanned output on a fixed
   seed and asserts allclose within a stated tolerance.
3. Benchmark tokens/sec before and after on the same box, same config, and record
   both rows (this rig is the slowest supported host; see
   `docs/phase2_training_perf.md`).
4. Only then consider enabling `compile_model` by default for the minimal core.

## Acceptance

- The loop is gone from the forward path, or reduced to a blocked scan.
- The equivalence test passes, and the bench shows a measured speedup or an
  explicit "no gain" result.
- `docs/phase2_training_perf.md` carries the before/after rows.
