# Phase 2 training — performance notes

## What was slow (2026-05-31)

The **minimal** and **stentor** cores used a Python `for step in range(T)` over
the full sequence length inside every layer. At `seq_len=1024` and 12 layers
that is tens of thousands of Python iterations **per optimizer step**, which
explains ~**60–80 s/step** on a 5060 Ti — not normal for a 30M model.

Sleep/hibernate during the first run made wall-clock even worse; stentor then
ran overnight on synthetic data.

## Fixes shipped

| Change | Effect |
| --- | --- |
| `diagonal_ssm_scan` in `minimal_ssm` | **minimal ~0.8–2 s/step** (was ~67 s) |
| Stentor: scan for core `h`, slimmer assoc loop | modest; still sequential assoc |
| Defaults: minimal **2500** steps @ **512** ctx | was 20k @ 1024 |
| Defaults: stentor **1200** steps @ **256** ctx | wall-clock budget |
| `packed_bin_iterator` pin_memory + non_blocking | less host overhead |
| TF32 + cuDNN benchmark | faster matmuls |
| Logs: `step_time`, `tok/s` | spot regressions |

## Windows checklist

- **Power & sleep:** Settings → System → Power → **Sleep: Never** (on AC).
- **Hibernate:** `powercfg /hibernate off` (optional).
- Screen off is fine; **system sleep is not**.
- Watch `step_time=` in logs; if it jumps to minutes, the machine slept.

## Sanity benchmark

```powershell
uv run python scripts/benchmark_train_step.py --steps 15 --minimal-only
uv run python scripts/benchmark_train_step.py --steps 10 --stentor-only
```

## Rough wall-clock (after fixes, 5060 Ti)

| Variant | ~s/step | Default steps | ~total |
| --- | --- | --- | --- |
| minimal | 1–2 | 2500 | ~1–1.5 h |
| stentor | 8–12 @ seq 256 | 1200 | ~2.5–4 h |

Override with `--steps` / `--seq-len` for longer science runs.

## Real data

```powershell
uv run python scripts/prepare_tokens_small.py
uv run python scripts/run_phase2_small_train.py --data-path data/tokens_small_tinystories.bin
```
