# Phase 2 `small` train — data provenance

Provenance for the real-token corpus used by the `small` training run summarised
in `runs/small/summary.json`. Addresses Task 3 of `docs/phase2_post_train_tasks.md`.

## Corpus

| Field | Value |
| --- | --- |
| Source dataset | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) |
| Split | `train` (streamed) |
| Tokenizer | GPT-2 BPE (`gpt2`) |
| Vocab cap | 32,000 (GPT-2 ids ≥ 32k dropped) |
| Tokens written | 80,000,000 |
| File | `data/tokens_small_tinystories.bin` (uint16) |
| Byte size | 160,000,000 (2 bytes/token) |
| Builder | `scripts/prepare_tokens_small.py` (`--corpus tinystories`) |
| Built | 2026-05-30 (filename fallback after a Windows lock on `tokens_small.bin`) |

Sidecar metadata: `data/tokens_small_tinystories.meta.json`. The `data/` directory
is gitignored (raw token bins are large/derivable); this note is the committed
record of how to regenerate it.

## Regenerate

```powershell
uv sync --extra research
uv run python scripts/prepare_tokens_small.py --corpus tinystories
# if data/tokens_small.bin is locked by a running train, the builder writes
# data/tokens_small_tinystories.bin instead (see _commit_tmp).
```

## Run that consumed it

`runs/small/summary.json` (real tokens, `synthetic: false`):

| Variant | Steps | seq_len | Final loss | Params |
| --- | --- | --- | --- | --- |
| minimal | 2,500 | 512 | 2.95 | 30.6M |
| stentor | 1,200 | 256 | 5.21 | 43.4M |

Steps were intentionally reduced from the original 20k default for local
iteration speed (per the perf work in `docs/phase2_training_perf.md`). A
full-length run remains a deliberate, user-gated decision and is **not** a
blocker for the go/no-go checklist (the "real tokens" criterion is met by the
run above). Stentor < minimal loss is logged in `docs/negative_results.md`.
