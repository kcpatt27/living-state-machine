"""Build ``data/tokens_small.bin`` for Phase 2 ``small`` training.

Downloads **TinyStories** from Hugging Face (streaming), tokenizes with GPT-2
BPE, keeps token ids in ``[0, vocab_size)``, and writes a flat **uint16** file
compatible with ``lsm.research.training.trainer.packed_bin_iterator``.

The ``small`` profile uses ``vocab_size=32_000``; GPT-2 ids above that are
dropped (rare subwords — most story text uses lower ids).

Usage::

    uv sync --extra research
    uv run python scripts/prepare_tokens_small.py
    uv run python scripts/prepare_tokens_small.py --max-tokens 25000000

Outputs (under ``data/``, gitignored)::

    tokens_small.bin
    tokens_small.meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "tokens_small.bin"
DEFAULT_VOCAB = 32_000
# Enough random windows for 20k steps @ batch=4, seq=1024 (see run_phase2_small_train).
DEFAULT_MAX_TOKENS = 80_000_000

CORPORA = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "split": "train",
        "text_key": "text",
    },
    "tinystories-valid": {
        "path": "roneneldan/TinyStories",
        "split": "validation",
        "text_key": "text",
    },
}


def _commit_tmp(tmp_path: Path, out_path: Path) -> Path:
    """Move temp bin into place; fall back if target is locked (e.g. training run)."""
    import shutil

    try:
        if out_path.exists():
            out_path.unlink()
        tmp_path.replace(out_path)
        return out_path
    except PermissionError:
        alt = out_path.with_name(f"{out_path.stem}_tinystories{out_path.suffix}")
        log.warning(
            "%s is locked (training run?). Writing %s instead.",
            out_path,
            alt,
        )
        if alt.exists():
            alt.unlink()
        shutil.move(str(tmp_path), str(alt))
        return alt


def _load_tokenizer():
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _iter_texts(corpus: str):
    from datasets import load_dataset

    spec = CORPORA[corpus]
    log.info("streaming %s split=%s", spec["path"], spec["split"])
    ds = load_dataset(spec["path"], split=spec["split"], streaming=True)
    key = spec["text_key"]
    for row in ds:
        text = row.get(key) if isinstance(row, dict) else None
        if text and str(text).strip():
            yield str(text)


def build_bin(
    out_path: Path,
    *,
    corpus: str,
    vocab_size: int,
    max_tokens: int,
    append_eos: bool,
) -> dict[str, object]:
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; choose from {list(CORPORA)}")

    tokenizer = _load_tokenizer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".bin.tmp")

    written = 0
    dropped = 0
    stories = 0
    t0 = time.perf_counter()

    with open(tmp_path, "wb") as fh:
        for text in _iter_texts(corpus):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if append_eos:
                ids.append(tokenizer.eos_token_id)
            if not ids:
                continue
            stories += 1
            arr = np.asarray(ids, dtype=np.int64)
            over = arr >= vocab_size
            dropped += int(over.sum())
            kept = arr[~over]
            if kept.size == 0:
                continue
            kept.astype(np.uint16).tofile(fh)
            written += int(kept.size)
            if written >= max_tokens:
                break
            if stories % 5000 == 0:
                elapsed = time.perf_counter() - t0
                log.info(
                    "progress: %d stories, %.1fM tokens, %.0f tok/s",
                    stories,
                    written / 1e6,
                    written / max(elapsed, 1e-6),
                )

    if written < vocab_size + 1025:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"only {written} tokens written — need more data or a larger corpus")

    # Trim file to exact max if we overshot on last story.
    need_bytes = min(written, max_tokens) * 2
    with open(tmp_path, "r+b") as fh:
        fh.truncate(need_bytes)
    written = need_bytes // 2

    final_path = _commit_tmp(tmp_path, out_path)
    elapsed = time.perf_counter() - t0
    meta = {
        "out": str(final_path),
        "corpus": corpus,
        "hf_dataset": CORPORA[corpus]["path"],
        "split": CORPORA[corpus]["split"],
        "tokenizer": "gpt2",
        "vocab_size": vocab_size,
        "num_tokens": written,
        "num_stories": stories,
        "dropped_ids_ge_vocab": dropped,
        "append_eos": append_eos,
        "bytes": written * 2,
        "elapsed_s": round(elapsed, 2),
    }
    meta_path = final_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "wrote %s (%.1fM tokens, %.1f MB, %d stories, %d ids dropped) in %.1fs",
        final_path,
        written / 1e6,
        (written * 2) / 1e6,
        stories,
        dropped,
        elapsed,
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .bin path (default: data/tokens_small.bin)",
    )
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPORA),
        default="tinystories",
        help="HF streaming corpus (default: tinystories train)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB,
        help="Keep token ids in [0, vocab_size); must match scale_configs.small",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Stop after this many tokens (default: {DEFAULT_MAX_TOKENS:,})",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Do not append GPT-2 EOS after each story",
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    try:
        meta = build_bin(
            out,
            corpus=args.corpus,
            vocab_size=args.vocab_size,
            max_tokens=args.max_tokens,
            append_eos=not args.no_eos,
        )
        print(json.dumps(meta, indent=2))
    except Exception as exc:
        log.exception("prepare failed")
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
