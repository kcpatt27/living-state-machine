"""Export SFT JSONL from local LSM episodic log (L15 dataset v0).

Reads ``$LSM_MEMORY_DIR/episodes.sqlite``, pairs user/assistant turns, and
skips assistant replies the verifier logged as ``reject``.

Usage::

    uv run python scripts/export_sft_from_lsm_home.py
    uv run python scripts/export_sft_from_lsm_home.py --out data/sft_v0.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from lsm.config import get_settings

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "sft_v0.jsonl"


def _rejected_candidates(verifier_db: Path) -> set[str]:
    if not verifier_db.exists():
        return set()
    conn = sqlite3.connect(str(verifier_db))
    try:
        rows = conn.execute(
            "SELECT candidate FROM verifier_events WHERE decision = 'reject'"
        ).fetchall()
    finally:
        conn.close()
    return {str(r[0]) for r in rows if r and r[0]}


def _load_turns(episodes_db: Path) -> list[dict[str, Any]]:
    if not episodes_db.exists():
        return []
    conn = sqlite3.connect(str(episodes_db))
    try:
        rows = conn.execute(
            "SELECT turn_id, session_id, role, content, ts, meta "
            "FROM turns ORDER BY session_id, ts ASC"
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for turn_id, session_id, role, content, ts, meta_raw in rows:
        meta: dict[str, Any] = {}
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                meta = {}
        out.append(
            {
                "turn_id": str(turn_id),
                "session_id": str(session_id),
                "role": str(role),
                "content": str(content),
                "ts": float(ts),
                "meta": meta,
            }
        )
    return out


def build_sft_rows(
    turns: list[dict[str, Any]],
    *,
    rejected: set[str],
) -> list[dict[str, Any]]:
    """Pair user messages with the following assistant reply in each session."""
    by_session: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        by_session.setdefault(turn["session_id"], []).append(turn)

    rows: list[dict[str, Any]] = []
    for session_id, session_turns in by_session.items():
        pending_user: str | None = None
        for turn in session_turns:
            role = turn["role"]
            text = turn["content"].strip()
            if not text:
                continue
            if role == "user":
                pending_user = text
                continue
            if role != "assistant" or pending_user is None:
                continue
            if text in rejected:
                pending_user = None
                continue
            rows.append(
                {
                    "session_id": session_id,
                    "messages": [
                        {"role": "user", "content": pending_user},
                        {"role": "assistant", "content": text},
                    ],
                }
            )
            pending_user = None
    return rows


def export_sft(
    *,
    episodes_db: Path,
    verifier_db: Path,
    out_path: Path,
) -> dict[str, Any]:
    rejected = _rejected_candidates(verifier_db)
    turns = _load_turns(episodes_db)
    rows = build_sft_rows(turns, rejected=rejected)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "out": str(out_path),
        "episodes_db": str(episodes_db),
        "verifier_db": str(verifier_db),
        "turns_read": len(turns),
        "pairs_written": len(rows),
        "rejected_skipped": len(rejected),
    }
    log.info("wrote %d SFT pairs -> %s", len(rows), out_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--home", type=Path, default=None, help="LSM_HOME override")
    parser.add_argument("--memory-dir", type=Path, default=None, help="LSM_MEMORY_DIR override")
    args = parser.parse_args()

    settings = get_settings()
    home = args.home or settings.home
    memory_dir = args.memory_dir or settings.memory_dir
    episodes_db = memory_dir / "episodes.sqlite"
    verifier_db = home / "verifier_log.sqlite"

    stats = export_sft(
        episodes_db=episodes_db,
        verifier_db=verifier_db,
        out_path=args.out,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
