"""Write a temp eval YAML with overridden run.seeds (for protocol hardening)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Source experiment YAML.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/_protocol_tmp"),
        help="Directory for patched configs.",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run = raw.setdefault("run", {})
    run["seeds"] = [int(s) for s in args.seeds]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.config.stem}_seeds{'-'.join(str(s) for s in args.seeds)}.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(out.as_posix())


if __name__ == "__main__":
    main()
    sys.exit(0)
