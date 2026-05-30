"""Generate recall_pref_v2 experiment YAMLs from v1 templates."""

from __future__ import annotations

from pathlib import Path

import yaml

V2_TASK = {"name": "recall_pref_v2", "num_items": 15, "distractor_turns": 45}
RUN = {"seeds": [0, 1, 2]}
EMBED = {"kind": "hash", "dim": 384}
BASE = Path("experiments")

STEMS = [
    "raw_llm",
    "raw_llm_matched",
    "naive_rag",
    "lsm",
    "lsm_no_assoc",
    "lsm_no_habituation",
    "lsm_no_writer",
    "lsm_flat_memory",
    "lsm_multi_memory",
]


def main() -> None:
    for stem in STEMS:
        src = BASE / f"sample_recall_pref_v1_{stem}.yaml"
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        cfg["name"] = f"sample_recall_pref_v2_{stem}"
        cfg["task"] = dict(V2_TASK)
        run: dict[str, object] = {"seeds": list(RUN["seeds"])}
        if stem == "raw_llm_matched":
            run["max_context_turns"] = 4
        cfg["run"] = run
        cfg["embedder"] = dict(EMBED)
        if cfg.get("system", {}).get("kind") == "lsm":
            cfg["system"]["state_updater"] = "facts_backed"
        out = BASE / f"sample_recall_pref_v2_{stem}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(out)

    hybrid_src = BASE / "sample_recall_pref_v1_lsm_hybrid_verifier.yaml"
    hybrid = yaml.safe_load(hybrid_src.read_text(encoding="utf-8"))
    hybrid["name"] = "sample_recall_pref_v2_lsm_hybrid_verifier"
    hybrid["task"] = dict(V2_TASK)
    hybrid["run"] = dict(RUN)
    hybrid["embedder"] = dict(EMBED)
    hybrid["system"]["state_updater"] = "facts_backed"
    out = BASE / "sample_recall_pref_v2_lsm_hybrid_verifier.yaml"
    out.write_text(yaml.safe_dump(hybrid, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
