#!/usr/bin/env python3
"""Expand a T-MCTS question set into N rollouts per question for RL sampling.

Each question with id X becomes N rows with ids X__r0 ... X__r{N-1}.
After sampling with run_inference.py, filter_rollout_traces.py re-groups
the rollouts by the base id.

Usage:
    python expand_rollouts.py --input ../train_data/LOGIC-X-8B/rl_prompts.jsonl \
                              --output outputs/rl/input_5x.jsonl --n-rollouts 5
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expand_sample(sample: dict[str, Any], n_rollouts: int) -> list[dict[str, Any]]:
    base_id = sample.get("id", "")
    if not base_id:
        raise ValueError(f"sample missing id: keys={list(sample.keys())}")

    out: list[dict[str, Any]] = []
    for k in range(n_rollouts):
        row = dict(sample)
        row["base_id"] = base_id
        row["rollout_index"] = k
        row["id"] = f"{base_id}__r{k}"
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Expand questions into N rollouts each")
    p.add_argument("--input", required=True, help="Source question jsonl")
    p.add_argument("--output", required=True, help="Expanded jsonl path")
    p.add_argument("--n-rollouts", type=int, default=5)
    p.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = p.parse_args()

    out_path = os.path.abspath(args.output)
    meta_path = os.path.join(os.path.dirname(out_path), "expand_meta.json")

    if os.path.isfile(out_path) and not args.force:
        n_lines = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"[skip] exists: {out_path} ({n_lines} lines). Use --force to rebuild.")
        return

    samples = load_jsonl(args.input)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tool_counts: Counter[str] = Counter()
    n_out = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for sample in samples:
            tool_counts[str(sample.get("ground_truth_tool", "?"))] += 1
            for row in expand_sample(sample, args.n_rollouts):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
                if n_out % 5000 == 0:
                    fh.flush()
        fh.flush()

    meta = {
        "input": os.path.abspath(args.input),
        "output": out_path,
        "n_source": len(samples),
        "n_rollouts": args.n_rollouts,
        "n_expanded": n_out,
        "tool_counts": dict(tool_counts),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"[ok] wrote {n_out} rows -> {out_path}")
    print(f"[ok] meta -> {meta_path}")


if __name__ == "__main__":
    main()
