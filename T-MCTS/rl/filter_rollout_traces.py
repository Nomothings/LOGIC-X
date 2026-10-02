#!/usr/bin/env python3
"""Aggregate N rollouts per question from run_inference traces; keep 1..K/N correct.

This implements the rejection-sampling filter for RL: a question is kept for
the next optimization round when the current policy solves it in some but not
all rollouts (neither always wrong — likely unlearnable — nor always right —
no gradient signal under a correctness reward).

Preserves full trace_internal records (including complete messages).
Writes keep/discard with a flush after each question for real-time durability.

Usage:
    python filter_rollout_traces.py --eval-dir outputs/rl/LOGIC-X-8B/eval \
                                    --source ../train_data/LOGIC-X-8B/rl_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterator, Optional

ROLLOUT_SUFFIX_RE = re.compile(r"^(.*)__r(\d+)$")


def parse_base_id(sample_id: str) -> tuple[str, Optional[int]]:
    m = ROLLOUT_SUFFIX_RE.match(sample_id or "")
    if not m:
        return sample_id, None
    return m.group(1), int(m.group(2))


def iter_trace_files(run_dir: str) -> list[str]:
    files = []
    for name in sorted(os.listdir(run_dir)):
        if name == "trace_internal.jsonl" or (
            name.startswith("trace_internal_") and name.endswith(".jsonl")
        ):
            files.append(os.path.join(run_dir, name))
    return files


def latest_run_dir(eval_root: str) -> str:
    if not os.path.isdir(eval_root):
        raise FileNotFoundError(f"eval root not found: {eval_root}")
    runs = sorted(
        d
        for d in os.listdir(eval_root)
        if d.startswith("run_") and os.path.isdir(os.path.join(eval_root, d))
    )
    if not runs:
        raise FileNotFoundError(f"no run_* under {eval_root}")
    return os.path.join(eval_root, runs[-1])


def load_source_by_id(path: str) -> dict[str, dict[str, Any]]:
    """Map original question id -> source sample (without rollout fields)."""
    out: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            base = row.get("base_id") or parse_base_id(row.get("id", ""))[0]
            # Prefer first occurrence of each base (all rollouts share fields)
            if base not in out:
                clean = {
                    k: v
                    for k, v in row.items()
                    if k not in ("base_id", "rollout_index")
                }
                clean["id"] = base
                out[base] = clean
    return out


def iter_traces(run_dir: str) -> Iterator[dict[str, Any]]:
    for path in iter_trace_files(run_dir):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Filter RL candidates from inference traces")
    p.add_argument("--eval-dir", required=True, help="Inference output root (.../MODEL/eval)")
    p.add_argument("--source", required=True,
                   help="Question jsonl used for rollouts (to recover sample fields by id)")
    p.add_argument("--model-name", default="",
                   help="Model name stored in rl_meta (default: parent of eval-dir)")
    p.add_argument("--n-rollouts", type=int, default=5)
    p.add_argument("--min-correct", type=int, default=1,
                   help="Inclusive lower bound on n_correct to keep")
    p.add_argument("--max-correct", type=int, default=3,
                   help="Inclusive upper bound on n_correct to keep")
    p.add_argument("--output-dir", default="",
                   help="Where to write keep/discard/summary (default: parent of eval-dir)")
    args = p.parse_args()

    eval_root = os.path.abspath(args.eval_dir)
    run_dir = latest_run_dir(eval_root)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(eval_root)
    model_name = args.model_name or os.path.basename(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    keep_path = os.path.join(out_dir, "rl_keep.jsonl")
    discard_path = os.path.join(out_dir, "rl_discard.jsonl")
    summary_path = os.path.join(out_dir, "filter_summary.json")

    print(f"[info] run_dir={run_dir}")
    print(f"[info] loading source map from {args.source}")
    source_map = load_source_by_id(os.path.abspath(args.source))

    # base_id -> list indexed by rollout_index
    buckets: dict[str, list[Optional[dict[str, Any]]]] = defaultdict(
        lambda: [None] * args.n_rollouts
    )
    orphan = 0
    for tr in iter_traces(run_dir):
        qid = tr.get("sample_id", "")
        base, ridx = parse_base_id(qid)
        if ridx is None or ridx < 0 or ridx >= args.n_rollouts:
            orphan += 1
            continue
        buckets[base][ridx] = tr  # keep full record, including messages

    # Real-time append writers
    keep_fh = open(keep_path, "w", encoding="utf-8")
    discard_fh = open(discard_path, "w", encoding="utf-8")

    hist: Counter[int] = Counter()
    n_keep = n_discard = n_incomplete = 0

    try:
        for base in sorted(buckets.keys()):
            rollouts = buckets[base]
            present = [r for r in rollouts if r is not None]
            if len(present) < args.n_rollouts:
                n_incomplete += 1
                # Still emit discard with whatever we have, marked incomplete
                n_correct = sum(1 for r in present if r.get("success"))
                reason = "incomplete_rollouts"
                keep = False
            else:
                n_correct = sum(1 for r in rollouts if r and r.get("success"))
                keep = args.min_correct <= n_correct <= args.max_correct
                reason = "keep" if keep else (
                    "all_wrong" if n_correct == 0 else "too_easy"
                )

            hist[n_correct] += 1
            src = source_map.get(base, {"id": base})
            record = dict(src)
            record["rl_meta"] = {
                "model": model_name,
                "n_correct": n_correct,
                "n_total": args.n_rollouts,
                "n_present": len(present),
                "keep": keep,
                "reason": reason,
            }
            # Full traces, ordered by rollout_index; None slots become missing markers
            record["rollouts"] = [
                r if r is not None else {"sample_id": f"{base}__r{i}", "missing": True}
                for i, r in enumerate(rollouts)
            ]

            line = json.dumps(record, ensure_ascii=False) + "\n"
            if keep:
                keep_fh.write(line)
                keep_fh.flush()
                n_keep += 1
            else:
                discard_fh.write(line)
                discard_fh.flush()
                n_discard += 1
    finally:
        keep_fh.close()
        discard_fh.close()

    summary = {
        "model": model_name,
        "run_dir": run_dir,
        "n_base_questions": len(buckets),
        "n_keep": n_keep,
        "n_discard": n_discard,
        "n_incomplete": n_incomplete,
        "orphan_traces": orphan,
        "correct_hist": {str(k): hist[k] for k in range(0, args.n_rollouts + 1)},
        "keep_rate": (n_keep / len(buckets)) if buckets else 0.0,
        "min_correct": args.min_correct,
        "max_correct": args.max_correct,
        "keep_path": keep_path,
        "discard_path": discard_path,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()

    print(f"[ok] keep={n_keep} discard={n_discard} incomplete={n_incomplete}")
    print(f"[ok] hist={dict(sorted((int(k), v) for k, v in summary['correct_hist'].items()))}")
    print(f"[ok] {summary_path}")


if __name__ == "__main__":
    main()
