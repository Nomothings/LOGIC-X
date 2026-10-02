#!/usr/bin/env python3
"""
T-MCTS — training trajectory generation runner.

Runs the teacher-guided tool-integrated reasoning pipeline over a set of
questions and exports SFT data:

    trace_internal.jsonl   raw traces (all attempts, full messages)
    sft_trajectories.jsonl full multi-turn trajectories (ShareGPT format)
    sft_atomic_tasks.jsonl single-turn skill items derived from successful
                           traces (tool selection / code generation / repair)
    failed_samples.jsonl   failed questions
    metrics.json           summary statistics

Usage:
    python generate_trajectories.py --input train_data/LOGIC-X-8B/rl_prompts.jsonl --output outputs/gen

Resume (auto-appends to the most recent run):
    python generate_trajectories.py --input train_data/LOGIC-X-8B/rl_prompts.jsonl --output outputs/gen
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime

# Make the repo-level `tmcts` package importable when run from this folder
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tmcts.data import load_samples
from tmcts.llm.client import init_pool
from tmcts.llm.raw_logger import init_raw_logger
from tmcts.pipeline import run_sample_guided
from tmcts.exporter import export_to_sharegpt
from tmcts.augmentation import augment_trajectory, generate_single_task_items
from tmcts.metrics import Metrics
from tmcts.trace import Trace

logger = logging.getLogger("tmcts.generate")


# ======================================================================
# IncrementalWriter — thread-safe, auto-splitting JSONL writer
# ======================================================================

class IncrementalWriter:
    MAX_RECORDS = 5000

    def __init__(self, run_dir: str, prefix: str, append: bool = False):
        self._dir = run_dir
        self._prefix = prefix
        self._lock = threading.Lock()
        self._count = 0
        self._file_idx = 0
        self._fh = None

        if append:
            # Find the last existing file and count its records
            existing = sorted(
                f for f in os.listdir(run_dir)
                if f == f"{prefix}.jsonl" or f.startswith(f"{prefix}_")
            )
            if existing:
                last = existing[-1]
                if last == f"{prefix}.jsonl":
                    self._file_idx = 0
                else:
                    try:
                        self._file_idx = int(last[len(prefix) + 1:].replace(".jsonl", ""))
                    except ValueError:
                        self._file_idx = 0
                last_path = os.path.join(run_dir, last)
                self._count = sum(1 for _ in open(last_path, "r", encoding="utf-8"))
                # If the last file is full, start a new one
                if self._count >= self.MAX_RECORDS:
                    self._file_idx += 1
                    self._count = 0
                    self._open_new()
                else:
                    self._fh = open(last_path, "a", encoding="utf-8")
                return

        self._open_new()

    def _open_new(self) -> None:
        if self._fh:
            self._fh.close()
        if self._file_idx == 0:
            path = os.path.join(self._dir, f"{self._prefix}.jsonl")
        else:
            path = os.path.join(self._dir, f"{self._prefix}_{self._file_idx:03d}.jsonl")
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            self._count += 1
            if self._count % self.MAX_RECORDS == 0:
                self._file_idx += 1
                self._open_new()

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


# ======================================================================
# Helpers
# ======================================================================

def _build_failed_sample(sample: dict, all_attempts: list) -> dict:
    """Record a failed sample."""
    first_content = ""
    if all_attempts:
        for msg in all_attempts[0].messages:
            if msg.get("role") == "assistant":
                first_content = msg.get("content", "")
                break
    return {
        "id": sample.get("id", ""),
        "category": sample.get("category", ""),
        "gold_tool": sample.get("ground_truth_tool", ""),
        "level": sample.get("level", 0),
        "split": sample.get("split", ""),
        "ground_truth": sample.get("ground_truth_answer", ""),
        "fail_reason": all_attempts[-1].fail_reason if all_attempts else "?",
        "question": sample.get("question", {}),
        "first_assistant_content": first_content,
    }


# ======================================================================
# Main orchestrator
# ======================================================================

def run(samples: list[dict],
        *,
        output_dir: str,
        workers: int = 64,
        max_repair: int = 2,
        new_run: bool = False,
        limit: int | None = None,
        search_mode: str = "linear"):
    """Run the T-MCTS generation pipeline on *samples*.

    Each sample gets exactly one attempt. When *new_run* is False, the
    runner auto-resumes from the most recent run directory under
    *output_dir*.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ---- Resolve run_dir ----
    completed_ids: set[str] = set()
    append_mode = False

    if new_run or not _latest_run(output_dir):
        run_name = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, run_name)
    else:
        run_dir = _latest_run(output_dir)
        append_mode = True
        for fname in sorted(os.listdir(run_dir)):
            if fname.startswith("trace_internal") and fname.endswith(".jsonl"):
                with open(os.path.join(run_dir, fname)) as f:
                    for line in f:
                        try:
                            completed_ids.add(json.loads(line).get("sample_id", ""))
                        except Exception:
                            pass
        logger.info("Auto-resume: %d completed ids from %s", len(completed_ids), run_dir)

    os.makedirs(run_dir, exist_ok=True)

    log_handler = logging.FileHandler(os.path.join(run_dir, "run.log"))
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(log_handler)
    logger.setLevel(logging.INFO)

    # Init API pool
    init_pool()
    init_raw_logger(os.path.join(run_dir, "raw_api_calls.parquet"))

    samples = [s for s in samples if s.get("id", "") not in completed_ids]
    if limit:
        samples = samples[:limit]
    total = len(samples)
    logger.info("Samples: %d (filtered), workers=%d, max_repair=%d",
                total, workers, max_repair)

    # ---- Writers (append mode when resuming) ----
    int_writer = IncrementalWriter(run_dir, "trace_internal", append=append_mode)
    traj_writer = IncrementalWriter(run_dir, "sft_trajectories", append=append_mode)
    atomic_writer = IncrementalWriter(run_dir, "sft_atomic_tasks", append=append_mode)
    fail_writer = IncrementalWriter(run_dir, "failed_samples", append=append_mode)
    stats_writer = IncrementalWriter(run_dir, "search_stats", append=append_mode)

    # ---- Execute ----
    completed = pass_cnt = fail_cnt = 0
    metrics = Metrics()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        if search_mode == "tree":
            from tmcts.search import run_case_tree
            futures = {ex.submit(run_case_tree, s): i for i, s in enumerate(samples)}
        else:
            futures = {ex.submit(run_sample_guided, s, max_repair, 0): i for i, s in enumerate(samples)}

        for fut in as_completed(futures):
            idx = futures.pop(fut)
            sample = samples[idx]
            try:
                result = fut.result()
            except Exception as e:
                logger.error("Sample %d crashed: %s", idx, e)
                completed += 1
                continue

            completed += 1
            qid = sample.get("id", "")

            if search_mode == "tree":
                # ---- T-MCTS tree search result ----
                stats_writer.write(result.stats)
                for v in result.verified:
                    int_writer.write(asdict(v))

                if result.kept is not None:
                    pass_cnt += 1
                    traj_item = export_to_sharegpt(result.kept)
                    if traj_item:
                        traj_item["metadata"]["source"] = "sft_tree_search"
                        traj_writer.write(traj_item)
                    metrics.update(sample, result.kept, [result.kept])
                else:
                    fail_cnt += 1
                    fail_writer.write({
                        "id": sample.get("id", ""),
                        "category": sample.get("category", ""),
                        "gold_tool": sample.get("ground_truth_tool", ""),
                        "level": sample.get("level", 0),
                        "ground_truth": sample.get("ground_truth_answer", ""),
                        "fail_reason": "no_verified_trajectory",
                        "question": sample.get("question", {}),
                        "search_stats": result.stats,
                    })

                if completed % 50 == 0:
                    elapsed = time.time() - t0
                    logger.info("Progress: %d/%d | pass=%d fail=%d | %.1f case/s",
                                completed, len(samples), pass_cnt, fail_cnt,
                                completed / elapsed if elapsed > 0 else 0)
                continue

            success_trace, attempts = result

            # Write internal traces
            for a in attempts:
                int_writer.write(asdict(a))

            if success_trace is not None:
                pass_cnt += 1

                # Full trajectory
                traj_item = export_to_sharegpt(success_trace)
                if traj_item:
                    traj_item["metadata"]["source"] = "sft_original"
                    traj_writer.write(traj_item)

                # Augmented trajectory (repair -> first-try-correct)
                if success_trace.repair_count > 0:
                    try:
                        aug_msgs = augment_trajectory(success_trace, sample)
                        if aug_msgs is not None:
                            at = Trace(
                                sample_id=success_trace.sample_id,
                                category=success_trace.category,
                                level=success_trace.level,
                                split=success_trace.split,
                                gold_tool=success_trace.gold_tool,
                                selected_tool=success_trace.selected_tool,
                                ground_truth=success_trace.ground_truth,
                                final_answer=success_trace.final_answer,
                                success=True, repair_count=0,
                                tool_call_count=success_trace.tool_call_count,
                                tools_used=success_trace.tools_used,
                                attempt_index=0,
                                messages=aug_msgs,
                            )
                            ai = export_to_sharegpt(at)
                            if ai:
                                ai["metadata"]["source"] = "sft_augmented"
                                traj_writer.write(ai)
                    except Exception as e:
                        logger.warning("augment failed %s: %s", qid, e)

                # Single-turn skill items
                try:
                    for it in generate_single_task_items(success_trace, sample):
                        atomic_writer.write(it)
                except Exception as e:
                    logger.warning("single_task failed %s: %s", qid, e)

            else:
                fail_cnt += 1
                fail_writer.write(_build_failed_sample(sample, attempts))
                fr = attempts[-1].fail_reason if attempts else "?"
                logger.info("[%d/%d] FAIL | %s | fail=%s", completed, total, qid[:55], fr)

            metrics.update(sample, success_trace, attempts)

            # Periodic progress
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                acc = pass_cnt / (pass_cnt + fail_cnt) * 100 if (pass_cnt + fail_cnt) > 0 else 0
                logger.info("Progress: %d/%d (%.1f%%) | %.1f case/s | pass=%d fail=%d acc=%.1f%%",
                            completed, total, completed / total * 100, rate,
                            pass_cnt, fail_cnt, acc)

    # ---- Close ----
    int_writer.close()
    traj_writer.close()
    atomic_writer.close()
    fail_writer.close()
    stats_writer.close()

    wall = time.time() - t0

    # ---- Summary ----
    total_done = pass_cnt + fail_cnt
    acc = pass_cnt / total_done * 100 if total_done > 0 else 0
    logger.info("=" * 60)
    logger.info("T-MCTS generation complete (search=%s)", search_mode)
    logger.info("=" * 60)
    logger.info("  Samples:        %d", total_done)
    logger.info("  Pass:           %d (%.1f%%)", pass_cnt, acc)
    logger.info("  Fail:           %d", fail_cnt)
    logger.info("  Wall time:      %.1fs", wall)
    logger.info("  Trajectories:   %d", traj_writer.count)
    logger.info("  Atomic items:   %d", atomic_writer.count)
    logger.info("  Failed samples: %d", fail_writer.count)
    logger.info("  Output:         %s", run_dir)
    logger.info("=" * 60)

    metrics_dict = metrics.to_dict()
    metrics_dict["wall_time_seconds"] = round(wall, 1)
    metrics_dict["search_mode"] = search_mode
    metrics_dict["sft_trajectories"] = traj_writer.count
    metrics_dict["sft_atomic_tasks"] = atomic_writer.count
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)


def _latest_run(output_dir: str) -> str | None:
    """Return the path of the most recent run_* directory under *output_dir*, or None."""
    if not os.path.isdir(output_dir):
        return None
    dirs = sorted(
        d for d in os.listdir(output_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(output_dir, d))
    )
    return os.path.join(output_dir, dirs[-1]) if dirs else None


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="T-MCTS training trajectory generation")
    p.add_argument("--input", required=True, help="Path to input .json or .jsonl (question set)")
    p.add_argument("--output", default="outputs/generate", help="Output root directory")
    p.add_argument("--new", dest="new_run", action="store_true",
                   help="Create a fresh run directory (disable auto-resume)")
    p.add_argument("--workers", type=int, default=64, help="Number of parallel workers")
    p.add_argument("--max_repair", type=int, default=2, help="Max repair attempts per code generation (linear mode)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    p.add_argument("--search", choices=["linear", "tree"], default="linear",
                   help="Generation strategy: linear guided loop, or T-MCTS tree search "
                        "(UCB selection + solver-grounded backpropagation)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for _l in ("httpx", "openai", "httpcore"):
        logging.getLogger(_l).setLevel(logging.WARNING)

    samples = load_samples(args.input)
    logger.info("Loaded %d samples from %s", len(samples), args.input)

    run(
        samples,
        output_dir=args.output,
        workers=args.workers,
        max_repair=args.max_repair,
        new_run=args.new_run,
        limit=args.limit,
        search_mode=args.search,
    )


if __name__ == "__main__":
    main()
