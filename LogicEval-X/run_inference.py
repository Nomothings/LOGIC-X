#!/usr/bin/env python3
"""
LogicEval-X evaluation runner (T-MCTS inference + scoring).

Two modes:
    --local MODEL_PATH    Launch vLLM from a local model, then evaluate it.
    --online              Use the API endpoint pool from .env directly.

Autonomous mode — the model decides repair/switch/answer on its own,
exactly one attempt per question.

Usage:
    # Local model (auto-start vLLM on GPUs 0,1,2,3)
    python run_inference.py --input data/logiceval_x.jsonl --output outputs/eval --local ./LOGIC-X-8B

    # Local model on specific GPUs
    python run_inference.py --input data/logiceval_x.jsonl --output outputs/eval --local ./LOGIC-X-8B --gpus 0,1

    # Online API (reads API_* from .env)
    python run_inference.py --input data/logiceval_x.jsonl --output outputs/eval --online

Output:
    {run_dir}/trace_internal.jsonl   per-attempt traces
    {run_dir}/failed_samples.jsonl   failed questions (for analysis)
    {run_dir}/metrics.json           accuracy, tool-selection rates, etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time

# Make the repo-level `tmcts` package importable when run from this folder
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tmcts.data import load_samples
from tmcts.llm.client import init_pool
from tmcts.llm.raw_logger import init_raw_logger
from tmcts.pipeline import run_sample
from tmcts.metrics import Metrics
from tmcts.trace import Trace

logger = logging.getLogger("tmcts.eval")


# ======================================================================
# IncrementalWriter (shared)
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

        os.makedirs(run_dir, exist_ok=True)

        if append:
            existing = sorted(
                f for f in os.listdir(run_dir)
                if f == f"{prefix}.jsonl" or f.startswith(f"{prefix}_")
            )
            if existing:
                last = existing[-1]
                if last != f"{prefix}.jsonl":
                    try:
                        self._file_idx = int(last[len(prefix) + 1:].replace(".jsonl", ""))
                    except ValueError:
                        self._file_idx = 0
                last_path = os.path.join(run_dir, last)
                self._count = sum(1 for _ in open(last_path, encoding="utf-8"))
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
        suffix = "" if self._file_idx == 0 else f"_{self._file_idx:03d}"
        path = os.path.join(self._dir, f"{self._prefix}{suffix}.jsonl")
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
# vLLM helpers (--local mode)
# ======================================================================

class VLLMProcess:
    """Manage a local vLLM server process."""

    def __init__(self, model_path: str, gpus: str = "0,1,2,3", port: int = 8009,
                 max_model_len: int = 16384):
        gpu_list = gpus.split(",")
        tp_size = len(gpu_list)
        workers = min(tp_size * 32, 16)

        self.port = port
        self.gpus = gpus
        self.workers = workers
        self.model_path = model_path
        self.max_model_len = max_model_len
        self._process = None

    def start(self) -> None:
        """Launch vLLM and wait until healthy."""
        # Kill anything already on this port
        self._kill_port()

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--served-model-name", "model",
            "--tensor-parallel-size", str(len(self.gpus.split(","))),
            "--gpu-memory-utilization", "0.70",
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", "32",
            "--max-num-batched-tokens", "8192",
            "--enforce-eager",
            "--trust-remote-code",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpus

        logger.info("Starting vLLM: model=%s GPUs=%s port=%d TP=%d",
                    self.model_path, self.gpus, self.port, len(self.gpus.split(",")))

        vllm_log = open(os.path.join(os.path.dirname(self.model_path), "vllm_eval.log"), "w")
        self._process = subprocess.Popen(
            cmd, stdout=vllm_log, stderr=subprocess.STDOUT,
            env=env, preexec_fn=os.setsid,
        )

        # Wait for health
        import urllib.request
        max_wait = 600
        start = time.time()
        health_url = f"http://localhost:{self.port}/health"
        while time.time() - start < max_wait:
            try:
                resp = urllib.request.urlopen(health_url, timeout=3)
                if resp.status == 200:
                    elapsed = time.time() - start
                    logger.info("vLLM ready (%.0fs)", elapsed)
                    return
            except Exception:
                pass
            time.sleep(3)

        raise RuntimeError(f"vLLM failed to start within {max_wait}s")

    def stop(self) -> None:
        """Kill vLLM and clean up."""
        if self._process is None:
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except Exception:
                pass
        self._kill_port()
        self._process = None

    def _kill_port(self) -> None:
        """Kill anything listening on our port."""
        try:
            import shutil
            if shutil.which("lsof"):
                result = subprocess.run(
                    ["lsof", "-ti", f":{self.port}"],
                    capture_output=True, text=True,
                )
                for pid in result.stdout.strip().split("\n"):
                    if pid:
                        os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass
        time.sleep(1)


# ======================================================================
# Eval runner
# ======================================================================

def run_eval(samples: list[dict],
             *,
             output_dir: str,
             workers: int = 64,
             new_run: bool = False,
             limit: int | None = None,
             vllm: VLLMProcess | None = None):
    """Run evaluation on *samples*.  Autonomous mode, one attempt per question.

    If *vllm* is provided, it is already started and will be shut down on exit.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime
    from dataclasses import asdict

    # ---- Resolve run_dir ----
    completed_qids: set[str] = set()
    append_mode = False

    if new_run or not _latest_run(output_dir):
        run_name = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, run_name)
    else:
        run_dir = _latest_run(output_dir)
        append_mode = True
        for fname in sorted(os.listdir(run_dir)):
            if fname.startswith("trace_internal") and fname.endswith(".jsonl"):
                with open(os.path.join(run_dir, fname)) as fh:
                    for line in fh:
                        try:
                            completed_qids.add(json.loads(line).get("sample_id", ""))
                        except Exception:
                            pass
        logger.info("Auto-resume: %d completed ids from %s", len(completed_qids), run_dir)

    os.makedirs(run_dir, exist_ok=True)

    # ---- Logging ----
    log_handler = logging.FileHandler(os.path.join(run_dir, "run.log"))
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(log_handler)

    # ---- Init API ----
    init_pool()
    init_raw_logger(os.path.join(run_dir, "raw_api_calls.parquet"))

    # Filter already-completed
    if completed_qids:
        samples = [s for s in samples if s.get("id", "") not in completed_qids]
    if limit:
        samples = samples[:limit]
    total = len(samples)

    workers_effective = workers
    if vllm is not None:
        workers_effective = vllm.workers
    logger.info("Eval: %d samples | workers=%d | autonomous | local=%s",
                total, workers_effective, vllm is not None)

    # ---- Writers ----
    int_writer = IncrementalWriter(run_dir, "trace_internal", append=append_mode)
    fail_writer = IncrementalWriter(run_dir, "failed_samples", append=append_mode)

    # ---- Run ----
    completed = pass_cnt = fail_cnt = 0
    metrics = Metrics()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_sample, s, 0): i for i, s in enumerate(samples)}

        for fut in as_completed(futures):
            idx = futures.pop(fut)
            sample = samples[idx]
            try:
                success_trace, all_attempts = fut.result()
            except Exception as exc:
                logger.error("Sample %d crashed: %s", idx, exc)
                completed += 1
                continue

            completed += 1
            qid = sample.get("id", "")

            # Write traces
            for a in all_attempts:
                int_writer.write(asdict(a))

            if success_trace is not None:
                pass_cnt += 1
            else:
                fail_cnt += 1
                if all_attempts:
                    fail_writer.write(_build_failed_sample(sample, all_attempts[-1]))

            metrics.update(sample, success_trace, all_attempts)

            if completed % 20 == 0 or completed == total:
                elapsed = time.time() - t0
                done = pass_cnt + fail_cnt
                acc = pass_cnt / done * 100 if done > 0 else 0
                md = metrics.to_dict()
                ov = md["overall"]
                ts = md["tool_selection"]
                mt = md["multi_turn"]
                rp = md["repair"]
                sw = md["tool_switch"]
                logger.info("[%d/%d] acc=%.1f%% pass=%d fail=%d | tool=%.1f%% | avg_tc=%.1f avg_llm=%.1f | repair=%.1f%% switch=%.1f%% | %.1f/s",
                            completed, total, acc, pass_cnt, fail_cnt,
                            ts["accuracy_vs_gold"] * 100,
                            mt["avg_tool_calls"], mt["avg_llm_turns"],
                            rp["repair_rate"] * 100, sw["switch_rate"] * 100,
                            completed / elapsed if elapsed > 0 else 0)

    # ---- Close ----
    int_writer.close()
    fail_writer.close()
    wall = time.time() - t0

    # ---- Summary ----
    total_done = pass_cnt + fail_cnt
    metrics_dict = metrics.to_dict()
    ov = metrics_dict["overall"]
    ts = metrics_dict["tool_selection"]
    te = metrics_dict["tool_execution"]
    mt = metrics_dict["multi_turn"]
    rp = metrics_dict["repair"]
    sw = metrics_dict["tool_switch"]
    ft = metrics_dict["first_try"]

    logger.info("=" * 60)
    logger.info("EVAL Complete  |  Wall: %.0fs", wall)
    logger.info("=" * 60)
    logger.info("  Samples:  %d  |  Pass: %d (%.1f%%)  |  Fail: %d",
                total_done, ov["pass"], ov["pass_rate"] * 100, ov["fail"])
    logger.info("---")
    logger.info("  Tool selection vs gold:  %.1f%% (%d/%d)",
                ts["accuracy_vs_gold"] * 100, ts["correct"], ts["correct"] + ts["wrong"])
    logger.info("  Tool execution success:  %.1f%% (%d/%d)",
                te["success_rate"] * 100, te["success"], te["total_runs"])
    logger.info("---")
    logger.info("  Avg tool calls:  %.2f  |  Avg LLM turns:  %.2f",
                mt["avg_tool_calls"], mt["avg_llm_turns"])
    logger.info("  Multi-call rate: %.1f%% (%d/%d)",
                mt["multi_call_rate"] * 100, mt["samples_with_multi_call"], total_done)
    logger.info("  First-try rate:  %.1f%% (%d/%d)",
                ft["rate"] * 100, ft["success"], ft["total"])
    logger.info("---")
    logger.info("  Repair rate:    %.1f%% (%d/%d)  |  success: %.1f%% (%d/%d)",
                rp["repair_rate"] * 100, rp["samples_with_repair"], total_done,
                rp["success_rate"] * 100, rp["successes"], rp["total_attempts"])
    logger.info("  Switch rate:    %.1f%% (%d/%d)  |  transitions: %s",
                sw["switch_rate"] * 100, sw["samples_with_switch"], total_done,
                json.dumps(sw["transitions"], ensure_ascii=False))
    logger.info("---")
    for fr, cnt in metrics_dict["fail_reasons"].items():
        logger.info("  fail: %s = %d", fr, cnt)
    logger.info("=" * 60)

    metrics_dict["wall_time_seconds"] = round(wall, 1)
    metrics_dict["completed_samples"] = total_done
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_dict, fh, indent=2, ensure_ascii=False)


def _build_failed_sample(sample: dict, trace: Trace) -> dict:
    return {
        "id": sample.get("id", ""),
        "category": sample.get("category", ""),
        "gold_tool": sample.get("ground_truth_tool", ""),
        "level": sample.get("level", 0),
        "split": sample.get("split", ""),
        "ground_truth": sample.get("ground_truth_answer", ""),
        "fail_reason": getattr(trace, "fail_reason", "?"),
        "question": sample.get("question", {}),
    }


def _latest_run(output_dir: str) -> str | None:
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
    p = argparse.ArgumentParser(description="LogicEval-X evaluation (T-MCTS inference)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", type=str, metavar="MODEL_PATH",
                      help="Evaluate a local model (auto-launch vLLM)")
    mode.add_argument("--online", action="store_true",
                      help="Evaluate using the online API pool from .env")

    p.add_argument("--input", required=True, help="Path to input .json or .jsonl")
    p.add_argument("--output", default="outputs/eval", help="Output root directory")
    p.add_argument("--gpus", default="0,1,2,3", help="GPUs for vLLM (--local only)")
    p.add_argument("--port", type=int, default=8009, help="vLLM port (--local only)")
    p.add_argument("--max-model-len", type=int, default=16384,
                   help="vLLM max model length (--local only)")
    p.add_argument("--new", dest="new_run", action="store_true",
                   help="Fresh run directory (disable auto-resume)")
    p.add_argument("--workers", type=int, default=64,
                   help="Online-mode workers. Local mode auto-calculates from TP size.")
    p.add_argument("--limit", type=int, default=None, help="Limit samples")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    for _l in ("httpx", "openai", "httpcore"):
        logging.getLogger(_l).setLevel(logging.WARNING)

    samples = load_samples(args.input)
    logger.info("Loaded %d samples from %s", len(samples), args.input)

    vllm = None
    try:
        if args.local:
            # ---- Local: start vLLM, then evaluate ----
            if not os.path.isdir(args.local):
                sys.exit(f"Model path not found: {args.local}")

            # Force-override all API endpoints to point to local vLLM.
            # (setdefault is too weak — .env is already loaded at import time.)
            local_base = f"http://localhost:{args.port}/v1"
            tp_size = len(args.gpus.split(","))
            for i in [1, 2, 3]:
                os.environ[f"API_{i}_BASE"] = local_base
                os.environ[f"API_{i}_KEY"] = "local"
                os.environ[f"API_{i}_MODEL"] = "model"
                os.environ[f"API_{i}_WORKER_NUM"] = str(min(tp_size * 32, 16))

            vllm = VLLMProcess(
                model_path=args.local, gpus=args.gpus,
                port=args.port, max_model_len=args.max_model_len,
            )
            vllm.start()
            logger.info("vLLM online at http://localhost:%d/v1 — beginning eval", args.port)

        else:
            # ---- Online: endpoints come from .env ----
            if not os.getenv("API_1_BASE"):
                sys.exit("API_1_BASE not set in .env — cannot use --online")

        run_eval(
            samples,
            output_dir=args.output,
            workers=args.workers,
            new_run=args.new_run,
            limit=args.limit,
            vllm=vllm,
        )

    finally:
        if vllm is not None:
            logger.info("Shutting down vLLM...")
            vllm.stop()


if __name__ == "__main__":
    main()
