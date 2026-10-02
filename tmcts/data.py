"""Data loader and task formatter for T-MCTS.

Sample schema (JSONL, one object per line):
    {
        "id": "logiceval-0001",
        "question": {"context": str, "question": str, "options": [str, ...]},
        "ground_truth_tool": "z3",
        "ground_truth_answer": "B",
        "category": "optional grouping label",
        "level": 0,
        "split": "optional split label"
    }
"""

from __future__ import annotations

import gzip
import json
import os
from typing import List

VALID_TOOLS = {"prover9", "z3", "pyke", "minizinc"}


def _normalize_split(raw_split: str) -> str:
    clean = raw_split.lower().replace(".json", "").replace(".jsonl", "").strip()
    mapping = {"validation": "dev", "val": "dev", "eval": "dev"}
    return mapping.get(clean, clean)


def load_samples(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # Transparent gzip support for *.jsonl.gz files
    opener = gzip.open if path.endswith(".gz") else open
    root, ext = os.path.splitext(path[:-3] if path.endswith(".gz") else path)
    ext = ext.lower()
    if ext == ".json":
        with opener(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        samples = raw if isinstance(raw, list) else [raw]
    elif ext == ".jsonl":
        samples = []
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file extension: {ext!r}")

    for s in samples:
        if "split" in s:
            s["split"] = _normalize_split(s["split"])

    return samples


# ---------------------------------------------------------------------------
# Task input formatter
# ---------------------------------------------------------------------------

def _format_options(options: list) -> str:
    return "\n".join(options)


def build_task_input(sample: dict) -> str:
    from tmcts.prompts import PROMPTS

    question: dict = sample.get("question", {})
    context = question.get("context", "")
    qtext = question.get("question", "")
    options: list = question.get("options", [])

    context_block = f"Context:\n{context.strip()}\n" if (context and context.strip()) else ""
    formatted = _format_options(options)

    task = PROMPTS["TASK_INPUT"]
    task = task.replace("%{CONTEXT_BLOCK}", context_block)
    task = task.replace("%{QUESTION}", qtext or "")
    task = task.replace("%{OPTIONS}", formatted)
    return task
