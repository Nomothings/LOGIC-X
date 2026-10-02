"""
Tool registry and sandboxed execution for T-MCTS.

Four declarative symbolic tools:
    prover9  — FOL theorem proving
    z3       — SMT constraint solving (SMT-LIB2)
    pyke     — Datalog rule inference (pure-Python engine)
    minizinc — Finite-domain CSP modeling (MiniZinc)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Binary paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_BIN = os.environ.get("TMCTS_TOOLS_BIN", os.path.join(_REPO_ROOT, "tools_bin"))
_TOOLS_BIN = os.path.abspath(_TOOLS_BIN)

_Z3_BIN = os.path.join(os.path.dirname(sys.executable), "z3")
if not os.path.isfile(_Z3_BIN):
    _Z3_BIN = "z3"

_PROVER9_BIN = os.path.join(_TOOLS_BIN, "prover9", "bin", "prover9")
_MINIZINC_BIN = os.path.join(_TOOLS_BIN, "minizinc", "bin", "minizinc")
_MINIZINC_SHARE = os.path.join(_TOOLS_BIN, "minizinc", "share", "minizinc")
_MINIZINC_SOLVERS = os.path.join(_MINIZINC_SHARE, "solvers")
_MINIZINC_LIB = os.path.join(_TOOLS_BIN, "minizinc", "lib")

_DATALOG_ENGINE = os.path.join(_REPO_ROOT, "tools", "datalog_engine.py")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_NAMES = ["prover9", "z3", "pyke", "minizinc"]

TOOL_REGISTRY: Dict[str, dict] = {
    "prover9": {
        "description": "First-order logic entailment and theorem proving. Used for True/False/Uncertain problems.",
        "hint": "Write prover9 FOL input that solves this problem.",
        "error_markers": ["parse error", "syntax error", "%%ERROR", "segmentation fault", "Fatal error"],
        "success_markers": ["THEOREM PROVED", "SEARCH FAILED", "exhausted", "end of proof", "PROOF-FOUND"],
    },
    "z3": {
        "description": "SMT constraint solving via SMT-LIB2 format. Used for scheduling, ordering, assignment.",
        "hint": "Write SMT-LIB2 input that solves this problem.",
        "error_markers": ["(error", "unsupported", "unknown logic", "Segmentation fault", "invalid"],
        "success_markers": ["sat", "unsat", "unknown", "model"],
    },
    "pyke": {
        "description": "Rule-based inference using Datalog (forward-chaining engine). Used for fact/rule reasoning.",
        "hint": "Write a Datalog program that solves this problem.",
        "error_markers": ["ParseError:", "Error:", "Traceback", "RecursionError", "MemoryError"],
        "success_markers": ["TRUE:", "FALSE:"],
    },
    "minizinc": {
        "description": "Finite-domain constraint modeling and solving. Used for ordering, assignment, combinatorial reasoning.",
        "hint": "Write a MiniZinc model that solves this problem.",
        "error_markers": ["Error:", "error:", "=====ERROR=====", "MiniZinc: evaluation error", "parse error"],
        "success_markers": ["----------", "=========="],
    },
}


def get_tool_hint(tool_name: str) -> str:
    return TOOL_REGISTRY.get(tool_name, {}).get("hint", "")


# ---------------------------------------------------------------------------
# Sandboxed runners
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: list, code: str, suffix: str, timeout: int = 30,
                    env_extra: Optional[Dict[str, str]] = None) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        result = subprocess.run(cmd + [tmp_path], capture_output=True, text=True,
                                timeout=timeout, env=env)
        out, err = result.stdout.strip(), result.stderr.strip()
        parts = [p for p in [out, err] if p]
        return "\n".join(parts) if parts else "(empty output)"
    except subprocess.TimeoutExpired:
        return f"TimeoutError: execution exceeded {timeout}s"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _run_prover9(code: str, timeout: int = 30) -> str:
    if not os.path.isfile(_PROVER9_BIN):
        return f"Prover9Error: binary not found at {_PROVER9_BIN}"
    return _run_subprocess([_PROVER9_BIN, "-f"], code, ".in", timeout=timeout)


def _run_z3(code: str, timeout: int = 30) -> str:
    return _run_subprocess([_Z3_BIN, "-smt2"], code, ".smt2", timeout=timeout)


def _run_pyke(code: str, timeout: int = 30) -> str:
    return _run_subprocess([sys.executable, _DATALOG_ENGINE], code, ".dl", timeout=timeout)


def _run_minizinc(code: str, timeout: int = 30) -> str:
    if not os.path.isfile(_MINIZINC_BIN):
        return f"MiniZincError: binary not found at {_MINIZINC_BIN}"
    env_extra = {
        "MZN_SOLVER_PATH": _MINIZINC_SOLVERS,
        "MZN_STDLIB_DIR": _MINIZINC_SHARE,
        "LD_LIBRARY_PATH": f"{_MINIZINC_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    return _run_subprocess([_MINIZINC_BIN, "--solver", "Gecode"], code, ".mzn",
                           timeout=timeout, env_extra=env_extra)


_RUNNER_MAP = {"prover9": _run_prover9, "z3": _run_z3, "pyke": _run_pyke, "minizinc": _run_minizinc}


def run_tool(tool_name: str, code: str, timeout: int = 30) -> str:
    if tool_name not in _RUNNER_MAP:
        raise ValueError(f"Unknown tool: '{tool_name}'. Available: {TOOL_NAMES}")
    return _RUNNER_MAP[tool_name](code, timeout=timeout)


def is_execution_success(tool_name: str, output: str) -> bool:
    if not output or not output.strip():
        return False
    stripped = output.strip()
    if stripped.startswith(("TimeoutError:", "Prover9Error:", "MiniZincError:")):
        return False
    for marker in TOOL_REGISTRY.get(tool_name, {}).get("error_markers", []):
        if marker.lower() in stripped.lower():
            return False
    return True


# ---------------------------------------------------------------------------
# OpenAI-compatible tool definitions (for export)
# ---------------------------------------------------------------------------

def get_openai_tools() -> list:
    param_descs = {
        "prover9": "FOL formula input for prover9. formulas(assumptions). ... end_of_list.",
        "z3": "SMT-LIB2 input for z3. (set-logic ...), (declare-const ...), (assert ...), (check-sat).",
        "pyke": "Datalog input for pyke. Facts: pred(c1, c2). Rules: head(V) :- body(V). Queries: ?- pred(args).",
        "minizinc": "MiniZinc model for finite-domain CSP. var 1..N: x; constraint x < y; solve satisfy;",
    }
    return [{"type": "function", "function": {
        "name": name, "description": info["description"],
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": param_descs.get(name, info["description"])},
        }, "required": ["code"]},
    }} for name, info in TOOL_REGISTRY.items()]
