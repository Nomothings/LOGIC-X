#!/usr/bin/env python3
"""Unit tests for tmcts.rewards (Eq. 2 / Eq. 10).

Run:  python3 T-MCTS/tests/test_rewards.py   (from the repo root)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Stub third-party deps when running in a bare environment
import types
if "openai" not in sys.modules:
    m = types.ModuleType("openai"); m.OpenAI = object; sys.modules["openai"] = m
if "httpx" not in sys.modules:
    m = types.ModuleType("httpx"); m.Client = lambda **kw: None; sys.modules["httpx"] = m

from tmcts import rewards
from tmcts.pipeline import build_seed_messages, execute_and_append

GOOD_SMT = "(set-logic QF_LIA)\n(declare-const a Int)\n(assert (> a 5))\n(check-sat)\n"
BAD_SMT = "(set-logic QF_LIA)\n(assert (> a 5)\n"  # unbalanced -> z3 error

SYS = {"role": "system", "content": "sys"}
USER = {"role": "user", "content": "Solve the following problem..."}


def tc(name, code):
    return {"role": "assistant",
            "content": f"<think>\nreason\n</think>\n<tool_call>\n"
                       + json.dumps({"name": name, "arguments": {"code": code}})
                       + "\n</tool_call>"}


def resp(output):
    return {"role": "user", "content": f"<tool_response>\n{output}\n</tool_response>"}


def ans(letter):
    return {"role": "assistant",
            "content": f"<answer>\nsolved.\nThe final answer is {letter}\n</answer>"}


def test_first_try_success():
    msgs = [SYS, USER, tc("z3", GOOD_SMT), resp("sat\n(a = 6)"), ans("B")]
    assert rewards.action_labels(msgs) == ["call", "answer"], rewards.action_labels(msgs)
    assert rewards.trajectory_return(msgs, "B") == 1.0          # 0.8 + 0.2
    assert rewards.is_solver_verified(msgs, "B") is True
    steps = rewards.step_rewards(msgs, "B")
    assert abs(sum(steps) - (0.8 + 0.2)) < 1e-9, steps
    print("[ok] first-try success: R_T=1.0, verified")


def test_repair():
    msgs = [SYS, USER, tc("z3", BAD_SMT), resp("(error line 3 column 2.)"),
            tc("z3", GOOD_SMT), resp("sat"), ans("B")]
    labels = rewards.action_labels(msgs)
    assert labels == ["call", "revise", "answer"], labels
    rt = rewards.trajectory_return(msgs, "B")                   # 0.8 + 0.2 - 0.05
    assert abs(rt - 0.95) < 1e-9, rt
    steps = rewards.step_rewards(msgs, "B")                     # Eq. 2: 0.8 + 0.2/2 - 0.05
    assert abs(sum(steps) - (0.8 + 0.1 - 0.05)) < 1e-9, steps
    assert rewards.is_solver_verified(msgs, "B") is True
    print("[ok] repair: R_T=0.95, labels=call/revise/answer")


def test_switch():
    msgs = [SYS, USER, tc("z3", BAD_SMT), resp("(error ...)"),
            tc("pyke", "p(a). ?- p(a)."), resp("TRUE: p(a)"), ans("B")]
    labels = rewards.action_labels(msgs)
    assert labels == ["call", "reselect", "answer"], labels
    rt = rewards.trajectory_return(msgs, "B")
    assert abs(rt - 0.95) < 1e-9, rt
    print("[ok] switch: R_T=0.95, labels=call/reselect/answer")


def test_wrong_answer():
    msgs = [SYS, USER, tc("z3", GOOD_SMT), resp("sat"), ans("A")]
    rt = rewards.trajectory_return(msgs, "B")                   # 0 + 0.2
    assert abs(rt - 0.2) < 1e-9, rt
    assert rewards.is_solver_verified(msgs, "B") is False
    print("[ok] wrong answer: R_T=0.2, not verified")


def test_answer_only():
    msgs = [SYS, USER, ans("B")]
    rt = rewards.trajectory_return(msgs, "B")                   # R_ans only, no invocation
    assert abs(rt - 0.8) < 1e-9, rt
    assert rewards.is_solver_verified(msgs, "B") is False       # no executable program
    print("[ok] answer-only: R_T=0.8 but NOT verified (no tool program)")


def test_real_execution_roundtrip():
    """Execute real z3 via the sandbox and check the validity flags."""
    sample = {"id": "t", "question": {"context": "a > 5", "question": "q?",
                                      "options": ["A) x", "B) y"]}}
    seed = build_seed_messages(sample)
    out_ok = execute_and_append(seed, "z3", GOOD_SMT)
    assert "sat" in out_ok
    seed.append({"role": "assistant", "content": "ans-placeholder"})
    assert rewards.final_program_executes(seed) is True

    msgs2 = [SYS, USER, tc("z3", BAD_SMT)]
    out_bad = execute_and_append(msgs2, "z3", BAD_SMT)
    assert "(error" in out_bad
    assert rewards.final_program_executes(msgs2) is False
    print("[ok] real z3 execution: valid/invalid programs detected")


def test_clip():
    msgs = [SYS, USER, tc("z3", BAD_SMT), resp("(error 1)"),
            tc("pyke", "?- x."), resp("(error 2)"),
            tc("z3", BAD_SMT), resp("(error 3)"), ans("A")]
    rt = rewards.trajectory_return(msgs, "B")                   # 0 + 0 - 0.15 -> clipped to 0
    assert rt == 0.0, rt
    print("[ok] clipping: negative raw return clipped to 0")


if __name__ == "__main__":
    test_first_try_success()
    test_repair()
    test_switch()
    test_wrong_answer()
    test_answer_only()
    test_real_execution_roundtrip()
    test_clip()
    print("ALL REWARDS TESTS PASSED")
