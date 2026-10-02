#!/usr/bin/env python3
"""Behavioural tests for the T-MCTS tree search (stub LLM + real z3).

Run:  python3 T-MCTS/tests/test_search.py   (from the repo root)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import types
if "openai" not in sys.modules:
    m = types.ModuleType("openai"); m.OpenAI = object; sys.modules["openai"] = m
if "httpx" not in sys.modules:
    m = types.ModuleType("httpx"); m.Client = lambda **kw: None; sys.modules["httpx"] = m

from tmcts import search as search_mod
from tmcts.search import SearchConfig, run_case_tree
from tmcts.tree import TreeNode, action_key, backup, ucb_select

GOOD_SMT = "(set-logic QF_LIA)\n(declare-const a Int)\n(assert (> a 5))\n(check-sat)\n"
BAD_SMT = "(set-logic QF_LIA)\n(assert (> a 5)\n"  # unbalanced -> z3 error


def _tc_text(code):
    return ('<think>\nformalize\n</think>\n<tool_call>\n'
            + json.dumps({"name": "z3", "arguments": {"code": code}})
            + '\n</tool_call>')


class FakeEndpoint:
    def acquire(self, timeout=None):
        return True

    def release(self):
        pass


class FakePool:
    def pick_endpoint_blocking(self):
        return FakeEndpoint()


def install_stub_llm(candidate_texts, rollout_answer="B"):
    """Candidate sampling returns scripted texts; rollout turns always answer."""
    counter = {"i": 0}

    def fake_llm(messages, **kw):
        last = messages[-1]
        if last.get("role") == "user" and "<tool_response>" in last.get("content", ""):
            return f"<answer>\nThe final answer is {rollout_answer}\n</answer>"
        idx = min(counter["i"], len(candidate_texts) - 1)
        counter["i"] += 1
        return candidate_texts[idx]

    search_mod.call_llm = fake_llm
    search_mod.get_pool = lambda: FakePool()
    return counter


def test_tree_ucb_units():
    root = TreeNode(messages=[])
    a = TreeNode(messages=[], parent=root, action={"type": "answer", "answer": "A"})
    b = TreeNode(messages=[], parent=root, action={"type": "answer", "answer": "B"})
    root.children = [a, b]
    backup(a, 1.0)
    backup(a, 1.0)
    # b is unvisited -> infinite UCB -> selected first (root.n_visits=2 via backup)
    assert ucb_select(root, 1.0) is b
    backup(b, 0.0)  # root.n_visits=3, b: n=1, q=0
    # a (q=1.0) now wins: 1.0 + sqrt(ln3/2) > 0.0 + sqrt(ln3/1)
    assert ucb_select(root, 1.0) is a
    assert action_key({"type": "answer", "answer": "A"}) == "answer::A"
    print("[ok] tree: UCB selection and action keys")


def test_search_finds_verified_and_backprops():
    install_stub_llm([_tc_text(BAD_SMT), _tc_text(GOOD_SMT), _tc_text(GOOD_SMT)])
    sample = {
        "id": "logiceval-9001",
        "question": {"context": "a is greater than 5.", "question": "Which is true?",
                     "options": ["A) a<=5", "B) a>5"]},
        "ground_truth_tool": "z3",
        "ground_truth_answer": "B",
    }
    cfg = SearchConfig(runs_per_case=1, max_rollouts=4, expansion_k=2, max_depth=3, c=1.0)
    result = run_case_tree(sample, cfg)

    # bad branch: z3 error -> answer B (correct answer, broken program, not verified)
    # good branch: sat -> answer B -> verified with R_T = 1.0
    assert result.stats["verified_total"] >= 1, result.stats
    assert result.kept is not None
    assert result.kept.success is True
    assert result.kept.final_answer == "B"
    # exactly one tool call in the kept autonomous trajectory
    calls = [m for m in result.kept.messages
             if m["role"] == "assistant" and "<tool_call>" in m["content"]]
    assert len(calls) == 1, len(calls)
    assert "sat" in result.kept.messages[-2]["content"]
    # search statistics recorded
    assert result.stats["rollouts_total"] >= 2, result.stats
    assert result.stats["nodes_total"] >= 3, result.stats
    assert result.stats["kept_tokens"] > 0
    print("[ok] search: verified trajectory found, kept autonomous-format, stats emitted:",
          result.stats)


def test_search_budget_and_dedup():
    # third candidate duplicates the good one -> must not create a third child
    install_stub_llm([_tc_text(BAD_SMT), _tc_text(GOOD_SMT), _tc_text(GOOD_SMT),
                      _tc_text(GOOD_SMT), _tc_text(GOOD_SMT)])
    sample = {
        "id": "logiceval-9002",
        "question": {"context": "a is greater than 5.", "question": "Which is true?",
                     "options": ["A) a<=5", "B) a>5"]},
        "ground_truth_tool": "z3",
        "ground_truth_answer": "B",
    }
    cfg = SearchConfig(runs_per_case=1, max_rollouts=2, expansion_k=2, max_depth=3, c=1.0)
    result = run_case_tree(sample, cfg)
    assert result.stats["rollouts_total"] == 2, result.stats
    # dedup check at node level
    node = TreeNode(messages=[])
    node.add_candidate({"type": "answer", "answer": "A"})
    node.add_candidate({"type": "answer", "answer": "A"})
    assert len(node.untried) == 1
    print("[ok] search: rollout budget respected, duplicate candidates deduped")


def test_search_depth_cap():
    install_stub_llm([_tc_text(GOOD_SMT)])
    sample = {
        "id": "logiceval-9003",
        "question": {"context": "a is greater than 5.", "question": "Which is true?",
                     "options": ["A) a<=5", "B) a>5"]},
        "ground_truth_tool": "z3",
        "ground_truth_answer": "B",
    }
    # depth 2 = one tool call + one answer turn; the rollout must not exceed it
    cfg = SearchConfig(runs_per_case=1, max_rollouts=1, expansion_k=1, max_depth=2, c=1.0)
    result = run_case_tree(sample, cfg)
    assert result.stats["verified_total"] == 1, result.stats
    n_asst = sum(1 for m in result.verified[0].messages if m["role"] == "assistant")
    assert n_asst == 2, n_asst
    print("[ok] search: depth cap honoured")


def test_no_verified():
    # only broken programs and a wrong rollout answer -> nothing verified, kept=None
    install_stub_llm([_tc_text(BAD_SMT)], rollout_answer="A")
    sample = {
        "id": "logiceval-9004",
        "question": {"context": "a is greater than 5.", "question": "Which is true?",
                     "options": ["A) a<=5", "B) a>5"]},
        "ground_truth_tool": "z3",
        "ground_truth_answer": "B",
    }
    cfg = SearchConfig(runs_per_case=1, max_rollouts=2, expansion_k=1, max_depth=3, c=1.0)
    result = run_case_tree(sample, cfg)
    assert result.kept is None
    assert result.stats["verified_total"] == 0
    print("[ok] search: no verified trajectory -> kept=None")


if __name__ == "__main__":
    test_tree_ucb_units()
    test_search_finds_verified_and_backprops()
    test_search_budget_and_dedup()
    test_search_depth_cap()
    test_no_verified()
    print("ALL SEARCH TESTS PASSED")
