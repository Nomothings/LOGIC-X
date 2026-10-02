"""
Solver-grounded rewards for T-MCTS.

Implements the reward model of the paper:

  Step level (Eq. 2):
      r_t = lambda_ans * [t == T] * R_ans
          + (lambda_exec / M) * [t in C(tau)] * v_t
          - beta_switch * [a_t == RESELECT]
          - beta_repair * [a_t == REVISE]

  Trajectory level (Eq. 10):
      R_T(tau) = clip01( lambda_ans * R_ans + lambda_exec * R_exec
                         - beta_switch * n_switch - beta_repair * n_repair )

where
  R_ans    final-answer correctness (parsed <answer> letter vs ground truth)
  R_exec   whether the final symbolic program executes on its solver
  v_t      per-invocation execution validity
  n_switch solver reselection count (RESELECT)
  n_repair feedback-based revision count (REVISE)

All helpers operate on the raw message list so they can be applied both to
finished traces and to partial trajectories inside the search tree.
"""

from __future__ import annotations

import math
import os
import re
from typing import List, Optional

from tmcts.metrics import _extract_tool_outcomes, _extract_tool_sequence

LAMBDA_ANS = float(os.getenv("TMCTS_LAMBDA_ANS", "0.8"))
LAMBDA_EXEC = float(os.getenv("TMCTS_LAMBDA_EXEC", "0.2"))
BETA_SWITCH = float(os.getenv("TMCTS_BETA_SWITCH", "0.05"))
BETA_REPAIR = float(os.getenv("TMCTS_BETA_REPAIR", "0.05"))

_RE_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

# Action labels (paper action space: Formulate / Select / Call / Revise /
# Reselect / Answer).  In this implementation Formulate, Select and Call are
# fused into a single emitted tool call, so a decision is one of:
CALL, REVISE, RESELECT, ANSWER = "call", "revise", "reselect", "answer"


def _assistant_decisions(messages: list[dict]) -> list[str]:
    """Assistant message contents in order."""
    return [m["content"] for m in messages if m.get("role") == "assistant"]


def parse_answer(messages: list[dict]) -> Optional[str]:
    """Extract the final answer letter from the last <answer> block."""
    texts = _assistant_decisions(messages)
    for text in reversed(texts):
        m = _RE_ANSWER.search(text)
        if m:
            inner = m.group(1).strip()
            lm = re.search(r"The final answer is\s+([A-Z])", inner)
            if lm:
                return lm.group(1)
            letters = re.findall(r"\b([A-Z])\b", inner)
            if letters:
                return letters[-1]
    return None


def answer_correct(messages: list[dict], ground_truth: str) -> bool:
    """R_ans: final-answer correctness."""
    ans = parse_answer(messages)
    return ans is not None and ans == ground_truth


def _invocation_validity(messages: list[dict]) -> List[bool]:
    """v_t for each solver invocation, in order (True = executed cleanly)."""
    return [not oc["is_error"] for oc in _extract_tool_outcomes(messages)]


def final_program_executes(messages: list[dict]) -> bool:
    """R_exec: whether the last solver invocation executed without error."""
    flags = _invocation_validity(messages)
    return bool(flags) and flags[-1]


def action_labels(messages: list[dict]) -> List[str]:
    """Label every assistant decision with its action type.

    Rules (consistent with the metrics module's repair/switch accounting):
      - first tool call                       -> CALL
      - tool call retrying the same tool right
        after a failed execution              -> REVISE
      - tool call switching to a new tool      -> RESELECT
      - <answer> turn                         -> ANSWER
    """
    tools = _extract_tool_sequence(messages)
    outcomes = _extract_tool_outcomes(messages)
    labels: List[str] = []
    prev_tool: Optional[str] = None
    for i, tool in enumerate(tools):
        if prev_tool is None:
            labels.append(CALL)
        else:
            prev_ok = outcomes[i - 1]["is_error"] is False if i - 1 < len(outcomes) else False
            if tool == prev_tool and not prev_ok:
                labels.append(REVISE)
            else:
                labels.append(RESELECT)
        prev_tool = tool
    labels.append(ANSWER)  # the terminal decision of a completed trajectory
    return labels


def step_rewards(messages: list[dict], ground_truth: str) -> List[float]:
    """Per-step rewards r_t (Eq. 2), one entry per assistant decision."""
    flags = _invocation_validity(messages)
    n_invocations = len(flags)
    M = max(1, n_invocations)

    decisions = _assistant_decisions(messages)
    tools = _extract_tool_sequence(messages)
    labels = action_labels(messages)

    rewards: List[float] = []
    invocation_idx = 0
    for i in range(len(decisions)):
        r = 0.0
        if labels[i] == ANSWER:
            if answer_correct(messages, ground_truth):
                r += LAMBDA_ANS
        else:
            if invocation_idx < n_invocations:
                v_t = 1.0 if flags[invocation_idx] else 0.0
                r += (LAMBDA_EXEC / M) * v_t
                invocation_idx += 1
        if labels[i] == RESELECT:
            r -= BETA_SWITCH
        if labels[i] == REVISE:
            r -= BETA_REPAIR
        rewards.append(r)
    return rewards


def trajectory_return(messages: list[dict], ground_truth: str) -> float:
    """Solver-grounded trajectory return R_T (Eq. 10)."""
    r_ans = 1.0 if answer_correct(messages, ground_truth) else 0.0
    r_exec = 1.0 if final_program_executes(messages) else 0.0
    labels = action_labels(messages)
    n_switch = labels.count(RESELECT)
    n_repair = labels.count(REVISE)
    raw = (LAMBDA_ANS * r_ans + LAMBDA_EXEC * r_exec
           - BETA_SWITCH * n_switch - BETA_REPAIR * n_repair)
    return min(1.0, max(0.0, raw))


def is_solver_verified(messages: list[dict], ground_truth: str) -> bool:
    """A trajectory is solver-verified when the final answer is correct and
    the final symbolic program executes on its solver."""
    return answer_correct(messages, ground_truth) and final_program_executes(messages)


def approx_tokens(messages: list[dict]) -> int:
    """Rough token count (characters / 4) for budget statistics."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return int(math.ceil(chars / 4))
