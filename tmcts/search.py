"""
T-MCTS: tool-integrated Monte Carlo tree search.

Solver-guided search over tool-use trajectories (one tree per problem):

  Selection        descend from the root by the UCB rule of Eq. (1)
  Expansion        sample candidate tool calls / answers at the selected
                   node (temperature > 0, deduplicated by (tool, code))
  Tool-grounded    from the new child, continue autonomously — execute the
  Rollout          candidate, feed back solver output, decide next — until
                   an <answer> or the depth budget is reached
  Solver-based     the trajectory return R_T (Eq. 10) is backpropagated
  Backpropagation  through the visited nodes

A trajectory is solver-verified when its final answer is correct and its
final symbolic program executes.  After the search budget is spent, one
random solver-verified trajectory is retained for SFT (keeping diversity
instead of always picking the highest-return path).

The search only consumes the same primitives as the linear pipeline
(build_seed_messages / execute_and_append / call_llm / parse_step_decision),
so trajectories produced here have exactly the autonomous format the SFT
and evaluation code expect.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

from tmcts.llm.client import call_llm, get_pool
from tmcts.llm.parser import parse_step_decision
from tmcts.metrics import _extract_tool_sequence
from tmcts.pipeline import build_seed_messages, execute_and_append
from tmcts.rewards import (
    action_labels,
    approx_tokens,
    is_solver_verified,
    parse_answer,
    trajectory_return,
    RESELECT,
    REVISE,
)
from tmcts.trace import Trace
from tmcts.tree import TreeNode, backup, select_expandable, tree_stats

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    c: float = float(os.getenv("TMCTS_C", "1.0"))                 # exploration (Eq. 1)
    expansion_k: int = int(os.getenv("TMCTS_EXPANSION_K", "3"))   # candidates per expansion
    max_rollouts: int = int(os.getenv("TMCTS_MAX_ROLLOUTS", "16"))
    max_depth: int = int(os.getenv("TMCTS_MAX_DEPTH", "6"))       # assistant turns per trajectory
    runs_per_case: int = int(os.getenv("TMCTS_RUNS_PER_CASE", "3"))
    cand_temp: float = float(os.getenv("TMCTS_CAND_TEMP", "0.9"))


@dataclass
class TreeResult:
    kept: Optional[Trace]                    # randomly retained verified trajectory
    verified: List[Trace] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _sample_candidates(messages: list[dict], endpoint, k: int, temp: float,
                       qid: str) -> List[dict]:
    """Sample k candidate actions for the state *messages*.

    Each candidate keeps the raw response text so the expanded child state
    reproduces the exact conversation the model saw.
    """
    candidates: List[dict] = []
    for j in range(k):
        try:
            text = call_llm(messages, endpoint=endpoint, temperature=temp,
                            question_id=qid, turn_idx=-1)
        except Exception as exc:
            logger.warning("candidate sampling failed for %s: %s", qid, exc)
            continue
        decision = parse_step_decision(text)
        if decision is None:
            continue
        if decision["type"] == "answer":
            candidates.append({"type": "answer", "answer": decision["answer"], "text": text})
        else:
            candidates.append({
                "type": "call",
                "tool_name": decision["tool_name"],
                "code": decision["code"],
                "text": text,
            })
    return candidates


def _rollout(messages: list[dict], action: dict, sample: dict,
             endpoint, cfg: SearchConfig) -> list[dict]:
    """Tool-grounded rollout from an expanded+executed state.

    *messages* ends either with an <answer> candidate (terminal) or with the
    solver feedback of the executed candidate.  The rollout continues
    autonomously (greedy temperature) until an <answer> or the depth budget.
    Rollout turns are not added to the tree — the tree grows one decision
    per iteration through expansion, so revisits deepen it.
    """
    qid = sample.get("id", "")
    messages = list(messages)

    if action["type"] == "answer":
        return messages

    while sum(1 for m in messages if m["role"] == "assistant") < cfg.max_depth:
        try:
            text = call_llm(messages, endpoint=endpoint,
                            question_id=qid, turn_idx=-1)
        except Exception as exc:
            logger.warning("rollout llm call failed for %s: %s", qid, exc)
            break
        messages.append({"role": "assistant", "content": text})
        decision = parse_step_decision(text)
        if decision is None:
            break
        if decision["type"] == "answer":
            break
        execute_and_append(messages, decision["tool_name"], decision["code"])

    return messages


def _to_trace(sample: dict, messages: list[dict]) -> Trace:
    """Wrap a finished trajectory message list into a Trace record."""
    labels = action_labels(messages)
    tools = _extract_tool_sequence(messages)
    return Trace(
        sample_id=sample.get("id", ""),
        category=sample.get("category", ""),
        level=sample.get("level", 0),
        split=sample.get("split", ""),
        gold_tool=sample.get("ground_truth_tool", ""),
        ground_truth=sample.get("ground_truth_answer", ""),
        selected_tool=tools[0] if tools else None,
        final_answer=parse_answer(messages),
        success=is_solver_verified(messages, sample.get("ground_truth_answer", "")),
        repair_count=labels.count(REVISE),
        tool_call_count=len(tools),
        tools_used=[tools[-1]] if tools else [],
        attempt_index=0,
        messages=messages,
    )


def _run_once(sample: dict, endpoint, cfg: SearchConfig,
              verified: List[Trace]) -> dict:
    """One independent tree search run over a problem.

    The tree alternates action nodes (an assistant decision) and execution
    nodes (the deterministic solver feedback of that decision), so the
    search can branch at every decision point along a trajectory.
    """
    qid = sample.get("id", "")
    ground_truth = sample.get("ground_truth_answer", "")

    root = TreeNode(messages=build_seed_messages(sample))
    rollouts = 0

    while rollouts < cfg.max_rollouts:
        node = select_expandable(root, cfg.c)
        if node is None:
            break

        # Expansion is only meaningful at decision points where the model
        # speaks next (root prompt or solver feedback).
        if node.messages[-1].get("role") != "user":
            node.exhausted = True
            continue

        if not node.untried and not node.children:
            for cand in _sample_candidates(node.messages, endpoint,
                                           cfg.expansion_k, cfg.cand_temp, qid):
                node.add_candidate(cand)
        action = node.pop_candidate()
        if action is None:
            node.exhausted = True
            backup(node, 0.0)
            continue

        child = node.expand(action, action["text"])

        # Execute a tool-call candidate and append the execution state as a
        # deterministic child so later iterations can branch on the feedback.
        if action["type"] == "call":
            exec_messages = list(child.messages)
            execute_and_append(exec_messages, action["tool_name"], action["code"])
            exec_node = TreeNode(
                messages=exec_messages,
                parent=child,
                action={"type": "execute", "tool_name": action["tool_name"]},
                depth=child.depth,
            )
            child.children.append(exec_node)
            backup_node = exec_node
        else:
            backup_node = child

        # ---- Tool-grounded rollout ----
        messages = _rollout(backup_node.messages, action, sample, endpoint, cfg)

        # ---- Solver-based backpropagation ----
        ret = trajectory_return(messages, ground_truth)
        backup(backup_node, ret)
        rollouts += 1

        if is_solver_verified(messages, ground_truth):
            verified.append(_to_trace(sample, messages))

    stats = tree_stats(root)
    stats["rollouts"] = rollouts
    stats["verified"] = sum(1 for t in verified)
    return stats


def run_case_tree(sample: dict, cfg: Optional[SearchConfig] = None) -> TreeResult:
    """Run the full T-MCTS search for one problem.

    Performs ``runs_per_case`` independent tree searches, collects all
    solver-verified trajectories and randomly retains one for SFT.
    """
    cfg = cfg or SearchConfig()
    qid = sample.get("id", "")

    pool = get_pool()
    endpoint = pool.pick_endpoint_blocking()
    if not endpoint.acquire(timeout=30):
        raise RuntimeError(f"Failed to acquire slot on {endpoint.name}")

    verified: List[Trace] = []
    run_stats = []
    try:
        for _ in range(cfg.runs_per_case):
            run_stats.append(_run_once(sample, endpoint, cfg, verified))
    finally:
        endpoint.release()

    kept = random.choice(verified) if verified else None
    if kept is not None:
        kept = Trace(**{**kept.__dict__})  # decouple from the verified list

    stats = {
        "id": qid,
        "runs": cfg.runs_per_case,
        "rollouts_total": sum(s["rollouts"] for s in run_stats),
        "verified_total": len(verified),
        "nodes_total": sum(s["nodes"] for s in run_stats),
        "kept_tokens": approx_tokens(kept.messages) if kept else 0,
        "avg_verified_tokens": (
            sum(approx_tokens(t.messages) for t in verified) / len(verified)
            if verified else 0
        ),
    }
    return TreeResult(kept=kept, verified=verified, stats=stats)
