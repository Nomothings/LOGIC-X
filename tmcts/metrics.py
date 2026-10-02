"""Metrics accumulator for T-MCTS evaluation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

_RE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_RE_TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_RE_ERROR = re.compile(
    r"(?i)\b(error|unsupported|syntax error|parse error|type error|"
    r"segmentation fault|bus error|aborted|killed|%%ERROR|"
    r"wrong number of arguments|unbound variable|undefined|not recognized|invalid)\b"
)


def _extract_tool_sequence(messages: list) -> list[str]:
    """Extract ordered list of tool names from assistant messages."""
    tools = []
    for m in messages:
        if m.get("role") == "assistant":
            tc = _RE_TOOL_CALL.search(m.get("content", ""))
            if tc:
                try:
                    import json
                    tools.append(json.loads(tc.group(1).strip()).get("name", "?"))
                except Exception:
                    tools.append("?")
    return tools


def _extract_tool_outcomes(messages: list) -> list[dict]:
    """Extract each tool execution's result from <tool_response> messages.

    Returns list of {output_preview, is_error, tool_call_idx}.
    """
    outcomes = []
    tc_idx = 0  # which tool call this outcome corresponds to (1-indexed)
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")

        if role == "assistant" and _RE_TOOL_CALL.search(content):
            tc_idx += 1

        if role == "user" and "<tool_response>" in content:
            tr = _RE_TOOL_RESPONSE.search(content)
            output = tr.group(1).strip() if tr else content
            outcomes.append({
                "tool_call_idx": tc_idx,
                "is_error": bool(_RE_ERROR.search(output)),
                "output_preview": output[:200],
                "output_len": len(output),
            })

    return outcomes


class Metrics:
    def __init__(self):
        # Core counts
        self.pass_cnt = 0
        self.fail_cnt = 0
        self.fail_reasons: Dict[str, int] = {}

        # Accumulators for averages (computed over all samples)
        self.total_tool_calls = 0
        self.total_llm_turns = 0
        self.total_tool_executions = 0   # total tool runs (may exceed tool_calls in multi-turn)

        # Tool selection accuracy vs ground_truth_tool
        self.tool_correct = 0
        self.tool_wrong = 0
        self._tool_right_dist: Dict[str, int] = {}    # per-tool correct counts
        self._tool_wrong_dist: Dict[str, int] = {}    # per-tool wrong counts
        self.tool_selected_dist: Dict[str, int] = {}  # what the model chose
        self.tool_gold_dist: Dict[str, int] = {}      # what the gold label says

        # Repair / switch stats
        self.repair_attempts = 0    # error -> same tool retry
        self.repair_successes = 0   # error -> same tool retry -> later success
        self.tool_switches = 0      # tool A -> tool B (different)
        self.samples_with_repair = 0
        self.samples_with_switch = 0
        self.samples_with_multi_call = 0  # 2+ tool calls

        # Tool execution reliability
        self.tool_exec_success = 0   # tool ran without error
        self.tool_exec_fail = 0      # tool returned error
        self.tool_exec_success_dist: Dict[str, int] = {}
        self.tool_exec_fail_dist: Dict[str, int] = {}

        # Tool-switch matrix: (from_tool, to_tool) -> count
        self.tool_transitions: Dict[str, int] = {}

        # First-try success: model picks right tool + code runs on first attempt
        self.first_try_success = 0   # only 1 tool call and it succeeded
        self.first_try_total = 0     # all samples that made at least 1 tool call

    # ------------------------------------------------------------------
    def update(self, sample: dict, success_trace, all_attempts: list):
        """Feed one sample result into the accumulator."""
        trace = success_trace if success_trace is not None else (
            all_attempts[-1] if all_attempts else None
        )
        if trace is None:
            return

        is_pass = success_trace is not None
        if is_pass:
            self.pass_cnt += 1
        else:
            self.fail_cnt += 1
            fr = getattr(trace, "fail_reason", "unknown") or "unknown"
            self.fail_reasons[fr] = self.fail_reasons.get(fr, 0) + 1

        messages = getattr(trace, "messages", [])
        tools = _extract_tool_sequence(messages)
        outcomes = _extract_tool_outcomes(messages)

        n_tool_calls = len(tools)
        n_llm_turns = sum(1 for m in messages if m.get("role") == "assistant")
        n_executions = len(outcomes)

        self.total_tool_calls += n_tool_calls
        self.total_llm_turns += n_llm_turns
        self.total_tool_executions += n_executions

        gold_tool = sample.get("ground_truth_tool", "") or getattr(trace, "gold_tool", "") or ""

        # ---- Tool selection accuracy ----
        first_tool = tools[0] if tools else None
        if first_tool:
            self.tool_selected_dist[first_tool] = self.tool_selected_dist.get(first_tool, 0) + 1
        if gold_tool:
            self.tool_gold_dist[gold_tool] = self.tool_gold_dist.get(gold_tool, 0) + 1
            if first_tool == gold_tool:
                self.tool_correct += 1
                self._tool_right_dist[gold_tool] = self._tool_right_dist.get(gold_tool, 0) + 1
            elif first_tool:
                self.tool_wrong += 1
                self._tool_wrong_dist[first_tool] = self._tool_wrong_dist.get(first_tool, 0) + 1

        # ---- Multi-call ----
        if n_tool_calls >= 2:
            self.samples_with_multi_call += 1

        # ---- Tool transitions ----
        prev_tool = None
        has_switch = False
        for t in tools:
            if prev_tool and t != prev_tool:
                self.tool_switches += 1
                has_switch = True
                key = f"{prev_tool} -> {t}"
                self.tool_transitions[key] = self.tool_transitions.get(key, 0) + 1
            prev_tool = t
        if has_switch:
            self.samples_with_switch += 1

        # ---- Repair tracking ----
        # A repair = same tool called again after the previous call errored
        has_repair = False
        repair_ok = False
        for i, tool in enumerate(tools):
            if i == 0:
                continue
            # Find the outcome for the previous tool call
            prev_outcome = outcomes[i - 1] if i - 1 < len(outcomes) else None
            if prev_outcome and prev_outcome["is_error"] and tool == tools[i - 1]:
                self.repair_attempts += 1
                has_repair = True
                # Was this repair successful? (check if this tool call's outcome is OK)
                this_outcome = outcomes[i] if i < len(outcomes) else None
                if this_outcome and not this_outcome["is_error"]:
                    repair_ok = True
        if has_repair:
            self.samples_with_repair += 1
        if repair_ok:
            self.repair_successes += 1

        # ---- Tool execution reliability ----
        for oc in outcomes:
            if oc["is_error"]:
                self.tool_exec_fail += 1
            else:
                self.tool_exec_success += 1
            # Per-tool breakdown (use the tool_call associated with this outcome)
            tc_idx = oc["tool_call_idx"] - 1  # 0-indexed
            if 0 <= tc_idx < len(tools):
                tool_name = tools[tc_idx]
                if oc["is_error"]:
                    self.tool_exec_fail_dist[tool_name] = self.tool_exec_fail_dist.get(tool_name, 0) + 1
                else:
                    self.tool_exec_success_dist[tool_name] = self.tool_exec_success_dist.get(tool_name, 0) + 1

        # ---- First-try success ----
        if n_tool_calls >= 1:
            self.first_try_total += 1
            if n_tool_calls == 1 and is_pass:
                self.first_try_success += 1

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        total = self.pass_cnt + self.fail_cnt
        n = total if total > 0 else 1

        # Per-tool accuracy
        tool_accuracy_by_tool = {}
        all_tools = set(self._tool_right_dist) | set(self._tool_wrong_dist) | set(self.tool_selected_dist) | set(self.tool_gold_dist)
        for tool_name in sorted(all_tools):
            right = self._tool_right_dist.get(tool_name, 0)
            wrong = self._tool_wrong_dist.get(tool_name, 0)
            tool_accuracy_by_tool[tool_name] = {
                "selected_as_first": self.tool_selected_dist.get(tool_name, 0),
                "gold_labeled": self.tool_gold_dist.get(tool_name, 0),
                "correct_when_selected": right,
                "wrong_when_selected": wrong,
            }

        # Actual tool accuracy from our counter
        tool_sel_total = self.tool_correct + self.tool_wrong
        tool_sel_acc = self.tool_correct / tool_sel_total if tool_sel_total > 0 else 0.0

        # Per-tool execution success rate
        tool_exec_stats = {}
        for tool_name in set(list(self.tool_exec_success_dist) + list(self.tool_exec_fail_dist)):
            s = self.tool_exec_success_dist.get(tool_name, 0)
            f = self.tool_exec_fail_dist.get(tool_name, 0)
            tot = s + f
            tool_exec_stats[tool_name] = {
                "executions": tot,
                "success": s,
                "fail": f,
                "success_rate": s / tot if tot > 0 else 0.0,
            }

        return {
            "overall": {
                "completed": total,
                "pass": self.pass_cnt,
                "fail": self.fail_cnt,
                "pass_rate": self.pass_cnt / n if total > 0 else 0.0,
            },
            "tool_selection": {
                "accuracy_vs_gold": round(tool_sel_acc, 4),
                "correct": self.tool_correct,
                "wrong": self.tool_wrong,
                "selected_distribution": dict(sorted(
                    self.tool_selected_dist.items(), key=lambda x: -x[1]
                )),
                "gold_distribution": dict(sorted(
                    self.tool_gold_dist.items(), key=lambda x: -x[1]
                )),
                "by_tool": tool_accuracy_by_tool,
            },
            "tool_execution": {
                "total_runs": self.total_tool_executions,
                "success": self.tool_exec_success,
                "fail": self.tool_exec_fail,
                "success_rate": self.tool_exec_success / max(1, self.total_tool_executions),
                "by_tool": tool_exec_stats,
            },
            "multi_turn": {
                "avg_tool_calls": self.total_tool_calls / n,
                "avg_llm_turns": self.total_llm_turns / n,
                "samples_with_multi_call": self.samples_with_multi_call,
                "multi_call_rate": self.samples_with_multi_call / n,
            },
            "repair": {
                "total_attempts": self.repair_attempts,
                "successes": self.repair_successes,
                "success_rate": self.repair_successes / max(1, self.repair_attempts),
                "samples_with_repair": self.samples_with_repair,
                "repair_rate": self.samples_with_repair / n,
            },
            "tool_switch": {
                "total_switches": self.tool_switches,
                "samples_with_switch": self.samples_with_switch,
                "switch_rate": self.samples_with_switch / n,
                "transitions": dict(sorted(
                    self.tool_transitions.items(), key=lambda x: -x[1]
                )),
            },
            "first_try": {
                "success": self.first_try_success,
                "total": self.first_try_total,
                "rate": self.first_try_success / max(1, self.first_try_total),
            },
            "fail_reasons": dict(sorted(
                self.fail_reasons.items(), key=lambda x: -x[1]
            )),
        }
