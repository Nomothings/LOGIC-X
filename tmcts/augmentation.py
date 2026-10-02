"""
Data augmentation for SFT training.

1. augment_trajectory() — an LLM rewrites a repair trace into a
   first-try-correct trace
2. generate_single_task_items() — splits a successful trace into
   single-turn QA items (tool selection / code generation / code repair)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from tmcts.data import build_task_input
from tmcts.llm.parser import parse_tool_call as validate_tool_call
from tmcts.prompts import PROMPTS
from tmcts.tools import TOOL_NAMES, TOOL_REGISTRY, get_openai_tools, get_tool_hint
from tmcts.trace import Trace

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ======================================================================
# Part 1: augment_trajectory (LLM rewrite repair -> first-try-correct)
# ======================================================================

_AUGMENT_PROMPT = (
    "You are a data augmentation analyst for symbolic reasoning training.\n"
    "\n"
    "Below is a trajectory where an LLM wrote buggy code, execution failed,\n"
    "and the LLM then REPAIRED it with correct code.  Your job: rewrite the\n"
    "INITIAL code-generation response so it reads as if the LLM produced\n"
    "the CORRECT code on its very first attempt — no repair ever happened.\n"
    "\n"
    "REWRITING RULES:\n"
    "- Preserve the ORIGINAL reasoning style, length, and verbosity.\n"
    "- Remove ALL language that references a previous error, fix, or "
    "correction (e.g. \"I forgot\", \"the error was\", \"adding missing\", "
    "\"fixing the syntax\", \"repairing\", \"corrected version\").\n"
    "- If the original reasoning describes a WRONG formalisation, replace it\n"
    "  with reasoning that describes the CORRECT formalisation as if that\n"
    "  was always the plan.\n"
    "- Keep the same tool choice.  Do NOT change the tool name.\n"
    "- Use the CORRECT code block EXACTLY as provided below — do NOT alter "
    "a single character of the code.\n"
    "- If the original response was already correct before the rewrite, your\n"
    "  output should be identical to the original.\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "Output ONLY the rewritten response — no commentary, no markdown fences.\n"
    "End exactly with:\n"
    "\n"
    "<tool_call>TOOL_NAME</tool_call>\n"
    "<code>\n"
    "... the corrected code ...\n"
    "</code>\n"
    "\n"
    "# Problem\n"
    "Question: {question}\n"
    "Options: {options}\n"
    "\n"
    "# Original (buggy) code-generation response\n"
    "{original_response}\n"
    "\n"
    "# Error output from the failed execution\n"
    "{error_output}\n"
    "\n"
    "# CORRECT code — COPY THIS EXACTLY, DO NOT MODIFY:\n"
    "{correct_code}\n"
    "\n"
    "# Rewritten first-attempt response:\n"
)


def _find_repair_segments(messages: list[dict]) -> Tuple[int, int, int, int] | None:
    """Locate (code_gen_idx, repair_user_idx, repair_resp_idx, result_user_idx)."""
    _REPAIR_SENTINEL = "The previous tool execution failed."
    _INTERP_SENTINEL = "Based on the latest tool output"

    # Find last repair assistant
    repair_resp_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] != "assistant":
            continue
        tc = _TOOL_CALL_RE.search(messages[i].get("content", ""))
        if tc is None:
            continue
        for j in range(i - 1, -1, -1):
            if messages[j]["role"] == "user" and messages[j].get("content", "").startswith(_REPAIR_SENTINEL):
                repair_resp_idx = i
                break
        if repair_resp_idx is not None:
            break
    if repair_resp_idx is None:
        return None

    # Repair user
    repair_user_idx = None
    for i in range(repair_resp_idx - 1, -1, -1):
        if messages[i]["role"] == "user" and messages[i].get("content", "").startswith(_REPAIR_SENTINEL):
            repair_user_idx = i
            break
    if repair_user_idx is None:
        return None

    # Buggy code-gen assistant — the FIRST assistant with a tool_call
    # (walking forward from 0, not backward from repair position, because
    # multi-repair traces have multiple repair assistants between the
    # original code-gen and the final repair).
    code_gen_idx = None
    for i in range(len(messages)):
        if messages[i]["role"] == "assistant" and _TOOL_CALL_RE.search(messages[i].get("content", "")):
            code_gen_idx = i
            break
    if code_gen_idx is None:
        return None

    # Result interpretation user
    result_user_idx = None
    for i in range(repair_resp_idx + 1, len(messages)):
        if messages[i]["role"] == "user" and messages[i].get("content", "").startswith(_INTERP_SENTINEL):
            result_user_idx = i
            break
    if result_user_idx is None:
        return None

    return (code_gen_idx, repair_user_idx, repair_resp_idx, result_user_idx)


def augment_trajectory(trace: Trace, sample: dict) -> list[dict] | None:
    """LLM rewrite: repair trajectory -> first-try-correct trajectory.

    Returns a new messages list, or None on failure.
    """
    if trace.repair_count <= 0 or not trace.success or not trace.messages:
        return None

    segments = _find_repair_segments(trace.messages)
    if segments is None:
        return None

    code_gen_idx, repair_user_idx, repair_resp_idx, result_user_idx = segments
    messages = trace.messages

    from tmcts.llm.parser import parse_tool_call

    _ERR_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

    # Guard: only augment when the repair actually produced working code.
    # If the code after repair still failed, the trace succeeded only
    # because the LLM reasoned past the error — the "correct code" does
    # not exist and augmenting would produce a garbage first-try-correct
    # sample.
    from tmcts.tools import is_execution_success as _exec_ok
    interp_content = messages[result_user_idx].get("content", "")
    output_match = _ERR_RE.search(interp_content)
    if output_match:
        last_tool_output = output_match.group(1)
        repair_tool = parse_tool_call(messages[repair_resp_idx].get("content", ""))
        if repair_tool and not _exec_ok(repair_tool["tool_name"], last_tool_output):
            logger.info("augment: repair code still fails for %s — skipping", trace.sample_id)
            return None

    # Build prompt
    question: dict = sample.get("question", {}) if isinstance(sample, dict) else {}
    qtext = question.get("question", "")
    options = "\n".join(question.get("options", []))

    repair_user_content = messages[repair_user_idx].get("content", "")
    err_match = _ERR_RE.search(repair_user_content)
    error_output = err_match.group(0) if err_match else repair_user_content

    # Extract the final correct code from the repair response
    repair_tc = parse_tool_call(messages[repair_resp_idx].get("content", ""))
    correct_code = repair_tc["code"] if repair_tc else "(unable to extract)"
    # Escape braces for str.format() (MiniZinc uses {} in set/array literals)
    correct_code_safe = correct_code.replace("{", "{{").replace("}", "}}")

    prompt = _AUGMENT_PROMPT.format(
        question=qtext,
        options=options,
        original_response=messages[code_gen_idx].get("content", ""),
        error_output=error_output,
        correct_code=correct_code_safe,
    )

    # Call the LLM — bypass the pool semaphore (augment is a one-off call;
    # the main pipeline already holds slots for its multi-turn sessions).
    from tmcts.llm.client import get_pool, GLOBAL_MAX_TOKENS
    try:
        ep = get_pool().pick_endpoint_blocking()
        client = ep.create_client(timeout=60)
        response = client.chat.completions.create(
            model=ep.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=GLOBAL_MAX_TOKENS,
            temperature=0.3,
        )
        augmented_response = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("augment LLM failed for %s: %s", trace.sample_id, e)
        return None

    # Validate with the dual-format parser
    if validate_tool_call(augmented_response) is None:
        logger.info("augment: LLM response unparseable for %s — using local fallback", trace.sample_id)
        # Local fallback: extract the repaired <tool_call> JSON, wrap in empty <think>
        raw_repair = messages[repair_resp_idx].get("content", "")
        tc_match = _TOOL_CALL_RE.search(raw_repair)
        if tc_match:
            augmented_response = (
                "<think>\n</think>\n\n"
                f"<tool_call>\n{tc_match.group(1)}\n</tool_call>"
            )
        else:
            return None

    # Build new messages: [0..code_gen-1] + augmented + [result..end]
    new_messages = [dict(messages[i]) for i in range(code_gen_idx)]
    new_messages.append({"role": "assistant", "content": augmented_response})
    for i in range(result_user_idx, len(messages)):
        new_messages.append(dict(messages[i]))

    logger.info("augment: %s — %d->%d messages (dropped %d repair turns)",
                trace.sample_id, len(messages), len(new_messages),
                len(messages) - len(new_messages))

    return new_messages


# ======================================================================
# Part 2: single-task QA generation
# ======================================================================

def _tool_descriptions_minimal() -> str:
    return "\n".join(f"  {n} — {TOOL_REGISTRY[n]['description']}" for n in TOOL_NAMES)


def _parse_tc_json(text: str) -> Optional[dict]:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def build_tool_selection_prompt(sample: dict) -> str:
    task_input = build_task_input(sample)
    return "\n".join([
        "Select the most suitable symbolic reasoning tool for this problem.",
        "",
        "Available tools:",
        "",
        _tool_descriptions_minimal(),
        "",
        "Requirements:",
        "- Choose the best tool for this problem.",
        "- Do NOT generate code — just name the tool.",
        "- Briefly explain your reasoning before the tag.",
        "",
        "End your response with:",
        "",
        "<tool_call>",
        '{"name": "TOOL_NAME", "arguments": {"code": ""}}',
        "</tool_call>",
        "",
        "IMPORTANT: The tool_call must be valid JSON.  Do NOT add text after </tool_call>.",
        "",
        task_input,
    ])


def build_code_generation_prompt(sample: dict, tool_name: str) -> str:
    task_input = build_task_input(sample)
    hint = get_tool_hint(tool_name)
    return "\n".join([
        f"You will use **{tool_name}** to solve this problem.",
        f"{hint}",
        "",
        "Generate self-contained, executable code in the tool's native format.",
        "- Briefly explain your formalization before the code.",
        "",
        "End your response with:",
        "",
        "<tool_call>",
        f'{{"name": "{tool_name}", "arguments": {{"code": "... your code ..."}}}}',
        "</tool_call>",
        "",
        "IMPORTANT: The tool_call must be valid JSON.  Do NOT add text after </tool_call>.",
        "",
        task_input,
    ])


def build_code_repair_prompt(sample: dict, tool_name: str, buggy_code: str, error_output: str) -> str:
    task_input = build_task_input(sample)
    hint = get_tool_hint(tool_name)
    return "\n".join([
        "The following code for this problem failed with an error. Fix the code.",
        "",
        f"Tool: {tool_name}",
        f"{hint}",
        "",
        "Requirements:",
        "- Do NOT change the tool.",
        "- Fix the code based on the error message below.",
        "- The repaired code must be executable and self-contained.",
        "",
        "Briefly explain the cause of the error.",
        "End your response with:",
        "",
        "<tool_call>",
        f'{{"name": "{tool_name}", "arguments": {{"code": "... your repaired code ..."}}}}',
        "</tool_call>",
        "",
        "IMPORTANT: The tool_call must be valid JSON.  Do NOT add text after </tool_call>.",
        "",
        task_input,
        "",
        "=== Buggy Code ===",
        buggy_code,
        "",
        "=== Error Output ===",
        error_output,
    ])


def _find_repair_triplet(messages: list[dict]) -> Optional[Tuple[str, str, str, str]]:
    _REPAIR_SENTINEL = "The previous tool execution failed."

    repair_resp_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] != "assistant":
            continue
        tc = _parse_tc_json(messages[i].get("content", ""))
        if tc is None:
            continue
        for j in range(i - 1, -1, -1):
            if messages[j]["role"] == "user" and messages[j].get("content", "").startswith(_REPAIR_SENTINEL):
                repair_resp_idx = i
                break
        if repair_resp_idx is not None:
            break
    if repair_resp_idx is None:
        return None

    error_user_idx = None
    buggy_idx = None
    for i in range(repair_resp_idx - 1, -1, -1):
        content = messages[i].get("content", "")
        if error_user_idx is None and messages[i]["role"] == "user" and content.startswith(_REPAIR_SENTINEL):
            error_user_idx = i
            continue
        if (error_user_idx is not None and messages[i]["role"] == "assistant"
                and _parse_tc_json(messages[i].get("content", "")) is not None):
            buggy_idx = i
            break
    if buggy_idx is None or error_user_idx is None:
        return None

    buggy = _parse_tc_json(messages[buggy_idx].get("content", ""))
    repair = _parse_tc_json(messages[repair_resp_idx].get("content", ""))
    if buggy is None or repair is None:
        return None

    buggy_tool = buggy["name"]
    buggy_code = buggy.get("arguments", {}).get("code", "") if isinstance(buggy.get("arguments"), dict) else ""
    repair_code = repair.get("arguments", {}).get("code", "") if isinstance(repair.get("arguments"), dict) else ""

    _ERR_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
    repair_user_content = messages[error_user_idx].get("content", "")
    err_match = _ERR_RE.search(repair_user_content)
    error_content = err_match.group(0) if err_match else repair_user_content[:500]

    return (buggy_tool, buggy_code, error_content, repair_code)


def _build_single_conv(prompt: str, tool_name: str, code: str, reasoning: str | None) -> list[dict]:
    tc_json = json.dumps({"name": tool_name, "arguments": {"code": code}})
    tc_block = "<tool_call>\n" + tc_json + "\n</tool_call>"
    if reasoning:
        if reasoning.startswith("<think>") and "</think>" in reasoning:
            fc_value = f"{reasoning}\n\n{tc_block}"
        else:
            fc_value = f"<think>\n{reasoning}\n</think>\n\n{tc_block}"
    else:
        fc_value = f"<think>\n</think>\n\n{tc_block}"
    return [
        {"from": "human", "value": prompt},
        {"from": "function_call", "value": fc_value},
    ]


def generate_single_task_items(trace: Trace, sample: dict) -> List[Dict[str, Any]]:
    if not trace.success or not trace.messages:
        return []

    correct_tool = trace.selected_tool or (trace.tools_used[0] if trace.tools_used else sample.get("ground_truth_tool", ""))
    if correct_tool not in TOOL_NAMES:
        return []

    # Extract correct code from the last assistant message using this tool
    correct_code = None
    for i in range(len(trace.messages) - 1, -1, -1):
        if trace.messages[i]["role"] != "assistant":
            continue
        tc = _parse_tc_json(trace.messages[i].get("content", ""))
        if tc and tc.get("name") == correct_tool:
            correct_code = tc.get("arguments", {}).get("code", "") if isinstance(tc.get("arguments"), dict) else ""
            break
    if not correct_code:
        return []

    system = PROMPTS["SYMBOLIC_REASONING_SYSTEM"]

    # Reasoning from the first assistant message
    first_reasoning = None
    for msg in trace.messages:
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            tc_clean = _TOOL_CALL_RE.sub("", content).strip()
            if len(tc_clean) > 30:
                first_reasoning = tc_clean
            break

    # Repair reasoning
    repair_reasoning = None
    _REPAIR_SENTINEL = "The previous tool execution failed."
    if trace.repair_count > 0:
        for i in range(len(trace.messages) - 1, -1, -1):
            if (trace.messages[i]["role"] == "assistant" and i > 0
                    and trace.messages[i - 1]["role"] == "user"
                    and trace.messages[i - 1].get("content", "").startswith(_REPAIR_SENTINEL)):
                content = trace.messages[i].get("content", "")
                tc_clean = _TOOL_CALL_RE.sub("", content).strip()
                if len(tc_clean) > 30:
                    repair_reasoning = tc_clean
                break

    items: List[Dict[str, Any]] = []

    # A: Tool Selection
    items.append({
        "id": f"task_toolsel::{trace.sample_id}",
        "conversations": _build_single_conv(
            build_tool_selection_prompt(sample), correct_tool, "", first_reasoning,
        ),
        "system": system,
        "tools": json.dumps(get_openai_tools()),
        "metadata": {
            "source": "sft_single_task",
            "category": trace.category, "gold_tool": trace.gold_tool,
            "selected_tool": correct_tool, "level": trace.level,
            "task": "tool_selection",
        },
    })

    # B: Code Generation
    items.append({
        "id": f"task_codegen::{trace.sample_id}",
        "conversations": _build_single_conv(
            build_code_generation_prompt(sample, correct_tool),
            correct_tool, correct_code, first_reasoning,
        ),
        "system": system,
        "tools": json.dumps(get_openai_tools()),
        "metadata": {
            "source": "sft_single_task",
            "category": trace.category, "gold_tool": trace.gold_tool,
            "selected_tool": correct_tool, "level": trace.level,
            "task": "code_generation",
        },
    })

    # C: Code Repair (only when a repair happened)
    if trace.repair_count > 0:
        triplet = _find_repair_triplet(trace.messages)
        if triplet is not None:
            tool, buggy_code, error_output, fixed_code = triplet
            items.append({
                "id": f"task_coderepair::{trace.sample_id}",
                "conversations": _build_single_conv(
                    build_code_repair_prompt(sample, tool, buggy_code, error_output),
                    tool, fixed_code, repair_reasoning,
                ),
                "system": system,
                "tools": json.dumps(get_openai_tools()),
                "metadata": {
                    "source": "sft_single_task",
                    "category": trace.category, "gold_tool": trace.gold_tool,
                    "selected_tool": tool, "level": trace.level,
                    "task": "code_repair",
                },
            })

    return items
