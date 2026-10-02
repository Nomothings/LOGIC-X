"""
Core pipeline — multi-turn tool-integrated reasoning loop.

Two entry points:
  - run_sample:          autonomous evaluation (the model decides every step)
  - run_sample_guided:   teacher-guided generation used to build SFT data
                         (repair / interpretation prompts are injected so the
                         generated trajectories demonstrate recovery behaviour)

State machine: TOOL_CODE_GENERATION -> execute -> model decides -> loop
                    ^_________________________________|
                       the model can emit another <tool_call> to loop back

Single hard limit: MAX_LLM_CALLS = 6.
Every call_llm() increments round_count.  When round_count reaches 6,
the sample is stopped and returned as a failed sample.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from tmcts.data import build_task_input
from tmcts.llm.client import call_llm, get_pool
from tmcts.llm.parser import parse_tool_call, parse_step_decision
from tmcts.prompts import PROMPTS
from tmcts.tools import TOOL_NAMES, TOOL_REGISTRY, get_tool_hint, is_execution_success, run_tool
from tmcts.trace import Trace
from tmcts.validator import FailReason, validate_final_answer

logger = logging.getLogger(__name__)

MAX_LLM_CALLS = 6
API_TIMEOUT = 60   # per-API-call timeout in seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tool_hints_block() -> str:
    """Build the tool format hints block for the code-generation prompt."""
    parts = []
    for name in TOOL_NAMES:
        info = TOOL_REGISTRY.get(name, {})
        desc = info.get("description", name)
        hint = get_tool_hint(name)
        parts.append(f"  {name} — {desc}\n    {hint}")
    return "\n\n".join(parts)


def _init_trace(sample: dict, attempt_index: int = 0) -> Trace:
    return Trace(
        sample_id=sample.get("id", ""),
        category=sample.get("category", ""),
        level=sample.get("level", 0),
        split=sample.get("split", ""),
        gold_tool=sample.get("ground_truth_tool", ""),
        ground_truth=sample.get("ground_truth_answer", ""),
        attempt_index=attempt_index,
    )


# ---------------------------------------------------------------------------
# Autonomous evaluation pipeline
# ---------------------------------------------------------------------------

def run_sample(
    sample: dict,
    attempt_index: int = 0,
) -> Tuple[Optional[Trace], List[Trace]]:
    """Autonomous evaluation pipeline.

    Matches the SFT training format:
      system -> user(code_gen) -> assistant(tool_call)
           -> [ user(<tool_response>) -> assistant(repair/switch/answer) ]*

    No repair_prompt or interp_prompt — the model decides autonomously.
    Tool output is fed as a <tool_response> wrapped in a user message,
    matching the Qwen3 template rendering exactly.

    Returns (success_trace, all_attempts).
    """
    qid = sample.get("id", "")

    pool = get_pool()
    endpoint = pool.pick_endpoint_blocking()
    if not endpoint.acquire(timeout=30):
        raise RuntimeError(f"Failed to acquire slot on {endpoint.name}")

    trace = _init_trace(sample, attempt_index=attempt_index)
    all_attempts: List[Trace] = [trace]
    round_count = 0

    try:
        messages = [{"role": "system", "content": PROMPTS["SYMBOLIC_REASONING_SYSTEM"]}]

        # ---- Step 1: code generation ----
        task_input = build_task_input(sample)
        code_gen_prompt = PROMPTS["TOOL_CODE_GENERATION"].replace(
            "%{TOOL_FORMAT_HINTS}", _build_tool_hints_block(),
        )
        messages.append({"role": "user", "content": task_input + "\n\n" + code_gen_prompt})

        code_response = call_llm(messages, endpoint=endpoint,
                                 question_id=qid, turn_idx=round_count,
                                 timeout=API_TIMEOUT)
        round_count += 1
        messages.append({"role": "assistant", "content": code_response})

        tool_call = parse_tool_call(code_response)
        if tool_call is None:
            trace.fail_reason = FailReason.TOOL_CALL_PARSE_ERROR.value
            trace.messages = messages
            return None, all_attempts

        current_tool = tool_call["tool_name"]
        code = tool_call["code"]
        trace.selected_tool = current_tool
        tool_call_count = 1
        execution_ever_succeeded = False

        # ---- Step 2: autonomous loop ----
        final_answer = None

        while final_answer is None and round_count < MAX_LLM_CALLS:
            # Execute tool
            output = run_tool(current_tool, code)
            if is_execution_success(current_tool, output):
                execution_ever_succeeded = True

            # Feed output matching the Qwen3 template rendering:
            #   role="tool" -> <|im_start|>user\n<tool_response>\n{output}\n</tool_response><|im_end|>
            # Using role="user" + <tool_response> avoids the OpenAI API
            # tool_calls requirement.
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{output}\n</tool_response>",
            })

            # The model autonomously decides the next step
            decision_response = call_llm(messages, endpoint=endpoint,
                                         question_id=qid, turn_idx=round_count,
                                         timeout=API_TIMEOUT)
            round_count += 1
            messages.append({"role": "assistant", "content": decision_response})

            decision = parse_step_decision(decision_response)
            if decision is None:
                trace.fail_reason = FailReason.FINAL_ANSWER_PARSE_ERROR.value
                trace.messages = messages
                return None, all_attempts

            if decision["type"] == "answer":
                final_answer = decision["answer"]
                break

            # Continue with another tool call
            current_tool = decision["tool_name"]
            code = decision["code"]
            tool_call_count += 1

        # ---- Post-loop ----
        if final_answer is None:
            trace.fail_reason = FailReason.TOOL_CALL_LIMIT_EXCEEDED.value
            trace.messages = messages
            return None, all_attempts

        trace.final_answer = final_answer
        trace.tool_call_count = tool_call_count
        trace.tools_used = [current_tool]

        answer_fail = validate_final_answer(final_answer, sample.get("ground_truth_answer", ""))
        if answer_fail is not None:
            if answer_fail == FailReason.ANSWER_WRONG.value and not execution_ever_succeeded:
                trace.fail_reason = FailReason.CODE_EXECUTION_FAILED.value
            else:
                trace.fail_reason = answer_fail
            trace.messages = messages
            return None, all_attempts

        trace.success = True
        trace.messages = messages
        return trace, all_attempts

    except Exception as exc:
        logger.warning("Unexpected error for %s: %s", qid, exc)
        trace.fail_reason = FailReason.UNKNOWN_ERROR.value
        trace.messages = messages if 'messages' in dir() else []
        return None, all_attempts

    finally:
        endpoint.release()


# ---------------------------------------------------------------------------
# Guided generation pipeline (for SFT data construction)
# ---------------------------------------------------------------------------

def run_sample_guided(
    sample: dict,
    max_repair: int = 2,
    attempt_index: int = 0,
) -> Tuple[Optional[Trace], List[Trace]]:
    """Teacher-guided generation pipeline with in-band repair/interp prompts.

    Unlike the autonomous evaluation pipeline, this injects TOOL_CODE_REPAIR
    and RESULT_INTERPRETATION prompts as user messages.  This guides the
    generation model to produce the repair/switch/answer behaviour we want
    in the SFT data.

    The generated messages record contains these prompts so the resulting
    training data reflects the exact conversation the model saw.
    """
    qid = sample.get("id", "")

    pool = get_pool()
    endpoint = pool.pick_endpoint_blocking()
    if not endpoint.acquire(timeout=30):
        raise RuntimeError(f"Failed to acquire slot on {endpoint.name}")

    trace = _init_trace(sample, attempt_index=attempt_index)
    all_attempts: List[Trace] = [trace]
    round_count = 0

    try:
        messages = [{"role": "system", "content": PROMPTS["SYMBOLIC_REASONING_SYSTEM"]}]

        # ---- Step 1: code generation (round 1) ----
        task_input = build_task_input(sample)
        code_gen_prompt = PROMPTS["TOOL_CODE_GENERATION"].replace(
            "%{TOOL_FORMAT_HINTS}", _build_tool_hints_block(),
        )
        messages.append({"role": "user", "content": task_input + "\n\n" + code_gen_prompt})

        code_response = call_llm(messages, endpoint=endpoint,
                                 question_id=qid, turn_idx=round_count,
                                 timeout=API_TIMEOUT)
        round_count += 1
        messages.append({"role": "assistant", "content": code_response})

        tool_call = parse_tool_call(code_response)
        if tool_call is None:
            trace.fail_reason = FailReason.TOOL_CALL_PARSE_ERROR.value
            trace.messages = messages
            return None, all_attempts

        current_tool = tool_call["tool_name"]
        code = tool_call["code"]
        trace.selected_tool = current_tool
        tool_call_count = 1

        # ---- Step 2: execute + repair + decision loop ----
        final_answer = None
        execution_ever_succeeded = False

        while final_answer is None and round_count < MAX_LLM_CALLS:
            last_output = ""

            # ---- 2a: execute + repair (each repair = 1 LLM call) ----
            for repair_idx in range(max_repair + 1):
                if round_count >= MAX_LLM_CALLS:
                    break

                output = run_tool(current_tool, code)
                last_output = output

                if is_execution_success(current_tool, output):
                    execution_ever_succeeded = True
                    break

                if repair_idx == max_repair:
                    break

                repair_prompt = PROMPTS["TOOL_CODE_REPAIR"].replace(
                    "%{ERROR_OUTPUT}", output,
                ).replace("%{TOOL_HINT}", get_tool_hint(current_tool))
                messages.append({"role": "user", "content": repair_prompt})

                repair_response = call_llm(messages, endpoint=endpoint,
                                           question_id=qid, turn_idx=round_count,
                                           timeout=API_TIMEOUT)
                round_count += 1
                messages.append({"role": "assistant", "content": repair_response})

                repaired_call = parse_tool_call(repair_response)
                if repaired_call is None:
                    trace.fail_reason = FailReason.REPAIR_PARSE_ERROR.value
                    trace.messages = messages
                    return None, all_attempts

                code = repaired_call["code"]
                trace.repair_count += 1

            # ---- 2b: round limit after repair ----
            if round_count >= MAX_LLM_CALLS:
                break

            # ---- 2c: result interpretation (1 LLM call) ----
            interp_prompt = PROMPTS["RESULT_INTERPRETATION"].replace(
                "%{TOOL_OUTPUT}", last_output,
            )
            messages.append({"role": "user", "content": interp_prompt})

            decision_response = call_llm(messages, endpoint=endpoint,
                                         question_id=qid, turn_idx=round_count,
                                         timeout=API_TIMEOUT)
            round_count += 1
            messages.append({"role": "assistant", "content": decision_response})

            decision = parse_step_decision(decision_response)
            if decision is None:
                trace.fail_reason = FailReason.FINAL_ANSWER_PARSE_ERROR.value
                trace.messages = messages
                return None, all_attempts

            if decision["type"] == "answer":
                final_answer = decision["answer"]
                break

            # Switch tool — loop back to execute+repair
            current_tool = decision["tool_name"]
            code = decision["code"]
            tool_call_count += 1

        # ---- Post-loop ----
        if final_answer is None:
            trace.fail_reason = FailReason.TOOL_CALL_LIMIT_EXCEEDED.value
            trace.messages = messages
            return None, all_attempts

        trace.final_answer = final_answer
        trace.tool_call_count = tool_call_count
        trace.tools_used = [current_tool]

        answer_fail = validate_final_answer(final_answer, sample.get("ground_truth_answer", ""))
        if answer_fail is not None:
            if answer_fail == FailReason.ANSWER_WRONG.value and not execution_ever_succeeded:
                trace.fail_reason = FailReason.CODE_EXECUTION_FAILED.value
            else:
                trace.fail_reason = answer_fail
            trace.messages = messages
            return None, all_attempts

        trace.success = True
        trace.messages = messages
        return trace, all_attempts

    except Exception as exc:
        logger.warning("Unexpected error for %s: %s", qid, exc)
        trace.fail_reason = FailReason.UNKNOWN_ERROR.value
        trace.messages = messages if 'messages' in dir() else []
        return None, all_attempts

    finally:
        endpoint.release()


# ---------------------------------------------------------------------------
# Shared primitives for the tree search (tmcts.search)
# ---------------------------------------------------------------------------

def build_seed_messages(sample: dict) -> list[dict]:
    """Build the root tool-use state: system + task + code-generation prompt."""
    messages = [{"role": "system", "content": PROMPTS["SYMBOLIC_REASONING_SYSTEM"]}]

    task_input = build_task_input(sample)
    code_gen_prompt = PROMPTS["TOOL_CODE_GENERATION"].replace(
        "%{TOOL_FORMAT_HINTS}", _build_tool_hints_block(),
    )
    messages.append({"role": "user", "content": task_input + "\n\n" + code_gen_prompt})
    return messages


def execute_and_append(messages: list[dict], tool_name: str, code: str) -> str:
    """Execute a solver program and append its feedback to the state.

    The output is wrapped in <tool_response> inside a user message, matching
    the chat-template rendering used everywhere else in this codebase.
    """
    output = run_tool(tool_name, code)
    messages.append({
        "role": "user",
        "content": f"<tool_response>\n{output}\n</tool_response>",
    })
    return output
