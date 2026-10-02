"""
Tag parsers for T-MCTS — supports two <tool_call> formats:

1. JSON (normal traces):
   <tool_call>{"name":"prover9","arguments":{"code":"..."}}</tool_call>

2. Tag+Code (augmentation prompt output):
   <tool_call>prover9</tool_call>
   <code>...</code>
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from tmcts.tools import TOOL_NAMES

_RE_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_RE_CODE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)
_RE_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _extract_json_object(text: str) -> str | None:
    """Extract a valid JSON object by finding matching braces.

    Handles trailing text that LLMs sometimes emit after ``}}`` inside
    ``<tool_call>`` (e.g. reasoning appended before ``</tool_call>``).
    """
    i = text.find("{")
    if i < 0:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for j in range(i, len(text)):
        ch = text[j]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


def parse_tool_call(text: str) -> Optional[Dict[str, str]]:
    matches = _RE_TOOL_CALL.findall(text)
    if not matches:
        return None

    for match_text in reversed(matches):
        stripped = match_text.strip()

        # Format 1: JSON
        json_candidate = _extract_json_object(stripped)
        if json_candidate is not None:
            try:
                tc = json.loads(json_candidate)
            except (json.JSONDecodeError, TypeError):
                pass
            else:
                if isinstance(tc, dict) and "name" in tc and "arguments" in tc:
                    name = tc["name"]
                    if name in TOOL_NAMES:
                        code = tc["arguments"].get("code", "") if isinstance(tc["arguments"], dict) else ""
                        return {"tool_name": name, "code": code}
            continue

        # Format 2: plain tool name + <code> block
        if stripped in TOOL_NAMES:
            code_match = _RE_CODE.search(text)
            code = code_match.group(1).strip() if code_match else ""
            return {"tool_name": stripped, "code": code}

    return None


def parse_answer(text: str) -> Optional[str]:
    matches = _RE_ANSWER.findall(text)
    if not matches:
        return None

    answer_text = matches[-1].strip()

    m = re.search(r"The final answer is\s+([A-Z])", answer_text)
    if m:
        return m.group(1)

    letters = re.findall(r"\b([A-Z])\b", answer_text)
    if letters:
        return letters[-1]

    return None


def parse_step_decision(text: str) -> Optional[Dict[str, str]]:
    answer = parse_answer(text)
    if answer is not None:
        return {"type": "answer", "answer": answer}

    tool_call = parse_tool_call(text)
    if tool_call is not None:
        return {"type": "tool_call", "tool_name": tool_call["tool_name"], "code": tool_call["code"]}

    return None
