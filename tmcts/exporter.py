"""
ShareGPT SFT export (Qwen3 native format).

Converts pipeline messages (user/assistant roles, text-based tool calls)
to ShareGPT format (human/gpt/function_call/observation).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from tmcts.tools import get_openai_tools
from tmcts.trace import Trace

_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_TAGCODE_RE = re.compile(
    r"<tool_call>(\w+)</tool_call>\s*<code>\s*(.*?)\s*</code>", re.DOTALL,
)


def export_to_sharegpt(trace: Trace) -> Optional[Dict[str, Any]]:
    if not trace.messages:
        return None

    conversations = []
    system_content = ""

    for msg in trace.messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            system_content = content

        elif role == "user":
            m = _TOOL_RESPONSE_RE.search(content)
            if not m:
                conversations.append({"from": "human", "value": content})
            else:
                tool_output = m.group(1).strip()
                if tool_output:
                    conversations.append({"from": "observation", "value": tool_output})

        elif role == "assistant":
            # ---- Format 1: JSON <tool_call>{"name":"t","arguments":...}</tool_call> ----
            m_json = _TOOL_CALL_JSON_RE.search(content)
            if m_json:
                tc_json = m_json.group(1).strip()
                try:
                    tc_obj = json.loads(tc_json)
                except json.JSONDecodeError:
                    tc_obj = None

                if tc_obj and "name" in tc_obj and "arguments" in tc_obj:
                    reasoning = content[:m_json.start()].strip()
                    if reasoning:
                        if reasoning.startswith("<think>") and reasoning.endswith("</think>"):
                            wrapped = content
                        else:
                            wrapped = f"<think>\n{reasoning}\n</think>\n\n{m_json.group(0)}"
                    else:
                        wrapped = content
                    conversations.append({"from": "function_call", "value": wrapped})
                    continue
                # fall through to TagCode

            # ---- Format 2: Tag+Code <tool_call>NAME</tool_call><code>...</code> ----
            m_tc = _TOOL_CALL_TAGCODE_RE.search(content)
            if m_tc:
                tool_name = m_tc.group(1)
                code = m_tc.group(2).strip()
                # Re-build as JSON-style <tool_call> for consistent output
                tc_json = json.dumps({"name": tool_name, "arguments": {"code": code}})
                tc_block = f"<tool_call>\n{tc_json}\n</tool_call>"

                reasoning = content[:m_tc.start()].strip()
                if reasoning:
                    if reasoning.startswith("<think>") and reasoning.endswith("</think>"):
                        wrapped = f"{reasoning}\n\n{tc_block}"
                    else:
                        wrapped = f"<think>\n{reasoning}\n</think>\n\n{tc_block}"
                else:
                    wrapped = f"<think>\n</think>\n\n{tc_block}"

                conversations.append({"from": "function_call", "value": wrapped})
                continue

            # ---- Plain gpt (answer, no tool_call) ----
            conversations.append({"from": "gpt", "value": content})

    return {
        "id": trace.sample_id,
        "conversations": conversations,
        "system": system_content,
        "tools": json.dumps(get_openai_tools()),
        "metadata": {
            "category": trace.category,
            "gold_tool": trace.gold_tool,
            "selected_tool": trace.selected_tool,
            "ground_truth": trace.ground_truth,
            "final_answer": trace.final_answer,
            "repair_count": trace.repair_count,
            "level": trace.level,
        },
    }
