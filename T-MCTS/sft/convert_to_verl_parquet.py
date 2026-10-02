#!/usr/bin/env python3
"""
Convert T-MCTS ShareGPT SFT data -> verl-compatible parquet.

Input (ShareGPT format, as produced by generate_trajectories.py):
    {
        "id": "...",
        "conversations": [
            {"from": "human",          "value": "<problem text>"},
            {"from": "function_call",  "value": "<think>...</think><tool_call>{JSON}</tool_call>"},
            {"from": "observation",    "value": "<raw tool output>"},
            {"from": "gpt",            "value": "<think>...</think><answer>...</answer>"},
            ...
        ],
        "system": "<system prompt>",
        "tools": "[{...}, {...}]"      # JSON string
    }

Output (verl MultiTurnSFTDataset):
    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "<think>...</think>",
             "tool_calls": [{"type": "function",
                             "function": {"name": "...", "arguments": "{...}"}}]},
            {"role": "tool",      "content": "..."},
            {"role": "assistant", "content": "<think>...</think><answer>...</answer>"},
            ...
        ],
        "tools": [{"type": "function", "function": {...}}, ...]   # parsed list
    }

Key conversions:
  - human -> user (content as-is)
  - function_call -> assistant (extract <think> -> content, <tool_call> -> tool_calls)
  - observation -> tool (content as-is)
  - gpt -> assistant (if <tool_call> present, split; otherwise content as-is)
  - system -> first message with role="system"
  - tools: parse JSON string -> list

Usage:
    python convert_to_verl_parquet.py --input  ../train_data/LOGIC-X-8B/sft_trajectories.jsonl \
                                      --output ../train_data/LOGIC-X-8B/sft_trajectories.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

# Regex to extract <tool_call> block (handle nested JSON)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_call_block(text: str) -> dict | None:
    """Extract and parse <tool_call>{...}</tool_call> from text.

    Returns the parsed dict with name and arguments, or None if not found.
    """
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def strip_tool_call_from_content(text: str) -> str:
    """Remove the <tool_call>...</tool_call> block from text, keeping <think> etc."""
    return TOOL_CALL_RE.sub("", text).strip()


def build_tool_calls(tool_call_data: dict) -> list[dict]:
    """Convert a parsed tool_call dict to the OpenAI tool_calls format.

    Input:  {"name": "prover9", "arguments": {"code": "..."}}
    Output: [{"type": "function", "function": {"name": "prover9", "arguments": "{\"code\": \"...\"}"}}]
    """
    name = tool_call_data.get("name", "")
    arguments = tool_call_data.get("arguments", {})
    # OpenAI format requires arguments to be a JSON string.
    # Use compact separators to match the ShareGPT data style.
    # (The Qwen3 template's tojson filter re-serializes the outer structure
    #  anyway, but compact arguments minimize token-level divergence.)
    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))
    else:
        arguments_str = str(arguments)

    return [{
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_str,
        }
    }]


def convert_sample(sample: dict) -> dict | None:
    """Convert a single ShareGPT sample to verl format. Returns None if invalid."""
    messages = []

    # 1. Prepend system message
    system_text = sample.get("system", "")
    if system_text:
        messages.append({"role": "system", "content": system_text})

    # 2. Convert conversation turns
    convs = sample.get("conversations", [])
    for msg in convs:
        role = msg.get("from", "")
        content = msg.get("value", "")

        if role == "human":
            messages.append({"role": "user", "content": content})

        elif role == "observation":
            messages.append({"role": "tool", "content": content})

        elif role in ("function_call", "gpt"):
            # Only assistant-role messages can have real tool_calls.
            # (human messages sometimes contain <tool_call> examples in instructions.)
            tool_call_data = parse_tool_call_block(content)

            if tool_call_data:
                # Split: <think>...</think> -> content, <tool_call> -> tool_calls
                clean_content = strip_tool_call_from_content(content)
                assistant_msg = {
                    "role": "assistant",
                    "content": clean_content,
                    "tool_calls": build_tool_calls(tool_call_data),
                }
            else:
                # Pure assistant message (final answer or intermediate reasoning)
                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                }

            messages.append(assistant_msg)

        else:
            # Unknown role — keep as-is
            messages.append({"role": role, "content": content})

    if not messages:
        return None

    result = {"messages": messages}

    # 3. Parse tools (JSON string -> list)
    tools = sample.get("tools", None)
    if tools is not None:
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                print(f"  WARNING: failed to parse tools JSON for sample {sample.get('id', '?')}")
                tools = None
        if tools:
            result["tools"] = tools

    return result


def convert_file(input_path: str, output_path: str) -> int:
    """Convert a ShareGPT JSONL file (optionally gzip-compressed) to verl parquet."""
    import gzip
    opener = gzip.open if input_path.endswith(".gz") else open
    rows = []
    errors = 0
    with opener(input_path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            converted = convert_sample(record)
            if converted:
                rows.append(converted)
            else:
                errors += 1

    if errors:
        print(f"  {errors} errors skipped")

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert ShareGPT SFT data to verl parquet")
    p.add_argument("--input", required=True, help="Input ShareGPT .jsonl")
    p.add_argument("--output", required=True, help="Output .parquet path")
    args = p.parse_args()

    n = convert_file(args.input, args.output)
    print(f"[ok] {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
