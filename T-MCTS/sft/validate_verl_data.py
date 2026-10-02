#!/usr/bin/env python3
"""
Validate verl parquet data by tokenizing through the Qwen3 chat template.
Mimics verl MultiTurnSFTDataset's validation logic without needing ray.

Key: parquet stores nested data as numpy arrays. We must convert these to
plain lists/dicts before tokenization (same as verl's
convert_nested_value_to_list_recursive).

Usage:
    python validate_verl_data.py --parquet ../train_data/LOGIC-X-8B/sft_trajectories.parquet \
                                 --model /path/to/qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import os
import traceback

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def to_list_recursive(obj):
    """Convert numpy arrays to lists/dicts recursively (same as verl's function)."""
    if isinstance(obj, dict):
        return {k: to_list_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_list_recursive(e) for e in obj]
    elif isinstance(obj, np.ndarray):
        return to_list_recursive(obj.tolist())
    else:
        return obj


def check_schema(df: pd.DataFrame, label: str) -> dict:
    """Validate parquet schema (convert ndarray first)."""
    print(f"\n{'='*60}")
    print(f" [{label}] Schema Check")
    print(f"{'='*60}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    assert "messages" in df.columns, "FATAL: 'messages' column missing!"

    role_dist = {}
    assistant_with_tc = 0
    assistant_total = 0
    assistant_no_tc = 0
    tool_total = 0
    empty = 0

    for i, row in df.iterrows():
        msgs = to_list_recursive(row["messages"])
        if not msgs:
            empty += 1
            continue

        for msg in msgs:
            role = msg.get("role", "???")
            role_dist[role] = role_dist.get(role, 0) + 1

            if role == "assistant":
                assistant_total += 1
                tcs = msg.get("tool_calls")
                if tcs and len(tcs) > 0:
                    assistant_with_tc += 1
                    for tc in tcs:
                        assert isinstance(tc, dict), f"tc not dict: {type(tc)} sample {i}"
                        assert tc.get("type") == "function", f"Bad type: {tc} sample {i}"
                        fn = tc.get("function", {})
                        assert "name" in fn, f"Missing name sample {i}"
                        assert "arguments" in fn, f"Missing arguments sample {i}"
                        assert isinstance(fn["arguments"], str), \
                            f"arguments not str: {type(fn['arguments'])} sample {i}"
                        try:
                            json.loads(fn["arguments"])
                        except json.JSONDecodeError:
                            print(f"  WARNING: bad args JSON, sample {i}")
                else:
                    assistant_no_tc += 1
            elif role == "tool":
                tool_total += 1

    print(f"  Roles: {json.dumps(role_dist, indent=4)}")
    print(f"  assistant total: {assistant_total}")
    print(f"    with tool_calls: {assistant_with_tc}")
    print(f"    without: {assistant_no_tc}")
    print(f"  tool messages: {tool_total}")
    print(f"  empty rows: {empty}")

    assert empty == 0, f"{empty} empty rows!"
    assert assistant_total > 0, "No assistant messages!"
    print("  Schema PASSED")

    return {"rows": len(df), "role_dist": role_dist,
            "assistant_with_tc": assistant_with_tc, "tool_total": tool_total}


def check_tokenization(df: pd.DataFrame, label: str, tokenizer, num_samples: int = 20):
    """Tokenize random samples and verify."""
    print(f"\n{'='*60}")
    print(f" [{label}] Tokenization Check ({num_samples} samples)")
    print(f"{'='*60}")

    import random
    rng = random.Random(42)
    indices = rng.sample(range(len(df)), min(num_samples, len(df)))

    errors = 0
    max_len = 0
    total_len = 0

    for idx in indices:
        row = df.iloc[idx]
        msgs = to_list_recursive(row["messages"])
        tools = None
        if "tools" in df.columns:
            t = row.get("tools")
            if t is not None and not (isinstance(t, float) and np.isnan(t)):
                tools = to_list_recursive(t) if isinstance(t, (np.ndarray, list)) else None

        try:
            result = tokenizer.apply_chat_template(
                msgs,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
            tl = len(result if isinstance(result, list) else result["input_ids"])
            total_len += tl
            if tl > max_len:
                max_len = tl

            # Decode and sanity check
            decoded = tokenizer.decode(result if isinstance(result, list) else result["input_ids"],
                                       skip_special_tokens=False)
            # Verify tool call presence in decoded output
            has_tc_in_msg = any(
                m.get("role") == "assistant" and m.get("tool_calls") and len(m["tool_calls"]) > 0
                for m in msgs
            )
            if has_tc_in_msg:
                assert "<tool_call>" in decoded or "tool_call" in decoded.lower(), \
                    f"Sample {idx}: tool_calls not rendered in template output"

        except Exception as e:
            print(f"  ERROR sample {idx}: {e}")
            # Show message structure for debugging
            for j, m in enumerate(msgs):
                role = m.get("role", "?")
                content_len = len(m.get("content", ""))
                tcs = m.get("tool_calls")
                has_tc = tcs is not None and len(tcs) > 0
                print(f"    [{j}] {role}: content_len={content_len}, has_tool_calls={has_tc}")
                if has_tc:
                    for tc in tcs:
                        fn = tc.get("function", {})
                        print(f"         tool: {fn.get('name', '?')}, args_len={len(fn.get('arguments', ''))}")
            errors += 1
            if errors <= 3:
                traceback.print_exc()

    print(f"  Errors: {errors}/{len(indices)}")
    print(f"  Max token length: {max_len}")
    print(f"  Avg token length: {total_len/max(1, len(indices)-errors):.0f}")

    if errors > 0:
        print("  FAILED")
        return False
    print("  PASSED")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Validate verl SFT parquet data")
    p.add_argument("--parquet", required=True, help="Path to the .parquet file to validate")
    p.add_argument("--model", required=True, help="Tokenizer/model path (e.g. Qwen3-8B)")
    p.add_argument("--num-samples", type=int, default=30, help="Random samples to tokenize")
    args = p.parse_args()

    print("=" * 60)
    print(" verl Data Validation")
    print(f" Model: {args.model}")
    print(f" Parquet: {args.parquet}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_passed = True

    df = pd.read_parquet(args.parquet)
    label = os.path.basename(args.parquet)

    # 1. Schema
    try:
        check_schema(df, label)
    except Exception as e:
        print(f"  SCHEMA FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    # 2. Tokenization
    try:
        if not check_tokenization(df, label, tokenizer, num_samples=args.num_samples):
            all_passed = False
    except Exception as e:
        print(f"  TOKENIZATION FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL CHECKS PASSED — verl data is ready!")
    else:
        print(" SOME CHECKS FAILED — see above")
    print("=" * 60)


if __name__ == "__main__":
    main()
