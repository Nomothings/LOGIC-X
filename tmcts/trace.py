"""Trace dataclass — a single tool-integrated reasoning attempt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Trace:
    sample_id: str
    category: str
    level: int
    split: str
    gold_tool: str
    selected_tool: Optional[str] = None
    ground_truth: str = ""
    final_answer: Optional[str] = None
    success: bool = False
    fail_reason: Optional[str] = None
    repair_count: int = 0
    tool_call_count: int = 1
    tools_used: list = field(default_factory=list)
    attempt_index: int = 0
    messages: List[Dict[str, str]] = field(default_factory=list)
