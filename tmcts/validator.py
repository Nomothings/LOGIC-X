"""Trace validation."""

from enum import Enum
from typing import Optional


class FailReason(str, Enum):
    TOOL_CALL_PARSE_ERROR = "tool_call_parse_error"
    CODE_EXECUTION_FAILED = "code_execution_failed"
    REPAIR_PARSE_ERROR = "repair_parse_error"
    REPAIR_FAILED = "repair_failed"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"
    FINAL_ANSWER_PARSE_ERROR = "final_answer_parse_error"
    ANSWER_WRONG = "answer_wrong"
    UNKNOWN_ERROR = "unknown_error"


def validate_final_answer(final_answer: Optional[str], ground_truth: str) -> Optional[str]:
    if final_answer is None:
        return FailReason.FINAL_ANSWER_PARSE_ERROR.value
    if final_answer != ground_truth:
        return FailReason.ANSWER_WRONG.value
    return None
