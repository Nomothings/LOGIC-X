"""
Raw API call logger for T-MCTS.
Records every LLM call to a parquet file for analysis/debugging.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_raw_logger: Optional["RawApiLogger"] = None


class RawApiLogger:
    def __init__(self, parquet_path: str):
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        self._parquet_path = parquet_path
        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._count = 0
        logger.info("RawApiLogger: %s", parquet_path)

    def log(self, question_id: str, attempt_idx: int, turn_idx: int,
            endpoint_name: str, model: str, messages: list,
            response_text: str, raw_choice_json: str,
            finish_reason: str, latency_ms: float) -> None:
        import json

        with self._lock:
            self._buffer.append({
                "question_id": question_id,
                "attempt_idx": attempt_idx,
                "turn_idx": turn_idx,
                "endpoint_name": endpoint_name,
                "model": model,
                "messages_json": json.dumps(messages, ensure_ascii=False),
                "response_text": response_text,
                "raw_choice_json": raw_choice_json,
                "finish_reason": finish_reason,
                "latency_ms": round(latency_ms, 1),
            })
            self._count += 1
            if len(self._buffer) >= 200:
                self._flush()

    def _flush(self) -> None:
        import pandas as pd
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        if os.path.exists(self._parquet_path):
            existing = pd.read_parquet(self._parquet_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(self._parquet_path, index=False)
        self._buffer.clear()

    def close(self) -> None:
        self._flush()
        logger.info("RawApiLogger closed: %d total calls -> %s", self._count, self._parquet_path)


def init_raw_logger(parquet_path: str) -> RawApiLogger:
    global _raw_logger
    _raw_logger = RawApiLogger(parquet_path)
    return _raw_logger


def get_raw_logger() -> Optional[RawApiLogger]:
    return _raw_logger
