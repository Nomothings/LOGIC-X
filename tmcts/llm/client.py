"""
LLM client with multi-endpoint pool, concurrency control, and circuit breaker.

Endpoints are configured through the .env file (see .env.example). Multiple
endpoints can be combined into one pool; requests are routed to the endpoint
with the most free slots.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------

_ENV_FILE = None


def _load_dotenv() -> None:
    global _ENV_FILE
    if _ENV_FILE is not None:
        return
    # Walk up from this file to find .env
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        env_path = Path.cwd() / ".env"
    _ENV_FILE = env_path
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

GLOBAL_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
GLOBAL_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
GLOBAL_TIMEOUT = float(os.getenv("TIMEOUT", "120.0"))

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    PERMANENTLY_DISABLED = "permanently_disabled"


FAILURE_THRESHOLD = 5
COOLDOWN_SECONDS = 120
RATE_LIMIT_COOLDOWN = 10
TRANSIENT_COOLDOWN = 10
FALLBACK_FAILURE_RATE = 0.5
FALLBACK_MIN_CALLS = 10


@dataclass
class _EndpointStats:
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0


# ---------------------------------------------------------------------------
# APIEndpoint
# ---------------------------------------------------------------------------


@dataclass
class _EndpointCfg:
    name: str
    api_key: str
    base_url: str
    model: str
    worker_num: int


class APIEndpoint:
    def __init__(self, cfg: _EndpointCfg):
        self.name = cfg.name
        self.api_key = cfg.api_key
        self.base_url = cfg.base_url
        self.model = cfg.model
        self.max_workers = cfg.worker_num
        self.current_workers = cfg.worker_num
        self._semaphore = threading.BoundedSemaphore(self.current_workers)
        self._lock = threading.Lock()
        self._circuit_state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._cooldown_until: float = 0.0
        self._stats = _EndpointStats()
        self._rate_limit_streak: int = 0

    def acquire(self, timeout: float = 30) -> bool:
        return self._semaphore.acquire(timeout=timeout)

    def release(self) -> None:
        self._semaphore.release()

    @property
    def available_slots(self) -> int:
        try:
            return self._semaphore._value
        except Exception:
            return self.current_workers

    @property
    def is_healthy(self) -> bool:
        with self._lock:
            if self._circuit_state == CircuitState.PERMANENTLY_DISABLED:
                return False
            if self._circuit_state == CircuitState.OPEN:
                if time.time() >= self._cooldown_until:
                    self._circuit_state = CircuitState.HALF_OPEN
                    self._consecutive_successes = 0
                    return True
                return False
            return True

    @property
    def circuit_state(self) -> CircuitState:
        with self._lock:
            return self._circuit_state

    def report_success(self) -> None:
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_successes += 1
            self._consecutive_failures = 0
            self._rate_limit_streak = 0
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= 2:
                    self._circuit_state = CircuitState.CLOSED

    def report_failure(self, error: Exception) -> None:
        error_str = str(error)
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_failures += 1

            if _is_permanent(error_str):
                self._circuit_state = CircuitState.PERMANENTLY_DISABLED
                logger.warning("[%s] PERMANENTLY DISABLED: %s", self.name, error_str[:120])
                return

            if _is_rate_limit(error_str):
                self._rate_limit_streak += 1
                self._circuit_state = CircuitState.OPEN
                self._cooldown_until = time.time() + RATE_LIMIT_COOLDOWN
                self._maybe_fallback(streak=self._rate_limit_streak)
                return

            if _is_transient(error_str):
                self._circuit_state = CircuitState.OPEN
                self._cooldown_until = time.time() + TRANSIENT_COOLDOWN
                # Transient errors don't count toward the 5-failure breaker
                # and don't trigger worker fallback.
                return

            self._consecutive_failures += 1
            if (self._circuit_state == CircuitState.HALF_OPEN or
                    self._consecutive_failures >= FAILURE_THRESHOLD):
                self._circuit_state = CircuitState.OPEN
                self._cooldown_until = time.time() + COOLDOWN_SECONDS

            self._maybe_fallback()

    def _maybe_fallback(self, streak: int = 0) -> None:
        s = self._stats
        if s.total_calls < FALLBACK_MIN_CALLS:
            return
        if s.total_failures / s.total_calls < FALLBACK_FAILURE_RATE:
            return
        old = self.current_workers
        if old >= 128:
            base = 32
        elif old >= 64:
            base = 16
        elif old >= 16:
            base = 8
        else:
            base = 4
        step = base * max(1, streak)
        new = max(4, old - step)
        if new == old:
            return
        logger.warning("[%s] worker fallback: %d -> %d", self.name, old, new)
        self.current_workers = new
        while self._semaphore.acquire(blocking=False):
            pass
        self._semaphore = threading.BoundedSemaphore(new)

    def batch_recover(self) -> None:
        with self._lock:
            if self._circuit_state == CircuitState.PERMANENTLY_DISABLED:
                return
            if self._circuit_state == CircuitState.OPEN:
                if time.time() >= self._cooldown_until:
                    self._circuit_state = CircuitState.HALF_OPEN
                    self._consecutive_successes = 0
                    self._consecutive_failures = 0
                return
            self._consecutive_failures = 0

    def create_client(self, timeout: Optional[float] = None) -> OpenAI:
        http_c = httpx.Client(verify=True, timeout=timeout or GLOBAL_TIMEOUT)
        return OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http_c)


def _is_rate_limit(error_str: str) -> bool:
    lower = error_str.lower()
    return any(kw in lower for kw in (
        "429", "rate limit", "rate_limit", "too many requests",
        "upstream saturation", "overloaded",
    ))


def _is_transient(error_str: str) -> bool:
    lower = error_str.lower()
    return any(kw in lower for kw in (
        "500", "502", "503", "504",
        "connection reset", "connection error", "connection refused",
        "read timeout", "read timed out", "eof",
        "internal server error", "bad gateway", "service unavailable",
        "temporarily unavailable", "upstream error", "server error",
    ))


def _is_permanent(error_str: str) -> bool:
    lower = error_str.lower()
    if any(kw in lower for kw in ("429", "rate limit", "rate_limit",
            "too many requests", "saturated")):
        return False
    return any(kw in lower for kw in (
        "401", "403", "insufficient_quota", "billing", "invalid api key",
    ))


# ---------------------------------------------------------------------------
# APIPool
# ---------------------------------------------------------------------------


class APIPool:
    def __init__(self, endpoints: list[APIEndpoint]):
        self._endpoints = endpoints

    @property
    def total_capacity(self) -> int:
        return sum(ep.current_workers for ep in self._endpoints if ep.is_healthy)

    def pick_endpoint(self) -> Optional[APIEndpoint]:
        candidates = [ep for ep in self._endpoints if ep.is_healthy]
        return max(candidates, key=lambda ep: ep.available_slots) if candidates else None

    def pick_endpoint_blocking(self, poll_interval: float = 2.0) -> APIEndpoint:
        while True:
            ep = self.pick_endpoint()
            if ep is not None:
                return ep
            time.sleep(poll_interval)

    def batch_recover(self) -> None:
        for ep in self._endpoints:
            ep.batch_recover()

    def summary(self) -> str:
        lines = ["API Pool:"]
        for ep in self._endpoints:
            lines.append(
                f"  [{ep.name}] {ep.model} @ {ep.base_url} "
                f"workers={ep.current_workers}/{ep.max_workers} "
                f"circuit={ep.circuit_state.value}"
            )
        lines.append(f"  total capacity: {self.total_capacity}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton + init
# ---------------------------------------------------------------------------

_pool: Optional[APIPool] = None


def _load_endpoints() -> list[_EndpointCfg]:
    endpoints = []
    for i in range(1, 21):
        key = os.getenv(f"API_{i}_KEY")
        if not key:
            break
        base = os.getenv(f"API_{i}_BASE", "").strip()
        model = os.getenv(f"API_{i}_MODEL", "").strip()
        workers = int(os.getenv(f"API_{i}_WORKER_NUM", "4"))
        if not base or not model:
            continue
        endpoints.append(_EndpointCfg(name=f"api_{i}", api_key=key, base_url=base,
                                       model=model, worker_num=workers))
    return endpoints


def init_pool() -> APIPool:
    global _pool
    cfgs = _load_endpoints()
    if not cfgs:
        raise RuntimeError("No API endpoints configured. Add API_1_KEY/BASE/MODEL to .env")
    endpoints = [APIEndpoint(cfg) for cfg in cfgs]
    _pool = APIPool(endpoints)
    logger.info("API pool: %d endpoints, capacity=%d", len(endpoints), _pool.total_capacity)
    logger.info("\n%s", _pool.summary())
    return _pool


def get_pool() -> APIPool:
    if _pool is None:
        raise RuntimeError("API pool not initialised — call init_pool() first.")
    return _pool


# ---------------------------------------------------------------------------
# call_llm (unified entry point)
# ---------------------------------------------------------------------------

# Re-export for convenience
from tmcts.llm.raw_logger import get_raw_logger  # noqa: E402


def call_llm(
    messages: list[dict],
    tools: Optional[list] = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    endpoint: Optional[APIEndpoint] = None,
    question_id: str = "",
    turn_idx: int = -1,
) -> str:
    pool = get_pool()

    # Stateless mode: auto-acquire from pool
    if endpoint is None:
        endpoint = pool.pick_endpoint_blocking()
        if not endpoint.acquire(timeout=30):
            raise RuntimeError(f"Timed out waiting for slot on {endpoint.name}")
        try:
            return _call_endpoint(endpoint, messages, tools, temperature, timeout,
                                  question_id, turn_idx)
        finally:
            endpoint.release()

    # Session-affinity mode
    return _call_endpoint(endpoint, messages, tools, temperature, timeout,
                          question_id, turn_idx)


def _call_endpoint(
    endpoint: APIEndpoint,
    messages: list[dict],
    tools: Optional[list],
    temperature: Optional[float],
    timeout: Optional[float],
    question_id: str,
    turn_idx: int,
) -> str:
    client = endpoint.create_client(timeout=timeout)
    t0 = time.time()

    try:
        kwargs = dict(
            model=endpoint.model,
            messages=messages,
            max_tokens=GLOBAL_MAX_TOKENS,
            temperature=temperature if temperature is not None else GLOBAL_TEMPERATURE,
        )
        if tools:
            kwargs["tools"] = tools
        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        endpoint.report_success()

        # Log
        _log_call(question_id, turn_idx, endpoint, messages, content, response, t0)

        return content

    except Exception as exc:
        endpoint.report_failure(exc)
        raise


def _log_call(question_id: str, turn_idx: int, endpoint: APIEndpoint,
              messages: list[dict], result: str, response, t0: float) -> None:
    try:
        raw_logger = get_raw_logger()
        if raw_logger is None:
            return

        latency_ms = (time.time() - t0) * 1000

        try:
            raw_json = response.choices[0].model_dump_json()
        except Exception:
            raw_json = json.dumps({"finish_reason": str(response.choices[0].finish_reason)})

        raw_logger.log(
            question_id=question_id,
            attempt_idx=-1,
            turn_idx=turn_idx,
            endpoint_name=endpoint.name,
            model=endpoint.model,
            messages=messages,
            response_text=result,
            raw_choice_json=raw_json,
            finish_reason=str(response.choices[0].finish_reason),
            latency_ms=latency_ms,
        )
    except Exception:
        pass
