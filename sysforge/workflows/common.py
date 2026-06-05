from __future__ import annotations

import statistics
import time
import traceback
from dataclasses import dataclass, field


@dataclass
class RetryBudget:
    max_retries: int
    retries_used: int = 0
    history: list[dict] = field(default_factory=list)

    @property
    def retries_left(self) -> int:
        return max(0, self.max_retries - self.retries_used)


@dataclass
class CommandResult:
    status: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str
    stdout_tail: str = ""
    stderr_tail: str = ""
    elapsed_s: float = 0.0
    failure_summary: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def consume_retry(state: RetryBudget, entry: dict) -> bool:
    state.history.append(entry)
    state.retries_used += 1
    return state.retries_used <= state.max_retries


def deadline_exceeded(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() > deadline


def tail_text(text: str, size: int = 4000) -> str:
    if len(text) <= size:
        return text
    return text[-size:]


def median_float(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def spread_pct(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    median_value = median_float(values)
    if median_value == 0.0:
        return 0.0
    return (max(values) - min(values)) / median_value * 100.0


def accepted_sample_confidence(base_confidence: float, sample_count: int) -> float:
    return min(0.99, base_confidence + 0.1 * max(0, sample_count - 1))


def workflow_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def append_workflow_error(result, label: str, exc: Exception, *, include_traceback: bool = True) -> None:
    message = f"{label}: {exc}"
    if include_traceback:
        message = f"{message}\n{traceback.format_exc()}"
    result.errors.append(message)
