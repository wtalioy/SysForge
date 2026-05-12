from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from time import strftime


@dataclass
class RetryBudget:
    max_retries: int
    retries_used: int = 0
    history: list[dict] = field(default_factory=list)

    @property
    def retries_left(self) -> int:
        return max(0, self.max_retries - self.retries_used)


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


def accepted_sample_confidence(base_confidence: float, sample_count: int) -> float:
    return min(0.99, base_confidence + 0.1 * max(0, sample_count - 1))


def stamp_finished(result) -> None:
    result.finished_at = strftime("%Y-%m-%dT%H:%M:%S")


def append_workflow_error(result, label: str, exc: Exception, *, include_traceback: bool = True) -> None:
    message = f"{label}: {exc}"
    if include_traceback:
        message = f"{message}\n{traceback.format_exc()}"
    result.errors.append(message)
