import time

from sysforge.workflows.common import (
    RetryBudget,
    accepted_sample_confidence,
    consume_retry,
    deadline_exceeded,
    tail_text,
)


def test_consume_retry_updates_history_and_budget():
    budget = RetryBudget(max_retries=2)
    assert consume_retry(budget, {"kind": "compile"}) is True
    assert budget.retries_used == 1
    assert budget.retries_left == 1
    assert budget.history == [{"kind": "compile"}]
    assert consume_retry(budget, {"kind": "runtime"}) is True
    assert budget.retries_left == 0
    assert consume_retry(budget, {"kind": "plausibility"}) is False


def test_deadline_exceeded_checks_monotonic_time():
    assert deadline_exceeded(time.monotonic() - 1.0) is True
    assert deadline_exceeded(time.monotonic() + 10.0) is False
    assert deadline_exceeded(None) is False


def test_tail_text_handles_short_and_long_input():
    assert tail_text("abc", 10) == "abc"
    assert tail_text("abcdef", 3) == "def"


def test_accepted_sample_confidence_matches_existing_growth():
    assert accepted_sample_confidence(0.8, 1) == 0.8
    assert accepted_sample_confidence(0.8, 3) == 0.99
