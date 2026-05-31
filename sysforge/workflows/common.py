from __future__ import annotations

import time
import traceback
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import strftime


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


def accepted_sample_confidence(base_confidence: float, sample_count: int) -> float:
    return min(0.99, base_confidence + 0.1 * max(0, sample_count - 1))


def stamp_finished(result) -> None:
    result.finished_at = strftime("%Y-%m-%dT%H:%M:%S")


def append_workflow_error(result, label: str, exc: Exception, *, include_traceback: bool = True) -> None:
    message = f"{label}: {exc}"
    if include_traceback:
        message = f"{message}\n{traceback.format_exc()}"
    result.errors.append(message)


def run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    logs_dir: Path,
    label: str,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> CommandResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{label}.stdout.log"
    stderr_path = logs_dir / f"{label}.stderr.log"
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        elapsed = time.monotonic() - start
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        status = "passed" if proc.returncode == 0 else "failed"
        failure = "" if proc.returncode == 0 else tail_text(proc.stderr or proc.stdout)
        return CommandResult(
            status=status,
            command=command,
            returncode=proc.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            stdout_tail=tail_text(proc.stdout),
            stderr_tail=tail_text(proc.stderr),
            elapsed_s=elapsed,
            failure_summary=failure,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")
        return CommandResult(
            status="failed",
            command=command,
            returncode=124,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            stdout_tail=tail_text(str(stdout)),
            stderr_tail=tail_text(str(stderr)),
            elapsed_s=elapsed,
            failure_summary=f"timeout after {timeout_s:.1f}s",
        )
