from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    ok: bool
    rc: int
    stdout: str
    stderr: str
    wallclock_s: float
    timed_out: bool = False


def run_binary(binary: Path, args: list[str] | None = None, *,
               timeout_s: float = 30.0, env: dict[str, str] | None = None) -> RunResult:
    cmd = [str(binary), *(args or [])]
    started = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as exc:
        return RunResult(False, -1, exc.stdout or "", exc.stderr or "",
                         time.monotonic() - started, timed_out=True)
    return RunResult(result.returncode == 0, result.returncode, result.stdout, result.stderr,
                     time.monotonic() - started)
