from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NcuResult:
    ok: bool
    rc: int
    stdout: str
    stderr: str
    rows: list[dict[str, str]]


def find_ncu() -> str | None:
    return shutil.which("ncu") or (Path("/usr/local/bin/ncu").exists() and "/usr/local/bin/ncu") or None


def _parse_csv(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    start = 0
    for index, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            start = index
            break
    if start >= len(lines):
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def profile(binary: Path, metrics: list[str], *, args: list[str] | None = None,
            timeout_s: float = 120.0) -> NcuResult:
    ncu = find_ncu()
    if ncu is None:
        return NcuResult(False, -1, "", "ncu not found on PATH", [])
    cmd = [
        ncu, "-f", "--target-processes", "all",
        "--metrics", ",".join(metrics), "--csv",
        str(binary), *(args or []),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return NcuResult(False, -1, exc.stdout or "", f"ncu timeout after {timeout_s}s", [])
    rows = _parse_csv(result.stdout) if result.returncode == 0 else []
    return NcuResult(result.returncode == 0, result.returncode, result.stdout, result.stderr, rows)
