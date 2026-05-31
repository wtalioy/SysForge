from __future__ import annotations

import csv
import io
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _to_float(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def aggregate_per_metric(rows: list[dict[str, str]], requested: list[str]) -> dict[str, dict[str, Any]]:
    by_metric: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("Metric Name") or row.get("Metric_Name") or ""
        if not name:
            continue
        value_str = row.get("Metric Value") or row.get("Metric_Value") or ""
        unit = row.get("Metric Unit") or row.get("Metric_Unit") or ""
        numeric = _to_float(value_str)
        entry = by_metric.setdefault(name, {"unit": unit, "raw": []})
        if numeric is not None:
            entry["raw"].append(numeric)
        elif value_str:
            entry.setdefault("raw_text", []).append(value_str)

    for entry in by_metric.values():
        raw_values = entry.get("raw") or []
        if raw_values:
            entry["value"] = float(statistics.median(raw_values))
            entry["min"] = min(raw_values)
            entry["max"] = max(raw_values)
            entry["samples"] = len(raw_values)
            entry["values"] = raw_values
        else:
            entry["value"] = None
            entry["samples"] = 0

    ordered: dict[str, dict[str, Any]] = {}
    for metric in requested:
        if metric in by_metric:
            ordered[metric] = by_metric[metric]
        else:
            ordered[metric] = {"value": None, "unit": "", "samples": 0, "error": "metric not present in ncu output"}
    for metric, entry in by_metric.items():
        if metric not in ordered:
            ordered[metric] = entry
    return ordered


def format_error(result: NcuResult) -> str:
    combined = (result.stderr or "") + "\n" + (result.stdout or "")
    hint = ""
    if "ERR_NVGPUCTRPERM" in combined:
        hint = (
            " [hint: ncu lacks GPU perf-counter access; run nvidia-modprobe -u -c=0 as root or enable counters via /etc/modprobe.d/. The eval server is expected to have this pre-enabled.]"
        )
    elif "ncu not found" in combined:
        hint = " [hint: Nsight Compute (ncu) is not installed or not on PATH]"
    return f"ncu profile failed (rc={result.rc}): {combined.strip()[-1200:]}{hint}"


def profile(binary: Path, metrics: list[str], *, args: list[str] | None = None,
            timeout_s: float = 120.0) -> NcuResult:
    return profile_command([str(binary), *(args or [])], metrics, timeout_s=timeout_s)


def profile_command(
    cmd: list[str],
    metrics: list[str],
    *,
    timeout_s: float = 120.0,
) -> NcuResult:
    ncu = find_ncu()
    if ncu is None:
        return NcuResult(False, -1, "", "ncu not found on PATH", [])
    ncu_cmd = [
        ncu, "-f", "--target-processes", "all",
        "--metrics", ",".join(metrics), "--csv",
        *cmd,
    ]
    try:
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return NcuResult(False, -1, exc.stdout or "", f"ncu timeout after {timeout_s}s", [])
    rows = _parse_csv(result.stdout) if result.returncode == 0 else []
    return NcuResult(result.returncode == 0, result.returncode, result.stdout, result.stderr, rows)
