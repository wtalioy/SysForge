from __future__ import annotations

import json
import statistics
from typing import Any

from .prompting import analyze_metrics, fix_reference_compile, generate_reference_gemm
from ...agent.llm import LLMError
from ...runtime import Config
from ...integrations import compiler, executor, ncu
from ...integrations.workspace import Workspace
from .models import AnalysisOutcome


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

    for name, entry in by_metric.items():
        raw_values = entry.get("raw") or []
        if raw_values:
            entry["value"] = statistics.median(raw_values)
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


def _compile_reference(config: Config, workspace: Workspace, source: str):
    source_path = workspace.write_source("gemm_reference", 1, source)
    binary_path = workspace.binary_path("gemm_reference", 1)
    compile_result = compiler.compile_cuda(
        source_path, binary_path, arch=config.nvcc_arch, timeout_s=config.compile_timeout_s
    )
    if compile_result.ok:
        return compile_result, binary_path
    fix = fix_reference_compile(source=source, compile_result=compile_result)
    fixed_source = fix.get("source") or ""
    source_path = workspace.write_source("gemm_reference", 2, fixed_source)
    binary_path = workspace.binary_path("gemm_reference", 2)
    compile_result = compiler.compile_cuda(
        source_path, binary_path, arch=config.nvcc_arch, timeout_s=config.compile_timeout_s
    )
    return compile_result, binary_path


def _format_ncu_error(ncu_result) -> str:
    combined = (ncu_result.stderr or "") + "\n" + (ncu_result.stdout or "")
    hint = ""
    if "ERR_NVGPUCTRPERM" in combined:
        hint = (
            " [hint: ncu lacks GPU perf-counter access; run nvidia-modprobe -u -c=0 as root or enable counters via /etc/modprobe.d/. The eval server is expected to have this pre-enabled.]"
        )
    elif "ncu not found" in combined:
        hint = " [hint: Nsight Compute (ncu) is not installed or not on PATH]"
    return f"ncu profile failed (rc={ncu_result.rc}): {combined.strip()[-1200:]}{hint}"


def run_analysis(config: Config, workspace: Workspace, metrics: list[str]) -> AnalysisOutcome:
    output = AnalysisOutcome(metrics_requested=list(metrics))
    try:
        source = generate_reference_gemm()
    except LLMError as exc:
        output.error = f"LLM generate_gemm failed: {exc}"
        return output

    try:
        compile_result, binary_path = _compile_reference(config, workspace, source)
    except LLMError as exc:
        output.error = f"LLM fix_compile failed: {exc}"
        return output
    if not compile_result.ok:
        output.error = f"gemm compile failed: {compile_result.stderr[-500:]}"
        return output

    run_result = executor.run_binary(binary_path, timeout_s=config.run_timeout_s)
    if not run_result.ok:
        output.error = f"gemm run failed rc={run_result.rc}: {run_result.stderr[-500:]}"
        return output

    ncu_result = ncu.profile(binary_path, metrics, timeout_s=config.ncu_timeout_s)
    if not ncu_result.ok:
        output.error = _format_ncu_error(ncu_result)
        output.ncu_csv_tail = ncu_result.stdout[-2000:]
        return output

    output.ncu_csv_tail = ncu_result.stdout[-4000:]
    output.per_metric = aggregate_per_metric(ncu_result.rows, metrics)

    per_metric_compact = {
        metric: {"value": entry.get("value"), "unit": entry.get("unit"), "samples": entry.get("samples")}
        for metric, entry in output.per_metric.items()
    }
    try:
        diagnosis = analyze_metrics(
            metrics=json.dumps(metrics, indent=2),
            ncu_csv=json.dumps(per_metric_compact, indent=2) + "\n\n--- raw CSV ---\n" + ncu_result.stdout[-4000:],
        )
        if isinstance(diagnosis, dict):
            output.bottleneck = str(diagnosis.get("bottleneck", "unknown"))
            evidence = diagnosis.get("evidence", [])
            if isinstance(evidence, list):
                output.evidence = evidence
            recommendations = diagnosis.get("recommendations", [])
            if isinstance(recommendations, list):
                output.recommendations = [str(item) for item in recommendations]
            output.summary = str(diagnosis.get("summary", ""))
    except LLMError as exc:
        output.error = f"LLM analyze_metrics failed: {exc}"
    return output
