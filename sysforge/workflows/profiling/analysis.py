from __future__ import annotations

import json

from .prompting import analyze_metrics, fix_reference_compile, generate_reference_gemm
from ...agent.llm import LLMError
from ...runtime import Config
from ...integrations import compiler, executor, ncu
from ...integrations.workspace import Workspace
from ..common import tail_text
from .models import AnalysisOutcome


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
        output.error = f"gemm compile failed: {tail_text(compile_result.stderr, 500)}"
        return output

    run_result = executor.run_binary(binary_path, timeout_s=config.run_timeout_s)
    if not run_result.ok:
        output.error = f"gemm run failed rc={run_result.rc}: {tail_text(run_result.stderr, 500)}"
        return output

    ncu_result = ncu.profile(binary_path, metrics, timeout_s=config.ncu_timeout_s)
    if not ncu_result.ok:
        output.error = ncu.format_error(ncu_result)
        output.ncu_csv_tail = tail_text(ncu_result.stdout, 2000)
        return output

    output.ncu_csv_tail = tail_text(ncu_result.stdout, 4000)
    output.per_metric = ncu.aggregate_per_metric(ncu_result.rows, metrics)

    per_metric_compact = {
        metric: {"value": entry.get("value"), "unit": entry.get("unit"), "samples": entry.get("samples")}
        for metric, entry in output.per_metric.items()
    }
    try:
        diagnosis = analyze_metrics(
            metrics=json.dumps(metrics, indent=2),
            ncu_csv=json.dumps(per_metric_compact, indent=2) + "\n\n--- raw CSV ---\n" + tail_text(ncu_result.stdout, 4000),
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
