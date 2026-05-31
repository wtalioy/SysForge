from __future__ import annotations

import json
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import CommandResult, tail_text
from .benchmark import run_benchmark
from .correctness import run_correctness as run_correctness_check
from .correctness import run_stress_correctness as run_stress_correctness_check
from .models import RuntimeBenchmarkSummary


@dataclass
class RuntimeHarnessResult(CommandResult):
    benchmark: RuntimeBenchmarkSummary | None = None


@dataclass(frozen=True)
class RuntimeHarnessConfig:
    model_config_path: Path
    weight_dir: Path
    submission_root: Path
    device: str = "auto"
    timeout_s: float = 360.0
    benchmark_warmup: int = 2
    benchmark_repeat: int = 5
    benchmark_runs: int = 3
    benchmark_discard_runs: int = 1
    benchmark_case_mode: str = "robust"
    run_stress: bool = True
    run_benchmark: bool = True

    @classmethod
    def from_env(cls, *, target_dir: Path, submission_root: Path) -> "RuntimeHarnessConfig":
        return cls(
            model_config_path=target_dir / "model_config.json",
            weight_dir=target_dir / "weights",
            submission_root=submission_root,
            device=os.environ.get("OPTIMIZE_RUNTIME_DEVICE", "auto"),
            timeout_s=float(os.environ.get("OPTIMIZE_RUNTIME_TIMEOUT_S", "360")),
            benchmark_warmup=int(os.environ.get("OPTIMIZE_RUNTIME_BENCHMARK_WARMUP", "2")),
            benchmark_repeat=int(os.environ.get("OPTIMIZE_RUNTIME_BENCHMARK_REPEAT", "5")),
            benchmark_runs=int(os.environ.get("OPTIMIZE_RUNTIME_BENCHMARK_RUNS", "3")),
            benchmark_discard_runs=int(os.environ.get("OPTIMIZE_RUNTIME_BENCHMARK_DISCARD_RUNS", "1")),
            benchmark_case_mode=os.environ.get("OPTIMIZE_RUNTIME_BENCHMARK_CASE_MODE", "robust"),
            run_stress=os.environ.get("OPTIMIZE_RUNTIME_STRESS", "1") == "1",
            run_benchmark=os.environ.get("OPTIMIZE_RUNTIME_BENCHMARK", "1") == "1",
        )


class OptimizeRuntimeHarness:
    def __init__(self, config: RuntimeHarnessConfig, logs_dir: Path) -> None:
        self.config = config
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def run_correctness(self, engine_path: Path, *, label: str) -> RuntimeHarnessResult:
        return self._run_and_log(
            label=f"{label}_correctness",
            fn=lambda: run_correctness_check(
                engine_path=str(engine_path),
                model_config_path=str(self.config.model_config_path),
                weight_dir=str(self.config.weight_dir),
                device=self.config.device,
            ),
        )

    def run_stress_correctness(self, engine_path: Path, *, label: str) -> RuntimeHarnessResult:
        return self._run_and_log(
            label=f"{label}_stress",
            fn=lambda: run_stress_correctness_check(
                engine_path=str(engine_path),
                model_config_path=str(self.config.model_config_path),
                weight_dir=str(self.config.weight_dir),
                device=self.config.device,
            ),
        )

    def run_quick_benchmark(self, engine_path: Path, *, label: str) -> RuntimeHarnessResult:
        run_results: list[RuntimeHarnessResult] = []
        parsed_runs: list[RuntimeBenchmarkSummary] = []
        kept_runs = max(1, self.config.benchmark_runs)
        discard_runs = max(0, self.config.benchmark_discard_runs)
        for run_index in range(discard_runs + kept_runs):
            output_json = self.logs_dir / f"{label}_benchmark_r{run_index + 1}.json"
            result = self._run_and_log(
                label=f"{label}_benchmark_r{run_index + 1}",
                fn=lambda output_json=output_json: self._benchmark_payload(engine_path, output_json),
            )
            run_results.append(result)
            if not result.passed:
                return result
            if run_index < discard_runs:
                continue
            try:
                parsed_runs.append(parse_benchmark_file(
                    output_json,
                    required_cases=required_cases_for_mode(self.config.benchmark_case_mode),
                ))
            except ValueError as exc:
                result.status = "failed"
                result.failure_summary = str(exc)
                return result
        final = run_results[-1]
        final.benchmark = aggregate_benchmark_summaries(parsed_runs)
        final.stdout_tail = "\n".join(result.stdout_tail for result in run_results)
        return final

    def _benchmark_payload(self, engine_path: Path, output_json: Path) -> list[dict[str, object]]:
        payload = run_benchmark(
            engine_path=str(engine_path),
            model_config_path=str(self.config.model_config_path),
            weight_dir=str(self.config.weight_dir),
            device=self.config.device,
            warmup=self.config.benchmark_warmup,
            repeat=self.config.benchmark_repeat,
            case_mode=self.config.benchmark_case_mode,
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _run_and_log(self, *, label: str, fn) -> RuntimeHarnessResult:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.logs_dir / f"{label}.stdout.log"
        stderr_path = self.logs_dir / f"{label}.stderr.log"
        start = time.monotonic()
        try:
            payload = fn()
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            stdout_path.write_text("", encoding="utf-8")
            stderr = traceback.format_exc()
            stderr_path.write_text(stderr, encoding="utf-8")
            return RuntimeHarnessResult(
                status="failed",
                command=[label],
                returncode=1,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                stderr_tail=tail_text(stderr),
                elapsed_s=elapsed,
                failure_summary=tail_text(str(exc)),
            )
        elapsed = time.monotonic() - start
        stdout = json.dumps(payload, indent=2)
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return RuntimeHarnessResult(
            status="passed",
            command=[label],
            returncode=0,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            stdout_tail=tail_text(stdout),
            elapsed_s=elapsed,
        )


def required_cases_for_mode(case_mode: str) -> set[str]:
    required_cases = {"prefill", "decode", "mixed"}
    if case_mode == "robust":
        required_cases.update({"varied_prefill", "long_decode", "churn"})
    return required_cases


def parse_benchmark_file(path: Path, *, required_cases: set[str] | None = None) -> RuntimeBenchmarkSummary:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark JSON file parse failed: {exc}") from exc
    return parse_benchmark_payload(payload, required_cases=required_cases)


def parse_benchmark_payload(payload: Any, *, required_cases: set[str] | None = None) -> RuntimeBenchmarkSummary:
    if not isinstance(payload, list):
        raise ValueError("benchmark JSON payload was not a list")
    required_cases = required_cases or {"prefill", "decode", "mixed"}
    by_case: dict[str, dict[str, Any]] = {str(item.get("case_name")): item for item in payload}
    if not required_cases.issubset(set(by_case)):
        missing = ", ".join(sorted(required_cases - set(by_case)))
        extra = ", ".join(sorted(set(by_case) - required_cases))
        raise ValueError(f"benchmark cases mismatch; missing={missing or 'none'} extra={extra or 'none'}")
    if len(by_case) != len(payload):
        raise ValueError("benchmark JSON payload contains duplicate case names")

    def metric(case_name: str, key: str) -> float:
        value = by_case.get(case_name, {}).get(key, 0.0)
        metric_value = float(value or 0.0)
        if not (metric_value >= 0.0 and metric_value < float("inf")):
            raise ValueError(f"benchmark metric {case_name}.{key} is not finite")
        return metric_value

    total_tps_values = [float(item.get("tokens_per_second") or 0.0) for item in payload]
    peak_values = [float(item.get("peak_memory_mb") or 0.0) for item in payload]
    case_tps = {case_name: metric(case_name, "tokens_per_second") for case_name in by_case}
    case_decode_tps = {case_name: metric(case_name, "decode_tokens_per_second") for case_name in by_case}
    return RuntimeBenchmarkSummary(
        prefill_tokens_per_second=metric("prefill", "tokens_per_second"),
        decode_tokens_per_second=metric("decode", "decode_tokens_per_second"),
        mixed_tokens_per_second=metric("mixed", "tokens_per_second"),
        total_tokens_per_second=sum(total_tps_values),
        peak_memory_mb=max(peak_values) if peak_values else 0.0,
        raw_results=payload,
        benchmark_runs=[payload],
        run_count=1,
        case_tokens_per_second=case_tps,
        case_decode_tokens_per_second=case_decode_tps,
    )


def aggregate_benchmark_summaries(summaries: list[RuntimeBenchmarkSummary]) -> RuntimeBenchmarkSummary:
    if not summaries:
        return RuntimeBenchmarkSummary(run_count=0)

    def median(values: list[float]) -> float:
        return float(statistics.median(values)) if values else 0.0

    def spread(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        med = median(values)
        if med == 0.0:
            return 0.0
        return (max(values) - min(values)) / med * 100.0

    prefill = [summary.prefill_tokens_per_second for summary in summaries]
    decode = [summary.decode_tokens_per_second for summary in summaries]
    mixed = [summary.mixed_tokens_per_second for summary in summaries]
    total = [summary.total_tokens_per_second for summary in summaries]
    peak = [summary.peak_memory_mb for summary in summaries]
    raw_runs = [summary.raw_results for summary in summaries]
    all_case_names = sorted({case_name for summary in summaries for case_name in summary.case_tokens_per_second})
    case_tps = {
        case_name: median([summary.case_tokens_per_second.get(case_name, 0.0) for summary in summaries])
        for case_name in all_case_names
    }
    case_decode_tps = {
        case_name: median([summary.case_decode_tokens_per_second.get(case_name, 0.0) for summary in summaries])
        for case_name in all_case_names
    }
    median_mixed = median(mixed)
    selected_index = min(
        range(len(summaries)),
        key=lambda index: abs(summaries[index].mixed_tokens_per_second - median_mixed),
    )
    return RuntimeBenchmarkSummary(
        prefill_tokens_per_second=median(prefill),
        decode_tokens_per_second=median(decode),
        mixed_tokens_per_second=median(mixed),
        total_tokens_per_second=median(total),
        peak_memory_mb=max(peak) if peak else 0.0,
        raw_results=summaries[selected_index].raw_results,
        benchmark_runs=raw_runs,
        run_count=len(summaries),
        spread_pct={
            "prefill": spread(prefill),
            "decode": spread(decode),
            "mixed": spread(mixed),
            "total": spread(total),
        },
        case_tokens_per_second=case_tps,
        case_decode_tokens_per_second=case_decode_tps,
    )
