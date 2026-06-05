from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from ..common import CommandResult, tail_text
from .benchmark import aggregate_benchmark_summaries, benchmark_case_names, parse_benchmark_file, run_benchmark
from .correctness import run_correctness_case
from .defaults import DEFAULT_RUNTIME_WORKFLOW_CONFIG, RuntimeWorkflowConfig
from .models import RuntimeBenchmarkSummary
from .runtime_io import write_json_file


@dataclass
class RuntimeHarnessResult(CommandResult):
    benchmark: RuntimeBenchmarkSummary | None = None


class OptimizeRuntimeHarness:
    def __init__(
        self,
        logs_dir: Path,
        config: RuntimeWorkflowConfig = DEFAULT_RUNTIME_WORKFLOW_CONFIG,
    ) -> None:
        self.config = config
        self.model_config_path = config.target_dir / "model_config.json"
        self.weight_dir = config.target_dir / "weights"
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def run_correctness(self, engine_path: Path, *, label: str, case: str = "basic") -> RuntimeHarnessResult:
        return self._run_and_log(
            label=f"{label}_{case}_correctness",
            fn=lambda: run_correctness_case(
                engine_path=str(engine_path),
                model_config_path=str(self.model_config_path),
                weight_dir=str(self.weight_dir),
                case=case,
                device=self.config.device,
            ),
        )

    def run_benchmark_suite(self, engine_path: Path, *, label: str) -> RuntimeHarnessResult:
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
                    required_cases=set(benchmark_case_names(self.config.benchmark_case_mode)),
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
            model_config_path=str(self.model_config_path),
            weight_dir=str(self.weight_dir),
            device=self.config.device,
            warmup=self.config.benchmark_warmup,
            repeat=self.config.benchmark_repeat,
            case_mode=self.config.benchmark_case_mode,
        )
        write_json_file(output_json, payload)
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
