from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from time import strftime

from ..core.runtime import RuntimeContext
from ..profiling.analysis import AnalysisRunner
from ..profiling.models import ProbeRecord
from ..profiling.probe_runner import ProbeCoordinator
from ..profiling.targets import load_target_spec, partition_targets
from .base import Workflow, WorkflowResult


@dataclass
class ProfilingWorkflowResult(WorkflowResult):
    target_spec_path: str
    env_hints: dict
    targets_input: list[str] = field(default_factory=list)
    routed: dict[str, list[str]] = field(default_factory=dict)
    hardware: dict[str, dict] = field(default_factory=dict)
    analysis: dict | None = None
    trace: list[dict] = field(default_factory=list)


class ProfilingWorkflowRunner:
    def _record_probe_result(self, result: ProfilingWorkflowResult, target: str, outcome) -> None:
        result.hardware[target] = ProbeRecord.from_outcome(outcome).to_output()
        result.trace.append({
            "target": target,
            "attempts": [attempt.to_dict() for attempt in outcome.attempts],
        })

    def _promote_analysis_metrics(self, result: ProfilingWorkflowResult, analysis) -> None:
        for name, entry in (analysis.per_metric or {}).items():
            samples = int(entry.get("samples") or 0)
            has_value = entry.get("value") is not None
            confidence = 0.95 if (has_value and samples >= 1) else 0.0
            result.hardware[name] = {
                "value": entry.get("value"),
                "unit": entry.get("unit", ""),
                "confidence": confidence,
                "samples": samples,
                "min": entry.get("min"),
                "max": entry.get("max"),
                "source": "ncu",
                "error": entry.get("error", ""),
            }

    def run(self, context: RuntimeContext) -> ProfilingWorkflowResult:
        config = context.config
        result = ProfilingWorkflowResult(
            workflow="profiling",
            started_at=context.started_at,
            finished_at=context.started_at,
            target_spec_path=str(config.target_spec_path),
            env_hints=context.env_hints,
        )

        try:
            targets = load_target_spec(config.target_spec_path)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"failed to read target_spec: {exc}")
            targets = []
        result.targets_input = targets

        probe_targets, metric_targets = partition_targets(targets)
        result.routed = {"probe": probe_targets, "analysis_metrics": metric_targets}

        if probe_targets:
            probe_coordinator = ProbeCoordinator(config, context.workspace, context.env_hints)
            for target in probe_targets:
                try:
                    outcome = probe_coordinator.solve(target)
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"probe {target}: {exc}\n{traceback.format_exc()}")
                    continue
                self._record_probe_result(result, target, outcome)

        if metric_targets:
            try:
                analysis = AnalysisRunner(config, context.workspace).run(metric_targets)
                result.analysis = analysis.to_dict()
                self._promote_analysis_metrics(result, analysis)
                if analysis.error:
                    result.errors.append(f"analysis: {analysis.error}")
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"analysis: {exc}\n{traceback.format_exc()}")

        result.finished_at = strftime("%Y-%m-%dT%H:%M:%S")
        return result


def ProfilingWorkflow() -> Workflow:
    return Workflow(
        name="profiling",
        description="Profile probe targets and NCU metrics.",
        runner=ProfilingWorkflowRunner().run,
    )
