from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...agent import BaseAgent
from ..base import WorkflowResult
from ..common import append_workflow_error, workflow_timestamp
from ..registry import register_workflow
from .analysis import run_analysis
from .probe_runner import ProbeCoordinator
from .targets import load_target_spec, partition_targets


@dataclass
class ProfilingWorkflowResult(WorkflowResult):
    target_spec_path: str
    env_hints: dict
    targets_input: list[str] = field(default_factory=list)
    routed: dict[str, list[str]] = field(default_factory=dict)
    hardware: dict[str, dict] = field(default_factory=dict)
    analysis: dict | None = None
    trace: list[dict] = field(default_factory=list)


@register_workflow(
    name="profiling",
    description="Profile probe targets and NCU metrics.",
)
class ProfilingAgent(BaseAgent):
    def run(self) -> ProfilingWorkflowResult:
        config = self.context.config
        result = ProfilingWorkflowResult(
            workflow="profiling",
            started_at=self.context.started_at,
            finished_at=self.context.started_at,
            target_spec_path=str(config.target_spec_path),
            env_hints=self.context.env_hints,
        )

        try:
            targets = load_target_spec(config.target_spec_path)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"failed to read target_spec: {exc}")
            targets = []
        result.targets_input = targets

        probe_targets, metric_targets = partition_targets(targets)
        result.routed = {"probe": probe_targets, "analysis_metrics": metric_targets}
        self.record_trace(action="targets_partitioned", probe=len(probe_targets), metrics=len(metric_targets))

        if probe_targets:
            probe_coordinator = ProbeCoordinator(config, self.context.workspace, self.context.env_hints)
            for target in probe_targets:
                try:
                    outcome = probe_coordinator.solve(target)
                except Exception as exc:  # noqa: BLE001
                    append_workflow_error(result, f"probe {target}", exc)
                    continue
                result.hardware[target] = {
                    "value": outcome.value,
                    "unit": outcome.unit,
                    "confidence": outcome.confidence,
                    "reasoning": outcome.reasoning,
                    "accepted_samples": outcome.accepted_samples,
                    "source": "probe",
                    "error": outcome.error,
                }
                result.trace.append({
                    "target": target,
                    "attempts": [asdict(attempt) for attempt in outcome.attempts],
                })

        if metric_targets:
            try:
                analysis = run_analysis(config, self.context.workspace, metric_targets)
                result.analysis = asdict(analysis)
                for name, entry in (analysis.per_metric or {}).items():
                    samples = int(entry.get("samples") or 0)
                    has_value = entry.get("value") is not None
                    result.hardware[name] = {
                        "value": entry.get("value"),
                        "unit": entry.get("unit", ""),
                        "confidence": 0.95 if (has_value and samples >= 1) else 0.0,
                        "samples": samples,
                        "min": entry.get("min"),
                        "max": entry.get("max"),
                        "source": "ncu",
                        "error": entry.get("error", ""),
                    }
                if analysis.error:
                    result.errors.append(f"analysis: {analysis.error}")
            except Exception as exc:  # noqa: BLE001
                append_workflow_error(result, "analysis", exc)

        result.finished_at = workflow_timestamp()
        return result
