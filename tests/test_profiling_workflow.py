from pathlib import Path

from sysforge.core.config import load_config
from sysforge.core.runtime import build_runtime_context
from sysforge.profiling.models import AnalysisOutcome, ProbeOutcome
from sysforge.workflows.profiling import ProfilingWorkflowRunner


def test_profiling_workflow_assembles_probe_and_analysis_output(monkeypatch, tmp_path: Path):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir()
    workspace_dir.mkdir()
    (target_dir / "target_spec.json").write_text(
        '{"targets":["dram_latency_cycles","sm__throughput.avg.pct_of_peak_sustained_elapsed"]}'
    )
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    probe_outcome = ProbeOutcome(
        target="dram_latency_cycles",
        unit="cycles",
        value=432.0,
        confidence=0.8,
        reasoning="good",
        accepted_samples=[{"version": 1, "value": 432.0}],
    )
    analysis_outcome = AnalysisOutcome(
        metrics_requested=["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        per_metric={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": {
                "value": 82.7,
                "unit": "%",
                "samples": 2,
                "min": 82.6,
                "max": 82.8,
            }
        },
        bottleneck="memory_bound",
    )

    monkeypatch.setattr("sysforge.workflows.profiling.ProbeCoordinator.solve", lambda self, target: probe_outcome)
    monkeypatch.setattr("sysforge.workflows.profiling.AnalysisRunner.run", lambda self, metrics: analysis_outcome)

    context = build_runtime_context(load_config())
    result = ProfilingWorkflowRunner().run(context)

    assert result.workflow == "profiling"
    assert result.targets_input == [
        "dram_latency_cycles",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    assert result.routed["probe"] == ["dram_latency_cycles"]
    assert result.routed["analysis_metrics"] == ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert result.hardware["dram_latency_cycles"]["source"] == "probe"
    assert result.hardware["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["source"] == "ncu"
    assert result.analysis["bottleneck"] == "memory_bound"
