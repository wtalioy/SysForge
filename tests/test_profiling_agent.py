from pathlib import Path

from sysforge.runtime import build_runtime_context, load_config
from sysforge.workflows.profiling.agent import ProfilingAgent
from sysforge.workflows.profiling.models import AnalysisOutcome


def test_profiling_agent_collects_probe_and_analysis_errors(monkeypatch, tmp_path: Path):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir()
    workspace_dir.mkdir()
    (target_dir / "target_spec.json").write_text(
        '{"targets":["dram_latency_cycles","sm__throughput.avg.pct_of_peak_sustained_elapsed"]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    def fail_probe(self, target):
        raise RuntimeError(f"probe failed for {target}")

    analysis_outcome = AnalysisOutcome(
        metrics_requested=["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        per_metric={
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": {
                "value": 91.2,
                "unit": "%",
                "samples": 1,
            }
        },
        error="ncu summary incomplete",
    )

    monkeypatch.setattr("sysforge.workflows.profiling.agent.ProbeCoordinator.solve", fail_probe)
    monkeypatch.setattr(
        "sysforge.workflows.profiling.agent.run_analysis",
        lambda config, workspace, metrics: analysis_outcome,
    )

    context = build_runtime_context(load_config())
    result = ProfilingAgent(context).run()

    assert result.routed == {
        "probe": ["dram_latency_cycles"],
        "analysis_metrics": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    }
    assert result.hardware["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["confidence"] == 0.95
    assert result.hardware["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["source"] == "ncu"
    assert any("probe dram_latency_cycles: probe failed for dram_latency_cycles" in error for error in result.errors)
    assert "analysis: ncu summary incomplete" in result.errors


def test_profiling_agent_handles_unreadable_target_spec(monkeypatch, tmp_path: Path):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir()
    workspace_dir.mkdir()
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))
    monkeypatch.setattr(
        "sysforge.workflows.profiling.agent.load_target_spec",
        lambda path: (_ for _ in ()).throw(OSError("missing target spec")),
    )

    context = build_runtime_context(load_config())
    result = ProfilingAgent(context).run()

    assert result.targets_input == []
    assert result.routed == {"probe": [], "analysis_metrics": []}
    assert result.errors == ["failed to read target_spec: missing target spec"]
