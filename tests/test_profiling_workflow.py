from pathlib import Path

from sysforge.runtime import build_runtime_context, load_config
from sysforge.workflows.profiling.agent import ProfilingAgent, ProfilingWorkflowResult
from sysforge.workflows.profiling.models import AnalysisOutcome, Attempt, ProbeOutcome
from sysforge.workflows.registry import build_registry


def test_registry_workflow_delegates_to_profiling_agent(monkeypatch, tmp_path: Path):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir()
    workspace_dir.mkdir()
    (target_dir / "target_spec.json").write_text('{"targets":[]}', encoding="utf-8")
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    context = build_runtime_context(load_config())
    expected = ProfilingWorkflowResult(
        workflow="profiling",
        started_at=context.started_at,
        finished_at=context.started_at,
        target_spec_path=str(context.config.target_spec_path),
        env_hints=context.env_hints,
    )
    captured = {}

    def fake_run(self):
        captured["context"] = self.context
        return expected

    monkeypatch.setattr(ProfilingAgent, "run", fake_run)

    result = build_registry()["profiling"].run(context)

    assert result is expected
    assert captured["context"] is context


def test_profiling_agent_assembles_probe_and_analysis_output(monkeypatch, tmp_path: Path):
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

    probe_outcome = ProbeOutcome(
        target="dram_latency_cycles",
        unit="cycles",
        value=432.0,
        confidence=0.8,
        reasoning="good",
        attempts=[
            Attempt(
                version=1,
                phase="run",
                source_path="probe.cu",
                compile_ok=True,
                compile_stderr="",
                run_ok=True,
                run_rc=0,
                run_stdout_tail="RESULT 432",
                run_stderr_tail="",
                extracted={"value": 432.0},
                plausible=True,
                reject_reason="",
                rationale="ok",
            )
        ],
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

    monkeypatch.setattr(
        "sysforge.workflows.profiling.agent.ProbeCoordinator.solve",
        lambda self, target: probe_outcome,
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.agent.run_analysis",
        lambda config, workspace, metrics: analysis_outcome,
    )

    context = build_runtime_context(load_config())
    result = ProfilingAgent(context).run()

    assert result.workflow == "profiling"
    assert result.target_spec_path == str(context.config.target_spec_path)
    assert result.targets_input == [
        "dram_latency_cycles",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    assert result.routed["probe"] == ["dram_latency_cycles"]
    assert result.routed["analysis_metrics"] == ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert result.hardware["dram_latency_cycles"]["source"] == "probe"
    assert result.hardware["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["source"] == "ncu"
    assert result.analysis["bottleneck"] == "memory_bound"
    assert result.trace == [
        {
            "target": "dram_latency_cycles",
            "attempts": [
                {
                    "version": 1,
                    "phase": "run",
                    "source_path": "probe.cu",
                    "compile_ok": True,
                    "compile_stderr": "",
                    "run_ok": True,
                    "run_rc": 0,
                    "run_stdout_tail": "RESULT 432",
                    "run_stderr_tail": "",
                    "extracted": {"value": 432.0},
                    "plausible": True,
                    "reject_reason": "",
                    "rationale": "ok",
                }
            ],
        }
    ]
