from pathlib import Path
from types import SimpleNamespace

from sysforge.runtime import load_config
from sysforge.integrations.workspace import Workspace
from sysforge.workflows.profiling.probe_runner import ProbeCoordinator


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace(
        probes_dir=tmp_path / "probes",
        build_dir=tmp_path / "build",
        logs_dir=tmp_path / "logs",
    )


def test_probe_coordinator_uses_regex_fallback_and_median_reruns(monkeypatch, tmp_path: Path):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir()
    workspace_dir.mkdir()
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    monkeypatch.setattr(
        "sysforge.workflows.profiling.prompting.generate_probe",
        lambda ctx, target, spec: {
            "source": "__global__ void k() {}",
            "args": [128],
            "parse_hint": "read RESULT line",
            "rationale": "baseline probe",
        },
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.prompting.extract",
        lambda ctx, target, spec, stdout, parse_hint: {
            "value": None,
            "unit": spec.unit,
            "confidence": 0.0,
            "reasoning": "llm missed it",
        },
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.probe_runner.compiler.compile_cuda",
        lambda *args, **kwargs: SimpleNamespace(ok=True, stderr="", stdout="", cmd=["nvcc"], duration_s=0.1),
    )

    run_outputs = iter(
        [
            SimpleNamespace(ok=True, rc=0, stdout="RESULT dram_latency_cycles=500 unit=cycles\n", stderr="", timed_out=False, wallclock_s=0.1),
            SimpleNamespace(ok=True, rc=0, stdout="RESULT dram_latency_cycles=400 unit=cycles\n", stderr="", timed_out=False, wallclock_s=0.1),
            SimpleNamespace(ok=True, rc=0, stdout="RESULT dram_latency_cycles=450 unit=cycles\n", stderr="", timed_out=False, wallclock_s=0.1),
        ]
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.probe_runner.executor.run_binary",
        lambda *args, **kwargs: next(run_outputs),
    )

    config = load_config()
    coordinator = ProbeCoordinator(config, _workspace(workspace_dir), hints={})
    outcome = coordinator.solve("dram_latency_cycles")

    assert outcome.error == ""
    assert outcome.value == 450.0
    assert outcome.unit == "cycles"
    assert outcome.confidence > 0.6
    assert len(outcome.accepted_samples) == 3
    assert outcome.accepted_samples[0]["value"] == 500.0
    assert "Median of 3 accepted sample(s)" in outcome.reasoning


def test_probe_coordinator_returns_best_effort_implausible_value(monkeypatch, tmp_path: Path):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir()
    workspace_dir.mkdir()
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    monkeypatch.setattr(
        "sysforge.workflows.profiling.prompting.generate_probe",
        lambda ctx, target, spec: {
            "source": "__global__ void k() {}",
            "args": [],
            "parse_hint": "",
            "rationale": "implausible probe",
        },
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.prompting.extract",
        lambda ctx, target, spec, stdout, parse_hint: {
            "value": 0.5,
            "unit": spec.unit,
            "confidence": 0.2,
            "reasoning": "parsed tiny value",
        },
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.probe_runner.compiler.compile_cuda",
        lambda *args, **kwargs: SimpleNamespace(ok=True, stderr="", stdout="", cmd=["nvcc"], duration_s=0.1),
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.probe_runner.executor.run_binary",
        lambda *args, **kwargs: SimpleNamespace(
            ok=True,
            rc=0,
            stdout="RESULT dram_latency_cycles=0.5 unit=cycles\n",
            stderr="",
            timed_out=False,
            wallclock_s=0.1,
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.profiling.probe_runner.consume_retry",
        lambda state, entry: False,
    )

    config = load_config()
    coordinator = ProbeCoordinator(config, _workspace(workspace_dir), hints={})
    outcome = coordinator.solve("dram_latency_cycles")

    assert outcome.error == ""
    assert outcome.value == 0.5
    assert outcome.unit == "cycles"
    assert outcome.confidence == 0.1
    assert "best-effort implausible value" in outcome.reasoning
