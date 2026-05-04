from sysforge.core import build_runtime_context, load_config
from sysforge.workflows.registry import build_registry


def test_registry_exposes_expected_workflows():
    registry = build_registry()
    assert set(registry.workflows) >= {"profiling", "optimize-lora"}


def test_optimize_lora_workflow_is_scaffolded():
    registry = build_registry()
    workflow = registry.get("optimize-lora")
    assert workflow.name == "optimize-lora"
    assert "LoRA" in workflow.description or "lora" in workflow.description.lower()


def test_optimize_lora_workflow_creates_submission_root_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    (tmp_path / "target").mkdir()
    monkeypatch.chdir(tmp_path)

    registry = build_registry()
    workflow = registry.get("optimize-lora")
    context = build_runtime_context(load_config())

    result = workflow.run(context)

    artifact_path = tmp_path / "optimized_lora.cu"
    assert artifact_path.exists()
    assert result.status == "bootstrap_ready"
    assert result.artifact_created is True
    assert result.promoted_source_path == str(artifact_path)
    assert "torch::Tensor forward" in artifact_path.read_text(encoding="utf-8")
