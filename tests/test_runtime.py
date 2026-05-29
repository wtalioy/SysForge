from sysforge.runtime import build_runtime_context, load_config


def test_load_config_creates_workspace_layout_from_environment(monkeypatch, tmp_path):
    target_dir = tmp_path / "target"
    workspace_dir = tmp_path / "workspace"
    target_dir.mkdir()
    monkeypatch.setenv("TARGET_DIR", str(target_dir))
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setenv("BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("BASE_MODEL", "model")
    monkeypatch.setenv("NVCC_ARCH", "sm_90")

    config = load_config()

    assert config.target_dir == target_dir
    assert config.workspace_dir == workspace_dir
    assert config.target_spec_path == target_dir / "target_spec.json"
    assert config.output_path == workspace_dir / "output.json"
    assert config.api_key == "key"
    assert config.base_url == "http://localhost:8000/v1"
    assert config.base_model == "model"
    assert config.nvcc_arch == "sm_90"
    assert config.build_dir.is_dir()
    assert config.probes_dir.is_dir()
    assert config.logs_dir.is_dir()


def test_build_runtime_context_wraps_config_workspace_and_environment_hints(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    (tmp_path / "target").mkdir()
    monkeypatch.setattr("sysforge.runtime.runtime.weak_hint", lambda: {"gpu": "stub"})

    config = load_config()
    context = build_runtime_context(config)

    assert context.config is config
    assert context.env_hints == {"gpu": "stub"}
    assert context.workspace.probes_dir == config.probes_dir
    assert context.workspace.build_dir == config.build_dir
    assert context.workspace.logs_dir == config.logs_dir
    assert "T" in context.started_at
