from pathlib import Path

from sysforge.cli.app import build_parser, main


def test_build_parser_has_workflow_subcommands():
    parser = build_parser()
    args = parser.parse_args(["profiling"])
    assert args.workflow == "profiling"
    args2 = parser.parse_args(["optimize-lora"])
    assert args2.workflow == "optimize-lora"


def test_cli_dispatches_selected_workflow(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    (tmp_path / "target").mkdir()

    captured = {}

    def fake_write_output(path, payload):
        captured["path"] = path
        captured["payload"] = payload

    class FakeWorkflow:
        name = "profiling"
        description = "fake"

        def run(self, context):
            return {"workflow": self.name, "started_at": context.started_at}

    class FakeRegistry:
        def items(self):
            return [("profiling", FakeWorkflow())]

        def get(self, name):
            assert name == "profiling"
            return FakeWorkflow()

    monkeypatch.setattr("sysforge.cli.app.build_registry", lambda: FakeRegistry())
    monkeypatch.setattr("sysforge.cli.app.write_output", fake_write_output)
    rc = main(["profiling"])
    assert rc == 0
    assert captured["payload"]["workflow"] == "profiling"


def test_run_sh_invokes_optimize_lora():
    run_sh = Path(__file__).resolve().parent.parent / "run.sh"
    text = run_sh.read_text(encoding="utf-8")
    assert "python3 -m sysforge.main optimize-lora" in text
