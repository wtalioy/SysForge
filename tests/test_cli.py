from dataclasses import dataclass

from sysforge.cli.app import build_parser, main
from sysforge.cli.app import write_output


class FakeWorkflow:
    name = "sample"
    description = "fake"

    def run(self, context):
        return {"workflow": self.name, "started_at": context.started_at}


def test_build_parser_uses_registered_workflow_names(monkeypatch):
    monkeypatch.setattr("sysforge.cli.app.build_registry", lambda: {"sample": FakeWorkflow()})

    parser = build_parser()
    args = parser.parse_args(["sample"])

    assert args.workflow == "sample"


def test_cli_dispatches_selected_workflow(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    (tmp_path / "target").mkdir()

    captured = {}

    def fake_write_output(path, payload):
        captured["path"] = path
        captured["payload"] = payload

    monkeypatch.setattr("sysforge.cli.app.build_registry", lambda: {"sample": FakeWorkflow()})
    monkeypatch.setattr("sysforge.cli.app.write_output", fake_write_output)
    rc = main(["sample"])
    assert rc == 0
    assert captured["payload"]["workflow"] == "sample"


def test_write_output_serializes_dataclasses_and_plain_payloads(tmp_path):
    @dataclass
    class Payload:
        workflow: str
        count: int

    dataclass_path = tmp_path / "nested" / "dataclass.json"
    write_output(dataclass_path, Payload(workflow="sample", count=2))
    assert '"workflow": "sample"' in dataclass_path.read_text(encoding="utf-8")

    plain_path = tmp_path / "plain.json"
    write_output(plain_path, {"ok": True})
    assert plain_path.read_text(encoding="utf-8").strip() == '{\n  "ok": true\n}'
