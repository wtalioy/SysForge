import pytest

from sysforge.agent.base import BaseAgent
from sysforge.workflows import registry


def test_register_workflow_adds_agent_runner_without_concrete_workflow_assumptions(monkeypatch):
    monkeypatch.setattr(registry, "_WORKFLOW_DEFS", [])
    monkeypatch.setattr(registry, "_discover_workflow_modules", lambda: None)

    @registry.register_workflow(name="sample", description="Sample workflow")
    class SampleAgent(BaseAgent):
        def run(self):
            return {"workflow": "sample", "trace": self.trace}

    workflows = registry.build_registry()

    assert set(workflows) == {"sample"}
    assert workflows["sample"].description == "Sample workflow"
    assert workflows["sample"].runner.__name__ == "run_SampleAgent"


def test_build_registry_rejects_duplicate_workflow_names(monkeypatch):
    monkeypatch.setattr(registry, "_discover_workflow_modules", lambda: None)
    monkeypatch.setattr(
        registry,
        "_WORKFLOW_DEFS",
        [
            ("duplicate", "first", lambda context: None),
            ("duplicate", "second", lambda context: None),
        ],
    )

    with pytest.raises(ValueError, match="workflow 'duplicate' is already registered"):
        registry.build_registry()
