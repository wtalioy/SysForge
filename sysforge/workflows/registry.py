from __future__ import annotations

from dataclasses import dataclass

from .base import Workflow
from .optimize_lora import OptimizeLoraWorkflow
from .profiling import ProfilingWorkflow


@dataclass(frozen=True)
class WorkflowRegistry:
    workflows: dict[str, Workflow]

    def get(self, name: str) -> Workflow:
        return self.workflows[name]

    def items(self):
        return self.workflows.items()


def build_registry() -> WorkflowRegistry:
    workflows = {
        workflow.name: workflow
        for workflow in [ProfilingWorkflow(), OptimizeLoraWorkflow()]
    }
    return WorkflowRegistry(workflows=workflows)
