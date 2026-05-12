from __future__ import annotations

import importlib
import pkgutil
from typing import Callable

from ..agent import BaseAgent
from ..runtime import RuntimeContext

from .base import Workflow

WorkflowRunner = Callable[[RuntimeContext], object]
WorkflowRegistry = dict[str, Workflow]
_WORKFLOW_DEFS: list[tuple[str, str, WorkflowRunner]] = []


def _agent_runner(agent_cls: type[BaseAgent]) -> WorkflowRunner:
    def runner(context: RuntimeContext) -> object:
        return agent_cls(context).run()

    runner.__name__ = f"run_{agent_cls.__name__}"
    return runner


def register_workflow(*, name: str, description: str) -> Callable[[type[BaseAgent]], type[BaseAgent]]:
    def decorator(agent_cls: type[BaseAgent]) -> type[BaseAgent]:
        _WORKFLOW_DEFS.append((name, description, _agent_runner(agent_cls)))
        return agent_cls

    return decorator


def _discover_workflow_modules() -> None:
    package = importlib.import_module(__package__)
    for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{__package__}."):
        if module_info.name.endswith(".agent"):
            importlib.import_module(module_info.name)


def build_registry() -> WorkflowRegistry:
    _discover_workflow_modules()
    workflows: dict[str, Workflow] = {}
    for name, description, runner in _WORKFLOW_DEFS:
        if name in workflows:
            raise ValueError(f"workflow '{name}' is already registered")
        workflows[name] = Workflow(name=name, description=description, runner=runner)
    return workflows
