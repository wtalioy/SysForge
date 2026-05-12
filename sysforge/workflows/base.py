from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..runtime import RuntimeContext


@dataclass
class WorkflowResult:
    workflow: str
    started_at: str
    finished_at: str
    errors: list[str] = field(default_factory=list, init=False)


@dataclass(frozen=True)
class Workflow:
    name: str
    description: str
    runner: Callable[[RuntimeContext], WorkflowResult]

    def run(self, context: RuntimeContext) -> WorkflowResult:
        return self.runner(context)
