from __future__ import annotations

from dataclasses import dataclass
from time import strftime

from ..integrations.gpu_info import weak_hint
from ..integrations.workspace import Workspace
from .config import Config


@dataclass(frozen=True)
class RuntimeContext:
    config: Config
    workspace: Workspace
    env_hints: dict
    started_at: str


def build_runtime_context(config: Config) -> RuntimeContext:
    workspace = Workspace(config.probes_dir, config.build_dir, config.logs_dir)
    return RuntimeContext(
        config=config,
        workspace=workspace,
        env_hints=weak_hint(),
        started_at=strftime("%Y-%m-%dT%H:%M:%S"),
    )
