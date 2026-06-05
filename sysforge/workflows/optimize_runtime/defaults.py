from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeWorkflowConfig:
    target_dir: Path = Path("target")
    device: str = "auto"
    benchmark_warmup: int = 2
    benchmark_repeat: int = 5
    benchmark_runs: int = 3
    benchmark_discard_runs: int = 1
    run_stress: bool = True
    run_benchmark: bool = True
    max_llm_rounds: int = 2
    max_llm_strategies_per_round: int = 2


DEFAULT_RUNTIME_WORKFLOW_CONFIG = RuntimeWorkflowConfig()
