from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Attempt:
    version: int
    phase: str
    source_path: str
    compile_ok: bool
    compile_stderr: str
    run_ok: bool
    run_rc: int
    run_stdout_tail: str
    run_stderr_tail: str
    extracted: dict[str, Any] | None
    plausible: bool
    reject_reason: str
    rationale: str


@dataclass
class ProbeOutcome:
    target: str
    unit: str
    value: float | None
    confidence: float
    reasoning: str
    attempts: list[Attempt] = field(default_factory=list)
    accepted_samples: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class AnalysisOutcome:
    metrics_requested: list[str]
    per_metric: dict[str, dict[str, Any]] = field(default_factory=dict)
    bottleneck: str = "unknown"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""
    ncu_csv_tail: str = ""
    error: str = ""
