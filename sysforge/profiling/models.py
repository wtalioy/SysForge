from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
class ProbeRecord:
    value: float | None
    unit: str
    confidence: float
    reasoning: str
    accepted_samples: list[dict[str, Any]]
    source: str
    error: str

    @classmethod
    def from_outcome(cls, outcome: ProbeOutcome) -> "ProbeRecord":
        return cls(
            value=outcome.value,
            unit=outcome.unit,
            confidence=outcome.confidence,
            reasoning=outcome.reasoning,
            accepted_samples=outcome.accepted_samples,
            source="probe",
            error=outcome.error,
        )

    def to_output(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
