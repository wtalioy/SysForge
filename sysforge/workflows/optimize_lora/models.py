from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..base import WorkflowResult

if TYPE_CHECKING:
    from .profiling import ProfileSummary


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    values: tuple[Any, ...]
    default: Any | None = None
    description: str = ""


@dataclass(frozen=True)
class CandidateFamilyDraft:
    family_name: str
    source_template: str
    parameters: tuple[ParameterSpec, ...] = ()
    concrete_variants: tuple[dict[str, Any], ...] = ()
    rationale: str = ""
    expected_bottleneck: str = ""


@dataclass
class ConcreteCandidate:
    candidate_id: str
    family_name: str
    source: str
    parameter_values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateFeedback:
    candidate_id: str
    parameter_values: dict[str, Any]
    outcome: str
    summary: str
    speedup: float | None = None
    weakest_shape: int | None = None


@dataclass(frozen=True)
class RoundFeedback:
    round_index: int
    family_name: str
    improved: bool
    summary: str = ""
    best_tier2_speedup: float = 0.0
    second_tier2_speedup: float = 0.0
    challenger_margin_pct: float = 0.0
    challenger_separation_guard_pct: float = 0.0
    close_frontier: bool = False


@dataclass
class BenchmarkResult:
    shape_d: int
    warmup: int
    iters: int
    median_ms: float
    min_ms: float = 0.0
    max_ms: float = 0.0
    spread_pct: float = 0.0


@dataclass
class ShapeBenchmarkResult(BenchmarkResult):
    reference_median_ms: float = 0.0
    speedup_vs_reference: float = 0.0
    speedup_vs_best: float | None = None


@dataclass
class BenchmarkTierSummary:
    tier_name: str
    shapes: list[int]
    correctness_passed: bool
    shape_results: list[ShapeBenchmarkResult] = field(default_factory=list)
    max_abs_err: float = 0.0
    rel_l2_err: float = 0.0
    geometric_mean_speedup: float = 0.0
    worst_regression_pct: float = 0.0
    improvement_over_best_pct: float = 0.0
    largest_shape_median_ms: float = 0.0
    noise_guard_pct: float = 0.0
    rerun_used: bool = False
    failure_reason: str = ""


@dataclass
class CandidateCompileResult:
    status: str
    source_hash: str
    module_name: str
    source_path: str
    build_dir: str
    log_path: str
    error: str = ""
    duration_s: float = 0.0


@dataclass
class CandidateEvaluation:
    shape_d: int = 0
    correctness_passed: bool = False
    max_abs_err: float = 0.0
    rel_l2_err: float = 0.0
    benchmark: BenchmarkResult | None = None
    tier_summaries: list[BenchmarkTierSummary] = field(default_factory=list)

    def add_tier_summary(self, summary: BenchmarkTierSummary) -> None:
        self.tier_summaries = [existing for existing in self.tier_summaries if existing.tier_name != summary.tier_name]
        prior_correctness = self.correctness_passed
        self.tier_summaries.append(summary)
        self.correctness_passed = (
            summary.correctness_passed
            if len(self.tier_summaries) == 1
            else prior_correctness and summary.correctness_passed
        )
        self.max_abs_err = max(self.max_abs_err, summary.max_abs_err)
        self.rel_l2_err = max(self.rel_l2_err, summary.rel_l2_err)
        if summary.shape_results:
            first = summary.shape_results[0]
            self.shape_d = first.shape_d
            self.benchmark = BenchmarkResult(
                shape_d=first.shape_d,
                warmup=first.warmup,
                iters=first.iters,
                median_ms=first.median_ms,
                min_ms=first.min_ms,
                max_ms=first.max_ms,
                spread_pct=first.spread_pct,
            )

    def tier(self, name: str) -> BenchmarkTierSummary | None:
        for summary in self.tier_summaries:
            if summary.tier_name == name:
                return summary
        return None


@dataclass
class CandidateRecord:
    candidate_id: str
    family: str
    source_hash: str
    module_name: str
    source_path: str
    entrypoint_name: str = "forward"
    origin: str = "seed"
    comparison_summary: str = ""
    created_at: str = ""
    updated_at: str = ""
    compile: CandidateCompileResult | None = None
    evaluation: CandidateEvaluation | None = None
    failure_stage: str = ""
    failure_summary: str = ""
    profile_summary: ProfileSummary | None = None
    parameter_values: dict[str, object] = field(default_factory=dict)


@dataclass
class OptimizeLoraResult(WorkflowResult):
    status: str
    summary: str
    submission_root: str
    promoted_source_path: str
    artifact_created: bool
    bootstrap_family: str = "baseline"
    verified_baseline: bool = False
    current_best_candidate_id: str = ""
    current_best_family: str = ""
    validation_shape: int = 0
    benchmark_summary: BenchmarkResult | None = None
    benchmark_tiers: list[BenchmarkTierSummary] = field(default_factory=list)
    finalist_summary: BenchmarkTierSummary | None = None
    best_candidate_kind: str = "baseline"
    seed_count_run: int = 0
    mutation_count_run: int = 0
    seed_variants_screened: int = 0
    mutation_variants_screened: int = 0
    winner_confirmed: bool = False
    winner_confirmation_candidate_id: str = ""
    compile_time_s_total: float = 0.0
    variants_benchmarked: int = 0
    best_tier2_speedup: float = 0.0
    best_tier3_speedup: float = 0.0
    profiling_used: bool = False
    skipped_steps: list[dict] = field(default_factory=list)
    candidates: list[CandidateRecord] = field(default_factory=list)
    controller_trace: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
