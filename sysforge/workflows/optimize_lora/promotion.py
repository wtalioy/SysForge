from __future__ import annotations

import math
from dataclasses import dataclass

from .models import BenchmarkTierSummary


TIER1 = "tier1"
TIER2 = "tier2"
TIER3 = "tier3"

DEFAULT_NOISE_GUARD_PCT = 1.5
DEFAULT_NOISE_GUARD_MS = 0.05
DEFAULT_MAX_REGRESSION_PCT = 2.0


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    improvement_pct: float
    worst_regression_pct: float
    noise_guard_pct: float
    close_result: bool = False


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values))


def threshold_pct_for_summaries(
    challenger: BenchmarkTierSummary,
    incumbent: BenchmarkTierSummary,
    *,
    noise_guard_pct: float = DEFAULT_NOISE_GUARD_PCT,
    noise_guard_ms: float = DEFAULT_NOISE_GUARD_MS,
) -> float:
    largest_ms = max(incumbent.largest_shape_median_ms, challenger.largest_shape_median_ms, 1e-9)
    ms_equivalent_pct = (noise_guard_ms / largest_ms) * 100.0
    return max(noise_guard_pct, ms_equivalent_pct)


def summarize_regression_pct(summary: BenchmarkTierSummary) -> float:
    regressions = []
    for shape_result in summary.shape_results:
        if shape_result.speedup_vs_best is None:
            continue
        regressions.append(max(0.0, (1.0 - shape_result.speedup_vs_best) * 100.0))
    return max(regressions, default=0.0)


def compare_tier_summaries(
    challenger: BenchmarkTierSummary,
    incumbent: BenchmarkTierSummary,
    *,
    noise_guard_pct: float = DEFAULT_NOISE_GUARD_PCT,
    noise_guard_ms: float = DEFAULT_NOISE_GUARD_MS,
    max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
) -> PromotionDecision:
    threshold_pct = threshold_pct_for_summaries(
        challenger,
        incumbent,
        noise_guard_pct=noise_guard_pct,
        noise_guard_ms=noise_guard_ms,
    )
    challenger.worst_regression_pct = summarize_regression_pct(challenger)
    incumbent_best = max(incumbent.geometric_mean_speedup, 1e-12)
    improvement_pct = ((challenger.geometric_mean_speedup / incumbent_best) - 1.0) * 100.0
    challenger.improvement_over_best_pct = improvement_pct
    challenger.noise_guard_pct = threshold_pct

    if not challenger.correctness_passed:
        return PromotionDecision(
            promote=False,
            reason="reject_correctness_failed",
            improvement_pct=improvement_pct,
            worst_regression_pct=challenger.worst_regression_pct,
            noise_guard_pct=threshold_pct,
        )
    if challenger.worst_regression_pct > max_regression_pct:
        return PromotionDecision(
            promote=False,
            reason="keep_incumbent_regression",
            improvement_pct=improvement_pct,
            worst_regression_pct=challenger.worst_regression_pct,
            noise_guard_pct=threshold_pct,
        )
    if improvement_pct > threshold_pct:
        return PromotionDecision(
            promote=True,
            reason="promote_clear_win",
            improvement_pct=improvement_pct,
            worst_regression_pct=challenger.worst_regression_pct,
            noise_guard_pct=threshold_pct,
        )
    if abs(improvement_pct) <= threshold_pct:
        return PromotionDecision(
            promote=False,
            reason="keep_incumbent_close_result",
            improvement_pct=improvement_pct,
            worst_regression_pct=challenger.worst_regression_pct,
            noise_guard_pct=threshold_pct,
            close_result=True,
        )
    return PromotionDecision(
        promote=False,
        reason="keep_incumbent_slower",
        improvement_pct=improvement_pct,
        worst_regression_pct=challenger.worst_regression_pct,
        noise_guard_pct=threshold_pct,
    )


def separation_pct(
    leader: BenchmarkTierSummary,
    challenger: BenchmarkTierSummary,
) -> float:
    challenger_speedup = max(challenger.geometric_mean_speedup, 1e-12)
    return ((leader.geometric_mean_speedup / challenger_speedup) - 1.0) * 100.0


def select_best_finalist(
    finalists: list[tuple[str, BenchmarkTierSummary]],
) -> tuple[str, BenchmarkTierSummary] | None:
    viable = [
        (candidate_id, summary)
        for candidate_id, summary in finalists
        if summary.correctness_passed
    ]
    if not viable:
        return None
    return min(
        viable,
        key=lambda item: (
            -item[1].geometric_mean_speedup,
            item[1].largest_shape_median_ms,
            item[1].worst_regression_pct,
            item[0],
        ),
    )
