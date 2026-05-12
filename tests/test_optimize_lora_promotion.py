from sysforge.workflows.optimize_lora.models import BenchmarkTierSummary, ShapeBenchmarkResult
from sysforge.workflows.optimize_lora.promotion import (
    compare_tier_summaries,
    separation_pct,
    select_best_finalist,
    threshold_pct_for_summaries,
)


def _summary(*, candidate_ms: float, reference_ms: float, speedup_vs_best: float, geo: float | None = None):
    speedup_vs_reference = reference_ms / candidate_ms
    return BenchmarkTierSummary(
        tier_name="tier2",
        shapes=[3584, 4608],
        correctness_passed=True,
        shape_results=[
            ShapeBenchmarkResult(
                shape_d=3584,
                warmup=1,
                iters=3,
                median_ms=candidate_ms,
                reference_median_ms=reference_ms,
                speedup_vs_reference=speedup_vs_reference,
                speedup_vs_best=speedup_vs_best,
            ),
            ShapeBenchmarkResult(
                shape_d=4608,
                warmup=1,
                iters=3,
                median_ms=candidate_ms,
                reference_median_ms=reference_ms,
                speedup_vs_reference=speedup_vs_reference,
                speedup_vs_best=speedup_vs_best,
            ),
        ],
        geometric_mean_speedup=geo if geo is not None else speedup_vs_reference,
        largest_shape_median_ms=candidate_ms,
    )


def test_threshold_pct_uses_larger_of_percent_and_ms_equivalent():
    incumbent = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0)
    challenger = _summary(candidate_ms=0.96, reference_ms=2.0, speedup_vs_best=1.01)
    assert threshold_pct_for_summaries(challenger, incumbent) >= 1.5


def test_compare_promotes_clear_win():
    incumbent = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0, geo=2.0)
    challenger = _summary(candidate_ms=0.9, reference_ms=2.0, speedup_vs_best=1.05, geo=2.25)
    decision = compare_tier_summaries(challenger, incumbent)
    assert decision.promote is True
    assert decision.reason == "promote_clear_win"


def test_compare_keeps_incumbent_for_close_result():
    incumbent = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0, geo=2.0)
    challenger = _summary(candidate_ms=0.995, reference_ms=2.0, speedup_vs_best=1.005, geo=2.01)
    decision = compare_tier_summaries(challenger, incumbent)
    assert decision.promote is False
    assert decision.reason == "keep_incumbent_close_result"
    assert decision.close_result is True


def test_separation_pct_reports_percent_gap_between_challengers():
    leader = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0, geo=2.04)
    runner_up = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0, geo=2.0)
    assert round(separation_pct(leader, runner_up), 6) == 2.0


def test_compare_rejects_major_regression():
    incumbent = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0, geo=2.0)
    challenger = _summary(candidate_ms=0.9, reference_ms=2.0, speedup_vs_best=0.96, geo=2.25)
    decision = compare_tier_summaries(challenger, incumbent)
    assert decision.promote is False
    assert decision.reason == "keep_incumbent_regression"


def test_select_best_finalist_prefers_speed_then_tiebreaks():
    first = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0, geo=2.1)
    first.worst_regression_pct = 0.8
    second = _summary(candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.0, geo=2.1)
    second.worst_regression_pct = 0.4
    winner = select_best_finalist([("candidate-a", first), ("candidate-b", second)])
    assert winner is not None
    assert winner[0] == "candidate-b"
