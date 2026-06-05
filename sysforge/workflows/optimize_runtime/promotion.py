from __future__ import annotations

import shutil
from pathlib import Path

from .models import RuntimeBenchmarkSummary, RuntimeCandidateRecord
from .runtime_io import load_engine_module


def candidate_score(candidate: RuntimeCandidateRecord) -> tuple[float, float, float, float, float, float]:
    benchmark = candidate.benchmark or RuntimeBenchmarkSummary()
    case_tps = benchmark.case_tokens_per_second
    case_decode_tps = benchmark.case_decode_tokens_per_second
    return (
        benchmark.mixed_tokens_per_second,
        case_tps.get("churn", 0.0),
        case_decode_tps.get("long_decode", 0.0),
        benchmark.decode_tokens_per_second,
        case_tps.get("varied_prefill", 0.0),
        -benchmark.peak_memory_mb,
    )


def choose_winner(
    candidates: list[RuntimeCandidateRecord],
    *,
    incumbent_id: str = "kv-cache-baseline",
    noise_guard: float = 0.01,
    robust_regression_guard: float = 0.10,
) -> RuntimeCandidateRecord | None:
    eligible = [candidate for candidate in candidates if candidate.correctness_passed and candidate.benchmark is not None]
    if not eligible:
        return None
    incumbent = next((candidate for candidate in eligible if candidate.candidate_id == incumbent_id), None)
    ranked = sorted(eligible, key=candidate_score, reverse=True)
    if incumbent is None:
        return ranked[0]
    for challenger in ranked:
        if challenger.candidate_id == incumbent.candidate_id:
            continue
        if _passes_promotion_guard(
            incumbent,
            challenger,
            noise_guard=noise_guard,
            robust_regression_guard=robust_regression_guard,
        ):
            return challenger
    return incumbent


def _passes_promotion_guard(
    incumbent: RuntimeCandidateRecord,
    challenger: RuntimeCandidateRecord,
    *,
    noise_guard: float,
    robust_regression_guard: float,
) -> bool:
    incumbent_score = candidate_score(incumbent)[0]
    challenger_score = candidate_score(challenger)[0]
    if incumbent_score <= 0 and challenger_score <= 0:
        return False
    if incumbent_score > 0:
        incumbent_spread = (incumbent.benchmark or RuntimeBenchmarkSummary()).spread_pct.get("mixed", 0.0) / 100.0
        challenger_spread = (challenger.benchmark or RuntimeBenchmarkSummary()).spread_pct.get("mixed", 0.0) / 100.0
        effective_guard = max(
            noise_guard,
            challenger_spread,
        )
        mixed_gain = (challenger_score - incumbent_score) / incumbent_score
        robust_regression = _has_robust_regression(
            incumbent,
            challenger,
            regression_guard=robust_regression_guard,
        )
        if challenger_spread > incumbent_spread:
            effective_guard = max(effective_guard, incumbent_spread)
        elif robust_regression:
            effective_guard = max(effective_guard, incumbent_spread)
        if mixed_gain < effective_guard:
            return False
        if mixed_gain < robust_regression_guard and robust_regression:
            return False
    return True


def _has_robust_regression(
    incumbent: RuntimeCandidateRecord,
    challenger: RuntimeCandidateRecord,
    *,
    regression_guard: float,
) -> bool:
    incumbent_benchmark = incumbent.benchmark or RuntimeBenchmarkSummary()
    challenger_benchmark = challenger.benchmark or RuntimeBenchmarkSummary()
    checks = (
        (
            incumbent_benchmark.decode_tokens_per_second,
            challenger_benchmark.decode_tokens_per_second,
        ),
        (
            incumbent_benchmark.prefill_tokens_per_second,
            challenger_benchmark.prefill_tokens_per_second,
        ),
        (
            incumbent_benchmark.case_tokens_per_second.get("churn", 0.0),
            challenger_benchmark.case_tokens_per_second.get("churn", 0.0),
        ),
        (
            incumbent_benchmark.case_decode_tokens_per_second.get("long_decode", 0.0),
            challenger_benchmark.case_decode_tokens_per_second.get("long_decode", 0.0),
        ),
        (
            incumbent_benchmark.case_tokens_per_second.get("varied_prefill", 0.0),
            challenger_benchmark.case_tokens_per_second.get("varied_prefill", 0.0),
        ),
    )
    for incumbent_value, challenger_value in checks:
        if incumbent_value > 0 and challenger_value < incumbent_value * (1.0 - regression_guard):
            return True
    return False


def promote_candidate(candidate: RuntimeCandidateRecord, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate.engine_path, destination)
    smoke_import(destination)
    return destination


def smoke_import(engine_path: Path) -> None:
    module = load_engine_module(str(engine_path))
    if not hasattr(module, "create_engine"):
        raise AttributeError("promoted engine does not define create_engine")
