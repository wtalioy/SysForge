from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .models import BenchmarkResult, BenchmarkTierSummary, CandidateEvaluation, ShapeBenchmarkResult
from .promotion import geometric_mean, summarize_regression_pct


PUBLIC_D_MIN = 3584
PUBLIC_D_MAX = 4608
DEFAULT_RANK = 16
REFERENCE_KEY = "__reference__"


@dataclass(frozen=True)
class HarnessConfig:
    validation_shape: int = 4096
    rank: int = DEFAULT_RANK
    seed: int = 0
    warmup: int = 2
    iters: int = 5
    enforce_public_range: bool = True
    tier1_shapes: tuple[int, ...] = (4096, 4608)
    tier2_shapes: tuple[int, ...] = (3584, 4096, 4608)
    tier3_shapes: tuple[int, ...] = (3584, 4096, 4608)
    screen_warmup: int = 1
    screen_iters: int = 2

    @classmethod
    def from_env(cls) -> HarnessConfig:
        validation_shape = int(os.environ.get("OPTIMIZE_LORA_VALIDATION_SHAPE", "4096"))
        test_mode = os.environ.get("OPTIMIZE_LORA_TEST_MODE", "") == "1"
        if test_mode:
            tier1_shapes = (validation_shape,)
            tier2_shapes = (validation_shape, validation_shape + 16, validation_shape + 32)
            tier3_shapes = (validation_shape, validation_shape + 16)
        else:
            tier1_shapes = (4096, 4608)
            tier2_shapes = (3584, 4096, 4608)
            tier3_shapes = (3584, 4096, 4608)
        for env_name, attr in (
            ("OPTIMIZE_LORA_TIER1_SHAPES", "tier1_shapes"),
            ("OPTIMIZE_LORA_TIER2_SHAPES", "tier2_shapes"),
            ("OPTIMIZE_LORA_TIER3_SHAPES", "tier3_shapes"),
        ):
            raw = os.environ.get(env_name)
            if raw:
                parsed = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
                if attr == "tier1_shapes":
                    tier1_shapes = parsed
                elif attr == "tier2_shapes":
                    tier2_shapes = parsed
                else:
                    tier3_shapes = parsed
        return cls(
            validation_shape=validation_shape,
            rank=DEFAULT_RANK,
            seed=int(os.environ.get("OPTIMIZE_LORA_SEED", "0")),
            warmup=int(os.environ.get("OPTIMIZE_LORA_BENCH_WARMUP", "2")),
            iters=int(os.environ.get("OPTIMIZE_LORA_BENCH_ITERS", "5")),
            enforce_public_range=not test_mode,
            tier1_shapes=tier1_shapes,
            tier2_shapes=tier2_shapes,
            tier3_shapes=tier3_shapes,
            screen_warmup=int(os.environ.get("OPTIMIZE_LORA_SCREEN_WARMUP", "1")),
            screen_iters=int(os.environ.get("OPTIMIZE_LORA_SCREEN_ITERS", "2")),
        )


@dataclass(frozen=True)
class InputFixture:
    shape_d: int
    W: torch.Tensor
    X: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor


def validate_shape(d: int, *, enforce_public_range: bool = True) -> None:
    if d <= 0:
        raise ValueError(f"shape d must be positive, got {d}")
    if enforce_public_range and not (PUBLIC_D_MIN <= d <= PUBLIC_D_MAX):
        raise ValueError(f"shape d must be within [{PUBLIC_D_MIN}, {PUBLIC_D_MAX}], got {d}")


def generate_synthetic_inputs(
    d: int,
    *,
    r: int = DEFAULT_RANK,
    seed: int = 0,
    device: str = "cuda",
    enforce_public_range: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(d, enforce_public_range=enforce_public_range)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    W = torch.randn((d, d), device=device, dtype=torch.float32, generator=generator).contiguous()
    X = torch.randn((d, d), device=device, dtype=torch.float32, generator=generator).contiguous()
    A = torch.randn((d, r), device=device, dtype=torch.float32, generator=generator).contiguous()
    B = torch.randn((d, r), device=device, dtype=torch.float32, generator=generator).contiguous()
    return W, X, A, B


def reference_impl(W: torch.Tensor, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def correctness_metrics(y: torch.Tensor, y_ref: torch.Tensor) -> tuple[bool, float, float]:
    diff = (y - y_ref).float()
    max_abs_err = diff.abs().max().item()
    rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    return passed, max_abs_err, rel_l2_err


def benchmark_forward(
    forward_fn,
    W: torch.Tensor,
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    shape_d: int,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    for _ in range(warmup):
        _ = forward_fn(W, X, A, B)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = forward_fn(W, X, A, B)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    min_ms = samples[0] if samples else 0.0
    max_ms = samples[-1] if samples else 0.0
    median_ms = samples[len(samples) // 2] if samples else 0.0
    spread_pct = (((max_ms - min_ms) / max(median_ms, 1e-9)) * 100.0) if samples else 0.0
    return BenchmarkResult(
        shape_d=shape_d,
        warmup=warmup,
        iters=iters,
        median_ms=median_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        spread_pct=spread_pct,
    )


class OptimizeLoraHarness:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self._fixture_cache: dict[tuple[int, int], InputFixture] = {}
        self._reference_output_cache: dict[int, torch.Tensor] = {}
        self._timing_cache: dict[tuple[str, int, int, int], BenchmarkResult] = {}
        self._timing_call_counts: dict[tuple[str, int, int, int], int] = {}

    def prepare_fixture(self, shape_d: int | None = None) -> InputFixture:
        prepared_shape = shape_d or self.config.validation_shape
        key = (prepared_shape, self.config.seed)
        cached = self._fixture_cache.get(key)
        if cached is not None:
            return cached
        W, X, A, B = generate_synthetic_inputs(
            prepared_shape,
            r=self.config.rank,
            seed=self.config.seed,
            enforce_public_range=self.config.enforce_public_range,
        )
        fixture = InputFixture(shape_d=prepared_shape, W=W, X=X, A=A, B=B)
        self._fixture_cache[key] = fixture
        return fixture

    def reference_output(self, shape_d: int) -> torch.Tensor:
        cached = self._reference_output_cache.get(shape_d)
        if cached is not None:
            return cached
        fixture = self.prepare_fixture(shape_d)
        output = reference_impl(fixture.W, fixture.X, fixture.A, fixture.B)
        self._reference_output_cache[shape_d] = output
        return output

    def tier_shapes(self, tier_name: str) -> tuple[int, ...]:
        return {"tier1": self.config.tier1_shapes, "tier2": self.config.tier2_shapes, "tier3": self.config.tier3_shapes}[tier_name]

    def _timed_result(self, cache_key: str, forward_fn, *, shape_d: int, warmup: int, iters: int) -> BenchmarkResult:
        key = (cache_key, shape_d, warmup, iters)
        cached = self._timing_cache.get(key)
        if cached is not None:
            return cached
        fixture = self.prepare_fixture(shape_d)
        result = benchmark_forward(
            forward_fn,
            fixture.W,
            fixture.X,
            fixture.A,
            fixture.B,
            shape_d=shape_d,
            warmup=warmup,
            iters=iters,
        )
        self._timing_cache[key] = result
        self._timing_call_counts[key] = self._timing_call_counts.get(key, 0) + 1
        return result

    def timing_call_count(self, cache_key: str, *, shape_d: int, warmup: int, iters: int) -> int:
        return self._timing_call_counts.get((cache_key, shape_d, warmup, iters), 0)

    def reference_evaluation(self, *, tier_name: str, shapes: tuple[int, ...], warmup: int | None = None, iters: int | None = None) -> CandidateEvaluation:
        summary = self.evaluate_tier(
            REFERENCE_KEY,
            reference_impl,
            tier_name=tier_name,
            shapes=shapes,
            incumbent_key=None,
            incumbent_forward=None,
            warmup=warmup,
            iters=iters,
        )
        evaluation = CandidateEvaluation()
        evaluation.add_tier_summary(summary)
        evaluation.correctness_passed = summary.correctness_passed
        return evaluation

    def evaluate_tier(
        self,
        candidate_key: str,
        forward_fn,
        *,
        tier_name: str,
        shapes: tuple[int, ...],
        incumbent_key: str | None = None,
        incumbent_forward=None,
        warmup: int | None = None,
        iters: int | None = None,
    ) -> BenchmarkTierSummary:
        warmup = self.config.warmup if warmup is None else warmup
        iters = self.config.iters if iters is None else iters
        shape_results: list[ShapeBenchmarkResult] = []
        max_abs_err = 0.0
        rel_l2_err = 0.0
        for shape_d in shapes:
            fixture = self.prepare_fixture(shape_d)
            y_student = forward_fn(fixture.W, fixture.X, fixture.A, fixture.B)
            y_ref = self.reference_output(shape_d)
            passed, shape_abs_err, shape_rel_l2_err = correctness_metrics(y_student, y_ref)
            max_abs_err = max(max_abs_err, shape_abs_err)
            rel_l2_err = max(rel_l2_err, shape_rel_l2_err)
            if not passed:
                return BenchmarkTierSummary(
                    tier_name=tier_name,
                    shapes=list(shapes),
                    correctness_passed=False,
                    shape_results=shape_results,
                    max_abs_err=max_abs_err,
                    rel_l2_err=rel_l2_err,
                    failure_reason=f"correctness_failed_shape_{shape_d}",
                )

            candidate_ms = self._timed_result(candidate_key, forward_fn, shape_d=shape_d, warmup=warmup, iters=iters)
            reference_ms = self._timed_result(REFERENCE_KEY, reference_impl, shape_d=shape_d, warmup=warmup, iters=iters)
            incumbent_ms = None
            if incumbent_key is not None:
                if incumbent_key == REFERENCE_KEY:
                    incumbent_ms = reference_ms.median_ms
                elif incumbent_forward is not None:
                    incumbent_ms = self._timed_result(
                        incumbent_key,
                        incumbent_forward,
                        shape_d=shape_d,
                        warmup=warmup,
                        iters=iters,
                    ).median_ms
            shape_results.append(
                ShapeBenchmarkResult(
                    shape_d=shape_d,
                    warmup=warmup,
                    iters=iters,
                    median_ms=candidate_ms.median_ms,
                    min_ms=candidate_ms.min_ms,
                    max_ms=candidate_ms.max_ms,
                    spread_pct=candidate_ms.spread_pct,
                    reference_median_ms=reference_ms.median_ms,
                    speedup_vs_reference=reference_ms.median_ms / candidate_ms.median_ms,
                    speedup_vs_best=(incumbent_ms / candidate_ms.median_ms) if incumbent_ms is not None else None,
                )
            )

        summary = BenchmarkTierSummary(
            tier_name=tier_name,
            shapes=list(shapes),
            correctness_passed=True,
            shape_results=shape_results,
            max_abs_err=max_abs_err,
            rel_l2_err=rel_l2_err,
            geometric_mean_speedup=geometric_mean([row.speedup_vs_reference for row in shape_results]),
            largest_shape_median_ms=max(
                (row.median_ms for row in shape_results if row.shape_d == max(shapes)),
                default=shape_results[-1].median_ms if shape_results else 0.0,
            ),
        )
        summary.worst_regression_pct = summarize_regression_pct(summary)
        summary.noise_guard_pct = max((row.spread_pct for row in shape_results), default=0.0)
        return summary

    def evaluate_batch_tier(
        self,
        entrypoints: dict[str, object],
        *,
        tier_name: str,
        incumbent_key: str | None = None,
        incumbent_forward=None,
        warmup: int | None = None,
        iters: int | None = None,
    ) -> dict[str, BenchmarkTierSummary]:
        results: dict[str, BenchmarkTierSummary] = {}
        shapes = self.tier_shapes(tier_name)
        for candidate_key, forward_fn in entrypoints.items():
            results[candidate_key] = self.evaluate_tier(
                candidate_key,
                forward_fn,
                tier_name=tier_name,
                shapes=shapes,
                incumbent_key=incumbent_key,
                incumbent_forward=incumbent_forward,
                warmup=warmup,
                iters=iters,
            )
        return results
