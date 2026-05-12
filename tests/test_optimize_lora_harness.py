import pytest

torch = pytest.importorskip("torch")

from sysforge.workflows.optimize_lora.harness import HarnessConfig, OptimizeLoraHarness, REFERENCE_KEY, correctness_metrics, generate_synthetic_inputs, reference_impl  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_generate_synthetic_inputs_shapes_and_dtypes():
    W, X, A, B = generate_synthetic_inputs(3584, seed=7)
    assert W.shape == (3584, 3584)
    assert X.shape == (3584, 3584)
    assert A.shape == (3584, 16)
    assert B.shape == (3584, 16)
    assert W.dtype == torch.float32
    assert W.is_cuda


def test_reference_impl_matches_direct_expression():
    W, X, A, B = generate_synthetic_inputs(3712, seed=11)
    y_ref = reference_impl(W, X, A, B)
    y_direct = W @ X + A @ (B.transpose(0, 1).contiguous() @ X)
    assert torch.allclose(y_ref, y_direct, rtol=1e-4, atol=1e-4)


def test_correctness_metrics_report_expected_values():
    base = torch.tensor([[1.0, 2.0]], device="cuda")
    same = torch.tensor([[1.0, 2.0]], device="cuda")
    passed, max_abs_err, rel_l2_err = correctness_metrics(base, same)
    assert passed is True
    assert max_abs_err == 0.0
    assert rel_l2_err == 0.0


def test_harness_caches_reference_timings():
    harness = OptimizeLoraHarness(HarnessConfig(validation_shape=3584, seed=5, warmup=1, iters=2, enforce_public_range=True))
    first = harness.evaluate_tier(REFERENCE_KEY, reference_impl, tier_name="tier1", shapes=(3584,), warmup=1, iters=2)
    second = harness.evaluate_tier(REFERENCE_KEY, reference_impl, tier_name="tier1", shapes=(3584,), warmup=1, iters=2)
    assert first.correctness_passed is True
    assert second.correctness_passed is True
    assert harness.timing_call_count(REFERENCE_KEY, shape_d=3584, warmup=1, iters=2) == 1


def test_harness_reference_evaluation_uses_virtual_baseline():
    harness = OptimizeLoraHarness(HarnessConfig(validation_shape=3584, seed=5, warmup=1, iters=2, enforce_public_range=True))
    evaluation = harness.reference_evaluation(tier_name="tier2", shapes=(3584, 4096))
    assert evaluation.correctness_passed is True
    assert evaluation.tier("tier2") is not None


def test_harness_evaluates_batch_entrypoints():
    harness = OptimizeLoraHarness(HarnessConfig(validation_shape=3584, seed=5, warmup=1, iters=2, enforce_public_range=True))
    results = harness.evaluate_batch_tier(
        {
            "candidate_a": reference_impl,
            "candidate_b": reference_impl,
        },
        tier_name="tier1",
        warmup=1,
        iters=2,
    )
    assert set(results) == {"candidate_a", "candidate_b"}
    assert all(summary.correctness_passed for summary in results.values())
