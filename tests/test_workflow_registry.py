from sysforge.runtime import build_runtime_context, load_config
from sysforge.workflows.optimize_lora.models import CandidateFamilyDraft, ParameterSpec
from sysforge.workflows.optimize_lora.models import BenchmarkTierSummary, CandidateEvaluation, ShapeBenchmarkResult
from sysforge.workflows.registry import build_registry


def _tier_summary(
    tier_name: str,
    shapes: list[int],
    *,
    candidate_ms: float = 1.0,
    reference_ms: float = 2.0,
    speedup_vs_best: float | None = 1.01,
    correctness_passed: bool = True,
) -> BenchmarkTierSummary:
    return BenchmarkTierSummary(
        tier_name=tier_name,
        shapes=shapes,
        correctness_passed=correctness_passed,
        shape_results=[
            ShapeBenchmarkResult(
                shape_d=shape,
                warmup=1,
                iters=3,
                median_ms=candidate_ms,
                reference_median_ms=reference_ms,
                speedup_vs_reference=reference_ms / candidate_ms,
                speedup_vs_best=speedup_vs_best,
            )
            for shape in shapes
        ],
        geometric_mean_speedup=reference_ms / candidate_ms,
        largest_shape_median_ms=candidate_ms,
    )


class _FakeModule:
    def __init__(self):
        self.forward = lambda W, X, A, B: W


def test_registry_exposes_expected_workflows():
    registry = build_registry()
    assert set(registry) >= {"profiling", "optimize-lora"}


def test_optimize_lora_workflow_uses_llm_authored_family_search(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPTIMIZE_LORA_TEST_MODE", "1")
    monkeypatch.setenv("OPTIMIZE_LORA_MAX_LLM_ROUNDS", "1")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_MODEL", "test-model")
    (tmp_path / "target").mkdir()
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(
            family_name="llm_family",
            source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // {{BLOCK_X}}",
            parameters=(ParameterSpec(name="BLOCK_X", values=(16, 24, 32)),),
            rationale="llm-authored search family",
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(family_name="unused", source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}", parameters=()),
    )

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.harness.OptimizeLoraHarness.reference_evaluation",
        lambda self, *, tier_name, shapes, warmup=None, iters=None: CandidateEvaluation(
            correctness_passed=True,
            tier_summaries=[_tier_summary(tier_name, list(shapes), candidate_ms=2.0, reference_ms=2.0)],
        ),
    )

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.build.CandidateBuilder.load_candidate",
        lambda self, candidate: (
            type("CompileResult", (), {
                "status": "built",
                "source_hash": candidate.source_hash,
                "module_name": candidate.module_name,
                "source_path": candidate.source_path,
                "build_dir": "/tmp",
                "log_path": "/tmp/build.log",
                "error": "",
                "duration_s": 1.0,
            })(),
            _FakeModule(),
        ),
    )

    def fake_evaluate_tier(self, candidate_key, forward_fn, *, tier_name, shapes, incumbent_key=None, incumbent_forward=None, warmup=None, iters=None):
        if candidate_key == "__reference__":
            return _tier_summary(tier_name, list(shapes), candidate_ms=2.0, reference_ms=2.0)
        if candidate_key.endswith("v1"):
            return _tier_summary(tier_name, list(shapes), candidate_ms=0.8, reference_ms=2.0, speedup_vs_best=1.2)
        return _tier_summary(tier_name, list(shapes), candidate_ms=0.95, reference_ms=2.0, speedup_vs_best=1.05)

    monkeypatch.setattr("sysforge.workflows.optimize_lora.harness.OptimizeLoraHarness.evaluate_tier", fake_evaluate_tier)

    workflow = build_registry()["optimize-lora"]
    result = workflow.run(build_runtime_context(load_config()))

    assert result.status == "searched" or result.status == "optimized"
    assert result.seed_count_run >= 1
    assert result.current_best_family == "llm_family"
    assert any(entry["action"] == "family_generated" for entry in result.controller_trace)


def test_optimize_lora_workflow_records_family_generation_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("OPTIMIZE_LORA_TEST_MODE", "1")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_MODEL", "test-model")
    (tmp_path / "target").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("proxy down")),
    )

    workflow = build_registry()["optimize-lora"]
    result = workflow.run(build_runtime_context(load_config()))

    assert result.status == "searched"
    assert "generate_candidate_family failed" in (result.errors[0] if result.errors else "")
