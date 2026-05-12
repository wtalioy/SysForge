from __future__ import annotations

import time
from pathlib import Path

from sysforge.runtime import build_runtime_context, load_config
from sysforge.workflows.optimize_lora.models import CandidateFamilyDraft, ParameterSpec
from sysforge.workflows.optimize_lora.harness import HarnessConfig, REFERENCE_KEY
from sysforge.workflows.optimize_lora.models import (
    BenchmarkTierSummary,
    CandidateEvaluation,
    CandidateRecord,
    ShapeBenchmarkResult,
)
from sysforge.workflows.optimize_lora.agent import OptimizeLoraAgent, SearchConfig
from sysforge.workflows.optimize_lora.build import BASELINE_SOURCE


def _tier_summary(
    tier_name: str,
    *,
    shapes: list[int] | None = None,
    candidate_ms: float = 1.0,
    reference_ms: float = 2.0,
    speedup_vs_best: float | None = 1.01,
    correctness_passed: bool = True,
    spread_pct: float = 0.0,
    rerun_used: bool = False,
) -> BenchmarkTierSummary:
    shapes = shapes or [4096]
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
                min_ms=candidate_ms,
                max_ms=candidate_ms,
                spread_pct=spread_pct,
                reference_median_ms=reference_ms,
                speedup_vs_reference=reference_ms / candidate_ms,
                speedup_vs_best=speedup_vs_best,
            )
            for shape in shapes
        ],
        geometric_mean_speedup=reference_ms / candidate_ms,
        largest_shape_median_ms=candidate_ms,
        noise_guard_pct=spread_pct,
        rerun_used=rerun_used,
        failure_reason="" if correctness_passed else f"correctness_failed_shape_{shapes[0]}",
    )


class _Module:
    def __init__(self) -> None:
        self.forward = lambda W, X, A, B: W


class _Builder:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.compile_calls = 0

    def register_candidate(self, *, candidate_id: str, family: str, source: str, entrypoint_name: str = "forward") -> CandidateRecord:
        path = self.tmp_path / f"{candidate_id}.cu"
        path.write_text(source, encoding="utf-8")
        return CandidateRecord(
            candidate_id=candidate_id,
            family=family,
            source_hash=candidate_id,
            module_name=f"{candidate_id}_mod",
            source_path=str(path),
            entrypoint_name=entrypoint_name,
        )

    def load_candidate(self, candidate: CandidateRecord):
        from sysforge.workflows.optimize_lora.models import CandidateCompileResult

        self.compile_calls += 1
        source = Path(candidate.source_path).read_text(encoding="utf-8")
        if "BROKEN" in source:
            return CandidateCompileResult(
                status="failed",
                source_hash=candidate.source_hash,
                module_name=candidate.module_name,
                source_path=candidate.source_path,
                build_dir="/tmp",
                log_path="/tmp/build.log",
                error="compile failed",
                duration_s=1.0,
            ), None
        return CandidateCompileResult(
            status="built",
            source_hash=candidate.source_hash,
            module_name=candidate.module_name,
            source_path=candidate.source_path,
            build_dir="/tmp",
            log_path="/tmp/build.log",
            error="",
            duration_s=1.0,
        ), _Module()


class _Harness:
    def __init__(self) -> None:
        self.config = HarnessConfig()
        self.calls: list[tuple[str, str, int | None, int | None]] = []

    def tier_shapes(self, tier_name: str) -> tuple[int, ...]:
        if tier_name == "tier1":
            return (4096,)
        if tier_name == "tier2":
            return (3584, 4096, 4608)
        return (3584, 4096)

    def reference_evaluation(self, *, tier_name: str, shapes: tuple[int, ...], warmup=None, iters=None):
        evaluation = CandidateEvaluation()
        evaluation.add_tier_summary(_tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0, reference_ms=2.0))
        evaluation.correctness_passed = True
        return evaluation

    def evaluate_tier(self, candidate_key, _forward_fn, *, tier_name, shapes, incumbent_key=None, incumbent_forward=None, warmup=None, iters=None):
        self.calls.append((candidate_key, tier_name, warmup, iters))
        if candidate_key == REFERENCE_KEY:
            return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0, reference_ms=2.0)
        if "repaired_family" in candidate_key:
            if tier_name == "tier3":
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=0.75, reference_ms=2.0, speedup_vs_best=1.2)
            return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=0.8, reference_ms=2.0, speedup_vs_best=1.2)
        if "search_family" in candidate_key:
            offset = 0.85 if candidate_key.endswith("v1") else 0.95
            return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=offset, reference_ms=2.0, speedup_vs_best=1.1)
        return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.05)


def _context(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TARGET_DIR", str(tmp_path / "target"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    (tmp_path / "target").mkdir()
    monkeypatch.chdir(tmp_path)
    return build_runtime_context(load_config())


def test_agent_instantiates_llm_family_parameter_space(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=3, max_llm_rounds=1),
    )
    family = CandidateFamilyDraft(
        family_name="search_family",
        source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // {{BLOCK_X}} {{USE_EPILOGUE}}",
        parameters=(
            ParameterSpec(name="BLOCK_X", values=(16, 32)),
            ParameterSpec(name="USE_EPILOGUE", values=(True, False)),
        ),
    )
    concrete = agent._instantiate_family(family, round_index=1)
    assert len(concrete) == 3
    assert "{{BLOCK_X}}" not in concrete[0].source
    assert any("true" in item.source or "false" in item.source for item in concrete)


def test_agent_instantiates_explicit_concrete_variants(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=2, max_llm_rounds=1),
    )
    family = CandidateFamilyDraft(
        family_name="explicit_family",
        source_template="{{FORWARD_BODY}}",
        concrete_variants=(
            {"FORWARD_BODY": "auto out = torch::matmul(W, X); return out;"},
            {"FORWARD_BODY": "auto out = torch::mm(W, X); return out;"},
            {"FORWARD_BODY": "auto out = torch::addmm(torch::matmul(W, X), A, B.transpose(0, 1).contiguous()); return out;"},
        ),
    )
    concrete = agent._instantiate_family(family, round_index=1)
    assert len(concrete) == 2
    assert concrete[0].parameter_values["FORWARD_BODY"] == "auto out = torch::matmul(W, X); return out;"
    assert concrete[1].parameter_values["FORWARD_BODY"] == "auto out = torch::mm(W, X); return out;"


def test_agent_skips_sources_already_seen_in_run(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=3, max_llm_rounds=1),
    )
    agent.bootstrap_baseline()
    family = CandidateFamilyDraft(
        family_name="dupe_family",
        source_template=BASELINE_SOURCE,
        parameters=(),
    )
    concrete = agent._instantiate_family(family, round_index=1)
    assert concrete == []


def test_agent_repairs_failed_family_and_promotes_winner(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(
            family_name="broken_family",
            source_template="BROKEN {{BLOCK_X}}",
            parameters=(ParameterSpec(name="BLOCK_X", values=(16,)),),
            rationale="broken first draft",
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.repair_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(
            family_name="repaired_family",
            source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // {{BLOCK_X}}",
            parameters=(ParameterSpec(name="BLOCK_X", values=(16,)),),
            rationale="fixed compile error",
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(family_name="unused", source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}", parameters=()),
    )

    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(
            max_family_variants=2,
            min_seed_variants=1,
            max_llm_rounds=1,
            ),
    )
    result = agent.run()
    assert result.current_best_family == "repaired_family"
    assert result.best_candidate_kind == "optimized"
    assert any(entry["action"] == "family_repaired" for entry in result.controller_trace)


def test_agent_regenerates_family_after_revision_failure(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    generated = iter(
        [
            CandidateFamilyDraft(
                family_name="first_family",
                source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // first {{BLOCK_X}}",
                parameters=(ParameterSpec(name="BLOCK_X", values=(16, 32)),),
                rationale="first try",
            ),
            CandidateFamilyDraft(
                family_name="second_family",
                source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // second {{BLOCK_X}}",
                parameters=(ParameterSpec(name="BLOCK_X", values=(48, 64)),),
                rationale="fresh retry",
            ),
        ]
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: next(generated),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("duplicate body")),
    )

    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(
            max_family_variants=2,
            min_seed_variants=2,
            max_llm_rounds=2,
            clear_winner_speedup=3.0,
            ),
    )
    result = agent.run()
    assert any(entry["action"] == "family_regenerated" for entry in result.controller_trace)
    assert any(candidate.family == "second_family" for candidate in result.candidates)


def test_agent_uses_local_fallback_family_after_duplicate_history_revision_failure(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(
            family_name="first_family",
            source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // first {{BLOCK_X}}",
            parameters=(ParameterSpec(name="BLOCK_X", values=(16, 32)),),
            rationale="first try",
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("revise_candidate_family.txt failed validation: FORWARD_BODY reuses concrete bodies that already appeared in recent history")
        ),
    )

    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(
            max_family_variants=2,
            min_seed_variants=2,
            max_llm_rounds=2,
            clear_winner_speedup=3.0,
            ),
    )
    result = agent.run()
    assert any(entry["action"] == "family_fallback_local" for entry in result.controller_trace)
    assert any(candidate.family == "first_family_fallback" for candidate in result.candidates)


def test_agent_uses_clean_local_fallback_after_recoverable_revision_failure(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    generate_calls = 0

    def _generate(**kwargs):
        nonlocal generate_calls
        generate_calls += 1
        return CandidateFamilyDraft(
            family_name="first_family",
            source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // first {{BLOCK_X}}",
            parameters=(ParameterSpec(name="BLOCK_X", values=(16, 32)),),
            rationale="first try",
        )

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        _generate,
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("APITimeoutError: Request timed out.")),
    )

    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(
            max_family_variants=2,
            min_seed_variants=2,
            max_llm_rounds=2,
            clear_winner_speedup=3.0,
            ),
    )
    result = agent.run()
    assert generate_calls == 1
    assert result.errors == []
    assert any(entry["action"] == "family_fallback_local" and entry["reason"] == "llm_revise_unavailable" for entry in result.controller_trace)
    assert any(candidate.family == "first_family_fallback" for candidate in result.candidates)


def test_fallback_family_prefers_inplace_addmm_variant(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=3, max_llm_rounds=1),
    )
    family = agent._fallback_candidate_family(
        previous_family_name="prior",
        incumbent_source="auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); return torch::addmm(torch::matmul(W, X), A, tmp);",
    )
    assert family.parameters[0].default == "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;"
    assert any("auto out = torch::mm(W, X);" in value for value in family.parameters[0].values)
    assert any("auto tmp = torch::mm(bt, X); auto out = torch::matmul(W, X);" in value for value in family.parameters[0].values)
    assert any("auto out = torch::empty({W.size(0), X.size(1)}, X.options()); at::mm_out(out, W, X);" in value for value in family.parameters[0].values)
    assert any("at::mm_out(out, W, X);" in value for value in family.parameters[0].values)
    assert any("at::mm_out(" in value for value in family.parameters[0].values)


def test_body_priority_score_prefers_mm_addmm_and_penalizes_extra_traffic(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=3, max_llm_rounds=1),
    )
    fast_shape = CandidateRecord(candidate_id="fast", family="probe", source_hash="fast", module_name="m", source_path="/tmp/f", entrypoint_name="forward")
    fast_shape.parameter_values = {
        "FORWARD_BODY": "auto bt = B.transpose(0, 1); auto tmp = torch::mm(bt, X); auto out = torch::mm(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;"
    }
    weak_shape = CandidateRecord(candidate_id="weak", family="probe", source_hash="weak", module_name="m", source_path="/tmp/w", entrypoint_name="forward")
    weak_shape.parameter_values = {
        "FORWARD_BODY": "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::empty({W.size(0), X.size(1)}, W.options()); out.copy_(torch::matmul(W, X)); out.addmm_(A, tmp, 1.0, 1.0); return out;"
    }
    assert agent._body_priority_score(fast_shape) > agent._body_priority_score(weak_shape)


def test_tier2_shortlist_prefers_distinct_execution_plans(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=4, max_full_evaluations_per_round=2, max_llm_rounds=1),
    )

    def make_candidate(candidate_id: str, body: str, tier1_geo: float, incumbent_ratio: float) -> CandidateRecord:
        record = CandidateRecord(candidate_id=candidate_id, family="probe", source_hash=candidate_id, module_name="m", source_path=f"/tmp/{candidate_id}", entrypoint_name="forward")
        record.parameter_values = {"FORWARD_BODY": body}
        evaluation = CandidateEvaluation()
        evaluation.add_tier_summary(
            BenchmarkTierSummary(
                tier_name="tier1",
                shapes=[4096, 4608],
                correctness_passed=True,
                shape_results=[
                    ShapeBenchmarkResult(shape_d=4096, warmup=1, iters=2, median_ms=1.0, reference_median_ms=2.0, speedup_vs_reference=tier1_geo, speedup_vs_best=incumbent_ratio),
                    ShapeBenchmarkResult(shape_d=4608, warmup=1, iters=2, median_ms=1.0, reference_median_ms=2.0, speedup_vs_reference=tier1_geo, speedup_vs_best=incumbent_ratio),
                ],
                geometric_mean_speedup=tier1_geo,
                largest_shape_median_ms=1.0,
            )
        )
        record.evaluation = evaluation
        return record

    c1 = make_candidate("c1", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.04, 1.02)
    c2 = make_candidate("c2", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); return torch::addmm(torch::matmul(W, X), A, tmp);", 1.05, 1.021)
    c3 = make_candidate("c3", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::mm(bt, X); auto out = torch::mm(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.03, 1.019)

    shortlist = agent._round_evaluator.select_tier2_shortlist([c1, c2, c3])
    assert [candidate.candidate_id for candidate in shortlist] == ["c2", "c3"]


def test_agent_reruns_close_tier1_candidates_before_shortlist(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)

    class _RerunHarness(_Harness):
        def evaluate_tier(self, candidate_key, _forward_fn, *, tier_name, shapes, incumbent_key=None, incumbent_forward=None, warmup=None, iters=None):
            self.calls.append((candidate_key, tier_name, warmup, iters))
            if candidate_key == REFERENCE_KEY:
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0, reference_ms=2.0)
            if tier_name == "tier1" and warmup == 2 and iters == 4:
                if candidate_key.endswith("v1"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.01, reference_ms=2.0, speedup_vs_best=1.01, spread_pct=0.2, rerun_used=True)
                if candidate_key.endswith("v2"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=0.98, reference_ms=2.0, speedup_vs_best=1.03, spread_pct=0.2, rerun_used=True)
            if tier_name == "tier1":
                if candidate_key.endswith("v1"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.02, spread_pct=2.0)
                if candidate_key.endswith("v2"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.01, reference_ms=2.0, speedup_vs_best=1.019, spread_pct=2.0)
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.2, reference_ms=2.0, speedup_vs_best=0.95, spread_pct=0.2)
            if candidate_key.endswith("v2"):
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=0.9, reference_ms=2.0, speedup_vs_best=1.04)
            return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=0.95, reference_ms=2.0, speedup_vs_best=1.02)

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(
            family_name="close_family",
            source_template="{{FORWARD_BODY}}",
            concrete_variants=(
                {"FORWARD_BODY": "auto out = torch::matmul(W, X); return out;"},
                {"FORWARD_BODY": "auto out = torch::mm(W, X); return out;"},
                {"FORWARD_BODY": "auto tmp = torch::empty_like(X); return tmp;"},
            ),
            rationale="near-tie probe",
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(family_name="unused", source_template="{{FORWARD_BODY}}", concrete_variants=()),
    )

    harness = _RerunHarness()
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=harness,
        config=SearchConfig(
            max_family_variants=3,
            max_full_evaluations_per_round=2,
            max_llm_rounds=1,
            tier1_rerun_warmup=2,
            tier1_rerun_iters=4,
            tier1_rerun_band_pct=1.5,
            ),
    )
    result = agent.run()
    rerun_calls = [call for call in harness.calls if call[1] == "tier1" and call[2] == 2 and call[3] == 4]
    assert [call[0] for call in rerun_calls] == ["close_family-r1-v1", "close_family-r1-v2"]
    rerun_entries = [entry for entry in result.controller_trace if entry["action"] == "tier1_rerun"]
    assert [entry["candidate_id"] for entry in rerun_entries] == ["close_family-r1-v1", "close_family-r1-v2"]
    rerun_summary = next(candidate.evaluation.tier("tier1") for candidate in result.candidates if candidate.candidate_id == "close_family-r1-v2")
    assert rerun_summary.rerun_used is True
    assert rerun_summary.noise_guard_pct >= 1.5


def test_agent_defers_clear_winner_when_tier2_frontier_is_close(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    revisions: list[str] = []

    def _family(name: str) -> CandidateFamilyDraft:
        return CandidateFamilyDraft(
            family_name=name,
            source_template="{{FORWARD_BODY}}",
            concrete_variants=(
                {"FORWARD_BODY": "auto out = torch::matmul(W, X); return out;"},
                {"FORWARD_BODY": "auto out = torch::mm(W, X); return out;"},
                {"FORWARD_BODY": "auto tmp = torch::empty_like(X); return tmp;"},
            ),
            rationale=f"{name} rationale",
        )

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: _family("close_family"),
    )

    def _revise_candidate_family(**kwargs):
        revisions.append(kwargs["family"].family_name)
        return _family("revised_family")

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        _revise_candidate_family,
    )

    class _CloseFrontierHarness(_Harness):
        def evaluate_tier(self, candidate_key, _forward_fn, *, tier_name, shapes, incumbent_key=None, incumbent_forward=None, warmup=None, iters=None):
            self.calls.append((candidate_key, tier_name, warmup, iters))
            if candidate_key == REFERENCE_KEY:
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0, reference_ms=2.0)
            family_name = candidate_key.split("-r", 1)[0]
            if tier_name == "tier1":
                if candidate_key.endswith("v1"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.0, reference_ms=2.0, speedup_vs_best=1.03, spread_pct=2.0)
                if candidate_key.endswith("v2"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.01, reference_ms=2.0, speedup_vs_best=1.029, spread_pct=2.0)
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.2, reference_ms=2.0, speedup_vs_best=0.95, spread_pct=0.3)
            if family_name == "close_family":
                if candidate_key.endswith("v1"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.0510, reference_ms=2.0, speedup_vs_best=1.0510)
                if candidate_key.endswith("v2"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.0502, reference_ms=2.0, speedup_vs_best=1.0502)
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.01, reference_ms=2.0, speedup_vs_best=1.01)
            if candidate_key.endswith("v1"):
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.09, reference_ms=2.0, speedup_vs_best=1.09)
            if candidate_key.endswith("v2"):
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.03, reference_ms=2.0, speedup_vs_best=1.03)
            return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.0, reference_ms=2.0, speedup_vs_best=1.0)

    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_CloseFrontierHarness(),
        config=SearchConfig(
            max_family_variants=3,
            max_full_evaluations_per_round=2,
            max_llm_rounds=2,
            clear_winner_speedup=1.05,
            tier1_rerun_warmup=2,
            tier1_rerun_iters=4,
            ),
    )
    result = agent.run()
    assert revisions == ["close_family"]
    assert any(entry["action"] == "clear_winner_deferred" for entry in result.controller_trace)
    assert any(entry["action"] == "round_started" and entry["round_index"] == 2 for entry in result.controller_trace)
    deferred = next(entry for entry in result.controller_trace if entry["action"] == "clear_winner_deferred")
    assert deferred["challenger_margin_pct"] < deferred["challenger_separation_guard_pct"]


def test_agent_stops_on_clear_winner_when_tier2_margin_is_clear(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    revisions: list[str] = []

    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(
            family_name="clear_family",
            source_template="{{FORWARD_BODY}}",
            concrete_variants=(
                {"FORWARD_BODY": "auto out = torch::matmul(W, X); return out;"},
                {"FORWARD_BODY": "auto out = torch::mm(W, X); return out;"},
                {"FORWARD_BODY": "auto tmp = torch::empty_like(X); return tmp;"},
            ),
            rationale="clear frontier",
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: revisions.append(kwargs["family"].family_name) or CandidateFamilyDraft(family_name="unused", source_template="{{FORWARD_BODY}}", concrete_variants=()),
    )

    class _ClearFrontierHarness(_Harness):
        def evaluate_tier(self, candidate_key, _forward_fn, *, tier_name, shapes, incumbent_key=None, incumbent_forward=None, warmup=None, iters=None):
            self.calls.append((candidate_key, tier_name, warmup, iters))
            if candidate_key == REFERENCE_KEY:
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0, reference_ms=2.0)
            if tier_name == "tier1":
                if candidate_key.endswith("v1"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=0.95, reference_ms=2.0, speedup_vs_best=1.06, spread_pct=0.2)
                if candidate_key.endswith("v2"):
                    return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.05, reference_ms=2.0, speedup_vs_best=1.02, spread_pct=0.2)
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=1.2, reference_ms=2.0, speedup_vs_best=0.98, spread_pct=0.2)
            if candidate_key.endswith("v1"):
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.08, reference_ms=2.0, speedup_vs_best=1.08)
            if candidate_key.endswith("v2"):
                return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.03, reference_ms=2.0, speedup_vs_best=1.03)
            return _tier_summary(tier_name, shapes=list(shapes), candidate_ms=2.0 / 1.0, reference_ms=2.0, speedup_vs_best=1.0)

    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_ClearFrontierHarness(),
        config=SearchConfig(
            max_family_variants=3,
            max_full_evaluations_per_round=2,
            max_llm_rounds=2,
            clear_winner_speedup=1.05,
            ),
    )
    result = agent.run()
    assert revisions == []
    assert any(entry["action"] == "search_stopped" and entry["reason"] == "clear_winner" for entry in result.controller_trace)
    assert not any(entry["action"] == "clear_winner_deferred" for entry in result.controller_trace)
    assert not any(entry["action"] == "round_started" and entry["round_index"] == 2 for entry in result.controller_trace)


def test_finalists_include_alternative_plan_when_available(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(final_confirmation_candidates=2, max_close_finalists=2, max_llm_rounds=1),
    )

    def make_candidate(candidate_id: str, body: str, tier2_geo: float) -> CandidateRecord:
        record = CandidateRecord(candidate_id=candidate_id, family="probe", source_hash=candidate_id, module_name="m", source_path=f"/tmp/{candidate_id}", entrypoint_name="forward")
        record.parameter_values = {"FORWARD_BODY": body}
        evaluation = CandidateEvaluation()
        evaluation.add_tier_summary(_tier_summary("tier2", candidate_ms=2.0 / tier2_geo, reference_ms=2.0))
        record.evaluation = evaluation
        return record

    c1 = make_candidate("c1", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.06)
    c2 = make_candidate("c2", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); return torch::addmm(torch::matmul(W, X), A, tmp);", 1.055)
    c3 = make_candidate("c3", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::mm(bt, X); auto out = torch::mm(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.05)

    finalists = agent._round_evaluator.select_finalists([c1, c2, c3])
    assert [candidate.candidate_id for candidate in finalists] == ["c1", "c3"]


def test_finalists_expand_for_close_tier2_cluster(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(
            final_confirmation_candidates=3,
            max_close_finalists=4,
            max_llm_rounds=1,
            ),
    )

    def make_candidate(candidate_id: str, body: str, tier2_geo: float) -> CandidateRecord:
        record = CandidateRecord(candidate_id=candidate_id, family="probe", source_hash=candidate_id, module_name="m", source_path=f"/tmp/{candidate_id}", entrypoint_name="forward")
        record.parameter_values = {"FORWARD_BODY": body}
        evaluation = CandidateEvaluation()
        evaluation.add_tier_summary(_tier_summary("tier2", candidate_ms=2.0 / tier2_geo, reference_ms=2.0))
        record.evaluation = evaluation
        return record

    c1 = make_candidate("c1", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::mm(bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.0510)
    c2 = make_candidate("c2", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::empty({B.size(1), X.size(1)}, X.options()); at::mm_out(tmp, bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.0504)
    c3 = make_candidate("c3", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.0501)
    c4 = make_candidate("c4", "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::mm(bt, X); auto out = torch::mm(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;", 1.0430)

    finalists = agent._round_evaluator.select_finalists([c1, c2, c3, c4])
    assert [candidate.candidate_id for candidate in finalists] == ["c1", "c2", "c3", "c4"]


def test_final_confirmation_uses_top_candidates(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.generate_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(
            family_name="search_family",
            source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // {{BLOCK_X}}",
            parameters=(ParameterSpec(name="BLOCK_X", values=(16, 32)),),
            rationale="search tiles",
        ),
    )
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.agent.family_agent.revise_candidate_family",
        lambda **kwargs: CandidateFamilyDraft(family_name="unused", source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}", parameters=()),
    )

    harness = _Harness()
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=harness,
        config=SearchConfig(
            max_family_variants=2,
            min_seed_variants=2,
            max_llm_rounds=1,
            final_confirmation_candidates=2,
            ),
    )
    result = agent.run()
    tier3_calls = [call for call in harness.calls if call[1] == "tier3"]
    assert len(tier3_calls) >= 2
    assert result.winner_confirmed is True
    assert result.winner_confirmation_candidate_id == result.current_best_candidate_id


def test_agent_deduplicates_rendered_variants(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=4, min_seed_variants=3, max_llm_rounds=1),
    )
    family = CandidateFamilyDraft(
        family_name="duplicate_family",
        source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}",
        parameters=(ParameterSpec(name="BLOCK_X", values=(16, 24, 32)),),
    )
    concrete = agent._instantiate_family(family, round_index=1)
    assert len(concrete) == 1


def test_agent_fills_distinct_variants_before_applying_family_cap(monkeypatch, tmp_path: Path):
    context = _context(monkeypatch, tmp_path)
    agent = OptimizeLoraAgent(
        context,
        builder=_Builder(tmp_path),
        harness=_Harness(),
        config=SearchConfig(max_family_variants=2, min_seed_variants=2, max_llm_rounds=1),
    )
    family = CandidateFamilyDraft(
        family_name="duplicate_then_distinct_family",
        source_template="PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {} // {{A}} {{B}}",
        parameters=(
            ParameterSpec(name="A", values=("same", "same", "distinct")),
            ParameterSpec(name="B", values=("same",)),
        ),
    )
    concrete = agent._instantiate_family(family, round_index=1)
    assert len(concrete) == 2
    assert concrete[0].source != concrete[1].source
