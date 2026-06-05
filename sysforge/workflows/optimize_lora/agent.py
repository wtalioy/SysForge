from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import torch
from ...agent import SearchAgent, StoppingPolicy
from ...runtime import RuntimeContext
from ..artifacts import source_digest
from ..common import append_workflow_error, workflow_timestamp
from ..registry import register_workflow
from .defaults import DEFAULT_LORA_SEARCH_CONFIG, SearchConfig
from .prompting import revise_candidate_family, generate_candidate_family
from .build import CandidateBuilder
from .families import expand_family_mappings, render_family_source
from .harness import OptimizeLoraHarness
from .models import (
    CandidateFamilyDraft,
    CandidateRecord,
    ConcreteCandidate,
    OptimizeLoraResult,
    ParameterSpec,
)
from .profiling import CandidateProfiler
from .promotion import TIER2
from .round_evaluator import RoundEvaluator
from .templates import BASELINE_SOURCE, extract_forward_body


ARTIFACT_NAME = "optimized_lora.cu"


@register_workflow(
    name="optimize-lora",
    description="Bootstrap and optimize a LoRA-style CUDA extension.",
)
class OptimizeLoraAgent(SearchAgent):
    def __init__(
        self,
        context: RuntimeContext,
        *,
        builder: CandidateBuilder | None = None,
        harness: OptimizeLoraHarness | None = None,
        profiler: CandidateProfiler | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        config = config or DEFAULT_LORA_SEARCH_CONFIG
        stop_policy = StoppingPolicy(max_stalled_rounds=config.max_stalled_rounds)
        super().__init__(context, stop_policy=stop_policy)
        self.builder = builder or CandidateBuilder(context.workspace)
        self.harness = harness or OptimizeLoraHarness()
        self.profiler = profiler or CandidateProfiler(context.workspace, self.harness.config)
        artifact_path = Path.cwd() / ARTIFACT_NAME
        artifact_path.write_text(BASELINE_SOURCE, encoding="utf-8")
        self.result = OptimizeLoraResult(
            workflow="optimize-lora",
            started_at=context.started_at,
            finished_at=context.started_at,
            status="confirmed_baseline",
            summary="Bootstrapped the optimize-lora search.",
            submission_root=str(Path.cwd()),
            promoted_source_path=str(artifact_path),
            artifact_created=artifact_path.exists(),
            bootstrap_family="baseline",
            validation_shape=self.harness.config.validation_shape,
            notes=[
                "Bootstrap artifact is written before validation begins.",
                "The optimize-lora search space is authored by the LLM as parameterized candidate families.",
            ],
        )
        self.current_best: CandidateRecord | None = None
        self.baseline: CandidateRecord | None = None
        self.tried_family_names: list[str] = []
        self.config = config
        self._forwards: dict[str, Callable] = {}
        self._round_top_candidates: list[CandidateRecord] = []
        self._round_evaluator = RoundEvaluator(self)

    def run(self) -> OptimizeLoraResult:
        self.log(
            "starting optimize-lora "
            f"(validation_shape={self.harness.config.validation_shape}, "
            f"tier1_shapes={list(self.harness.config.tier1_shapes)}, "
            f"tier2_shapes={list(self.harness.config.tier2_shapes)}, "
            f"tier3_shapes={list(self.harness.config.tier3_shapes)})"
        )
        try:
            self.bootstrap_baseline()
            self.run_family_search()
            self.log("running final confirmation for top candidates")
            self._round_evaluator.finalize_winner()
            if self.result.winner_confirmed and self.result.best_candidate_kind == "optimized":
                self.result.status = "optimized"
                self.result.summary = "Confirmed a non-baseline optimized candidate after LLM-authored family search."
            else:
                self.result.status = "searched"
                self.result.summary = "Completed LLM-authored family search and kept the strongest verified artifact."
        except Exception as exc:  # noqa: BLE001
            append_workflow_error(self.result, "optimize-lora failed", exc)
            self.result.status = "failed"
            self.result.summary = str(exc)
        self.log(
            "finished optimize-lora "
            f"(status={self.result.status}, winner={self.result.current_best_candidate_id or 'none'}, "
            f"winner_kind={self.result.best_candidate_kind}, tier2={self.result.best_tier2_speedup:.4f}, "
            f"tier3={self.result.best_tier3_speedup:.4f})"
        )
        self.result.controller_trace = list(self.trace)
        self.result.finished_at = workflow_timestamp()
        return self.result

    def bootstrap_baseline(self) -> None:
        self.log(
            "validating baseline reference "
            f"on tier2 shapes {list(self.harness.tier_shapes(TIER2))}"
        )
        baseline = CandidateRecord(
            candidate_id="baseline-v0",
            family="baseline",
            source_hash="baseline_virtual",
            module_name="reference_impl",
            source_path=str(Path.cwd() / ARTIFACT_NAME),
            entrypoint_name="forward",
            origin="bootstrap",
            created_at=workflow_timestamp(),
            updated_at=workflow_timestamp(),
        )
        baseline.evaluation = self.harness.reference_evaluation(tier_name=TIER2, shapes=self.harness.tier_shapes(TIER2))
        baseline.comparison_summary = "Reference PyTorch implementation established as the virtual incumbent."
        self.baseline = baseline
        if not any(existing.candidate_id == baseline.candidate_id for existing in self.result.candidates):
            self.result.candidates.append(baseline)
        self.result.verified_baseline = True
        self._sync_best_result_state(baseline, best_kind="baseline")
        tier2 = baseline.evaluation.tier(TIER2)
        if tier2 is not None:
            self.log(
                f"baseline ready (candidate={baseline.candidate_id}, tier2_speedup={tier2.geometric_mean_speedup:.4f})"
            )
        self.record_trace(action="bootstrap_baseline_reference", candidate_id=baseline.candidate_id)

    def log(self, message: str) -> None:
        print(f"[sysforge][optimize-lora] {message}", flush=True)

    def _incumbent_forward_body(self) -> str:
        if self.current_best is None:
            return extract_forward_body(BASELINE_SOURCE)
        return extract_forward_body(Path(self.current_best.source_path).read_text(encoding="utf-8"))

    def _family_prompt_context(self) -> dict[str, object]:
        return {
            "env_summary": self._env_summary(),
            "tried_family_names": self.tried_family_names,
            "recent_body_history": self._recent_body_history(),
            "recent_plan_summary": self._recent_plan_summary(),
            "recent_body_values": self._recent_body_values(),
        }

    def _advance_family(
        self,
        *,
        family: CandidateFamilyDraft,
        incumbent_source: str,
        round_feedback: str,
    ) -> CandidateFamilyDraft:
        previous_family_name = family.family_name

        def recover(reason: str, detail: str = "") -> CandidateFamilyDraft:
            fallback = self._fallback_candidate_family(
                previous_family_name=previous_family_name,
                incumbent_source=incumbent_source,
            )
            trace = {
                "action": "family_fallback_local",
                "previous_family_name": previous_family_name,
                "family_name": fallback.family_name,
                "reason": reason,
            }
            if detail:
                trace["detail"] = detail
            self.record_trace(**trace)
            return fallback

        try:
            return revise_candidate_family(
                family=family,
                incumbent_source=incumbent_source,
                round_feedback=round_feedback,
                **self._family_prompt_context(),
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            self.record_trace(action="family_revision_failed", family_name=family.family_name, error=error)
            if "recent history" in error:
                return recover("duplicate_recent_history")
            if self._is_recoverable_llm_failure(exc):
                self.result.notes.append(f"Recovered from revise failure with local fallback family: {previous_family_name}")
                return recover("llm_revise_unavailable", error)
            self.result.errors.append(f"revise_candidate_family failed: {exc}")
            try:
                regenerated = generate_candidate_family(
                    baseline_source=incumbent_source,
                    min_distinct_variants=self.config.min_seed_variants,
                    **self._family_prompt_context(),
                )
            except Exception as regen_exc:  # noqa: BLE001
                if not self._is_recoverable_llm_failure(regen_exc):
                    self.result.errors.append(f"generate_candidate_family after revise failure failed: {regen_exc}")
                    self.record_trace(
                        action="family_regeneration_failed",
                        family_name=previous_family_name,
                        error=str(regen_exc),
                    )
                return recover(
                    "regeneration_failed" if not self._is_recoverable_llm_failure(regen_exc) else "llm_generate_unavailable",
                    str(regen_exc),
                )
            self.record_trace(
                action="family_regenerated",
                previous_family_name=previous_family_name,
                family_name=regenerated.family_name,
            )
            return regenerated

    def run_family_search(self) -> None:
        self.log("requesting initial candidate family from LLM")
        family = generate_candidate_family(
            baseline_source=self._incumbent_forward_body(),
            min_distinct_variants=self.config.min_seed_variants,
            **self._family_prompt_context(),
        )
        for _ in range(self.config.max_llm_rounds):
            round_index = self.begin_round(family.family_name)
            planned_variants = len(expand_family_mappings(family))
            self.log(
                f"round {round_index} started "
                f"(family={family.family_name}, planned_variants={planned_variants}, "
                f"stalled_rounds={self.state.stalled_rounds})"
            )
            self.tried_family_names.append(family.family_name)
            try:
                feedback = self._round_evaluator.evaluate_family_round(family=family, round_index=round_index)
            except Exception as exc:  # noqa: BLE001
                self.result.errors.append(f"evaluate_family_round failed: {exc}")
                self.log(f"round {round_index} failed for family={family.family_name}: {exc}")
                self.record_trace(action="family_round_failed", family_name=family.family_name, error=str(exc))
                break
            self.finish_round(improved=feedback.improved)
            self.log(
                f"round {round_index} finished "
                f"(improved={feedback.improved}, best_tier2={feedback.best_tier2_speedup:.4f}, "
                f"runner_up_tier2={feedback.second_tier2_speedup:.4f})"
            )
            if feedback.best_tier2_speedup >= self.config.clear_winner_speedup:
                if feedback.close_frontier:
                    self.log(
                        f"round {round_index} has a near winner but frontier is still close "
                        f"(best={feedback.best_tier2_speedup:.4f}, runner_up={feedback.second_tier2_speedup:.4f})"
                    )
                    self.record_trace(
                        action="clear_winner_deferred",
                        reason="close_frontier",
                        best_tier2_speedup=feedback.best_tier2_speedup,
                        second_tier2_speedup=feedback.second_tier2_speedup,
                        challenger_margin_pct=feedback.challenger_margin_pct,
                        challenger_separation_guard_pct=feedback.challenger_separation_guard_pct,
                    )
                else:
                    self.log(
                        f"stopping search after round {round_index}: clear winner at "
                        f"{feedback.best_tier2_speedup:.4f}x"
                    )
                    self.record_trace(
                        action="search_stopped",
                        reason="clear_winner",
                        best_tier2_speedup=feedback.best_tier2_speedup,
                        challenger_margin_pct=feedback.challenger_margin_pct,
                        challenger_separation_guard_pct=feedback.challenger_separation_guard_pct,
                    )
                    break
            decision = self.stop_decision()
            if decision.should_stop:
                self.log(f"stopping search after round {round_index}: {decision.reason}")
                self.result.skipped_steps.append({"step_name": "stop", "reason": decision.reason})
                self.record_trace(action="search_stopped", reason=decision.reason)
                break
            self.log(f"requesting revised family after round {round_index}")
            family = self._advance_family(
                family=family,
                incumbent_source=self._incumbent_forward_body(),
                round_feedback=feedback.summary,
            )

    def _candidate_plan_label(self, candidate: CandidateRecord | None) -> str:
        if candidate is None:
            return "other"
        body = candidate.parameter_values.get("FORWARD_BODY")
        return self._selection_plan_label(body) if isinstance(body, str) else "other"

    def _analyze_body(self, body: str) -> tuple[str, str, int]:
        split_cat = "torch::cat(" in body or ".narrow(" in body
        materialized = "dense.add_(" in body or "auto low = torch::matmul(A, tmp);" in body
        mm_out = "at::mm_out(" in body
        mm = "torch::mm(" in body
        matmul = "torch::matmul(" in body
        bt_contiguous = "B.transpose(0, 1).contiguous()" in body
        bt_view = "B.transpose(0, 1)" in body
        inplace_addmm = "addmm_(" in body
        functional_addmm = "torch::addmm(" in body
        copy = "copy_(" in body
        clone = "clone()" in body

        if split_cat:
            selection = "split_cat"
        elif materialized:
            selection = "materialized_low_rank"
        else:
            selection = ",".join(
                [
                    "mm_out_lowrank" if mm_out else "mm_family" if mm else "matmul_family" if matmul else "other",
                    "bt_contiguous" if bt_contiguous else "bt_view" if bt_view else "bt_other",
                    "addmm_accum" if inplace_addmm or functional_addmm else "other_accum",
                ]
            )

        execution_parts: list[str] = []
        if mm or mm_out:
            execution_parts.append("mm_family")
        elif matmul:
            execution_parts.append("matmul_family")
        if bt_contiguous:
            execution_parts.append("bt_contiguous")
        elif bt_view:
            execution_parts.append("bt_view")
        if inplace_addmm:
            execution_parts.append("inplace_addmm")
        elif functional_addmm:
            execution_parts.append("functional_addmm")
        elif materialized:
            execution_parts.append("materialized_low_rank")
        if mm_out:
            execution_parts.append("mm_out")
        if copy:
            execution_parts.append("copy")
        if clone:
            execution_parts.append("clone")
        if split_cat:
            execution_parts.append("split_cat")

        priority = 0
        priority += 4 if mm else 0
        priority += 3 if mm_out else 0
        priority += 2 if inplace_addmm else 0
        priority += 1 if "B.transpose(0, 1);" in body else 0
        priority -= 1 if ".contiguous()" in body else 0
        priority -= 3 if copy else 0
        priority -= 3 if clone else 0
        priority -= 2 if "torch::addmm(torch::matmul(" in body else 0
        priority -= 3 if materialized else 0
        priority -= 4 if split_cat else 0
        return selection, ",".join(execution_parts) if execution_parts else "other", priority

    def _selection_plan_label(self, body: str) -> str:
        return self._analyze_body(body)[0]

    def _env_summary(self) -> str:
        hints = self.context.env_hints or {}
        rows = hints.get("nvidia_smi_unique") or []
        row = rows[0] if rows else {}
        parts = [
            f"gpu={row.get('name', 'unknown')}",
            f"cc={row.get('compute_cap', '?')}",
            f"sm_clock_mhz={row.get('clocks.sm', '?')}",
            f"sm_clock_max_mhz={row.get('clocks.max.sm', '?')}",
            f"vram_mib={row.get('memory.total', '?')}",
            f"torch={getattr(torch, '__version__', '?')}",
            f"torch_cuda={getattr(getattr(torch, 'version', None), 'cuda', '?')}",
            f"nvcc_arch={self.context.config.nvcc_arch or 'default'}",
        ]
        return " ".join(parts)

    def _recent_body_history(self, limit: int = 6) -> str:
        rows: list[str] = []
        for candidate in reversed(self.result.candidates):
            body = candidate.parameter_values.get("FORWARD_BODY")
            if not isinstance(body, str):
                continue
            rows.append(
                json.dumps(
                    {
                        "candidate_id": candidate.candidate_id,
                        "comparison_summary": candidate.comparison_summary,
                        "failure_stage": candidate.failure_stage,
                        "body": body,
                    },
                    ensure_ascii=True,
                )
            )
            if len(rows) >= limit:
                break
        return "\n".join(reversed(rows)) if rows else "(none)"

    def _recent_body_values(self, limit: int = 6) -> list[str]:
        bodies: list[str] = []
        for candidate in reversed(self.result.candidates):
            body = candidate.parameter_values.get("FORWARD_BODY")
            if not isinstance(body, str) or body in bodies:
                continue
            bodies.append(body)
            if len(bodies) >= limit:
                break
        return list(reversed(bodies))

    def _execution_plan_label(self, body: str) -> str:
        return self._analyze_body(body)[1]

    def _recent_plan_summary(self, limit: int = 6) -> str:
        rows: list[str] = []
        seen: set[str] = set()
        for candidate in reversed(self.result.candidates):
            body = candidate.parameter_values.get("FORWARD_BODY")
            if not isinstance(body, str):
                continue
            label = self._execution_plan_label(body)
            if label in seen:
                continue
            seen.add(label)
            outcome = candidate.failure_stage or candidate.comparison_summary or "observed"
            rows.append(f"{label}: {outcome}")
            if len(rows) >= limit:
                break
        return "\n".join(reversed(rows)) if rows else "(none)"

    def _seen_source_hashes(self) -> set[str]:
        hashes = {candidate.source_hash for candidate in self.result.candidates if candidate.source_hash}
        hashes.add(source_digest(BASELINE_SOURCE))
        return hashes

    def _is_recoverable_llm_failure(self, exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "timeout",
            "timed out",
            "apitimeouterror",
            "connection",
            "temporarily unavailable",
            "service unavailable",
            "rate limit",
            "overloaded",
        )
        return any(marker in text for marker in markers)

    def _fallback_candidate_family(self, *, previous_family_name: str, incumbent_source: str) -> CandidateFamilyDraft:
        base_body = incumbent_source.strip()
        variants: list[str] = []
        for body in (
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::mm(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::mm(bt, X); auto out = torch::mm(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::mm(bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::mm(bt, X); auto out = torch::empty({W.size(0), X.size(1)}, X.options()); at::mm_out(out, W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); auto out = torch::empty({W.size(0), X.size(1)}, X.options()); at::mm_out(out, W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::empty({B.size(1), X.size(1)}, X.options()); at::mm_out(tmp, bt, X); auto out = torch::matmul(W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::empty({B.size(1), X.size(1)}, X.options()); at::mm_out(tmp, bt, X); auto out = torch::empty({W.size(0), X.size(1)}, X.options()); at::mm_out(out, W, X); out.addmm_(A, tmp, 1.0, 1.0); return out;",
            "auto bt = B.transpose(0, 1).contiguous(); auto tmp = torch::matmul(bt, X); return torch::addmm(torch::matmul(W, X), A, tmp);",
            base_body,
        ):
            if body and body not in variants:
                variants.append(body)
        default = variants[0]
        return CandidateFamilyDraft(
            family_name=f"{previous_family_name}_fallback",
            source_template="{{FORWARD_BODY}}",
            parameters=(
                ParameterSpec(name="FORWARD_BODY", values=tuple(variants), default=default, description="local fallback ATen body"),
            ),
            rationale="Deterministic ATen fallback family used when LLM revision cannot produce a novel valid family.",
            expected_bottleneck="launch_bound",
        )

    def _instantiate_family(self, family: CandidateFamilyDraft, *, round_index: int) -> list[ConcreteCandidate]:
        candidates: list[ConcreteCandidate] = []
        seen_source_hashes = self._seen_source_hashes()
        for mapping in expand_family_mappings(family):
            source = render_family_source(family.source_template, mapping)
            digest = source_digest(source)
            if digest in seen_source_hashes:
                continue
            seen_source_hashes.add(digest)
            candidates.append(
                ConcreteCandidate(
                    candidate_id=f"{family.family_name}-r{round_index}-v{len(candidates) + 1}",
                    family_name=family.family_name,
                    source=source,
                    parameter_values=mapping,
                )
            )
            if len(candidates) >= self.config.max_family_variants:
                break
        return candidates

    def _register_concrete_candidate(self, concrete: ConcreteCandidate, *, round_index: int) -> CandidateRecord:
        record = self.builder.register_candidate(
            candidate_id=concrete.candidate_id,
            family=concrete.family_name,
            source=concrete.source,
            entrypoint_name="forward",
        )
        record.origin = "seed_family" if round_index == 1 else "family_revision"
        record.parameter_values = dict(concrete.parameter_values)
        record.created_at = workflow_timestamp()
        record.updated_at = record.created_at
        return record

    def _body_priority_score(self, candidate: CandidateRecord | None) -> int:
        if candidate is None:
            return 0
        body = candidate.parameter_values.get("FORWARD_BODY")
        if not isinstance(body, str):
            return 0
        return self._analyze_body(body)[2]

    def _write_promoted_source(self, source_path: str) -> None:
        (Path.cwd() / ARTIFACT_NAME).write_text(Path(source_path).read_text(encoding="utf-8"), encoding="utf-8")

    def _promote_candidate(self, candidate: CandidateRecord, *, reason: str) -> None:
        self._write_promoted_source(candidate.source_path)
        self._sync_best_result_state(candidate, best_kind="optimized")
        candidate.comparison_summary = reason
        self.record_trace(action="candidate_promoted", candidate_id=candidate.candidate_id, reason=reason)

    def _sync_best_result_state(self, candidate: CandidateRecord, *, best_kind: str) -> None:
        self.current_best = candidate
        self.result.current_best_candidate_id = candidate.candidate_id
        self.result.current_best_family = candidate.family
        self.result.best_candidate_kind = best_kind
        self.result.benchmark_summary = candidate.evaluation.benchmark
        self.result.benchmark_tiers = list(candidate.evaluation.tier_summaries)
        tier2 = candidate.evaluation.tier(TIER2)
        if tier2 is not None:
            self.result.best_tier2_speedup = tier2.geometric_mean_speedup
