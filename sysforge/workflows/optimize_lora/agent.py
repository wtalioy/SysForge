from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import strftime
from typing import Callable
import torch
from ...agent import SearchAgent, StoppingPolicy
from ...agent.llm import has_llm_config
from ...runtime import RuntimeContext
from ..common import stamp_finished
from ..registry import register_workflow
from . import prompting as family_agent
from .build import BASELINE_SOURCE, CandidateBuilder, source_hash
from .harness import OptimizeLoraHarness, HarnessConfig
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
from .templates import extract_forward_body, render_source_from_body


ARTIFACT_NAME = "optimized_lora.cu"
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_SINGLE_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_RAW_DOUBLE_BRACE_RE = re.compile(r"\{\{.*?\}\}")


@dataclass(frozen=True)
class SearchConfig:
    max_family_variants: int = 6
    min_seed_variants: int = 3
    max_full_evaluations_per_round: int = 3
    max_llm_rounds: int = 6
    final_confirmation_candidates: int = 3
    max_close_finalists: int = 4
    clear_winner_speedup: float = 1.05
    profile_enabled: bool = True
    final_confirm_warmup: int = 2
    final_confirm_iters: int = 6
    tier1_rerun_warmup: int = 2
    tier1_rerun_iters: int = 4
    tier1_rerun_band_pct: float = 1.5
    max_stalled_rounds: int = 3

    @classmethod
    def from_env(cls) -> SearchConfig:
        env_int = lambda name, default: int(os.environ.get(name, str(default)))
        env_float = lambda name, default: float(os.environ.get(name, str(default)))
        return cls(
            max_family_variants=env_int("OPTIMIZE_LORA_MAX_FAMILY_VARIANTS", 6),
            min_seed_variants=env_int("OPTIMIZE_LORA_MIN_SEED_VARIANTS", 3),
            max_full_evaluations_per_round=env_int("OPTIMIZE_LORA_MAX_FULL_EVALS_PER_ROUND", 3),
            max_llm_rounds=env_int("OPTIMIZE_LORA_MAX_LLM_ROUNDS", 6),
            final_confirmation_candidates=env_int("OPTIMIZE_LORA_FINAL_CONFIRMATION_CANDIDATES", 3),
            max_close_finalists=env_int("OPTIMIZE_LORA_MAX_CLOSE_FINALISTS", 4),
            clear_winner_speedup=env_float("OPTIMIZE_LORA_CLEAR_WINNER_SPEEDUP", 1.05),
            profile_enabled=os.environ.get("OPTIMIZE_LORA_PROFILE_ENABLED", "1") == "1",
            final_confirm_warmup=env_int("OPTIMIZE_LORA_FINAL_CONFIRM_WARMUP", 2),
            final_confirm_iters=env_int("OPTIMIZE_LORA_FINAL_CONFIRM_ITERS", 6),
            tier1_rerun_warmup=env_int("OPTIMIZE_LORA_TIER1_RERUN_WARMUP", 2),
            tier1_rerun_iters=env_int("OPTIMIZE_LORA_TIER1_RERUN_ITERS", 4),
            tier1_rerun_band_pct=env_float("OPTIMIZE_LORA_TIER1_RERUN_BAND_PCT", 1.5),
            max_stalled_rounds=env_int("OPTIMIZE_LORA_MAX_STALLED_ROUNDS", 3),
        )

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
        config = config or SearchConfig.from_env()
        stop_policy = StoppingPolicy(max_stalled_rounds=config.max_stalled_rounds)
        super().__init__(context, stop_policy=stop_policy)
        self.builder = builder or CandidateBuilder(context.workspace)
        self.harness = harness or OptimizeLoraHarness(HarnessConfig.from_env())
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
            llm_enabled=has_llm_config(
                api_key=context.config.api_key,
                base_url=context.config.base_url,
                model=context.config.base_model,
            ),
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
            f"(llm_enabled={self.result.llm_enabled}, "
            f"validation_shape={self.harness.config.validation_shape}, "
            f"tier1_shapes={list(self.harness.config.tier1_shapes)}, "
            f"tier2_shapes={list(self.harness.config.tier2_shapes)}, "
            f"tier3_shapes={list(self.harness.config.tier3_shapes)})"
        )
        self.bootstrap_baseline()
        if self.result.llm_enabled:
            self.run_family_search()
        else:
            self.log("llm search unavailable; keeping validated baseline only")
            self.record_trace(action="family_search_skipped", reason="llm_disabled")
        self.log("running final confirmation for top candidates")
        self._round_evaluator.finalize_winner()
        if self.result.winner_confirmed and self.result.best_candidate_kind == "optimized":
            self.result.status = "optimized"
            self.result.summary = "Confirmed a non-baseline optimized candidate after LLM-authored family search."
        elif self.result.llm_enabled:
            self.result.status = "searched"
            self.result.summary = "Completed LLM-authored family search and kept the strongest verified artifact."
        else:
            self.result.status = "confirmed_baseline"
            self.result.summary = "Validated the bootstrap baseline because the LLM search loop was unavailable."
        self.log(
            "finished optimize-lora "
            f"(status={self.result.status}, winner={self.result.current_best_candidate_id or 'none'}, "
            f"winner_kind={self.result.best_candidate_kind}, tier2={self.result.best_tier2_speedup:.4f}, "
            f"tier3={self.result.best_tier3_speedup:.4f})"
        )
        self.result.controller_trace = list(self.trace)
        stamp_finished(self.result)
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
            created_at=strftime("%Y-%m-%dT%H:%M:%S"),
            updated_at=strftime("%Y-%m-%dT%H:%M:%S"),
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
            return family_agent.revise_candidate_family(
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
                regenerated = family_agent.generate_candidate_family(
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
        try:
            family = family_agent.generate_candidate_family(
                baseline_source=self._incumbent_forward_body(),
                min_distinct_variants=self.config.min_seed_variants,
                **self._family_prompt_context(),
            )
        except Exception as exc:  # noqa: BLE001
            self.result.errors.append(f"generate_candidate_family failed: {exc}")
            self.log(f"initial family generation failed: {exc}")
            self.record_trace(action="family_generation_failed", error=str(exc))
            return
        for _ in range(self.config.max_llm_rounds):
            round_index = self.begin_round(family.family_name)
            planned_variants = len(self._iter_family_mappings(family))
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
        hashes.add(source_hash(BASELINE_SOURCE))
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

    def _iter_family_mappings(self, family: CandidateFamilyDraft) -> list[dict[str, object]]:
        if family.concrete_variants:
            return [dict(mapping) for mapping in family.concrete_variants]
        param_specs = list(family.parameters)
        combinations = list(itertools.product(*([spec.values for spec in param_specs] if param_specs else [()])))
        if not param_specs:
            return [{} for _ in combinations]
        default_values = tuple(
            spec.default if spec.default in spec.values else spec.values[min(len(spec.values) // 2, len(spec.values) - 1)]
            for spec in param_specs
        )
        ordered = ([default_values] if default_values in combinations else []) + [
            values for values in combinations if values != default_values
        ]
        seen: set[tuple[tuple[str, object], ...]] = set()
        mappings: list[dict[str, object]] = []
        for values in ordered:
            mapping = {spec.name: value for spec, value in zip(param_specs, values, strict=True)}
            signature = tuple(sorted(mapping.items()))
            if signature in seen:
                continue
            seen.add(signature)
            mappings.append(mapping)
        return mappings

    def _instantiate_family(self, family: CandidateFamilyDraft, *, round_index: int) -> list[ConcreteCandidate]:
        candidates: list[ConcreteCandidate] = []
        seen_source_hashes = self._seen_source_hashes()
        for mapping in self._iter_family_mappings(family):
            source = self._render_family_source(family.source_template, mapping)
            digest = source_hash(source)
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

    def _render_family_source(self, source_template: str, parameter_values: dict[str, object]) -> str:
        rendered = source_template
        for name, value in parameter_values.items():
            text = "true" if isinstance(value, bool) and value else "false" if isinstance(value, bool) else str(value)
            rendered = rendered.replace(f"{{{{{name}}}}}", text)
            rendered = rendered.replace(f"{{{name}}}", text)
        unresolved = _PLACEHOLDER_RE.findall(rendered)
        unresolved.extend(name for name in _SINGLE_PLACEHOLDER_RE.findall(rendered) if name in parameter_values)
        unresolved.extend(_RAW_DOUBLE_BRACE_RE.findall(rendered))
        if unresolved:
            raise ValueError(f"unresolved family parameters: {', '.join(sorted(set(unresolved)))}")
        if "PYBIND11_MODULE" not in rendered:
            return render_source_from_body(rendered)
        return rendered

    def _register_concrete_candidate(self, concrete: ConcreteCandidate, *, round_index: int) -> CandidateRecord:
        record = self.builder.register_candidate(
            candidate_id=concrete.candidate_id,
            family=concrete.family_name,
            source=concrete.source,
            entrypoint_name="forward",
        )
        record.origin = "seed_family" if round_index == 1 else "family_revision"
        record.parameter_values = dict(concrete.parameter_values)
        record.created_at = strftime("%Y-%m-%dT%H:%M:%S")
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
