from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..common import workflow_timestamp
from .prompting import revise_candidate_family, repair_candidate_family
from .harness import REFERENCE_KEY, reference_impl
from .models import CandidateEvaluation, CandidateFeedback, CandidateRecord, ConcreteCandidate, RoundFeedback
from .promotion import TIER1, TIER2, TIER3, compare_tier_summaries, select_best_finalist, separation_pct, threshold_pct_for_summaries

if TYPE_CHECKING:
    from .agent import OptimizeLoraAgent
    from .models import CandidateFamilyDraft


class RoundEvaluator:
    def __init__(self, owner: OptimizeLoraAgent) -> None:
        self.owner = owner

    def evaluate_family_round(self, *, family: CandidateFamilyDraft, round_index: int, allow_repair: bool = True) -> RoundFeedback:
        concrete_candidates = self.owner._instantiate_family(family, round_index=round_index)
        self.owner.log(
            f"round {round_index}: instantiated {len(concrete_candidates)} concrete candidates "
            f"for family={family.family_name}"
        )
        if round_index == 1 and len(concrete_candidates) < self.owner.config.min_seed_variants:
            self.owner.record_trace(
                action="family_distinct_variants_short",
                round_index=round_index,
                family_name=family.family_name,
                distinct_variants=len(concrete_candidates),
            )
            if allow_repair:
                family = revise_candidate_family(
                    family=family,
                    incumbent_source=self.owner._incumbent_forward_body(),
                    round_feedback=(
                        "The initial family rendered too few distinct concrete candidates after local substitution. "
                        f"Distinct concrete variants={len(concrete_candidates)} but at least {self.owner.config.min_seed_variants} are required. "
                        "Every declared parameter must materially change the emitted source or a meaningful API/algorithm choice. "
                        "Return a revised family with at least 3 distinct concrete variants."
                    ),
                    **self.owner._family_prompt_context(),
                )
                concrete_candidates = self.owner._instantiate_family(family, round_index=round_index)
        self.owner.record_trace(action="family_generated", round_index=round_index, family_name=family.family_name, rationale=family.rationale)
        is_seed_round = round_index == 1
        feedback_rows, successful, first_failure = self._screen_candidates(
            concrete_candidates=concrete_candidates,
            round_index=round_index,
        )
        self.owner.log(
            f"round {round_index}: screen complete "
            f"(successful={len(successful)}, failed={len(feedback_rows) - len(successful)})"
        )
        if is_seed_round:
            self.owner.result.seed_variants_screened = len(successful)
        else:
            self.owner.result.mutation_variants_screened += len(successful)
        self.rerun_close_tier1_candidates(successful, incumbent=self.owner.current_best or self.owner.baseline)
        ranked = self.select_tier2_shortlist(successful)
        if ranked:
            self.owner.log(
                f"round {round_index}: selected {len(ranked)} candidates for tier2 "
                f"({', '.join(candidate.candidate_id for candidate in ranked)})"
            )
        else:
            self.owner.log(f"round {round_index}: no candidates advanced to tier2")

        improved = False
        round_start_incumbent = self.owner.current_best
        evaluated_tier2: list[CandidateRecord] = []
        for candidate in ranked:
            assert round_start_incumbent is not None
            full = self._evaluate_candidate(candidate, tier_name=TIER2, incumbent=round_start_incumbent)
            evaluated_tier2.append(candidate)
            self._maybe_profile_candidate(candidate)
            decision = compare_tier_summaries(full, round_start_incumbent.evaluation.tier(TIER2))
            candidate.comparison_summary = decision.reason
            self.owner.log(
                f"round {round_index}: tier2 result for {candidate.candidate_id} "
                f"(geo={full.geometric_mean_speedup:.4f}, promote={decision.promote})"
            )
            self.owner.record_trace(
                action="candidate_ranked",
                round_index=round_index,
                candidate_id=candidate.candidate_id,
                incumbent_id=round_start_incumbent.candidate_id,
                reason=decision.reason,
                improvement_pct=decision.improvement_pct,
                geo_speedup=full.geometric_mean_speedup,
            )
            if decision.promote:
                improved = True
                self.owner._promote_candidate(candidate, reason=decision.reason)

        if allow_repair and not ranked and first_failure is not None and first_failure.failure_stage in {"compile", "runtime"}:
            family = repair_candidate_family(
                env_summary=self.owner._env_summary(),
                family=family,
                candidate_source=Path(first_failure.source_path).read_text(encoding="utf-8"),
                parameter_values=first_failure.parameter_values,
                failure_stage=first_failure.failure_stage or "compile",
                failure_summary=first_failure.failure_summary or "compile_failed",
            )
            self.owner.record_trace(action="family_repaired", round_index=round_index, family_name=family.family_name)
            return self.evaluate_family_round(family=family, round_index=round_index, allow_repair=False)

        if ranked:
            self.owner._round_top_candidates.extend(ranked)
        return self._build_round_feedback(
            family=family,
            round_index=round_index,
            improved=improved,
            feedback_rows=feedback_rows,
            evaluated_tier2=evaluated_tier2,
        )

    def finalize_winner(self) -> None:
        ranked_pool = [candidate for candidate in self.owner._round_top_candidates if candidate.evaluation and candidate.evaluation.tier(TIER2)]
        if self.owner.baseline is not None:
            ranked_pool.append(self.owner.baseline)
        finalists = self.select_finalists(list({candidate.candidate_id: candidate for candidate in ranked_pool}.values()))

        rows: list[tuple[str, object]] = []
        summaries: dict[str, CandidateRecord] = {}
        for candidate in finalists:
            candidate_key, forward_fn = (
                (REFERENCE_KEY, reference_impl)
                if candidate.family == "baseline"
                else (candidate.candidate_id, self.owner._forwards[candidate.candidate_id])
            )
            self.owner.log(
                f"final confirmation: benchmarking {candidate.candidate_id} "
                f"on tier3 shapes {list(self.owner.harness.tier_shapes(TIER3))}"
            )
            try:
                summary = self.owner.harness.evaluate_tier(
                    candidate_key,
                    forward_fn,
                    tier_name=TIER3,
                    shapes=self.owner.harness.tier_shapes(TIER3),
                    warmup=self.owner.config.final_confirm_warmup,
                    iters=self.owner.config.final_confirm_iters,
                )
            except Exception as exc:  # noqa: BLE001
                self.owner.result.errors.append(f"final_confirmation failed for {candidate.candidate_id}: {exc}")
                self.owner.record_trace(action="final_confirmation_failed", candidate_id=candidate.candidate_id, error=str(exc))
                continue
            candidate.evaluation.add_tier_summary(summary)
            rows.append((candidate.candidate_id, summary))
            summaries[candidate.candidate_id] = candidate
            self.owner.record_trace(action="final_confirmation_candidate", candidate_id=candidate.candidate_id, geo_speedup=summary.geometric_mean_speedup)
            self.owner.log(
                f"final confirmation: {candidate.candidate_id} geo_speedup={summary.geometric_mean_speedup:.4f}"
            )

        best = select_best_finalist(rows)
        if best is None:
            fallback = self.owner.current_best or self.owner.baseline
            if fallback is None or fallback.evaluation is None or fallback.evaluation.tier(TIER2) is None:
                self.owner.result.winner_confirmed = False
                self.owner.result.finalist_summary = None
                return
            self.owner.result.winner_confirmed = True
            self.owner.result.winner_confirmation_candidate_id = fallback.candidate_id
            self.owner.result.finalist_summary = fallback.evaluation.tier(TIER2)
            self.owner.result.best_tier3_speedup = self.owner.result.finalist_summary.geometric_mean_speedup
            self.owner.record_trace(action="final_winner_fallback", candidate_id=fallback.candidate_id, reason="tier3_unavailable")
            return

        best_id, finalist_summary = best
        winner = summaries[best_id]
        if winner.family != "baseline":
            self.owner._write_promoted_source(winner.source_path)
            best_kind = "optimized"
        else:
            best_kind = "baseline"
        self.owner._sync_best_result_state(winner, best_kind=best_kind)
        self.owner.result.winner_confirmation_candidate_id = winner.candidate_id
        self.owner.result.winner_confirmed = True
        self.owner.result.finalist_summary = finalist_summary
        self.owner.result.best_tier3_speedup = finalist_summary.geometric_mean_speedup
        self.owner.log(
            f"final winner: {winner.candidate_id} "
            f"(family={winner.family}, tier3_speedup={finalist_summary.geometric_mean_speedup:.4f})"
        )
        self.owner.record_trace(action="final_winner", candidate_id=winner.candidate_id, geo_speedup=finalist_summary.geometric_mean_speedup)

    def select_tier2_shortlist(self, successful: list[CandidateRecord]) -> list[CandidateRecord]:
        ranked = sorted(successful, key=self._tier1_rank_key, reverse=True)
        shortlist: list[CandidateRecord] = []
        used_plan_labels: set[str] = set()
        limit = self.owner.config.max_full_evaluations_per_round
        for candidate in ranked:
            label = self.owner._candidate_plan_label(candidate)
            if label in used_plan_labels:
                continue
            shortlist.append(candidate)
            used_plan_labels.add(label)
            if len(shortlist) >= limit:
                return shortlist
        for candidate in ranked:
            if candidate not in shortlist:
                shortlist.append(candidate)
            if len(shortlist) >= limit:
                break
        return shortlist

    def select_finalists(self, candidates: list[CandidateRecord]) -> list[CandidateRecord]:
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.evaluation.tier(TIER2).geometric_mean_speedup if candidate.evaluation and candidate.evaluation.tier(TIER2) else 0.0,
                self._tier1_incumbent_geomean(candidate),
                candidate.candidate_id,
            ),
            reverse=True,
        )
        if not ranked:
            return []
        finalists = [ranked[0]]
        limit = self.owner.config.final_confirmation_candidates
        top_label = self.owner._candidate_plan_label(ranked[0])
        for candidate in ranked[1:]:
            if len(finalists) >= limit:
                break
            if self.owner._candidate_plan_label(candidate) != top_label:
                finalists.append(candidate)
                break
        close_limit = max(limit, self.owner.config.max_close_finalists)
        top_summary = ranked[0].evaluation.tier(TIER2) if ranked[0].evaluation else None
        if top_summary is not None:
            for candidate in ranked[1:]:
                if len(finalists) >= close_limit:
                    break
                if candidate in finalists or candidate.evaluation is None:
                    continue
                candidate_summary = candidate.evaluation.tier(TIER2)
                if candidate_summary is None:
                    continue
                margin_pct = separation_pct(top_summary, candidate_summary)
                guard_pct = threshold_pct_for_summaries(
                    top_summary,
                    candidate_summary,
                    noise_guard_pct=max(top_summary.noise_guard_pct, candidate_summary.noise_guard_pct),
                )
                if margin_pct <= guard_pct:
                    finalists.append(candidate)
        for candidate in ranked[1:]:
            if len(finalists) >= close_limit:
                break
            if candidate not in finalists:
                finalists.append(candidate)
        return finalists[:close_limit]

    def rerun_close_tier1_candidates(self, successful: list[CandidateRecord], *, incumbent: CandidateRecord | None) -> None:
        if len(successful) <= 1 or incumbent is None:
            return
        ranked = sorted(successful, key=self._tier1_rank_key, reverse=True)
        cutoff = ranked[min(self.owner.config.max_full_evaluations_per_round, len(ranked)) - 1]
        cutoff_score = self._tier1_incumbent_geomean(cutoff)
        if cutoff_score <= 0.0:
            return
        rerun_targets = []
        for candidate in ranked:
            tier1 = candidate.evaluation.tier(TIER1) if candidate.evaluation is not None else None
            if tier1 is None or tier1.rerun_used:
                continue
            gap_pct = abs((self._tier1_incumbent_geomean(candidate) / cutoff_score) - 1.0) * 100.0
            band_pct = max(self._tier1_noise_band_pct(candidate), self._tier1_noise_band_pct(cutoff))
            if gap_pct <= band_pct:
                rerun_targets.append(candidate)
        if len(rerun_targets) <= 1:
            return
        for candidate in rerun_targets:
            summary = self._evaluate_candidate(
                candidate,
                tier_name=TIER1,
                incumbent=incumbent,
                warmup=self.owner.config.tier1_rerun_warmup,
                iters=self.owner.config.tier1_rerun_iters,
            )
            summary.rerun_used = True
            summary.noise_guard_pct = max(summary.noise_guard_pct, self.owner.config.tier1_rerun_band_pct)
            self.owner.record_trace(
                action="tier1_rerun",
                candidate_id=candidate.candidate_id,
                warmup=self.owner.config.tier1_rerun_warmup,
                iters=self.owner.config.tier1_rerun_iters,
                geo_speedup=summary.geometric_mean_speedup,
            )

    def _screen_candidates(
        self,
        *,
        concrete_candidates: list[ConcreteCandidate],
        round_index: int,
    ) -> tuple[list[CandidateFeedback], list[CandidateRecord], CandidateRecord | None]:
        feedback_rows: list[CandidateFeedback] = []
        successful: list[CandidateRecord] = []
        first_failure: CandidateRecord | None = None

        def fail(record: CandidateRecord, outcome: str, summary: str, *, stage: str = "", trace_action: str = "") -> None:
            nonlocal first_failure
            if stage:
                record.failure_stage = stage
                record.failure_summary = summary
            if first_failure is None:
                first_failure = record
            if trace_action:
                self.owner.record_trace(action=trace_action, candidate_id=record.candidate_id, error=summary)
            feedback_rows.append(
                CandidateFeedback(
                    candidate_id=record.candidate_id,
                    parameter_values=dict(record.parameter_values),
                    outcome=outcome,
                    summary=summary,
                )
            )

        for concrete in concrete_candidates:
            record = self.owner._register_concrete_candidate(concrete, round_index=round_index)
            if round_index == 1:
                self.owner.result.seed_count_run += 1
            else:
                self.owner.result.mutation_count_run += 1
            if not any(existing.candidate_id == record.candidate_id for existing in self.owner.result.candidates):
                self.owner.result.candidates.append(record)
            if not self._compile_single(record):
                fail(record, "compile_failed", record.failure_summary or "compile_failed")
                continue
            try:
                summary = self._evaluate_candidate(record, tier_name=TIER1, incumbent=self.owner.current_best or self.owner.baseline)
            except Exception as exc:  # noqa: BLE001
                fail(record, "runtime_failed", str(exc), stage="runtime", trace_action="candidate_runtime_failed")
                continue
            if not summary.correctness_passed:
                fail(record, "correctness_failed", summary.failure_reason, stage="correctness")
                continue
            successful.append(record)
            feedback_rows.append(
                CandidateFeedback(
                    candidate_id=record.candidate_id,
                    parameter_values=dict(record.parameter_values),
                    outcome="tier1_ok",
                    summary=f"tier1 geo_speedup={summary.geometric_mean_speedup:.4f}",
                    speedup=summary.geometric_mean_speedup,
                    weakest_shape=min(summary.shape_results, key=lambda row: row.speedup_vs_reference).shape_d if summary.shape_results else None,
                )
            )
        return feedback_rows, successful, first_failure

    def _build_round_feedback(
        self,
        *,
        family: CandidateFamilyDraft,
        round_index: int,
        improved: bool,
        feedback_rows: list[CandidateFeedback],
        evaluated_tier2: list[CandidateRecord],
    ) -> RoundFeedback:
        best_tier2_speedup = second_tier2_speedup = challenger_margin_pct = challenger_separation_guard_pct = 0.0
        close_frontier = False
        tier2_ranked = self.rank_round_tier2_candidates(evaluated_tier2)
        if tier2_ranked:
            leader_summary = tier2_ranked[0].evaluation.tier(TIER2)
            best_tier2_speedup = leader_summary.geometric_mean_speedup
            if len(tier2_ranked) > 1:
                runner_up_summary = tier2_ranked[1].evaluation.tier(TIER2)
                second_tier2_speedup = runner_up_summary.geometric_mean_speedup
                challenger_margin_pct = separation_pct(leader_summary, runner_up_summary)
                challenger_separation_guard_pct = threshold_pct_for_summaries(
                    leader_summary,
                    runner_up_summary,
                    noise_guard_pct=max(leader_summary.noise_guard_pct, runner_up_summary.noise_guard_pct),
                )
                close_frontier = challenger_margin_pct <= challenger_separation_guard_pct

        records_by_id = {candidate.candidate_id: candidate for candidate in self.owner.result.candidates}
        top_successes = {
            row.candidate_id
            for row in sorted(
                (row for row in feedback_rows if row.outcome == "tier1_ok"),
                key=lambda row: (
                    -(row.speedup or 0.0),
                    -self._tier1_incumbent_geomean(records_by_id.get(row.candidate_id)),
                    -self._tier1_largest_shape_speedup(records_by_id.get(row.candidate_id)),
                    -self.owner._body_priority_score(records_by_id.get(row.candidate_id)),
                    row.candidate_id,
                ),
            )[: self.owner.config.max_full_evaluations_per_round]
        }
        failure_counts: dict[str, int] = {}
        failure_examples: list[dict[str, object]] = []
        payload: dict[str, object] = {
            "round": round_index,
            "family": family.family_name,
            "rationale": family.rationale,
            "expected_bottleneck": family.expected_bottleneck,
            "improved": improved,
            "incumbent": None,
            "top_candidates": [],
            "failures": {"counts": failure_counts, "examples": failure_examples},
            "frontier": {
                "best_tier2_speedup": round(best_tier2_speedup, 6),
                "second_tier2_speedup": round(second_tier2_speedup, 6),
                "challenger_margin_pct": round(challenger_margin_pct, 6),
                "challenger_separation_guard_pct": round(challenger_separation_guard_pct, 6),
                "close_frontier": close_frontier,
            },
        }
        if self.owner.current_best is not None and self.owner.current_best.evaluation is not None:
            tier2 = self.owner.current_best.evaluation.tier(TIER2)
            if tier2 is not None:
                payload["incumbent"] = {
                    "candidate_id": self.owner.current_best.candidate_id,
                    "geo_speedup": round(tier2.geometric_mean_speedup, 6),
                    "largest_shape_median_ms": round(tier2.largest_shape_median_ms, 6),
                    "worst_regression_pct": round(tier2.worst_regression_pct, 6),
                    "shapes": [
                        {"shape_d": shape.shape_d, "median_ms": round(shape.median_ms, 6), "speedup_vs_reference": round(shape.speedup_vs_reference, 6)}
                        for shape in tier2.shape_results
                    ],
                }
        for row in feedback_rows:
            record = records_by_id.get(row.candidate_id)
            if row.outcome != "tier1_ok":
                failure_counts[row.outcome] = failure_counts.get(row.outcome, 0) + 1
                if len(failure_examples) < 3:
                    failure_examples.append(
                        {"candidate_id": row.candidate_id, "outcome": row.outcome, "params": row.parameter_values, "summary": row.summary}
                    )
                continue
            if row.candidate_id not in top_successes:
                continue
            candidate_payload: dict[str, object] = {
                "candidate_id": row.candidate_id,
                "speedup": None if row.speedup is None else round(row.speedup, 6),
                "weakest_shape": row.weakest_shape,
                "params": row.parameter_values,
                "summary": row.summary,
            }
            if record is not None and record.evaluation is not None:
                tier1 = record.evaluation.tier(TIER1)
                if tier1 is not None:
                    candidate_payload["largest_shape_median_ms"] = round(tier1.largest_shape_median_ms, 6)
                    candidate_payload["worst_regression_pct"] = round(tier1.worst_regression_pct, 6)
                    candidate_payload["tier1_shapes"] = [
                        {
                            "shape_d": shape.shape_d,
                            "median_ms": round(shape.median_ms, 6),
                            "speedup_vs_reference": round(shape.speedup_vs_reference, 6),
                            "speedup_vs_best": None if shape.speedup_vs_best is None else round(shape.speedup_vs_best, 6),
                        }
                        for shape in tier1.shape_results
                    ]
            if record is not None and record.profile_summary is not None:
                candidate_payload["profile"] = {
                    "shape_d": record.profile_summary.shape_d,
                    "kernel_count": record.profile_summary.kernel_count,
                    "top_kernels": record.profile_summary.top_kernels,
                    "bottleneck_hints": record.profile_summary.bottleneck_hints,
                    "headline_metrics": {
                        name: details.get("value")
                        for name, details in record.profile_summary.metrics.items()
                        if name in (
                            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                            "sm__warps_active.avg.pct_of_peak_sustained_active",
                            "smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg",
                            "smsp__pcsamp_warps_issue_stalled_short_scoreboard.avg",
                        )
                    },
                }
            payload["top_candidates"].append(candidate_payload)
        return RoundFeedback(
            round_index=round_index,
            family_name=family.family_name,
            improved=improved,
            summary=json.dumps(payload, indent=2, sort_keys=True),
            best_tier2_speedup=best_tier2_speedup,
            second_tier2_speedup=second_tier2_speedup,
            challenger_margin_pct=challenger_margin_pct,
            challenger_separation_guard_pct=challenger_separation_guard_pct,
            close_frontier=close_frontier,
        )

    def rank_round_tier2_candidates(self, candidates: list[CandidateRecord]) -> list[CandidateRecord]:
        return sorted(
            [candidate for candidate in candidates if candidate.evaluation is not None and candidate.evaluation.tier(TIER2) is not None],
            key=lambda candidate: (
                candidate.evaluation.tier(TIER2).geometric_mean_speedup,
                self._tier1_incumbent_geomean(candidate),
                candidate.candidate_id,
            ),
            reverse=True,
        )

    def _tier1_noise_band_pct(self, candidate: CandidateRecord | None) -> float:
        tier1 = candidate.evaluation.tier(TIER1) if candidate is not None and candidate.evaluation is not None else None
        return max(self.owner.config.tier1_rerun_band_pct, tier1.noise_guard_pct) if tier1 is not None else self.owner.config.tier1_rerun_band_pct

    def _tier1_rank_key(self, candidate: CandidateRecord) -> tuple[float, float, float, float, str]:
        tier1 = candidate.evaluation.tier(TIER1) if candidate.evaluation else None
        return (
            self._tier1_incumbent_geomean(candidate),
            -((tier1.worst_regression_pct if tier1 is not None else 0.0)),
            self._tier1_largest_shape_speedup(candidate),
            tier1.geometric_mean_speedup if tier1 is not None else 0.0,
            candidate.candidate_id,
        )

    def _tier1_incumbent_geomean(self, candidate: CandidateRecord | None) -> float:
        tier1 = candidate.evaluation.tier(TIER1) if candidate is not None and candidate.evaluation is not None else None
        values = [row.speedup_vs_best for row in tier1.shape_results if row.speedup_vs_best is not None] if tier1 is not None else []
        product = 1.0
        for value in values:
            product *= max(value, 1e-12)
        return product ** (1.0 / len(values)) if values else 0.0

    def _tier1_largest_shape_speedup(self, candidate: CandidateRecord | None) -> float:
        tier1 = candidate.evaluation.tier(TIER1) if candidate is not None and candidate.evaluation is not None else None
        if tier1 is None or not tier1.shape_results:
            return 0.0
        largest_shape = max(row.shape_d for row in tier1.shape_results)
        return next((row.speedup_vs_reference for row in tier1.shape_results if row.shape_d == largest_shape), 0.0)

    def _compile_single(self, candidate: CandidateRecord) -> bool:
        self.owner.log(f"compiling {candidate.candidate_id} (family={candidate.family})")
        compile_result, module = self.owner.builder.load_candidate(candidate)
        candidate.compile = compile_result
        self.owner.result.compile_time_s_total += compile_result.duration_s
        if module is None:
            candidate.failure_stage = "compile"
            candidate.failure_summary = compile_result.error or "compile_failed"
            self.owner.log(
                f"compile failed for {candidate.candidate_id}: {candidate.failure_summary} "
                f"(log={compile_result.log_path})"
            )
            self.owner.record_trace(action="compile_failed", candidate_id=candidate.candidate_id, error=candidate.failure_summary)
            return False
        self.owner._forwards[candidate.candidate_id] = getattr(module, candidate.entrypoint_name)
        status = "cached" if compile_result.status == "cache_hit" else "built"
        self.owner.log(
            f"compile {status} for {candidate.candidate_id} "
            f"(duration_s={compile_result.duration_s:.3f})"
        )
        return True

    def _evaluate_candidate(
        self,
        candidate: CandidateRecord,
        *,
        tier_name: str,
        incumbent: CandidateRecord | None,
        warmup: int | None = None,
        iters: int | None = None,
    ):
        self.owner.log(
            f"benchmarking {candidate.candidate_id} on {tier_name} "
            f"shapes {list(self.owner.harness.tier_shapes(tier_name))}"
        )
        candidate_key, forward_fn = (
            (REFERENCE_KEY, reference_impl)
            if candidate.family == "baseline"
            else (candidate.candidate_id, self.owner._forwards[candidate.candidate_id])
        )
        summary = self.owner.harness.evaluate_tier(
            candidate_key,
            forward_fn,
            tier_name=tier_name,
            shapes=self.owner.harness.tier_shapes(tier_name),
            incumbent_key=None if incumbent is None else REFERENCE_KEY if incumbent.family == "baseline" else incumbent.candidate_id,
            incumbent_forward=self.owner._forwards[incumbent.candidate_id] if incumbent is not None and incumbent.family != "baseline" else None,
            warmup=warmup if warmup is not None else self.owner.harness.config.screen_warmup if tier_name == TIER1 else None,
            iters=iters if iters is not None else self.owner.harness.config.screen_iters if tier_name == TIER1 else None,
        )
        if candidate.evaluation is None:
            candidate.evaluation = CandidateEvaluation()
        candidate.evaluation.add_tier_summary(summary)
        candidate.updated_at = workflow_timestamp()
        if tier_name == TIER1:
            self.owner.result.variants_benchmarked += 1
        self.owner.log(
            f"{tier_name} complete for {candidate.candidate_id} "
            f"(geo={summary.geometric_mean_speedup:.4f}, correctness={summary.correctness_passed})"
        )
        return summary

    def _maybe_profile_candidate(self, candidate: CandidateRecord) -> None:
        if not self.owner.config.profile_enabled:
            return
        if candidate.family == "baseline" or candidate.compile is None:
            self.owner.record_trace(action="profile_skipped", candidate_id=candidate.candidate_id, reason="baseline_or_uncompiled")
            return
        if candidate.profile_summary is not None:
            return
        tier2 = candidate.evaluation.tier(TIER2) if candidate.evaluation else None
        shape_d = max(tier2.shapes) if tier2 and tier2.shapes else self.owner.harness.config.validation_shape
        self.owner.log(f"profiling {candidate.candidate_id} at shape_d={shape_d}")
        candidate.profile_summary = self.owner.profiler.profile_candidate(candidate=candidate, compile_result=candidate.compile, shape_d=shape_d)
        if not candidate.profile_summary.error:
            self.owner.result.profiling_used = True
            self.owner.log(f"profile complete for {candidate.candidate_id}")
            self.owner.record_trace(action="profiled_candidate", candidate_id=candidate.candidate_id, shape_d=shape_d)
            return
        self.owner.log(f"profile failed for {candidate.candidate_id}: {candidate.profile_summary.error}")
        self.owner.record_trace(action="profile_failed", candidate_id=candidate.candidate_id, shape_d=shape_d, error=candidate.profile_summary.error)
