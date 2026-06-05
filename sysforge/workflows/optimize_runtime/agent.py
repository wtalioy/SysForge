from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ...agent import BaseAgent
from ...runtime import RuntimeContext
from ..artifacts import source_digest
from ..common import append_workflow_error, workflow_timestamp
from ..registry import register_workflow
from .defaults import DEFAULT_RUNTIME_WORKFLOW_CONFIG, RuntimeWorkflowConfig
from .harness import OptimizeRuntimeHarness
from .models import EngineStrategy, RuntimeCandidateRecord, RuntimeOptimizationResult
from .promotion import candidate_score, choose_winner, promote_candidate
from .renderer import render_engine
from .strategies import INITIAL_STRATEGIES, SEARCH_STRATEGIES, strategy_key
from .prompting import generate_strategy_batch


@register_workflow(
    name="optimize-runtime",
    description="Generate, validate, benchmark, and promote an LLM inference runtime.",
)
class OptimizeRuntimeAgent(BaseAgent):
    def __init__(
        self,
        context: RuntimeContext,
        *,
        harness: OptimizeRuntimeHarness | None = None,
        workflow_config: RuntimeWorkflowConfig = DEFAULT_RUNTIME_WORKFLOW_CONFIG,
    ) -> None:
        super().__init__(context=context)
        self.submission_root = Path.cwd()
        self.candidate_dir = context.workspace.probes_dir / "optimize_runtime"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.harness = harness or OptimizeRuntimeHarness(
            logs_dir=context.workspace.logs_dir / "optimize_runtime",
            config=workflow_config,
        )
        self.max_llm_rounds = workflow_config.max_llm_rounds
        self.max_llm_strategies_per_round = workflow_config.max_llm_strategies_per_round
        self.result = RuntimeOptimizationResult(
            workflow="optimize-runtime",
            started_at=context.started_at,
            finished_at=context.started_at,
            status="started",
            summary="Starting runtime optimization.",
            submission_root=str(self.submission_root),
            promoted_engine_path=str(self.submission_root / "engine.py"),
            artifact_created=False,
            notes=[
                "Candidates are rendered from a fixed correctness-first engine template.",
                "LLM strategy search proposes schema-validated strategy JSON only.",
            ],
        )

    def run(self) -> RuntimeOptimizationResult:
        self.log(
            "starting optimize-runtime "
            f"(device={self.harness.config.device}, "
            f"benchmark_runs={self.harness.config.benchmark_runs}, "
            f"benchmark_discard_runs={self.harness.config.benchmark_discard_runs}, "
            f"llm_rounds={self.max_llm_rounds})"
        )
        try:
            self._prepare_public_weights_if_needed()
            candidates = self._initial_candidates()
            self._write_initial_engine(candidates)
            self._evaluate_candidates(candidates)
            self._run_llm_strategy_search(candidates)
            winner = choose_winner(candidates)
            if winner is None:
                self.result.status = "failed"
                self.result.summary = "No candidate passed correctness and benchmark gates."
                self.log("no promotable candidate found")
            else:
                promote_candidate(winner, self.submission_root / "engine.py")
                self.result.status = "optimized" if winner.origin == "llm" else "confirmed_baseline"
                self.result.summary = f"Promoted {winner.candidate_id}."
                self.result.artifact_created = (self.submission_root / "engine.py").exists()
                self.result.correctness_passed = winner.correctness_passed
                self.result.benchmark_summary = winner.benchmark
                self.log(f"promoted candidate {winner.candidate_id} to engine.py")
                smoke = self.harness.check_correctness(self.submission_root / "engine.py", label="promoted")
                if not smoke.passed:
                    self.result.status = "failed"
                    self.result.summary = f"Promoted candidate failed final correctness check: {smoke.failure_summary}"
                    self.result.errors.append(self.result.summary)
                else:
                    self.log("promoted engine final correctness check passed")
                    final_benchmark = self.harness.run_benchmark_suite(
                        self.submission_root / "engine.py",
                        label="promoted",
                    )
                    if final_benchmark.passed and final_benchmark.benchmark is not None:
                        self.result.benchmark_summary = final_benchmark.benchmark
                        self.result.notes.append("Promoted engine final benchmark passed.")
                        self.log(
                            "promoted engine final benchmark passed "
                            f"(mixed={final_benchmark.benchmark.mixed_tokens_per_second:.2f}, "
                            f"decode={final_benchmark.benchmark.decode_tokens_per_second:.2f}, "
                            f"mixed_spread={final_benchmark.benchmark.spread_pct.get('mixed', 0.0):.2f}%, "
                            f"churn={final_benchmark.benchmark.case_tokens_per_second.get('churn', 0.0):.2f}, "
                            f"varied_prefill="
                            f"{final_benchmark.benchmark.case_tokens_per_second.get('varied_prefill', 0.0):.2f})"
                        )
                    else:
                        self.result.status = "failed"
                        self.result.summary = f"Promoted candidate failed final benchmark: {final_benchmark.failure_summary}"
                        self.result.errors.append(self.result.summary)
        except Exception as exc:  # noqa: BLE001
            append_workflow_error(self.result, "optimize-runtime failed", exc)
            self.result.status = "failed"
            self.result.summary = str(exc)
        self.result.candidates = list(self.result.candidates)
        self.result.controller_trace = list(self.trace)
        self.result.artifact_created = (self.submission_root / "engine.py").exists()
        self.result.finished_at = workflow_timestamp()
        self.log(f"finished optimize-runtime (status={self.result.status})")
        return self.result

    def _prepare_public_weights_if_needed(self) -> None:
        config_path = self.harness.model_config_path
        weight_path = self.harness.weight_dir / "model.pt"
        if weight_path.exists() or not config_path.exists():
            return
        self.log("preparation: generating missing toy weights")
        command = [
            sys.executable,
            str(self.submission_root / "scripts" / "generate_toy_weights.py"),
            "--config",
            str(config_path),
            "--output",
            str(weight_path),
        ]
        subprocess.run(command, cwd=self.submission_root, check=True)

    def _initial_candidates(self) -> list[RuntimeCandidateRecord]:
        return [
            self._render_candidate(spec.candidate_id, spec.origin, spec.strategy)
            for spec in INITIAL_STRATEGIES
        ]

    def _render_candidate(
        self,
        candidate_id: str,
        origin: str,
        strategy: EngineStrategy,
    ) -> RuntimeCandidateRecord:
        source = render_engine(strategy)
        digest = source_digest(source, length=16)
        path = self.candidate_dir / candidate_id / "engine.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        candidate = RuntimeCandidateRecord(
            candidate_id=candidate_id,
            strategy=strategy,
            engine_path=str(path),
            source_hash=digest,
            origin=origin,
        )
        self.result.candidates.append(candidate)
        self.record_trace(
            action="candidate_rendered",
            candidate_id=candidate_id,
            origin=origin,
            source_hash=digest,
            strategy=strategy.to_dict(),
        )
        self.log(f"rendered {origin} candidate {candidate_id} ({digest})")
        return candidate

    def _write_initial_engine(self, candidates: list[RuntimeCandidateRecord]) -> None:
        baseline = next(candidate for candidate in candidates if candidate.candidate_id == "kv-cache-baseline")
        promote_candidate(baseline, self.submission_root / "engine.py")
        self.result.artifact_created = True
        self.log("bootstrap: wrote initial kv-cache baseline engine.py before validation")

    def _evaluate_candidates(self, candidates: list[RuntimeCandidateRecord]) -> None:
        for candidate in candidates:
            self._evaluate_candidate(candidate)

    def _evaluate_candidate(self, candidate: RuntimeCandidateRecord) -> None:
        engine_path = Path(candidate.engine_path)
        self.log(f"correctness gate: {candidate.candidate_id}")
        correctness = self.harness.check_correctness(engine_path, label=candidate.candidate_id)
        candidate.correctness_passed = correctness.passed
        if not correctness.passed:
            candidate.failure_stage = "correctness"
            candidate.failure_summary = correctness.failure_summary
            self.log(f"correctness failed: {candidate.candidate_id}: {correctness.failure_summary}")
            return
        self.log(f"correctness passed: {candidate.candidate_id}")
        self.log(f"benchmark: {candidate.candidate_id}")
        benchmark = self.harness.run_benchmark_suite(engine_path, label=candidate.candidate_id)
        if not benchmark.passed or benchmark.benchmark is None:
            candidate.failure_stage = "benchmark"
            candidate.failure_summary = benchmark.failure_summary
            self.log(f"benchmark failed: {candidate.candidate_id}: {benchmark.failure_summary}")
            return
        candidate.benchmark = benchmark.benchmark
        self.log(
            "benchmark passed: "
            f"{candidate.candidate_id} "
            f"prefill={benchmark.benchmark.prefill_tokens_per_second:.2f} "
            f"decode={benchmark.benchmark.decode_tokens_per_second:.2f} "
            f"mixed={benchmark.benchmark.mixed_tokens_per_second:.2f} "
            f"runs={benchmark.benchmark.run_count} "
            f"mixed_spread={benchmark.benchmark.spread_pct.get('mixed', 0.0):.2f}% "
            f"decode_spread={benchmark.benchmark.spread_pct.get('decode', 0.0):.2f}% "
            f"churn={benchmark.benchmark.case_tokens_per_second.get('churn', 0.0):.2f} "
            f"long_decode={benchmark.benchmark.case_decode_tokens_per_second.get('long_decode', 0.0):.2f} "
            f"varied_prefill={benchmark.benchmark.case_tokens_per_second.get('varied_prefill', 0.0):.2f}"
        )

    def _run_llm_strategy_search(self, candidates: list[RuntimeCandidateRecord]) -> None:
        for round_index in range(1, max(0, self.max_llm_rounds) + 1):
            should_continue = self._run_llm_strategy_round(candidates, round_index=round_index)
            if not should_continue:
                break

    def _run_llm_strategy_round(self, candidates: list[RuntimeCandidateRecord], *, round_index: int = 1) -> bool:
        self.log(f"LLM strategy round {round_index}: generating strategy JSON")

        existing_keys = {strategy_key(candidate.strategy) for candidate in candidates}
        strategy_catalog = self._llm_strategy_catalog(existing_keys)
        incumbent = self._incumbent_candidate(candidates)
        evidence = {
            "policy": {
                "goal": "select catalogue candidates most likely to improve the promoted runtime on measured evidence",
                "max_new_candidates": self.max_llm_strategies_per_round,
                "round": round_index,
                "requested_rounds": self.max_llm_rounds,
                "promotion_priority": [
                    "mixed_tokens_per_second",
                    "churn tokens_per_second",
                    "long_decode decode_tokens_per_second",
                    "decode_tokens_per_second",
                    "varied_prefill tokens_per_second",
                    "lower peak_memory_mb",
                ],
                "avoid": [
                    "full recompute variants",
                    "duplicate strategies",
                    "unsupported implementation work",
                ],
            },
            "incumbent_id": incumbent.candidate_id if incumbent is not None else "",
            "best_by_mixed": self._best_candidate_id(candidates, "mixed"),
            "best_by_case": {
                "churn": self._best_candidate_id(candidates, "churn"),
                "long_decode": self._best_candidate_id(candidates, "long_decode_decode"),
                "varied_prefill": self._best_candidate_id(candidates, "varied_prefill"),
            },
            "candidates": [self._candidate_evidence(candidate, incumbent) for candidate in candidates],
        }
        if not strategy_catalog:
            self.log(f"LLM strategy round {round_index} skipped because no novel catalogue entries remain")
            self.result.strategy_rounds.append(
                {"round": round_index, "strategy_count": 0, "strategies": [], "reason": "empty_catalog"}
            )
            return False
        try:
            strategies = generate_strategy_batch(
                evidence=evidence,
                strategy_catalog=strategy_catalog,
                max_strategies=self.max_llm_strategies_per_round,
            )
        except Exception as exc:  # noqa: BLE001
            self.result.errors.append(f"LLM strategy generation failed: {exc}")
            self.log(f"LLM strategy generation failed: {exc}")
            return False
        novel_strategies = []
        duplicate_strategies = []
        for strategy in strategies:
            if strategy_key(strategy) in existing_keys:
                duplicate_strategies.append(strategy)
            else:
                existing_keys.add(strategy_key(strategy))
                novel_strategies.append(strategy)
        self.result.strategy_rounds.append(
            {
                "round": round_index,
                "strategy_count": len(strategies),
                "novel_strategy_count": len(novel_strategies),
                "duplicate_strategy_count": len(duplicate_strategies),
                "candidate_catalog": strategy_catalog,
                "strategies": [strategy.to_dict() for strategy in strategies],
                "duplicates": [strategy.to_dict() for strategy in duplicate_strategies],
            }
        )
        if not novel_strategies:
            self.log(f"LLM strategy round {round_index} produced no novel strategies after duplicate filtering")
            return False
        for index, strategy in enumerate(novel_strategies, start=1):
            candidate = self._render_candidate(f"llm-r{round_index}-strategy-{index}", "llm", strategy)
            candidates.append(candidate)
            self._evaluate_candidate(candidate)
        return True

    def _llm_strategy_catalog(self, existing_keys: set[tuple[str, str, str, str, str, str, str]]) -> list[dict[str, object]]:
        catalog = []
        for spec in SEARCH_STRATEGIES:
            if strategy_key(spec.strategy) in existing_keys:
                continue
            catalog.append(
                {
                    "catalog_id": spec.candidate_id,
                    "rationale": spec.rationale,
                    "techniques": list(spec.techniques),
                    "strategy": spec.strategy.to_dict(),
                }
            )
        return catalog

    def _incumbent_candidate(self, candidates: list[RuntimeCandidateRecord]) -> RuntimeCandidateRecord | None:
        winner = choose_winner(candidates)
        if winner is not None:
            return winner
        return next((candidate for candidate in candidates if candidate.candidate_id == "kv-cache-baseline"), None)

    def _best_candidate_id(self, candidates: list[RuntimeCandidateRecord], metric: str) -> str:
        scored = [
            (self._candidate_metric(candidate, metric), candidate.candidate_id)
            for candidate in candidates
            if candidate.benchmark is not None
        ]
        if not scored:
            return ""
        return max(scored)[1]

    def _candidate_metric(self, candidate: RuntimeCandidateRecord, metric: str) -> float:
        benchmark = candidate.benchmark
        if benchmark is None:
            return 0.0
        if metric == "mixed":
            return benchmark.mixed_tokens_per_second
        if metric == "decode":
            return benchmark.decode_tokens_per_second
        if metric == "churn":
            return benchmark.case_tokens_per_second.get("churn", 0.0)
        if metric == "long_decode_decode":
            return benchmark.case_decode_tokens_per_second.get("long_decode", 0.0)
        if metric == "varied_prefill":
            return benchmark.case_tokens_per_second.get("varied_prefill", 0.0)
        return 0.0

    def _candidate_evidence(
        self,
        candidate: RuntimeCandidateRecord,
        incumbent: RuntimeCandidateRecord | None,
    ) -> dict[str, object]:
        benchmark = candidate.benchmark
        metrics: dict[str, float] = {}
        if benchmark is not None:
            metrics = {
                "mixed_tps": benchmark.mixed_tokens_per_second,
                "decode_tps": benchmark.decode_tokens_per_second,
                "churn_tps": benchmark.case_tokens_per_second.get("churn", 0.0),
                "long_decode_tps": benchmark.case_tokens_per_second.get("long_decode", 0.0),
                "long_decode_decode_tps": benchmark.case_decode_tokens_per_second.get("long_decode", 0.0),
                "varied_prefill_tps": benchmark.case_tokens_per_second.get("varied_prefill", 0.0),
                "peak_memory_mb": benchmark.peak_memory_mb,
                "mixed_spread_pct": benchmark.spread_pct.get("mixed", 0.0),
                "spread_pct": {key: round(value, 4) for key, value in benchmark.spread_pct.items()},
                "run_count": benchmark.run_count,
                "case_tps": {key: round(value, 4) for key, value in benchmark.case_tokens_per_second.items()},
                "case_decode_tps": {
                    key: round(value, 4) for key, value in benchmark.case_decode_tokens_per_second.items()
                },
                "promotion_score": [round(value, 4) for value in candidate_score(candidate)],
            }
            if incumbent is not None and incumbent.benchmark is not None:
                for metric_name in ("mixed", "decode", "churn", "long_decode_decode", "varied_prefill"):
                    base = self._candidate_metric(incumbent, metric_name)
                    value = self._candidate_metric(candidate, metric_name)
                    if base > 0:
                        metrics[f"{metric_name}_vs_incumbent_pct"] = (value - base) / base * 100.0
        return {
            "candidate_id": candidate.candidate_id,
            "origin": candidate.origin,
            "strategy": candidate.strategy.to_dict(),
            "passed": candidate.correctness_passed and candidate.benchmark is not None,
            "metrics": metrics,
            "failure_stage": candidate.failure_stage,
            "failure_summary": candidate.failure_summary[:1000],
        }

    def log(self, message: str) -> None:
        print(f"[sysforge][optimize-runtime] {message}", flush=True)
