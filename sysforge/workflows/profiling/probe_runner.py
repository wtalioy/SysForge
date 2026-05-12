from __future__ import annotations

import time

from . import prompting as profiling_prompts
from ...agent.llm import LLMError
from ...runtime import Config
from ...integrations import compiler, executor
from ...integrations.workspace import Workspace
from ..common import RetryBudget, accepted_sample_confidence, consume_retry, deadline_exceeded, tail_text
from .models import Attempt, ProbeOutcome
from .targets import looks_like_ncu_metric, strategy_for
from .validation import check_plausible_range, extract_unit_from_result, extract_value_from_result, sanity_check


class ProbeCoordinator:
    def __init__(self, config: Config, workspace: Workspace, hints: dict) -> None:
        self.config = config
        self.workspace = workspace
        self.hints = hints
        self.prompt_ctx = profiling_prompts.ProfilingPromptContext(config=config, hints=hints)

    def _repair_compile(self, target, source, compile_result, state: RetryBudget):
        if not consume_retry(state, {"version": self._version, "stderr": compile_result.stderr}):
            return None, None
        return profiling_prompts.fix_compile(self.prompt_ctx, target, source, compile_result, state.history), "fix_compile"

    def _repair_runtime(self, target, source, run_result, state: RetryBudget):
        if not consume_retry(
            state,
            {"version": self._version, "rc": run_result.rc, "stdout": run_result.stdout, "stderr": run_result.stderr},
        ):
            return None, None
        return profiling_prompts.fix_runtime(self.prompt_ctx, target, source, run_result, state.history), "fix_runtime"

    def _repair_implausible(self, target, spec, source, stdout, value, reason, state: RetryBudget):
        if not consume_retry(state, {"version": self._version, "value": value, "reason": reason}):
            return None, None
        return profiling_prompts.fix_implausible(
            self.prompt_ctx,
            target,
            spec,
            source,
            stdout,
            value,
            reason,
            state.history,
        ), "fix_implausible"

    def _advance_after_failed_attempt(
        self,
        *,
        outcome: ProbeOutcome,
        attempt,
        state: RetryBudget,
        exhausted_error: str,
        llm_error_prefix: str,
        append_attempt: bool = True,
        repair_call,
    ):
        if append_attempt:
            outcome.attempts.append(attempt)
        if state.retries_left <= 0 or deadline_exceeded(self._deadline):
            outcome.error = exhausted_error
            return None, None
        try:
            generation, phase = repair_call()
        except LLMError as exc:
            outcome.error = f"{llm_error_prefix}: {exc}"
            return None, None
        if generation is None:
            outcome.error = exhausted_error
            return None, None
        self._version += 1
        return generation, phase

    def _return_best_effort_implausible(self, outcome: ProbeOutcome, *, spec, numeric_value: float | None, reason: str) -> None:
        if numeric_value is not None:
            outcome.value = numeric_value
            outcome.unit = spec.unit
            outcome.confidence = 0.1
            outcome.reasoning = f"returning best-effort implausible value: {reason}"
            outcome.error = ""
            return
        outcome.error = "exhausted plausibility retries without numeric value"

    def _compile_probe_attempt(self, *, target: str, source: str, phase: str, rationale: str):
        source_path = self.workspace.write_source(target, self._version, source)
        binary_path = self.workspace.binary_path(target, self._version)
        compile_result = compiler.compile_cuda(
            source_path, binary_path, arch=self.config.nvcc_arch, timeout_s=self.config.compile_timeout_s
        )
        attempt = Attempt(
            version=self._version,
            phase=phase,
            source_path=str(source_path),
            compile_ok=compile_result.ok,
            compile_stderr=tail_text(compile_result.stderr, 2000),
            run_ok=False,
            run_rc=0,
            run_stdout_tail="",
            run_stderr_tail="",
            extracted=None,
            plausible=False,
            reject_reason="",
            rationale=rationale,
        )
        return attempt, compile_result, binary_path

    def solve(self, target: str) -> ProbeOutcome:
        spec = strategy_for(target)
        outcome = ProbeOutcome(target=target, unit=spec.unit, value=None, confidence=0.0, reasoning="")
        if looks_like_ncu_metric(target):
            outcome.error = (
                f"target '{target}' has ncu metric/attribute shape; it should be routed to the analysis workflow path, not the probe path."
            )
            return outcome

        deadline = time.monotonic() + self.config.per_target_wallclock_s
        self._deadline = deadline
        compile_state = RetryBudget(self.config.max_compile_fixes)
        runtime_state = RetryBudget(self.config.max_runtime_fixes)
        plausibility_state = RetryBudget(self.config.max_plausibility_retries)

        try:
            generation = profiling_prompts.generate_probe(self.prompt_ctx, target, spec)
        except LLMError as exc:
            outcome.error = f"LLM generate failed: {exc}"
            return outcome

        self._version = 1
        phase = "generate"
        while True:
            if deadline_exceeded(deadline):
                outcome.error = outcome.error or "per-target wallclock exceeded"
                break

            source = generation.get("source") or ""
            args = generation.get("args") or []
            if not isinstance(args, list):
                args = []
            parse_hint = generation.get("parse_hint") or ""
            rationale = generation.get("rationale") or ""
            attempt, compile_result, binary_path = self._compile_probe_attempt(
                target=target,
                source=source,
                phase=phase,
                rationale=rationale,
            )

            if not compile_result.ok:
                generation, phase = self._advance_after_failed_attempt(
                    outcome=outcome,
                    attempt=attempt,
                    state=compile_state,
                    exhausted_error="exhausted compile retries",
                    llm_error_prefix="LLM fix_compile failed",
                    repair_call=lambda: self._repair_compile(target, source, compile_result, compile_state),
                )
                if generation is None:
                    break
                continue

            run_result = executor.run_binary(
                binary_path,
                args=[str(arg) for arg in args],
                timeout_s=self.config.run_timeout_s,
            )
            attempt.run_ok = run_result.ok
            attempt.run_rc = run_result.rc
            attempt.run_stdout_tail = tail_text(run_result.stdout, 4000)
            attempt.run_stderr_tail = tail_text(run_result.stderr, 2000)

            if not run_result.ok:
                generation, phase = self._advance_after_failed_attempt(
                    outcome=outcome,
                    attempt=attempt,
                    state=runtime_state,
                    exhausted_error="exhausted runtime retries",
                    llm_error_prefix="LLM fix_runtime failed",
                    repair_call=lambda: self._repair_runtime(target, source, run_result, runtime_state),
                )
                if generation is None:
                    break
                continue

            extracted = profiling_prompts.extract(self.prompt_ctx, target, spec, run_result.stdout, parse_hint)
            if extracted.get("value") is None:
                fallback_value = extract_value_from_result(target, run_result.stdout)
                if fallback_value is not None:
                    extracted = {
                        "value": fallback_value,
                        "unit": spec.unit,
                        "confidence": 0.5,
                        "reasoning": "regex-matched RESULT line",
                    }
            value = extracted.get("value") if isinstance(extracted, dict) else None
            try:
                numeric_value = float(value) if value is not None else None
            except (TypeError, ValueError):
                numeric_value = None
            attempt.extracted = extracted

            plausible, reason = check_plausible_range(spec, numeric_value)
            if plausible:
                sanity_reason = sanity_check(target, spec, run_result.stdout, numeric_value, self.hints)
                if sanity_reason:
                    plausible = False
                    reason = sanity_reason

            attempt.plausible = plausible
            attempt.reject_reason = reason
            outcome.attempts.append(attempt)

            if plausible:
                outcome.accepted_samples.append({
                    "version": self._version,
                    "value": numeric_value,
                    "confidence": extracted.get("confidence", 0.5),
                    "reasoning": extracted.get("reasoning", "") if isinstance(extracted, dict) else "",
                })
                remaining = max(0.0, deadline - time.monotonic())
                extra_runs = 0
                while extra_runs < 2 and remaining > self.config.run_timeout_s + 2:
                    rerun = executor.run_binary(
                        binary_path,
                        args=[str(arg) for arg in args],
                        timeout_s=self.config.run_timeout_s,
                    )
                    if not rerun.ok:
                        break
                    rerun_value = extract_value_from_result(target, rerun.stdout)
                    if rerun_value is None:
                        break
                    rerun_ok, _ = check_plausible_range(spec, rerun_value)
                    if not rerun_ok:
                        break
                    outcome.accepted_samples.append({
                        "version": f"{self._version}_rerun{extra_runs + 1}",
                        "value": rerun_value,
                        "confidence": 0.8,
                        "reasoning": "re-run of accepted probe for stability",
                    })
                    extra_runs += 1
                    remaining = max(0.0, deadline - time.monotonic())
                values = sorted(sample["value"] for sample in outcome.accepted_samples if sample["value"] is not None)
                outcome.value = values[len(values) // 2] if values else numeric_value
                extracted_unit = extracted.get("unit") if isinstance(extracted, dict) else None
                stdout_unit = extract_unit_from_result(target, run_result.stdout)
                outcome.unit = (
                    spec.unit if spec.unit and spec.unit != "unknown"
                    else (stdout_unit or (extracted_unit if extracted_unit else spec.unit))
                )
                base_conf = float(extracted.get("confidence", 0.5)) if isinstance(extracted, dict) else 0.5
                outcome.confidence = accepted_sample_confidence(base_conf, len(outcome.accepted_samples))
                extracted_reasoning = extracted.get("reasoning", "") if isinstance(extracted, dict) else ""
                outcome.reasoning = (
                    f"Median of {len(outcome.accepted_samples)} accepted sample(s); raw values={values}. "
                    f"{extracted_reasoning}"
                ).strip()
                break

            if plausibility_state.retries_left <= 0 or deadline_exceeded(deadline):
                self._return_best_effort_implausible(outcome, spec=spec, numeric_value=numeric_value, reason=reason)
                break

            generation, phase = self._advance_after_failed_attempt(
                outcome=outcome,
                attempt=attempt,
                state=plausibility_state,
                exhausted_error="exhausted plausibility retries without numeric value",
                llm_error_prefix="LLM fix_implausible failed",
                append_attempt=False,
                repair_call=lambda: self._repair_implausible(
                    target, spec, source, run_result.stdout, numeric_value, reason, plausibility_state
                ),
            )
            if generation is None:
                if not outcome.error.startswith("LLM"):
                    self._return_best_effort_implausible(outcome, spec=spec, numeric_value=numeric_value, reason=reason)
                break

        return outcome
