from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.config import Config
from ..integrations import compiler, executor
from ..integrations.llm import LLMError
from ..integrations.workspace import Workspace
from .models import Attempt, ProbeOutcome
from .prompting import ProbePrompter
from .targets import looks_like_ncu_metric, strategy_for
from .validation import check_plausible_range, extract_unit_from_result, extract_value_from_result, sanity_check


def tail(text: str, size: int = 4000) -> str:
    if len(text) <= size:
        return text
    return text[-size:]


@dataclass
class RepairState:
    retries_left: int
    history: list[dict] = field(default_factory=list)


class ProbeExecutionEngine:
    def __init__(self, config: Config, workspace: Workspace) -> None:
        self.config = config
        self.workspace = workspace

    def compile_attempt(self, target: str, version: int, source: str, phase: str, rationale: str):
        source_path = self.workspace.write_source(target, version, source)
        binary_path = self.workspace.binary_path(target, version)
        compile_result = compiler.compile_cuda(
            source_path, binary_path, arch=self.config.nvcc_arch, timeout_s=self.config.compile_timeout_s
        )
        attempt = Attempt(
            version=version,
            phase=phase,
            source_path=str(source_path),
            compile_ok=compile_result.ok,
            compile_stderr=tail(compile_result.stderr, 2000),
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

    def run_attempt(self, binary_path, args):
        return executor.run_binary(binary_path, args=[str(arg) for arg in args], timeout_s=self.config.run_timeout_s)


class ProbeCoordinator:
    def __init__(self, config: Config, workspace: Workspace, hints: dict) -> None:
        self.config = config
        self.workspace = workspace
        self.hints = hints
        self.prompter = ProbePrompter(config, hints)
        self.execution = ProbeExecutionEngine(config, workspace)

    def _normalize_generation(self, generation: dict) -> tuple[str, list[str], str, str]:
        source = generation.get("source") or ""
        args = generation.get("args") or []
        if not isinstance(args, list):
            args = []
        parse_hint = generation.get("parse_hint") or ""
        rationale = generation.get("rationale") or ""
        return source, args, parse_hint, rationale

    def _extract_numeric_result(self, target: str, spec, stdout: str, extracted: dict) -> tuple[dict, float | None]:
        if extracted.get("value") is None:
            fallback_value = extract_value_from_result(target, stdout)
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
        return extracted, numeric_value

    def _append_accepted_sample(self, outcome: ProbeOutcome, version, value: float | None, extracted: dict, reasoning: str):
        outcome.accepted_samples.append({
            "version": version,
            "value": value,
            "confidence": extracted.get("confidence", 0.5),
            "reasoning": reasoning,
        })

    def _finalize_outcome(self, outcome: ProbeOutcome, target: str, spec, stdout: str, extracted: dict, numeric_value: float | None):
        values = [sample["value"] for sample in outcome.accepted_samples if sample["value"] is not None]
        values.sort()
        outcome.value = values[len(values) // 2] if values else numeric_value
        extracted_unit = extracted.get("unit") if isinstance(extracted, dict) else None
        stdout_unit = extract_unit_from_result(target, stdout)
        outcome.unit = (
            spec.unit if spec.unit and spec.unit != "unknown"
            else (stdout_unit or (extracted_unit if extracted_unit else spec.unit))
        )
        base_conf = float(extracted.get("confidence", 0.5)) if isinstance(extracted, dict) else 0.5
        outcome.confidence = min(0.99, base_conf + 0.1 * (len(outcome.accepted_samples) - 1))
        extracted_reasoning = extracted.get("reasoning", "") if isinstance(extracted, dict) else ""
        outcome.reasoning = (
            f"Median of {len(outcome.accepted_samples)} accepted sample(s); raw values={values}. "
            f"{extracted_reasoning}"
        ).strip()

    def _repair_compile(self, target, source, compile_result, state: RepairState, deadline: float):
        state.retries_left -= 1
        state.history.append({"version": self._version, "stderr": compile_result.stderr})
        return self.prompter.fix_compile(target, source, compile_result, state.history, deadline), "fix_compile"

    def _repair_runtime(self, target, source, run_result, state: RepairState, deadline: float):
        state.retries_left -= 1
        state.history.append({"version": self._version, "rc": run_result.rc, "stdout": run_result.stdout, "stderr": run_result.stderr})
        return self.prompter.fix_runtime(target, source, run_result, state.history, deadline), "fix_runtime"

    def _repair_implausible(self, target, spec, source, stdout, value, reason, state: RepairState, deadline: float):
        state.retries_left -= 1
        state.history.append({"version": self._version, "value": value, "reason": reason})
        return self.prompter.fix_implausible(target, spec, source, stdout, value, reason, state.history, deadline), "fix_implausible"

    def solve(self, target: str) -> ProbeOutcome:
        spec = strategy_for(target)
        outcome = ProbeOutcome(target=target, unit=spec.unit, value=None, confidence=0.0, reasoning="")
        if looks_like_ncu_metric(target):
            outcome.error = (
                f"target '{target}' has ncu metric/attribute shape; it should be routed to the analysis workflow path, not the probe path."
            )
            return outcome

        deadline = time.monotonic() + self.config.per_target_wallclock_s
        compile_state = RepairState(self.config.max_compile_fixes)
        runtime_state = RepairState(self.config.max_runtime_fixes)
        plausibility_state = RepairState(self.config.max_plausibility_retries)

        try:
            generation = self.prompter.generate_probe(target, spec, deadline)
        except LLMError as exc:
            outcome.error = f"LLM generate failed: {exc}"
            return outcome

        self._version = 1
        phase = "generate"
        while True:
            if time.monotonic() > deadline:
                outcome.error = outcome.error or "per-target wallclock exceeded"
                break

            source, args, parse_hint, rationale = self._normalize_generation(generation)
            attempt, compile_result, binary_path = self.execution.compile_attempt(
                target, self._version, source, phase, rationale
            )

            if not compile_result.ok:
                outcome.attempts.append(attempt)
                if compile_state.retries_left <= 0 or time.monotonic() > deadline:
                    outcome.error = "exhausted compile retries"
                    break
                try:
                    generation, phase = self._repair_compile(target, source, compile_result, compile_state, deadline)
                except LLMError as exc:
                    outcome.error = f"LLM fix_compile failed: {exc}"
                    break
                self._version += 1
                continue

            run_result = self.execution.run_attempt(binary_path, args)
            attempt.run_ok = run_result.ok
            attempt.run_rc = run_result.rc
            attempt.run_stdout_tail = tail(run_result.stdout, 4000)
            attempt.run_stderr_tail = tail(run_result.stderr, 2000)

            if not run_result.ok:
                outcome.attempts.append(attempt)
                if runtime_state.retries_left <= 0 or time.monotonic() > deadline:
                    outcome.error = "exhausted runtime retries"
                    break
                try:
                    generation, phase = self._repair_runtime(target, source, run_result, runtime_state, deadline)
                except LLMError as exc:
                    outcome.error = f"LLM fix_runtime failed: {exc}"
                    break
                self._version += 1
                continue

            extracted = self.prompter.extract(target, spec, run_result.stdout, parse_hint, deadline)
            extracted, numeric_value = self._extract_numeric_result(target, spec, run_result.stdout, extracted)
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
                self._append_accepted_sample(
                    outcome,
                    self._version,
                    numeric_value,
                    extracted,
                    extracted.get("reasoning", "") if isinstance(extracted, dict) else "",
                )
                remaining = max(0.0, deadline - time.monotonic())
                extra_runs = 0
                while extra_runs < 2 and remaining > self.config.run_timeout_s + 2:
                    rerun = self.execution.run_attempt(binary_path, args)
                    if not rerun.ok:
                        break
                    rerun_value = extract_value_from_result(target, rerun.stdout)
                    if rerun_value is None:
                        break
                    rerun_ok, _ = check_plausible_range(spec, rerun_value)
                    if not rerun_ok:
                        break
                    self._append_accepted_sample(
                        outcome,
                        f"{self._version}_rerun{extra_runs + 1}",
                        rerun_value,
                        {"confidence": 0.8},
                        "re-run of accepted probe for stability",
                    )
                    extra_runs += 1
                    remaining = max(0.0, deadline - time.monotonic())
                self._finalize_outcome(outcome, target, spec, run_result.stdout, extracted, numeric_value)
                break

            if plausibility_state.retries_left <= 0 or time.monotonic() > deadline:
                if numeric_value is not None:
                    outcome.value = numeric_value
                    outcome.unit = spec.unit
                    outcome.confidence = 0.1
                    outcome.reasoning = f"returning best-effort implausible value: {reason}"
                else:
                    outcome.error = "exhausted plausibility retries without numeric value"
                break

            try:
                generation, phase = self._repair_implausible(
                    target, spec, source, run_result.stdout, numeric_value, reason, plausibility_state, deadline
                )
            except LLMError as exc:
                outcome.error = f"LLM fix_implausible failed: {exc}"
                break
            self._version += 1

        return outcome
