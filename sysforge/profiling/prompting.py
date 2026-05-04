from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..core.config import Config
from ..integrations.llm import LLMError, chat_json


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs) -> str:
    return load_prompt(name).format(**kwargs)


SYSTEM_PROMPT = load_prompt("system.txt")


class ProbePrompter:
    def __init__(self, config: Config, hints: dict) -> None:
        self.config = config
        self.hints = hints

    def llm_budget(self, deadline: float) -> float:
        return max(5.0, deadline - time.monotonic())

    def _history_prompt(
        self,
        *,
        deadline: float,
        prompt_name: str,
        formatter: Callable[[list[dict]], str],
        history_entries: list[dict],
        temperature: float,
        **kwargs,
    ):
        prompt = render_prompt(
            prompt_name,
            history=formatter(history_entries),
            **kwargs,
        )
        return chat_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=temperature,
            retries=1,
            deadline_s=self.llm_budget(deadline),
        )

    def _hint_line(self) -> str:
        rows = (self.hints.get("nvidia_smi_unique") or []) if isinstance(self.hints, dict) else []
        if not rows:
            return "(none)"
        row = rows[0]
        return (
            f"gpu={row.get('name', '?')} cc={row.get('compute_cap', '?')} "
            f"clocks_max_mhz={row.get('clocks.max.sm', '?')} "
            f"clocks_now_mhz={row.get('clocks.sm', '?')} "
            f"vram_mib={row.get('memory.total', '?')} "
            "(POSSIBLY SPOOFED or clock-locked; do NOT report as answer)"
        )

    def generate_probe(self, target, spec, deadline):
        prompt = render_prompt(
            "generate_probe.txt",
            target=target,
            category=spec.category,
            unit=spec.unit,
            strategy=spec.strategy,
            description=spec.description,
            plausible_min=spec.plausible_min,
            plausible_max=spec.plausible_max,
            hints=self._hint_line(),
            run_timeout_s=int(self.config.run_timeout_s),
            run_timeout_s_safe=max(1, int(self.config.run_timeout_s) - 5),
            num_trials=32,
        )
        return chat_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.3,
            retries=1,
            deadline_s=self.llm_budget(deadline),
        )

    def fix_compile(self, target, source, compile_result, history_entries, deadline):
        return self._history_prompt(
            deadline=deadline,
            prompt_name="fix_compile_error.txt",
            formatter=lambda history: "\n".join(
                f"  #{i}: v{entry['version']} stderr=\"{(entry.get('stderr') or '')[:500].replace(chr(10), ' | ')}\""
                for i, entry in enumerate(history, 1)
            ) or "  (none)",
            history_entries=history_entries,
            temperature=0.2,
            target=target,
            cmd=" ".join(compile_result.cmd),
            stdout=(compile_result.stdout or "")[-3000:],
            stderr=(compile_result.stderr or "")[-3000:],
            source=source,
        )

    def fix_runtime(self, target, source, run_result, history_entries, deadline):
        return self._history_prompt(
            deadline=deadline,
            prompt_name="fix_runtime_error.txt",
            formatter=lambda history: "\n".join(
                f"  #{i}: v{entry['version']} rc={entry.get('rc')} stderr=\"{(entry.get('stderr') or '')[:300].replace(chr(10), ' | ')}\" stdout=\"{(entry.get('stdout') or '')[:300].replace(chr(10), ' | ')}\""
                for i, entry in enumerate(history, 1)
            ) or "  (none)",
            history_entries=history_entries,
            temperature=0.2,
            target=target,
            rc=run_result.rc,
            wallclock_s=f"{run_result.wallclock_s:.3f}",
            timed_out=str(run_result.timed_out),
            stdout=(run_result.stdout or "")[-3000:],
            stderr=(run_result.stderr or "")[-3000:],
            source=source,
        )

    def fix_implausible(self, target, spec, source, stdout, value, reason, history_entries, deadline):
        return self._history_prompt(
            deadline=deadline,
            prompt_name="fix_implausible.txt",
            formatter=lambda history: "\n".join(
                f"  #{i}: v{entry['version']} value={entry['value']} {spec.unit} reason=\"{entry['reason']}\""
                for i, entry in enumerate(history, 1)
            ) or "  (none)",
            history_entries=history_entries,
            temperature=0.3,
            target=target,
            value=str(value),
            unit=spec.unit,
            plausible_min=spec.plausible_min,
            plausible_max=spec.plausible_max,
            reason=reason,
            stdout=(stdout or "")[-3000:],
            source=source,
        )

    def extract(self, target, spec, stdout, parse_hint, deadline):
        try:
            obj = chat_json(
                render_prompt(
                    "extract_value.txt",
                    target=target,
                    unit=spec.unit,
                    parse_hint=parse_hint or "(none given)",
                    stdout=(stdout or "")[-3000:],
                ),
                temperature=0.0,
                retries=1,
                deadline_s=self.llm_budget(deadline),
            )
            if isinstance(obj, dict):
                return obj
        except LLMError:
            pass
        return {"value": None, "unit": spec.unit, "confidence": 0.0, "reasoning": "LLM extraction failed"}
