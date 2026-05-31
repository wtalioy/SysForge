from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...agent.llm import LLMError
from ...agent.prompts import cuda_prompt_file, json_prompt
from ...runtime import Config
from ..common import tail_text


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class ProfilingPromptContext:
    config: Config
    hints: dict


def _json(prompt_name: str, *, temperature: float = 0.2, retries: int = 1, **kwargs) -> dict:
    obj = json_prompt(
        PROMPT_DIR,
        prompt_name,
        temperature=temperature,
        retries=retries,
        **kwargs,
    )
    return obj if isinstance(obj, dict) else {}


def hint_line(hints: dict) -> str:
    rows = (hints.get("nvidia_smi_unique") or []) if isinstance(hints, dict) else []
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

def generate_probe(ctx: ProfilingPromptContext, target, spec):
    return _json(
        "generate_probe.txt",
        temperature=0.3,
        target=target,
        category=spec.category,
        unit=spec.unit,
        strategy=spec.strategy,
        description=spec.description,
        plausible_min=spec.plausible_min,
        plausible_max=spec.plausible_max,
        hints=hint_line(ctx.hints),
        run_timeout_s=int(ctx.config.run_timeout_s),
        run_timeout_s_safe=max(1, int(ctx.config.run_timeout_s) - 5),
        num_trials=32,
    )


def fix_compile(ctx: ProfilingPromptContext, target, source, compile_result, history_entries):
    return _json(
        "fix_compile_error.txt",
        history="\n".join(
            f"  #{i}: v{entry['version']} stderr=\"{(entry.get('stderr') or '')[:500].replace(chr(10), ' | ')}\""
            for i, entry in enumerate(history_entries, 1)
        ) or "  (none)",
        target=target,
        cmd=" ".join(compile_result.cmd),
        stdout=tail_text(compile_result.stdout or "", 3000),
        stderr=tail_text(compile_result.stderr or "", 3000),
        source=source,
    )


def fix_runtime(ctx: ProfilingPromptContext, target, source, run_result, history_entries):
    return _json(
        "fix_runtime_error.txt",
        history="\n".join(
            f"  #{i}: v{entry['version']} rc={entry.get('rc')} stderr=\"{(entry.get('stderr') or '')[:300].replace(chr(10), ' | ')}\" stdout=\"{(entry.get('stdout') or '')[:300].replace(chr(10), ' | ')}\""
            for i, entry in enumerate(history_entries, 1)
        ) or "  (none)",
        target=target,
        rc=run_result.rc,
        wallclock_s=f"{run_result.wallclock_s:.3f}",
        timed_out=str(run_result.timed_out),
        stdout=tail_text(run_result.stdout or "", 3000),
        stderr=tail_text(run_result.stderr or "", 3000),
        source=source,
    )


def fix_implausible(ctx: ProfilingPromptContext, target, spec, source, stdout, value, reason, history_entries):
    return _json(
        "fix_implausible.txt",
        temperature=0.3,
        history="\n".join(
            f"  #{i}: v{entry['version']} value={entry['value']} {spec.unit} reason=\"{entry['reason']}\""
            for i, entry in enumerate(history_entries, 1)
        ) or "  (none)",
        target=target,
        value=str(value),
        unit=spec.unit,
        plausible_min=spec.plausible_min,
        plausible_max=spec.plausible_max,
        reason=reason,
        stdout=tail_text(stdout or "", 3000),
        source=source,
    )


def extract(ctx: ProfilingPromptContext, target, spec, stdout, parse_hint):
    try:
        obj = _json(
            "extract_value.txt",
            temperature=0.0,
            target=target,
            unit=spec.unit,
            parse_hint=parse_hint or "(none given)",
            stdout=tail_text(stdout or "", 3000),
        )
        if obj:
            return obj
    except LLMError:
        pass
    return {"value": None, "unit": spec.unit, "confidence": 0.0, "reasoning": "LLM extraction failed"}


def generate_reference_gemm() -> str:
    return cuda_prompt_file(
        PROMPT_DIR,
        "generate_gemm.txt",
        temperature=0.2,
    )


def fix_reference_compile(source: str, compile_result) -> dict:
    return _json(
        "fix_compile_error.txt",
        target="gemm_reference",
        cmd=" ".join(compile_result.cmd),
        stdout=tail_text(compile_result.stdout, 2000),
        stderr=tail_text(compile_result.stderr, 2000),
        source=source,
        history="(none)",
    )


def analyze_metrics(metrics: str, ncu_csv: str) -> dict:
    return _json(
        "analyze_metrics.txt",
        metrics=metrics,
        ncu_csv=ncu_csv,
    )
