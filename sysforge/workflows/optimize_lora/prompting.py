from __future__ import annotations

from pathlib import Path

from ...agent.prompts import json_prompt
from .families import family_concrete_variants_json, family_parameters_json, render_family_template
from .models import CandidateFamilyDraft, ParameterSpec


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _top_level_call_argument_count(source: str, marker: str) -> int | None:
    start = source.find(marker)
    if start < 0:
        return None
    depth = 1
    commas = 0
    has_content = False
    index = start + len(marker)
    while index < len(source):
        char = source[index]
        if char == "(":
            depth += 1
            has_content = True
        elif char == ")":
            depth -= 1
            if depth == 0:
                return 0 if not has_content else commas + 1
        elif char == "," and depth == 1:
            commas += 1
        elif not char.isspace():
            has_content = True
        index += 1
    return None


def _contains_invalid_addmm_arity(source: str) -> bool:
    marker = "torch::addmm("
    offset = 0
    while True:
        start = source.find(marker, offset)
        if start < 0:
            return False
        count = _top_level_call_argument_count(source[start:], marker)
        if count not in {3, 5}:
            return True
        offset = start + len(marker)


def _coerce_parameter(raw: dict) -> ParameterSpec:
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError("parameter name is required")
    values_raw = raw.get("values") or []
    values = tuple(values_raw) if isinstance(values_raw, list) else (values_raw,)
    if not values:
        raise ValueError(f"parameter {name} requires at least one value")
    default = raw.get("default", values[0])
    if default not in values:
        raise ValueError(f"parameter {name} default must be one of its values")
    return ParameterSpec(
        name=name,
        values=values,
        default=default,
        description=str(raw.get("description") or ""),
    )


def _coerce_family(obj: dict) -> CandidateFamilyDraft:
    parameters = tuple(_coerce_parameter(item) for item in (obj.get("parameters") or []) if isinstance(item, dict))
    concrete_variants_raw = obj.get("concrete_variants") or []
    concrete_variants: list[dict[str, object]] = []
    for index, item in enumerate(concrete_variants_raw):
        if not isinstance(item, dict):
            raise ValueError(f"concrete_variants[{index}] must be an object")
        values = item.get("parameter_values", item)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"concrete_variants[{index}] must provide a non-empty parameter_values object")
        concrete_variants.append({str(name): value for name, value in values.items()})
    return CandidateFamilyDraft(
        family_name=str(obj.get("family_name") or "llm_family"),
        source_template=str(obj.get("source_template") or "").strip(),
        parameters=parameters,
        concrete_variants=tuple(concrete_variants),
        rationale=str(obj.get("rationale") or ""),
        expected_bottleneck=str(obj.get("expected_bottleneck") or ""),
    )


def _family_json_prompt(prompt_name: str, **kwargs) -> CandidateFamilyDraft:
    validator_feedback = "(none)"
    min_distinct_variants = int(kwargs.pop("min_distinct_variants", 1))
    forbidden_bodies = {str(body) for body in (kwargs.pop("recent_body_values", []) or []) if str(body)}
    last_error = ""
    for _ in range(4):
        obj = json_prompt(
            PROMPT_DIR,
            prompt_name,
            system_name="json_system.txt",
            temperature=0.35,
            retries=2,
            validator_feedback=validator_feedback,
            **kwargs,
        )
        if not isinstance(obj, dict):
            last_error = f"{prompt_name} did not return a JSON object"
            validator_feedback = last_error
            continue
        family = _coerce_family(obj)
        problems: list[str] = []
        if not family.source_template:
            problems.append("source_template is empty")
        if "..." in family.source_template:
            problems.append("source_template contains ellipsis or pseudocode placeholders; every line must be real compilable code")
        if "PYBIND11_MODULE" not in family.source_template and any(
            marker in family.source_template
            for marker in ("#include", "TORCH_LIBRARY", "TORCH_LIBRARY_IMPL", "namespace {", "using namespace", "__global__")
        ):
            problems.append(
                "source_template mixes top-level C++ or registration constructs into a forward-body template; "
                "return either a complete .cu file with PYBIND11_MODULE or a plain forward body with statements only"
            )
        if "PYBIND11_MODULE" in family.source_template and "torch::Tensor forward(" not in family.source_template:
            problems.append("full-file source_template must define the required torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) entrypoint")
        if _contains_invalid_addmm_arity(family.source_template):
            problems.append("source_template uses invalid torch::addmm arity; valid forms are addmm(input, mat1, mat2) or addmm(input, mat1, mat2, beta, alpha)")
        total_variants = 0
        if family.concrete_variants:
            total_variants = len(family.concrete_variants)
            rendered_sources: set[str] = set()
            for index, mapping in enumerate(family.concrete_variants):
                try:
                    rendered = render_family_template(family.source_template, mapping)
                except ValueError as exc:
                    problems.append(f"concrete_variants[{index}] {exc}")
                    continue
                if rendered == family.source_template:
                    problems.append(f"concrete_variants[{index}] does not affect source_template")
                body_value = mapping.get("FORWARD_BODY")
                if isinstance(body_value, str) and body_value in forbidden_bodies:
                    problems.append(f"concrete_variants[{index}] reuses a FORWARD_BODY from recent history: {body_value[:120]}")
                if isinstance(body_value, str) and _contains_invalid_addmm_arity(body_value):
                    problems.append("concrete_variants uses invalid torch::addmm arity; valid forms are addmm(input, mat1, mat2) or addmm(input, mat1, mat2, beta, alpha)")
                rendered_sources.add(rendered)
            if len(rendered_sources) < total_variants:
                problems.append("concrete_variants contains duplicate rendered sources")
        else:
            total_variants = 1
            for parameter in family.parameters:
                total_variants *= max(1, len(parameter.values))
                if f"{{{parameter.name}}}" in family.source_template:
                    problems.append(
                        f"parameter {parameter.name} uses unsupported single-brace placeholder; use double braces"
                    )
                if f"{{{{{parameter.name}}}}}" not in family.source_template:
                    if parameter.name == "FORWARD_BODY":
                        problems.append("parameter FORWARD_BODY never appears in source_template; for body-first families, source_template should usually be exactly {{FORWARD_BODY}}")
                    else:
                        problems.append(f"parameter {parameter.name} never appears in source_template")
                if parameter.name == "FORWARD_BODY":
                    duplicate_values = [str(value) for value in parameter.values if str(value) in forbidden_bodies]
                    if duplicate_values:
                        preview = " | ".join(value[:120] for value in duplicate_values[:2])
                        problems.append(f"FORWARD_BODY reuses concrete bodies that already appeared in recent history: {preview}")
                    if any(_contains_invalid_addmm_arity(str(value)) for value in parameter.values):
                        problems.append("FORWARD_BODY uses invalid torch::addmm arity; valid forms are addmm(input, mat1, mat2) or addmm(input, mat1, mat2, beta, alpha)")
        if total_variants < min_distinct_variants:
            problems.append(f"family only exposes {total_variants} parameter combinations but needs at least {min_distinct_variants}")
        if not problems:
            return family
        last_error = "; ".join(problems)
        validator_feedback = f"Your previous JSON was rejected because {last_error}. Return a corrected family."
    raise ValueError(f"{prompt_name} failed validation: {last_error}")


def generate_candidate_family(
    *,
    env_summary: str,
    baseline_source: str,
    tried_family_names: list[str],
    recent_body_history: str,
    recent_plan_summary: str,
    recent_body_values: list[str],
    min_distinct_variants: int = 3,
) -> CandidateFamilyDraft:
    return _family_json_prompt(
        "generate_candidate_family.txt",
        min_distinct_variants=min_distinct_variants,
        env_summary=env_summary,
        baseline_source=baseline_source,
        tried_family_names=", ".join(tried_family_names) if tried_family_names else "(none)",
        recent_body_history=recent_body_history,
        recent_plan_summary=recent_plan_summary,
        recent_body_values=recent_body_values,
    )


def repair_candidate_family(
    *,
    env_summary: str,
    family: CandidateFamilyDraft,
    candidate_source: str,
    parameter_values: dict[str, object],
    failure_stage: str,
    failure_summary: str,
) -> CandidateFamilyDraft:
    return _family_json_prompt(
        "repair_candidate_family.txt",
        env_summary=env_summary,
        family_name=family.family_name,
        family_rationale=family.rationale,
        source_template=family.source_template,
        family_parameters=family_parameters_json(family),
        family_concrete_variants=family_concrete_variants_json(family),
        parameter_values=parameter_values,
        candidate_source=candidate_source,
        failure_stage=failure_stage,
        failure_summary=failure_summary,
    )


def revise_candidate_family(
    *,
    env_summary: str,
    family: CandidateFamilyDraft,
    incumbent_source: str,
    round_feedback: str,
    tried_family_names: list[str],
    recent_body_history: str,
    recent_plan_summary: str,
    recent_body_values: list[str],
) -> CandidateFamilyDraft:
    return _family_json_prompt(
        "revise_candidate_family.txt",
        env_summary=env_summary,
        family_name=family.family_name,
        family_rationale=family.rationale,
        source_template=family.source_template,
        family_parameters=family_parameters_json(family),
        family_concrete_variants=family_concrete_variants_json(family),
        incumbent_source=incumbent_source,
        round_feedback=round_feedback,
        tried_family_names=", ".join(tried_family_names) if tried_family_names else "(none)",
        recent_body_history=recent_body_history,
        recent_plan_summary=recent_plan_summary,
        recent_body_values=recent_body_values,
    )
