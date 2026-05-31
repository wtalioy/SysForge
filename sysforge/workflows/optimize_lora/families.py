from __future__ import annotations

import itertools
import json
import re

from .models import CandidateFamilyDraft
from .templates import render_source_from_body


_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_RAW_DOUBLE_BRACE_RE = re.compile(r"\{\{.*?\}\}")


def render_family_template(source_template: str, parameter_values: dict[str, object]) -> str:
    rendered = source_template
    for name, value in parameter_values.items():
        if f"{{{name}}}" in rendered:
            raise ValueError(f"single-brace family parameter is not supported: {name}")
        text = format_family_value(value)
        rendered = rendered.replace(f"{{{{{name}}}}}", text)
    return rendered


def expand_family_mappings(family: CandidateFamilyDraft) -> list[dict[str, object]]:
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


def render_family_source(source_template: str, parameter_values: dict[str, object]) -> str:
    rendered = render_family_template(source_template, parameter_values)
    unresolved = _PLACEHOLDER_RE.findall(rendered)
    unresolved.extend(_RAW_DOUBLE_BRACE_RE.findall(rendered))
    if unresolved:
        raise ValueError(f"unresolved family parameters: {', '.join(sorted(set(unresolved)))}")
    if "PYBIND11_MODULE" not in rendered:
        return render_source_from_body(rendered)
    return rendered


def format_family_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def family_parameters_json(family: CandidateFamilyDraft) -> str:
    return json.dumps(
        [
            {
                "name": parameter.name,
                "values": list(parameter.values),
                "default": parameter.default,
                "description": parameter.description,
            }
            for parameter in family.parameters
        ],
        indent=2,
    )


def family_concrete_variants_json(family: CandidateFamilyDraft) -> str:
    return json.dumps(list(family.concrete_variants), indent=2)
