from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import source_digest
from .models import EngineStrategy
from .templates import ENGINE_TEMPLATE

FORBIDDEN_SOURCE_MARKERS = (
    "target/model_config.json",
    "ReferenceModel",
    "optimize_runtime.reference_model",
)


def render_engine(strategy: EngineStrategy) -> str:
    strategy = EngineStrategy.from_mapping(strategy.to_dict())
    source = ENGINE_TEMPLATE.replace(
        "__STRATEGY_JSON__",
        json.dumps(strategy.to_dict(), sort_keys=True),
    )
    for marker in FORBIDDEN_SOURCE_MARKERS:
        if marker in source:
            raise ValueError(f"rendered engine source contains forbidden marker: {marker}")
    return source


def write_engine(path: Path, strategy: EngineStrategy) -> str:
    source = render_engine(strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return source_digest(source, length=16)
