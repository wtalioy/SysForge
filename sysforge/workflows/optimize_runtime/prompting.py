from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...agent.prompts import json_prompt
from .models import EngineStrategy
from .strategies import strategy_key


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def generate_strategy_batch(
    *,
    evidence: dict[str, Any],
    strategy_catalog: list[dict[str, Any]] | None = None,
    max_strategies: int = 1,
) -> list[EngineStrategy]:
    strategy_catalog = strategy_catalog or []
    payload = json_prompt(
        PROMPT_DIR,
        "generate_strategy_batch.txt",
        system_name="json_system.txt",
        temperature=0.2,
        retries=1,
        evidence_json=json.dumps(evidence, indent=2, sort_keys=True),
        strategy_catalog_json=json.dumps(strategy_catalog, indent=2, sort_keys=True),
        max_strategies=max_strategies,
    )
    return _parse_strategy_payload(payload, max_strategies=max_strategies, strategy_catalog=strategy_catalog)


def _parse_strategy_payload(
    payload: Any,
    *,
    max_strategies: int,
    strategy_catalog: list[dict[str, Any]] | None = None,
) -> list[EngineStrategy]:
    if not isinstance(payload, dict):
        raise ValueError("strategy response must be a JSON object")
    strategy_catalog = strategy_catalog or []
    catalog_by_id = {str(item.get("catalog_id")): item for item in strategy_catalog if item.get("catalog_id")}
    raw_catalog_ids = payload.get("selected_catalog_ids", [])
    if payload.get("strategies"):
        raise ValueError("strategy response must select catalogue IDs only")
    if not isinstance(raw_catalog_ids, list):
        raise ValueError("strategy response field 'selected_catalog_ids' must be a list")
    strategies: list[EngineStrategy] = []
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    mapped_items: list[dict[str, Any]] = []
    for raw_id in raw_catalog_ids:
        item = catalog_by_id.get(str(raw_id))
        if item is not None and isinstance(item.get("strategy"), dict):
            mapped_items.append(item["strategy"])
    for item in mapped_items:
        if len(strategies) >= max_strategies:
            break
        if not isinstance(item, dict):
            continue
        strategy = EngineStrategy.from_mapping(item)
        key = strategy_key(strategy)
        if key in seen:
            continue
        seen.add(key)
        strategies.append(strategy)
    return strategies
