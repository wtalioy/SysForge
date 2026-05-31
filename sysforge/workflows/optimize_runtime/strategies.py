from __future__ import annotations

from dataclasses import dataclass

from .models import EngineStrategy


@dataclass(frozen=True)
class StrategySpec:
    candidate_id: str
    origin: str
    strategy: EngineStrategy
    rationale: str = ""
    techniques: tuple[str, ...] = ()


def strategy_key(strategy: EngineStrategy) -> tuple[str, str, str, str, str, str, str]:
    return (
        strategy.prefill_policy,
        strategy.kv_policy,
        strategy.attention_policy,
        strategy.decode_attention_policy,
        strategy.cache_growth_policy,
        strategy.cache_layout_policy,
        strategy.norm_policy,
    )


INITIAL_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        candidate_id="bootstrap-recompute",
        origin="bootstrap",
        strategy=EngineStrategy(
            prefill_policy="per_request",
            kv_policy="none",
            attention_policy="manual",
        ),
        rationale="Correctness bootstrap and debug reference.",
        techniques=("real per-layer KV cache reference",),
    ),
    StrategySpec(
        candidate_id="kv-cache-baseline",
        origin="baseline",
        strategy=EngineStrategy(
            prefill_policy="group_by_length",
            kv_policy="per_request_prealloc",
            attention_policy="sdpa_prefill_only",
            decode_attention_policy="sdpa_by_length",
            cache_growth_policy="decode_slack_64",
        ),
        rationale="Live-promoted KV-cache baseline using SDPA prefill/decode and small decode KV headroom.",
        techniques=("real per-layer KV cache", "batched prefill and decode", "PyTorch SDPA", "custom cache operations"),
    ),
)


SEARCH_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        candidate_id="pad-batch-sdpa-decode-slack64-prealloc",
        origin="llm",
        strategy=EngineStrategy(
            prefill_policy="pad_batch",
            kv_policy="per_request_prealloc",
            attention_policy="sdpa_prefill_only",
            decode_attention_policy="sdpa_by_length",
            cache_growth_policy="decode_slack_64",
        ),
        rationale="Combine the live-promoted slack-64 cache policy with padded prefill for churn and varied-prefill traces.",
        techniques=("PyTorch SDPA", "batched prefill and decode", "custom cache operations"),
    ),
    StrategySpec(
        candidate_id="pad-batch-sdpa-decode-prealloc",
        origin="llm",
        strategy=EngineStrategy(
            prefill_policy="pad_batch",
            kv_policy="per_request_prealloc",
            attention_policy="sdpa_prefill_only",
            decode_attention_policy="sdpa_by_length",
        ),
        rationale="Isolate padded mixed-length prefill on the current SDPA decode baseline without cache-growth changes.",
        techniques=("PyTorch SDPA", "batched prefill and decode", "better request-state management"),
    ),
    StrategySpec(
        candidate_id="sdpa-prefill-decode-slack128-prealloc",
        origin="llm",
        strategy=EngineStrategy(
            prefill_policy="group_by_length",
            kv_policy="per_request_prealloc",
            attention_policy="sdpa_prefill_only",
            decode_attention_policy="sdpa_by_length",
            cache_growth_policy="decode_slack_128",
        ),
        rationale="Current promoted attention path with medium decode headroom for long-decode/churn traces.",
        techniques=("PyTorch SDPA", "custom cache operations", "better request-state management"),
    ),
    StrategySpec(
        candidate_id="pad-batch-sdpa-decode-slack-prealloc",
        origin="llm",
        strategy=EngineStrategy(
            prefill_policy="pad_batch",
            kv_policy="per_request_prealloc",
            attention_policy="sdpa_prefill_only",
            decode_attention_policy="sdpa_by_length",
            cache_growth_policy="decode_slack_128",
        ),
        rationale="Aggressive mixed-length prefill plus decode-heavy candidate with strong robust-case evidence.",
        techniques=("PyTorch SDPA", "custom attention path", "custom cache operations", "better memory layout"),
    ),
    StrategySpec(
        candidate_id="sdpa-prefill-decode-slack256-prealloc",
        origin="llm",
        strategy=EngineStrategy(
            prefill_policy="group_by_length",
            kv_policy="per_request_prealloc",
            attention_policy="sdpa_prefill_only",
            decode_attention_policy="sdpa_by_length",
            cache_growth_policy="decode_slack_256",
        ),
        rationale="Current promoted attention path with larger decode headroom for machines where realloc/copy churn dominates.",
        techniques=("PyTorch SDPA", "custom cache operations", "better request-state management"),
    ),
)
