from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..common import workflow_timestamp
from ..base import WorkflowResult


PREFILL_POLICIES = ("per_request", "pad_batch", "group_by_length")
KV_POLICIES = ("none", "per_request_cat", "per_request_prealloc")
ATTENTION_POLICIES = ("manual", "sdpa_prefill_only")
DECODE_ATTENTION_POLICIES = ("manual", "sdpa_by_length")
CACHE_GROWTH_POLICIES = ("power2", "decode_slack_64", "decode_slack_128", "decode_slack_256")
CACHE_LAYOUT_POLICIES = ("standard", "transposed_k")
NORM_POLICIES = ("torch", "triton_rmsnorm")


def _require_member(field_name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        joined = ", ".join(allowed)
        raise ValueError(f"{field_name} must be one of: {joined}; got {value!r}")
    return value


@dataclass(frozen=True)
class EngineStrategy:
    prefill_policy: str = "per_request"
    kv_policy: str = "per_request_prealloc"
    attention_policy: str = "manual"
    decode_attention_policy: str = "manual"
    cache_growth_policy: str = "power2"
    cache_layout_policy: str = "standard"
    norm_policy: str = "torch"

    def __post_init__(self) -> None:
        _require_member("prefill_policy", self.prefill_policy, PREFILL_POLICIES)
        _require_member("kv_policy", self.kv_policy, KV_POLICIES)
        _require_member("attention_policy", self.attention_policy, ATTENTION_POLICIES)
        _require_member("decode_attention_policy", self.decode_attention_policy, DECODE_ATTENTION_POLICIES)
        _require_member("cache_growth_policy", self.cache_growth_policy, CACHE_GROWTH_POLICIES)
        _require_member("cache_layout_policy", self.cache_layout_policy, CACHE_LAYOUT_POLICIES)
        _require_member("norm_policy", self.norm_policy, NORM_POLICIES)
        if self.kv_policy != "per_request_prealloc" and self.cache_growth_policy != "power2":
            raise ValueError("cache_growth_policy variants require kv_policy='per_request_prealloc'")
        if self.kv_policy != "per_request_prealloc" and self.cache_layout_policy != "standard":
            raise ValueError("cache_layout_policy variants require kv_policy='per_request_prealloc'")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "EngineStrategy":
        allowed = {
            "prefill_policy",
            "kv_policy",
            "attention_policy",
            "decode_attention_policy",
            "cache_growth_policy",
            "cache_layout_policy",
            "norm_policy",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown strategy field(s): {', '.join(sorted(unknown))}")
        return cls(**{key: payload[key] for key in allowed if key in payload})

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RuntimeBenchmarkSummary:
    prefill_tokens_per_second: float = 0.0
    decode_tokens_per_second: float = 0.0
    mixed_tokens_per_second: float = 0.0
    total_tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    benchmark_runs: list[list[dict[str, Any]]] = field(default_factory=list)
    run_count: int = 1
    spread_pct: dict[str, float] = field(default_factory=dict)
    case_tokens_per_second: dict[str, float] = field(default_factory=dict)
    case_decode_tokens_per_second: dict[str, float] = field(default_factory=dict)


@dataclass
class RuntimeCandidateRecord:
    candidate_id: str
    strategy: EngineStrategy
    engine_path: str
    source_hash: str
    origin: str = "seed"
    correctness_passed: bool = False
    stress_passed: bool = False
    benchmark: RuntimeBenchmarkSummary | None = None
    failure_stage: str = ""
    failure_summary: str = ""
    created_at: str = field(default_factory=workflow_timestamp)


@dataclass
class RuntimeOptimizationResult(WorkflowResult):
    status: str
    summary: str
    submission_root: str
    promoted_engine_path: str
    artifact_created: bool
    correctness_passed: bool = False
    stress_passed: bool = False
    benchmark_summary: RuntimeBenchmarkSummary | None = None
    candidates: list[RuntimeCandidateRecord] = field(default_factory=list)
    strategy_rounds: list[dict[str, Any]] = field(default_factory=list)
    controller_trace: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
