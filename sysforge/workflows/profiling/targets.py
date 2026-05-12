from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProbeSpec:
    category: str
    unit: str
    strategy: str
    plausible_min: float
    plausible_max: float
    description: str


CATALOG: dict[str, ProbeSpec] = {
    "l1_latency_cycles": ProbeSpec(
        "latency_l1", "cycles", "pointer_chase_l1", 15, 120,
        "L1/data-cache hit latency via dependent pointer-chase fitting in L1 (< 16 KB per SM). Typical: 20-40 cycles.",
    ),
    "l2_latency_cycles": ProbeSpec(
        "latency_l2", "cycles", "pointer_chase_l2", 150, 600,
        "L2 hit latency: pointer-chase over a working set that spills L1 but fits L2 (e.g. 1-4 MB, still inside typical L2). Typical: 200-300 cycles.",
    ),
    "dram_latency_cycles": ProbeSpec(
        "latency_dram", "cycles", "pointer_chase_dram", 300, 1500,
        "DRAM/global-memory latency: pointer-chase over a working set LARGER than L2 (>= 128 MB for modern GPUs) with a random permutation and stride >= 128 B to defeat stride+stream prefetchers. Typical Ampere/Ada: 400-700 cycles. A value <300 almost certainly means L2 residency contamination.",
    ),
    "l2_cache_capacity_kb": ProbeSpec(
        "capacity_cache", "KB", "latency_size_sweep", 256, 262144,
        "L2 physical size inferred from the latency-vs-working-set-size cliff.",
    ),
    "shared_mem_peak_bw_gbps": ProbeSpec(
        "bandwidth_shmem", "GB/s", "shmem_copy_loop", 500, 30000,
        "Effective peak shared-memory bandwidth via many-block tight copy/reduction.",
    ),
    "global_mem_peak_bw_gbps": ProbeSpec(
        "bandwidth_global", "GB/s", "strided_copy_kernel", 50, 4000,
        "Effective peak global-memory bandwidth via large-vector copy/triad.",
    ),
    "actual_boost_clock_mhz": ProbeSpec(
        "clock", "MHz", "sustained_clock_measure", 100, 3000,
        "Stable core clock measured as (clock64() cycles) / (elapsed wallclock seconds), around the same kernel invocation. Do NOT compute from FLOP/s.",
    ),
    "bank_conflict_penalty_cycles": ProbeSpec(
        "conflict_delta", "cycles", "bank_conflict_delta", 0, 200,
        "Extra cycles per shared-memory access caused by N-way bank conflict, measured as latency delta vs conflict-free access.",
    ),
    "max_shmem_per_block_kb": ProbeSpec(
        "capacity_shmem", "KB", "shmem_config_sweep", 8, 256,
        "Largest dynamic shared-memory allocation per block that still launches successfully.",
    ),
    "sm_count": ProbeSpec(
        "topology_smcount", "count", "distinct_smid_count", 1, 200,
        "Number of visible SMs inferred by launching >> N_SM blocks, having each block record the %smid it runs on, and counting DISTINCT values. Must NOT return number of blocks or cudaDeviceProp::multiProcessorCount as the answer.",
    ),
}


FALLBACK = ProbeSpec(
    category="unknown",
    unit="unknown",
    strategy="llm_choose",
    plausible_min=float("-inf"),
    plausible_max=float("inf"),
    description="Target is not in the built-in catalog; ask the LLM to classify and design.",
)


def get_spec(name: str) -> ProbeSpec | None:
    return CATALOG.get(name)


def strategy_for(target: str) -> ProbeSpec:
    spec = CATALOG.get(target)
    if spec is not None:
        return spec
    lower = target.lower()
    for key, catalog_spec in CATALOG.items():
        if key in lower or lower in key:
            return catalog_spec
    return FALLBACK


def looks_like_ncu_metric(name: str) -> bool:
    normalized = name.strip()
    if not normalized:
        return False
    if normalized in CATALOG:
        return False
    return "__" in normalized


def partition_targets(targets: Iterable[str]) -> tuple[list[str], list[str]]:
    probes: list[str] = []
    metrics: list[str] = []
    for target in targets:
        (metrics if looks_like_ncu_metric(target) else probes).append(target)
    return probes, metrics


def load_target_spec(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    targets = data.get("targets") or data.get("metrics") or []
    if not isinstance(targets, list):
        return []
    return [str(target).strip() for target in targets if str(target).strip()]
