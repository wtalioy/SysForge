from __future__ import annotations

import re

from .targets import ProbeSpec


_KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")
_SWEEP_RE = re.compile(r"sweep[a-z_]*\s*=\s*([0-9eE.+,:\-]+)")


def extract_result_line(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.lstrip().startswith("RESULT "):
            return line
    return ""


def parse_kv_pairs(result_line: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in _KV_RE.findall(result_line):
        try:
            output[key] = float(value)
        except ValueError:
            continue
    return output


def extract_unit_from_result(target: str, stdout: str) -> str | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("RESULT") or target not in stripped:
            continue
        match = re.search(r"unit\s*=\s*(\S+)", stripped)
        if match:
            return match.group(1)
    return None


def extract_value_from_result(target: str, stdout: str) -> float | None:
    pattern = re.compile(r"RESULT\s+" + re.escape(target) + r"\s*=\s*([-+0-9.eE]+)")
    match = pattern.search(stdout)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def check_plausible_range(spec: ProbeSpec, value: float | None) -> tuple[bool, str]:
    if value is None:
        return False, "no numeric value extracted"
    if spec.plausible_min <= value <= spec.plausible_max:
        return True, ""
    return False, (
        f"value {value} outside plausible range [{spec.plausible_min}, {spec.plausible_max}] {spec.unit}"
    )


def _rule_degenerate_stats(target, spec, kv, value, line, hints):
    for lo, hi in (("min", "max"), ("cycles_min", "cycles_max"), ("ms_min", "ms_max")):
        if lo in kv and hi in kv and kv[lo] == kv[hi] and kv[lo] > 1e6:
            return (
                f"suspicious: {lo} == {hi} == {kv[lo]:.4g}; probable counter overflow or the loop was compiler-collapsed / optimized away"
            )
    return ""


def _rule_bare_zero(target, spec, kv, value, line, hints):
    if value is not None and value == 0 and not spec.category.endswith("_delta"):
        return "suspicious: reported value is exactly 0; the measurement loop was probably optimized away or the unit conversion dropped the value"
    return ""


def _rule_latency_stride(target, spec, kv, value, line, hints):
    if "stride_bytes" in kv and kv["stride_bytes"] < 128:
        return (
            f"suspicious: stride_bytes={kv['stride_bytes']} < 128 on a latency pointer-chase; consecutive hops share a 128-byte cache line, so the probe measures burst access, not true DRAM/L2 latency"
        )
    return ""


def _rule_dram_latency_too_low(target, spec, kv, value, line, hints):
    if value is not None and value < 300:
        return (
            f"suspicious: dram_latency={value} cycles is below L2-hit latency. Your pointer chain almost certainly REVISITS nodes faster than L2 can evict. Ensure the chain visits at least 1M DISTINCT nodes before looping, spaced >= 128 B apart, and that iters >> chain_length so the measurement timer covers at least one full chain traversal. A 2 GiB allocation does not help if the chain only uses a few thousand nodes of it."
        )
    return ""


def _parse_sweep(line: str) -> list[tuple[float, float]]:
    match = _SWEEP_RE.search(line)
    if not match:
        return []
    output: list[tuple[float, float]] = []
    for token in match.group(1).split(","):
        parts = token.split(":")
        try:
            size = float(parts[0])
            latency = float(parts[1]) if len(parts) > 1 else float("nan")
            output.append((size, latency))
        except ValueError:
            continue
    return output


def _rule_capacity_sweep(target, spec, kv, value, line, hints):
    if value is None:
        return ""
    sweep_max = kv.get("sweep_max_kb") or kv.get("sweep_max")
    points = _parse_sweep(line)
    if points and sweep_max is None:
        sweep_max = max(point[0] for point in points)
    if sweep_max is not None and abs(sweep_max - value) / max(1.0, sweep_max) < 0.05:
        return f"suspicious: reported capacity={value} equals sweep_max={sweep_max}; the sweep never passed through the latency cliff. Extend upward."
    if points and len(points) >= 3:
        latencies = [point[1] for point in points if point[1] == point[1]]
        if latencies and max(latencies) - min(latencies) < 1e-6:
            return f"suspicious: every sweep point reports identical latency {latencies[0]}; the inner measurement loop is returning a constant (likely compiler-collapsed, zero iters, or a missing device-sync)."
    match = re.search(r"sweep_med_cpl\s*=\s*([0-9eE.+,\-]+)", line)
    if match:
        try:
            values = [float(x) for x in match.group(1).split(",") if x]
        except ValueError:
            values = []
        if len(values) >= 4:
            peak = max(values)
            tail = min(values[-3:])
            if peak > 0 and tail < peak * 0.5:
                return (
                    f"suspicious: sweep_med_cpl is non-monotonic (peak={peak:.1f}, tail-min={tail:.1f}); a cache-capacity probe should show latency RISE and stay elevated past the cliff, not fall. You are timing total work, not per-load latency. Time the INNER chase loop with clock64() and divide by the number of dependent loads executed."
                )
    return ""


def _rule_conflict_delta(target, spec, kv, value, line, hints):
    if value is None:
        return ""
    working_set = kv.get("working_set_bytes") or (kv.get("smem_kb") or 0) * 1024
    if working_set and working_set < 4096:
        return (
            f"suspicious: working_set_bytes={working_set} is trivially small; conflict and conflict-free kernels both complete in a handful of cycles. Use a shared-memory array of at least 32 KiB and many repeated accesses."
        )
    if value < 2.0:
        stride = kv.get("stride") or kv.get("conflict_stride") or kv.get("stride_bytes")
        return (
            f"suspicious: reported conflict delta={value} cycles is below measurement noise. NVIDIA shared memory has 32 banks of 4 bytes. A stride of {stride} does not create bank collisions: 128-byte stride distributes threads across all 32 banks (conflict-free), and any stride that is a multiple of 128 B likewise spreads them. To force a 32-way conflict, 32 threads of one warp must address the SAME bank: use smem[tid*32] (i.e. each warp-lane reads smem[0], smem[32], smem[64], ... -- all in bank 0 when the array element is 4 bytes)."
        )
    free_med = kv.get("free_median") or kv.get("free_cycles") or kv.get("conflict_free_cycles")
    conf_med = kv.get("conf_median") or kv.get("conflict_cycles") or kv.get("conflicting_cycles")
    if free_med is not None and conf_med is not None and abs(free_med - conf_med) < 2.0:
        stride = kv.get("stride") or kv.get("conflict_stride")
        return (
            f"suspicious: conflict-free ({free_med}) and 'conflicting' ({conf_med}) kernels differ by <2 cyc; stride={stride} is NOT producing bank collisions. NVIDIA GPUs have 32 banks of 4 bytes each; to force a 32-way conflict, 32 threads of one warp must access addresses that all fall in the SAME bank. Use smem[tid*32] (4B elements): each lane hits smem[0], smem[32], smem[64], ..., all in bank 0."
        )
    return ""


def _rule_bandwidth_undersampled(target, spec, kv, value, line, hints):
    iterations = kv.get("iters_last") or kv.get("iters_per_thread") or kv.get("loads_per_thread")
    if iterations is None:
        return ""
    if iterations < 64:
        return (
            f"suspicious: each thread did only {iterations} loads/iterations; at that scale per-kernel launch overhead dominates the timing. Each thread must issue 64-256 vector loads from its own contiguous stripe."
        )
    if iterations > 4096:
        return (
            f"suspicious: loads_per_thread={iterations} is enormous. If the working set has fewer elements than (blocks*threads*loads_per_thread), you are revisiting cached elements and the bandwidth accounting is wrong. Either reduce loads_per_thread to 64-1024, or grow the working set so each element is touched exactly ONCE per timed kernel."
        )
    return ""


def _rule_bw_shmem_not_shmem(target, spec, kv, value, line, hints):
    if value is not None and value < 3000:
        working_set = kv.get("working_set_bytes") or 0
        if working_set > 200_000:
            return (
                f"suspicious: shared_mem BW={value} GB/s with working_set_bytes={working_set} is two orders of magnitude below hardware peak. A shared-memory probe must operate on block-local __shared__ arrays (<= 100 KiB per block); a large working_set means you are measuring GLOBAL memory."
            )
    return ""


def _rule_smcount_blocks(target, spec, kv, value, line, hints):
    if value is None:
        return ""
    for blocks_key in ("blocks", "max_blocks", "num_blocks", "grid"):
        if blocks_key in kv and abs(kv[blocks_key] - value) < 0.5:
            distinct = kv.get("distinct_smid_count")
            if distinct is None or abs(distinct - value) > 0.5:
                return f"suspicious: sm_count={value} appears to be the block count ({blocks_key}={kv[blocks_key]}), not distinct %smid values"
    if value > 128:
        return f"suspicious: measured sm_count={value} is larger than any known single-GPU SM count; probe likely reporting something else"
    return ""


def _rule_clock_vs_hint(target, spec, kv, value, line, hints):
    if value is None:
        return ""
    hint_rows = hints.get("nvidia_smi_unique") or []
    if not hint_rows:
        return ""
    try:
        max_sm_mhz = float(hint_rows[0].get("clocks.max.sm", "0"))
    except (TypeError, ValueError):
        return ""
    if max_sm_mhz > 0 and value > max_sm_mhz * 1.15:
        return f"suspicious: measured clock {value} MHz exceeds nvidia-smi max boost {max_sm_mhz} MHz by >15%; probe is mis-deriving MHz."
    return ""


UNIVERSAL_RULES = [_rule_degenerate_stats, _rule_bare_zero]

CATEGORY_RULES = {
    "latency_l1": [_rule_latency_stride],
    "latency_l2": [_rule_latency_stride],
    "latency_dram": [_rule_latency_stride, _rule_dram_latency_too_low],
    "capacity_cache": [_rule_capacity_sweep],
    "capacity_shmem": [],
    "bandwidth_shmem": [_rule_bw_shmem_not_shmem, _rule_bandwidth_undersampled],
    "bandwidth_global": [_rule_bandwidth_undersampled],
    "conflict_delta": [_rule_conflict_delta],
    "topology_smcount": [_rule_smcount_blocks],
    "clock": [_rule_clock_vs_hint],
}


def sanity_check(target: str, spec: ProbeSpec, stdout: str, value: float | None, hints: dict) -> str:
    line = extract_result_line(stdout)
    if not line:
        return ""
    kv = parse_kv_pairs(line)
    for rule in [*UNIVERSAL_RULES, *CATEGORY_RULES.get(spec.category, [])]:
        reason = rule(target, spec, kv, value, line, hints)
        if reason:
            return reason
    return ""
