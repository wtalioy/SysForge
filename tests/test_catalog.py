from sysforge.profiling.targets import CATALOG, strategy_for
from sysforge.profiling.validation import check_plausible_range


def test_catalog_has_section_1_7_targets():
    for name in [
        "l1_latency_cycles", "l2_latency_cycles", "dram_latency_cycles",
        "l2_cache_capacity_kb", "shared_mem_peak_bw_gbps",
        "global_mem_peak_bw_gbps", "actual_boost_clock_mhz",
        "bank_conflict_penalty_cycles", "max_shmem_per_block_kb", "sm_count",
    ]:
        assert name in CATALOG


def test_plausibility_checks():
    spec = CATALOG["dram_latency_cycles"]
    ok, _ = check_plausible_range(spec, 500)
    assert ok
    ok, reason = check_plausible_range(spec, 50)
    assert not ok and "outside plausible range" in reason


def test_strategy_fallback_on_unknown_target():
    spec = strategy_for("totally_made_up_metric_xyz")
    assert spec.strategy in ("llm_choose",)
