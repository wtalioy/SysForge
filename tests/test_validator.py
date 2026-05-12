from sysforge.workflows.profiling.targets import CATALOG
from sysforge.workflows.profiling.validation import (
    check_plausible_range,
    extract_result_line,
    extract_unit_from_result,
    extract_value_from_result,
    parse_kv_pairs,
    sanity_check,
)


def test_result_parsers():
    so = "noise\nRESULT sm_count=82 unit=count samples=32 blocks=4096\nmore noise\n"
    assert extract_result_line(so).startswith("RESULT sm_count=")
    assert extract_value_from_result("sm_count", so) == 82.0
    assert extract_unit_from_result("sm_count", so) == "count"
    kv = parse_kv_pairs(extract_result_line(so))
    assert kv["sm_count"] == 82 and kv["blocks"] == 4096


def test_plausible_range():
    spec = CATALOG["dram_latency_cycles"]
    assert check_plausible_range(spec, 500) == (True, "")
    ok, reason = check_plausible_range(spec, 50)
    assert not ok and "outside plausible range" in reason


def test_sanity_rule_latency_stride_guard():
    spec = CATALOG["dram_latency_cycles"]
    ok_line = "RESULT dram_latency_cycles=500 unit=cycles stride_bytes=128"
    assert sanity_check("dram_latency_cycles", spec, ok_line, 500.0, {}) == ""
    bad = "RESULT dram_latency_cycles=500 unit=cycles stride_bytes=4"
    assert "stride_bytes" in sanity_check("dram_latency_cycles", spec, bad, 500.0, {})


def test_sanity_rule_dram_latency_too_low():
    spec = CATALOG["dram_latency_cycles"]
    line = "RESULT dram_latency_cycles=41 unit=cycles stride_bytes=4096"
    # first rule fires (stride) because it's checked before the <300 rule
    r = sanity_check("dram_latency_cycles", spec, line, 41.0, {})
    assert r  # some rejection
    # now with legal stride, the <300 rule fires
    line2 = "RESULT dram_latency_cycles=41 unit=cycles stride_bytes=128"
    r2 = sanity_check("dram_latency_cycles", spec, line2, 41.0, {})
    assert "below L2-hit latency" in r2


def test_sanity_rule_capacity_all_identical_sweep():
    spec = CATALOG["l2_cache_capacity_kb"]
    line = ("RESULT l2_cache_capacity_kb=262144 unit=KB "
            "sweep_kb=64:0.00,128:0.00,256:0.00,131072:0.00")
    r = sanity_check("l2_cache_capacity_kb", spec, line, 262144.0, {})
    assert r  # caught
    assert ("identical latency" in r) or ("sweep_max" in r)


def test_sanity_rule_conflict_delta_zero():
    spec = CATALOG["bank_conflict_penalty_cycles"]
    line = ("RESULT bank_conflict_penalty_cycles=-0.0 unit=cycles "
            "free_median=4.51 conf_median=4.51 stride=2 smem_kb=32")
    r = sanity_check("bank_conflict_penalty_cycles", spec, line, 0.0, {})
    assert "identical" in r or "stride" in r


def test_sanity_rule_shmem_bw_too_low_and_global():
    spec = CATALOG["shared_mem_peak_bw_gbps"]
    line = ("RESULT shared_mem_peak_bw_gbps=644 unit=GB/s "
            "working_set_bytes=21495808 blocks=328 threads_per_block=256")
    r = sanity_check("shared_mem_peak_bw_gbps", spec, line, 644.0, {})
    assert "GLOBAL" in r


def test_sanity_rule_smcount_reports_blocks():
    spec = CATALOG["sm_count"]
    line = "RESULT sm_count=256 unit=count max_blocks=256"
    r = sanity_check("sm_count", spec, line, 256.0, {})
    assert r  # caught (either blocks-match or >128 rule)


def test_sanity_rule_clock_exceeds_hint():
    spec = CATALOG["actual_boost_clock_mhz"]
    line = "RESULT actual_boost_clock_mhz=4000 unit=MHz"
    hints = {"nvidia_smi_unique": [{"clocks.max.sm": "2100"}]}
    r = sanity_check("actual_boost_clock_mhz", spec, line, 4000.0, hints)
    assert "exceeds" in r


def test_sanity_passes_good_result():
    spec = CATALOG["dram_latency_cycles"]
    line = ("RESULT dram_latency_cycles=534 unit=cycles stride_bytes=128 "
            "chain_len=2000000 iters=8192 median=534")
    assert sanity_check("dram_latency_cycles", spec, line, 534.0, {}) == ""


def test_sanity_rule_capacity_nonmonotonic_sweep():
    spec = CATALOG["l2_cache_capacity_kb"]
    # This is the exact antipattern seen in group2: cpl drops past the reported
    # capacity instead of staying at DRAM level.
    line = ("RESULT l2_cache_capacity_kb=32768 unit=KB sweep_kb=256,512,1024,2048,4096,8192 "
            "sweep_med_cpl=253.22,253.28,211.32,137.67,77.15,147.80")
    r = sanity_check("l2_cache_capacity_kb", spec, line, 32768.0, {})
    assert "non-monotonic" in r, r


def test_sanity_rule_conflict_too_small():
    spec = CATALOG["bank_conflict_penalty_cycles"]
    # stride=128 distributes across 32 banks -> no conflicts, delta ~0.75 cyc.
    line = ("RESULT bank_conflict_penalty_cycles=0.75 unit=cycles stride_bytes=128 "
            "smem_kb=32 chain_len=8192 free_median=0.63 conflict_median=1.38")
    r = sanity_check("bank_conflict_penalty_cycles", spec, line, 0.75, {})
    assert "noise" in r or "below" in r, r


def test_sanity_rule_bandwidth_undersampled():
    spec = CATALOG["global_mem_peak_bw_gbps"]
    line = ("RESULT global_mem_peak_bw_gbps=400 unit=GB/s "
            "working_set_bytes=1073741824 iters_last=4")
    r = sanity_check("global_mem_peak_bw_gbps", spec, line, 400.0, {})
    assert "loads/iterations" in r, r
    # Too-many branch: loads_per_thread > 4096 → revisit accounting broken
    line_big = ("RESULT global_mem_peak_bw_gbps=1.01 unit=GB/s "
                "working_set_bytes=1073741824 loads_per_thread=51200")
    r_big = sanity_check("global_mem_peak_bw_gbps", spec, line_big, 1.01, {})
    assert "enormous" in r_big or "revisit" in r_big, r_big
    # Same undersampled check for shared-memory BW
    spec2 = CATALOG["shared_mem_peak_bw_gbps"]
    line2 = ("RESULT shared_mem_peak_bw_gbps=9000 unit=GB/s loads_per_thread=32")
    r2 = sanity_check("shared_mem_peak_bw_gbps", spec2, line2, 9000.0, {})
    assert "loads/iterations" in r2, r2
