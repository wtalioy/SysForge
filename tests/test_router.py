from sysforge.workflows.profiling.targets import looks_like_ncu_metric, partition_targets


def test_ncu_metric_names_classified():
    assert looks_like_ncu_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert looks_like_ncu_metric("dram__bytes_read.sum.per_second")
    assert looks_like_ncu_metric("launch__sm_count")
    assert looks_like_ncu_metric("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")


def test_device_attribute_names_classified_as_ncu():
    """ncu device attributes (device__attribute_*) are also ncu namespace."""
    assert looks_like_ncu_metric("device__attribute_max_gpu_frequency_khz")
    assert looks_like_ncu_metric("device__attribute_max_mem_frequency_khz")
    assert looks_like_ncu_metric("device__attribute_fb_bus_width")


def test_unknown_ncu_namespaces_still_route_to_analysis():
    """Router is catalog-first, so any unseen `xxx__yyy` name is treated as ncu."""
    assert looks_like_ncu_metric("crop__bytes.sum")
    assert looks_like_ncu_metric("zrop__read_requests.sum.per_second")
    assert looks_like_ncu_metric("some_future_namespace__whatever")


def test_catalog_entries_are_never_ncu_even_when_underscored():
    """A catalogued probe name wins even if it might look ncu-ish."""
    # none of our current catalog entries contain "__", but the rule is defensive
    assert not looks_like_ncu_metric("dram_latency_cycles")
    assert not looks_like_ncu_metric("sm_count")


def test_unknown_non_underscored_names_route_to_probe():
    """Unknown hardware intrinsics (fallback LLM-choose) stay on the probe path."""
    assert not looks_like_ncu_metric("register_file_size_per_sm_kb")
    assert not looks_like_ncu_metric("warp_scheduler_issue_rate_per_cycle")


def test_hardware_targets_not_classified_as_ncu():
    assert not looks_like_ncu_metric("dram_latency_cycles")
    assert not looks_like_ncu_metric("actual_boost_clock_mhz")
    assert not looks_like_ncu_metric("l2_cache_capacity_kb")
    assert not looks_like_ncu_metric("max_shmem_per_block_kb")


def test_partition_sample_spec():
    targets = [
        "launch__sm_count",
        "dram__bytes_read.sum.per_second",
        "dram__bytes_write.sum.per_second",
    ]
    probes, metrics = partition_targets(targets)
    assert probes == []
    assert metrics == targets


def test_partition_mixed():
    probes, metrics = partition_targets([
        "dram_latency_cycles", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "actual_boost_clock_mhz",
    ])
    assert probes == ["dram_latency_cycles", "actual_boost_clock_mhz"]
    assert metrics == ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
