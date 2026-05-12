from sysforge.workflows.optimize_lora.profiling import summarize_profile


def test_summarize_profile_extracts_kernel_count_and_hints():
    rows = [
        {"Kernel Name": "kernel_a", "Metric Name": "dram__throughput.avg.pct_of_peak_sustained_elapsed", "Metric Value": "82", "Metric Unit": "%"},
        {"Kernel Name": "kernel_a", "Metric Name": "sm__warps_active.avg.pct_of_peak_sustained_active", "Metric Value": "35", "Metric Unit": "%"},
        {"Kernel Name": "kernel_b", "Metric Name": "smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg", "Metric Value": "20", "Metric Unit": "%"},
    ]
    summary = summarize_profile(rows, shape_d=4608)
    assert summary.shape_d == 4608
    assert summary.kernel_count == 2
    assert summary.top_kernels == [{"name": "kernel_a", "samples": 2}, {"name": "kernel_b", "samples": 1}]
    assert "memory_bound" in summary.bottleneck_hints
    assert "low_occupancy" in summary.bottleneck_hints
    assert "long_scoreboard_stalls" in summary.bottleneck_hints
