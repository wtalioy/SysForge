from sysforge.profiling.analysis import aggregate_per_metric


# A slice of the CSV the server actually returned, with 2 invocations per metric.
_ROWS = [
    {"Metric Name": "device__attribute_fb_bus_width", "Metric Value": "384",
     "Metric Unit": ""},
    {"Metric Name": "device__attribute_max_gpu_frequency_khz", "Metric Value": "1695000",
     "Metric Unit": ""},
    {"Metric Name": "device__attribute_max_mem_frequency_khz", "Metric Value": "9751000",
     "Metric Unit": ""},
    {"Metric Name": "dram__bytes_read.sum.per_second", "Metric Value": "111028419924.22",
     "Metric Unit": "byte/second"},
    {"Metric Name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
     "Metric Value": "82.71", "Metric Unit": "%"},
    {"Metric Name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
     "Metric Value": "82.68", "Metric Unit": "%"},
    {"Metric Name": "dram__bytes_read.sum.per_second", "Metric Value": "111080690038.95",
     "Metric Unit": "byte/second"},
]


def test_aggregate_median_across_invocations():
    out = aggregate_per_metric(_ROWS, [
        "device__attribute_fb_bus_width",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum.per_second",
        "metric_not_returned",
    ])
    assert out["device__attribute_fb_bus_width"]["value"] == 384
    # median of [82.71, 82.68] = 82.695
    assert abs(out["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["value"] - 82.695) < 1e-6
    assert out["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["unit"] == "%"
    assert out["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["samples"] == 2
    # requested but not emitted
    assert out["metric_not_returned"]["value"] is None
    assert out["metric_not_returned"].get("error", "").startswith("metric not present")


def test_aggregate_preserves_request_order():
    requested = [
        "dram__bytes_read.sum.per_second",
        "device__attribute_fb_bus_width",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    out = aggregate_per_metric(_ROWS, requested)
    assert list(out.keys())[:3] == requested
