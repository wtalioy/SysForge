from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ...integrations import ncu
from ...integrations.workspace import Workspace
from ..profiling.analysis import aggregate_per_metric
from .harness import HarnessConfig


PROFILE_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard.avg",
]


@dataclass
class ProfileSummary:
    shape_d: int
    kernel_count: int
    top_kernels: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, dict[str, float | int | str | None]] = field(default_factory=dict)
    bottleneck_hints: list[str] = field(default_factory=list)
    error: str = ""


def _metric_value(per_metric: dict[str, dict[str, object]], name: str) -> float | None:
    entry = per_metric.get(name, {})
    value = entry.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_profile(rows: list[dict[str, str]], *, shape_d: int) -> ProfileSummary:
    per_metric = aggregate_per_metric(rows, PROFILE_METRICS)
    kernel_counts: dict[str, int] = {}
    for row in rows:
        kernel = (
            row.get("Kernel Name")
            or row.get("Kernel_Name")
            or row.get("Kernel")
            or row.get("Kernel_Name_Base")
            or ""
        ).strip()
        if kernel:
            kernel_counts[kernel] = kernel_counts.get(kernel, 0) + 1

    hints: list[str] = []
    dram = _metric_value(per_metric, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    sm = _metric_value(per_metric, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    warps = _metric_value(per_metric, "sm__warps_active.avg.pct_of_peak_sustained_active")
    long_scoreboard = _metric_value(per_metric, "smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg")
    if dram is not None and dram >= 70.0:
        hints.append("memory_bound")
    if warps is not None and warps < 40.0:
        hints.append("low_occupancy")
    if long_scoreboard is not None and long_scoreboard >= 15.0:
        hints.append("long_scoreboard_stalls")
    if sm is not None and dram is not None and sm >= 75.0 and dram < 50.0:
        hints.append("compute_heavy")

    return ProfileSummary(
        shape_d=shape_d,
        kernel_count=len(kernel_counts),
        top_kernels=[
            {"name": name, "samples": samples}
            for name, samples in sorted(kernel_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        ],
        metrics=per_metric,
        bottleneck_hints=hints,
    )


class CandidateProfiler:
    def __init__(self, workspace: Workspace, harness_config: HarnessConfig) -> None:
        self.workspace = workspace
        self.harness_config = harness_config

    def _script_path(self, candidate) -> Path:
        return self.workspace.probes_dir / f"profile_{candidate.source_hash}.py"

    def _write_driver_script(self, *, candidate, compile_result, shape_d: int) -> Path:
        script_path = self._script_path(candidate)
        script = f"""from __future__ import annotations

import importlib
import json
import sys

import torch

sys.path.insert(0, {json.dumps(compile_result.build_dir)})
module = importlib.import_module({json.dumps(candidate.module_name)})

generator = torch.Generator(device="cuda")
generator.manual_seed({self.harness_config.seed})
W = torch.randn(({shape_d}, {shape_d}), device="cuda", dtype=torch.float32, generator=generator).contiguous()
X = torch.randn(({shape_d}, {shape_d}), device="cuda", dtype=torch.float32, generator=generator).contiguous()
A = torch.randn(({shape_d}, {self.harness_config.rank}), device="cuda", dtype=torch.float32, generator=generator).contiguous()
B = torch.randn(({shape_d}, {self.harness_config.rank}), device="cuda", dtype=torch.float32, generator=generator).contiguous()

_ = module.forward(W, X, A, B)
torch.cuda.synchronize()
_ = module.forward(W, X, A, B)
torch.cuda.synchronize()
"""
        script_path.write_text(script, encoding="utf-8")
        return script_path

    def profile_candidate(self, *, candidate, compile_result, shape_d: int) -> ProfileSummary:
        script_path = self._write_driver_script(
            candidate=candidate,
            compile_result=compile_result,
            shape_d=shape_d,
        )
        result = ncu.profile_command(
            ["python3", str(script_path)],
            PROFILE_METRICS,
        )
        if not result.ok:
            return ProfileSummary(shape_d=shape_d, kernel_count=0, error=result.stderr or result.stdout)
        return summarize_profile(result.rows, shape_d=shape_d)
