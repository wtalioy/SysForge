from __future__ import annotations

from typing import Any

from .benchmark import run_benchmark
from .correctness import run_correctness_case
from .runtime_io import write_json_file


def run_public_checks(
    *,
    engine_path: str = "engine.py",
    model_config_path: str = "target/model_config.json",
    weight_dir: str = "target/weights",
    device: str = "auto",
    case_mode: str = "public",
    warmup: int = 1,
    repeat: int = 3,
    benchmark_output_path: str | None = None,
) -> dict[str, Any]:
    correctness = run_correctness_case(
        engine_path=engine_path,
        model_config_path=model_config_path,
        weight_dir=weight_dir,
        case="basic",
        device=device,
    )
    benchmark = run_benchmark(
        engine_path=engine_path,
        model_config_path=model_config_path,
        weight_dir=weight_dir,
        device=device,
        case_mode=case_mode,
        warmup=warmup,
        repeat=repeat,
    )
    if benchmark_output_path:
        write_json_file(benchmark_output_path, benchmark)
    return {"correctness": correctness, "benchmark": benchmark}
