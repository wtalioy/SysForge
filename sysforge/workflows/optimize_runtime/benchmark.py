from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..common import median_float, spread_pct
from .events import decode_event, prefill_event, random_token_ids, remove_event
from .models import RuntimeBenchmarkSummary
from .runtime_io import load_engine_module, read_json_file, resolve_device


PUBLIC_CASE_NAMES = ("prefill", "decode", "mixed")
ROBUST_CASE_NAMES = ("varied_prefill", "long_decode", "churn")
BENCHMARK_CASE_NAMES = (*PUBLIC_CASE_NAMES, *ROBUST_CASE_NAMES)


@dataclass
class BenchmarkCaseResult:
    case_name: str
    elapsed_ms: float
    prefill_tokens: int
    decode_tokens: int
    total_tokens: int
    tokens_per_second: float
    decode_tokens_per_second: float
    peak_memory_mb: float
    elapsed_ms_samples: list[float]
    tokens_per_second_samples: list[float]
    decode_tokens_per_second_samples: list[float]
    spread_pct: dict[str, float]


def _sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _run_timed_events(engine, events: list[dict], device: str):
    prefill_tokens = 0
    decode_tokens = 0

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    _sync_device(device)
    start = time.perf_counter()

    with torch.no_grad():
        for event in events:
            op = event["op"]

            if op == "prefill":
                request_ids = event["request_ids"]
                input_ids = event["input_ids"]
                engine.prefill(request_ids, input_ids)
                prefill_tokens += sum(int(x.numel()) for x in input_ids)

            elif op == "decode":
                request_ids = event["request_ids"]
                token_ids = event["token_ids"]
                engine.decode(request_ids, token_ids)
                decode_tokens += int(token_ids.numel())

            elif op == "remove":
                engine.remove(event["request_ids"])

            else:
                raise ValueError(f"unknown op: {op}")

    _sync_device(device)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000.0
    total_tokens = prefill_tokens + decode_tokens

    peak_memory_mb = 0.0
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return elapsed_ms, prefill_tokens, decode_tokens, total_tokens, peak_memory_mb


def _build_prefill_events(batch_size, prompt_len, vocab_size, device):
    request_ids = list(range(batch_size))
    input_ids = [random_token_ids(vocab_size, prompt_len, device) for _ in range(batch_size)]
    return [prefill_event(request_ids, input_ids), remove_event(request_ids)]


def _build_decode_events(batch_size, prompt_len, decode_steps, vocab_size, device):
    request_ids = list(range(batch_size))
    input_ids = [random_token_ids(vocab_size, prompt_len, device) for _ in range(batch_size)]

    events = [prefill_event(request_ids, input_ids)]

    for _ in range(decode_steps):
        events.append(decode_event(request_ids, random_token_ids(vocab_size, batch_size, device)))

    events.append(remove_event(request_ids))
    return events


def _build_mixed_events(vocab_size, device):
    events = []
    active = set()
    next_request_id = 0

    schedule = [
        ("prefill", 4, 64),
        ("decode", 4, None),
        ("decode", 4, None),
        ("prefill", 2, 128),
        ("decode", 6, None),
        ("remove", 2, None),
        ("prefill", 4, 32),
        ("decode", 8, None),
        ("decode", 8, None),
        ("remove", 8, None),
    ]

    for op, count, prompt_len in schedule:
        if op == "prefill":
            request_ids = list(range(next_request_id, next_request_id + count))
            next_request_id += count
            active.update(request_ids)
            input_ids = [random_token_ids(vocab_size, prompt_len, device) for _ in request_ids]
            events.append(prefill_event(request_ids, input_ids))

        elif op == "decode":
            request_ids = sorted(active)[:count]
            events.append(decode_event(request_ids, random_token_ids(vocab_size, len(request_ids), device)))

        elif op == "remove":
            request_ids = sorted(active)[:count]
            for rid in request_ids:
                active.remove(rid)
            events.append(remove_event(request_ids))

    if active:
        events.append(remove_event(sorted(active)))

    return events


def _build_varied_prefill_events(vocab_size, device):
    request_ids = [101, 7, 42, 1009, 3, 88]
    lengths = [17, 64, 9, 96, 33, 128]
    input_ids = [random_token_ids(vocab_size, length, device) for length in lengths]
    return [prefill_event(request_ids, input_ids), remove_event(request_ids)]


def _build_long_decode_events(batch_size, prompt_len, decode_steps, vocab_size, device):
    request_ids = [1000 + i * 13 for i in range(batch_size)]
    input_ids = [random_token_ids(vocab_size, prompt_len, device) for _ in request_ids]
    events = [prefill_event(request_ids, input_ids)]
    for step in range(decode_steps):
        step_request_ids = list(reversed(request_ids)) if step % 2 else list(request_ids)
        events.append(decode_event(step_request_ids, random_token_ids(vocab_size, batch_size, device)))
    events.append(remove_event(request_ids))
    return events


def _build_churn_events(vocab_size, device):
    events = []
    active = []
    next_request_id = 500
    for round_index, prompt_len in enumerate([24, 80, 12, 144, 40]):
        request_ids = [next_request_id + offset * 17 for offset in range(3)]
        next_request_id += 100
        active.extend(request_ids)
        input_ids = [random_token_ids(vocab_size, prompt_len + offset * 3, device) for offset in range(3)]
        events.append(prefill_event(request_ids, input_ids))
        decode_ids = list(reversed(active[-min(len(active), 6):]))
        events.append(decode_event(decode_ids, random_token_ids(vocab_size, len(decode_ids), device)))
        if round_index % 2 == 1 and active:
            remove_ids = active[:2]
            active = active[2:]
            events.append(remove_event(remove_ids))
    if active:
        events.append(remove_event(active))
    return events


def _run_benchmark_case(case_name, engine_module, model_config, weight_dir, events, device, warmup, repeat):
    engine = engine_module.create_engine(model_config, weight_dir, device)
    for _ in range(warmup):
        _run_timed_events(engine, events, device)

    measurements = [_run_timed_events(engine, events, device) for _ in range(repeat)]
    measurements.sort(key=lambda x: x[0])
    elapsed_ms, prefill_tokens, decode_tokens, total_tokens, peak_memory_mb = measurements[len(measurements) // 2]
    elapsed_s = elapsed_ms / 1000.0
    elapsed_samples = [value[0] for value in measurements]
    token_samples = [value[3] / (value[0] / 1000.0) for value in measurements]
    decode_samples = [value[2] / (value[0] / 1000.0) if value[2] else 0.0 for value in measurements]

    return BenchmarkCaseResult(
        case_name=case_name,
        elapsed_ms=elapsed_ms,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        total_tokens=total_tokens,
        tokens_per_second=total_tokens / elapsed_s,
        decode_tokens_per_second=decode_tokens / elapsed_s if decode_tokens else 0.0,
        peak_memory_mb=peak_memory_mb,
        elapsed_ms_samples=elapsed_samples,
        tokens_per_second_samples=token_samples,
        decode_tokens_per_second_samples=decode_samples,
        spread_pct={
            "elapsed_ms": spread_pct(elapsed_samples),
            "tokens_per_second": spread_pct(token_samples),
            "decode_tokens_per_second": spread_pct(decode_samples),
        },
    )


def _build_benchmark_cases(vocab_size: int, device: str):
    builders = {
        "prefill": lambda: _build_prefill_events(batch_size=4, prompt_len=128, vocab_size=vocab_size, device=device),
        "decode": lambda: _build_decode_events(
            batch_size=8,
            prompt_len=32,
            decode_steps=16,
            vocab_size=vocab_size,
            device=device,
        ),
        "mixed": lambda: _build_mixed_events(vocab_size=vocab_size, device=device),
        "varied_prefill": lambda: _build_varied_prefill_events(vocab_size=vocab_size, device=device),
        "long_decode": lambda: _build_long_decode_events(
            batch_size=6,
            prompt_len=48,
            decode_steps=32,
            vocab_size=vocab_size,
            device=device,
        ),
        "churn": lambda: _build_churn_events(vocab_size=vocab_size, device=device),
    }
    return {case_name: builders[case_name]() for case_name in BENCHMARK_CASE_NAMES}


def run_benchmark(
    *,
    engine_path: str,
    model_config_path: str,
    weight_dir: str,
    device: str = "auto",
    warmup: int = 1,
    repeat: int = 3,
) -> list[dict[str, object]]:
    device = resolve_device(device)
    torch.manual_seed(0)

    model_config = read_json_file(model_config_path)

    vocab_size = int(model_config["vocab_size"])
    engine_module = load_engine_module(engine_path)
    cases = _build_benchmark_cases(vocab_size, device)
    return [
        asdict(
            _run_benchmark_case(
                case_name=case_name,
                engine_module=engine_module,
                model_config=model_config,
                weight_dir=weight_dir,
                events=events,
                device=device,
                warmup=warmup,
                repeat=repeat,
            )
        )
        for case_name, events in cases.items()
    ]


def parse_benchmark_file(path: Path, *, required_cases: set[str] | None = None) -> RuntimeBenchmarkSummary:
    try:
        payload = read_json_file(path)
    except ValueError as exc:
        raise ValueError(f"benchmark JSON file parse failed: {exc}") from exc
    return parse_benchmark_payload(payload, required_cases=required_cases)


def parse_benchmark_payload(payload: Any, *, required_cases: set[str] | None = None) -> RuntimeBenchmarkSummary:
    if not isinstance(payload, list):
        raise ValueError("benchmark JSON payload was not a list")
    by_case: dict[str, dict[str, Any]] = {str(item.get("case_name")): item for item in payload}
    if len(by_case) != len(payload):
        raise ValueError("benchmark JSON payload contains duplicate case names")
    required_cases = required_cases or set(BENCHMARK_CASE_NAMES)
    if not required_cases.issubset(set(by_case)):
        missing = ", ".join(sorted(required_cases - set(by_case)))
        extra = ", ".join(sorted(set(by_case) - required_cases))
        raise ValueError(f"benchmark cases mismatch; missing={missing or 'none'} extra={extra or 'none'}")

    def metric(case_name: str, key: str) -> float:
        value = by_case.get(case_name, {}).get(key, 0.0)
        metric_value = float(value or 0.0)
        if not (metric_value >= 0.0 and metric_value < float("inf")):
            raise ValueError(f"benchmark metric {case_name}.{key} is not finite")
        return metric_value

    total_tps_values = [float(item.get("tokens_per_second") or 0.0) for item in payload]
    peak_values = [float(item.get("peak_memory_mb") or 0.0) for item in payload]
    case_tps = {case_name: metric(case_name, "tokens_per_second") for case_name in by_case}
    case_decode_tps = {case_name: metric(case_name, "decode_tokens_per_second") for case_name in by_case}
    return RuntimeBenchmarkSummary(
        prefill_tokens_per_second=metric("prefill", "tokens_per_second"),
        decode_tokens_per_second=metric("decode", "decode_tokens_per_second"),
        mixed_tokens_per_second=metric("mixed", "tokens_per_second"),
        total_tokens_per_second=sum(total_tps_values),
        peak_memory_mb=max(peak_values) if peak_values else 0.0,
        raw_results=payload,
        benchmark_runs=[payload],
        run_count=1,
        case_tokens_per_second=case_tps,
        case_decode_tokens_per_second=case_decode_tps,
    )


def aggregate_benchmark_summaries(summaries: list[RuntimeBenchmarkSummary]) -> RuntimeBenchmarkSummary:
    if not summaries:
        return RuntimeBenchmarkSummary(run_count=0)

    prefill = [summary.prefill_tokens_per_second for summary in summaries]
    decode = [summary.decode_tokens_per_second for summary in summaries]
    mixed = [summary.mixed_tokens_per_second for summary in summaries]
    total = [summary.total_tokens_per_second for summary in summaries]
    peak = [summary.peak_memory_mb for summary in summaries]
    raw_runs = [summary.raw_results for summary in summaries]
    all_case_names = sorted({case_name for summary in summaries for case_name in summary.case_tokens_per_second})
    case_tps = {
        case_name: median_float([summary.case_tokens_per_second.get(case_name, 0.0) for summary in summaries])
        for case_name in all_case_names
    }
    case_decode_tps = {
        case_name: median_float([summary.case_decode_tokens_per_second.get(case_name, 0.0) for summary in summaries])
        for case_name in all_case_names
    }
    median_mixed = median_float(mixed)
    selected_index = min(
        range(len(summaries)),
        key=lambda index: abs(summaries[index].mixed_tokens_per_second - median_mixed),
    )
    return RuntimeBenchmarkSummary(
        prefill_tokens_per_second=median_float(prefill),
        decode_tokens_per_second=median_float(decode),
        mixed_tokens_per_second=median_float(mixed),
        total_tokens_per_second=median_float(total),
        peak_memory_mb=max(peak) if peak else 0.0,
        raw_results=summaries[selected_index].raw_results,
        benchmark_runs=raw_runs,
        run_count=len(summaries),
        spread_pct={
            "prefill": spread_pct(prefill),
            "decode": spread_pct(decode),
            "mixed": spread_pct(mixed),
            "total": spread_pct(total),
        },
        case_tokens_per_second=case_tps,
        case_decode_tokens_per_second=case_decode_tps,
    )
