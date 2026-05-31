from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import torch

from .runtime_io import load_engine_module, load_json_file, resolve_device


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
    input_ids = [
        torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
        for _ in range(batch_size)
    ]

    return [
        {"op": "prefill", "request_ids": request_ids, "input_ids": input_ids},
        {"op": "remove", "request_ids": request_ids},
    ]


def _build_decode_events(batch_size, prompt_len, decode_steps, vocab_size, device):
    request_ids = list(range(batch_size))
    input_ids = [
        torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
        for _ in range(batch_size)
    ]

    events = [{"op": "prefill", "request_ids": request_ids, "input_ids": input_ids}]

    for _ in range(decode_steps):
        token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long, device=device)
        events.append({"op": "decode", "request_ids": request_ids, "token_ids": token_ids})

    events.append({"op": "remove", "request_ids": request_ids})
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
            input_ids = [
                torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
                for _ in request_ids
            ]
            events.append({"op": "prefill", "request_ids": request_ids, "input_ids": input_ids})

        elif op == "decode":
            request_ids = sorted(active)[:count]
            token_ids = torch.randint(0, vocab_size, (len(request_ids),), dtype=torch.long, device=device)
            events.append({"op": "decode", "request_ids": request_ids, "token_ids": token_ids})

        elif op == "remove":
            request_ids = sorted(active)[:count]
            for rid in request_ids:
                active.remove(rid)
            events.append({"op": "remove", "request_ids": request_ids})

    if active:
        events.append({"op": "remove", "request_ids": sorted(active)})

    return events


def _build_varied_prefill_events(vocab_size, device):
    request_ids = [101, 7, 42, 1009, 3, 88]
    lengths = [17, 64, 9, 96, 33, 128]
    input_ids = [
        torch.randint(0, vocab_size, (length,), dtype=torch.long, device=device)
        for length in lengths
    ]
    return [
        {"op": "prefill", "request_ids": request_ids, "input_ids": input_ids},
        {"op": "remove", "request_ids": request_ids},
    ]


def _build_long_decode_events(batch_size, prompt_len, decode_steps, vocab_size, device):
    request_ids = [1000 + i * 13 for i in range(batch_size)]
    input_ids = [
        torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
        for _ in request_ids
    ]
    events = [{"op": "prefill", "request_ids": request_ids, "input_ids": input_ids}]
    for step in range(decode_steps):
        step_request_ids = list(reversed(request_ids)) if step % 2 else list(request_ids)
        token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long, device=device)
        events.append({"op": "decode", "request_ids": step_request_ids, "token_ids": token_ids})
    events.append({"op": "remove", "request_ids": request_ids})
    return events


def _build_churn_events(vocab_size, device):
    events = []
    active = []
    next_request_id = 500
    for round_index, prompt_len in enumerate([24, 80, 12, 144, 40]):
        request_ids = [next_request_id + offset * 17 for offset in range(3)]
        next_request_id += 100
        active.extend(request_ids)
        input_ids = [
            torch.randint(0, vocab_size, (prompt_len + offset * 3,), dtype=torch.long, device=device)
            for offset in range(3)
        ]
        events.append({"op": "prefill", "request_ids": request_ids, "input_ids": input_ids})
        decode_ids = list(reversed(active[-min(len(active), 6):]))
        token_ids = torch.randint(0, vocab_size, (len(decode_ids),), dtype=torch.long, device=device)
        events.append({"op": "decode", "request_ids": decode_ids, "token_ids": token_ids})
        if round_index % 2 == 1 and active:
            remove_ids = active[:2]
            active = active[2:]
            events.append({"op": "remove", "request_ids": remove_ids})
    if active:
        events.append({"op": "remove", "request_ids": active})
    return events


def _spread_pct(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    median_value = float(statistics.median(values))
    if median_value == 0.0:
        return 0.0
    return (max(values) - min(values)) / median_value * 100.0


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
            "elapsed_ms": _spread_pct(elapsed_samples),
            "tokens_per_second": _spread_pct(token_samples),
            "decode_tokens_per_second": _spread_pct(decode_samples),
        },
    )


def _build_benchmark_cases(case_mode: str, vocab_size: int, device: str):
    cases = {
        "prefill": _build_prefill_events(batch_size=4, prompt_len=128, vocab_size=vocab_size, device=device),
        "decode": _build_decode_events(batch_size=8, prompt_len=32, decode_steps=16, vocab_size=vocab_size, device=device),
        "mixed": _build_mixed_events(vocab_size=vocab_size, device=device),
    }
    if case_mode == "robust":
        cases.update(
            {
                "varied_prefill": _build_varied_prefill_events(vocab_size=vocab_size, device=device),
                "long_decode": _build_long_decode_events(
                    batch_size=6,
                    prompt_len=48,
                    decode_steps=32,
                    vocab_size=vocab_size,
                    device=device,
                ),
                "churn": _build_churn_events(vocab_size=vocab_size, device=device),
            }
        )
    return cases


def run_benchmark(
    *,
    engine_path: str,
    model_config_path: str,
    weight_dir: str,
    device: str = "auto",
    warmup: int = 1,
    repeat: int = 3,
    case_mode: str = "public",
) -> list[dict[str, object]]:
    device = resolve_device(device)
    torch.manual_seed(0)

    model_config = load_json_file(model_config_path)

    vocab_size = int(model_config["vocab_size"])
    engine_module = load_engine_module(engine_path)
    cases = _build_benchmark_cases(case_mode, vocab_size, device)
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
