from __future__ import annotations

import torch

from .reference_model import ReferenceModel
from .runtime_io import load_engine_module, load_json_file, resolve_device


def assert_logits_close(name, student_logits, reference_logits, atol: float, rtol: float) -> None:
    student = student_logits.detach().float().cpu()
    reference = reference_logits.detach().float().cpu()
    if student.shape != reference.shape:
        raise AssertionError(
            f"{name}: shape mismatch, student={tuple(student.shape)}, reference={tuple(reference.shape)}"
        )
    if not torch.allclose(student, reference, atol=atol, rtol=rtol):
        diff = (student - reference).abs()
        ref_scale = reference.abs().clamp_min(1e-12)
        raise AssertionError(
            f"{name}: logits mismatch, max_abs={float(diff.max()):.6g}, "
            f"max_rel={float((diff / ref_scale).max()):.6g}"
        )


def select_last_logits(ref_model: ReferenceModel, input_ids: torch.Tensor) -> torch.Tensor:
    return ref_model.forward(input_ids.unsqueeze(0))[0, -1, :]


def build_basic_events(vocab_size: int, device: str) -> list[dict]:
    torch.manual_seed(7)
    return [
        {
            "op": "prefill",
            "name": "single_prefill",
            "request_ids": [0],
            "input_ids": [torch.randint(0, vocab_size, (11,), device=device)],
        },
        {
            "op": "decode",
            "name": "single_decode",
            "request_ids": [0],
            "token_ids": torch.randint(0, vocab_size, (1,), device=device),
        },
        {
            "op": "prefill",
            "name": "multi_prefill",
            "request_ids": [1, 2],
            "input_ids": [
                torch.randint(0, vocab_size, (7,), device=device),
                torch.randint(0, vocab_size, (13,), device=device),
            ],
        },
        {
            "op": "decode",
            "name": "multi_decode",
            "request_ids": [0, 1, 2],
            "token_ids": torch.randint(0, vocab_size, (3,), device=device),
        },
        {"op": "remove", "request_ids": [1]},
        {
            "op": "prefill",
            "name": "insert_after_remove",
            "request_ids": [3],
            "input_ids": [torch.randint(0, vocab_size, (5,), device=device)],
        },
        {
            "op": "decode",
            "name": "decode_after_remove",
            "request_ids": [0, 2, 3],
            "token_ids": torch.randint(0, vocab_size, (3,), device=device),
        },
        {"op": "remove", "request_ids": [0, 2, 3]},
    ]


def build_stress_events(vocab_size: int, device: str) -> list[dict]:
    torch.manual_seed(17)
    return [
        {
            "op": "prefill",
            "name": "non_contiguous_prefill",
            "request_ids": [101, 7, 42],
            "input_ids": [
                torch.randint(0, vocab_size, (5,), device=device),
                torch.randint(0, vocab_size, (9,), device=device),
                torch.randint(0, vocab_size, (3,), device=device),
            ],
        },
        {
            "op": "prefill",
            "name": "same_length_prefill_bucket",
            "request_ids": [55, 56],
            "input_ids": [
                torch.randint(0, vocab_size, (6,), device=device),
                torch.randint(0, vocab_size, (6,), device=device),
            ],
        },
        {
            "op": "decode",
            "name": "reordered_decode",
            "request_ids": [56, 42, 101, 7, 55],
            "token_ids": torch.randint(0, vocab_size, (5,), device=device),
        },
        {
            "op": "decode",
            "name": "multi_step_decode",
            "request_ids": [7, 42],
            "token_ids": torch.randint(0, vocab_size, (2,), device=device),
        },
        {
            "op": "prefill",
            "name": "replacement_prefill",
            "request_ids": [42],
            "input_ids": [torch.randint(0, vocab_size, (4,), device=device)],
        },
        {"op": "remove", "request_ids": [7, 999]},
        {
            "op": "decode",
            "name": "decode_after_remove",
            "request_ids": [42, 101],
            "token_ids": torch.randint(0, vocab_size, (2,), device=device),
        },
    ]


def compare_engine_with_reference(
    *,
    engine_path: str,
    model_config: dict,
    weight_dir: str,
    events: list[dict],
    device: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    engine = load_engine_module(engine_path).create_engine(model_config, weight_dir, device)
    ref_model = ReferenceModel(model_config, weight_dir, device)
    request_tokens: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for event in events:
            if event["op"] == "prefill":
                request_ids = event["request_ids"]
                input_ids = [ids.to(device=device, dtype=torch.long) for ids in event["input_ids"]]
                student = engine.prefill(request_ids, input_ids)
                expected = []
                for rid, ids in zip(request_ids, input_ids):
                    request_tokens[int(rid)] = ids.clone()
                    expected.append(select_last_logits(ref_model, ids))
                assert_logits_close(event["name"], student, torch.stack(expected, dim=0), atol, rtol)
            elif event["op"] == "decode":
                request_ids = event["request_ids"]
                token_ids = event["token_ids"].to(device=device, dtype=torch.long)
                student = engine.decode(request_ids, token_ids)
                expected = []
                for rid, token in zip(request_ids, token_ids):
                    rid = int(rid)
                    request_tokens[rid] = torch.cat([request_tokens[rid], token.reshape(1)])
                    expected.append(select_last_logits(ref_model, request_tokens[rid]))
                assert_logits_close(event["name"], student, torch.stack(expected, dim=0), atol, rtol)
            elif event["op"] == "remove":
                engine.remove(event["request_ids"])
                for rid in event["request_ids"]:
                    request_tokens.pop(int(rid), None)
            else:
                raise ValueError(f"unknown correctness op: {event['op']}")


def run_correctness(
    *,
    engine_path: str,
    model_config_path: str,
    weight_dir: str,
    device: str = "auto",
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, object]:
    device = resolve_device(device)
    model_config = load_json_file(model_config_path)
    compare_engine_with_reference(
        engine_path=engine_path,
        model_config=model_config,
        weight_dir=weight_dir,
        events=build_basic_events(int(model_config["vocab_size"]), device),
        device=device,
        atol=atol,
        rtol=rtol,
    )
    return {
        "status": "passed",
        "engine": engine_path,
        "model_config": model_config_path,
        "device": device,
        "case": "basic",
        "atol": atol,
        "rtol": rtol,
    }


def run_stress_correctness(
    *,
    engine_path: str,
    model_config_path: str,
    weight_dir: str,
    device: str = "auto",
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, object]:
    device = resolve_device(device)
    model_config = load_json_file(model_config_path)
    compare_engine_with_reference(
        engine_path=engine_path,
        model_config=model_config,
        weight_dir=weight_dir,
        events=build_stress_events(int(model_config["vocab_size"]), device),
        device=device,
        atol=atol,
        rtol=rtol,
    )
    return {
        "status": "passed",
        "engine": engine_path,
        "model_config": model_config_path,
        "device": device,
        "case": "stress",
        "atol": atol,
        "rtol": rtol,
    }
