from __future__ import annotations

import torch

from .events import decode_event, prefill_event, random_token_ids, remove_event
from .reference_model import ReferenceModel
from .runtime_io import load_engine_module, read_json_file, resolve_device


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
        prefill_event([0], [random_token_ids(vocab_size, 11, device)], name="single_prefill"),
        decode_event([0], random_token_ids(vocab_size, 1, device), name="single_decode"),
        prefill_event(
            [1, 2],
            [random_token_ids(vocab_size, 7, device), random_token_ids(vocab_size, 13, device)],
            name="multi_prefill",
        ),
        decode_event([0, 1, 2], random_token_ids(vocab_size, 3, device), name="multi_decode"),
        remove_event([1]),
        prefill_event([3], [random_token_ids(vocab_size, 5, device)], name="insert_after_remove"),
        decode_event([0, 2, 3], random_token_ids(vocab_size, 3, device), name="decode_after_remove"),
        remove_event([0, 2, 3]),
    ]


def build_stress_events(vocab_size: int, device: str) -> list[dict]:
    torch.manual_seed(17)
    return [
        prefill_event(
            [101, 7, 42],
            [
                random_token_ids(vocab_size, 5, device),
                random_token_ids(vocab_size, 9, device),
                random_token_ids(vocab_size, 3, device),
            ],
            name="non_contiguous_prefill",
        ),
        prefill_event(
            [55, 56],
            [random_token_ids(vocab_size, 6, device), random_token_ids(vocab_size, 6, device)],
            name="same_length_prefill_bucket",
        ),
        decode_event([56, 42, 101, 7, 55], random_token_ids(vocab_size, 5, device), name="reordered_decode"),
        decode_event([7, 42], random_token_ids(vocab_size, 2, device), name="multi_step_decode"),
        prefill_event([42], [random_token_ids(vocab_size, 4, device)], name="replacement_prefill"),
        remove_event([7, 999]),
        decode_event([42, 101], random_token_ids(vocab_size, 2, device), name="decode_after_remove"),
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


def run_correctness_case(
    *,
    engine_path: str,
    model_config_path: str,
    weight_dir: str,
    case: str,
    device: str = "auto",
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, object]:
    device = resolve_device(device)
    model_config = read_json_file(model_config_path)
    if case == "basic":
        events = build_basic_events(int(model_config["vocab_size"]), device)
    elif case == "stress":
        events = build_stress_events(int(model_config["vocab_size"]), device)
    else:
        raise ValueError(f"unknown correctness case: {case}")
    compare_engine_with_reference(
        engine_path=engine_path,
        model_config=model_config,
        weight_dir=weight_dir,
        events=events,
        device=device,
        atol=atol,
        rtol=rtol,
    )
    return {
        "status": "passed",
        "engine": engine_path,
        "model_config": model_config_path,
        "device": device,
        "case": case,
        "atol": atol,
        "rtol": rtol,
    }
