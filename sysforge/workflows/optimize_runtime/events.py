from __future__ import annotations

import torch


def random_token_ids(vocab_size: int, length: int, device: str) -> torch.Tensor:
    return torch.randint(0, vocab_size, (length,), dtype=torch.long, device=device)


def prefill_event(request_ids: list[int], input_ids: list[torch.Tensor], *, name: str | None = None) -> dict:
    event = {"op": "prefill", "request_ids": request_ids, "input_ids": input_ids}
    if name is not None:
        event["name"] = name
    return event


def decode_event(request_ids: list[int], token_ids: torch.Tensor, *, name: str | None = None) -> dict:
    event = {"op": "decode", "request_ids": request_ids, "token_ids": token_ids}
    if name is not None:
        event["name"] = name
    return event


def remove_event(request_ids: list[int]) -> dict:
    return {"op": "remove", "request_ids": request_ids}
