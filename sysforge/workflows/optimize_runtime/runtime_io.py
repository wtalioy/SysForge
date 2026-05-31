from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_engine_module(engine_path: str):
    spec = importlib.util.spec_from_file_location("student_engine", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load engine from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json_file(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)
