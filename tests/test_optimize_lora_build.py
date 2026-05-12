from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sysforge.integrations.workspace import Workspace  # noqa: E402
from sysforge.workflows.optimize_lora.build import BASELINE_SOURCE, CandidateBuilder, module_name_for_hash, source_hash  # noqa: E402
from sysforge.workflows.optimize_lora.harness import reference_impl  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace(probes_dir=tmp_path / "probes", build_dir=tmp_path / "build", logs_dir=tmp_path / "logs")


class _FakeCompiledModule:
    def __init__(self) -> None:
        self.forward = reference_impl


def test_source_hash_and_module_name_are_stable():
    digest = source_hash(BASELINE_SOURCE)
    assert digest == source_hash(BASELINE_SOURCE)
    assert module_name_for_hash(digest).startswith("optimized_lora_")


def test_candidate_builder_loads_single_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sysforge.workflows.optimize_lora.build.load",
        lambda name, sources, verbose, extra_cuda_cflags, with_cuda, build_directory: _FakeCompiledModule(),
    )
    builder = CandidateBuilder(_workspace(tmp_path))
    candidate = builder.register_candidate(
        candidate_id="candidate-a",
        family="llm_revision",
        source=BASELINE_SOURCE,
    )
    compile_result, module = builder.load_candidate(candidate)
    assert compile_result.status == "built"
    assert compile_result.duration_s >= 0.0
    assert module is not None

    d, r = 64, 16
    W = torch.randn((d, d), device="cuda", dtype=torch.float32)
    X = torch.randn((d, d), device="cuda", dtype=torch.float32)
    A = torch.randn((d, r), device="cuda", dtype=torch.float32)
    B = torch.randn((d, r), device="cuda", dtype=torch.float32)
    y_student = getattr(module, candidate.entrypoint_name)(W, X, A, B)
    y_ref = reference_impl(W, X, A, B)
    assert torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4)


def test_candidate_builder_caches_identical_source(tmp_path, monkeypatch):
    compile_calls = {"count": 0}

    def fake_load(name, sources, verbose, extra_cuda_cflags, with_cuda, build_directory):
        compile_calls["count"] += 1
        return _FakeCompiledModule()

    monkeypatch.setattr("sysforge.workflows.optimize_lora.build.load", fake_load)
    builder = CandidateBuilder(_workspace(tmp_path))
    first = builder.register_candidate(candidate_id="candidate-a", family="llm_revision", source=BASELINE_SOURCE)
    second = builder.register_candidate(candidate_id="candidate-b", family="llm_revision", source=BASELINE_SOURCE)

    first_result, first_module = builder.load_candidate(first)
    second_result, second_module = builder.load_candidate(second)

    assert first_result.status == "built"
    assert second_result.status == "cache_hit"
    assert first_module is second_module
    assert compile_calls["count"] == 1


def test_candidate_builder_caches_compile_failures(tmp_path, monkeypatch):
    compile_calls = {"count": 0}

    def fake_load(name, sources, verbose, extra_cuda_cflags, with_cuda, build_directory):
        compile_calls["count"] += 1
        raise RuntimeError("nvcc failed")

    monkeypatch.setattr("sysforge.workflows.optimize_lora.build.load", fake_load)
    builder = CandidateBuilder(_workspace(tmp_path))
    candidate = builder.register_candidate(candidate_id="candidate-failed", family="llm_revision", source=BASELINE_SOURCE)

    first_result, first_module = builder.load_candidate(candidate)
    second_result, second_module = builder.load_candidate(candidate)

    assert first_result.status == "failed"
    assert second_result.status == "failed"
    assert "nvcc failed" in first_result.error
    assert first_module is None
    assert second_module is None
    assert compile_calls["count"] == 1
