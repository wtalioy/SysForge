from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from ...integrations.workspace import Workspace
from .models import CandidateCompileResult, CandidateRecord
from .templates import BASELINE_SOURCE


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def module_name_for_hash(source_digest: str) -> str:
    return f"optimized_lora_{source_digest[:12]}"


def _append_log(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content.rstrip() + "\n")


CUDA_ONLY_INCLUDE_RE = re.compile(
    r"^\s*#include\s+<(?:ATen/cuda/CUDAContext\.h|cuda\.h|cuda_runtime\.h)>\s*$",
    re.MULTILINE,
)
CUDA_SYNTAX_RE = re.compile(
    r"__global__|__device__|__host__|<<<|>>>|\bcuda(?:Malloc|Free|Memcpy|Memset|DeviceSynchronize|GetLastError|Stream|Event)\b|"
    r"\bCU(?:device|context|stream|event|function|module)\b|\bthreadIdx\b|\bblockIdx\b|\bblockDim\b|\bgridDim\b",
)
STALE_LOCK_AGE_S = 300


def _detect_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}.{minor}"


@contextmanager
def _torch_arch_list_env():
    prior = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if prior:
        yield
        return
    detected = _detect_arch_list()
    os.environ["TORCH_CUDA_ARCH_LIST"] = detected
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)


class CandidateBuilder:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._module_cache: dict[str, object] = {}
        self._compile_cache: dict[str, CandidateCompileResult] = {}

    def register_candidate(
        self,
        *,
        candidate_id: str,
        family: str,
        source: str,
        entrypoint_name: str = "forward",
    ) -> CandidateRecord:
        digest = source_hash(source)
        source_path = self.workspace.candidate_source_path(digest)
        if not source_path.exists():
            source_path.write_text(source, encoding="utf-8")
        return CandidateRecord(
            candidate_id=candidate_id,
            family=family,
            source_hash=digest,
            module_name=module_name_for_hash(digest),
            source_path=str(source_path),
            entrypoint_name=entrypoint_name,
        )

    def _source_uses_cuda(self, candidate: CandidateRecord) -> bool:
        source = Path(candidate.source_path).read_text(encoding="utf-8")
        return CUDA_SYNTAX_RE.search(source) is not None

    def _prepare_build_source(self, candidate: CandidateRecord, build_dir: Path) -> tuple[Path, bool]:
        source_path = Path(candidate.source_path)
        source = source_path.read_text(encoding="utf-8")
        if CUDA_SYNTAX_RE.search(source):
            return source_path, True
        cpp_path = build_dir / f"{candidate.module_name}.cpp"
        cpp_source = CUDA_ONLY_INCLUDE_RE.sub("", source).strip() + "\n"
        cpp_path.parent.mkdir(parents=True, exist_ok=True)
        cpp_path.write_text(cpp_source, encoding="utf-8")
        return cpp_path, False

    def _clear_stale_lock(self, build_dir: Path) -> None:
        lock_path = build_dir / "lock"
        if not lock_path.exists():
            return
        age_s = time.time() - lock_path.stat().st_mtime
        if age_s < STALE_LOCK_AGE_S:
            return
        lock_path.unlink(missing_ok=True)

    def _ensure_backend_build_dir(self, candidate: CandidateRecord, backend: str) -> Path:
        legacy_build_dir = self.workspace.candidate_build_dir(candidate.source_hash)
        build_dir = legacy_build_dir / backend
        build_dir.mkdir(parents=True, exist_ok=True)
        self._clear_stale_lock(build_dir)
        legacy_lock = legacy_build_dir / "lock"
        if legacy_lock.exists():
            age_s = time.time() - legacy_lock.stat().st_mtime
            if age_s >= STALE_LOCK_AGE_S:
                legacy_lock.unlink(missing_ok=True)
        legacy_ninja = legacy_build_dir / "build.ninja"
        if legacy_ninja.exists() and not any(build_dir.iterdir()):
            for path in legacy_build_dir.iterdir():
                if path.name == backend:
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        return build_dir

    def load_candidate(self, candidate: CandidateRecord) -> tuple[CandidateCompileResult, object | None]:
        cached_module = self._module_cache.get(candidate.source_hash)
        if cached_module is not None:
            cached = self._compile_cache[candidate.source_hash]
            return CandidateCompileResult(
                status="cache_hit",
                source_hash=cached.source_hash,
                module_name=cached.module_name,
                source_path=cached.source_path,
                build_dir=cached.build_dir,
                log_path=cached.log_path,
                error=cached.error,
                duration_s=0.0,
            ), cached_module
        cached_failure = self._compile_cache.get(candidate.source_hash)
        if cached_failure is not None:
            return cached_failure, None

        log_path = self.workspace.candidate_log_path(candidate.source_hash)
        backend = "cuda" if self._source_uses_cuda(candidate) else "cpp"
        build_dir = self._ensure_backend_build_dir(candidate, backend)
        build_source_path, with_cuda = self._prepare_build_source(candidate, build_dir)
        backend = "cuda" if with_cuda else "cpp"
        _append_log(
            log_path,
            f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] build_start module={candidate.module_name} "
            f"source={candidate.source_path} build_source={build_source_path} backend={backend} build_dir={build_dir}",
        )
        started = time.monotonic()
        try:
            with _torch_arch_list_env() if with_cuda else nullcontext():
                module = load(
                    name=candidate.module_name,
                    sources=[str(build_source_path)],
                    verbose=False,
                    extra_cuda_cflags=["-O3"],
                    with_cuda=with_cuda,
                    build_directory=str(build_dir),
                )
        except Exception as exc:  # noqa: BLE001
            duration_s = time.monotonic() - started
            error = f"{type(exc).__name__}: {exc}"
            _append_log(
                log_path,
                f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] build_failed {error}\n{traceback.format_exc()}",
            )
            result = CandidateCompileResult(
                status="failed",
                source_hash=candidate.source_hash,
                module_name=candidate.module_name,
                source_path=candidate.source_path,
                build_dir=str(build_dir),
                log_path=str(log_path),
                error=error,
                duration_s=duration_s,
            )
            self._compile_cache[candidate.source_hash] = result
            return result, None

        duration_s = time.monotonic() - started
        _append_log(log_path, f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] build_ok module={candidate.module_name} duration_s={duration_s:.3f}")
        result = CandidateCompileResult(
            status="built",
            source_hash=candidate.source_hash,
            module_name=candidate.module_name,
            source_path=candidate.source_path,
            build_dir=str(build_dir),
            log_path=str(log_path),
            error="",
            duration_s=duration_s,
        )
        self._compile_cache[candidate.source_hash] = result
        self._module_cache[candidate.source_hash] = module
        return result, module
