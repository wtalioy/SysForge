from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompileResult:
    ok: bool
    binary: Path | None
    stdout: str
    stderr: str
    cmd: list[str]


def find_nvcc() -> str | None:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    for candidate in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12/bin/nvcc",
                      "/usr/local/cuda-11.8/bin/nvcc"):
        if Path(candidate).exists():
            return candidate
    return None


def compile_cuda(source_path: Path, binary_path: Path, *,
                 arch: str = "", timeout_s: float = 60.0,
                 extra_flags: list[str] | None = None) -> CompileResult:
    nvcc = find_nvcc()
    if nvcc is None:
        return CompileResult(False, None, "", "nvcc not found on PATH", [])
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nvcc, str(source_path), "-O3", "-std=c++17", "-lineinfo", "-o", str(binary_path)]
    if arch:
        cmd.extend(["-arch", arch])
    if extra_flags:
        cmd.extend(extra_flags)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return CompileResult(False, None, exc.stdout or "", f"nvcc timeout after {timeout_s}s", cmd)
    ok = result.returncode == 0 and binary_path.exists()
    return CompileResult(ok, binary_path if ok else None, result.stdout, result.stderr, cmd)
