from __future__ import annotations

import re
from pathlib import Path


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(name: str) -> str:
    return _SAFE.sub("_", name).strip("_") or "probe"


class Workspace:
    def __init__(self, probes_dir: Path, build_dir: Path, logs_dir: Path) -> None:
        self.probes_dir = probes_dir
        self.build_dir = build_dir
        self.logs_dir = logs_dir
        for directory in (probes_dir, build_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def source_path(self, target: str, attempt: int) -> Path:
        return self.probes_dir / f"{safe_name(target)}_v{attempt}.cu"

    def binary_path(self, target: str, attempt: int) -> Path:
        return self.build_dir / f"{safe_name(target)}_v{attempt}"

    def write_source(self, target: str, attempt: int, source: str) -> Path:
        path = self.source_path(target, attempt)
        path.write_text(source)
        return path

    def write_log(self, target: str, content: str) -> Path:
        path = self.logs_dir / f"{safe_name(target)}.log"
        with path.open("a") as handle:
            handle.write(content + "\n")
        return path

    def candidate_source_path(self, source_hash: str) -> Path:
        return self.probes_dir / f"candidate_{safe_name(source_hash)}.cu"

    def candidate_build_dir(self, source_hash: str) -> Path:
        return self.build_dir / f"candidate_{safe_name(source_hash)}"

    def candidate_log_path(self, source_hash: str) -> Path:
        return self.logs_dir / f"candidate_{safe_name(source_hash)}.log"
