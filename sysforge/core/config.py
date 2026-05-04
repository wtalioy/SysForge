from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except Exception:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv_if_present()


@dataclass(frozen=True)
class Config:
    target_dir: Path
    workspace_dir: Path
    build_dir: Path
    probes_dir: Path
    logs_dir: Path
    target_spec_path: Path
    output_path: Path
    api_key: str
    base_url: str
    base_model: str
    max_compile_fixes: int = 4
    max_runtime_fixes: int = 3
    max_plausibility_retries: int = 2
    per_target_wallclock_s: float = 240.0
    compile_timeout_s: float = 60.0
    run_timeout_s: float = 45.0
    ncu_timeout_s: float = 120.0
    nvcc_arch: str = ""


def load_config() -> Config:
    target_dir = Path(os.environ.get("TARGET_DIR", "/target"))
    workspace_dir = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
    workspace_dir.mkdir(parents=True, exist_ok=True)

    build_dir = workspace_dir / "_sysforge" / "build"
    probes_dir = workspace_dir / "_sysforge" / "probes"
    logs_dir = workspace_dir / "_sysforge" / "logs"
    for directory in (build_dir, probes_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return Config(
        target_dir=target_dir,
        workspace_dir=workspace_dir,
        build_dir=build_dir,
        probes_dir=probes_dir,
        logs_dir=logs_dir,
        target_spec_path=target_dir / "target_spec.json",
        output_path=workspace_dir / "output.json",
        api_key=os.environ.get("API_KEY", ""),
        base_url=os.environ.get("BASE_URL", ""),
        base_model=os.environ.get("BASE_MODEL", ""),
        nvcc_arch=os.environ.get("NVCC_ARCH", ""),
    )
