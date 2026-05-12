#!/usr/bin/env bash
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -d "/usr/local/cuda/bin" ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

python3 - <<'PY' || exit 1
try:
    import dotenv  # noqa: F401
    import httpx  # noqa: F401
    import openai  # noqa: F401
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "Missing Python dependency for SysForge runtime. "
        "Install requirements.txt into the current environment first. "
        f"Details: {type(exc).__name__}: {exc}"
    )
PY

cd "$HERE"
exec python3 -m sysforge.main optimize-lora
