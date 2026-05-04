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

pip3 install -q --disable-pip-version-check -r "$HERE/requirements.txt" \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --default-timeout 30 || \
  pip3 install -q --disable-pip-version-check -r "$HERE/requirements.txt" || true

cd "$HERE"
exec python3 -m sysforge.main optimize-lora
