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

cd "$HERE"
pip3 install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
exec python3 -m sysforge.main optimize-lora
