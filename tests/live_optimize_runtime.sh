#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

export TARGET_DIR="${TARGET_DIR:-$HERE/target}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$HERE/workspace/live_optimize_runtime}"

echo "[live-optimize-runtime] python: $("$PYTHON_BIN" --version)"
"$PYTHON_BIN" - <<'PY'
import torch
print("[live-optimize-runtime] torch:", torch.__version__)
print("[live-optimize-runtime] cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[live-optimize-runtime] gpu:", torch.cuda.get_device_name(0))
PY

rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

bash run.sh | tee "$WORKSPACE_DIR/live.log"

test -f engine.py
test -f results.log

"$PYTHON_BIN" - <<'PY'
import json
import os
from sysforge.workflows.optimize_runtime.public_checks import run_public_checks

print(json.dumps(run_public_checks(
    engine_path="engine.py",
    model_config_path=os.path.join(os.environ["TARGET_DIR"], "model_config.json"),
    weight_dir=os.path.join(os.environ["TARGET_DIR"], "weights"),
    device="auto",
    case_mode="robust",
    warmup=2,
    repeat=3,
    benchmark_output_path=os.path.join(os.environ["WORKSPACE_DIR"], "promoted_benchmark.json"),
), indent=2))
PY

grep -qi "bootstrap" results.log
grep -qi "kv-cache" results.log
grep -qi "LLM" results.log
grep -qi "correctness" results.log
grep -qi "benchmark" results.log
grep -qi "promoted engine final correctness smoke passed" results.log
grep -qi "promoted engine final benchmark passed" results.log
cp results.log "$WORKSPACE_DIR/results.log"
