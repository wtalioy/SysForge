#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

LIVE_ROOT="${LIVE_ROOT:-${OUT_ROOT:-/tmp/sysforge_live}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$LIVE_ROOT/optimize_runtime/$RUN_ID}"
export TARGET_DIR="${TARGET_DIR:-$HERE/target}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$RUN_DIR/workspace}"
RUN_LOG="$RUN_DIR/run.log"
RESULTS_COPY="$RUN_DIR/results.log"
OUTPUT_COPY="$RUN_DIR/output.json"
export BENCHMARK_JSON="$RUN_DIR/promoted_benchmark.json"

echo "[live-optimize-runtime] python: $("$PYTHON_BIN" --version)"
"$PYTHON_BIN" - <<'PY'
import torch
print("[live-optimize-runtime] torch:", torch.__version__)
print("[live-optimize-runtime] cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[live-optimize-runtime] gpu:", torch.cuda.get_device_name(0))
PY

rm -rf "$RUN_DIR"
mkdir -p "$WORKSPACE_DIR"

bash run.sh | tee "$RUN_LOG"

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
    warmup=2,
    repeat=3,
    benchmark_output_path=os.environ["BENCHMARK_JSON"],
), indent=2))
PY

grep -qi "bootstrap" results.log
grep -qi "kv-cache" results.log
grep -qi "LLM" results.log
grep -qi "correctness" results.log
grep -qi "benchmark" results.log
grep -qi "promoted engine final correctness check passed" results.log
grep -qi "promoted engine final benchmark passed" results.log
cp results.log "$RESULTS_COPY"
if [ -f "$WORKSPACE_DIR/output.json" ]; then
  cp "$WORKSPACE_DIR/output.json" "$OUTPUT_COPY"
fi

echo "============================================================"
echo "[live-optimize-runtime] run dir        = $RUN_DIR"
echo "[live-optimize-runtime] run log        = $RUN_LOG"
echo "[live-optimize-runtime] output.json    = $OUTPUT_COPY"
echo "[live-optimize-runtime] results.log    = $RESULTS_COPY"
echo "[live-optimize-runtime] benchmark json = $BENCHMARK_JSON"
echo "[live-optimize-runtime] workspace data = $WORKSPACE_DIR/_sysforge"
echo "============================================================"
