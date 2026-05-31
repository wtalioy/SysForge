#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LIVE_ROOT="${LIVE_ROOT:-${OUT_ROOT:-/tmp/sysforge_live}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$LIVE_ROOT/optimize_lora/$RUN_ID}"
TARGET_DIR="${TARGET_DIR:-$RUN_DIR/target}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$RUN_DIR/workspace}"
RUN_LOG="$RUN_DIR/run.log"
RESULT_OUT="$RUN_DIR/result.out"
HARNESS_JSON="$RUN_DIR/harness_summary.json"
OUTPUT_COPY="$RUN_DIR/output.json"
ARTIFACT_PATH="$ROOT/optimized_lora.cu"

fail() {
  echo "[live-optimize-lora] $*" >&2
  exit 1
}

mkdir -p "$RUN_DIR" "$TARGET_DIR" "$WORKSPACE_DIR"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [ -d "/usr/local/cuda/bin" ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
  fail "conda is required to activate the base environment"
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
[ -n "$CONDA_BASE" ] || fail "unable to resolve conda base path"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || fail "conda activation script not found under $CONDA_BASE"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base >/dev/null 2>&1 || fail "failed to activate conda environment 'base'"
[ "${CONDA_DEFAULT_ENV:-}" = "base" ] || fail "conda activated unexpected environment '${CONDA_DEFAULT_ENV:-}'"

command -v python3 >/dev/null 2>&1 || fail "python3 not found in conda base environment"
command -v nvcc >/dev/null 2>&1 || fail "nvcc not found in conda base environment"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
nvidia-smi -L >/dev/null 2>&1 || fail "no visible NVIDIA GPU detected"

[ -n "${BASE_MODEL:-}" ] || fail "BASE_MODEL is required for a real optimize-lora live test"

export TARGET_DIR WORKSPACE_DIR
export OPTIMIZE_LORA_VALIDATION_SHAPE="${OPTIMIZE_LORA_VALIDATION_SHAPE:-4096}"
export OPTIMIZE_LORA_BENCH_WARMUP="${OPTIMIZE_LORA_BENCH_WARMUP:-1}"
export OPTIMIZE_LORA_BENCH_ITERS="${OPTIMIZE_LORA_BENCH_ITERS:-3}"
export OPTIMIZE_LORA_MAX_FAMILY_VARIANTS="${OPTIMIZE_LORA_MAX_FAMILY_VARIANTS:-6}"
export OPTIMIZE_LORA_MAX_FULL_EVALS_PER_ROUND="${OPTIMIZE_LORA_MAX_FULL_EVALS_PER_ROUND:-3}"
export OPTIMIZE_LORA_MAX_LLM_ROUNDS="${OPTIMIZE_LORA_MAX_LLM_ROUNDS:-6}"
export OPTIMIZE_LORA_CLEAR_WINNER_SPEEDUP="${OPTIMIZE_LORA_CLEAR_WINNER_SPEEDUP:-1.05}"
export OPTIMIZE_LORA_PROFILE_ENABLED="${OPTIMIZE_LORA_PROFILE_ENABLED:-0}"
export OPTIMIZE_LORA_FINAL_CONFIRM_WARMUP="${OPTIMIZE_LORA_FINAL_CONFIRM_WARMUP:-2}"
export OPTIMIZE_LORA_FINAL_CONFIRM_ITERS="${OPTIMIZE_LORA_FINAL_CONFIRM_ITERS:-6}"
export OPTIMIZE_LORA_MIN_SEED_VARIANTS="${OPTIMIZE_LORA_MIN_SEED_VARIANTS:-3}"
export OPTIMIZE_LORA_MAX_STALLED_ROUNDS="${OPTIMIZE_LORA_MAX_STALLED_ROUNDS:-3}"
export OPTIMIZE_LORA_TIER1_SHAPES="${OPTIMIZE_LORA_TIER1_SHAPES:-4096,4608}"
export OPTIMIZE_LORA_TIER2_SHAPES="${OPTIMIZE_LORA_TIER2_SHAPES:-3584,4096,4608}"
export OPTIMIZE_LORA_TIER3_SHAPES="${OPTIMIZE_LORA_TIER3_SHAPES:-3584,4096,4608}"
export OPTIMIZE_LORA_SCREEN_WARMUP="${OPTIMIZE_LORA_SCREEN_WARMUP:-1}"
export OPTIMIZE_LORA_SCREEN_ITERS="${OPTIMIZE_LORA_SCREEN_ITERS:-2}"
export HARNESS_SHAPES="${HARNESS_SHAPES:-4096,3584}"
export RESULT_OUT HARNESS_JSON ARTIFACT_PATH OUTPUT_COPY
export LIVE_TEST_BUILD_DIR="$RUN_DIR/harness_build"
export LIVE_PASS_SPEEDUP_GUARD="${LIVE_PASS_SPEEDUP_GUARD:-1.01}"

cd "$ROOT"

echo "============================================================"
echo "[live-optimize-lora] root       = $ROOT"
echo "[live-optimize-lora] run dir    = $RUN_DIR"
echo "[live-optimize-lora] workspace  = $WORKSPACE_DIR"
echo "[live-optimize-lora] target dir = $TARGET_DIR"
echo "[live-optimize-lora] python     = $(command -v python3)"
echo "[live-optimize-lora] conda env  = ${CONDA_DEFAULT_ENV:-}"
echo "============================================================"

RUN_TIMEOUT="${SYSFORGE_RUN_TIMEOUT:-30m}"
if command -v timeout >/dev/null 2>&1; then
  timeout "$RUN_TIMEOUT" bash "$ROOT/run.sh" 2>&1 | tee "$RUN_LOG"
else
  bash "$ROOT/run.sh" 2>&1 | tee "$RUN_LOG"
fi

[ -s "$ARTIFACT_PATH" ] || fail "optimized_lora.cu was not created at the repo root"
grep -q 'torch::Tensor forward' "$ARTIFACT_PATH" || fail "optimized_lora.cu is missing torch::Tensor forward"
grep -q 'PYBIND11_MODULE' "$ARTIFACT_PATH" || fail "optimized_lora.cu is missing PYBIND11_MODULE"

[ -f "$WORKSPACE_DIR/output.json" ] || fail "workspace output.json was not produced"
cp "$WORKSPACE_DIR/output.json" "$OUTPUT_COPY"

python3 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def benchmark(fn, W, X, A, B, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn(W, X, A, B)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(W, X, A, B)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def reference_impl(W, X, A, B):
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


artifact_path = Path(os.environ["ARTIFACT_PATH"]).resolve()
output_path = Path(os.environ["OUTPUT_COPY"]).resolve()
result_out = Path(os.environ["RESULT_OUT"]).resolve()
harness_json = Path(os.environ["HARNESS_JSON"]).resolve()
build_dir = Path(os.environ["LIVE_TEST_BUILD_DIR"]).resolve()
build_dir.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false in conda base environment")

major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

workflow_output = json.loads(output_path.read_text(encoding="utf-8"))
if workflow_output.get("workflow") != "optimize-lora":
    raise SystemExit(f"unexpected workflow in output.json: {workflow_output.get('workflow')!r}")
if workflow_output.get("status") not in {"optimized", "searched", "confirmed_baseline"}:
    raise SystemExit(f"unexpected optimize-lora status: {workflow_output.get('status')!r}")
if not workflow_output.get("llm_enabled"):
    raise SystemExit("output.json reports llm_enabled=false")
if not workflow_output.get("candidates"):
    raise SystemExit("output.json has no candidate records")
if workflow_output.get("benchmark_summary") is None:
    raise SystemExit("output.json is missing benchmark_summary")
if workflow_output.get("finalist_summary") is None:
    raise SystemExit("output.json is missing finalist_summary")
if os.environ.get("OPTIMIZE_LORA_PROFILE_ENABLED") == "1":
    if not workflow_output.get("profiling_used"):
        profile_errors = [
            (candidate.get("profile_summary") or {}).get("error", "").strip()
            for candidate in (workflow_output.get("candidates") or [])
            if candidate.get("family") != "baseline"
        ]
        profile_errors = [error for error in profile_errors if error]
        if profile_errors:
            raise SystemExit(
                "profiling was requested but no successful profile completed; first profiler error:\n"
                + profile_errors[0]
            )
        raise SystemExit("profiling was requested but output.json reports profiling_used=false")
    profiled = [
        candidate
        for candidate in (workflow_output.get("candidates") or [])
        if candidate.get("family") != "baseline"
        and (candidate.get("profile_summary") or {}).get("error", "") == ""
    ]
    if not profiled:
        raise SystemExit("profiling was requested but no non-baseline candidate has a successful profile_summary")

promoted_source = Path(workflow_output.get("promoted_source_path", "")).resolve()
if promoted_source != artifact_path:
    raise SystemExit(f"promoted_source_path mismatch: expected {artifact_path}, saw {promoted_source}")

module_name = f"optimized_lora_live_{os.getpid()}"
module = load(
    name=module_name,
    sources=[str(artifact_path)],
    verbose=False,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    build_directory=str(build_dir),
)

shapes = [int(part.strip()) for part in os.environ["HARNESS_SHAPES"].split(",") if part.strip()]
warmup = int(os.environ.get("OPTIMIZE_LORA_BENCH_WARMUP", "1"))
iters = int(os.environ.get("OPTIMIZE_LORA_BENCH_ITERS", "3"))

results = []
overall_passed = True
max_abs_err = 0.0
max_rel_l2_err = 0.0

for offset, shape_d in enumerate(shapes):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(17 + offset)
    W = torch.randn((shape_d, shape_d), device="cuda", dtype=torch.float32, generator=gen).contiguous()
    X = torch.randn((shape_d, shape_d), device="cuda", dtype=torch.float32, generator=gen).contiguous()
    A = torch.randn((shape_d, 16), device="cuda", dtype=torch.float32, generator=gen).contiguous()
    B = torch.randn((shape_d, 16), device="cuda", dtype=torch.float32, generator=gen).contiguous()

    with torch.no_grad():
        y_student = module.forward(W, X, A, B)
        y_ref = reference_impl(W, X, A, B)

    diff = (y_student - y_ref).float()
    shape_max_abs_err = diff.abs().max().item()
    shape_rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4)
    overall_passed = overall_passed and passed
    max_abs_err = max(max_abs_err, shape_max_abs_err)
    max_rel_l2_err = max(max_rel_l2_err, shape_rel_l2_err)

    student_ms = benchmark(module.forward, W, X, A, B, warmup=warmup, iters=iters) if passed else None
    torch_ms = benchmark(reference_impl, W, X, A, B, warmup=warmup, iters=iters) if passed else None
    speedup = (torch_ms / student_ms) if passed and student_ms else 0.0

    results.append(
        {
            "shape_d": shape_d,
            "correct": passed,
            "max_abs_err": shape_max_abs_err,
            "rel_l2_err": shape_rel_l2_err,
            "student_median_ms": student_ms,
            "torch_median_ms": torch_ms,
            "speedup": speedup,
        }
    )

lines = []
for entry in results:
    lines.extend(
        [
            f"shape_d: {entry['shape_d']}",
            f"correct: {entry['correct']}",
            f"max_abs_err: {entry['max_abs_err']}",
            f"rel_l2_err: {entry['rel_l2_err']}",
            f"student_median_ms: {entry['student_median_ms']}",
            f"torch_median_ms: {entry['torch_median_ms']}",
            f"speedup: {entry['speedup']}",
            "",
        ]
    )
lines.extend(
    [
        f"overall_passed: {overall_passed}",
        f"overall_max_abs_err: {max_abs_err}",
        f"overall_rel_l2_err: {max_rel_l2_err}",
    ]
)
result_out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

payload = {
    "artifact_path": str(artifact_path),
    "workflow_status": workflow_output.get("status"),
    "llm_enabled": workflow_output.get("llm_enabled"),
    "candidate_count": len(workflow_output.get("candidates") or []),
    "current_best_candidate_id": workflow_output.get("current_best_candidate_id"),
    "best_candidate_kind": workflow_output.get("best_candidate_kind"),
    "seed_count_run": workflow_output.get("seed_count_run"),
    "mutation_count_run": workflow_output.get("mutation_count_run"),
    "seed_variants_screened": workflow_output.get("seed_variants_screened"),
    "mutation_variants_screened": workflow_output.get("mutation_variants_screened"),
    "winner_confirmed": workflow_output.get("winner_confirmed"),
    "winner_confirmation_candidate_id": workflow_output.get("winner_confirmation_candidate_id"),
    "best_tier2_speedup": workflow_output.get("best_tier2_speedup"),
    "best_tier3_speedup": workflow_output.get("best_tier3_speedup"),
    "compile_time_s_total": workflow_output.get("compile_time_s_total"),
    "variants_benchmarked": workflow_output.get("variants_benchmarked"),
    "skipped_steps": workflow_output.get("skipped_steps") or [],
    "errors": workflow_output.get("errors") or [],
    "overall_passed": overall_passed,
    "overall_max_abs_err": max_abs_err,
    "overall_rel_l2_err": max_rel_l2_err,
    "max_harness_speedup": max((entry["speedup"] for entry in results), default=0.0),
    "results": results,
}
harness_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

if not overall_passed:
    raise SystemExit("live harness correctness check failed")
PY

CLASSIFICATION="$(python3 - <<'PY'
from __future__ import annotations
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["HARNESS_JSON"]).read_text(encoding="utf-8"))
guard = float(os.environ["LIVE_PASS_SPEEDUP_GUARD"])
best_kind = payload.get("best_candidate_kind")
seed_count = int(payload.get("seed_count_run") or 0)
mutation_count = int(payload.get("mutation_count_run") or 0)
seed_variants = int(payload.get("seed_variants_screened") or 0)
mutation_variants = int(payload.get("mutation_variants_screened") or 0)
winner_confirmed = bool(payload.get("winner_confirmed"))
overall_passed = bool(payload.get("overall_passed"))
best_tier3 = float(payload.get("best_tier3_speedup") or 0.0)
max_harness = float(payload.get("max_harness_speedup") or 0.0)

strong_guard = max(1.01, guard)
consistent_guard = 1.005

if (
    overall_passed
    and best_kind == "optimized"
    and winner_confirmed
    and seed_count >= 3
    and (
        best_tier3 >= strong_guard
        or (best_tier3 >= consistent_guard and max_harness >= consistent_guard)
    )
):
    print("PASS_OPTIMIZED")
elif overall_passed and winner_confirmed and seed_count >= 3:
    print("PASS_SEARCHED")
elif overall_passed and seed_count < 3 and mutation_count == 0 and seed_variants < 3 and mutation_variants == 0:
    print("FAIL_NOT_ENOUGH_SEARCH")
else:
    print("FAIL_WORKFLOW")
PY
)"

echo "============================================================"
echo "[live-optimize-lora] $CLASSIFICATION"
echo "[live-optimize-lora] run log        = $RUN_LOG"
echo "[live-optimize-lora] output.json    = $OUTPUT_COPY"
echo "[live-optimize-lora] result.out     = $RESULT_OUT"
echo "[live-optimize-lora] harness json   = $HARNESS_JSON"
echo "[live-optimize-lora] workspace data = $WORKSPACE_DIR/_sysforge"
python3 - <<'PY'
from __future__ import annotations
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["HARNESS_JSON"]).read_text(encoding="utf-8"))
print(f"[live-optimize-lora] winner         = {payload.get('current_best_candidate_id')}")
print(f"[live-optimize-lora] winner kind    = {payload.get('best_candidate_kind')}")
print(f"[live-optimize-lora] seeds run      = {payload.get('seed_count_run')}")
print(f"[live-optimize-lora] mutations run  = {payload.get('mutation_count_run')}")
print(f"[live-optimize-lora] seed screened  = {payload.get('seed_variants_screened')}")
print(f"[live-optimize-lora] mut screened   = {payload.get('mutation_variants_screened')}")
print(f"[live-optimize-lora] winner confirm = {payload.get('winner_confirmed')}")
print(f"[live-optimize-lora] confirm id     = {payload.get('winner_confirmation_candidate_id')}")
print(f"[live-optimize-lora] best tier3     = {payload.get('best_tier3_speedup')}")
print(f"[live-optimize-lora] harness max    = {payload.get('max_harness_speedup')}")
print(f"[live-optimize-lora] compile total  = {payload.get('compile_time_s_total')}")
print(f"[live-optimize-lora] variants bench = {payload.get('variants_benchmarked')}")
errors = payload.get("errors") or []
if errors:
    print(f"[live-optimize-lora] first error    = {errors[0]}")
skipped_steps = payload.get("skipped_steps") or []
for skipped in skipped_steps:
    print(
        "[live-optimize-lora] skipped step   = "
        f"{skipped.get('step_name')} candidate={skipped.get('candidate_id')} "
        f"remaining={skipped.get('remaining_s')} reserve={skipped.get('required_reserve_s')}"
    )
PY
echo "============================================================"

if [ "$CLASSIFICATION" != "PASS_OPTIMIZED" ]; then
  fail "workflow did not produce a confirmed optimized winner with a broader seed sweep"
fi
