#!/usr/bin/env bash
# Run the full live-local test matrix against SysForge.
# Requires: GPU + nvcc on PATH, ncu for analysis groups, `clashon` proxy active
# for LLM calls, and SysForge/.env populated with API_KEY/BASE_URL/BASE_MODEL.
#
# Usage:
#   bash tests/live_profiling.sh                # run every group
#   bash tests/live_profiling.sh group1 group5  # run a subset
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LIVE_ROOT="${LIVE_ROOT:-${OUT_ROOT:-/tmp/sysforge_live}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$LIVE_ROOT/profiling/$RUN_ID}"
mkdir -p "$RUN_DIR"

export PATH="/usr/local/cuda/bin:$PATH"

declare -A TARGET_GROUPS
TARGET_GROUPS[group1]='["dram_latency_cycles","sm_count","actual_boost_clock_mhz"]'
TARGET_GROUPS[group2]='["l1_latency_cycles","l2_latency_cycles","l2_cache_capacity_kb","shared_mem_peak_bw_gbps","global_mem_peak_bw_gbps","bank_conflict_penalty_cycles","max_shmem_per_block_kb"]'
TARGET_GROUPS[group3]='["sm__throughput.avg.pct_of_peak_sustained_elapsed","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","dram__throughput.avg.pct_of_peak_sustained_elapsed","sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active","sm__warps_active.avg.pct_of_peak_sustained_active"]'
TARGET_GROUPS[group4]='["register_file_size_per_sm_kb","warp_scheduler_issue_rate_per_cycle","global_to_shared_async_copy_bw_gbps"]'
TARGET_GROUPS[group5]='["dram_latency_cycles","launch__sm_count","dram__bytes_read.sum.per_second","actual_boost_clock_mhz","l1tex__data_bank_conflicts_pipe_lsu.sum"]'

GROUP_ORDER=(group2 group1 group5 group3 group4)
if [ "$#" -gt 0 ]; then
  GROUP_ORDER=("$@")
fi

summary="$RUN_DIR/summary.txt"
: > "$summary"

for g in "${GROUP_ORDER[@]}"; do
  spec="${TARGET_GROUPS[$g]:-}"
  if [ -z "$spec" ]; then
    echo "unknown group: $g" | tee -a "$summary"
    continue
  fi

  gdir="$RUN_DIR/$g"
  tgt="$gdir/target"
  ws="$gdir/workspace"
  mkdir -p "$tgt" "$ws"
  printf '{"targets": %s}\n' "$spec" > "$tgt/target_spec.json"

  echo "============================================================"
  echo "[$g] spec = $spec"
  echo "[$g] workspace = $ws"
  echo "============================================================"

  start=$(date +%s)
  TARGET_DIR="$tgt" WORKSPACE_DIR="$ws" \
    bash "$ROOT/run.sh" 2>&1 | tee "$gdir/run.log"
  rc=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - start ))

  echo "[$g] rc=$rc elapsed=${elapsed}s" | tee -a "$summary"

  out="$ws/output.json"
  if [ -f "$out" ]; then
    python3 - "$out" "$g" >> "$summary" <<'PY'
import json, sys
path, group = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
hw = data.get("hardware") or {}
for k, v in hw.items():
    val = v.get("value")
    unit = v.get("unit")
    conf = v.get("confidence")
    err = v.get("error") or ""
    n = len(v.get("accepted_samples") or [])
    tag = f"ERR: {err}" if err else ""
    print(f"  [{group}] {k} = {val} {unit} (conf={conf}, samples={n}) {tag}".rstrip())
an = data.get("analysis")
if an:
    print(f"  [{group}] analysis.bottleneck = {an.get('bottleneck')}  err={an.get('error','')}")
errs = data.get("errors") or []
for e in errs:
    first = (e.splitlines()[0] if e else "")
    print(f"  [{group}] ERROR: {first}")
PY
    cp "$out" "$gdir/output.json"
  else
    echo "[$g] no output.json produced" | tee -a "$summary"
  fi
  echo "" | tee -a "$summary"
done

echo "============================================================"
echo "Summary written to $summary"
echo "Per-group artifacts under $RUN_DIR/<group>/{run.log,output.json,workspace/_sysforge/}"
