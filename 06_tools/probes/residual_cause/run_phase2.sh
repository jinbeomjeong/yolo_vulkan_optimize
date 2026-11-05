#!/usr/bin/env bash
# Phase 2: L1-only kernels, PMU matrix, cycle sampling. Output only under 11_logs/2026-11-06_residual_cause.
set -euo pipefail
D=/home/pi/workspace/yolo_vulkan_optimize/11_logs/2026-11-06_residual_cause
cd "$D/runs"
CORE="ref rowsplit local_params nobranch_nohoist hoist_y hoist_all local_hoist ref_notrap nobranch rowbuf split split_nohoist"
mkdir -p core
: > core/core.log
for run in 1 2 3 4 5; do
  for v in $CORE; do
    (cd core && taskset -c 3 "$D/bin/core_only" "$v") >> core/core.log 2>&1
  done
done
echo core done
OUT=$D/runs/counters RUNS=3 COUNTER_SETS="1 2" VARIANTS="$CORE" bash "$D/src/run_matrix.sh"
mkdir -p "$D/runs/perf"
for v in ref rowsplit nobranch split ref_notrap hoist_all; do
  (cd "$D/runs/perf" && taskset -c 3 perf record -q -e cycles:u -F 12000 -o "$v.data" -- "$D/bin/residual_probe" --variant "$v" --frames 600 --warmup 60) > "$D/runs/perf/$v.log" 2>&1
done
echo phase2 finished
