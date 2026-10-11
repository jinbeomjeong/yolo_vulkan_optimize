#!/usr/bin/env bash
# local_size re-sweep for yolo_decode.comp after the argmax shortcut (gpu_postprocess, round "local_sweep_argmax").
# Stage ① grew from 35 to 45 QPU instructions per loop iteration, so the 10x6 choice made in 2026-09 may have moved.
# GPU only, no NPU: run_sweep reads the fixture files and never opens a VDevice.
#
# Two things about run_sweep shape the protocol:
#   - the 26 local_size combinations are hardcoded (--local is ignored) and the 3 interleaved rounds are
#     hardcoded too, with the samples POOLED into one median -> run-level spread needs 3 separate processes.
#   - there is no [RESULT] gate and --validation does not apply. The only correctness net is the cand/frame
#     column, which must be identical across all 26 rows; if it moves, the packing/SWAR has a local-size bug.
# The default out-file is truncated on every run, so each run writes its own file.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
OUT=07_results/gpu_postprocess/runs/local_sweep_argmax/evidence
P=build/bin/yvo_postproc_probe
CROWD=build/out/crowd_fixtures
conds=(
  "default|04_data/raw_tensors|5"
  "mot20_02|$CROWD/mot20_02|10"
  "mot20_05|$CROWD/mot20_05|10"
)

soc_temp() { vcgencmd measure_temp | sed 's/[^0-9.]//g'; }
# Same gate as the emit_split / emit_shortcut rounds so this round stays in the same thermal band.
cool() {
  local t waited=0
  while t=$(soc_temp); awk "BEGIN{exit !($t > 75)}"; do
    [ "$waited" -ge 120 ] && { echo "[cool] giving up at ${t}C after ${waited}s"; return; }
    sleep 10; waited=$((waited + 10))
  done
  [ "$waited" -gt 0 ] && echo "[cool] waited ${waited}s down to ${t}C"
  return 0
}

for run in 1 2 3; do
  for c in "${conds[@]}"; do
    IFS='|' read -r id dir slots <<<"$c"
    cool
    echo "[$(date +%T)] run $run $id (v3d $(vcgencmd measure_clock v3d | cut -d= -f2), $(vcgencmd measure_temp | cut -d= -f2), $(vcgencmd get_throttled))"
    "$P" --probe sweep --fixture-dir "$dir" --slots "$slots" \
      --perf-frames 300 --warmup 30 \
      --out-dir "$OUT" --out-file "sweep_${id}_run${run}.txt" > /dev/null 2>&1
    grep -E '^\[best\]|^\[sweep\]' "$OUT/sweep_${id}_run${run}.txt" | tr '\n' ' '; echo
  done
done
