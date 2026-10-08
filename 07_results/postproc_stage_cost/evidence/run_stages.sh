#!/usr/bin/env bash
# Stage-cost measurement for yolo_decode.comp (postproc_stage_cost). GPU only, no NPU.
# Repeat (process) is the outer loop, condition the inner loop, so drift is spread across conditions.
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=07_results/postproc_stage_cost/evidence
P=build/bin/yvo_postproc_probe
CROWD=build/out/crowd_fixtures
conds=(
  "default_thr0.2|04_data/raw_tensors|0.2|5"
  "mot20_02_thr0.2|$CROWD/mot20_02|0.2|10"
  "mot20_05_thr0.2|$CROWD/mot20_05|0.2|10"
  "mot20_05_thr0.02|$CROWD/mot20_05|0.02|10"
  "mot20_05_thr0.003|$CROWD/mot20_05|0.003|10"
)
for run in 1 2 3; do
  for c in "${conds[@]}"; do
    IFS='|' read -r id dir thr slots <<<"$c"
    echo "[$(date +%T)] run $run $id (v3d $(vcgencmd measure_clock v3d | cut -d= -f2), $(vcgencmd measure_temp | cut -d= -f2))"
    "$P" --probe stages --fixture-dir "$dir" --score "$thr" --slots "$slots" \
      --perf-frames 500 --warmup 50 --rounds 3 > "$OUT/stages_${id}_run${run}.txt" 2>&1
    grep -E "RESULT|validation" "$OUT/stages_${id}_run${run}.txt" | tr '\n' ' '; echo
  done
done
# NMS stage cost on the same fixtures/thresholds (once each).
for c in "${conds[@]}"; do
  IFS='|' read -r id dir thr slots <<<"$c"
  "$P" --probe nms --fixture-dir "$dir" --score "$thr" > "$OUT/nms_${id}.txt" 2>&1
  grep -E "RESULT" "$OUT/nms_${id}.txt"
done
