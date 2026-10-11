#!/usr/bin/env bash
# emit split measurement for yolo_decode.comp stage 5 (postproc_stage_cost, round "emit_split").
# Splits stage 5 into its rescan share and its atomic share using two diagnostic builds.
# GPU only, no NPU. Same conditions and driver shape as the parent round so the numbers compare directly.
# Repeat (process) is the outer loop, condition the inner loop, so drift is spread across conditions.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
OUT=07_results/postproc_stage_cost/runs/emit_split/evidence
P=build/bin/yvo_postproc_probe
CROWD=build/out/crowd_fixtures
VARIANTS=S1,S2,S3,noatomic,norescan   # Full is always added by the probe
conds=(
  "default_thr0.2|04_data/raw_tensors|0.2|5"
  "mot20_02_thr0.2|$CROWD/mot20_02|0.2|10"
  "mot20_05_thr0.2|$CROWD/mot20_05|0.2|10"
  "mot20_05_thr0.02|$CROWD/mot20_05|0.02|10"
  "mot20_05_thr0.003|$CROWD/mot20_05|0.003|10"
)

soc_temp() { vcgencmd measure_temp | sed 's/[^0-9.]//g'; }
# The parent round ran at 71.9-74.1 C, so gate at the protocol threshold (75 C) rather than forcing a
# colder start: that keeps this round inside the same thermal band instead of creating a second condition.
cool() {
  local t waited=0
  while t=$(soc_temp); awk "BEGIN{exit !($t > 75)}"; do
    [ "$waited" -ge 120 ] && { echo "[cool] giving up at ${t}C after ${waited}s"; return; }
    sleep 10; waited=$((waited + 10))
  done
  [ "$waited" -gt 0 ] && echo "[cool] waited ${waited}s down to ${t}C"
  return 0
}

echo "== stages (variant differencing), variants=$VARIANTS"
for run in 1 2 3; do
  for c in "${conds[@]}"; do
    IFS='|' read -r id dir thr slots <<<"$c"
    cool
    echo "[$(date +%T)] run $run $id (v3d $(vcgencmd measure_clock v3d | cut -d= -f2), $(vcgencmd measure_temp | cut -d= -f2), $(vcgencmd get_throttled))"
    "$P" --probe stages --fixture-dir "$dir" --score "$thr" --slots "$slots" \
      --variants "$VARIANTS" --perf-frames 500 --warmup 50 --rounds 3 \
      > "$OUT/stages_${id}_run${run}.txt" 2>&1
    grep -E "RESULT|validation" "$OUT/stages_${id}_run${run}.txt" | tr '\n' ' '; echo
  done
done

# Passing-anchor distribution per group. CPU only (no GPU/NPU), so it is deterministic and runs once.
echo "== groups (passing-anchor distribution)"
for g in "default|04_data/raw_tensors|0.2" "mot20_02|$CROWD/mot20_02|0.2" "mot20_05|$CROWD/mot20_05|0.2"; do
  IFS='|' read -r id dir thr <<<"$g"
  "$P" --probe groups --fixture-dir "$dir" --score "$thr" > "$OUT/groups_${id}_thr${thr}.txt" 2>&1
  grep -E "RESULT|mean|avg" "$OUT/groups_${id}_thr${thr}.txt" | head -4
done
