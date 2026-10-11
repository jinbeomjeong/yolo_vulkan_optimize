#!/usr/bin/env bash
# argmax shortcut measurement for yolo_decode.comp stage 5 (postproc_stage_cost, round "emit_shortcut").
# `legacy` is the pre-change ①+⑤ compiled from the same source via YVO_DECODE_LEGACY_EMIT, so
# `Full - legacy` is the net effect measured in one binary with round interleaving - no cross-commit compare.
# `S1 - stop1_legacy` isolates the cost added to ①. GPU only, no NPU.
# Same conditions and driver shape as the parent round and the emit_split round, so the numbers compare directly.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
OUT=07_results/postproc_stage_cost/runs/emit_shortcut/evidence
P=build/bin/yvo_postproc_probe
CROWD=build/out/crowd_fixtures
VARIANTS=S1,S2,S3,stop1_legacy,legacy   # Full is always added by the probe
conds=(
  "default_thr0.2|04_data/raw_tensors|0.2|5"
  "mot20_02_thr0.2|$CROWD/mot20_02|0.2|10"
  "mot20_05_thr0.2|$CROWD/mot20_05|0.2|10"
  "mot20_05_thr0.02|$CROWD/mot20_05|0.02|10"
  "mot20_05_thr0.003|$CROWD/mot20_05|0.003|10"
)

soc_temp() { vcgencmd measure_temp | sed 's/[^0-9.]//g'; }
# Gate at the protocol threshold so this round stays in the same thermal band as the parent (71.9-74.1 C)
# and the emit_split round (71.4-74.7 C) instead of becoming a second condition.
cool() {
  local t waited=0
  while t=$(soc_temp); awk "BEGIN{exit !($t > 75)}"; do
    [ "$waited" -ge 120 ] && { echo "[cool] giving up at ${t}C after ${waited}s"; return; }
    sleep 10; waited=$((waited + 10))
  done
  [ "$waited" -gt 0 ] && echo "[cool] waited ${waited}s down to ${t}C"
  return 0
}

echo "== stages, variants=$VARIANTS  (no validation layer: it distorts GPU time)"
for run in 1 2 3; do
  for c in "${conds[@]}"; do
    IFS='|' read -r id dir thr slots <<<"$c"
    cool
    echo "[$(date +%T)] run $run $id (v3d $(vcgencmd measure_clock v3d | cut -d= -f2), $(vcgencmd measure_temp | cut -d= -f2), $(vcgencmd get_throttled))"
    "$P" --probe stages --fixture-dir "$dir" --score "$thr" --slots "$slots" \
      --variants "$VARIANTS" --perf-frames 500 --warmup 50 --rounds 3 \
      > "$OUT/stages_${id}_run${run}.txt" 2>&1
    grep -E "RESULT|multiset|validation" "$OUT/stages_${id}_run${run}.txt" | tr '\n' ' '; echo
  done
done
