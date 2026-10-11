#!/usr/bin/env bash
# Thermal soak (07_results/thermal_soak). Every other measurement in this repo is 900 frames (~26 s), so nothing
# says whether the operating points hold once the Hailo-8 heats up: at 90 C class its clock drops 400 -> 250 MHz,
# which could erase b2/s4's margin (34.99 fps vs a 30 fps source). This runs each config continuously for SOAK_MIN
# minutes on a looped crowd clip (worst GPU post-processing load) and logs, per 300-frame [progress] line:
#   wall-clock time, cumulative FPS, Hailo chip temp, SoC temp, throttled bits.
#
# Why not the usual bench reliability rule: measure skill §4 discards a run whose hailort.log has temperature/clock
# warnings. Here those warnings are the signal being looked for, so they are counted and time-stamped, not used
# to throw the run away.
#
# Three things this driver must get right:
#   - the [progress] FPS is CUMULATIVE (frames / elapsed since start), so a late slowdown is diluted. Interval FPS
#     is recomputed afterwards as 300 / (delta wall-clock between consecutive [progress] lines).
#   - [progress] is suppressed by --quiet, so --quiet is NOT passed.
#   - hailo_temp() from common.sh opens a hailo_platform Device(): it is only used while the device is free
#     (cool-down before a run). During a run the chip temperature comes from the pipeline's own [progress] line,
#     which reads it through the pipeline's existing VDevice handle (stage->chip_temperature()).
#
# The bench lock is taken so npu_guard blocks builds/tests for the whole soak (yvo_pipeline alone takes no lock).
# Each run executes with its run folder as CWD so hailort.log lands there, never in the repo root (STRUCTURE.md §3.5).
set -euo pipefail
PROJ=$(cd "$(dirname "$0")/../../.." && pwd)
OUT_DIR="$PROJ/07_results/thermal_soak"
COOL_C=${COOL_C:-65}
COOL_MAX_S=${COOL_MAX_S:-600}
SOAK_MIN=${SOAK_MIN:-40}
SOC_STOP_C=${SOC_STOP_C:-85}          # safety stop: the Pi itself throttles in this band
LOOP_VIDEO=${LOOP_VIDEO:-$PROJ/build/out/thermal_soak/mot20_05_loop.mp4}
HEF="$PROJ/03_models/yolov11m.hef"
# shellcheck source=../../../06_tools/bench/lib/common.sh
source "$PROJ/06_tools/bench/lib/common.sh"

[[ -x "$BIN_DIR/yvo_pipeline" ]] || { echo "missing $BIN_DIR/yvo_pipeline — build first" >&2; exit 1; }
[[ -f "$LOOP_VIDEO" ]] || { echo "missing $LOOP_VIDEO — make it with ffmpeg -stream_loop (see summary.md)" >&2; exit 1; }

bench_lock_acquire

# Same BASE as 06_tools/bench/nms_bench.sh so the operating points are the ones measured there.
BASE=(--video "$LOOP_VIDEO" --hef "$HEF" --mode dmabuf --layout nhcw --postproc gpu --postproc-alloc export
      --gpu-depth 3 --display off --nms gpu)
configs=("b2s4|2|4" "b8s16|8|16")

for c in "${configs[@]}"; do
  IFS='|' read -r id batch slots <<<"$c"
  rd="$OUT_DIR/raw/${id}_run1"
  mkdir -p "$rd"
  cool_down "before $id" hailo       # device is free here, so hailo_temp() is safe
  echo "[$(date +%T)] start $id  batch=$batch slots=$slots  soak=${SOAK_MIN}min  cooled ${COOL_WAITED}s  soc $(soc_temp) C"
  rc=0
  # SIGINT is the pipeline's graceful stop (handler sets a flag, normal shutdown order, exit 128+2).
  (cd "$rd" && exec timeout -s INT "${SOAK_MIN}m" env -u YVO_DEC_THREADS "$BIN_DIR/yvo_pipeline" \
      "${BASE[@]}" --batch "$batch" --slots "$slots" --csv-label "$id") 2>&1 \
  | while IFS= read -r line; do
      now=$(date +%s.%N)
      if [[ "$line" == "[progress]"* ]]; then
        soc=$(soc_temp)
        printf '%s soc=%s %s %s\n' "$now" "$soc" "$(throttled)" "$line"
        if awk -v t="$soc" -v c="$SOC_STOP_C" 'BEGIN{exit !(t!="nan" && t>=c)}'; then
          printf '%s [soak] SoC %s C >= %s C — safety stop\n' "$now" "$soc" "$SOC_STOP_C"
          pkill -INT -x yvo_pipeline || true     # -x, never -f (it would match this shell)
        fi
      else
        printf '%s %s\n' "$now" "$line"
      fi
    done > "$rd/stdout.log" || rc=$?
  read -r nwarn nerr < <(count_hailort_warnings "$rd/hailort.log")
  echo "[$(date +%T)] end   $id  rc=$rc  hailort warnings=$nwarn errors=$nerr  soc $(soc_temp) C  $(throttled)"
done
