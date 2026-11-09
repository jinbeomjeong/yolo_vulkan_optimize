#!/usr/bin/env bash
# 40-minute soak of the O2 operating point with the display on (follow-up 4, 2026-11-09).
# thermal_soak (2026-10-11) ran b2s4/b8s16 with the display off; O2 (b1/s1, paced input, queue 1) and the display path
# were never run for longer than ~1 minute. This runs O2 exactly as technique_ablation `base` (Vulkan composite
# fullscreen, matched, display pace source, recording on) for SOAK_MIN minutes on a looped copy of video_h265.mp4.
#
# Same structure as 07_results/thermal_soak/evidence/run_thermal_soak.sh:
#   - every [progress] line (every 300 processed frames, cumulative FPS + Hailo chip temp from the pipeline's own
#     VDevice) is prefixed with wall-clock time, SoC temp and throttled bits; --quiet is NOT passed.
#   - hailort.log temperature/clock warnings are the signal looked for, so they are counted, not used to discard the run.
#   - SIGINT (timeout -s INT) is the pipeline's graceful stop; -k gives a hard kill if it hangs (display path).
#   - SoC >= SOC_STOP_C is a safety stop (pkill -INT -x yvo_pipeline).
#   - --frame-log keeps every frame's timing (written once at the end) for latency drift over time.
# Start under: flock <hw.lock> env SUDO_ASKPASS=<askpass helper> OUT=<dir> BIN=<yvo_pipeline> LOOP_VIDEO=<file> bash run_o2_soak.sh
# (the sudo credential source is passed by the caller; this script does not name it)
set -u
R=/home/pi/workspace/yolo_vulkan_optimize
OUT=${OUT:?OUT required}; BIN=${BIN:?BIN required}; LOOP_VIDEO=${LOOP_VIDEO:?LOOP_VIDEO required}
H=$R/03_models/yolov11m.hef
PY=/home/pi/miniconda3/envs/hailo-yolo/bin/python
GOV=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
SOAK_MIN=${SOAK_MIN:-40}; SOC_STOP_C=${SOC_STOP_C:-85}; CONNECTOR=${CONNECTOR:-HDMI-A-1}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
ARGS=(--video "$LOOP_VIDEO" --hef "$H" --mode dmabuf --decoder hw --batch 1 --slots 1 --input-pace source --input-queue 1
      --layout nhcw --postproc gpu --nms gpu --display on --display-backend wayland --display-fullscreen
      --display-sync matched --display-pace source --dump-dir . --dump-every 100000000 --frame-log frame_log.csv
      --csv-label o2_soak)
LOCK=$R/build/.bench_running
rd=$OUT/raw/o2_display_run1
mkdir -p "$OUT/raw/_session" "$rd"

ORIG_GOV=$(cat $GOV)
cleanup() { sudo -A sh -c "echo $ORIG_GOV > $GOV"; rm -f "$LOCK"; echo "cleanup: governor $(cat $GOV), lock removed"; }
trap cleanup EXIT INT TERM
echo $$ > "$LOCK"
sudo -A sh -c "echo performance > $GOV"

hailo_temp() {
  (cd "$OUT/raw/_session" && "$PY" - <<'PYEOF' 2>/dev/null) || echo nan
from hailo_platform import Device
d = Device()
t = d.control.get_chip_temperature()
print(f"{max(t.ts0_temperature, t.ts1_temperature):.1f}")
d.release()
PYEOF
}
soc_temp() { awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }
throttled() { vcgencmd get_throttled 2>/dev/null; }
output_enabled() {
  timeout 5 wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '/^[^ \t]/ { inb = ($1 == c) } inb && /Enabled:/ { ok = ($2 == "yes") } END { exit !ok }'
}
t0=$SECONDS
while :; do
  h=$(hailo_temp); s=$(soc_temp)
  awk -v h="$h" -v s="$s" 'BEGIN{exit !(h!="nan" && h<=65 && s<=75)}' && break
  (( SECONDS - t0 > 900 )) && { echo "cool timeout h=$h s=$s"; break; }
  sleep 10
done
{
  date -Iseconds; uname -r
  echo "bin: $BIN md5 $(md5sum < "$BIN" | cut -d' ' -f1)"
  echo "git HEAD: $(git -C $R rev-parse HEAD)"
  echo "pcie: $(cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed) x$(cat /sys/bus/pci/devices/0001:01:00.0/current_link_width)"
  vcgencmd get_config v3d_freq; throttled
  echo "governor: $(cat $GOV) (was $ORIG_GOV)"
  echo "labwc: $(dpkg-query -W -f='${Version}' labwc 2>/dev/null)  hailort: $(dpkg-query -W -f='${Version}' hailort 2>/dev/null)"
  timeout 5 wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '/^[^ \t]/ { inb = ($1 == c) } inb && /current/ { print "mode: " $0; exit }'
  echo "loop video: $LOOP_VIDEO $(stat -c %s "$LOOP_VIDEO") B (ffmpeg -stream_loop 41 -c copy of 04_data/videos/video_h265.mp4)"
  echo "soak: ${SOAK_MIN} min (timeout -s INT -k 120), SoC safety stop ${SOC_STOP_C} C, YVO_DEC_THREADS=${YVO_DEC_THREADS:-<app default 1>}"
  echo "pre: hailo $h C soc $s C cool_wait_s $((SECONDS - t0))"
  echo "args: ${ARGS[*]}"
} > "$OUT/env_bench.txt"
output_enabled || { echo "$CONNECTOR not enabled — stop"; exit 1; }

echo "[$(date +%T)] start o2_display soak=${SOAK_MIN}min hailo $h C soc $s C"
rc=0
(cd "$rd" && exec timeout -s INT -k 120 "${SOAK_MIN}m" "$BIN" "${ARGS[@]}") 2>&1 \
| while IFS= read -r line; do
    now=$(date +%s.%N)
    if [[ "$line" == "[progress]"* ]]; then
      soc=$(soc_temp)
      printf '%s soc=%s %s %s\n' "$now" "$soc" "$(throttled)" "$line"
      if awk -v t="$soc" -v c="$SOC_STOP_C" 'BEGIN{exit !(t!="nan" && t>=c)}'; then
        printf '%s [soak] SoC %s C >= %s C — safety stop\n' "$now" "$soc" "$SOC_STOP_C"
        pkill -INT -x yvo_pipeline || true
      fi
    else
      printf '%s %s\n' "$now" "$line"
    fi
  done > "$rd/stdout.log"
rc=${PIPESTATUS[0]}
w=$(grep -ciE '\[warning\]' "$rd/hailort.log" 2>/dev/null); e=$(grep -ciE '\[error\]' "$rd/hailort.log" 2>/dev/null)
echo "[$(date +%T)] end o2_display rc=$rc hailort warnings=${w:-0} errors=${e:-0} soc $(soc_temp) C $(throttled)"
echo "post: hailo $(hailo_temp) C soc $(soc_temp) C rc $rc warn ${w:-0} err ${e:-0} $(throttled)" >> "$OUT/env_bench.txt"
