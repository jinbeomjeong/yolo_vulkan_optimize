#!/bin/bash
# Plane-path commit-gate margin sweep at the O2 operating point (b1, paced input queue 1), display on.
# Planes need >= 2 slots with b1 (b1/s1 + planes deadlocks, input_pace_bench.sh), so the plane configs are b1/s2.
# Configs (inner loop, same margin order as display_planes_e2e/runs/vblank_phase_20261027):
#   vk_b1s1        Vulkan composite O2 (control)
#   wlp_b1s2_m{8,4,0,12}  wayland-planes, --display-commit-margin-ms X (gate = 1000/fps - X ms)
# Args otherwise follow input_pace_bench.sh DISPLAY_RUN=1 (DCOMMON, PACE 1, warmup 30, 900 frames) + --frame-log.
# Each run: HDMI connector must be Enabled (wlr-randr), timeout -s KILL, cooling to Hailo <= 65 C and SoC <= 75 C.
# Start under: flock <hw.lock> env SUDO_ASKPASS=<askpass helper> OUT=<dir> BIN=<yvo_pipeline> bash run_o2_gate.sh
# (the sudo credential source is passed by the caller; this script does not name it)
set -u
R=/home/pi/workspace/yolo_vulkan_optimize
OUT=${OUT:?OUT required}
BIN=${BIN:?BIN required}
V=$R/04_data/videos/video_h265.mp4
H=$R/03_models/yolov11m.hef
PY=/home/pi/miniconda3/envs/hailo-yolo/bin/python
GOV=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
FRAMES=${FRAMES:-900}; WARMUP=${WARMUP:-30}; TIMEOUT_S=${TIMEOUT_S:-90}; RUNS=${RUNS:-3}
CONNECTOR=${CONNECTOR:-HDMI-A-1}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
DCOMMON="--mode dmabuf --layout nhcw --postproc gpu --nms gpu --display on --display-fullscreen --display-sync matched --display-pace source --display-warmup $WARMUP"
PACE="--input-pace source --input-queue 1"
configs=(
  "vk_b1s1|--display-backend wayland --batch 1 --slots 1"
  "wlp_b1s2_m8|--display-backend wayland-planes --batch 1 --slots 2 --display-commit-margin-ms 8"
  "wlp_b1s2_m4|--display-backend wayland-planes --batch 1 --slots 2 --display-commit-margin-ms 4"
  "wlp_b1s2_m0|--display-backend wayland-planes --batch 1 --slots 2 --display-commit-margin-ms 0"
  "wlp_b1s2_m12|--display-backend wayland-planes --batch 1 --slots 2 --display-commit-margin-ms 12"
)
LOCK=$R/build/.bench_running
mkdir -p "$OUT/raw/_session"

ORIG_GOV=$(cat $GOV)
cleanup() {
  sudo -A sh -c "echo $ORIG_GOV > $GOV"
  rm -f "$LOCK"
  echo "cleanup: governor $(cat $GOV), lock removed"
}
trap cleanup EXIT INT TERM
echo $$ > "$LOCK"
sudo -A sh -c "echo performance > $GOV"
echo "governor: $(cat $GOV) (was $ORIG_GOV)"

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
output_enabled() {
  timeout 5 wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '/^[^ \t]/ { inb = ($1 == c) } inb && /Enabled:/ { ok = ($2 == "yes") } END { exit !ok }'
}
cool() {
  local t0=$SECONDS h s
  while :; do
    h=$(hailo_temp); s=$(soc_temp)
    if awk -v h="$h" -v s="$s" 'BEGIN{exit !(h!="nan" && h<=65 && s<=75)}'; then break; fi
    (( SECONDS - t0 > 900 )) && { echo "cool timeout h=$h s=$s"; break; }
    sleep 10
  done
  COOL_WAIT=$((SECONDS - t0)); H_PRE=$h; S_PRE=$s
}
{
  date -Iseconds; uname -r
  echo "bin: $BIN md5 $(md5sum < "$BIN" | cut -d' ' -f1)"
  echo "git HEAD: $(git -C $R rev-parse HEAD)"
  echo "pcie: $(cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed) x$(cat /sys/bus/pci/devices/0001:01:00.0/current_link_width)"
  vcgencmd get_config v3d_freq; vcgencmd get_throttled
  echo "governor: $(cat $GOV)"
  echo "labwc: $(dpkg-query -W -f='${Version}' labwc 2>/dev/null)  hailort: $(dpkg-query -W -f='${Version}' hailort 2>/dev/null)"
  timeout 5 wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '/^[^ \t]/ { inb = ($1 == c) } inb && /current/ { print "mode: " $0; exit }'
  echo "video: $V frames $FRAMES warmup $WARMUP runs $RUNS timeout ${TIMEOUT_S}s YVO_DEC_THREADS=${YVO_DEC_THREADS:-<app default 1>}"
  echo "common: $DCOMMON $PACE --frame-log frame_log.csv"
  for c in "${configs[@]}"; do echo "config: $c"; done
} > "$OUT/env_bench.txt"

echo "== runs $(date -Iseconds)"
for run in $(seq 1 "$RUNS"); do
  for c in "${configs[@]}"; do
    id=${c%%|*}; args=${c#*|}
    d=$OUT/raw/${id}_run$run; mkdir -p "$d"; cd "$d" || exit 1
    cool
    if ! output_enabled; then echo "$id run$run SKIP: $CONNECTOR not enabled"; echo "skip no_$CONNECTOR" > note.txt; continue; fi
    echo "date $(date -Iseconds) hailo_pre $H_PRE soc_pre $S_PRE cool_wait_s $COOL_WAIT" > env.txt
    echo "args: --video $V --hef $H --frames $FRAMES $DCOMMON $PACE $args --frame-log frame_log.csv" >> env.txt
    timeout -s KILL "$TIMEOUT_S" "$BIN" --video "$V" --hef "$H" --frames "$FRAMES" $DCOMMON $PACE $args \
      --frame-log frame_log.csv > stdout.log 2> stderr.txt; rc=$?
    w=$(grep -ci '\[warning\]' hailort.log 2>/dev/null); e=$(grep -ci '\[error\]' hailort.log 2>/dev/null)
    echo "exit $rc hailort_warn ${w:-0} err ${e:-0} hailo_post $(hailo_temp) soc_post $(soc_temp)" >> env.txt
    echo "$id run$run rc=$rc warn=${w:-0} err=${e:-0} cool=${COOL_WAIT}s h=$H_PRE s=$S_PRE | $(grep -m1 -E 'late frames' stdout.log | tr -s ' ') | $(grep -m1 -E 'scanout' stdout.log | tr -s ' ')"
  done
done
echo "== done $(date -Iseconds)"
