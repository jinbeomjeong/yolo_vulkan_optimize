#!/bin/bash
# NPU latency window at b8/s16 (asap, host postproc, display off): YVO_DEC_THREADS=1 vs auto (0), 3 interleaved rounds,
# every run with --frame-log so GPU submit -> NPU done can be compared with the npu_ms column.
# Re-checks 07_results/dec_threads (297 vs 306 ms) against the npu_ms window-start effect found in
# technique_ablation/runs/npu_latency_window.
# Start under: flock <hw.lock> env SUDO_ASKPASS=<askpass helper> OUT=<dir> BIN=<yvo_pipeline> bash run_b8_window.sh
# (the sudo credential source is passed by the caller; this script does not name it)
set -u
R=/home/pi/workspace/yolo_vulkan_optimize
OUT=${OUT:?OUT required}
BIN=${BIN:?BIN required}
V=$R/04_data/videos/video_h265.mp4
H=$R/03_models/yolov11m.hef
PY=/home/pi/miniconda3/envs/hailo-yolo/bin/python
GOV=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
COMMON=(--video "$V" --hef "$H" --input-pace asap --mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host)
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
env_snap() {
  {
    date -Iseconds
    echo "bin: $BIN md5 $(md5sum < "$BIN" | cut -d' ' -f1)"
    echo "pcie: $(cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed) x$(cat /sys/bus/pci/devices/0001:01:00.0/current_link_width)"
    echo "hailo_pre: $H_PRE C  soc_pre: $S_PRE C  cool_wait_s: $COOL_WAIT"
    echo "governor: $(cat $GOV)  cur_khz $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)"
    vcgencmd get_throttled; vcgencmd measure_clock v3d
    echo "git HEAD: $(git -C $R rev-parse HEAD)"
    echo "YVO_DEC_THREADS=$1"
    echo "args: ${COMMON[*]} --frame-log frame_log.csv"
  } > env.txt
}

echo "== latency runs $(date -Iseconds)"
for run in 1 2 3; do
  for cfg in t1 auto; do
    thr=1; [[ $cfg == auto ]] && thr=0
    d=$OUT/raw/${cfg}_run$run; mkdir -p "$d"; cd "$d" || exit 1
    rm -f hailort.log
    cool; env_snap "$thr"
    YVO_DEC_THREADS=$thr timeout -s KILL 180s "$BIN" "${COMMON[@]}" --frame-log frame_log.csv \
      > stdout.log 2> stderr.txt; rc=$?
    w=$(grep -ci '\[warning\]' hailort.log 2>/dev/null); e=$(grep -ci '\[error\]' hailort.log 2>/dev/null)
    echo "exit $rc hailort_warn ${w:-0} err ${e:-0} hailo_post $(hailo_temp) soc_post $(soc_temp)" >> env.txt
    echo "$cfg run$run rc=$rc $(grep -m1 'npu latency' stdout.log) warn=${w:-0} err=${e:-0} cool=${COOL_WAIT}s h=$H_PRE s=$S_PRE"
  done
done
echo "== done $(date -Iseconds)"
