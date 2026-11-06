#!/bin/bash
# Phase 3: per-frame log to test "NPU latency shrinks when the callback -> next submit gap grows".
# new binary, base / input_copy, display off; plus base asap (no pacing gap). Run under flock hw.lock.
set -u
R=/home/pi/workspace/yolo_vulkan_optimize
SP=/tmp/claude-1000/-home-pi-workspace-yolo-vulkan-optimize/e4a7ad62-ae73-489e-9893-56a55ad8454a/scratchpad
OUT=$SP/npu_lat/raw/_gap
NEW=$R/build/bin/yvo_pipeline
V=$R/04_data/videos/video_h265.mp4
PY=/home/pi/miniconda3/envs/hailo-yolo/bin/python
PW=/home/pi/workspace/pwd.txt
GOV=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
LOCK=$R/build/.bench_running
BASEARGS=(--video "$V" --mode dmabuf --decoder hw --batch 1 --slots 1 --input-queue 1 --layout nhcw --postproc gpu --nms gpu --display off --frames 900)
mkdir -p "$OUT"; cd "$OUT" || exit 1
ORIG_GOV=$(cat $GOV)
cleanup() { sudo -S -p '' sh -c "echo $ORIG_GOV > $GOV" < "$PW"; rm -f "$LOCK"; echo "cleanup: governor $(cat $GOV)"; }
trap cleanup EXIT INT TERM
echo $$ > "$LOCK"
sudo -S -p '' sh -c "echo performance > $GOV" < "$PW"
hailo_temp() {
  "$PY" - <<'PYEOF' 2>/dev/null || echo nan
from hailo_platform import Device
d = Device(); t = d.control.get_chip_temperature()
print(f"{max(t.ts0_temperature, t.ts1_temperature):.1f}"); d.release()
PYEOF
}
soc_temp() { awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }
cool() { local t0=$SECONDS h s; while :; do h=$(hailo_temp); s=$(soc_temp)
  awk -v h="$h" -v s="$s" 'BEGIN{exit !(h!="nan" && h<=65 && s<=75)}' && break
  (( SECONDS - t0 > 900 )) && break; sleep 10; done; echo "cool ${h}C soc ${s}C wait $((SECONDS-t0))s"; }
for spec in "src_base|--input-pace source" "src_input_copy|--input-pace source --input-copy" "asap_base|--input-pace asap" "asap_input_copy|--input-pace asap --input-copy"; do
  name=${spec%%|*}; args=${spec#*|}
  cool
  "$NEW" "${BASEARGS[@]}" $args --frame-log "$OUT/$name.csv" > "$OUT/$name.stdout" 2>&1
  echo "$name rc=$? $(grep 'npu latency' $OUT/$name.stdout) $(grep -m1 '^wall' $OUT/$name.stdout)"
done
grep -ci '\[warning\]\|\[error\]' hailort.log 2>/dev/null; echo "== gap done"
