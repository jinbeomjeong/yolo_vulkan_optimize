#!/bin/bash
# NPU latency: base vs --input-copy, new (HEAD) vs old (pre-c97c274) binary, O2 display off, 3 interleaved rounds.
# Then a diagnostic phase: strace hailo map ioctl counts/durations and a dma_buf bufinfo snapshot.
# Must be started under: flock <scratchpad>/hw.lock bash run_npu_lat.sh
set -u
R=/home/pi/workspace/yolo_vulkan_optimize
SP=/tmp/claude-1000/-home-pi-workspace-yolo-vulkan-optimize/e4a7ad62-ae73-489e-9893-56a55ad8454a/scratchpad
OUT=$SP/npu_lat
NEW=$R/build/bin/yvo_pipeline
OLD=$SP/baseline_bin/yvo_pipeline
V=$R/04_data/videos/video_h265.mp4
PY=/home/pi/miniconda3/envs/hailo-yolo/bin/python
# sudo: the caller sets SUDO_ASKPASS (sudo -A); the credential source is not named here
GOV=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
COMMON=(--video "$V" --mode dmabuf --decoder hw --batch 1 --slots 1 --input-pace source --input-queue 1
        --layout nhcw --postproc gpu --nms gpu --display off)
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
    echo "bin: $1 md5 $(md5sum < "$1" | cut -d' ' -f1)"
    echo "pcie: $(cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed) x$(cat /sys/bus/pci/devices/0001:01:00.0/current_link_width)"
    echo "hailo_pre: $H_PRE C  soc_pre: $S_PRE C  cool_wait_s: $COOL_WAIT"
    echo "governor: $(cat $GOV)  cur_khz $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)"
    vcgencmd get_throttled; vcgencmd measure_clock v3d
    echo "git HEAD: $(git -C $R rev-parse HEAD)"
    echo "args: ${COMMON[*]} $2"
  } > env.txt
}

echo "== phase 1: latency runs $(date -Iseconds)"
for run in 1 2 3; do
  for bin in new old; do
    B=$NEW; [[ $bin == old ]] && B=$OLD
    for cfg in base input_copy; do
      extra=""; [[ $cfg == input_copy ]] && extra="--input-copy"
      d=$OUT/raw/${bin}_${cfg}_run$run; mkdir -p "$d"; cd "$d" || exit 1
      rm -f hailort.log
      cool; env_snap "$B" "$extra"
      timeout -s KILL 180s "$B" "${COMMON[@]}" $extra > stdout.txt 2> stderr.txt; rc=$?
      w=$(grep -ci '\[warning\]' hailort.log 2>/dev/null); e=$(grep -ci '\[error\]' hailort.log 2>/dev/null)
      echo "exit $rc hailort_warn ${w:-0} err ${e:-0} hailo_post $(hailo_temp) soc_post $(soc_temp)" >> env.txt
      echo "$bin $cfg run$run rc=$rc $(grep 'npu latency' stdout.txt) warn=${w:-0} err=${e:-0} cool=${COOL_WAIT}s h=$H_PRE s=$S_PRE"
    done
  done
done

echo "== phase 2: diagnostics $(date -Iseconds)"
D=$OUT/raw/_diag; mkdir -p "$D"
for bin in old new; do
  B=$NEW; [[ $bin == old ]] && B=$OLD
  for cfg in base input_copy; do
    extra=""; [[ $cfg == input_copy ]] && extra="--input-copy"
    cd "$D" || exit 1
    cool
    strace -f -T -e trace=ioctl -o strace_${bin}_${cfg}.txt "$B" "${COMMON[@]}" $extra --frames 300 \
      > strace_${bin}_${cfg}.stdout 2>&1
    echo "strace $bin $cfg rc=$?"
  done
done
# bufinfo snapshot during a run (size, exporter, attached devices of every dma-buf)
for bin in new; do
  for cfg in base input_copy; do
    extra=""; [[ $cfg == input_copy ]] && extra="--input-copy"
    cd "$D" || exit 1
    cool
    "$NEW" "${COMMON[@]}" $extra --frames 300 > bufinfo_${cfg}.stdout 2>&1 &
    p=$!; sleep 6
    sudo -A cat /sys/kernel/debug/dma_buf/bufinfo > bufinfo_${cfg}.txt 2>&1
    wait $p; echo "bufinfo $cfg rc=$?"
  done
done
echo "== done $(date -Iseconds)"
