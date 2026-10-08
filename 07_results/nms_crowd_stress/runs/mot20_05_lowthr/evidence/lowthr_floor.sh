#!/usr/bin/env bash
# Supplementary pipeline runs: lower --score on mot20_05 so candidates cross kMaxSelected (1024)
# and the CPU fallback runs on the main thread. b8s16, --nms gpu, display off, 900 frames x 3 runs, cooled.
set -u
P=/home/pi/workspace/yolo_vulkan_optimize
O=$P/07_results/nms_crowd_stress/runs/mot20_05_lowthr
PY=/home/pi/miniconda3/envs/hailo-yolo/bin/python
# folder exists: appending the floor threshold to the same run folder
mkdir -p $O/raw
echo $$ > $P/build/.bench_running; trap 'rm -f $P/build/.bench_running' EXIT
temp() { $PY -c "from hailo_platform import Device; d=Device(); t=d.control.get_chip_temperature(); print(f'{max(t.ts0_temperature,t.ts1_temperature):.1f}'); d.release()" 2>/dev/null; }
{ echo "# nms_crowd_stress lowthr $(date -Is)"; echo "git $(git -C $P rev-parse --short HEAD)"; uname -r
  echo "pcie $(cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed) x$(cat /sys/bus/pci/devices/0001:01:00.0/current_link_width)"
  vcgencmd measure_temp; vcgencmd get_throttled; vcgencmd measure_clock arm; echo "v3d_freq $(vcgencmd get_config v3d_freq)"
  echo "hailo $(temp) C"; echo "video 04_data/videos/mot20_05_h265.mp4"; } > /dev/null
BASE="--video $P/04_data/videos/mot20_05_h265.mp4 --mode dmabuf --layout nhcw --postproc gpu --postproc-alloc export --gpu-depth 3 --display off --batch 8 --slots 16 --nms gpu --frames 900"
for run in 1 2 3; do
  for thr in 0.003; do
    id="b8s16_gpu_thr$thr"; d=$O/raw/${id}_run$run; mkdir -p $d
    w=0; while :; do t=$(temp); awk -v t="$t" 'BEGIN{exit !(t<=65)}' && break; [[ $w -ge 120 ]] && break; sleep 5; w=$((w+5)); done
    echo "[$(date +%T)] run $run $id (hailo $t C, cooled ${w}s)"
    ( cd $d && timeout -s KILL 120 $P/build/bin/yvo_pipeline $BASE --score $thr > stdout.log 2>&1; echo "rc=$?" >> stdout.log )
    warn=$(grep -E '\[warning\]' $d/hailort.log 2>/dev/null | grep -cE '[Tt]emperature|[Tt]hrottl|[Oo]verheat|clock')
    grep -E "^wall|^cpu |^candidates|^nms fallback|^gpu nms|^cpu nms|^detections" $d/stdout.log | tr '\n' ' '; echo " hailo_warn=$warn"
  done
done
