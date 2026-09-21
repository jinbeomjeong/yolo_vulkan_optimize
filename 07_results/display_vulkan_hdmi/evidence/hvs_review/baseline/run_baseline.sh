#!/bin/bash
# 기준선: yvo_pipeline --display on (Wayland fullscreen), 900 프레임. gpu_stats·labwc CPU·debugfs state 를 함께 기록.
set -u
cd /home/pi/workspace/yvo_pipeline_port
E=docs/hvs_review/evidence/baseline
tag=$1; shift
export XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0
LPID=$(pgrep -x labwc | head -1)
cat /sys/class/drm/renderD128/device/gpu_stats > $E/${tag}_gpu_stats_before.txt
awk '{print $14+$15}' /proc/$LPID/stat > $E/${tag}_labwc_ticks_before.txt
t0=$(date +%s.%N)
./build/yvo_pipeline --video 04_data/videos/video_h265.mp4 --hef 03_models/yolov11m.hef --frames 900 "$@" > $E/${tag}_stdout.log 2>&1 &
P=$!
sleep 14
cp /sys/kernel/debug/dri/1/state $E/${tag}_state_mid.txt
wait $P
t1=$(date +%s.%N)
cat /sys/class/drm/renderD128/device/gpu_stats > $E/${tag}_gpu_stats_after.txt
awk '{print $14+$15}' /proc/$LPID/stat > $E/${tag}_labwc_ticks_after.txt
echo "wall_s $(echo "$t1 - $t0" | bc) clk_tck $(getconf CLK_TCK) labwc_pid $LPID" > $E/${tag}_meta.txt
