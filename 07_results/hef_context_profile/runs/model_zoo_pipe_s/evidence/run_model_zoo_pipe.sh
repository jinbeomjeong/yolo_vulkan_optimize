#!/usr/bin/env bash
# Model Zoo n/s vs m inside yvo_pipeline at O2 (display off): technique_ablation_bench.sh ONLY='^display_off$', HEF per model.
# The sudo credential source is passed by the caller (SUDO_ASKPASS); this script does not name it.
set -u
cd /home/pi/workspace/yolo_vulkan_optimize
SP=/tmp/claude-1000/-home-pi-workspace-yolo-vulkan-optimize/e4a7ad62-ae73-489e-9893-56a55ad8454a/scratchpad
gov=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
old=$(cat $gov)
restore() { sudo -A sh -c "echo $old > $gov"; echo "cleanup: governor $(cat $gov)"; }
trap restore EXIT
sudo -A sh -c "echo performance > $gov"; echo "governor $(cat $gov)"
for m in n s m; do
  flock $SP/hw.lock env HEF=$PWD/03_models/yolov11$m.hef ONLY='^display_off$' RUNS=3 TIMEOUT_S=180 \
    BIN_PIPELINE=$SP/sd_bin/yvo_pipeline OUT_DIR=$PWD/07_results/hef_context_profile/runs/model_zoo_pipe_$m \
    bash 06_tools/bench/technique_ablation_bench.sh
  echo "exit model $m: $?"
done
