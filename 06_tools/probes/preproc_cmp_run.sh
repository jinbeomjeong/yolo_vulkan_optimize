#!/usr/bin/env bash
# Formal runs of `yvo_preproc_bench --impl` (CPU vs GPU preprocessing under the same conditions)
# -> 07_results/preproc_cpu_vs_gpu. Lives in probes/ rather than bench/: a new bench/*.sh would also need the npu_guard
# bench list (C10). It still takes the bench lock (build/.bench_running) through bench/lib/common.sh.
#
# Same input for every implementation: decoded DRM_PRIME frames of 04_data/videos/video_h265.mp4 (decoder frame
# threading off), FRAMES timed frames after WARMUP frames, output 640x640 NHCW uint8. Decode is not timed.
# Settings (3 independent runs each; runs are the outer loop, settings the inner loop):
#   1-thread CPU settings and gpu are pinned to core 3 (taskset -c 3), 4-thread settings to cores 0-3.
#   CPU outputs go to cached dma-heap buffers unless the setting says malloc.
# Phases: timed runs -> accuracy pass (every frame vs CPU-ref, evidence/accuracy/) -> validation run of the GPU setting
#   (verification/) -> aggregation (results_raw.csv, results.csv, evidence/medians.txt).
# Before every timed run: SoC <= COOL_C (default 70 C, project rule <= 75 C) and get_throttled current bits (0..3) == 0.
# The governor is not changed here (root). Run it under a wrapper that sets it and restores it on exit, and pass
# EXPECT_GOVERNOR=<name> so the script refuses to measure under a different governor.
#
# Env: OUT_DIR (default 07_results/preproc_cpu_vs_gpu), RUNS (3), FRAMES (600), WARMUP (60), COOL_C (70),
#      COOL_MAX_S (300), ONLY (grep on setting), ACC_FRAMES (660), SKIP_ACC=1, SKIP_VALIDATION=1,
#      EXPECT_GOVERNOR, FORCE=1 to write into a non-empty OUT_DIR, DRYRUN=1 to print the commands only.
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJ/06_tools/bench/lib/common.sh"

OUT_DIR="${OUT_DIR:-$PROJ/07_results/preproc_cpu_vs_gpu}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-600}"
WARMUP="${WARMUP:-60}"
COOL_C="${COOL_C:-70}"
COOL_MAX_S="${COOL_MAX_S:-300}"
ONLY="${ONLY:-}"
ACC_FRAMES="${ACC_FRAMES:-660}"
BIN="$BIN_DIR/yvo_preproc_bench"
require_bin "$BIN" yvo_preproc_bench
check_only_exit

# name|cpus|args
CONFIGS=(
  "cpuref_t1|3|--impl cpuref"
  "cpuref_t1_permap|3|--impl cpuref --cpu-map perframe"
  "cpuref_t1_malloc|3|--impl cpuref --cpu-out malloc"
  "cpuref_t4|0-3|--impl cpuref --threads 4 --mem-threads 1"
  "cpuref_t4_mem4|0-3|--impl cpuref --threads 4 --mem-threads 4"
  "sws_rgb24_t1|3|--impl sws-rgb24"
  "sws_rgb24_t1_malloc|3|--impl sws-rgb24 --cpu-out malloc"
  "sws_rgb24_t4|0-3|--impl sws-rgb24 --threads 4 --mem-threads 1"
  "sws_gbrp_t1|3|--impl sws-gbrp"
  "sws_gbrp_t4|0-3|--impl sws-gbrp --threads 4 --mem-threads 1"
  "sws_fastbil_t1|3|--impl sws-rgb24 --sws-flags fast_bilinear"
  "sws_fastbil_t4|0-3|--impl sws-rgb24 --sws-flags fast_bilinear --threads 4 --mem-threads 1"
  "gpu|3|--impl gpu"
)
# Accuracy pass: one per distinct output (name|args). Thread counts are included to show t1 == t4 byte for byte.
ACC=(
  "cpuref_t1|--impl cpuref"
  "cpuref_t4|--impl cpuref --threads 4 --mem-threads 1"
  "cpuref_t4_mem4|--impl cpuref --threads 4 --mem-threads 4"
  "sws_rgb24_t1|--impl sws-rgb24"
  "sws_rgb24_t4|--impl sws-rgb24 --threads 4 --mem-threads 1"
  "sws_gbrp_t1|--impl sws-gbrp"
  "sws_gbrp_t4|--impl sws-gbrp --threads 4 --mem-threads 1"
  "sws_fastbil_t1|--impl sws-rgb24 --sws-flags fast_bilinear"
  "sws_fastbil_t4|--impl sws-rgb24 --sws-flags fast_bilinear --threads 4 --mem-threads 1"
  "gpu|--impl gpu"
)
COMMON=(--frames "$FRAMES" --warmup "$WARMUP")

if [[ "${DRYRUN:-0}" == "1" ]]; then
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r cfg cpus args <<< "$c"
    echo "taskset -c $cpus $BIN $args ${COMMON[*]} --frame-csv frames.csv  -> raw/${cfg}_run<N>/stdout.log"
  done
  for c in "${ACC[@]}"; do echo "$BIN ${c#*|} --accuracy --frames $ACC_FRAMES -> evidence/accuracy/${c%%|*}.log"; done
  exit 0
fi

gov_now() { cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor | sort -u | tr '\n' ' ' | sed 's/ $//'; }
if [[ -n "${EXPECT_GOVERNOR:-}" && "$(gov_now)" != "$EXPECT_GOVERNOR" ]]; then
  echo "governor is '$(gov_now)', expected '$EXPECT_GOVERNOR' - not measuring" >&2
  exit 1
fi

guard_out_dir
bench_lock_acquire
mkdir -p "$OUT_DIR/raw/_session"

throttle_now() { local v; v=$(vcgencmd get_throttled | sed -E 's/.*=0x([0-9a-fA-F]+)/\1/'); echo $(( 16#$v & 0xF )); }
{
  echo "date: $(date -Is)"
  echo "host: $(tr -d '\0' < /proc/device-tree/model)"
  echo "kernel: $(uname -r)"
  pkg_versions mesa-vulkan-drivers libdrm2 libavcodec61 libavutil59 libswscale8
  echo "v3d_freq config: $(vcgencmd get_config v3d_freq)"
  echo "clock v3d: $(vcgencmd measure_clock v3d)  arm: $(vcgencmd measure_clock arm)"
  echo "governor(cpu0..3): $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor | tr '\n' ' ')"
  echo "scaling_max_freq(cpu0): $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
  echo "soc temp: $(soc_temp) C  $(throttled)"
  echo "$(pcie_link)  (NPU not used)"
  echo "dma_heap: $(ls /dev/dma_heap | tr '\n' ' ')"
  env_git_head
  env_bin "$BIN"
  echo "video: $PROJ/04_data/videos/video_h265.mp4 (yvo_preproc_bench default)"
  echo "frames: $FRAMES timed after $WARMUP warm-up frames per run; accuracy pass: first $ACC_FRAMES frames"
  echo "configs:"; for c in "${CONFIGS[@]}"; do IFS='|' read -r cfg cpus args <<< "$c"; echo "  $cfg: taskset -c $cpus $args ${COMMON[*]}"; done
  echo "runs: $RUNS  cool: SoC <= ${COOL_C} C (max ${COOL_MAX_S}s) and throttled bits 0..3 == 0 before each run"
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

for ((run = 1; run <= RUNS; run++)); do
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r cfg cpus argstr <<< "$c"
    read -r -a args <<< "$argstr"
    [[ -n "$ONLY" && ! "$cfg" =~ $ONLY ]] && continue
    dir="$OUT_DIR/raw/${cfg}_run${run}"
    mkdir -p "$dir"
    cool_down "before $cfg run $run" soc
    while (( $(throttle_now) != 0 )); do echo "  throttled now ($(throttled)) - waiting"; sleep 10; done
    t_pre=$(soc_temp)
    echo "== $cfg run $run (SoC ${t_pre} C, cooled ${COOL_WAITED}s)"
    rc=0
    (cd "$dir" && taskset -c "$cpus" "$BIN" "${args[@]}" "${COMMON[@]}" --frame-csv frames.csv) > "$dir/stdout.log" 2>&1 || rc=$?
    echo "soc_pre ${t_pre} soc_post $(soc_temp) cooled_s ${COOL_WAITED} throttled $(throttled | sed "s/throttled=//") governor $(gov_now) cpus ${cpus} rc ${rc}" > "$dir/run_env.txt"
    grep -E '^\[RESULT\]' "$dir/stdout.log" | cut -c1-160 | sed 's/^/   /' || echo "   (no RESULT line, rc $rc)"
    [[ -s "$dir/hailort.log" ]] || rm -f "$dir/hailort.log"   # 0-byte HailoRT log (NPU not used) carries nothing
  done
done
echo "soc temp after timed runs: $(soc_temp) C  $(throttled)" >> "$OUT_DIR/env_bench.txt"

if [[ "${SKIP_ACC:-0}" != "1" ]]; then
  mkdir -p "$OUT_DIR/evidence/accuracy" "$OUT_DIR/samples"
  for c in "${ACC[@]}"; do
    name="${c%%|*}"; read -r -a args <<< "${c#*|}"
    echo "== accuracy $name"
    (cd "$OUT_DIR/raw/_session" && "$BIN" "${args[@]}" --accuracy --frames "$ACC_FRAMES" --png-frame 100 \
       --png-dir "$OUT_DIR/samples") > "$OUT_DIR/evidence/accuracy/${name}.log" 2>&1 || echo "   accuracy $name rc $?"
    grep -E '^\[ACCURACY\]' "$OUT_DIR/evidence/accuracy/${name}.log" | cut -c1-160 | sed 's/^/   /' || true
  done
fi

if [[ "${SKIP_VALIDATION:-0}" != "1" ]]; then
  mkdir -p "$OUT_DIR/verification"
  echo "== validation layer: gpu setting"
  (cd "$OUT_DIR/raw/_session" && VK_LOADER_LAYERS_ENABLE='*validation' "$BIN" --impl gpu --frames 200 --warmup 10) \
    > "$OUT_DIR/verification/validation_impl_gpu_200f.log" 2>&1 || echo "   validation rc $?"
  grep -E '^validation:' "$OUT_DIR/verification/validation_impl_gpu_200f.log" | sed 's/^/   /' || true
fi
[[ -s "$OUT_DIR/raw/_session/hailort.log" ]] || rm -f "$OUT_DIR/raw/_session/hailort.log"
rmdir "$OUT_DIR/raw/_session" 2>/dev/null || true

mkdir -p "$OUT_DIR/evidence"
python3 "$PROJ/06_tools/probes/preproc_cmp_aggregate.py" "$OUT_DIR" > "$OUT_DIR/evidence/medians.txt"
echo "results -> $OUT_DIR/results.csv, medians -> $OUT_DIR/evidence/medians.txt"
