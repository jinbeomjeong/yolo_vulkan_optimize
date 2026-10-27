#!/usr/bin/env bash
# Formal runs of `yvo_postproc_probe --probe memasym` (CPU<->GPU memory access asymmetry) -> 07_results/memory_asymmetry.
# Lives in probes/ rather than bench/: a new bench/*.sh would also need the npu_guard bench list (C10). It still takes
# the bench lock (build/.bench_running) through bench/lib/common.sh, so npu_guard blocks builds/tests while it runs.
#
# Settings (3 independent runs each; runs are the outer loop, settings the inner loop; CPU pinned to core 3):
#   bw_r256x3  --parts gpubw --bw-local 256 --bw-groups 3   (read-optimal dispatch shape, sweep 2026-10-27)
#   bw_w128x2  --parts gpubw --bw-local 128 --bw-groups 2   (write-optimal dispatch shape)
#   attr       --parts attr                                  (same-line latency, unaligned LDR / DC ZVA)
#   coh        --parts coherence --coh-trials 5              (sync omitted vs kept, 256 KiB / 8 MiB)
#   cpuwrite   --parts cpuwrite                              (memset / memcpy / byte stores, 1,228,800 B)
# Before every run: SoC <= COOL_C (default 70 C, project rule <= 75 C) and get_throttled current bits (0..3) == 0.
# The tracefs/kprobe evidence is a separate root run: sudo bash 06_tools/probes/memasym_trace.sh <OUT_DIR>/evidence/trace
#
# Env: OUT_DIR (default 07_results/memory_asymmetry), RUNS (3), COOL_C (70), COOL_MAX_S (300), ONLY (grep on setting),
#      FORCE=1 to write into a non-empty OUT_DIR, DRYRUN=1 to print the commands only.
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
source "$PROJ/06_tools/bench/lib/common.sh"

OUT_DIR="${OUT_DIR:-$PROJ/07_results/memory_asymmetry}"
RUNS="${RUNS:-3}"
COOL_C="${COOL_C:-70}"
COOL_MAX_S="${COOL_MAX_S:-300}"
ONLY="${ONLY:-}"
BIN="$BIN_DIR/yvo_postproc_probe"
require_bin "$BIN" yvo_postproc_probe
check_only_exit

CONFIGS=(
  "bw_r256x3|--parts gpubw --bw-local 256 --bw-groups 3"
  "bw_w128x2|--parts gpubw --bw-local 128 --bw-groups 2"
  "attr|--parts attr"
  "coh|--parts coherence --coh-trials 5"
  "cpuwrite|--parts cpuwrite"
)
COMMON=(--probe memasym --cpu 3 --rounds 3 --repeats 10 --out-dir "")

if [[ "${DRYRUN:-0}" == "1" ]]; then
  for c in "${CONFIGS[@]}"; do echo "$BIN ${COMMON[*]} ${c#*|}  -> raw/${c%%|*}_run<N>/stdout.log"; done
  exit 0
fi

guard_out_dir
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

throttle_now() { local v; v=$(vcgencmd get_throttled | sed -E 's/.*=0x([0-9a-fA-F]+)/\1/'); echo $(( 16#$v & 0xF )); }
{
  echo "date: $(date -Is)"
  echo "host: $(tr -d '\0' < /proc/device-tree/model)"
  echo "kernel: $(uname -r)"
  pkg_versions mesa-vulkan-drivers libdrm2
  echo "v3d_freq config: $(vcgencmd get_config v3d_freq)"
  echo "clock v3d: $(vcgencmd measure_clock v3d)  arm: $(vcgencmd measure_clock arm)"
  echo "governor(cpu0..3): $(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor | tr '\n' ' ')"
  echo "soc temp: $(soc_temp) C  $(throttled)"
  echo "$(pcie_link)  (NPU not used)"
  echo "CmaTotal/CmaFree: $(awk '/CmaTotal|CmaFree/ {printf "%s %s kB  ", $1, $2}' /proc/meminfo)"
  echo "dma_heap: $(ls /dev/dma_heap | tr '\n' ' ')"
  env_git_head
  env_bin "$BIN"
  echo "configs:"; for c in "${CONFIGS[@]}"; do echo "  ${c%%|*}: ${COMMON[*]} ${c#*|}"; done
  echo "runs: $RUNS  cool: SoC <= ${COOL_C} C (max ${COOL_MAX_S}s) and throttled bits 0..3 == 0 before each run"
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

for ((run = 1; run <= RUNS; run++)); do
  for c in "${CONFIGS[@]}"; do
    cfg="${c%%|*}"; read -r -a args <<< "${c#*|}"
    [[ -n "$ONLY" && ! "$cfg" =~ $ONLY ]] && continue
    dir="$OUT_DIR/raw/${cfg}_run${run}"
    mkdir -p "$dir"
    cool_down "before $cfg run $run" soc
    while (( $(throttle_now) != 0 )); do echo "  throttled now ($(throttled)) - waiting"; sleep 10; done
    t_pre=$(soc_temp)
    echo "== $cfg run $run (SoC ${t_pre} C, cooled ${COOL_WAITED}s)"
    rc=0
    (cd "$dir" && "$BIN" "${COMMON[@]}" "${args[@]}") > "$dir/stdout.log" 2>&1 || rc=$?
    echo "soc_pre ${t_pre} soc_post $(soc_temp) cooled_s ${COOL_WAITED} $(throttled) rc ${rc}" > "$dir/run_env.txt"
    grep -E '^\[RESULT\]|^\[check\]' "$dir/stdout.log" | sed 's/^/   /'
    [[ -s "$dir/hailort.log" ]] || rm -f "$dir/hailort.log"   # 0-byte HailoRT log (NPU not used) carries nothing
  done
done
echo "soc temp after: $(soc_temp) C  $(throttled)" >> "$OUT_DIR/env_bench.txt"
mkdir -p "$OUT_DIR/evidence"
python3 "$PROJ/06_tools/probes/memasym_aggregate.py" "$OUT_DIR" > "$OUT_DIR/evidence/medians.txt"
echo "medians -> $OUT_DIR/evidence/medians.txt"
