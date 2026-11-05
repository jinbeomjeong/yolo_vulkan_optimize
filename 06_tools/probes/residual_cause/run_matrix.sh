#!/usr/bin/env bash
# Interleaved timing runs of the residual probe plus the unmodified tool as a control. Output only under 11_logs.
# Env: OUT (dir, required), RUNS, VARIANTS (space separated), COUNTER_SETS ("" = timing only).
# Governor handling is in the wrapper gov_wrap.sh, which sets performance and restores ondemand.
set -euo pipefail
PROJ=/home/pi/workspace/yolo_vulkan_optimize
source "$PROJ/06_tools/bench/lib/common.sh"
D=$PROJ/11_logs/2026-11-06_residual_cause
OUT="${OUT:?}"
case "$OUT" in "$D"/*) ;; *) echo "OUT must be under $D" >&2; exit 2 ;; esac
RUNS="${RUNS:-5}"
read -r -a VARS <<< "${VARIANTS:?}"
COUNTER_SETS="${COUNTER_SETS:-}"
COOL_C=70
gov_now() { cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor | sort -u | tr '\n' ' ' | sed 's/ $//'; }
[[ "$(gov_now)" == performance ]] || { echo "governor '$(gov_now)' - not measuring" >&2; exit 1; }
bench_lock_acquire
mkdir -p "$OUT"
{
  echo "date: $(date -Is)"; echo "soc: $(soc_temp) $(throttled)"; echo "arm: $(vcgencmd measure_clock arm)"
  echo "tool md5: $(md5sum "$PROJ/build/bin/yvo_preproc_bench" | cut -d' ' -f1)"
  echo "probe md5: $(md5sum "$D/bin/residual_probe" | cut -d' ' -f1)"
  echo "git: $(git -C "$PROJ" rev-parse HEAD) dirty: $(git -C "$PROJ" status --short | wc -l)"
  echo "variants: ${VARS[*]}  runs: $RUNS  counters: $COUNTER_SETS"
} > "$OUT/env.txt"
nv=${#VARS[@]}
for ((run = 1; run <= RUNS; run++)); do
  for ((i = 0; i < nv; i++)); do
    v=${VARS[$(( (i + (run - 1) * 7) % nv ))]}   # rotate the start so a variant does not always follow the same one
    for cs in ${COUNTER_SETS:-0}; do
      dir="$OUT/${v}_c${cs}_run${run}"
      mkdir -p "$dir"
      while awk -v t="$(soc_temp)" -v c="$COOL_C" 'BEGIN{exit !(t > c)}'; do sleep 10; done
      while (( $(( 16#$(vcgencmd get_throttled | sed -E 's/.*=0x([0-9a-fA-F]+)/\1/') & 0xF )) != 0 )); do sleep 10; done
      t_pre=$(soc_temp)
      rc=0
      case "$v" in
        tool_ref)   (cd "$dir" && taskset -c 3 "$PROJ/build/bin/yvo_preproc_bench" --impl cpuref --frames 600 --warmup 60 --frame-csv frames.csv) > "$dir/stdout.log" 2>&1 || rc=$? ;;
        tool_split) (cd "$dir" && taskset -c 3 "$PROJ/build/bin/yvo_preproc_bench" --impl cpuref-split --frames 600 --warmup 60 --frame-csv frames.csv) > "$dir/stdout.log" 2>&1 || rc=$? ;;
        *)
          cargs=(); [[ "$cs" != 0 ]] && cargs=(--counters "$cs")
          (cd "$dir" && taskset -c 3 "$D/bin/residual_probe" --variant "$v" --frames 600 --warmup 60 "${cargs[@]}") > "$dir/stdout.log" 2>&1 || rc=$? ;;
      esac
      echo "soc_pre $t_pre soc_post $(soc_temp) throttled $(throttled) governor $(gov_now) rc $rc run $run" > "$dir/run_env.txt"
      echo "$(date +%T) $v c$cs run$run rc=$rc soc=$t_pre $(grep -E '^\[RESULT\]' "$dir/stdout.log" | cut -c1-120 || true)"
      [[ -s "$dir/hailort.log" ]] || rm -f "$dir/hailort.log"
    done
  done
done
echo "soc end: $(soc_temp) $(throttled)" >> "$OUT/env.txt"
echo finished
