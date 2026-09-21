#!/usr/bin/env bash
# Preprocess-stage performance: PiSP BE (ISP) path vs Vulkan shader path vs decode alone
#   Does not use the NPU (preprocess stage only). The full pipeline is 06_tools/bench/pipeline_bench.sh.
#
# Usage:
#   bash 06_tools/bench/isp_bench.sh                       # all (1816 frames), 3 runs per config
#   RUNS=1 FRAMES=300 bash 06_tools/bench/isp_bench.sh     # short
#   ONLY='isp_' bash 06_tools/bench/isp_bench.sh           # subset by config id (grep -E pattern)
#   OUT_DIR=/path bash ...                             # output directory
#
# There are two time axes. Do not mix them up.
#   wall   : host wall-clock. For ISP, right before/after BackendDevice::Run() (synchronous).
#   device : GPU device time from Vulkan timestamp queries. Excludes host submit/wait cost.
# So the GPU is measured in two configs: depth 2 (pipelined, throughput) and depth 1 (synchronous, for wall-clock comparison).
#
# Run order: repetition (run) is the outer loop, config the inner loop (spreads thermal bias evenly across configs).
# Cool-down: this stage doesn't use the NPU, so it keys on SoC temperature. If SoC temp exceeds HOT_C right before a run,
#       wait until it is at or below COOL_C (at most COOL_MAX_S s). Hailo chip temp is read only for snapshots.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
# They used to be copy-pasted into 7 scripts and had actually diverged (e.g. pcie_link's Gen5 branch).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ISP_BIN="${ISP_BIN:-$BIN_DIR/yvo_isp_bench}"
GPU_BIN="${GPU_BIN:-$BIN_DIR/yvo_preproc_bench}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/isp_resize}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"
WARMUP="${WARMUP:-60}"
SLOTS="${SLOTS:-4}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-66}"   # Pi 5 (CM5, active cooler) idles at SoC 65-66 C. Set lower and it will never get there.
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-72}"
APPEND="${APPEND:-0}"

require_bin "$ISP_BIN" yvo_isp_bench
require_bin "$GPU_BIN" yvo_preproc_bench
check_only_exit
guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw" "$OUT_DIR/samples"

# id|tool|args...
configs=(
  "decode_only|isp|--no-isp"
  "gpu_nhcw_d2|gpu|--layout nhcw --depth 2"
  "gpu_nhcw_d1|gpu|--layout nhcw --depth 1"
  "isp_A_nhwc|isp|--repack none"
  "isp_C_repack_cpu|isp|--repack cpu"
  "isp_C_repack_neon|isp|--repack neon"
)


{
  echo "# isp_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "model: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)  (0x80000 = past-event bit, not currently throttling)"
  vcgencmd measure_clock arm 2>/dev/null || true
  vcgencmd measure_clock v3d 2>/dev/null || true
  pcie_link
  echo "libpisp: $(pkg-config --modversion libpisp 2>/dev/null || echo NA)"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS warmup: $WARMUP slots: $SLOTS"
  echo "note: this measurement does not use the NPU (preprocess stage alone). The hailort warning column is always 0."
} > "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
CSV_HEADER="label,tool,variant,time_axis,frames,wall_s,fps,cpu_ms_per_frame,pre_ms_median,pre_ms_p90,pre_ms_mean,pre_ms_min,pre_ms_max,host_submit_us_median,host_wait_us_median,repack_ms_median,repack_ms_p90,sync_ms_median,n,ok,run,cool_wait_s,soc_t0,soc_t1,throttled,hailort_warnings,hailort_errors"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id tool extra <<<"$c"
    echo "$id" | grep -Eq "$ONLY" || continue
    COOL_WAITED=0
    t_now=$(soc_temp)
    if awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then
      cool_down "SoC temp ${t_now} C > ${HOT_C} C" soc   # this bench does not use the NPU
    fi
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    soc_t0=$(soc_temp)
    echo "[$(date +%T)] run $r/$RUNS $id (soc ${soc_t0} C)"
    set +e
    if [[ "$tool" == "isp" ]]; then
      # shellcheck disable=SC2086
      (cd "$run_dir" && "$ISP_BIN" --video "$VIDEO" --slots "$SLOTS" --warmup "$WARMUP" \
          "${frames_arg[@]}" $extra --no-png --csv-label "$id" > stdout.log 2>&1)
    else
      # The GPU has no warmup flag. Run with --repeat 2 and use the second (warmed-up) block.
      # shellcheck disable=SC2086
      (cd "$run_dir" && "$GPU_BIN" --video "$VIDEO" --slots 8 --repeat 2 \
          "${frames_arg[@]}" $extra --png-dir "$OUT_DIR/samples" --png-frame 100 > stdout.log 2>&1)
    fi
    rc=$?
    set -e
    soc_t1=$(soc_temp)
    thr=$(throttled)
    nwarn=0; nerr=0
    if [[ -f "$run_dir/hailort.log" ]]; then
      nwarn=$(grep -E '\[warning\]' "$run_dir/hailort.log" | grep -cE "$WARN_RE" || true)
      nerr=$(grep -c '\[error\]' "$run_dir/hailort.log" || true)
    fi

    if [[ "$tool" == "isp" ]]; then
      line=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
      if [[ -z "$line" ]]; then
        echo "  failed rc=$rc (see stdout.log)"; tail -5 "$run_dir/stdout.log"; continue
      fi
      # RESULT_CSV: label,mode,repack,slots,frames,wall_s,fps,cpu,med,p90,mean,min,max,rep_med,rep_p90,sync_med,n,ok
      IFS=',' read -r _lab mode repack _slots nfr wall fps cpu med p90 mean mn mx rmed rp90 smed nn ok <<<"$line"
      variant="$mode/$repack"
      echo "$id,isp,$variant,wall,$nfr,$wall,$fps,$cpu,$med,$p90,$mean,$mn,$mx,,,$rmed,$rp90,$smed,$nn,$ok,$r,$COOL_WAITED,$soc_t0,$soc_t1,$thr,$nwarn,$nerr" >> "$RAW"
      echo "  fps=$fps cpu=$cpu ms/f isp_median=$med ms repack_median=${rmed:-} rc=$rc"
    else
      # Parse preproc_bench's last [decode+gpu] block (the second of --repeat 2 = after warmup).
      mapfile -t gl < <(grep '^\[decode+gpu\]' "$run_dir/stdout.log")
      if (( ${#gl[@]} < 3 )); then
        echo "  failed rc=$rc (see stdout.log)"; tail -5 "$run_dir/stdout.log"; continue
      fi
      # last 3 lines = frames/gpu/host lines of the final repetition
      l_fr="${gl[-3]}"; l_gpu="${gl[-2]}"; l_host="${gl[-1]}"
      nfr=$(sed -E 's/.*frames=([0-9]+).*/\1/' <<<"$l_fr")
      wall=$(sed -E 's/.*wall=([0-9.]+) s.*/\1/' <<<"$l_fr")
      fps=$(sed -E 's/.*fps=([0-9.]+).*/\1/' <<<"$l_fr")
      cpu=$(sed -E 's/.*cpu=([0-9.]+) ms\/frame.*/\1/' <<<"$l_fr")
      med=$(sed -E 's/.*median=([0-9.]+).*/\1/' <<<"$l_gpu")
      p90=$(sed -E 's/.*p90=([0-9.]+).*/\1/' <<<"$l_gpu")
      mean=$(sed -E 's/.*mean=([0-9.]+).*/\1/' <<<"$l_gpu")
      mn=$(sed -E 's/.*min=([0-9.]+).*/\1/' <<<"$l_gpu")
      mx=$(sed -E 's/.*max=([0-9.]+) .*/\1/' <<<"$l_gpu")
      nn=$(sed -E 's/.*\(n=([0-9]+)\).*/\1/' <<<"$l_gpu")
      sub=$(sed -E 's/.*submit us: median=([0-9.]+).*/\1/' <<<"$l_host")
      wt=$(sed -E 's/.*wait us: median=([0-9.]+).*/\1/' <<<"$l_host")
      ok=$([[ $rc -eq 0 ]] && echo 1 || echo 0)
      echo "$id,gpu,$extra,device,$nfr,$wall,$fps,$cpu,$med,$p90,$mean,$mn,$mx,$sub,$wt,,,,$nn,$ok,$r,$COOL_WAITED,$soc_t0,$soc_t1,$thr,$nwarn,$nerr" >> "$RAW"
      echo "  fps=$fps cpu=$cpu ms/f gpu_device_median=$med ms submit+wait=$(awk -v a="$sub" -v b="$wt" 'BEGIN{printf "%.3f", (a+b)/1000}') ms rc=$rc"
    fi
  done
done

# Per-config medians
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)
num = ["fps", "wall_s", "cpu_ms_per_frame", "pre_ms_median", "pre_ms_p90", "pre_ms_mean", "pre_ms_min",
       "pre_ms_max", "host_submit_us_median", "host_wait_us_median", "repack_ms_median", "repack_ms_p90",
       "sync_ms_median", "soc_t1"]
def med(g, k):
    vals = [float(x[k]) for x in g if x[k] not in ("", None)]
    return f"{statistics.median(vals):.3f}" if vals else ""
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "tool", "variant", "time_axis", "runs", "frames"] + [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "cpu_ms_min", "cpu_ms_max", "hailort_warnings", "all_ok"])
    for lab in order:
        g = groups[lab]
        fps = [float(x["fps"]) for x in g]
        cpu = [float(x["cpu_ms_per_frame"]) for x in g]
        w.writerow([lab, g[0]["tool"], g[0]["variant"], g[0]["time_axis"], len(g), g[0]["frames"]] +
                   [med(g, k) for k in num] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}", f"{min(cpu):.3f}", f"{max(cpu):.3f}",
                    sum(int(x["hailort_warnings"]) for x in g), int(all(x["ok"] == "1" for x in g))])
print(open(sys.argv[2]).read())
EOF

# Sample PNGs (separate from measurement)
echo "[samples] ISP letterbox PNG (excluded from measurement)"
(cd "$OUT_DIR/samples" && "$ISP_BIN" --video "$VIDEO" --frames 200 --warmup 0 --repack none \
    --png-dir "$OUT_DIR/samples" --png-frame 100 > isp_sample.log 2>&1) || echo "  samples run failed"

{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)"
} >> "$OUT_DIR/env_bench.txt"
