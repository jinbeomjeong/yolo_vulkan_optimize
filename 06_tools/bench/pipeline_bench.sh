#!/usr/bin/env bash
# Integrated pipeline (yvo_pipeline) performance measurement
#   {ptr, dmabuf} × NHWC + dmabuf × NHCW, each {batch1/slots2, batch4/slots8, batch8/slots16} × RUNS times
#   NHCW is the path that eliminates HailoRT's CPU input conversion and works only with dma-buf input.
#
# Usage:
#   bash 06_tools/bench/pipeline_bench.sh                    # all (1816 frames), 3 runs per config
#   RUNS=1 FRAMES=300 bash 06_tools/bench/pipeline_bench.sh  # short
#   ONLY='dmabuf_b8' bash 06_tools/bench/pipeline_bench.sh   # subset by config id (grep -E pattern)
#   ONLY='nhcw' APPEND=1 bash ...                        # append to existing results_raw.csv (only under the same cool-down rules)
#   COOL_C=65 COOL_MAX_S=120 HOT_C=75 bash ...          # cool-down conditions
#   OUT_DIR=/path bash ...                               # output directory
#   SAMPLES=0 bash ...                                   # skip the sample PNG / detection JSONL dump run
#
# Run order: repetition (run) is the outer loop, config the inner loop (spreads thermal bias evenly across configs).
# Cool-down: if the previous run's hailort.log has temperature/throttle warnings, or Hailo chip temp exceeds HOT_C right before a run,
#       wait until chip temp is at or below COOL_C (at most COOL_MAX_S s). Chip temp is read via hailo_platform (Python).
# Output:
#   $OUT_DIR/raw/<id>_run<N>/{stdout.log,hailort.log}  per-run logs (hailort.log appears in the run directory)
#   $OUT_DIR/results_raw.csv   per-run RESULT_CSV rows + wait time and warning count
#   $OUT_DIR/results.csv       per-config medians
#   $OUT_DIR/samples/          detections.jsonl + box PNGs from one dmabuf batch8 run (separate from measurement)
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
# They used to be copy-pasted into 7 scripts and had actually diverged (e.g. pcie_link's Gen5 branch).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/pipeline_ffmpeg_vulkan_hailo}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
SAMPLES="${SAMPLES:-1}"
SAMPLE_LAYOUT="${SAMPLE_LAYOUT:-nhcw}"   # layout used for the sample dump
APPEND="${APPEND:-0}"                  # if 1, append to results_raw.csv instead of clearing it

require_bin "$BIN" yvo_pipeline
check_only_exit
guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

# id|mode|layout|batch|slots
configs=(
  "ptr_b1|ptr|nhwc|1|2"
  "ptr_b4|ptr|nhwc|4|8"
  "ptr_b8|ptr|nhwc|8|16"
  "dmabuf_b1|dmabuf|nhwc|1|2"
  "dmabuf_b4|dmabuf|nhwc|4|8"
  "dmabuf_b8|dmabuf|nhwc|8|16"
  "nhcw_b1|dmabuf|nhcw|1|2"
  "nhcw_b4|dmabuf|nhcw|4|8"
  "nhcw_b8|dmabuf|nhcw|8|16"
)


{
  echo "# pipeline_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
} > "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
CSV_HEADER="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,run,cool_wait_s,hailort_warnings,hailort_errors"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

prev_warn=0
frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id mode layout batch slots <<<"$c"
    echo "$id" | grep -Eq "$ONLY" || continue
    COOL_WAITED=0
    t_now=$(hailo_temp)
    if (( prev_warn > 0 )); then
      cool_down "hailort.log warnings: ${prev_warn}"
    elif awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then
      cool_down "chip temp ${t_now} C > ${HOT_C} C"
    fi
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    echo "[$(date +%T)] run $r/$RUNS $id (hailo $(hailo_temp) C)"
    set +e
    (cd "$run_dir" && "$BIN" --video "$VIDEO" --hef "$HEF" --mode "$mode" --layout "$layout" --postproc host \
        --batch "$batch" --slots "$slots" \
        "${frames_arg[@]}" --quiet --csv-label "${id}" > stdout.log 2>&1)
    rc=$?
    set -e
    nwarn=0; nerr=0
    if [[ -f "$run_dir/hailort.log" ]]; then
      nwarn=$(grep -E '\[warning\]' "$run_dir/hailort.log" | grep -cE "$WARN_RE" || true)
      nerr=$(grep -c '\[error\]' "$run_dir/hailort.log" || true)
    fi
    prev_warn=$nwarn
    line=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | sed 's/^RESULT_CSV,//' || true)
    if [[ -z "$line" ]]; then
      echo "  failed rc=$rc (see stdout.log)"; tail -5 "$run_dir/stdout.log"
      continue
    fi
    echo "$line,$r,$COOL_WAITED,$nwarn,$nerr" >> "$RAW"
    echo "  $(grep -E '^wall|^cpu ' "$run_dir/stdout.log" | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc"
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
num = ["fps", "wall_s", "cpu_ms_per_frame", "decode_ms_mean", "gpu_ms_mean", "gpu_ms_p90", "npu_ms_mean",
       "npu_ms_p90", "e2e_ms_mean", "e2e_ms_p90", "detections", "hailo_t1", "soc_t1"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "mode", "layout", "batch", "slots", "runs", "frames"] + [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "hailo_t1_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [statistics.median(float(x[k]) for x in g) for k in num]
        fps = [float(x["fps"]) for x in g]
        w.writerow([lab, g[0]["mode"], g[0]["layout"], g[0]["batch"], g[0]["slots"], len(g), g[0]["frames"]] +
                   [f"{v:.3f}" for v in med] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}", f"{max(float(x['hailo_t1']) for x in g):.1f}",
                    sum(int(x["hailort_warnings"]) for x in g), sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])
print(open(sys.argv[2]).read())
EOF

if [[ "$SAMPLES" == "1" ]]; then
  echo "[samples] dmabuf/$SAMPLE_LAYOUT batch8 --dump-dir (excluded from measurement)"
  rm -rf "$OUT_DIR/samples"; mkdir -p "$OUT_DIR/samples"
  (cd "$OUT_DIR/samples" && "$BIN" --video "$VIDEO" --hef "$HEF" --mode dmabuf --layout "$SAMPLE_LAYOUT" --postproc host --batch 8 "${frames_arg[@]}" \
      --dump-dir "$OUT_DIR/samples" --dump-every 300 > stdout.log 2>&1) || echo "  samples run failed"
fi
{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
} >> "$OUT_DIR/env_bench.txt"
