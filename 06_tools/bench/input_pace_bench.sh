#!/usr/bin/env bash
# Input pacing + leaky queue measurement — yvo_pipeline --input-pace source --input-queue N vs asap
#
# Why a new script instead of op_point_bench.sh:
#   op_point_bench.sh's raw CSV header is checked by APPEND=1 against archived results (07_results/op_point/),
#   and paced runs print a fourth line (RESULT_CSV4) whose fields that header has no room for. So that script is
#   left alone. Cool-down, environment snapshot, lock and loop structure follow op_point_bench.sh.
#
# What it measures: for b1/s1 (overloaded: processing > 33.3 ms), b2/s3 (about 98% utilisation) and b2/s4 (headroom),
#   frames arrive at the source frame rate (30 fps) into a leaky queue of N = 1, 2, 3 (full → drop the oldest).
#   The same three configs in asap mode are the controls. All on the current defaults: dmabuf, nhcw, postproc gpu,
#   nms gpu, display off.
#   ⚠ FPS and cpu_ms_per_frame of paced rows are per processed frame (plus polling): not comparable to the asap rows.
#     Compare paced rows by drop_rate, e2e_arr_* (source arrival → NPU completion) and queue_wait_*.
#
# Usage:
#   bash 06_tools/bench/input_pace_bench.sh                    # all, 3 runs per config, 900 arrived frames
#   RUNS=3 FRAMES=900 ONLY='b2s3' bash ...                     # subset only (regex on the config id)
#   BIN=/path/to/yvo_pipeline bash ...                         # pin binary (guards against rebuilds mid-measurement)
#   APPEND=1 FORCE=1 bash ...                                  # append to an existing results_raw.csv
#   DRYRUN=1 bash ...                                          # print the commands only (no lock, no run, no files)
#   CHECK_ONLY=1 bash ...                                      # path check only (lib/common.sh)
#   FRAME_LOG=1 bash ...                                       # also write raw/<run>/frame_log.csv (--frame-log; analyse with frame_log_stats.py)
#
# Run order: repetition (run) is the outer loop, config the inner loop — spreads thermal bias evenly across configs.
# --frames counts **arrived** frames with --input-pace source and submitted frames with asap (900 either way).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/input_pacing}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-900}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
APPEND="${APPEND:-0}"
DRYRUN="${DRYRUN:-0}"
fl_arg=(); [[ "${FRAME_LOG:-0}" == "1" ]] && fl_arg=(--frame-log frame_log.csv)   # relative: each run cds into its run dir

require_bin "$BIN" yvo_pipeline
check_only_exit

# id|free-form args. Per config: asap control first, then N = 1, 2, 3.
COMMON="--mode dmabuf --layout nhcw --postproc gpu --nms gpu --display off"
PACE="--input-pace source --input-queue"
configs=(
  "asap_b1s1 |$COMMON --batch 1 --slots 1"
  "src_b1s1_q1 |$COMMON --batch 1 --slots 1 $PACE 1"
  "src_b1s1_q2 |$COMMON --batch 1 --slots 1 $PACE 2"
  "src_b1s1_q3 |$COMMON --batch 1 --slots 1 $PACE 3"
  "asap_b2s3 |$COMMON --batch 2 --slots 3"
  "src_b2s3_q1 |$COMMON --batch 2 --slots 3 $PACE 1"
  "src_b2s3_q2 |$COMMON --batch 2 --slots 3 $PACE 2"
  "src_b2s3_q3 |$COMMON --batch 2 --slots 3 $PACE 3"
  "asap_b2s4 |$COMMON --batch 2 --slots 4"
  "src_b2s4_q1 |$COMMON --batch 2 --slots 4 $PACE 1"
  "src_b2s4_q2 |$COMMON --batch 2 --slots 4 $PACE 2"
  "src_b2s4_q3 |$COMMON --batch 2 --slots 4 $PACE 3"
)

frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN: OUT_DIR=$OUT_DIR RUNS=$RUNS FRAMES=$FRAMES ONLY=$ONLY"
  for ((r = 1; r <= RUNS; r++)); do
    for c in "${configs[@]}"; do
      IFS='|' read -r id args <<<"$c"
      id="${id// /}"
      echo "$id" | grep -Eq "$ONLY" || continue
      # shellcheck disable=SC2086  # args is word-split on purpose
      echo "(cd $OUT_DIR/raw/${id}_run${r} && $BIN --video $VIDEO --hef $HEF $(echo $args) ${frames_arg[*]} ${fl_arg[*]} --quiet --csv-label $id > stdout.log 2>&1)"
    done
  done
  exit 0
fi

guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

{
  echo "# input_pace_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  pkg_versions libavcodec61 hailort mesa-vulkan-drivers
  echo "bin: $BIN"
  echo "bin md5: $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "bin mtime: $(stat -c %y "$BIN")"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
# yvo_pipeline's 22 RESULT_CSV fields + the 21 RESULT_CSV4 fields after its label (nan for asap rows; its ok is pace_ok)
# + 4 fields appended by this script.
CSV4_COLS="input_queue,src_fps,arrived,processed,dropped,abandoned,drop_rate,max_drop_run,queue_depth_max,e2e_arr_mean,e2e_arr_median,e2e_arr_p90,e2e_arr_max,queue_wait_mean,queue_wait_p90,queue_wait_max,decode_late_mean,decode_late_p90,decode_late_max,held_frames_max,pace_ok"
CSV_HEADER="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,$CSV4_COLS,run,cool_wait_s,hailort_warnings,hailort_errors"
CSV4_NAN="$(printf 'nan%.0s,' $(seq 1 21))"; CSV4_NAN="${CSV4_NAN%,}"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

prev_warn=0
for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id args <<<"$c"
    id="${id// /}"
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
    # shellcheck disable=SC2086  # args is word-split on purpose
    (cd "$run_dir" && "$BIN" --video "$VIDEO" --hef "$HEF" $args \
        "${frames_arg[@]}" "${fl_arg[@]}" --quiet --csv-label "$id" > stdout.log 2>&1)
    rc=$?
    set -e
    read -r nwarn nerr <<<"$(count_hailort_warnings "$run_dir/hailort.log")"
    prev_warn=$nwarn
    line=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | sed 's/^RESULT_CSV,//' || true)
    if [[ -z "$line" ]]; then
      echo "  failed rc=$rc (see stdout.log)"; tail -5 "$run_dir/stdout.log"
      continue
    fi
    pace="$CSV4_NAN"
    if [[ "$args" == *"--input-pace source"* ]]; then
      pace=$(grep '^RESULT_CSV4,' "$run_dir/stdout.log" | cut -d, -f3- || true)
      if [[ -z "$pace" ]]; then
        echo "  paced run printed no RESULT_CSV4 rc=$rc (see stdout.log)"; tail -5 "$run_dir/stdout.log"
        continue
      fi
    fi
    echo "$line,$pace,$r,$COOL_WAITED,$nwarn,$nerr" >> "$RAW"
    echo "  $(grep -E '^wall|^arrived' "$run_dir/stdout.log" | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc"
  done
done

# Per-config medians (+ min/max of the verdict columns). Numeric columns are picked from the header, as in
# op_point_bench.sh. asap rows carry nan in the RESULT_CSV4 columns, so their medians stay nan.
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, math, statistics, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows"); sys.exit(0)

NON_NUM = {"label", "mode", "layout", "run"}          # identifiers — no median
def numeric(key):
    if key in NON_NUM:
        return False
    for r in rows:                                     # numeric column if every row parses as float (nan included)
        try: float(r[key])
        except (TypeError, ValueError): return False
    return True

num = [k for k in rows[0] if numeric(k)]
RANGE = ["fps", "e2e_ms_p90", "drop_rate", "e2e_arr_p90", "e2e_arr_max", "queue_wait_p90"]

def med(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return statistics.median(vals) if vals else float("nan")

order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)

with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")             # no CRLF (measure skill §5)
    w.writerow(["label", "mode", "layout", "runs"] + [f"{k}_median" for k in num] +
               [f"{k}_{m}" for k in RANGE for m in ("min", "max")] +
               ["hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        meds = [med(float(x[k]) for x in g) for k in num]
        rng = []
        for k in RANGE:
            v = [float(x[k]) for x in g if not math.isnan(float(x[k]))]
            rng += [f"{min(v):.3f}", f"{max(v):.3f}"] if v else ["nan", "nan"]
        w.writerow([lab, g[0]["mode"], g[0]["layout"], len(g)] + [f"{v:.3f}" for v in meds] + rng +
                   [sum(int(x["hailort_warnings"]) for x in g),
                    sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" and x["pace_ok"] in ("1", "nan") for x in g))])

# Print only the columns used for the verdict (the full set is in results.csv).
show = ["fps_median", "e2e_ms_p90_median", "drop_rate_median", "max_drop_run_median", "e2e_arr_mean_median",
        "e2e_arr_p90_median", "queue_wait_p90_median", "decode_late_p90_median", "held_frames_max_median", "all_ok"]
out = list(csv.DictReader(open(sys.argv[2])))
hdr = ["label"] + [c for c in show if c in out[0]]
wid = [max(len(h), *(len(r[h]) for r in out)) for h in hdr]
print(" | ".join(h.ljust(w) for h, w in zip(hdr, wid)))
print("-+-".join("-" * w for w in wid))
for r in out:
    print(" | ".join(r[h].ljust(w) for h, w in zip(hdr, wid)))
EOF

{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "bin md5 (after measurement): $(md5sum "$BIN" | cut -d' ' -f1)"
  vcgencmd get_throttled 2>/dev/null || true
} >> "$OUT_DIR/env_bench.txt"
