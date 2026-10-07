#!/usr/bin/env bash
# Full pipeline (incl. NPU) measurement: current GPU shader path vs ISP path (option A NHWC / option C NHCW)
#   Only the b8 operating point is measured (batch 8, slots 16). b1/b4 are not operating points and were dropped.
#
# Usage:
#   bash 06_tools/bench/isp_pipeline_bench.sh                 # all (1816 frames), 3 runs per config
#   RUNS=1 FRAMES=300 bash 06_tools/bench/isp_pipeline_bench.sh
#   ONLY='isp_' bash ...            # subset by config id (grep -E pattern)
#   DUMPS=0 bash ...                # skip the detection JSONL dump run
#   OUT_DIR=/path bash ...          # output directory (FORCE=1 needed if not empty)
#
# Output: $OUT_DIR/raw/<id>_run<N>/{stdout.log,hailort.log}, $OUT_DIR/raw/_session/hailort.log (temperature probe),
#       $OUT_DIR/results_raw.csv, results.csv, env_bench.txt, $OUT_DIR/samples/detections/<id>/ (detection dumps)
#
# Run order: repetition (run) is the outer loop, config the inner loop (spreads thermal bias evenly across configs).
# Cool-down (measure skill §2): if the previous run's hailort.log has temperature/throttle warnings or chip temp
#   exceeds HOT_C right before a run, wait until it is at or below COOL_C (at most COOL_MAX_S s).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
# They used to be copy-pasted into 7 scripts and had actually diverged (e.g. pcie_link's Gen5 branch).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
GPU_BIN="${GPU_BIN:-$BIN_DIR/yvo_pipeline}"
ISP_BIN="${ISP_BIN:-$BIN_DIR/yvo_isp_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/isp_resize/runs/pipeline}"   # round folders are runs/<round>/ (STRUCTURE.md §3)
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
DUMPS="${DUMPS:-1}"
APPEND="${APPEND:-0}"

require_bin "$GPU_BIN" yvo_pipeline
require_bin "$ISP_BIN" yvo_isp_pipeline
check_only_exit
guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

# id|tool|args
configs=(
  "gpu_nhcw_b8|gpu|--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"
  "isp_A_nhwc_b8|isp|--mode dmabuf --layout nhwc --batch 8 --slots 16"
  "isp_C_nhcw_b8|isp|--mode dmabuf --layout nhcw --repack cpu --batch 8 --slots 16"
)


# Measurement-window contamination check: mtime snapshot of 01_src/ and 02_shaders/ files (is another session editing them?)
src_snapshot() { find "$PROJ/01_src" "$PROJ/02_shaders" -type f -printf '%T@ %p\n' | sort | md5sum | cut -d' ' -f1; }
src_newest() { find "$PROJ/01_src" "$PROJ/02_shaders" -type f -printf '%TT %p\n' | sort | tail -1; }

SRC_BEFORE=$(src_snapshot)
{
  echo "# isp_pipeline_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "model: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)  (0x80000 = past-event bit)"
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  echo "hef: $HEF"
  echo "libpisp: $(pkg-config --modversion libpisp 2>/dev/null || echo NA)"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
  echo "src tree md5 (before): $SRC_BEFORE"
  echo "src newest (before): $(src_newest)"
} > "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
CSV_HEADER="label,tool,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,pre_ms_mean,pre_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,run,cool_wait_s,hailort_warnings,hailort_errors"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")
prev_warn=0

for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id tool extra <<<"$c"
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
    BIN=$([[ "$tool" == "gpu" ]] && echo "$GPU_BIN" || echo "$ISP_BIN")
    echo "[$(date +%T)] run $r/$RUNS $id (hailo $(hailo_temp) C)"
    set +e
    # shellcheck disable=SC2086
    (cd "$run_dir" && "$BIN" --video "$VIDEO" --hef "$HEF" $extra "${frames_arg[@]}" \
        --quiet --csv-label "$id" > stdout.log 2>&1)
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
      echo "  failed rc=$rc (see stdout.log)"; tail -8 "$run_dir/stdout.log"; continue
    fi
    # RESULT_CSV: label,mode,layout,... -> insert tool after label
    lab="${line%%,*}"; rest="${line#*,}"
    echo "$lab,$tool,$rest,$r,$COOL_WAITED,$nwarn,$nerr" >> "$RAW"
    echo "  $(grep -E '^wall|^cpu |^detections' "$run_dir/stdout.log" | tr '\n' ' ') warn=$nwarn err=$nerr rc=$rc"
  done
done

/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)
num = ["fps", "wall_s", "cpu_ms_per_frame", "decode_ms_mean", "pre_ms_mean", "pre_ms_p90", "npu_ms_mean",
       "npu_ms_p90", "e2e_ms_mean", "e2e_ms_p90", "detections", "hailo_t1", "soc_t1"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "tool", "mode", "layout", "batch", "slots", "runs", "frames"] +
               [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "cpu_ms_min", "cpu_ms_max", "detections_min", "detections_max",
                "hailo_t1_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [statistics.median(float(x[k]) for x in g) for k in num]
        fps = [float(x["fps"]) for x in g]
        cpu = [float(x["cpu_ms_per_frame"]) for x in g]
        det = [int(x["detections"]) for x in g]
        w.writerow([lab, g[0]["tool"], g[0]["mode"], g[0]["layout"], g[0]["batch"], g[0]["slots"], len(g),
                    g[0]["frames"]] + [f"{v:.3f}" for v in med] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}", f"{min(cpu):.3f}", f"{max(cpu):.3f}",
                    min(det), max(det), f"{max(float(x['hailo_t1']) for x in g):.1f}",
                    sum(int(x["hailort_warnings"]) for x in g), sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])
print(open(sys.argv[2]).read())
EOF

# Detection JSONL dump (separate from measurement — not included in timing)
if [[ "$DUMPS" == "1" ]]; then
  for c in "${configs[@]}"; do
    IFS='|' read -r id tool extra <<<"$c"
    echo "$id" | grep -Eq "$ONLY" || continue
    # The dump is a one-off run output separate from measurement, hence samples/ (STRUCTURE.md §3). Never writes outside OUT_DIR.
    # Before 2026-09-22 it was $OUT_DIR/../detections/<id> (= isp_resize/detections, now isp_resize/samples/detections).
    d="$OUT_DIR/samples/detections/$id"
    rm -rf "$d"; mkdir -p "$d"
    BIN=$([[ "$tool" == "gpu" ]] && echo "$GPU_BIN" || echo "$ISP_BIN")
    echo "[dumps] $id -> $d"
    t_now=$(hailo_temp)
    if awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then cool_down "before dump ${t_now} C"; fi
    set +e
    # shellcheck disable=SC2086
    (cd "$d" && "$BIN" --video "$VIDEO" --hef "$HEF" $extra "${frames_arg[@]}" \
        --dump-dir "$d" --dump-every 300 --quiet > stdout.log 2>&1) || echo "  dump failed ($id)"
    set -e
  done
fi

SRC_AFTER=$(src_snapshot)
{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)"
  echo "src tree md5 (after): $SRC_AFTER"
  echo "src newest (after): $(src_newest)"
  if [[ "$SRC_BEFORE" == "$SRC_AFTER" ]]; then
    echo "contamination check: no changes in 01_src/ or 02_shaders/ (md5 match)"
  else
    echo "contamination check: **files in 01_src/ or 02_shaders/ changed during the measurement window** (md5 mismatch)"
  fi
} >> "$OUT_DIR/env_bench.txt"
