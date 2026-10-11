#!/usr/bin/env bash
# Operating-point measurement — batch/slots/gpu_depth/postproc/display matrix
#
# Why not use pipeline_bench.sh:
#   pipeline_bench.sh's configs are fixed at 5 fields `id|mode|layout|batch|slots`, so
#   there is no room for --gpu-depth/--postproc/--display*. And that script's 9 rows are
#   a **regression-baseline asset** (this script's control group compares against it to judge session validity).
#   Changing the schema would make that asset irreproducible, so it is left alone and a new script is made.
#   The configs schema follows the proven precedent in this repo, isp_pipeline_bench.sh's `id|free-form args`.
#   Cool-down, environment snapshot and loop structure are copied verbatim from pipeline_bench.sh (unmodified).
#
# Usage:
#   bash 06_tools/bench/op_point_bench.sh                       # all, 3 runs per config
#   RUNS=3 FRAMES=900 ONLY='b2s2|b8s8' bash ...             # subset only
#   BIN=/path/to/yvo_pipeline bash ...                      # pin binary (guards against rebuilds mid-measurement)
#   APPEND=1 bash ...                                       # append to existing results_raw.csv
#   USE_GRID=1 OUT_DIR=... bash ...                         # batch x slots grid on current defaults
#
# Run order: repetition (run) is the outer loop, config the inner loop — spreads thermal bias evenly across configs.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, package versions, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
# They used to be copy-pasted into 7 scripts and had actually diverged (e.g. pcie_link's Gen5 branch).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/op_point}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-900}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
APPEND="${APPEND:-0}"

require_bin "$BIN" yvo_pipeline
check_only_exit
guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

# Display configs need Wayland. Look it up here instead of relying on the caller's environment.
# If absent, skip configs with --display and note that on screen and in env_bench.txt.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
  _wl=$(ls -t "$XDG_RUNTIME_DIR"/wayland-[0-9]* 2>/dev/null | head -1 || true)
  [[ -n "$_wl" ]] && export WAYLAND_DISPLAY="$(basename "$_wl")"
fi
HAVE_WAYLAND=0; [[ -n "${WAYLAND_DISPLAY:-}" ]] && HAVE_WAYLAND=1

# id|free-form args
#   Stage 1 (M1): batch/slot sweep. Display off, postproc host, --gpu-depth left at its default (3)
#   so results compare directly with the existing 07_results/pipeline_ffmpeg_vulkan_hailo/results.csv.
#   b2s2 goes first: it has a real risk of not exceeding 30 FPS, which we need to know first.
configs=(
  "b2s2      |--mode dmabuf --layout nhcw --batch 2 --slots 2 --postproc host"
  "b2s4      |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc host"
  "b4s4      |--mode dmabuf --layout nhcw --batch 4 --slots 4 --postproc host"
  "b8s8      |--mode dmabuf --layout nhcw --batch 8 --slots 8 --postproc host"
  "ctl_b4s8  |--mode dmabuf --layout nhcw --batch 4 --slots 8 --postproc host"    # control: previously 44.52 FPS
  "ctl_b8s16 |--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"   # control: previously 51.63 FPS
)

# Latency-minimizing sweep (select with ONLY='lat_'). Gives up 30 FPS to see how far latency can drop.
#   Fewer slots remove queue waiting; lower gpu_depth removes GPU-stage buffering.
#   s=1 is a fully serial path with no pipelining at all (unmeasured territory).
configs_lat=(
  "lat_b1s1d1 |--mode dmabuf --layout nhcw --batch 1 --slots 1 --gpu-depth 1 --postproc host"
  "lat_b1s2d1 |--mode dmabuf --layout nhcw --batch 1 --slots 2 --gpu-depth 1 --postproc host"
  "lat_b2s2d1 |--mode dmabuf --layout nhcw --batch 2 --slots 2 --gpu-depth 1 --postproc host"
  "lat_b2s3d2 |--mode dmabuf --layout nhcw --batch 2 --slots 3 --gpu-depth 2 --postproc host"
  "lat_b2s4d1 |--mode dmabuf --layout nhcw --batch 2 --slots 4 --gpu-depth 1 --postproc host"
)
[[ "${USE_LAT:-0}" == "1" ]] && configs=("${configs_lat[@]}")

# Batch x slots grid on the current defaults (select with USE_GRID=1): postproc gpu, nms gpu, gpu_depth 3, display off.
#   HailoRawRunner caps slots at get_async_max_queue_size: 8 for batch 1/2/4, 16 for batch 8 (2026-10-08).
configs_grid=(
  "g_b1s1  |--batch 1 --slots 1"
  "g_b1s2  |--batch 1 --slots 2"
  "g_b1s4  |--batch 1 --slots 4"
  "g_b1s8  |--batch 1 --slots 8"
  "g_b2s2  |--batch 2 --slots 2"
  "g_b2s3  |--batch 2 --slots 3"
  "g_b2s4  |--batch 2 --slots 4"
  "g_b2s6  |--batch 2 --slots 6"
  "g_b2s8  |--batch 2 --slots 8"
  "g_b4s4  |--batch 4 --slots 4"
  "g_b4s6  |--batch 4 --slots 6"
  "g_b4s8  |--batch 4 --slots 8"
  "g_b8s8  |--batch 8 --slots 8"
  "g_b8s12 |--batch 8 --slots 12"
  "g_b8s16 |--batch 8 --slots 16"
)
[[ "${USE_GRID:-0}" == "1" ]] && configs=("${configs_grid[@]}")

# ── end of copy ──

{
  echo "# op_point_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  pkg_versions "${PKGS_DISPLAY_BENCH[@]}"   # labwc/libwlroots matter: the wayland-planes COLOR_RANGE inversion is a compositor behavior
  echo "bin: $BIN"
  echo "bin md5: $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "bin mtime: $(stat -c %y "$BIN")"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
  if (( HAVE_WAYLAND )); then echo "wayland: $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
  else echo "wayland: none — configs with --display are skipped"; fi
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
# yvo_pipeline's 22 RESULT_CSV fields + 4 fields appended by this script.
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
    IFS='|' read -r id args <<<"$c"
    id="${id// /}"
    echo "$id" | grep -Eq "$ONLY" || continue
    if [[ "$args" == *"--display"* ]] && (( ! HAVE_WAYLAND )); then
      echo "[skip] $id — no Wayland"; continue
    fi
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
        "${frames_arg[@]}" --quiet --csv-label "$id" > stdout.log 2>&1)
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

# Per-config medians.
# Unlike pipeline_bench.sh, numeric columns are not hard-coded — they are picked from the header automatically.
# That way this script needs no change when yvo_pipeline adds columns to RESULT_CSV.
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows"); sys.exit(0)

NON_NUM = {"label", "mode", "layout", "run"}          # identifiers — no median
def numeric(key):
    if key in NON_NUM:
        return False
    for r in rows:                                     # numeric column if every row parses as float
        try: float(r[key])
        except (TypeError, ValueError): return False
    return True

num = [k for k in rows[0] if numeric(k)]

order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)

with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")             # no CRLF (measure skill §5)
    w.writerow(["label", "mode", "layout", "runs"] + [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [statistics.median(float(x[k]) for x in g) for k in num]
        fps = [float(x["fps"]) for x in g]
        w.writerow([lab, g[0]["mode"], g[0]["layout"], len(g)] + [f"{v:.3f}" for v in med] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}",
                    sum(int(x["hailort_warnings"]) for x in g),
                    sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])

# Print only the columns used for the verdict (the full set is in results.csv).
show = ["fps_median", "cpu_ms_per_frame_median", "npu_ms_mean_median",
        "e2e_ms_mean_median", "e2e_ms_p90_median", "detections_median", "hailo_t1_median"]
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
