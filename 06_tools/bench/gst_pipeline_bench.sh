#!/usr/bin/env bash
# App vs GStreamer research path: yvo_pipeline (dmabuf/NHCW) vs yvo_gst_pipeline (--sink hailo)
#   Same input video, same HEF, same InferModel batch and the same ring slots (= same NPU queue depth) on both sides.
#   app: ffmpeg DRM_PRIME decode -> SandLetterbox (Vulkan, NHCW) -> HailoRT (host NMS)
#   gst: filesrc ! qtdemux ! h265parse ! v4l2slh265dec ! yvosandletterbox (Vulkan, NHCW) ! yvohailosink (host NMS)
#   Both print RESULT_CSV with the same 22 columns (pipeline/main.cpp, tools/gst/gst_pipeline.cpp).
#   Differences in meaning (gst_pipeline.cpp RESULT_CSV comment): gst decode_ms_mean = nan (no blocking decode call),
#   gst e2e = letterbox arrival -> completion (app: decode return -> completion), gst wall/cpu = PLAYING -> EOS after
#   preroll (app: first decode -> drain).
#
# Usage:
#   bash 06_tools/bench/gst_pipeline_bench.sh                    # all (1816 frames), 3 runs per config
#   RUNS=1 FRAMES=300 bash 06_tools/bench/gst_pipeline_bench.sh  # short
#   ONLY='_b8' bash ...                                  # subset by config id (grep -E pattern)
#   ONLY='gst' APPEND=1 FORCE=1 bash ...                 # append to existing results_raw.csv (same cool-down rules only)
#   COOL_C=65 COOL_MAX_S=120 HOT_C=75 bash ...           # cool-down conditions
#   OUT_DIR=/path bash ...                               # output directory (default 07_results/gst_pipeline)
#   DETS=1 bash ...                                      # + one detection dump run per config (excluded from timing)
#   SAMPLES=1 bash ...                                   # DETS=1 + app box PNGs every 300 frames (default 0: no PNGs)
#   DRYRUN=1 bash ...                                    # print the exact command line of every run and exit
#                                                        # (no lock, no NPU/GPU, no temperature probe, writes nothing)
#   CHECK_ONLY=1 bash ...                                # binary check only (lib/common.sh)
#
# Run order: repetition (run) is the outer loop, config the inner loop (spreads thermal bias evenly across configs).
# Cool-down: if the previous run's hailort.log has temperature/throttle/clock warnings, or Hailo chip temp exceeds HOT_C
#       right before a run, wait until chip temp is at or below COOL_C (at most COOL_MAX_S s).
# Reliability: a run whose hailort.log has temperature/throttle/clock warnings is marked unreliable=1 (measure skill §4).
# Exit codes of the tools: app 0 pass / 1 fail / 2 bad args / 128+signo interrupted;
#       gst 0 pass / 1 runtime error / 2 bad args / 3 checks failed / 4 stall. Recorded per run in exit_code.
# Output:
#   $OUT_DIR/env_bench.txt           environment before and after (PCIe, bootloader, v3d_freq, packages, git HEAD)
#   $OUT_DIR/raw/<id>_run<N>/{stdout.log,hailort.log}  per-run logs (the run directory is the CWD)
#   $OUT_DIR/raw/_session/hailort.log                  temperature probe log
#   $OUT_DIR/results_raw.csv         per run: pipeline_bench.sh's 26 columns (RESULT_CSV 22 + run,cool_wait_s,
#                                    hailort_warnings,hailort_errors) + tool,exit_code,hailo_pre_c,unreliable.
#                                    label = config id. A run without a RESULT_CSV line is recorded with nan and ok=0.
#   $OUT_DIR/results_raw_gst.csv     per gst run: RESULT_GST_CSV (steady_fps, slot wait, stall flag) + run,exit_code
#   $OUT_DIR/results.csv             per config: pipeline_bench.sh's 26 columns (NaN-safe medians) + tool extras
#   DETS=1: $OUT_DIR/samples/<id>/{detections.jsonl,stdout.log,hailort.log}, app also frame00000.png
#           $OUT_DIR/evidence/compare_b<B>.{txt,csv}  compare_detections.py, ref = app_nhcw_b<B>, test = gst_b<B>
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
APP_BIN="${APP_BIN:-$BIN_DIR/yvo_pipeline}"
GST_BIN="${GST_BIN:-$BIN_DIR/yvo_gst_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/gst_pipeline}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
COOL_EVERY=0                           # fixed: this bench only cools on warnings or > HOT_C (lib/common.sh cool_before_run)
SAMPLES="${SAMPLES:-0}"                # if 1, DETS=1 and the app DETS run also writes a box PNG every 300 frames
DETS="${DETS:-0}"                      # if 1, one extra --dump-dir run per config into samples/ + compare_detections.py
DETS_PNG_EVERY="${DETS_PNG_EVERY:-100000}"   # app --dump-every for the DETS run (>= frames -> only frame00000.png)
if [[ "$SAMPLES" == "1" ]]; then DETS=1; DETS_PNG_EVERY=300; fi
APPEND="${APPEND:-0}"                  # if 1, append to results_raw*.csv instead of clearing them
DRYRUN="${DRYRUN:-0}"

require_bin "$APP_BIN" yvo_pipeline
require_bin "$GST_BIN" yvo_gst_pipeline
check_only_exit

# id|tool|batch|slots — the same slots on both sides give the same NPU queue depth
# (gst: yvosandletterbox ring slots, each returned in the NPU completion callback; app: GPU+NPU shared slots).
# app args are exactly pipeline_bench.sh's nhcw_b* configs (--mode dmabuf --layout nhcw).
configs=(
  "app_nhcw_b1|app|1|2"
  "gst_b1|gst|1|2"
  "app_nhcw_b4|app|4|8"
  "gst_b4|gst|4|8"
  "app_nhcw_b8|app|8|16"
  "gst_b8|gst|8|16"
)

frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

# build_cmd <tool> <id> <batch> <slots> [dump_dir] -> fills cmd[]
build_cmd() {
  local tool="$1" id="$2" batch="$3" slots="$4" dump="${5:-}"
  case "$tool" in
    app)
      cmd=("$APP_BIN" --video "$VIDEO" --hef "$HEF" --mode dmabuf --layout nhcw --postproc host
           --batch "$batch" --slots "$slots" --input-pace asap "${frames_arg[@]}")   # asap: pre-2026-10-27 app default, explicit since O2 became the default
      if [[ -n "$dump" ]]; then cmd+=(--dump-dir "$dump" --dump-every "$DETS_PNG_EVERY"); fi
      ;;
    gst)
      cmd=("$GST_BIN" --video "$VIDEO" --hef "$HEF" --sink hailo
           --batch "$batch" --slots "$slots" "${frames_arg[@]}")
      if [[ -n "$dump" ]]; then cmd+=(--dump-dir "$dump"); fi
      ;;
    *) echo "unknown tool in configs: $tool" >&2; exit 1 ;;
  esac
  cmd+=(--quiet --csv-label "$id")
}

selected=()
for c in "${configs[@]}"; do
  IFS='|' read -r id _ _ _ <<<"$c"
  if grep -Eq "$ONLY" <<<"$id"; then selected+=("$c"); fi
done
(( ${#selected[@]} > 0 )) || { echo "ONLY='$ONLY' matches no config id" >&2; exit 1; }

# DETS pairs: app_nhcw_b<B> (ref) with gst_b<B> (test), both selected
dets_pairs=()
for c in "${selected[@]}"; do
  IFS='|' read -r id tool batch _ <<<"$c"
  [[ "$tool" == "app" ]] || continue
  for c2 in "${selected[@]}"; do
    IFS='|' read -r id2 tool2 batch2 _ <<<"$c2"
    if [[ "$tool2" == "gst" && "$batch2" == "$batch" ]]; then dets_pairs+=("$batch|$id|$id2"); fi
  done
done

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN: $(basename "$0") OUT_DIR=$OUT_DIR RUNS=$RUNS FRAMES=$FRAMES ONLY='$ONLY' DETS=$DETS"
  if [[ -d "$OUT_DIR" ]] && [[ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "DRYRUN: note — OUT_DIR is not empty; the real run stops here without FORCE=1"
  fi
  for ((r = 1; r <= RUNS; r++)); do
    for c in "${selected[@]}"; do
      IFS='|' read -r id tool batch slots <<<"$c"
      build_cmd "$tool" "$id" "$batch" "$slots"
      printf '[run %d/%d %s] cd %q && ' "$r" "$RUNS" "$id" "$OUT_DIR/raw/${id}_run${r}"
      printf '%q ' "${cmd[@]}"; printf '> stdout.log 2>&1\n'
    done
  done
  if [[ "$DETS" == "1" ]]; then
    for c in "${selected[@]}"; do
      IFS='|' read -r id tool batch slots <<<"$c"
      build_cmd "$tool" "$id" "$batch" "$slots" "$OUT_DIR/samples/$id"
      printf '[dets %s] cd %q && ' "$id" "$OUT_DIR/samples/$id"
      printf '%q ' "${cmd[@]}"; printf '> stdout.log 2>&1\n'
    done
    for p in "${dets_pairs[@]}"; do
      IFS='|' read -r b ref test <<<"$p"
      printf '[compare b%s] %q %q %q %q --csv %q --label %q\n' "$b" "$PY_HAILO" \
        "$PROJ/06_tools/python/compare_detections.py" "$OUT_DIR/samples/$ref/detections.jsonl" \
        "$OUT_DIR/samples/$test/detections.jsonl" "$OUT_DIR/evidence/compare_b$b.csv" "${ref}_vs_${test}"
    done
  fi
  exit 0
fi

guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

# Environment snapshot (before / after). Package and git lines are what distinguishes the two paths' builds.
env_snapshot() {
  echo "# gst_pipeline_bench $1 $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  vcgencmd get_config v3d_freq 2>/dev/null || true
  echo "bootloader: $(vcgencmd bootloader_version 2>/dev/null | head -1)"
  pcie_link
  pkg_versions gstreamer1.0-plugins-bad libgstreamer1.0-0 hailort
  env_git_head
}

{
  env_snapshot before
  echo "bins: $(stat -c '%n %s B %y' "$APP_BIN" "$GST_BIN" | paste -sd ';')"
  echo "env: $(env | grep -E '^(GST_|VK_|HAILO)' | paste -sd ' ' || true)"
  echo "video: $VIDEO frames: $([[ "$FRAMES" == "0" ]] && echo all || echo "$FRAMES") runs: $RUNS only: $ONLY"
  echo "configs: ${selected[*]}"
} > "$OUT_DIR/env_bench.txt"
# A validation layer or GST_DEBUG left in the environment silently changes the timing.
if env | grep -qE '^(VK_LOADER_LAYERS_ENABLE|VK_INSTANCE_LAYERS|GST_DEBUG)='; then
  echo "WARNING: VK_*LAYERS*/GST_DEBUG set in the environment (recorded in env_bench.txt)" >&2
fi

RAW="$OUT_DIR/results_raw.csv"
GRAW="$OUT_DIR/results_raw_gst.csv"
# The first 26 columns are pipeline_bench.sh's results_raw.csv header, unchanged.
CSV_HEADER="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,run,cool_wait_s,hailort_warnings,hailort_errors,tool,exit_code,hailo_pre_c,unreliable"
GST_HEADER="label,sink,hold,slots,queue,batch,frames,wall_s,fps,steady_fps,cpu_ms_per_frame,gpu_ms_mean,gpu_ms_median,gpu_ms_p90,slot_wait_ms_mean,elem_sink_ms_mean,elem_sink_ms_p90,vk_errors,stalled,ok,run,exit_code"
for pair in "$RAW|$CSV_HEADER" "$GRAW|$GST_HEADER"; do
  f="${pair%%|*}"; h="${pair#*|}"
  if [[ "$APPEND" == "1" && -f "$f" ]]; then
    head -1 "$f" | grep -qx "$h" || { echo "APPEND=1 but the existing $f header differs" >&2; exit 1; }
  else
    echo "$h" > "$f"
  fi
done

prev_warn=0

# cool_before_run (lib/common.sh): pipeline_bench.sh's rule (COOL_EVERY=0). Sets t_pre and COOL_WAITED.

for ((r = 1; r <= RUNS; r++)); do
  for c in "${selected[@]}"; do
    IFS='|' read -r id tool batch slots <<<"$c"
    cool_before_run
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    build_cmd "$tool" "$id" "$batch" "$slots"
    echo "[$(date +%T)] run $r/$RUNS $id (hailo ${t_pre} C)"
    set +e
    (cd "$run_dir" && "${cmd[@]}" > stdout.log 2>&1)
    rc=$?
    set -e
    read -r nwarn nerr < <(count_hailort_warnings "$run_dir/hailort.log")
    prev_warn=$nwarn
    unreliable=0; (( nwarn > 0 )) && unreliable=1
    line=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
    if [[ -z "$line" ]] || [[ "$(awk -F, '{print NF}' <<<"$line")" != "22" ]]; then
      echo "  failed rc=$rc, no valid RESULT_CSV (see stdout.log) — recorded as nan, ok=0"
      tail -5 "$run_dir/stdout.log" || true
      line="$id,dmabuf,nhcw,$batch,$slots,0$(printf ',nan%.0s' {1..15}),0"
    elif (( rc != 0 )); then
      echo "  rc=$rc (RESULT_CSV present, see stdout.log)"
    fi
    echo "$line,$r,$COOL_WAITED,$nwarn,$nerr,$tool,$rc,$t_pre,$unreliable" >> "$RAW"
    if [[ "$tool" == "gst" ]]; then
      gline=$(grep '^RESULT_GST_CSV,' "$run_dir/stdout.log" | tail -1 | sed 's/^RESULT_GST_CSV,//' || true)
      if [[ -n "$gline" ]]; then echo "$gline,$r,$rc" >> "$GRAW"; fi
    fi
    echo "  $(grep -E '^wall|^cpu ' "$run_dir/stdout.log" | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc"
  done
done

# Per-config aggregation. The first 26 columns are pipeline_bench.sh's results.csv header, unchanged;
# medians are NaN-safe (gst decode_ms_mean is always nan, failed runs are all nan).
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" "$GRAW" <<'EOF'
import csv, math, os, statistics, sys


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def finite(vals):
    return [v for v in vals if not math.isnan(v)]


def nmed(vals):
    v = finite(vals)
    return statistics.median(v) if v else math.nan


def nmin(vals):
    v = finite(vals)
    return min(v) if v else math.nan


def nmax(vals):
    v = finite(vals)
    return max(v) if v else math.nan


rows = list(csv.DictReader(open(sys.argv[1], newline="")))
gst = {}
if os.path.isfile(sys.argv[3]):
    for g in csv.DictReader(open(sys.argv[3], newline="")):
        gst.setdefault(g["label"], []).append(g)
order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)
num = ["fps", "wall_s", "cpu_ms_per_frame", "decode_ms_mean", "gpu_ms_mean", "gpu_ms_p90", "npu_ms_mean",
       "npu_ms_p90", "e2e_ms_mean", "e2e_ms_p90", "detections", "hailo_t1", "soc_t1"]
summary = {}
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "mode", "layout", "batch", "slots", "runs", "frames"] + [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "hailo_t1_max", "hailort_warnings", "hailort_errors", "all_ok"] +
               ["tool", "runs_ok", "exit_codes", "unreliable_runs", "cool_wait_s_total", "hailo_pre_c_max",
                "gst_steady_fps_median", "gst_slot_wait_ms_median"])
    for lab in order:
        g = groups[lab]
        ok = [x for x in g if x["ok"] == "1" and x["exit_code"] == "0"]
        med = [nmed(fnum(x[k]) for x in g) for k in num]
        fps = [fnum(x["fps"]) for x in g]
        gg = gst.get(lab, [])
        steady = nmed(fnum(x["steady_fps"]) for x in gg) if gg else math.nan
        slotw = nmed(fnum(x["slot_wait_ms_mean"]) for x in gg) if gg else math.nan
        summary[lab] = (g[0]["tool"], int(g[0]["batch"]), dict(zip(num, med)))
        w.writerow([lab, g[0]["mode"], g[0]["layout"], g[0]["batch"], g[0]["slots"], len(g),
                    (ok[0] if ok else g[0])["frames"]] +
                   [f"{v:.3f}" for v in med] +
                   [f"{nmin(fps):.2f}", f"{nmax(fps):.2f}", f"{nmax(fnum(x['hailo_t1']) for x in g):.1f}",
                    sum(int(x["hailort_warnings"]) for x in g), sum(int(x["hailort_errors"]) for x in g),
                    int(len(ok) == len(g))] +
                   [g[0]["tool"], len(ok), ";".join(x["exit_code"] for x in g),
                    sum(x["unreliable"] == "1" for x in g), sum(int(x["cool_wait_s"]) for x in g),
                    f"{nmax(fnum(x['hailo_pre_c']) for x in g):.1f}",
                    "" if math.isnan(steady) else f"{steady:.2f}", "" if math.isnan(slotw) else f"{slotw:.3f}"])
print(open(sys.argv[2]).read())
# Console only: gst vs app at the same batch
for lab, (tool, b, m) in summary.items():
    if tool != "app":
        continue
    peer = next((l for l, (t, bb, _) in summary.items() if t == "gst" and bb == b), None)
    if peer is None:
        continue
    p = summary[peer][2]
    ratio = p["fps"] / m["fps"] if m["fps"] and not math.isnan(m["fps"]) else math.nan
    print(f"b{b}: fps app {m['fps']:.2f} / gst {p['fps']:.2f} (gst/app {ratio:.3f}), "
          f"cpu ms/f app {m['cpu_ms_per_frame']:.3f} / gst {p['cpu_ms_per_frame']:.3f}, "
          f"npu ms app {m['npu_ms_mean']:.2f} / gst {p['npu_ms_mean']:.2f}, "
          f"e2e ms app {m['e2e_ms_mean']:.2f} / gst {p['e2e_ms_mean']:.2f} (different start points)")
EOF

if [[ "$DETS" == "1" ]]; then
  echo "[dets] one --dump-dir run per config (excluded from measurement)"
  for c in "${selected[@]}"; do
    IFS='|' read -r id tool batch slots <<<"$c"
    cool_before_run
    sdir="$OUT_DIR/samples/$id"
    rm -rf "$sdir"; mkdir -p "$sdir"
    build_cmd "$tool" "$id" "$batch" "$slots" "$sdir"
    echo "[$(date +%T)] dets $id (hailo ${t_pre} C)"
    set +e
    (cd "$sdir" && "${cmd[@]}" > stdout.log 2>&1)
    rc=$?
    set -e
    read -r nwarn _ < <(count_hailort_warnings "$sdir/hailort.log")
    prev_warn=$nwarn
    (( rc == 0 )) || echo "  dets run failed rc=$rc (see $sdir/stdout.log)"
  done
  for p in "${dets_pairs[@]}"; do
    IFS='|' read -r b ref test <<<"$p"
    rj="$OUT_DIR/samples/$ref/detections.jsonl"; tj="$OUT_DIR/samples/$test/detections.jsonl"
    if [[ ! -s "$rj" || ! -s "$tj" ]]; then echo "  compare b$b skipped: detections.jsonl missing"; continue; fi
    mkdir -p "$OUT_DIR/evidence"
    "$PY_HAILO" "$PROJ/06_tools/python/compare_detections.py" "$rj" "$tj" \
        --csv "$OUT_DIR/evidence/compare_b$b.csv" --label "${ref}_vs_${test}" \
        > "$OUT_DIR/evidence/compare_b$b.txt" 2>&1 || echo "  compare b$b failed (see evidence/compare_b$b.txt)"
    grep -E '^match @ IoU|^detections' "$OUT_DIR/evidence/compare_b$b.txt" | sed "s/^/  b$b /" || true
  done
fi

{
  env_snapshot after
} >> "$OUT_DIR/env_bench.txt"
