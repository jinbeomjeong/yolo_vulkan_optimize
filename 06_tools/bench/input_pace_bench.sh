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
#   DISPLAY_RUN=1 FRAME_LOG=1 bash ...                         # display mode (below); default OUT_DIR runs/display_on
#
# Display mode (DISPLAY_RUN=1, 2026-10-12): paced input (queue 1) with --display on, to measure source arrival → screen
#   (RESULT_CSV5 a2p_*) and what the display adds over the display-off rows above. Configs (all --display-fullscreen
#   --display-sync matched --display-pace source --display-warmup WARMUP, same cut as display_planes_bench.sh):
#     dvk_b1s1  Vulkan composition (--display-backend wayland), b1/s1 = O2
#     dwlp_b1s2 / dwlp_b1s3  HVS planes (--display-backend wayland-planes), b1/s2, b1/s3 (planes + b1/s1 deadlocks)
#     dvk_b2s4  Vulkan, b2/s4        dwlp_b2s6  planes, b2/s6
#   Safety (verify skill §3-1): before every run wlr-randr must list CONNECTOR (HDMI-A-1) Enabled and labwc must not have
#   lost DRM master — otherwise the session stops (the swapchain path would block forever in vkQueuePresentKHR).
#   Every run is wrapped in `timeout -s KILL TIMEOUT_S` (default 90; 900 arrived frames ≈ 30 s). Never the kms backend.
#   Raw CSV (87 columns): RESULT_CSV 22 + RESULT_CSV4 21 + RESULT_CSV5 6 (after its label; its ok is a2p_ok)
#   + 31 display fields picked from RESULT_CSV2 / RESULT_CSV3 / the summary ("-" → nan) + 7 script columns.
#   Display-mode env: WARMUP=30 TIMEOUT_S=90 CONNECTOR=HDMI-A-1.
#
# Run order: repetition (run) is the outer loop, config the inner loop — spreads thermal bias evenly across configs.
# --frames counts **arrived** frames with --input-pace source and submitted frames with asap (900 either way).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
DISPLAY_RUN="${DISPLAY_RUN:-0}"
OUT_DEFAULT="$PROJ/07_results/input_pacing"; [[ "$DISPLAY_RUN" == "1" ]] && OUT_DEFAULT="$OUT_DEFAULT/runs/display_on"
OUT_DIR="$(realpath -m "${OUT_DIR:-$OUT_DEFAULT}")"   # absolute: runs cd into raw/<run>/
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
if [[ "$DISPLAY_RUN" == "1" ]]; then
  WARMUP="${WARMUP:-30}"
  TIMEOUT_S="${TIMEOUT_S:-90}"
  CONNECTOR="${CONNECTOR:-HDMI-A-1}"
  DRI_DIR="${DRI_DIR:-/sys/kernel/debug/dri/1}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
  [[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "WARMUP must be >= 0: $WARMUP" >&2; exit 2; }
  [[ "$FRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "DISPLAY_RUN=1 needs a positive FRAMES (timeout bound): $FRAMES" >&2; exit 2; }
  (( WARMUP < FRAMES )) || { echo "WARMUP ($WARMUP) must be < FRAMES ($FRAMES)" >&2; exit 2; }
  for t in timeout wlr-randr awk; do
    command -v "$t" >/dev/null || { echo "required tool not found: $t" >&2; exit 1; }
  done
fi
check_only_exit

# id|free-form args. Per config: asap control first, then N = 1, 2, 3.
COMMON="--mode dmabuf --layout nhcw --postproc gpu --nms gpu --display off"
PACE="--input-pace source --input-queue"
configs=(
  "asap_b1s1 |$COMMON --batch 1 --slots 1 --input-pace asap"
  "src_b1s1_q1 |$COMMON --batch 1 --slots 1 $PACE 1"
  "src_b1s1_q2 |$COMMON --batch 1 --slots 1 $PACE 2"
  "src_b1s1_q3 |$COMMON --batch 1 --slots 1 $PACE 3"
  "asap_b2s3 |$COMMON --batch 2 --slots 3 --input-pace asap"
  "src_b2s3_q1 |$COMMON --batch 2 --slots 3 $PACE 1"
  "src_b2s3_q2 |$COMMON --batch 2 --slots 3 $PACE 2"
  "src_b2s3_q3 |$COMMON --batch 2 --slots 3 $PACE 3"
  "asap_b2s4 |$COMMON --batch 2 --slots 4 --input-pace asap"
  "src_b2s4_q1 |$COMMON --batch 2 --slots 4 $PACE 1"
  "src_b2s4_q2 |$COMMON --batch 2 --slots 4 $PACE 2"
  "src_b2s4_q3 |$COMMON --batch 2 --slots 4 $PACE 3"
)
if [[ "$DISPLAY_RUN" == "1" ]]; then
  # Display mode: every config paced with queue 1. Backend args are explicit (defaults can move, archived commands must not).
  DCOMMON="--mode dmabuf --layout nhcw --postproc gpu --nms gpu --display on --display-fullscreen"
  DCOMMON+=" --display-sync matched --display-pace source --display-warmup $WARMUP"
  VK="--display-backend wayland"
  WLP="--display-backend wayland-planes"
  configs=(
    "dvk_b1s1  |$DCOMMON $VK --batch 1 --slots 1 $PACE 1"
    "dwlp_b1s2 |$DCOMMON $WLP --batch 1 --slots 2 $PACE 1"
    "dwlp_b1s3 |$DCOMMON $WLP --batch 1 --slots 3 $PACE 1"
    "dvk_b2s4  |$DCOMMON $VK --batch 2 --slots 4 $PACE 1"
    "dwlp_b2s6 |$DCOMMON $WLP --batch 2 --slots 6 $PACE 1"
  )
fi

# Display-mode guards (read-only). output_enabled: wlr-randr lists CONNECTOR with "Enabled: yes" (labwc not headless).
# master_of labwc: "y"/"n"/"-" from debugfs clients, "?" if unreadable — "n" means labwc lost DRM master.
output_enabled() {
  timeout 5 wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '
    /^[^ \t]/ { inb = ($1 == c) } inb && /^[ \t]+Enabled:/ { if ($2 == "yes") ok = 1 } END { exit !ok }'
}
master_of() {
  [[ -r "$DRI_DIR/clients" ]] || { echo "?"; return; }
  awk -v c="$1" 'NR > 1 && $1 == c { seen = 1; if ($4 == "y") m = 1 } END { print (m ? "y" : (seen ? "n" : "-")) }' \
    "$DRI_DIR/clients" 2>/dev/null || echo "?"
}
display_ready() { output_enabled && [[ "$(master_of labwc)" != "n" ]]; }

frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN: OUT_DIR=$OUT_DIR RUNS=$RUNS FRAMES=$FRAMES ONLY=$ONLY"
  to_pfx=""
  if [[ "$DISPLAY_RUN" == "1" ]]; then
    echo "DRYRUN: DISPLAY_RUN=1 WARMUP=$WARMUP TIMEOUT_S=$TIMEOUT_S CONNECTOR=$CONNECTOR env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
    echo "DRYRUN: now wlr-randr $CONNECTOR $(output_enabled && echo Enabled || echo NOT-enabled), labwc DRM master: $(master_of labwc)"
    to_pfx="timeout -s KILL $TIMEOUT_S "
  fi
  for ((r = 1; r <= RUNS; r++)); do
    for c in "${configs[@]}"; do
      IFS='|' read -r id args <<<"$c"
      id="${id// /}"
      echo "$id" | grep -Eq "$ONLY" || continue
      [[ "$DISPLAY_RUN" == "1" ]] && echo "[run $r $id] require: wlr-randr lists $CONNECTOR Enabled and labwc DRM master != n, else stop the session"
      # shellcheck disable=SC2086  # args is word-split on purpose
      echo "(cd $OUT_DIR/raw/${id}_run${r} && $to_pfx$BIN --video $VIDEO --hef $HEF $(echo $args) ${frames_arg[*]} ${fl_arg[*]} --quiet --csv-label $id > stdout.log 2>&1)"
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
  if [[ "$DISPLAY_RUN" == "1" ]]; then
    # Display runs also depend on the compositor (labwc / wlroots) and the V3D clock.
    pkg_versions labwc libwlroots-0.20
    vcgencmd get_config v3d_freq 2>/dev/null || true
    env_git_head
    echo "display mode: warmup $WARMUP, timeout -s KILL ${TIMEOUT_S}s, connector $CONNECTOR, frame_log ${FRAME_LOG:-0}"
    echo "wlr-randr $CONNECTOR: $(output_enabled && echo Enabled || echo NOT-enabled); labwc DRM master: $(master_of labwc)"
    timeout 5 wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '/^[^ \t]/ { inb = ($1 == c) } inb && /current/ { print "mode: " $0; exit }' || true
    echo "env: $(env | grep -E '^(VK_|HAILO|YVO_|WAYLAND_DISPLAY|XDG_RUNTIME_DIR)' | paste -sd ' ' || true)"
    echo "configs (id|args):"
    for c in "${configs[@]}"; do echo "  $c"; done
  fi
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
# yvo_pipeline's 22 RESULT_CSV fields + the 21 RESULT_CSV4 fields after its label (nan for asap rows; its ok is pace_ok)
# + 4 fields appended by this script.
CSV4_COLS="input_queue,src_fps,arrived,processed,dropped,abandoned,drop_rate,max_drop_run,queue_depth_max,e2e_arr_mean,e2e_arr_median,e2e_arr_p90,e2e_arr_max,queue_wait_mean,queue_wait_p90,queue_wait_max,decode_late_mean,decode_late_p90,decode_late_max,held_frames_max,pace_ok"
CSV_HEADER="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,$CSV4_COLS,run,cool_wait_s,hailort_warnings,hailort_errors"
CSV4_NAN="$(printf 'nan%.0s,' $(seq 1 21))"; CSV4_NAN="${CSV4_NAN%,}"
if [[ "$DISPLAY_RUN" == "1" ]]; then
  # Display mode: RESULT_CSV 22 + RESULT_CSV4 21 + RESULT_CSV5 6 + 31 display fields + 7 script columns = 87.
  #   Display fields (main.cpp print_summary; c2:N = RESULT_CSV2 field N after its prefix, c3:N = RESULT_CSV3):
  #   display_backend c2:11, display_warmup c3:2, gpu_post_mean c2:4, disp_convert_mean c2:14, disp_overlay_mean c2:16,
  #   present_interval_median c2:17, present_interval_p90 c3:5, glass_mean c2:18, glass_median c3:3, glass_p90 c2:19,
  #   glass_n c3:4, d2s_median c3:16, d2s_p90 c3:17, presented c2:20, display_dropped c2:21,
  #   never_reached (summary "N source frames never reached the display", planes only), recreates c2:22,
  #   slot_timeout_count c2:24, slot_wait_ms_mean/p90 (summary "slot wait ms"), late_frames c3:7, late_expected_gap c3:8,
  #   seq_gap_max c3:9, discarded c3:12, refresh_hz c3:13, scanout_zero_copy c2:30, plane_color_ok c2:31,
  #   hvs_underrun_delta c2:32, overlay_skipped c2:33, frame_cb_timeouts c2:34,
  #   scanout_ok = 1 if zero_copy 1, color_ok 1, underrun delta 0 (planes; verify skill §3-1), 0 if not, nan for Vulkan.
  #   ⚠ If main.cpp changes a RESULT_CSV2/3/5 field list, the c2/c3 indexes, N_CSV* and these names change together.
  N_CSV2=35; N_CSV3=18; N_CSV5=7   # fields after the "RESULT_CSVn," prefix (label included)
  CSV5_COLS="a2p_mean,a2p_median,a2p_p90,a2p_max,a2p_n,a2p_ok"
  DISP_COLS="display_backend,display_warmup,gpu_post_mean,disp_convert_mean,disp_overlay_mean,present_interval_median,present_interval_p90,glass_mean,glass_median,glass_p90,glass_n,d2s_median,d2s_p90,presented,display_dropped,never_reached,recreates,slot_timeout_count,slot_wait_ms_mean,slot_wait_ms_p90,late_frames,late_expected_gap,seq_gap_max,discarded,refresh_hz,scanout_zero_copy,plane_color_ok,hvs_underrun_delta,overlay_skipped,frame_cb_timeouts,scanout_ok"
  CSV1_COLS="${CSV_HEADER%%,input_queue,*}"
  CSV_HEADER="$CSV1_COLS,$CSV4_COLS,$CSV5_COLS,$DISP_COLS,run,cool_wait_s,hailort_warnings,hailort_errors,exit_code,unreliable,note"
fi
nan_n() { local i out=""; for ((i = 0; i < $1; i++)); do out+="${out:+,}nan"; done; printf '%s' "$out"; }
# disp_row <stdout.log> -> the 31 display fields (nan when a line is missing or has the wrong width).
disp_row() {
  local log="$1" l2 l3 sw never
  l2=$(grep '^RESULT_CSV2,' "$log" | tail -1 | sed 's/^RESULT_CSV2,//' || true)
  l3=$(grep '^RESULT_CSV3,' "$log" | tail -1 | sed 's/^RESULT_CSV3,//' || true)
  [[ "$(awk -F, '{print NF}' <<<"$l2")" == "$N_CSV2" ]] || l2="$(nan_n "$N_CSV2")"
  [[ "$(awk -F, '{print NF}' <<<"$l3")" == "$N_CSV3" ]] || l3="$(nan_n "$N_CSV3")"
  sw=$(awk '/^slot wait ms / { if (match($0, /mean [0-9.]+/)) m = substr($0, RSTART + 5, RLENGTH - 5)
                               if (match($0, /p90 [0-9.]+/)) p = substr($0, RSTART + 4, RLENGTH - 4) }
            END { printf "%s,%s", (m == "" ? "nan" : m), (p == "" ? "nan" : p) }' "$log")
  never=$(awk '/source frames never reached the display/ { v = $2 } END { print (v == "" ? "nan" : v) }' "$log")
  awk -F, -v l3="$l3" -v sw="$sw" -v never="$never" 'BEGIN { OFS = "," }
    function v(x) { return (x == "-" || x == "") ? "nan" : x }
    {
      n3 = split(l3, c, ",")
      split(sw, s, ",")
      ok = "nan"
      if ($30 != "-" && $30 != "nan") ok = ($30 == "1" && $31 == "1" && $32 == "0") ? 1 : 0
      print v($11), v(c[2]), v($4), v($14), v($16), v($17), v(c[5]), v($18), v(c[3]), v($19), v(c[4]), v(c[16]), v(c[17]),
            v($20), v($21), never, v($22), v($24), s[1], s[2], v(c[7]), v(c[8]), v(c[9]), v(c[12]), v(c[13]),
            v($30), v($31), v($32), v($33), v($34), ok
    }' <<<"$l2"
}
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

prev_warn=0
aborted=""
to_cmd=(); [[ "$DISPLAY_RUN" == "1" ]] && to_cmd=(timeout -s KILL "$TIMEOUT_S")   # verify skill §3-1
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
    if [[ "$DISPLAY_RUN" == "1" ]] && ! display_ready; then
      echo "display not ready before run $r $id: wlr-randr $CONNECTOR $(output_enabled && echo Enabled || echo NOT-enabled)," \
           "labwc DRM master $(master_of labwc) — stopping the session (the swapchain path would block forever)" >&2
      aborted="run $r $id"; break 2
    fi
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    echo "[$(date +%T)] run $r/$RUNS $id (hailo $(hailo_temp) C)"
    set +e
    # shellcheck disable=SC2086  # args is word-split on purpose
    (cd "$run_dir" && "${to_cmd[@]}" "$BIN" --video "$VIDEO" --hef "$HEF" $args \
        "${frames_arg[@]}" "${fl_arg[@]}" --quiet --csv-label "$id" > stdout.log 2>&1)
    rc=$?
    set -e
    read -r nwarn nerr <<<"$(count_hailort_warnings "$run_dir/hailort.log")"
    prev_warn=$nwarn
    if [[ "$DISPLAY_RUN" == "1" ]]; then
      # Display mode: always one row (nan + ok=0 for missing lines) so failed / killed runs stay visible.
      log="$run_dir/stdout.log"; notes=()
      (( rc == 137 )) && notes+=(timeout_kill)
      (( rc != 0 )) && notes+=("rc_$rc")
      line=$(grep '^RESULT_CSV,' "$log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
      if [[ "$(awk -F, '{print NF}' <<<"$line")" != "22" ]]; then
        notes+=(no_result_csv)
        line="$id,dmabuf,nhcw,$(grep -oE -- '--batch [0-9]+' <<<"$args" | cut -d' ' -f2),$(grep -oE -- '--slots [0-9]+' <<<"$args" | cut -d' ' -f2),$(nan_n 16),0"
      fi
      pace=$(grep '^RESULT_CSV4,' "$log" | tail -1 | cut -d, -f3- || true)
      [[ "$(awk -F, '{print NF}' <<<"$pace")" == "21" ]] || { notes+=(no_result_csv4); pace="$CSV4_NAN"; }
      a2p=$(grep '^RESULT_CSV5,' "$log" | tail -1 | cut -d, -f3- || true)
      [[ "$(awk -F, '{print NF}' <<<"$a2p")" == "$((N_CSV5 - 1))" ]] || { notes+=(no_result_csv5); a2p="$(nan_n $((N_CSV5 - 1))),0"; }
      disp=$(disp_row "$log")
      [[ "${disp##*,}" == "0" ]] && notes+=(scanout_fail)
      unreliable=0; (( nwarn > 0 )) && unreliable=1
      note="-"; (( ${#notes[@]} )) && note=$(IFS=';'; echo "${notes[*]}")
      echo "$line,$pace,$a2p,$disp,$r,$COOL_WAITED,$nwarn,$nerr,$rc,$unreliable,$note" >> "$RAW"
      echo "  $(grep -E '^arrived|^arrival-to-present|^display  ' "$log" | tr -s ' ' | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc note=$note"
      continue
    fi
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
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" "$DISPLAY_RUN" <<'EOF'
import csv, math, statistics, sys

DISPLAY = sys.argv[3] == "1"                          # display mode: more range columns, own verdict columns

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
if DISPLAY:
    RANGE += ["a2p_mean", "a2p_median", "a2p_p90", "a2p_max", "glass_median", "display_dropped", "late_frames"]

def row_ok(x):
    ok = x["ok"] == "1" and x["pace_ok"] in ("1", "nan")
    if DISPLAY:                                        # + arrival→present line, clean exit, planes scanout verdict
        ok = ok and x["a2p_ok"] == "1" and x["exit_code"] == "0" and x["scanout_ok"] in ("1", "nan")
    return ok

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
            v = []
            for x in g:
                try: fv = float(x[k])
                except ValueError: continue                # e.g. late_frames "unverified"
                if not math.isnan(fv): v.append(fv)
            rng += [f"{min(v):.3f}", f"{max(v):.3f}"] if v else ["nan", "nan"]
        w.writerow([lab, g[0]["mode"], g[0]["layout"], len(g)] + [f"{v:.3f}" for v in meds] + rng +
                   [sum(int(x["hailort_warnings"]) for x in g),
                    sum(int(x["hailort_errors"]) for x in g),
                    int(all(row_ok(x) for x in g))])

# Print only the columns used for the verdict (the full set is in results.csv).
show = ["fps_median", "e2e_ms_p90_median", "drop_rate_median", "max_drop_run_median", "e2e_arr_mean_median",
        "e2e_arr_p90_median", "queue_wait_p90_median", "decode_late_p90_median", "held_frames_max_median", "all_ok"]
if DISPLAY:
    show = ["fps_median", "drop_rate_median", "e2e_arr_median_median", "a2p_mean_median", "a2p_median_median",
            "a2p_p90_median", "a2p_max_median", "glass_median_median", "presented_median", "display_dropped_median",
            "late_frames_median", "all_ok"]
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
  [[ -n "$aborted" ]] && echo "ABORTED before $aborted: display not ready ($CONNECTOR / labwc DRM master)"
} >> "$OUT_DIR/env_bench.txt"
[[ -z "$aborted" ]] || { echo "session stopped before $aborted (display not ready)" >&2; exit 1; }
