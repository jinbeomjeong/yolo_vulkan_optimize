#!/usr/bin/env bash
# FFmpeg HEVC frame-thread count of yvo_pipeline — thread sweep + regression A/B (auto vs 1 thread)
#
# Why:
#   Until 2026-09-29 the app let FFmpeg pick its frame-thread count (thread_count = 0 -> 5 threads av:hevc:df0..4).
#   07_results/gst_pipeline/runs/perf_dec_threads/ showed that YVO_DEC_THREADS=1 (research knob read in
#   01_src/pipeline/main.cpp) lowers CPU by ~0.28 ms/frame at b8; the app default is now 1 (07_results/dec_threads).
#   Before changing the app default to 1, this script collects (a) a thread-count sweep at nhcw b8/s16 and
#   (b) auto-vs-1 A/B pairs across the other modes (ptr, nhwc dmabuf, b1/b4, gpu postproc, latency, display).
#
# Why not use op_point_bench.sh / display_bench.sh:
#   Their config schema has no per-run environment and a single global FRAMES; this experiment needs
#   YVO_DEC_THREADS per config and full-video and 900-frame configs in one interleaved session.
#   Cool-down, environment snapshot and loop structure follow op_point_bench.sh / gst_pipeline_bench.sh;
#   the RESULT_CSV + RESULT_CSV2 column names are display_bench.sh's (59-column header minus its 4 script fields).
#
# Usage:
#   bash 06_tools/bench/dec_threads_bench.sh                  # all, 3 runs per config
#   ONLY='^sw_' bash ...                                      # sweep only (grep -E on config id)
#   ONLY='_(auto|t1)$' SKIP_DISPLAY=1 bash ...                # A/B pairs without the display pair
#   DRYRUN=1 RUNS=1 bash ...                                  # print every run's exact command line and exit
#                                                             # (no lock, no NPU/GPU, no temperature probe, writes nothing)
#   CHECK_ONLY=1 bash ...                                     # binary check only (lib/common.sh)
#   BIN=/path/to/yvo_pipeline bash ...                        # pin binary (guards against rebuilds mid-measurement)
#   APPEND=1 FORCE=1 bash ...                                 # append to existing results_raw.csv
#   COOL_C=65 COOL_MAX_S=120 HOT_C=75 COOL_EVERY=1 bash ...   # cool-down conditions (see below)
#   DISP_TIMEOUT_S=90 DISP_OUTPUT=HDMI-A-1 bash ...           # display-run guard (see below)
#
# Config schema: id|threads|frames|args
#   threads = auto -> YVO_DEC_THREADS=0 for that child (FFmpeg auto; the app default is 1 since 2026-09-29)
#             N    -> YVO_DEC_THREADS=N only for that child (env YVO_DEC_THREADS=N)
#   frames  = all  -> no --frames (full video, 1816 frames) | N -> --frames N
#   args    = yvo_pipeline arguments; --video/--hef (absolute) and --quiet --csv-label <id> are added by this script.
#
# Run order: repetition (run) is the outer loop, config the inner loop (spreads thermal bias evenly across configs).
# Cool-down: COOL_EVERY=1 (default) cools the Hailo chip to <= COOL_C before **every** run, because the A/B deltas
#       looked for here (~0.3 ms/frame CPU) are smaller than the thermal drift between runs. With COOL_EVERY=0 only
#       the measure-skill rule applies (chip > HOT_C, or the previous run's hailort.log had temperature/clock warnings).
#       The previous-run-warning cool-down applies in both modes.
# Reliability: a run whose hailort.log has temperature/throttle/clock warnings is marked unreliable=1 (measure skill §4).
# Display configs (--display on):
#   - skipped entirely (no rows) when there is no Wayland socket or SKIP_DISPLAY=1; recorded in env_bench.txt.
#   - before each display run, wlr-randr must list DISP_OUTPUT (HDMI-A-1). With only a headless output the swapchain
#     path hangs in vkQueuePresentKHR and ignores SIGTERM (verify skill §3-1), so that run is skipped and recorded
#     as a row with exit_code=skip, note=skip_no_<output>.
#   - each display run is wrapped in `timeout -s KILL $DISP_TIMEOUT_S`; a kill shows up as exit_code 137 (note=timeout_kill).
# Exit codes of yvo_pipeline: 0 pass / 1 fail / 2 bad args / 128+signo interrupted / 137 killed by the timeout.
# Output (default OUT_DIR = 07_results/dec_threads):
#   $OUT_DIR/env_bench.txt        environment before and after (kernel, temps, throttled, clocks, v3d_freq, bootloader,
#                                 PCIe, packages, git HEAD, binary md5/mtime, Wayland, configs, skipped runs)
#   $OUT_DIR/raw/<id>_run<N>/{stdout.log,hailort.log}  per-run logs (the run directory is the CWD)
#   $OUT_DIR/raw/_session/hailort.log                  temperature probe log
#   $OUT_DIR/results_raw.csv      per run: 10 script columns + RESULT_CSV (22) + RESULT_CSV2 after label without its
#                                 duplicate ok (33) = 65 columns. Missing/malformed lines -> nan fields and ok=0.
#   $OUT_DIR/results.csv          per config: NaN-safe median/min/max of the key metrics, runs_ok, all_ok
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, package versions, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/dec_threads}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
COOL_EVERY="${COOL_EVERY:-1}"          # 1: cool to COOL_C before every run; 0: only the HOT_C / warning rule
APPEND="${APPEND:-0}"
DRYRUN="${DRYRUN:-0}"
SKIP_DISPLAY="${SKIP_DISPLAY:-0}"
DISP_TIMEOUT_S="${DISP_TIMEOUT_S:-90}" # verify skill §3-1: display runs are always wrapped in timeout -s KILL 90
DISP_OUTPUT="${DISP_OUTPUT:-HDMI-A-1}" # output wlr-randr must list before a display run

require_bin "$BIN" yvo_pipeline
check_only_exit

# The caller's YVO_DEC_THREADS must not leak into any run. Every child gets an explicit env YVO_DEC_THREADS=0|N,
# but the caller's value is still recorded in env_bench.txt.
CALLER_DEC_THREADS="${YVO_DEC_THREADS-<unset>}"

# id|threads|frames|args
configs=(
  # Sweep: nhcw b8/s16 host postproc, full video. sw_auto is FFmpeg auto (thread_count 0 -> 5 threads).
  "sw_t1            |1   |all|--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"
  "sw_t2            |2   |all|--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"
  "sw_t3            |3   |all|--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"
  "sw_t4            |4   |all|--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"
  "sw_auto          |auto|all|--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"
  # Regression A/B (auto vs 1). Arguments are those of pipeline_bench.sh (ptr_b8, dmabuf_b8, nhcw_b1, nhcw_b4),
  # op_point_bench.sh (b2s4 + display_bench.sh's disp_off, configs_lat) and display_bench.sh (disp_source).
  "ptr_b8_auto      |auto|all|--mode ptr --layout nhwc --batch 8 --slots 16 --postproc host"
  "ptr_b8_t1        |1   |all|--mode ptr --layout nhwc --batch 8 --slots 16 --postproc host"
  "nhwc_b8_auto     |auto|all|--mode dmabuf --layout nhwc --batch 8 --slots 16 --postproc host"
  "nhwc_b8_t1       |1   |all|--mode dmabuf --layout nhwc --batch 8 --slots 16 --postproc host"
  "nhcw_b1_auto     |auto|all|--mode dmabuf --layout nhcw --batch 1 --slots 2 --postproc host"
  "nhcw_b1_t1       |1   |all|--mode dmabuf --layout nhcw --batch 1 --slots 2 --postproc host"
  "nhcw_b4_auto     |auto|all|--mode dmabuf --layout nhcw --batch 4 --slots 8 --postproc host"
  "nhcw_b4_t1       |1   |all|--mode dmabuf --layout nhcw --batch 4 --slots 8 --postproc host"
  "gpu_b2s4_auto    |auto|900|--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu"
  "gpu_b2s4_t1      |1   |900|--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu"
  "lat_b1s1d1_auto  |auto|900|--mode dmabuf --layout nhcw --batch 1 --slots 1 --gpu-depth 1 --postproc host"
  "lat_b1s1d1_t1    |1   |900|--mode dmabuf --layout nhcw --batch 1 --slots 1 --gpu-depth 1 --postproc host"
  "lat_b2s2d1_auto  |auto|900|--mode dmabuf --layout nhcw --batch 2 --slots 2 --gpu-depth 1 --postproc host"
  "lat_b2s2d1_t1    |1   |900|--mode dmabuf --layout nhcw --batch 2 --slots 2 --gpu-depth 1 --postproc host"
  "disp_source_auto |auto|900|--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu --display on --display-fullscreen --display-pace source"
  "disp_source_t1   |1   |900|--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu --display on --display-fullscreen --display-pace source"
)

# Display configs need Wayland. Same detection as op_point_bench.sh / display_bench.sh (reads only XDG_RUNTIME_DIR).
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
  _wl=$(ls -t "$XDG_RUNTIME_DIR"/wayland-[0-9]* 2>/dev/null | head -1 || true)
  [[ -n "$_wl" ]] && export WAYLAND_DISPLAY="$(basename "$_wl")"
fi
HAVE_WAYLAND=0; [[ -n "${WAYLAND_DISPLAY:-}" ]] && HAVE_WAYLAND=1
HAVE_WLR_RANDR=0; command -v wlr-randr >/dev/null 2>&1 && HAVE_WLR_RANDR=1

is_display() { [[ " $1 " == *" --display on "* ]]; }

# Config parsing + validation, then ONLY / display filtering.
selected=()
config_skips=()
for c in "${configs[@]}"; do
  IFS='|' read -r id threads frames args <<<"$c"
  id="${id// /}"; threads="${threads// /}"; frames="${frames// /}"
  [[ "$threads" == "auto" || "$threads" =~ ^[1-9][0-9]*$ ]] || { echo "bad threads '$threads' in config $id" >&2; exit 1; }
  [[ "$frames" == "all" || "$frames" =~ ^[1-9][0-9]*$ ]] || { echo "bad frames '$frames' in config $id" >&2; exit 1; }
  grep -Eq "$ONLY" <<<"$id" || continue
  if is_display "$args"; then
    if [[ "$SKIP_DISPLAY" == "1" ]]; then config_skips+=("$id: SKIP_DISPLAY=1"); continue; fi
    if (( ! HAVE_WAYLAND )); then config_skips+=("$id: no Wayland socket in $XDG_RUNTIME_DIR"); continue; fi
  fi
  selected+=("$id|$threads|$frames|$args")
done
(( ${#selected[@]} > 0 )) || { echo "ONLY='$ONLY' matches no runnable config id" >&2; exit 1; }
for s in ${config_skips[@]+"${config_skips[@]}"}; do echo "[skip config] $s"; done

# build_cmd <id> <threads> <frames> <args> -> fills cmd[]
build_cmd() {
  local id="$1" threads="$2" frames="$3" args="$4" argv=()
  read -ra argv <<<"$args"
  cmd=()
  if is_display "$args"; then cmd+=(timeout -s KILL "$DISP_TIMEOUT_S"); fi
  if [[ "$threads" == "auto" ]]; then cmd+=(env YVO_DEC_THREADS=0)
  else cmd+=(env "YVO_DEC_THREADS=$threads"); fi
  cmd+=("$BIN" --video "$VIDEO" --hef "$HEF" "${argv[@]}")
  if [[ "$frames" != "all" ]]; then cmd+=(--frames "$frames"); fi
  cmd+=(--quiet --csv-label "$id")
}

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN: $(basename "$0") OUT_DIR=$OUT_DIR RUNS=$RUNS ONLY='$ONLY' COOL_EVERY=$COOL_EVERY SKIP_DISPLAY=$SKIP_DISPLAY"
  echo "DRYRUN: wayland=$([[ $HAVE_WAYLAND == 1 ]] && echo "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" || echo none)" \
       "wlr-randr=$([[ $HAVE_WLR_RANDR == 1 ]] && echo yes || echo no) caller YVO_DEC_THREADS=$CALLER_DEC_THREADS"
  if [[ -d "$OUT_DIR" ]] && [[ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "DRYRUN: note — OUT_DIR is not empty; the real run stops here without FORCE=1"
  fi
  for ((r = 1; r <= RUNS; r++)); do
    for c in "${selected[@]}"; do
      IFS='|' read -r id threads frames args <<<"$c"
      build_cmd "$id" "$threads" "$frames" "$args"
      if is_display "$args"; then
        echo "[run $r/$RUNS $id] (real run first requires: wlr-randr lists $DISP_OUTPUT, else skip)"
      fi
      printf '[run %d/%d %s] cd %q && ' "$r" "$RUNS" "$id" "$OUT_DIR/raw/${id}_run${r}"
      printf '%q ' "${cmd[@]}"; printf '> stdout.log 2>&1\n'
    done
  done
  exit 0
fi

guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

# wlr_outputs: "NAME:enabled NAME:disabled ..." from wlr-randr (one line), or empty.
wlr_outputs() {
  (( HAVE_WLR_RANDR )) || return 0
  timeout 5 wlr-randr 2>/dev/null | awk '
    /^[^ \t]/ { if (n != "") printf "%s:%s ", n, e; n = $1; e = "enabled" }
    /^[ \t]+Enabled:/ { e = ($2 == "no") ? "disabled" : "enabled" }
    END { if (n != "") printf "%s:%s", n, e }' || true
}

# Environment snapshot (before / after).
env_snapshot() {
  local dirty=""
  [[ -n "$(git -C "$PROJ" status --porcelain --untracked-files=no 2>/dev/null)" ]] && dirty=" (tracked changes)"
  echo "# dec_threads_bench $1 $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  throttled
  vcgencmd measure_clock arm 2>/dev/null || true
  vcgencmd get_config v3d_freq 2>/dev/null || true
  echo "bootloader: $(vcgencmd bootloader_version 2>/dev/null | head -1)"
  pcie_link
  pkg_versions "${PKGS_DISPLAY_BENCH[@]}"
  echo "git HEAD: $(git -C "$PROJ" rev-parse HEAD 2>/dev/null || echo '?')${dirty}"
  echo "bin: $BIN md5 $(md5sum "$BIN" | cut -d' ' -f1) mtime $(stat -c %y "$BIN")"
  if (( HAVE_WAYLAND )); then echo "wayland: $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY, wlr-randr outputs: $(wlr_outputs || true)"
  else echo "wayland: none — display configs skipped"; fi
}

{
  env_snapshot before
  echo "video: $VIDEO"
  echo "hef: $HEF"
  echo "runs: $RUNS only: $ONLY cool: COOL_C=$COOL_C COOL_MAX_S=$COOL_MAX_S HOT_C=$HOT_C COOL_EVERY=$COOL_EVERY"
  echo "display guard: output $DISP_OUTPUT required, timeout -s KILL $DISP_TIMEOUT_S, SKIP_DISPLAY=$SKIP_DISPLAY"
  echo "caller YVO_DEC_THREADS: $CALLER_DEC_THREADS (ignored: every child gets env YVO_DEC_THREADS=0 (auto) or =N)"
  echo "env: $(env | grep -E '^(VK_|HAILO|AV_|FFREPORT)' | paste -sd ' ' || true)"
  echo "configs (id|threads|frames|args, all):"
  for c in "${configs[@]}"; do echo "  $c"; done
  echo "selected: $(for c in "${selected[@]}"; do printf '%s ' "${c%%|*}"; done)"
  for s in ${config_skips[@]+"${config_skips[@]}"}; do echo "skipped config: $s"; done
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"
# A validation layer left in the environment silently changes the timing.
if env | grep -qE '^(VK_LOADER_LAYERS_ENABLE|VK_INSTANCE_LAYERS)='; then
  echo "WARNING: VK_*LAYERS* set in the environment (recorded in env_bench.txt)" >&2
fi

RAW="$OUT_DIR/results_raw.csv"
# 10 script columns + RESULT_CSV 22 + RESULT_CSV2 without label and its duplicate ok (33) = 65 columns.
# The 55 app columns use display_bench.sh's names (its 59-column header minus run,cool_wait_s,hailort_*).
#   ⚠ If main.cpp appends RESULT_CSV2 fields, N_CSV2 and this header must change together.
SCRIPT_COLS="id,threads,run,exit_code,cool_wait_s,hailort_warnings,hailort_errors,unreliable,hailo_pre_c,note"
CSV1_COLS="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok"
CSV2_COLS="postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,display,sync,pace,disp_convert_mean,disp_convert_p90,disp_overlay_mean,present_interval_median,e2e_glass_mean,e2e_glass_p90,presented,display_dropped,recreates,held_frames_max,slot_timeout_count,nms,gpu_nms_mean,gpu_nms_p90,nms_fallback,nms_verify_mismatch,scanout_zero_copy,plane_color_ok,hvs_underrun_delta,overlay_skipped,frame_cb_timeouts,zero_copy_feedback"
N_CSV1=22   # fields after "RESULT_CSV,"  (label ... ok)
N_CSV2=35   # fields after "RESULT_CSV2," (label + 10 postproc incl. ok + 14 display + 5 NMS + 6 scanout)
CSV_HEADER="$SCRIPT_COLS,$CSV1_COLS,$CSV2_COLS"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

nan_list() { local n="$1" i out=""; for ((i = 0; i < n; i++)); do out+="${out:+,}nan"; done; printf '%s' "$out"; }
arg_val() {   # arg_val <args> <flag> <default>
  local a=() i
  read -ra a <<<"$1"
  for ((i = 0; i + 1 < ${#a[@]}; i++)); do [[ "${a[i]}" == "$2" ]] && { printf '%s' "${a[i+1]}"; return; }; done
  printf '%s' "$3"
}

# make_row <id> <threads> <run> <rc> <cool> <nwarn> <nerr> <t_pre> <run_dir> <args> -> one results_raw.csv line
#   A missing or malformed RESULT_CSV / RESULT_CSV2 line becomes nan fields and forces ok=0.
make_row() {
  local id="$1" threads="$2" r="$3" rc="$4" cool="$5" nwarn="$6" nerr="$7" t_pre="$8" dir="$9" args="${10}"
  local log="$dir/stdout.log" l1="" l2="" notes=() unreliable=0
  (( nwarn > 0 )) && unreliable=1
  if [[ -f "$log" ]]; then
    l1=$(grep '^RESULT_CSV,' "$log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
    l2=$(grep '^RESULT_CSV2,' "$log" | tail -1 | sed 's/^RESULT_CSV2,//' || true)
  fi
  [[ "$rc" == "137" ]] && notes+=(timeout_kill)
  if [[ -z "$l1" || "$(awk -F, '{print NF}' <<<"$l1")" != "$N_CSV1" ]]; then
    notes+=(no_result_csv)
    l1="$id,$(arg_val "$args" --mode dmabuf),$(arg_val "$args" --layout nhwc),$(arg_val "$args" --batch nan),$(arg_val "$args" --slots nan),$(nan_list 16),0"
  fi
  if [[ -z "$l2" || "$(awk -F, '{print NF}' <<<"$l2")" != "$N_CSV2" ]]; then
    notes+=(no_result_csv2)
    l2="$(nan_list $((N_CSV2 - 2)))"
    l1="${l1%,*},0"   # CSV2 missing -> ok=0
  else
    # drop label (field 1) and the duplicate ok (field 10)
    l2=$(awk -F, 'BEGIN{OFS=","}{ out=""; for(i=2;i<=NF;i++){ if(i==10) continue;
                                   out = (out=="" ? $i : out OFS $i) } print out }' <<<"$l2")
  fi
  (( rc != 0 )) && [[ "$rc" != "137" ]] && notes+=("rc_$rc")
  local note="-"; (( ${#notes[@]} )) && note=$(IFS=';'; echo "${notes[*]}")
  echo "$id,$threads,$r,$rc,$cool,$nwarn,$nerr,$unreliable,$t_pre,$note,$l1,$l2"
}

prev_warn=0
skipped_runs=()

# cool_before_run: sets t_pre (chip temp right before the run) and COOL_WAITED.
cool_before_run() {
  local t_now
  COOL_WAITED=0
  t_now=$(hailo_temp)
  t_pre=$t_now
  if (( prev_warn > 0 )); then
    cool_down "hailort.log warnings: ${prev_warn}"; t_pre=$(hailo_temp)
  elif [[ "$COOL_EVERY" == "1" ]] && awk -v t="$t_now" -v c="$COOL_C" 'BEGIN{exit !(t=="nan" || t>c)}'; then
    cool_down "every run, chip ${t_now} C"; t_pre=$(hailo_temp)
  elif awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then
    cool_down "chip temp ${t_now} C > ${HOT_C} C"; t_pre=$(hailo_temp)
  fi
}

for ((r = 1; r <= RUNS; r++)); do
  for c in "${selected[@]}"; do
    IFS='|' read -r id threads frames args <<<"$c"
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    if is_display "$args"; then
      outs=$(wlr_outputs)
      if [[ " $outs " != *" $DISP_OUTPUT:enabled "* ]]; then
        why="skip_no_${DISP_OUTPUT}"
        (( HAVE_WLR_RANDR )) || outs="wlr-randr not installed"
        echo "[skip] run $r/$RUNS $id — wlr-randr lists no enabled $DISP_OUTPUT (outputs: ${outs:-none}); the swapchain path would hang"
        skipped_runs+=("$id run$r: $why (wlr-randr: ${outs:-none})")
        rm -rf "$run_dir"; mkdir -p "$run_dir"
        echo "skipped: wlr-randr outputs: ${outs:-none}" > "$run_dir/stdout.log"
        echo "$id,$threads,$r,skip,0,0,0,0,nan,$why,$id,$(arg_val "$args" --mode dmabuf),$(arg_val "$args" --layout nhwc),$(arg_val "$args" --batch nan),$(arg_val "$args" --slots nan),$(nan_list 16),0,$(nan_list $((N_CSV2 - 2)))" >> "$RAW"
        continue
      fi
    fi
    cool_before_run
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    build_cmd "$id" "$threads" "$frames" "$args"
    echo "[$(date +%T)] run $r/$RUNS $id threads=$threads (hailo ${t_pre} C, cooled ${COOL_WAITED}s)"
    set +e
    (cd "$run_dir" && "${cmd[@]}" > stdout.log 2>&1)
    rc=$?
    set -e
    read -r nwarn nerr < <(count_hailort_warnings "$run_dir/hailort.log")
    prev_warn=$nwarn
    row=$(make_row "$id" "$threads" "$r" "$rc" "$COOL_WAITED" "$nwarn" "$nerr" "$t_pre" "$run_dir" "$args")
    echo "$row" >> "$RAW"
    note=$(cut -d, -f10 <<<"$row")
    if [[ "$note" != "-" ]]; then echo "  rc=$rc note=$note (see stdout.log)"; tail -3 "$run_dir/stdout.log" || true; fi
    echo "  $(grep -E '^wall|^cpu |^held frames' "$run_dir/stdout.log" | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc"
  done
done

# Per-config aggregation (NaN-safe) + console comparison.
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'PYEOF'
import csv, math, statistics, sys


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


def ival(x):
    v = fnum(x)
    return 0 if math.isnan(v) else int(v)


def fmt(v, d=3):
    return "nan" if math.isnan(v) else f"{v:.{d}f}"


rows = list(csv.DictReader(open(sys.argv[1], newline="")))
if not rows:
    print("no rows"); sys.exit(0)
order, groups = [], {}
for r in rows:
    if r["id"] not in groups:
        order.append(r["id"]); groups[r["id"]] = []
    groups[r["id"]].append(r)

# metric column -> printed decimals
METRICS = [("fps", 2), ("cpu_ms_per_frame", 3), ("gpu_ms_mean", 3), ("npu_ms_mean", 2), ("e2e_ms_mean", 2),
           ("e2e_ms_p90", 2), ("detections", 0), ("held_frames_max", 0),
           ("e2e_glass_mean", 2), ("e2e_glass_p90", 2), ("presented", 0), ("display_dropped", 0)]
summary = {}
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")             # no CRLF (measure skill §5)
    hdr = ["id", "threads", "mode", "layout", "batch", "slots", "postproc", "display", "frames_median",
           "runs", "runs_ok", "exit_codes", "unreliable_runs", "hailort_warnings", "hailort_errors",
           "cool_wait_s_total", "hailo_pre_c_max", "notes"]
    for k, _ in METRICS:
        hdr += [f"{k}_median", f"{k}_min", f"{k}_max"]
    hdr += ["all_ok"]
    w.writerow(hdr)
    for cid in order:
        g = groups[cid]
        # ok run: exit 0, app ok=1, no hailort temperature/clock warning, and no silent display fallback (offscreen)
        ok = [x for x in g if x["exit_code"] == "0" and x["ok"] == "1" and x["hailort_warnings"] == "0"
              and x["display"] in ("off", "wayland")]
        runs = [x for x in g if x["exit_code"] != "skip"]
        stats = {}
        for k, _ in METRICS:
            vals = [fnum(x[k]) for x in g]
            stats[k] = (nmed(vals), nmin(vals), nmax(vals))
        disp = sorted({x["display"] for x in g if x["display"] not in ("nan", "")}) or ["nan"]
        notes = sorted({n for x in g for n in x["note"].split(";") if n not in ("-", "")})
        all_ok = int(len(ok) == len(g))
        summary[cid] = {"threads": g[0]["threads"], "stats": stats, "all_ok": all_ok, "runs_ok": len(ok),
                        "runs": len(g), "display": disp}
        row = [cid, g[0]["threads"], g[0]["mode"], g[0]["layout"], g[0]["batch"], g[0]["slots"],
               (ok[0] if ok else g[0])["postproc"], "/".join(disp), fmt(nmed(fnum(x["frames"]) for x in g), 0),
               len(runs), len(ok), ";".join(x["exit_code"] for x in g),
               sum(x["unreliable"] == "1" for x in g),
               sum(ival(x["hailort_warnings"]) for x in g),
               sum(ival(x["hailort_errors"]) for x in g),
               sum(ival(x["cool_wait_s"]) for x in g),
               fmt(nmax(fnum(x["hailo_pre_c"]) for x in g), 1), ";".join(notes) or "-"]
        for k, d in METRICS:
            row += [fmt(v, d) for v in stats[k]]
        row += [all_ok]
        w.writerow(row)


def med(cid, k):
    return summary[cid]["stats"][k][0]


def table(title, hdr, lines):
    print(title)
    wid = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)] if lines else [len(h) for h in hdr]
    print(" | ".join(h.ljust(x) for h, x in zip(hdr, wid)))
    print("-+-".join("-" * x for x in wid))
    for l in lines:
        print(" | ".join(c.ljust(x) for c, x in zip(l, wid)))
    print()


# Sweep table, ordered by thread count (auto last), deltas vs sw_auto.
sw = [c for c in order if c.startswith("sw_")]
sw.sort(key=lambda c: (summary[c]["threads"] == "auto", fnum(summary[c]["threads"])))
if sw:
    base = "sw_auto" if "sw_auto" in summary else None
    lines = []
    for c in sw:
        s = summary[c]
        fr = s["stats"]["fps"]
        cpu = med(c, "cpu_ms_per_frame")
        dcpu = cpu - med(base, "cpu_ms_per_frame") if base else math.nan
        lines.append([c, s["threads"], f"{fmt(fr[0], 2)} ({fmt(fr[1], 2)}..{fmt(fr[2], 2)})", fmt(cpu),
                      fmt(dcpu), fmt(med(c, "npu_ms_mean"), 2), fmt(med(c, "e2e_ms_mean"), 2),
                      fmt(med(c, "e2e_ms_p90"), 2), fmt(med(c, "held_frames_max"), 0),
                      fmt(med(c, "detections"), 0), f"{s['runs_ok']}/{s['runs']}"])
    table("== sweep (nhcw b8/s16, medians; d_cpu = vs sw_auto) ==",
          ["id", "threads", "fps (min..max)", "cpu_ms/f", "d_cpu", "npu_ms", "e2e_mean", "e2e_p90",
           "held_max", "dets", "ok"], lines)

# A/B pairs: <prefix>_auto vs <prefix>_t1, delta = t1 - auto.
AB = [("fps", 2), ("cpu_ms_per_frame", 3), ("e2e_ms_mean", 2), ("e2e_ms_p90", 2), ("held_frames_max", 0),
      ("detections", 0), ("e2e_glass_mean", 2), ("display_dropped", 0)]
DISP_KEYS = {"e2e_glass_mean", "display_dropped"}
lines = []
for c in order:
    if not c.endswith("_auto"):
        continue
    p = c[: -len("_auto")]
    t1 = p + "_t1"
    if t1 not in summary:
        continue
    line = [p]
    for k, d in AB:
        a, b = med(c, k), med(t1, k)
        if (math.isnan(a) and math.isnan(b)) or (k in DISP_KEYS and summary[c]["display"] == ["off"]):
            line.append("-")
            continue
        rel = f" ({(b - a) / a * 100:+.1f}%)" if a and not math.isnan(a) and not math.isnan(b) and k != "detections" else ""
        line.append(f"{fmt(a, d)}->{fmt(b, d)} {'+' if b - a >= 0 else ''}{fmt(b - a, d)}{rel}")
    line.append(f"{summary[c]['all_ok']}/{summary[t1]['all_ok']}")
    lines.append(line)
if lines:
    table("== A/B (t1 - auto, medians) ==", ["pair"] + [k for k, _ in AB] + ["all_ok a/t1"], lines)

bad = [c for c in order if not summary[c]["all_ok"]]
for c in order:
    # "off" = display disabled; anything else but wayland (e.g. offscreen) means a display run silently fell back.
    if any(x["display"] not in ("nan", "", "off", "wayland") for x in groups[c]):
        print(f"WARNING: {c} ran with display backend {'/'.join(summary[c]['display'])} (expected wayland)")
print("all_ok=0: " + (", ".join(bad) if bad else "none"))
PYEOF

{
  env_snapshot after
  echo "bin md5 (after measurement): $(md5sum "$BIN" | cut -d' ' -f1)"
  for s in ${skipped_runs[@]+"${skipped_runs[@]}"}; do echo "skipped run: $s"; done
} >> "$OUT_DIR/env_bench.txt"
