#!/usr/bin/env bash
# M4 — HDMI display-integrated e2e measurement (plan stages 4-5).
#
# Why not use op_point_bench.sh:
#   That script collects only the 22 `RESULT_CSV` fields. Display metrics (disp_convert, present interval,
#   glass-to-glass, held_frames_max ...) are in the **14 fields appended at the end** of the `RESULT_CSV2` line.
#   So this separate script joins the two lines into one row. Cool-down, environment snapshot and loop structure are
#   copied verbatim from op_point_bench.sh (unmodified).
#
# ⚠ Measurements must use --display-fullscreen. Windowed mode falls onto the uniform-scale path and
#   display conversion becomes 2.6x slower with no error or warning.
# ⚠ Split the axes: throughput verdicts use pace=asap, latency/drop verdicts use pace=source.
#   With pace=source FPS is pinned to the source 30, making the "FPS ±2%" verdict meaningless.
#
# Compositor CPU (2026-10-16): cpu_ms_per_frame is the app process only (getrusage RUSAGE_SELF). To get
#   "app + compositor" CPU/frame from the same runs, every run — display off too, since the idle compositor share is
#   part of a fair comparison — copies /proc/<labwc pid>/stat right before and right after yvo_pipeline into
#   raw/<id>_run<N>/labwc_stat_{pre,post}.txt (same method and file names as display_planes_bench.sh) and appends
#   labwc_cpu_ms (user+sys delta) and labwc_cpu_ms_per_frame (/ frames) as the last 2 CSV columns.
#   app + compositor = cpu_ms_per_frame + labwc_cpu_ms_per_frame. No labwc (headless) or a labwc that exited or was
#   replaced during the run -> both nan; the run itself goes on. Caveats: the labwc window is the whole yvo_pipeline
#   lifetime (init and teardown included), wider than the getrusage window of cpu_ms_per_frame; tick resolution is
#   1/CLK_TCK s (10 ms) per endpoint.
#
# System-wide CPU (2026-10-16): 06_tools/python/sys_cpu_window.py (system python3, stdlib only) per run, every config:
#   1. After the cool-down gate, an idle baseline with the app not running: --duration IDLE_S (default 10 s) ->
#      raw/<run>/syscpu_idle.{json,csv}. IDLE_S=0 skips it (idle_sched_ms_per_s and sys_net_ms_per_frame = nan).
#   2. Window mode, aligned to the app's own CPU window: the sampler opens raw/<run>/syscpu_{ctl,ack}.fifo and writes
#      raw/<run>/syscpu_ready; the app runs with YVO_PERF_CTL/YVO_PERF_ACK and announces enable/disable at exactly
#      the c0..c1 interval of cpu_ms_per_frame (perf_window() in 01_src/toolkit/tool_util.hpp), the sampler takes a
#      /proc snapshot at each and acks -> raw/<run>/syscpu.{json,csv}. Not ready within 5 s, a sampler exit code
#      other than 0 (3 = partial window) or a sampler still running 10 s after the app -> this run's sys_* = nan and
#      the run itself goes on (without YVO_PERF_* when not ready). The FIFOs are removed after the run.
#   Columns (per-frame = / frames of RESULT_CSV field 6): sys_win_ms, sys_{busy,sched,app,labwc,kthread,other}_ms_per_frame,
#   idle_sched_ms_per_s, sys_net_ms_per_frame = (sched_run_ms - idle_sched_ms_per_s * sys_win_ms / 1000) / frames.
#   Counting limits: busy is /proc/stat ticks and the kernel has CONFIG_IRQ_TIME_ACCOUNTING off, so hardirq time is
#   charged to whatever was interrupted (to idle: not counted at all); per-process classes are utime+stime ticks
#   (10 ms per process per endpoint); sched_run (/proc/schedstat) is ns-precise and is the column sys_net builds on.
#
# IRQ_TRACE=1 (2026-10-16): interrupt time over the same app CPU window, for the "hardirq time while idle is not
#   counted" limit above. This mode runs perf **under sudo** with irq tracepoints, system-wide, per run:
#     $SUDO perf record -a -e irq:irq_handler_{entry,exit},irq:softirq_{entry,exit} -D -1 \
#       --control fifo:raw/<run>/syscpu_{ctl,ack}.fifo -o raw/<run>/irq.perf.data -- sleep IRQ_MAX_S (600)
#   in place of the sampler and of the idle baseline (sys_* columns = nan; results_raw.csv keeps its 70 columns).
#   $SUDO = "sudo -A" when SUDO_ASKPASS is set, else "sudo -n" (fails at once without NOPASSWD/cached rights). The bench
#   tries "$SUDO true" before anything else and stops if it fails; no password or password file is referenced here.
#   The bench creates both FIFOs, queues "ping" and starts the app only after perf's "ack" (perf has opened both FIFOs
#   and is in its main loop, events disabled). After the app: $SUDO kill -INT <perf pid> (the process whose comm is
#   "perf" and whose command line has -o <run>/irq.perf.data; the "stop" control command if that fails), then
#   $SUDO perf script -F cpu,pid,tid,comm,time,event,trace > irq_script.txt, $SUDO chown to the bench user,
#   06_tools/python/irq_time_window.py --script irq_script.txt --frames <RESULT_CSV frames> --out irq.json, and
#   gzip -> irq_script.txt.gz. irq.perf.data is removed unless IRQ_KEEP_DATA=1. One row per run is appended to
#   OUT_DIR/irq_results_raw.csv (label,run + the parser's CSV columns; nan when the parser is missing or fails, and
#   the bench goes on); the console summary ends with the per-config uncounted_ms_per_frame median.
#   ⚠ The tracepoints add CPU work to every interrupt, so all CPU ms of an IRQ_TRACE run (app, labwc, FPS too) carry
#   that overhead: use the run for interrupt accounting only, never for headline CPU figures.
#
# Usage:
#   bash 06_tools/bench/display_bench.sh
#   RUNS=3 FRAMES=900 ONLY='disp_on' bash 06_tools/bench/display_bench.sh
#   CHECK_ONLY=1 bash 06_tools/bench/display_bench.sh                 # binary/path check only, exit
#   IDLE_S=0 bash 06_tools/bench/display_bench.sh                     # skip the per-run idle baseline
#   IRQ_TRACE=1 [IRQ_KEEP_DATA=1] [SUDO_ASKPASS=<prog>] bash ...       # irq tracepoints instead of the sampler (sudo)
#   DRYRUN=1 [IRQ_TRACE=1] RUNS=1 bash ...                             # print every run's commands, exit (no lock/files/sudo)
#   OUT_DIR=/path [FORCE=1] [APPEND=1] bash ...                        # APPEND needs the same 70-column header
#   Output: OUT_DIR (default 07_results/display_e2e) / {env_bench.txt, results_raw.csv (70 columns), results.csv,
#           raw/<id>_run<N>/{stdout.log, hailort.log, labwc_stat_{pre,post}.txt, syscpu{,_idle}.{json,csv},
#           syscpu_ready[, syscpu.err]}, raw/m5a_run1/, raw/_session/}
#   IRQ_TRACE=1: + irq_results_raw.csv; per run irq_perf.log, irq_script.txt.gz, irq.json[, irq.perf.data, irq_parse.err]
#           instead of the syscpu* files
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, package versions, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
# They used to be copy-pasted into 7 scripts and had actually diverged (e.g. pcie_link's Gen5 branch).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/display_e2e}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-900}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
APPEND="${APPEND:-0}"
IDLE_S="${IDLE_S:-10}"                                  # idle baseline seconds per run (0 = skip)
SYSCPU_PY="$PROJ/06_tools/python/sys_cpu_window.py"
SYSCPU_TIMEOUT_S="${SYSCPU_TIMEOUT_S:-1800}"            # window-mode sampler: max wait for enable+disable
APP_COMM="$(basename "$BIN" | cut -c1-15)"              # /proc/<pid>/comm is truncated to 15 characters
DRYRUN="${DRYRUN:-0}"                                   # 1 = print every run's commands and exit (no lock, files, sudo)
IRQ_TRACE="${IRQ_TRACE:-0}"                             # 1 = perf irq tracepoints instead of the sampler (header)
IRQ_KEEP_DATA="${IRQ_KEEP_DATA:-0}"                     # 1 = keep raw/<run>/irq.perf.data after perf script
IRQ_PY="${IRQ_PY:-$PROJ/06_tools/python/irq_time_window.py}"
IRQ_MAX_S="${IRQ_MAX_S:-600}"                           # perf's workload is "sleep IRQ_MAX_S": longest traceable run
IRQ_EVENTS="irq:irq_handler_entry,irq:irq_handler_exit,irq:softirq_entry,irq:softirq_exit"
IRQ_COLS="irq_window_s,hardirq_count,hardirq_ms,hardirq_idle_ms,softirq_count,softirq_ms,softirq_idle_ms,unmatched,uncounted_ms_per_frame"
IRQ_NCOLS=9                                             # fields in IRQ_COLS = fields of the parser's CSV line
IRQ_RAW="$OUT_DIR/irq_results_raw.csv"
IRQ_PID=""; IRQ_SUDO_PID=""; IRQ_READY=0; IRQ_OK=0
if [[ -n "${SUDO_ASKPASS:-}" ]]; then SUDO=(sudo -A); else SUDO=(sudo -n); fi   # never handles a password itself

# irq_record_argv / irq_script_argv <run_dir> -> IRQ_REC / IRQ_SCR: the perf command lines run under "${SUDO[@]}"
#   (shared by the real run and DRYRUN). The FIFOs are the sampler's paths, created by the bench (irq_start).
irq_record_argv() {
  IRQ_REC=(perf record -a -e "$IRQ_EVENTS" -D -1 --control "fifo:$1/syscpu_ctl.fifo,$1/syscpu_ack.fifo"
           -o "$1/irq.perf.data" -- sleep "$IRQ_MAX_S")
}
irq_script_argv() { IRQ_SCR=(perf script -i "$1/irq.perf.data" -F cpu,pid,tid,comm,time,event,trace); }

# irq_perf_pid <perf.data path>: pid of the perf process itself (not sudo, not its sleep), for "$SUDO kill -INT".
#   Command line matched with -o <path> (regex-escaped), then comm must be "perf"; prints nothing when not found.
irq_perf_pid() {
  local re p
  re=$(printf '%s' "$1" | sed 's/[][\\.*^$+?(){}|]/\\&/g')
  for p in $(pgrep -f -- "-o $re( |\$)" || true); do
    if [[ "$(cat "/proc/$p/comm" 2>/dev/null)" == "perf" ]]; then echo "$p"; return 0; fi
  done
}

# irq_abort: EXIT/INT/TERM trap of IRQ_TRACE runs; stops a perf still running so it does not outlive the bench.
irq_abort() {
  if [[ -n "$IRQ_SUDO_PID" && -n "$IRQ_PID" && -e "/proc/$IRQ_SUDO_PID" ]]; then
    "${SUDO[@]}" kill -INT "$IRQ_PID" < /dev/null 2>/dev/null || true
  fi
}

require_bin "$BIN" yvo_pipeline
[[ -f "$SYSCPU_PY" && -x /usr/bin/python3 ]] || { echo "sys_cpu_window.py or /usr/bin/python3 not found" >&2; exit 1; }
[[ "$IDLE_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "IDLE_S must be a number >= 0: '$IDLE_S'" >&2; exit 1; }
for _v in DRYRUN IRQ_TRACE IRQ_KEEP_DATA; do
  [[ "${!_v}" =~ ^[01]$ ]] || { echo "$_v must be 0 or 1: '${!_v}'" >&2; exit 1; }
done
if [[ "$IRQ_TRACE" == "1" ]]; then
  [[ "$IRQ_MAX_S" =~ ^[1-9][0-9]*$ ]] || { echo "IRQ_MAX_S must be a positive integer: '$IRQ_MAX_S'" >&2; exit 1; }
  { command -v perf && command -v gzip; } > /dev/null || { echo "IRQ_TRACE=1 needs perf and gzip in PATH" >&2; exit 1; }
  IRQ_PY_STATE="missing: irq_results_raw.csv rows will be nan (irq_script.txt.gz can be parsed later)"
  if [[ -f "$IRQ_PY" ]]; then
    _h=$(/usr/bin/python3 "$IRQ_PY" --csv-header 2>/dev/null | head -1 || true)
    if [[ "$_h" == "$IRQ_COLS" ]]; then IRQ_PY_STATE="present, --csv-header matches"
    else IRQ_PY_STATE="present, but --csv-header '$_h' != expected; rows with != $IRQ_NCOLS fields become nan"; fi
  fi
  echo "IRQ_TRACE: $(perf --version 2>&1) via '${SUDO[*]}'; parser $IRQ_PY ($IRQ_PY_STATE)"
fi
check_only_exit
if [[ "$IRQ_TRACE" == "1" && "$DRYRUN" != "1" ]] && ! "${SUDO[@]}" true < /dev/null; then
  echo "IRQ_TRACE=1 runs perf tracepoints under sudo, but '${SUDO[*]} true' failed." >&2
  echo "  Set SUDO_ASKPASS=<askpass program> (-> sudo -A), or make sudo -n work for this user, then retry." >&2
  exit 1
fi
if [[ "$DRYRUN" != "1" ]]; then   # DRYRUN: no guard, no lock, no files (it prints and exits after the configs below)
guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"
fi
if [[ "$IRQ_TRACE" == "1" && "$DRYRUN" != "1" ]]; then   # bench_lock_acquire's traps + stop a perf left running
  trap 'irq_abort; rm -f "$BENCH_LOCK"' EXIT
  trap 'irq_abort; rm -f "$BENCH_LOCK"; exit 130' INT; trap 'irq_abort; rm -f "$BENCH_LOCK"; exit 143' TERM
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
  _wl=$(ls -t "$XDG_RUNTIME_DIR"/wayland-[0-9]* 2>/dev/null | head -1 || true)
  [[ -n "$_wl" ]] && export WAYLAND_DISPLAY="$(basename "$_wl")"
fi
HAVE_WAYLAND=0; [[ -n "${WAYLAND_DISPLAY:-}" ]] && HAVE_WAYLAND=1

# id|free-form args. Operating point fixed at b2/s4 --postproc gpu; only the display axis varies.
configs=(
  "reg_host_off |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc host"
  "disp_off     |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu"
  "disp_asap    |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu --display on --display-fullscreen --display-pace asap"
  "disp_source  |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu --display on --display-fullscreen --display-pace source"
)

# DRYRUN=1: print the commands of every run (same loop filters) and exit. No lock, no files, no sudo, no NPU access
#   (hailo_temp and the cool-down are not called). Sampler lines mirror syscpu_idle/syscpu_start below.
if [[ "$DRYRUN" == "1" ]]; then
  dry() { echo "DRYRUN:   $*"; }
  echo "DRYRUN: $(basename "$0") OUT_DIR=$OUT_DIR RUNS=$RUNS FRAMES=$FRAMES ONLY='$ONLY' IDLE_S=$IDLE_S IRQ_TRACE=$IRQ_TRACE IRQ_KEEP_DATA=$IRQ_KEEP_DATA"
  echo "DRYRUN: wayland=$( (( HAVE_WAYLAND )) && echo "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" || echo none)"
  if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR" 2>/dev/null)" && "${FORCE:-0}" != "1" ]]; then
    echo "DRYRUN: note — OUT_DIR is not empty; the real run stops here without FORCE=1"
  fi
  if [[ "$IRQ_TRACE" == "1" ]]; then echo "DRYRUN: start-up sudo check"; dry "${SUDO[*]} true"; fi
  if [[ "${SKIP_M5A:-0}" != "1" ]] && (( HAVE_WAYLAND )) && [[ -x "$BIN_DIR/yvo_display_probe" ]]; then
    echo "DRYRUN: [M5a]"
    dry "cd $OUT_DIR/raw/m5a_run1 && $BIN_DIR/yvo_display_probe --probe convert --video $VIDEO --frames 300 --display-fullscreen --with-overlay --overlay-dets 5 > stdout.log 2>&1"
  fi
  dry_frames=""; [[ "$FRAMES" != "0" ]] && dry_frames="--frames $FRAMES"
  for ((r = 1; r <= RUNS; r++)); do
    for c in "${configs[@]}"; do
      IFS='|' read -r id args <<<"$c"
      id="${id// /}"
      echo "$id" | grep -Eq "$ONLY" || continue
      if [[ "$args" == *"--display on"* ]] && (( ! HAVE_WAYLAND )); then echo "DRYRUN: [skip] $id — no Wayland"; continue; fi
      d="$OUT_DIR/raw/${id}_run${r}"
      echo "DRYRUN: [$id run $r] (cool-down gate first)"
      if [[ "$IRQ_TRACE" == "1" ]]; then
        dry "mkfifo -m 600 $d/syscpu_ctl.fifo $d/syscpu_ack.fifo; queue 'ping' on ctl"
        irq_record_argv "$d"; dry "${SUDO[*]} ${IRQ_REC[*]} < /dev/null > $d/irq_perf.log 2>&1 &   # wait <= 10 s for 'ack'"
      else
        if awk -v s="$IDLE_S" 'BEGIN{exit !(s + 0 > 0)}'; then
          dry "/usr/bin/python3 $SYSCPU_PY --duration $IDLE_S --out $d/syscpu_idle.json --app-comm $APP_COMM --csv-fields > $d/syscpu_idle.csv"
        fi
        dry "/usr/bin/python3 $SYSCPU_PY --ctl $d/syscpu_ctl.fifo --ack $d/syscpu_ack.fifo --ready $d/syscpu_ready --out $d/syscpu.json --timeout $SYSCPU_TIMEOUT_S --app-comm $APP_COMM --csv-fields > $d/syscpu.csv &"
      fi
      dry "cd $d && env YVO_PERF_CTL=$d/syscpu_ctl.fifo YVO_PERF_ACK=$d/syscpu_ack.fifo $BIN --video $VIDEO --hef $HEF $args $dry_frames --quiet --csv-label $id > stdout.log 2>&1"
      if [[ "$IRQ_TRACE" == "1" ]]; then
        dry "${SUDO[*]} kill -INT <pid of perf with -o $d/irq.perf.data>"
        irq_script_argv "$d"; dry "${SUDO[*]} ${IRQ_SCR[*]} > $d/irq_script.txt"
        dry "${SUDO[*]} chown $(id -u):$(id -g) $d/irq.perf.data $d/irq_script.txt"
        [[ "$IRQ_KEEP_DATA" == "1" ]] || dry "rm -f $d/irq.perf.data"
        dry "/usr/bin/python3 $IRQ_PY --script $d/irq_script.txt --frames <RESULT_CSV frames> --out $d/irq.json >> $IRQ_RAW (with label,run)"
        dry "gzip -f $d/irq_script.txt"
      fi
    done
  done
  exit 0
fi

# ── end of copy ──

# M5a — GPU baseline with the NPU idle. Taken **right before M4, in the same thermal state**. Without it, one can't tell
# whether M4's GPU time is due to contention or code. No cool-down (intentional).
if [[ "${SKIP_M5A:-0}" != "1" ]] && (( HAVE_WAYLAND )); then
  PROBE="$BIN_DIR/yvo_display_probe"
  if [[ -x "$PROBE" ]]; then
    echo "[M5a] GPU baseline with NPU idle (yvo_display_probe --probe convert, fullscreen, 5 overlays)"
    # One run = one folder (STRUCTURE.md §3): raw/m5a_run1/{stdout.log (full), m5a_baseline.txt (excerpt)}
    # Before 2026-09-22 these were raw/m5a/m5a.log and m5a_baseline.txt at the root.
    m5a_dir="$OUT_DIR/raw/m5a_run1"
    mkdir -p "$m5a_dir"
    # ⚠ Pass --video as an absolute path. The probe runs in raw/m5a_run1, which breaks the default relative path.
    (cd "$m5a_dir" && "$PROBE" --probe convert --video "$VIDEO" --frames 300 --display-fullscreen \
        --with-overlay --overlay-dets 5 > stdout.log 2>&1) || true
    grep -E "display path|on-screen shader|on-screen overlay|present_wait|presented" \
      "$m5a_dir/stdout.log" | tee "$m5a_dir/m5a_baseline.txt"
  fi
fi

{
  echo "# display_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  for g in /sys/devices/system/cpu/cpufreq/policy*; do echo "cpufreq $(basename "$g"): governor $(cat "$g/scaling_governor") cur_khz $(cat "$g/scaling_cur_freq")"; done
  pcie_link
  pkg_versions "${PKGS_DISPLAY_BENCH[@]}"   # labwc/libwlroots matter: the wayland-planes COLOR_RANGE inversion is a compositor behavior
  echo "bin: $BIN"
  echo "bin md5: $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
  if (( HAVE_WAYLAND )); then echo "wayland: $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
  else echo "wayland: none — configs with --display are skipped"; fi
  if [[ "$IRQ_TRACE" == "1" ]]; then
    echo "irq_trace: 1 (perf irq tracepoints instead of sys_cpu_window.py; CPU ms carry tracepoint overhead)"
    echo "irq perf: $(perf --version 2>&1), sudo: ${SUDO[*]}, events: $IRQ_EVENTS, max_s: $IRQ_MAX_S, keep_data: $IRQ_KEEP_DATA"
    echo "irq parser: $IRQ_PY ($IRQ_PY_STATE)"
  fi
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
# 22 RESULT_CSV fields + 33 RESULT_CSV2 fields after label + 4 script fields + 2 labwc fields + 9 sys_* fields = 70 columns
#   RESULT_CSV2 = 8 postproc fields (ok duplicates RESULT_CSV's ok, so the awk below drops it) + 14 display + 5 GPU NMS fields
#                 + 6 plane-scanout fields ("-" except for wayland-planes / kms; see the RESULT_CSV2 comment in main.cpp)
#   labwc_cpu_ms, labwc_cpu_ms_per_frame = compositor CPU over the run (labwc_cpu_cols in lib/common.sh; nan if no labwc).
#   sys_* .. sys_net_ms_per_frame = system-wide CPU over the app's CPU window (sys_cpu_cols below; header comment).
#   ⚠ That line is appended wholesale via cut -f3-, so if main.cpp appends fields this header must change too.
#   The APPEND=1 header check refuses older headers: 48 columns (no NMS fields, 07_results/display_e2e/),
#   53 columns (no scanout fields), 59 columns (no labwc fields, before 2026-10-16) and 61 columns (no sys_* fields).
CSV_HEADER="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,display,sync,pace,disp_convert_mean,disp_convert_p90,disp_overlay_mean,present_interval_median,e2e_glass_mean,e2e_glass_p90,presented,display_dropped,recreates,held_frames_max,slot_timeout_count,nms,gpu_nms_mean,gpu_nms_p90,nms_fallback,nms_verify_mismatch,scanout_zero_copy,plane_color_ok,hvs_underrun_delta,overlay_skipped,frame_cb_timeouts,zero_copy_feedback,run,cool_wait_s,hailort_warnings,hailort_errors,labwc_cpu_ms,labwc_cpu_ms_per_frame,sys_win_ms,sys_busy_ms_per_frame,sys_sched_ms_per_frame,sys_app_ms_per_frame,sys_labwc_ms_per_frame,sys_kthread_ms_per_frame,sys_other_ms_per_frame,idle_sched_ms_per_s,sys_net_ms_per_frame"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi
if [[ "$IRQ_TRACE" == "1" ]]; then   # label,run + the parser's columns (irq_parse below); same APPEND rule as RAW
  if [[ "$APPEND" == "1" && -f "$IRQ_RAW" ]]; then
    head -1 "$IRQ_RAW" | grep -qx "label,run,$IRQ_COLS" || { echo "APPEND=1 but the existing $IRQ_RAW header differs" >&2; exit 1; }
  else
    echo "label,run,$IRQ_COLS" > "$IRQ_RAW"
  fi
fi

prev_warn=0
frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

# ── System-wide CPU (sys_cpu_window.py; see the header comment) ─────────────────────
# syscpu_alive <pid>: running and not a zombie (kill -0 also succeeds on an unreaped child).
syscpu_alive() { local s; s=$(cut -d' ' -f3 "/proc/$1/stat" 2>/dev/null) || return 1; [[ -n "$s" && "$s" != Z ]]; }

# syscpu_idle <run_dir>: idle baseline, app not running -> SYS_IDLE_CSV (CSV line; "" when IDLE_S=0 or on failure).
syscpu_idle() {
  SYS_IDLE_CSV=""
  awk -v s="$IDLE_S" 'BEGIN{exit !(s + 0 > 0)}' || return 0
  if /usr/bin/python3 "$SYSCPU_PY" --duration "$IDLE_S" --out "$1/syscpu_idle.json" --app-comm "$APP_COMM" \
       --csv-fields > "$1/syscpu_idle.csv" 2>> "$1/syscpu.err"; then
    SYS_IDLE_CSV=$(head -1 "$1/syscpu_idle.csv")
  else
    echo "  warning: idle baseline failed (see syscpu.err, syscpu_idle.json); idle_sched/sys_net = nan"
  fi
}

# syscpu_stop: SIGTERM the sampler (it writes a partial JSON and exits 3), SIGKILL after 2 s, reap it.
syscpu_stop() {
  local i
  kill -TERM "$SYS_PID" 2>/dev/null || true
  for ((i = 0; i < 40; i++)); do syscpu_alive "$SYS_PID" || break; sleep 0.05; done
  syscpu_alive "$SYS_PID" && kill -KILL "$SYS_PID" 2>/dev/null
  wait "$SYS_PID" 2>/dev/null || true
}

# syscpu_start <run_dir>: start the window-mode sampler in the background and wait (<= 5 s) for its ready file.
#   Ready -> SYS_PID, SYS_ENV=(YVO_PERF_CTL=.. YVO_PERF_ACK=..), and this shell also holds both FIFOs O_RDWR
#   (SYS_KEEP_CTL/ACK) until the app exits: should the sampler die early, the app's open() of the FIFOs still
#   returns, its writes never raise SIGPIPE and its ack wait only times out (1 s, one warning), so the run goes on.
#   Not ready -> sampler stopped, SYS_PID="" and SYS_ENV=() (the app runs without YVO_PERF_*, sys_* = nan).
syscpu_start() {
  local d="$1" i
  SYS_PID=""; SYS_ENV=(); SYS_KEEP_CTL=""; SYS_KEEP_ACK=""
  /usr/bin/python3 "$SYSCPU_PY" --ctl "$d/syscpu_ctl.fifo" --ack "$d/syscpu_ack.fifo" --ready "$d/syscpu_ready" \
      --out "$d/syscpu.json" --timeout "$SYSCPU_TIMEOUT_S" --app-comm "$APP_COMM" --csv-fields \
      > "$d/syscpu.csv" 2>> "$d/syscpu.err" &
  SYS_PID=$!
  for ((i = 0; i < 100; i++)); do
    [[ -s "$d/syscpu_ready" ]] && break
    syscpu_alive "$SYS_PID" || break
    sleep 0.05
  done
  if [[ -s "$d/syscpu_ready" ]] && syscpu_alive "$SYS_PID" &&
     exec {SYS_KEEP_CTL}<>"$d/syscpu_ctl.fifo" {SYS_KEEP_ACK}<>"$d/syscpu_ack.fifo"; then
    SYS_ENV=(YVO_PERF_CTL="$d/syscpu_ctl.fifo" YVO_PERF_ACK="$d/syscpu_ack.fifo")
  else
    echo "  warning: sys_cpu_window not ready within 5 s or exited (see syscpu.err); sys_* = nan, app runs without YVO_PERF_*"
    syscpu_stop
    SYS_PID=""
  fi
}

# syscpu_finish <run_dir>: call right after the app exits. Drops the held FIFO ends (a sampler that never got
#   "disable" then sees EOF and writes a partial JSON), waits <= 10 s, stops a sampler still running, removes the
#   FIFOs. Sampler exit 0 -> SYS_CSV = its CSV line; anything else (3 = partial, killed, 2 = error) -> SYS_CSV="".
syscpu_finish() {
  local d="$1" i src=0
  SYS_CSV=""
  if [[ -n "$SYS_KEEP_CTL" ]]; then exec {SYS_KEEP_CTL}>&- {SYS_KEEP_ACK}>&-; fi
  SYS_KEEP_CTL=""; SYS_KEEP_ACK=""
  if [[ -n "$SYS_PID" ]]; then
    for ((i = 0; i < 200; i++)); do syscpu_alive "$SYS_PID" || break; sleep 0.05; done
    if syscpu_alive "$SYS_PID"; then
      echo "  warning: sys_cpu_window still running 10 s after the app; stopped, sys_* = nan"
      syscpu_stop
    else
      wait "$SYS_PID" 2>/dev/null || src=$?
      if (( src == 0 )); then SYS_CSV=$(head -1 "$d/syscpu.csv")
      else echo "  warning: sys_cpu_window exit $src (see syscpu.json status); sys_* = nan"; fi
    fi
  fi
  SYS_PID=""
  rm -f "$d/syscpu_ctl.fifo" "$d/syscpu_ack.fifo"
  if [[ ! -s "$d/syscpu.err" ]]; then rm -f "$d/syscpu.err"; fi
}

# sys_cpu_cols <window CSV line> <idle CSV line> <frames> -> the 9 sys_* columns (no newline). CSV lines are
#   sys_cpu_window.py --csv-fields: win_ms,sys_busy_ms,sched_run_ms,app_ms,labwc_ms,kthread_ms,other_ms,softirq_ms,
#   irq_count. Each output field is nan when an input it needs is missing/nan, or frames is not > 0 (per-frame ones).
sys_cpu_cols() {
  awk -v w="$1" -v z="$2" -v n="$3" '
    function num(x) { return x ~ /^-?[0-9]+([.][0-9]+)?$/ }
    BEGIN {
      nw = split(w, a, ","); nz = split(z, b, ",")
      if (nw != 9) for (k = 1; k <= 9; k++) a[k] = "nan"
      fok = num(n) && n + 0 > 0
      out = num(a[1]) ? sprintf("%.1f", a[1]) : "nan"
      for (k = 2; k <= 7; k++) out = out "," ((fok && num(a[k])) ? sprintf("%.4f", a[k] / n) : "nan")
      iok = (nz == 9) && num(b[1]) && num(b[3]) && b[1] + 0 > 0
      isch = iok ? b[3] * 1000 / b[1] : 0
      out = out "," (iok ? sprintf("%.1f", isch) : "nan")
      net = fok && iok && num(a[1]) && num(a[3])
      out = out "," (net ? sprintf("%.4f", (a[3] - isch * a[1] / 1000) / n) : "nan")
      printf "%s", out
    }'
}

# ── IRQ_TRACE=1: perf irq tracepoints over the app's CPU window (header comment) ─────────────────────
# irq_start <run_dir>, in place of syscpu_idle + syscpu_start: creates both FIFOs, holds them O_RDWR (SYS_KEEP_CTL/ACK,
#   same reason as syscpu_start), queues "ping" on ctl, starts perf under sudo in the background (FIFO keepers closed
#   for it) and waits <= 10 s for "ack": perf answers only after opening both FIFOs and entering its main loop with
#   events disabled (-D -1). Ready -> IRQ_READY=1, IRQ_SUDO_PID (sudo), IRQ_PID (perf itself), SYS_ENV as in
#   syscpu_start. Not ready -> perf stopped, SYS_ENV=() (the app runs without YVO_PERF_*), this run's irq row = nan.
irq_start() {
  local d="$1" ack="" i
  SYS_PID=""; SYS_ENV=(); SYS_KEEP_CTL=""; SYS_KEEP_ACK=""; SYS_IDLE_CSV=""
  IRQ_PID=""; IRQ_SUDO_PID=""; IRQ_READY=0; IRQ_OK=0
  rm -f "$d/syscpu_ctl.fifo" "$d/syscpu_ack.fifo"
  if ! { mkfifo -m 600 "$d/syscpu_ctl.fifo" "$d/syscpu_ack.fifo" &&
         exec {SYS_KEEP_CTL}<>"$d/syscpu_ctl.fifo" {SYS_KEEP_ACK}<>"$d/syscpu_ack.fifo"; }; then
    echo "  warning: IRQ_TRACE FIFOs could not be created or opened; irq row = nan, app runs without YVO_PERF_*"
    return 0
  fi
  printf 'ping\n' >&"$SYS_KEEP_CTL"   # stays in the pipe until perf opens ctl
  irq_record_argv "$d"
  "${SUDO[@]}" "${IRQ_REC[@]}" < /dev/null > "$d/irq_perf.log" 2>&1 {SYS_KEEP_CTL}>&- {SYS_KEEP_ACK}>&- &
  IRQ_SUDO_PID=$!
  for ((i = 0; i < 50; i++)); do
    read -r -t 0.2 -u "$SYS_KEEP_ACK" ack && break
    syscpu_alive "$IRQ_SUDO_PID" || break
  done
  if [[ "$ack" == "ack" ]] && syscpu_alive "$IRQ_SUDO_PID"; then
    IRQ_READY=1
    IRQ_PID=$(irq_perf_pid "$d/irq.perf.data")
    [[ -n "$IRQ_PID" ]] || echo "  warning: perf pid not found; it will be stopped with the 'stop' control command"
    SYS_ENV=(YVO_PERF_CTL="$d/syscpu_ctl.fifo" YVO_PERF_ACK="$d/syscpu_ack.fifo")
  else
    echo "  warning: perf not ready within 10 s or exited (see irq_perf.log); irq row = nan, app runs without YVO_PERF_*"
    irq_stop "$d"
  fi
}

# irq_stop <run_dir>: SIGINT perf ($SUDO kill -INT; the "stop" control command when its pid is unknown or kill
#   fails), SIGKILL after 30 s, then reap sudo. perf's exit status is not checked: after SIGINT it reports its killed
#   workload (143), so irq_finish judges by irq.perf.data and perf script instead.
irq_stop() {
  local i
  [[ -n "$IRQ_SUDO_PID" ]] || return 0
  if syscpu_alive "$IRQ_SUDO_PID"; then
    [[ -n "$IRQ_PID" ]] || IRQ_PID=$(irq_perf_pid "$1/irq.perf.data")
    if ! { [[ -n "$IRQ_PID" ]] && "${SUDO[@]}" kill -INT "$IRQ_PID" < /dev/null; }; then
      if [[ -n "$SYS_KEEP_CTL" ]]; then printf 'stop\n' >&"$SYS_KEEP_CTL"; fi
    fi
    for ((i = 0; i < 600; i++)); do syscpu_alive "$IRQ_SUDO_PID" || break; sleep 0.05; done
    if syscpu_alive "$IRQ_SUDO_PID"; then
      echo "  warning: perf still running 30 s after SIGINT; SIGKILL (irq.perf.data may be truncated)"
      if [[ -n "$IRQ_PID" ]]; then "${SUDO[@]}" kill -KILL "$IRQ_PID" < /dev/null || true; fi
      for ((i = 0; i < 40; i++)); do syscpu_alive "$IRQ_SUDO_PID" || break; sleep 0.05; done
    fi
  fi
  if syscpu_alive "$IRQ_SUDO_PID"; then echo "  warning: sudo/perf pid $IRQ_SUDO_PID still alive; not reaped"
  else wait "$IRQ_SUDO_PID" 2>/dev/null || true; fi
  IRQ_SUDO_PID=""; IRQ_PID=""
}

# irq_finish <run_dir>, in place of syscpu_finish (right after the app exits): stops perf, drops the held FIFO ends,
#   removes the FIFOs, then (if perf was ready) $SUDO perf script > irq_script.txt and $SUDO chown of irq.perf.data and
#   irq_script.txt to the bench user; irq.perf.data is removed unless IRQ_KEEP_DATA=1. IRQ_OK=1 when perf script
#   succeeded. SYS_CSV="" (sys_* = nan).
irq_finish() {
  local d="$1"
  SYS_CSV=""; IRQ_OK=0
  if (( IRQ_READY )) && [[ -n "$IRQ_SUDO_PID" ]] && ! syscpu_alive "$IRQ_SUDO_PID"; then
    echo "  warning: perf exited before the app did (see irq_perf.log); the irq window may be partial"
  fi
  irq_stop "$d"
  if [[ -n "$SYS_KEEP_CTL" ]]; then exec {SYS_KEEP_CTL}>&- {SYS_KEEP_ACK}>&-; fi
  SYS_KEEP_CTL=""; SYS_KEEP_ACK=""
  rm -f "$d/syscpu_ctl.fifo" "$d/syscpu_ack.fifo"
  if grep -qi 'lost' "$d/irq_perf.log" 2>/dev/null; then
    echo "  warning: perf reported lost data: $(grep -i -m1 'lost' "$d/irq_perf.log")"
  fi
  if (( IRQ_READY )) && [[ -s "$d/irq.perf.data" ]]; then
    irq_script_argv "$d"
    if "${SUDO[@]}" "${IRQ_SCR[@]}" < /dev/null > "$d/irq_script.txt" 2>> "$d/irq_perf.log"; then IRQ_OK=1
    else echo "  warning: perf script failed (see irq_perf.log); irq row = nan"; fi
    "${SUDO[@]}" chown "$(id -u):$(id -g)" "$d/irq.perf.data" "$d/irq_script.txt" < /dev/null ||
      echo "  warning: chown of irq.perf.data / irq_script.txt failed"
  elif (( IRQ_READY )); then
    echo "  warning: no irq.perf.data (see irq_perf.log); irq row = nan"
  fi
  if [[ "$IRQ_KEEP_DATA" != "1" ]]; then rm -f "$d/irq.perf.data"; fi
}

# irq_pack <run_dir>: irq_script.txt -> irq_script.txt.gz (also for runs that failed before irq_parse).
irq_pack() { if [[ -f "$1/irq_script.txt" ]]; then gzip -f "$1/irq_script.txt"; fi; }

# irq_parse <run_dir> <label> <run> <frames>: irq_time_window.py on irq_script.txt -> one row of IRQ_RAW
#   (label,run + its CSV line). A missing parser, a non-zero exit, a line without IRQ_NCOLS fields or frames not > 0
#   -> nan row and a warning; the bench goes on. Then gzips irq_script.txt.
irq_parse() {
  local d="$1" line="" prc=0 nf
  if (( IRQ_OK )); then
    if [[ ! -f "$IRQ_PY" ]]; then
      echo "  warning: $IRQ_PY not found; irq row = nan"
    elif [[ ! "$4" =~ ^[0-9]+$ || "$4" == 0 ]]; then
      echo "  warning: frames '$4' is not > 0; irq row = nan"
    else
      line=$(/usr/bin/python3 "$IRQ_PY" --script "$d/irq_script.txt" --frames "$4" --out "$d/irq.json" \
               2>> "$d/irq_parse.err") || prc=$?
      line=${line%%$'\n'*}
      nf=$(awk -F, '{print NF}' <<<"$line")
      if (( prc != 0 )); then
        echo "  warning: irq_time_window.py exit $prc (see irq_parse.err); irq row = nan"; line=""
      elif [[ "$nf" != "$IRQ_NCOLS" ]]; then
        echo "  warning: irq_time_window.py printed $nf fields, expected $IRQ_NCOLS ('$line'); irq row = nan"; line=""
      fi
    fi
  fi
  [[ -n "$line" ]] || line=$(nan_list "$IRQ_NCOLS")
  echo "$2,$3,$line" >> "$IRQ_RAW"
  if [[ ! -s "$d/irq_parse.err" ]]; then rm -f "$d/irq_parse.err"; fi
  irq_pack "$d"
  echo "  irq: uncounted_ms/frame=${line##*,} hardirq_idle_ms=$(cut -d, -f4 <<<"$line") softirq_idle_ms=$(cut -d, -f7 <<<"$line")"
}

for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id args <<<"$c"
    id="${id// /}"
    echo "$id" | grep -Eq "$ONLY" || continue
    if [[ "$args" == *"--display on"* ]] && (( ! HAVE_WAYLAND )); then
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
    # System-wide CPU: idle baseline (app not running), then the window-mode sampler waits for the app's c0..c1.
    if [[ "$IRQ_TRACE" == "1" ]]; then irq_start "$run_dir"; else   # IRQ_TRACE: perf replaces both (header)
    syscpu_idle "$run_dir"
    syscpu_start "$run_dir"
    fi
    # Compositor CPU window: /proc/<labwc pid>/stat right before and right after the yvo_pipeline process (same pid).
    lpid=$(labwc_pid)
    if [[ -n "$lpid" ]]; then proc_stat "$lpid" "$run_dir/labwc_stat_pre.txt"; fi
    set +e
    # shellcheck disable=SC2086
    (cd "$run_dir" && env "${SYS_ENV[@]}" "$BIN" --video "$VIDEO" --hef "$HEF" $args \
        "${frames_arg[@]}" --quiet --csv-label "$id" > stdout.log 2>&1)
    rc=$?
    set -e
    if [[ -n "$lpid" ]]; then proc_stat "$lpid" "$run_dir/labwc_stat_post.txt"; fi
    if [[ "$IRQ_TRACE" == "1" ]]; then irq_finish "$run_dir"; else
    syscpu_finish "$run_dir"
    fi
    nwarn=0; nerr=0
    if [[ -f "$run_dir/hailort.log" ]]; then
      nwarn=$(grep -E '\[warning\]' "$run_dir/hailort.log" | grep -cE "$WARN_RE" || true)
      nerr=$(grep -c '\[error\]' "$run_dir/hailort.log" || true)
    fi
    prev_warn=$nwarn
    l1=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | sed 's/^RESULT_CSV,//' || true)
    # RESULT_CSV2 starts with label, so drop the first field and append.
    l2=$(grep '^RESULT_CSV2,' "$run_dir/stdout.log" | cut -d, -f3- || true)
    if [[ -z "$l1" || -z "$l2" ]]; then
      echo "  failed rc=$rc (see stdout.log)"; tail -5 "$run_dir/stdout.log"
      if [[ "$IRQ_TRACE" == "1" ]]; then irq_pack "$run_dir"; fi   # no results_raw row -> no irq row either
      continue
    fi
    # l2's 9th field (ok) duplicates l1's ok, so drop it.
    #   l2 = postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,[ok],14 display fields
    l2=$(awk -F, 'BEGIN{OFS=","}{ out=""; for(i=1;i<=NF;i++){ if(i==9) continue;
                                   out = (out=="" ? $i : out OFS $i) } print out }' <<<"$l2")
    # labwc_cpu_ms,labwc_cpu_ms_per_frame over the completed frames (field 6 of RESULT_CSV)
    lcols=$(labwc_cpu_cols "$run_dir/labwc_stat_pre.txt" "$run_dir/labwc_stat_post.txt" "$(cut -d, -f6 <<<"$l1")")
    # sys_win_ms .. sys_net_ms_per_frame over the same completed frames
    scols=$(sys_cpu_cols "$SYS_CSV" "$SYS_IDLE_CSV" "$(cut -d, -f6 <<<"$l1")")
    if [[ "$IRQ_TRACE" == "1" ]]; then irq_parse "$run_dir" "$id" "$r" "$(cut -d, -f6 <<<"$l1")"; fi
    echo "$l1,$l2,$r,$COOL_WAITED,$nwarn,$nerr,$lcols,$scols" >> "$RAW"
    echo "  $(grep -E '^wall|^cpu |^held frames' "$run_dir/stdout.log" | tr '\n' ' ') labwc_ms/frame=${lcols#*,} sys_net/frame=${scols##*,} hailo_warn=$nwarn err=$nerr rc=$rc"
  done
done

/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, math, statistics, sys

def median_nan(vals):   # nan runs (e.g. labwc_* without a compositor, sys_* of a failed sampler) are left out; all nan -> nan
    vals = [v for v in vals if not math.isnan(v)]
    return statistics.median(vals) if vals else math.nan

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows"); sys.exit(0)

NON_NUM = {"label", "mode", "layout", "run", "postproc", "alloc", "display", "sync", "pace"}
def numeric(key):
    if key in NON_NUM:
        return False
    for r in rows:
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
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "postproc", "display", "sync", "pace", "runs"] +
               [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [median_nan([float(x[k]) for x in g]) for k in num]
        fps = [float(x["fps"]) for x in g]
        w.writerow([lab, g[0]["postproc"], g[0]["display"], g[0]["sync"], g[0]["pace"], len(g)] +
                   [f"{v:.3f}" for v in med] + [f"{min(fps):.2f}", f"{max(fps):.2f}",
                    sum(int(x["hailort_warnings"]) for x in g),
                    sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])

show = ["fps_median", "cpu_ms_per_frame_median", "labwc_cpu_ms_per_frame_median", "sys_sched_ms_per_frame_median",
        "sys_kthread_ms_per_frame_median", "sys_net_ms_per_frame_median", "gpu_ms_mean_median", "e2e_ms_mean_median",
        "disp_convert_mean_median", "disp_overlay_mean_median", "present_interval_median_median",
        "e2e_glass_mean_median", "presented_median", "display_dropped_median",
        "held_frames_max_median", "slot_timeout_count_median", "detections_median"]
out = list(csv.DictReader(open(sys.argv[2])))
hdr = ["label"] + [c for c in show if c in out[0]]
wid = [max(len(h), *(len(r[h]) for r in out)) for h in hdr]
print(" | ".join(h.ljust(w) for h, w in zip(hdr, wid)))
print("-+-".join("-" * w for w in wid))
for r in out:
    print(" | ".join(r[h].ljust(w) for h, w in zip(hdr, wid)))
EOF

if [[ "$IRQ_TRACE" == "1" && -f "$IRQ_RAW" ]]; then   # per-config median of uncounted_ms_per_frame (nan rows left out)
/usr/bin/python3 - "$IRQ_RAW" <<'EOF'
import csv, math, statistics, sys

def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v

rows = list(csv.DictReader(open(sys.argv[1])))
order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(num(r.get("uncounted_ms_per_frame")))
out = [["label", "runs", "parsed", "uncounted_ms_per_frame_median", "min", "max"]]
for lab in order:
    v = [x for x in groups[lab] if not math.isnan(x)]
    f = (lambda y: f"{y:.4f}") if v else (lambda y: "nan")
    out.append([lab, str(len(groups[lab])), str(len(v)), f(statistics.median(v) if v else 0),
                f(min(v) if v else 0), f(max(v) if v else 0)])
print("\nIRQ_TRACE (irq_results_raw.csv; CPU numbers of these runs include tracepoint overhead)")
wid = [max(len(r[i]) for r in out) for i in range(len(out[0]))]
for k, r in enumerate(out):
    print(" | ".join(c.ljust(w) for c, w in zip(r, wid)))
    if k == 0:
        print("-+-".join("-" * w for w in wid))
EOF
fi

{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "bin md5 (after measurement): $(md5sum "$BIN" | cut -d' ' -f1)"
  vcgencmd get_throttled 2>/dev/null || true
} >> "$OUT_DIR/env_bench.txt"
