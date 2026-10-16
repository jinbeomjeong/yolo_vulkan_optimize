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
# Usage:
#   bash 06_tools/bench/display_bench.sh
#   RUNS=3 FRAMES=900 ONLY='disp_on' bash 06_tools/bench/display_bench.sh
#   CHECK_ONLY=1 bash 06_tools/bench/display_bench.sh                 # binary/path check only, exit
#   IDLE_S=0 bash 06_tools/bench/display_bench.sh                     # skip the per-run idle baseline
#   OUT_DIR=/path [FORCE=1] [APPEND=1] bash ...                        # APPEND needs the same 70-column header
#   Output: OUT_DIR (default 07_results/display_e2e) / {env_bench.txt, results_raw.csv (70 columns), results.csv,
#           raw/<id>_run<N>/{stdout.log, hailort.log, labwc_stat_{pre,post}.txt, syscpu{,_idle}.{json,csv},
#           syscpu_ready[, syscpu.err]}, raw/m5a_run1/, raw/_session/}
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

require_bin "$BIN" yvo_pipeline
[[ -f "$SYSCPU_PY" && -x /usr/bin/python3 ]] || { echo "sys_cpu_window.py or /usr/bin/python3 not found" >&2; exit 1; }
[[ "$IDLE_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "IDLE_S must be a number >= 0: '$IDLE_S'" >&2; exit 1; }
check_only_exit
guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

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
    syscpu_idle "$run_dir"
    syscpu_start "$run_dir"
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
    syscpu_finish "$run_dir"
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

{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "bin md5 (after measurement): $(md5sum "$BIN" | cut -d' ' -f1)"
  vcgencmd get_throttled 2>/dev/null || true
} >> "$OUT_DIR/env_bench.txt"
