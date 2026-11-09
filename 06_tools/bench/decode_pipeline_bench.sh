#!/usr/bin/env bash
# HW decode vs SW decode in the decided pipeline: yvo_pipeline O2 + Vulkan composition, the decoder is the only difference.
# --decoder sw copies each software-decoded frame into a dma-heap SAND128 NV12 buffer, so preprocessing onwards runs the same code.
#
# Configs (all: 04_data/videos/video_h265.mp4, batch 1, one frame in flight, frames arrive at 30 fps into a leaky queue of 1):
#   hw_t1    --decoder hw, YVO_DEC_THREADS=1  (same arguments as "after" in demo_compare_bench.sh: cross-check with its 1.492 ms/f)
#   sw_t1    --decoder sw, YVO_DEC_THREADS=1  (decoding runs on the main thread, inside next())
#   sw_auto  --decoder sw, YVO_DEC_THREADS=0  (libavcodec auto: frame threads, next() only collects a frame and copies it)
#
# Run folder raw/<config>_run<N>/:
#   stdout.log        includes "thread cpu ..." (per thread name) and, for sw, "sw decode copy: ..." and "sw decode pool: ..."
#   frame_log.csv     per-frame times (toolkit/frame_log.hpp), ms since the tool's t0
#   detections.jsonl  per processed frame, "frame" = source frame index
#   cpu_samples.csv   06_tools/python/proc_cpu_sampler.py, about every 0.5 s: target / system / labwc ticks
#   labwc_stat_{pre,post}.txt  compositor /proc stat around the run (labwc_cpu_cols in lib/common.sh)
#   hailort.log, frame00000.png (--dump-dir always saves source frame 0)
#
# The CPU governor is not touched here; wrap this script (measure skill §1). env_bench.txt records what was in effect.
#
# Usage:
#   bash 06_tools/bench/decode_pipeline_bench.sh                  # 3 runs per config, the whole video (1816 frames)
#   RUNS=1 FRAMES=300 ONLY=sw bash ...                            # subset (ONLY = regex on the config id)
#   BIN_PIPELINE=/path bash ...                                   # pin the binary (guards against rebuilds mid-measurement)
#   DRYRUN=1 bash ...                                             # print the commands only (no lock, no run, no files)
#   CHECK_ONLY=1 bash ...                                         # path check only (lib/common.sh)
#   OUT_DIR=/path FORCE=1 bash ...                                # FORCE=1 reuses a non-empty OUT_DIR
#
# Safety (verify skill §3-1): before every run wlr-randr must list CONNECTOR Enabled and labwc must not have lost DRM
# master, otherwise the session stops. Every run is wrapped in `timeout -s KILL TIMEOUT_S`. Never the kms backend.
# Run order: repetition (run) is the outer loop, config the inner loop. Cool-down before every run (COOL_EVERY=1).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN_PIPELINE="${BIN_PIPELINE:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/decode_benchmark/runs/pipeline_sw_hw}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"                  # arrived frames; 0 = the whole video
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-300}"
COOL_EVERY="${COOL_EVERY:-1}"
HOT_C="${HOT_C:-75}"
SOC_MAX_C="${SOC_MAX_C:-75}"           # project gate: SoC <= 75 C before a run
TIMEOUT_S="${TIMEOUT_S:-180}"          # 1816 frames at 30 fps = about 61 s
SAMPLE_S="${SAMPLE_S:-0.5}"
CONNECTOR="${CONNECTOR:-HDMI-A-1}"
DRI_DIR="${DRI_DIR:-/sys/kernel/debug/dri/1}"
DRYRUN="${DRYRUN:-0}"
SAMPLER_PY="$PROJ/06_tools/python/proc_cpu_sampler.py"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"

require_bin "$BIN_PIPELINE" yvo_pipeline
[[ -f "$SAMPLER_PY" && -x /usr/bin/python3 ]] || { echo "proc_cpu_sampler.py or /usr/bin/python3 not found" >&2; exit 1; }
[[ -f "$VIDEO" && -f "$HEF" ]] || { echo "video or HEF not found: $VIDEO $HEF" >&2; exit 1; }
[[ "$FRAMES" =~ ^[0-9]+$ ]] || { echo "FRAMES must be >= 0: $FRAMES" >&2; exit 2; }
for t in timeout wlr-randr awk; do
  command -v "$t" >/dev/null || { echo "required tool not found: $t" >&2; exit 1; }
done
check_only_exit

# id|YVO_DEC_THREADS|decoder. Every option that defines the operating point is explicit: defaults can move, archived commands must not.
# PIPE_ARGS is AFTER_ARGS of demo_compare_bench.sh.
PIPE_ARGS="--mode dmabuf --layout nhcw --postproc gpu --nms gpu --batch 1 --slots 1 --input-pace source --input-queue 1"
PIPE_ARGS+=" --display on --display-backend wayland --display-fullscreen --display-sync matched --display-pace source"
PIPE_ARGS+=" --dump-dir . --dump-every 100000000"
configs=(
  "hw_t1|1|hw"
  "sw_t1|1|sw"
  "sw_auto|0|sw"
)

# Read-only display guards, same as input_pace_bench.sh.
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

# The run's command after the cd: pid file (the tool execs, so the pid stays), timeout, CPU marker FIFO.
launcher='echo $$ > app.pid; exec "$@"'

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN: OUT_DIR=$OUT_DIR RUNS=$RUNS FRAMES=$FRAMES ONLY=$ONLY TIMEOUT_S=$TIMEOUT_S CONNECTOR=$CONNECTOR env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
  echo "DRYRUN: now wlr-randr $CONNECTOR $(output_enabled && echo Enabled || echo NOT-enabled), labwc DRM master: $(master_of labwc), governor: $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null || echo '?')"
  if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR" 2>/dev/null)" && "${FORCE:-0}" != "1" ]]; then
    echo "DRYRUN: note — OUT_DIR is not empty; the real run stops here without FORCE=1"
  fi
  for ((r = 1; r <= RUNS; r++)); do
    for c in "${configs[@]}"; do
      IFS='|' read -r id threads decoder <<<"$c"
      echo "$id" | grep -Eq "$ONLY" || continue
      d="$OUT_DIR/raw/${id}_run${r}"
      echo "[run $r $id] cool-down gate; require: wlr-randr lists $CONNECTOR Enabled and labwc DRM master != n, else stop the session"
      echo "  mkfifo $d/cpu_sync.fifo; /usr/bin/python3 $SAMPLER_PY --pid-file $d/app.pid --ctl $d/cpu_sync.fifo --out $d/cpu_samples.csv --interval $SAMPLE_S &"
      echo "  (cd $d && YVO_DEC_THREADS=$threads YVO_PERF_CTL=$d/cpu_sync.fifo timeout -s KILL $TIMEOUT_S bash -c '$launcher' _ $BIN_PIPELINE --video $VIDEO --hef $HEF $PIPE_ARGS --decoder $decoder ${frames_arg[*]} --frame-log frame_log.csv --quiet --csv-label $id > stdout.log 2>&1)"
    done
  done
  exit 0
fi

guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

{
  echo "# decode_pipeline_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  vcgencmd get_config v3d_freq 2>/dev/null || true
  pcie_link
  cpu_governor
  pkg_versions "${PKGS_DISPLAY_BENCH[@]}"
  env_git_head
  env_bin "$BIN_PIPELINE"; env_hef "$HEF"
  echo "video: $VIDEO md5 $(md5sum "$VIDEO" | cut -d' ' -f1) frames: ${FRAMES} (0 = all) runs: $RUNS"
  echo "timeout -s KILL ${TIMEOUT_S}s, connector $CONNECTOR, sampler interval ${SAMPLE_S}s, clk_tck $(getconf CLK_TCK)"
  echo "wlr-randr $CONNECTOR: $(output_enabled && echo Enabled || echo NOT-enabled); labwc DRM master: $(master_of labwc)"
  timeout 5 wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '/^[^ \t]/ { inb = ($1 == c) } inb && /current/ { print "mode: " $0; exit }' || true
  echo "env: $(env | grep -E '^(VK_|HAILO|YVO_|WAYLAND_DISPLAY|XDG_RUNTIME_DIR)' | paste -sd ' ' || true)"
  echo "args: $PIPE_ARGS"
  echo "configs (id|YVO_DEC_THREADS|decoder):"
  for c in "${configs[@]}"; do echo "  $c"; done
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
# RESULT_CSV 22 + RESULT_CSV4 21 (after its label) + 15 script columns = 58. The sw_* columns are nan for hw (per frame, "sw decode copy:" line).
CSV1_COLS="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok"
CSV4_COLS="input_queue,src_fps,arrived,processed,dropped,abandoned,drop_rate,max_drop_run,queue_depth_max,e2e_arr_mean,e2e_arr_median,e2e_arr_p90,e2e_arr_max,queue_wait_mean,queue_wait_p90,queue_wait_max,decode_late_mean,decode_late_p90,decode_late_max,held_frames_max,pace_ok"
SCRIPT_COLS="run,cool_wait_s,hailo_pre,soc_pre,hailort_warnings,hailort_errors,labwc_cpu_ms,labwc_cpu_ms_per_frame,sw_copy_ms,sw_memcpy_ms,sw_sync_ms,sw_decode_wait_ms,exit_code,unreliable,note"
echo "$CSV1_COLS,$CSV4_COLS,$SCRIPT_COLS" > "$RAW"

prev_warn=0
aborted=""
for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id threads decoder <<<"$c"
    echo "$id" | grep -Eq "$ONLY" || continue

    cool_before_run                      # hailo sensor; sets t_pre and COOL_WAITED
    cooled=$COOL_WAITED
    soc_now=$(soc_temp)
    if awk -v t="$soc_now" -v m="$SOC_MAX_C" 'BEGIN{exit !(t=="nan" || t>m)}'; then
      COOL_C=$SOC_MAX_C cool_down "soc ${soc_now} C > ${SOC_MAX_C} C" soc
      cooled=$((cooled + COOL_WAITED))
      soc_now=$(soc_temp)
    fi

    if ! display_ready; then
      echo "display not ready before run $r $id: wlr-randr $CONNECTOR $(output_enabled && echo Enabled || echo NOT-enabled)," \
           "labwc DRM master $(master_of labwc) — stopping the session" >&2
      aborted="run $r $id"; break 2
    fi

    run_dir="$OUT_DIR/raw/${id}_run${r}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    echo "[$(date +%T)] run $r/$RUNS $id (hailo $t_pre C, soc $soc_now C, throttled $(throttled))"
    fifo="$run_dir/cpu_sync.fifo"
    mkfifo -m 600 "$fifo"
    exec {keep_fd}<>"$fifo"              # the tool's write never raises SIGPIPE, even if the sampler died
    /usr/bin/python3 "$SAMPLER_PY" --pid-file "$run_dir/app.pid" --ctl "$fifo" --out "$run_dir/cpu_samples.csv" \
        --interval "$SAMPLE_S" --max-s "$((TIMEOUT_S + 30))" 2> "$run_dir/sampler.err" &
    sampler=$!
    lpid=$(labwc_pid)
    if [[ -n "$lpid" ]]; then proc_stat "$lpid" "$run_dir/labwc_stat_pre.txt"; fi
    set +e
    # shellcheck disable=SC2086  # args is word-split on purpose
    (cd "$run_dir" && YVO_DEC_THREADS="$threads" YVO_PERF_CTL="$fifo" timeout -s KILL "$TIMEOUT_S" bash -c "$launcher" _ "$BIN_PIPELINE" \
        --video "$VIDEO" --hef "$HEF" $PIPE_ARGS --decoder "$decoder" "${frames_arg[@]}" --frame-log frame_log.csv --quiet --csv-label "$id" \
        > stdout.log 2>&1)
    rc=$?
    set -e
    if [[ -n "$lpid" ]]; then proc_stat "$lpid" "$run_dir/labwc_stat_post.txt"; fi

    # The sampler sees the tool exit within one interval; if it does not, stop it (it still writes its CSV).
    for ((i = 0; i < 100; i++)); do   # state Z: exited, not reaped yet
      s=$(cut -d' ' -f3 "/proc/$sampler/stat" 2>/dev/null || echo Z); [[ "$s" == Z ]] && break; sleep 0.05
    done
    kill -TERM "$sampler" 2>/dev/null || true
    src=0; wait "$sampler" 2>/dev/null || src=$?
    exec {keep_fd}>&-
    rm -f "$fifo" "$run_dir/app.pid"
    [[ -s "$run_dir/sampler.err" ]] || rm -f "$run_dir/sampler.err"

    read -r nwarn nerr <<<"$(count_hailort_warnings "$run_dir/hailort.log")"
    prev_warn=$nwarn
    log="$run_dir/stdout.log"; notes=()
    (( rc == 137 )) && notes+=(timeout_kill)
    (( rc != 0 )) && notes+=("rc_$rc")
    (( src != 0 )) && notes+=("sampler_rc_$src")
    line=$(grep '^RESULT_CSV,' "$log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
    if [[ "$(awk -F, '{print NF}' <<<"$line")" != "22" ]]; then
      notes+=(no_result_csv)
      line="$id,,,,,$(nan_list 16),0"
    fi
    pace=$(grep '^RESULT_CSV4,' "$log" | tail -1 | cut -d, -f3- || true)
    if [[ "$(awk -F, '{print NF}' <<<"$pace")" != "21" ]]; then
      notes+=(no_result_csv4)
      pace="$(nan_list 20),0"
    fi
    lcols=$(labwc_cpu_cols "$run_dir/labwc_stat_pre.txt" "$run_dir/labwc_stat_post.txt" "$(cut -d, -f6 <<<"$line")")
    # "sw decode copy: mean A ms/frame (memcpy B + cache sync C), decode wait mean D ms/frame, N frames"
    swcols=$(sed -nE 's/^sw decode copy: mean ([0-9.]+) ms\/frame \(memcpy ([0-9.]+) \+ cache sync ([0-9.]+)\), decode wait mean ([0-9.]+) ms\/frame.*/\1,\2,\3,\4/p' "$log" | tail -1)
    if [[ -z "$swcols" ]]; then
      swcols=$(nan_list 4)
      [[ "$decoder" == "sw" ]] && notes+=(no_sw_copy_line)
    fi
    unreliable=0; (( nwarn > 0 )) && unreliable=1
    note="-"; (( ${#notes[@]} )) && note=$(IFS=';'; echo "${notes[*]}")
    echo "$line,$pace,$r,$cooled,$t_pre,$soc_now,$nwarn,$nerr,$lcols,$swcols,$rc,$unreliable,$note" >> "$RAW"
    echo "  $(grep -E '^wall|^cpu |^arrived|^display  |^sw decode copy' "$log" | tr -s ' ' | tr '\n' ' ') labwc_ms/frame=${lcols#*,} hailo_warn=$nwarn err=$nerr rc=$rc note=$note"
  done
done

# Per-config medians (measure skill: the median over independent runs). nan values are left out of a median.
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, math, statistics, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows"); sys.exit(0)

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan

NON_NUM = {"label", "mode", "layout", "run", "note"}
cols = [k for k in rows[0] if k not in NON_NUM]
order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)

def row_ok(x):
    return x["ok"] == "1" and x["pace_ok"] == "1" and x["exit_code"] == "0" and x["unreliable"] == "0"

out = []
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "runs"] + [f"{k}_median" for k in cols] + ["all_ok"])
    for lab in order:
        g = groups[lab]
        med = []
        for k in cols:
            v = [num(x[k]) for x in g]
            v = [y for y in v if not math.isnan(y)]
            med.append(f"{statistics.median(v):.3f}" if v else "nan")
        w.writerow([lab, len(g)] + med + [int(all(row_ok(x) for x in g))])
        idx = {k: med[i] for i, k in enumerate(cols)}
        out.append([lab, str(len(g)), idx["fps"], idx["cpu_ms_per_frame"], idx["labwc_cpu_ms_per_frame"], idx["sw_copy_ms"], idx["drop_rate"],
                    idx["e2e_arr_median"], idx["detections"], str(int(all(row_ok(x) for x in g)))])
hdr = ["label", "runs", "fps", "cpu_ms/f", "labwc_ms/f", "sw_copy_ms", "drop_rate", "e2e_arr_med", "dets", "all_ok"]
wid = [max(len(h), *(len(r[i]) for r in out)) for i, h in enumerate(hdr)]
print(" | ".join(h.ljust(w) for h, w in zip(hdr, wid)))
print("-+-".join("-" * w for w in wid))
for r in out:
    print(" | ".join(c.ljust(w) for c, w in zip(r, wid)))
EOF

{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "bin md5 (after measurement): $(md5sum "$BIN_PIPELINE" | cut -d' ' -f1)"
  vcgencmd get_throttled 2>/dev/null || true
  [[ -n "$aborted" ]] && echo "ABORTED before $aborted: display not ready ($CONNECTOR / labwc DRM master)"
} >> "$OUT_DIR/env_bench.txt"
[[ -z "$aborted" ]] || { echo "session stopped before $aborted (display not ready)" >&2; exit 1; }
