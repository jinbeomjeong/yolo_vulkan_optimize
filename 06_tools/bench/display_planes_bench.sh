#!/usr/bin/env bash
# HVS plane scanout display vs Vulkan composition — formal e2e measurement (experiment B).
# Plan: 07_results/display_vulkan_hdmi/notes/formal_measurement_plan.md §5 (experiment B), §6, §7.
#
# Why a separate script:
#   display_bench.sh has no --display-backend axis and is the reproduction reference of display_e2e (configs unchanged).
#   Structure (cool-down gate, interleaving, results_raw/results CSV, stdout metric extraction, V3D gpu_stats deltas)
#   follows nms_bench.sh (experiment A). Per-run side files (gpu_stats, labwc /proc stat, debugfs state mid-run) follow
#   07_results/display_vulkan_hdmi/evidence/hvs_review/baseline/run_baseline.sh.
#
# Configs (all b2/s$SLOTS (default 4), --postproc gpu --nms gpu, NPU on, fullscreen, --display-sync matched, every flag explicit):
#   vk_source  / vk_asap    Vulkan composition (--display-backend wayland: the backend `auto` resolves to, but strict —
#                           no silent offscreen fallback)
#   wlp_source / wlp_asap   --display-backend wayland-planes --display-color-range auto
#   kms_source / kms_asap   --display-backend kms --display-connector HDMI-A-1 --display-color-range auto
#   Every config gets the same --display-warmup WARMUP (default 30): kms's first commit is a blocking modeset
#   (kms_scanout.hpp B8/B10), and a fair comparison cuts the same N presentations everywhere. 30 = 1 s of the 30 fps
#   source, the cut hvs_kms_probe.cpp uses (`i < 30`); PlaneDisplay's debugfs start snapshot comes later (presented 60).
#
# Loop: repetition outer, config inner (interleaved). Within a repetition the kms configs run back to back so the
#   session switch happens once per repetition:  ... wlp_asap -> [activate tty] kms_source kms_asap [activate desktop].
#
# kms session switch (08_learn/07_video_display.md §7.12; no sudo, no lightdm stop):
#   The desktop (Type wayland|x11) and tty (Type tty, TTY tty*) sessions of this user on seat0 are found at run time
#   with loginctl (never hard-coded). Before the first kms run: `loginctl activate <tty>`, wait until labwc is no longer
#   DRM master. After the kms runs have **exited**: `loginctl activate <desktop>`, then require desktop Active, labwc
#   master y in debugfs clients and HDMI-A-1 Enabled in wlr-randr — otherwise stop (see the recovery note printed).
#   No VT switch ever happens while yvo_pipeline runs (the switch-back waits for the child to exit, also on Ctrl+C).
#
# Display safety: before every Vulkan / wayland-planes run, wlr-randr must list HDMI-A-1 as Enabled. If labwc is
#   headless, the Vulkan FIFO present blocks forever, so the script stops instead. debugfs clients (master column)
#   is recorded before and after every run.
#
# Usage:
#   bash 06_tools/bench/display_planes_bench.sh                       # 3 runs x 6 configs x 900 frames (user present: kms)
#   SKIP_KMS=1 bash ...                                               # vk + wlp only (no session switch)
#   ONLY='^wlp_' RUNS=1 FRAMES=300 bash ...                           # subset (grep -E on config id)
#   SLOTS=6 bash ...                                                  # --slots of every config (default 4; --batch stays 2)
#   DRYRUN=1 bash ...                                                 # print every command (incl. session switches), exit
#   CHECK_ONLY=1 bash ...                                             # binary/tool check only
#   SESSIONS_ONLY=1 bash ...                                          # read-only: print the session ids found, exit
#   OUT_DIR=/path FORCE=1 APPEND=1 bash ...                           # see guard below
#   WARMUP=30 TIMEOUT_S=90 MID_SNAP_S=<auto> COOL_C=65 COOL_MAX_S=120 HOT_C=75 COOL_EVERY=1 bash ...
#
# Output (default OUT_DIR = 07_results/display_planes_e2e; not created by anyone else — guard_out_dir):
#   env_bench.txt, results_raw.csv (one row per run), results.csv (per config), evidence/sessions.txt (kms switch log),
#   raw/<id>_run<N>/{stdout.log, hailort.log, meta.txt, gpu_stats_{pre,post}.txt, labwc_stat_{pre,post}.txt,
#                    labwc_fdinfo_{pre,post}.txt, yvo_stat_last.txt, yvo_fdinfo_last.txt, dri_state_mid.txt,
#                    dri_clients_{pre,post}.txt, wlr_randr_pre.txt (not for kms)}
#   raw/_session/hailort.log (temperature probe)
# results_raw.csv = 11 script columns + RESULT_CSV (22) + RESULT_CSV2 without label and its duplicate ok (33)
#                   + RESULT_CSV3 without label and ok (12 + d2s_mean/median/p90/n = 16)
#                   + 19 stdout / side-file columns + checks = 102 columns.
# Side-file columns: labwc_* = /proc stat ticks and drm-engine-* fdinfo deltas over the run (same labwc pid before and after);
#   yvo_* = the child's last /proc sample (<= SAMPLE_S before exit), i.e. its **whole lifetime including init**
#   (HailoRT / Vulkan setup) — the process CPU of the measurement window is cpu_ms_per_frame (getrusage) from RESULT_CSV;
#   v3d_* = gpu_stats queue runtime deltas, system-wide (all clients). All divided by completed frames.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, package versions, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/display_planes_e2e}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-900}"
ONLY="${ONLY:-.}"
WARMUP="${WARMUP:-30}"
SLOTS="${SLOTS:-4}"
TIMEOUT_S="${TIMEOUT_S:-90}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
COOL_EVERY="${COOL_EVERY:-1}"
APPEND="${APPEND:-0}"
DRYRUN="${DRYRUN:-0}"
SKIP_KMS="${SKIP_KMS:-0}"
SESSIONS_ONLY="${SESSIONS_ONLY:-0}"
CONNECTOR="${CONNECTOR:-HDMI-A-1}"
SWITCH_WAIT_S="${SWITCH_WAIT_S:-15}"   # max wait for master / output to settle after a session switch
SAMPLE_S="${SAMPLE_S:-0.25}"           # /proc sampling period of the running yvo_pipeline
GPU_STATS="/sys/class/drm/renderD128/device/gpu_stats"
DRI_DIR="${DRI_DIR:-/sys/kernel/debug/dri/1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"

[[ "$FRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "FRAMES must be a positive integer: $FRAMES" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "RUNS must be a positive integer: $RUNS" >&2; exit 2; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "WARMUP must be >= 0: $WARMUP" >&2; exit 2; }
[[ "$SLOTS" =~ ^[1-9][0-9]*$ ]] || { echo "SLOTS must be a positive integer: $SLOTS" >&2; exit 2; }
(( WARMUP < FRAMES )) || { echo "WARMUP ($WARMUP) must be < FRAMES ($FRAMES)" >&2; exit 2; }
# Mid-run debugfs state snapshot: half of the source-paced duration + 3 s of startup.
MID_SNAP_S="${MID_SNAP_S:-$(( FRAMES / 60 + 3 ))}"
require_bin "$BIN" yvo_pipeline
for t in timeout loginctl wlr-randr pgrep awk; do
  command -v "$t" >/dev/null || { echo "required tool not found: $t" >&2; exit 1; }
done
check_only_exit

# Shared arguments (everything explicit, also the CLI defaults: defaults can move, archived commands must not).
BASE_ARGS="--mode dmabuf --layout nhcw --postproc gpu --postproc-alloc export --nms gpu --gpu-depth 3 --batch 2 --slots $SLOTS"
BASE_ARGS+=" --display on --display-fullscreen --display-sync matched --display-warmup $WARMUP"
VK_ARGS="--display-backend wayland"
WLP_ARGS="--display-backend wayland-planes --display-color-range auto"
KMS_ARGS="--display-backend kms --display-connector $CONNECTOR --display-color-range auto"
# id|args — kms configs last so they run back to back inside one session switch.
configs=(
  "vk_source  |$BASE_ARGS $VK_ARGS --display-pace source"
  "vk_asap    |$BASE_ARGS $VK_ARGS --display-pace asap"
  "wlp_source |$BASE_ARGS $WLP_ARGS --display-pace source"
  "wlp_asap   |$BASE_ARGS $WLP_ARGS --display-pace asap"
  "kms_source |$BASE_ARGS $KMS_ARGS --display-pace source"
  "kms_asap   |$BASE_ARGS $KMS_ARGS --display-pace asap"
)
selected=()
for c in "${configs[@]}"; do
  IFS='|' read -r id args <<<"$c"; id="${id// /}"
  [[ "$SKIP_KMS" == "1" && "$id" == kms_* ]] && continue
  grep -Eq "$ONLY" <<<"$id" && selected+=("$id|$args")
done
(( ${#selected[@]} > 0 )) || { echo "ONLY='$ONLY' (SKIP_KMS=$SKIP_KMS) matches no config id" >&2; exit 1; }
is_kms() { [[ "$1" == kms_* ]]; }
any_kms=0
for c in "${selected[@]}"; do is_kms "${c%%|*}" && any_kms=1; done

# ── Sessions / master / output (read-only helpers) ───────────────────────────────────
# find_sessions: sets DESKTOP_SID / TTY_SID (empty if not exactly one candidate) and SESSIONS_DESC.
#   desktop = this user, seat0, Class user, Remote no, Type wayland|x11
#   tty     = this user, seat0, Class user, Remote no, Type tty, TTY tty*   (lowest VT if several)
find_sessions() {
  local me sid kv seat type class remote tty vt name active desk=() ttys=()
  me=$(id -un)
  SESSIONS_DESC=""
  for sid in $(loginctl list-sessions --no-legend 2>/dev/null | awk '{print $1}'); do
    kv=$(loginctl show-session "$sid" -p Name -p Seat -p Type -p Class -p Remote -p TTY -p VTNr -p Active 2>/dev/null) || continue
    name=$(sed -n 's/^Name=//p' <<<"$kv"); seat=$(sed -n 's/^Seat=//p' <<<"$kv"); type=$(sed -n 's/^Type=//p' <<<"$kv")
    class=$(sed -n 's/^Class=//p' <<<"$kv"); remote=$(sed -n 's/^Remote=//p' <<<"$kv"); tty=$(sed -n 's/^TTY=//p' <<<"$kv")
    vt=$(sed -n 's/^VTNr=//p' <<<"$kv"); active=$(sed -n 's/^Active=//p' <<<"$kv")
    SESSIONS_DESC+="session $sid name=$name seat=${seat:--} type=$type class=$class remote=$remote tty=${tty:--} vt=$vt active=$active"$'\n'
    [[ "$name" == "$me" && "$seat" == "seat0" && "$class" == "user" && "$remote" == "no" ]] || continue
    if [[ "$type" == "wayland" || "$type" == "x11" ]]; then desk+=("$sid")
    elif [[ "$type" == "tty" && "$tty" == tty* ]]; then ttys+=("$vt:$sid"); fi
  done
  DESKTOP_SID=""; TTY_SID=""
  (( ${#desk[@]} == 1 )) && DESKTOP_SID="${desk[0]}"
  (( ${#ttys[@]} >= 1 )) && TTY_SID=$(printf '%s\n' "${ttys[@]}" | sort -t: -k1,1n | head -1 | cut -d: -f2)
  SESSIONS_DESC+="desktop candidates: ${desk[*]:-none} -> DESKTOP_SID=${DESKTOP_SID:-none}"$'\n'
  SESSIONS_DESC+="tty candidates (vt:id): ${ttys[*]:-none} -> TTY_SID=${TTY_SID:-none}"
}
session_active() { [[ "$(loginctl show-session "$1" -p Active --value 2>/dev/null)" == "yes" ]]; }
# master_of <command>: "y" if a process with that command name holds DRM master on card1, "n" if listed but not,
# "?" if debugfs clients is unreadable.
master_of() {
  [[ -r "$DRI_DIR/clients" ]] || { echo "?"; return; }
  awk -v c="$1" 'NR > 1 && $1 == c { seen = 1; if ($4 == "y") m = 1 } END { print (m ? "y" : (seen ? "n" : "-")) }' \
    "$DRI_DIR/clients" 2>/dev/null || echo "?"
}
masters() { [[ -r "$DRI_DIR/clients" ]] && awk 'NR > 1 && $4 == "y" { printf "%s%s(%s)", (n++ ? " " : ""), $1, $2 }' "$DRI_DIR/clients" || echo "unreadable"; }
# output_enabled: wlr-randr lists $CONNECTOR with "Enabled: yes" (labwc not headless).
output_enabled() {
  wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '
    /^[^ \t]/ { inb = ($1 == c) } inb && /^[ \t]+Enabled:/ { if ($2 == "yes") ok = 1 } END { exit !ok }'
}
desktop_ready() {   # desktop_ready <desktop sid|""> -> 0 if the desktop can take Vulkan / wayland-planes runs
  { [[ -z "$1" ]] || session_active "$1"; } && output_enabled && [[ "$(master_of labwc)" != "n" ]]
}

if [[ "$SESSIONS_ONLY" == "1" || "$DRYRUN" == "1" ]]; then
  find_sessions
  echo "$SESSIONS_DESC"
  echo "DRM master on card1 now: $(masters); labwc master: $(master_of labwc)"
  if output_enabled; then echo "wlr-randr: $CONNECTOR Enabled"; else echo "wlr-randr: $CONNECTOR NOT enabled (or no compositor)"; fi
  [[ "$SESSIONS_ONLY" == "1" ]] && exit 0
fi

# build_cmd <id> <args> -> fills cmd[] (no `timeout` here; run_one adds it)
build_cmd() {
  local id="$1" args="$2" argv=()
  read -ra argv <<<"$args"
  cmd=(env -u YVO_DEC_THREADS "$BIN" --video "$VIDEO" --hef "$HEF" "${argv[@]}" --frames "$FRAMES" --quiet --csv-label "$id")
}

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN: $(basename "$0") OUT_DIR=$OUT_DIR RUNS=$RUNS FRAMES=$FRAMES WARMUP=$WARMUP SLOTS=$SLOTS ONLY='$ONLY' SKIP_KMS=$SKIP_KMS"
  echo "DRYRUN: TIMEOUT_S=$TIMEOUT_S MID_SNAP_S=$MID_SNAP_S COOL_EVERY=$COOL_EVERY env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
  for ((r = 1; r <= RUNS; r++)); do
    on_tty=0
    for c in "${selected[@]}"; do
      IFS='|' read -r id args <<<"$c"
      if is_kms "$id" && (( ! on_tty )); then
        echo "[run $r] loginctl activate ${TTY_SID:-<tty session: NOT FOUND>}   # then wait until labwc is not DRM master"
        on_tty=1
      elif ! is_kms "$id"; then
        echo "[run $r] check: wlr-randr lists $CONNECTOR Enabled, labwc DRM master, desktop session ${DESKTOP_SID:-<NOT FOUND>} active"
      fi
      build_cmd "$id" "$args"
      printf '[run %d/%d %s] cd %q && timeout -s KILL %s ' "$r" "$RUNS" "$id" "$OUT_DIR/raw/${id}_run${r}" "$TIMEOUT_S"
      printf '%q ' "${cmd[@]}"; printf '> stdout.log 2>&1\n'
    done
    if (( on_tty )); then
      echo "[run $r] (after the kms child exited) loginctl activate ${DESKTOP_SID:-<desktop session: NOT FOUND>}"
      echo "[run $r] require: desktop active, labwc DRM master y, wlr-randr $CONNECTOR Enabled — else stop"
    fi
  done
  exit 0
fi

# ── Archived-result protection, lock ────────────────────────────────────────────────
if [[ "$APPEND" == "1" ]]; then
  [[ "${FORCE:-0}" == "1" ]] || { echo "APPEND=1 modifies archived results: needs FORCE=1" >&2; exit 1; }
else
  guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1
fi

# Pre-flight (before the lock): the display path must be usable, and kms needs both sessions.
find_sessions
if (( any_kms )); then
  [[ -n "$DESKTOP_SID" && -n "$TTY_SID" ]] || {
    echo "kms configs selected but the sessions were not found (SKIP_KMS=1 to run without kms):" >&2
    echo "$SESSIONS_DESC" >&2; exit 1; }
fi
if ! desktop_ready "$DESKTOP_SID"; then
  echo "display not ready: desktop session ${DESKTOP_SID:-?} active? wlr-randr $CONNECTOR Enabled? labwc master=$(master_of labwc)" >&2
  echo "  (labwc headless -> the Vulkan FIFO present blocks forever). Stopping." >&2
  exit 1
fi

bench_lock_acquire
mkdir -p "$OUT_DIR/raw"
SESS_LOG="$OUT_DIR/evidence/sessions.txt"   # kms switch log (not directly in OUT_DIR: STRUCTURE.md R4)
ON_TTY=0          # 1 while the tty session is active (set by to_tty, cleared by to_desktop)
CHILD_PID=""      # running timeout/yvo_pipeline (the switch-back waits for it)

recovery_hint() {
  cat >&2 <<EOF
  Recovery (memory note labwc-master-recovery / 08_learn/07_video_display.md §7.12):
    cat $DRI_DIR/clients   # who holds master (column 'master')
    If Xwayland holds it: kill -9 Xwayland, then loginctl activate $TTY_SID; loginctl activate $DESKTOP_SID
    Then check: wlr-randr shows $CONNECTOR Enabled: yes and labwc master y.
EOF
}
to_tty() {
  mkdir -p "$OUT_DIR/evidence"; echo "[$(date +%T)] session: activate tty $TTY_SID (desktop $DESKTOP_SID); masters before: $(masters)" | tee -a "$SESS_LOG"
  loginctl activate "$TTY_SID"
  ON_TTY=1
  local w=0
  while [[ "$(master_of labwc)" == "y" ]] && (( w < SWITCH_WAIT_S * 4 )); do sleep 0.25; w=$((w + 1)); done
  echo "[$(date +%T)] session: tty active=$(session_active "$TTY_SID" && echo yes || echo no), masters: $(masters)" | tee -a "$SESS_LOG"
  [[ "$(master_of labwc)" != "y" ]]
}
to_desktop() {   # never called while a child runs
  echo "[$(date +%T)] session: activate desktop $DESKTOP_SID; masters before: $(masters)" | tee -a "$SESS_LOG"
  loginctl activate "$DESKTOP_SID"
  ON_TTY=0
  local w=0
  until desktop_ready "$DESKTOP_SID" && [[ "$(master_of labwc)" == "y" ]] || (( w >= SWITCH_WAIT_S * 4 )); do
    sleep 0.25; w=$((w + 1))
  done
  local ok=0
  desktop_ready "$DESKTOP_SID" && [[ "$(master_of labwc)" == "y" ]] && ok=1
  echo "[$(date +%T)] session: desktop active=$(session_active "$DESKTOP_SID" && echo yes || echo no)," \
       "labwc master=$(master_of labwc), masters: $(masters), $CONNECTOR $(output_enabled && echo Enabled || echo NOT-enabled)" \
       "-> $([[ $ok == 1 ]] && echo ok || echo FAILED)" | tee -a "$SESS_LOG"
  (( ok ))
}
cleanup() {
  local rc=$?
  # Never switch VT while yvo_pipeline runs: ask it to stop (timeout forwards SIGTERM; the app exits with its partial
  # summary, code 143) and wait for it (timeout -s KILL bounds it) before going back.
  if [[ -n "$CHILD_PID" ]]; then kill -TERM "$CHILD_PID" 2>/dev/null || true; wait "$CHILD_PID" 2>/dev/null || true; fi
  if (( ON_TTY )); then to_desktop || { echo "WARNING: desktop did not come back cleanly" >&2; recovery_hint; }; fi
  rm -f "$BENCH_LOCK"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

env_snapshot() {
  echo "# display_planes_bench $1 $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  throttled
  vcgencmd measure_clock arm 2>/dev/null || true
  vcgencmd measure_clock v3d 2>/dev/null || true
  vcgencmd get_config v3d_freq 2>/dev/null || true
  echo "bootloader: $(vcgencmd bootloader_version 2>/dev/null | head -1)"
  pcie_link
  pkg_versions "${PKGS_DISPLAY_BENCH[@]}"   # labwc / libwlroots: the wayland-planes COLOR_RANGE inversion is compositor behavior
  env_git_head
  env_bin "$BIN"
  echo "wlr-randr $CONNECTOR: $(output_enabled && echo Enabled || echo NOT-enabled); DRM masters: $(masters)"
  wlr-randr 2>/dev/null | awk -v c="$CONNECTOR" '/^[^ \t]/ { inb = ($1 == c) } inb && /current/ { print "mode: " $0; exit }'
}
{
  env_snapshot before
  echo "video: $VIDEO"
  echo "hef: $HEF"
  echo "runs: $RUNS frames: $FRAMES warmup: $WARMUP slots: $SLOTS only: $ONLY skip_kms: $SKIP_KMS timeout_s: $TIMEOUT_S mid_snap_s: $MID_SNAP_S"
  echo "cool: COOL_C=$COOL_C COOL_MAX_S=$COOL_MAX_S HOT_C=$HOT_C COOL_EVERY=$COOL_EVERY"
  echo "caller YVO_DEC_THREADS: ${YVO_DEC_THREADS-<unset>} (removed for every child: app default)"
  echo "env: $(env | grep -E '^(VK_|HAILO|AV_|FFREPORT|YVO_|WAYLAND_DISPLAY|XDG_RUNTIME_DIR)' | paste -sd ' ' || true)"
  echo "$SESSIONS_DESC"
  echo "configs (id|args):"
  for c in "${selected[@]}"; do echo "  $c"; done
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"
if env | grep -qE '^(VK_LOADER_LAYERS_ENABLE|VK_INSTANCE_LAYERS)='; then
  echo "WARNING: VK_*LAYERS* set in the environment (recorded in env_bench.txt)" >&2
fi

# ── Side files ─────────────────────────────────────────────────────────────────
# drm_fdinfo <pid> <out>: every /dev/dri fdinfo block of that process (drm-engine-* are cumulative ns per client).
drm_fdinfo() {
  local pid="$1" out="$2" fd tgt buf=""
  for fd in /proc/"$pid"/fd/*; do
    tgt=$(readlink "$fd" 2>/dev/null) || continue
    [[ "$tgt" == /dev/dri/* && -r /proc/$pid/fdinfo/${fd##*/} ]] || continue
    buf+="== fd ${fd##*/} -> $tgt"$'\n'"$(cat /proc/"$pid"/fdinfo/"${fd##*/}" 2>/dev/null)"$'\n'
  done
  printf '%s' "$buf" > "$out"
}
# proc_stat <pid> <out> lives in lib/common.sh (shared with display_bench.sh, 2026-10-16).
clients_snap() { cat "$DRI_DIR/clients" > "$1" 2>/dev/null || echo unreadable > "$1"; }

# ── stdout metrics ──────────────────────────────────────────────────────────────
# Parsed from the yvo_pipeline summary (01_src/pipeline/main.cpp, end of run()). nan when absent.
#   glass-to-glass  mean M  median D  p90 P  (n=N, ...)     -> (already RESULT_CSV2 mean/p90, RESULT_CSV3 median)
#   decode-to-screen mean M  median D  p90 P  (n=N, ...)    -> (already RESULT_CSV3 d2s_*)
#   slot wait ms    mean M  p90 P  (...)                    -> slot_wait_ms_mean, slot_wait_ms_p90
#   "  kms: flips F, flip_timeouts T, commit failures C, busy B, master_lost no|yes (...)" -> kms_*
#   "  kms: shutdown <result>"                                                               -> kms_shutdown (commas -> ;)
slot_wait_metrics() {   # slot_wait_metrics <stdout.log> -> "mean,p90" (nan,nan if the line is missing)
  awk '/^slot wait ms / { if (match($0, /mean [0-9.]+/)) m = substr($0, RSTART + 5, RLENGTH - 5)
                          if (match($0, /p90 [0-9.]+/)) p = substr($0, RSTART + 4, RLENGTH - 4) }
       END { printf "%s,%s", (m == "" ? "nan" : m), (p == "" ? "nan" : p) }' "$1" 2>/dev/null || printf 'nan,nan'
}
kms_metrics() {   # kms_metrics <stdout.log> -> "flips,timeouts,commit_failures,busy,master_lost,shutdown"
  awk '
    /^  kms: flips / {
      if (match($0, /flips [0-9]+/)) f = substr($0, RSTART + 6, RLENGTH - 6)
      if (match($0, /flip_timeouts [0-9]+/)) t = substr($0, RSTART + 14, RLENGTH - 14)
      if (match($0, /commit failures [0-9]+/)) c = substr($0, RSTART + 16, RLENGTH - 16)
      if (match($0, /busy [0-9]+/)) b = substr($0, RSTART + 5, RLENGTH - 5)
      if (match($0, /master_lost [a-z]+/)) m = substr($0, RSTART + 12, RLENGTH - 12)
    }
    /^  kms: shutdown / { s = substr($0, 17); gsub(/,/, ";", s) }
    END {
      printf "%s,%s,%s,%s,%s,%s", (f == "" ? "-" : f), (t == "" ? "-" : t), (c == "" ? "-" : c), (b == "" ? "-" : b),
        (m == "" ? "-" : m), (s == "" ? "-" : s)
    }' "$1" 2>/dev/null || printf -- '-,-,-,-,-,-'
}
# gpu_stats deltas / frames: "csd,tfu,bin+render,total" ms per frame (system-wide, all clients)
v3d_delta() {
  awk -v n="$3" '
    FNR == NR && NR > 1 { r0[$1] = $4; next }
    FNR > 1 { r1[$1] = $4 }
    END {
      if (n + 0 <= 0 || !("csd" in r0) || !("csd" in r1)) { printf "nan,nan,nan,nan"; exit }
      for (q in r1) { d[q] = (r1[q] - r0[q]) / 1e6 / n; tot += d[q] }
      printf "%.4f,%.4f,%.4f,%.4f", d["csd"], d["tfu"], d["bin"] + d["render"], tot
    }' "$1" "$2" 2>/dev/null || printf 'nan,nan,nan,nan'
}
# Sum of drm-engine-* ns over every block of an fdinfo dump (one process).
fdinfo_engine_ns() { awk '/^drm-engine-/ { s += $2 } END { printf "%.0f", s + 0 }' "$1" 2>/dev/null || echo 0; }
# stat_ticks <stat file> (utime + stime ticks) lives in lib/common.sh.

SCRIPT_COLS="id,backend,pace,run,exit_code,cool_wait_s,hailort_warnings,hailort_errors,unreliable,hailo_pre_c,note"
CSV1_COLS="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok"
CSV2_COLS="postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,display,sync,pace_cli,disp_convert_mean,disp_convert_p90,disp_overlay_mean,present_interval_median,e2e_glass_mean,e2e_glass_p90,presented,display_dropped,recreates,held_frames_max,slot_timeout_count,nms,gpu_nms_mean,gpu_nms_p90,nms_fallback,nms_verify_mismatch,scanout_zero_copy,plane_color_ok,hvs_underrun_delta,overlay_skipped,frame_cb_timeouts,zero_copy_feedback"
CSV3_COLS="display_warmup,glass_median,glass_n,present_interval_p90,present_interval_n,late_frames,late_expected_gap,seq_gap_max,seq_gaps,seq_breaks,discarded,refresh_hz,d2s_mean,d2s_median,d2s_p90,d2s_n"
EXTRA_COLS="slot_wait_ms_mean,slot_wait_ms_p90,kms_flips,kms_flip_timeouts,kms_commit_failures,kms_busy,kms_master_lost,kms_shutdown,labwc_pid,labwc_cpu_ms_per_frame,labwc_v3d_ms_per_frame,yvo_ticks_ms_per_frame,yvo_v3d_ms_per_frame,v3d_csd_ms_per_frame,v3d_tfu_ms_per_frame,v3d_binrender_ms_per_frame,v3d_total_ms_per_frame,wall_script_s,session,checks"
N_CSV1=22   # fields after "RESULT_CSV,"  (label ... ok)
N_CSV2=35   # fields after "RESULT_CSV2," (label + 10 postproc incl. ok + 14 display + 5 NMS + 6 scanout)
N_CSV3=18   # fields after "RESULT_CSV3," (label + 12 + ok + d2s_mean,d2s_median,d2s_p90,d2s_n)
#   ⚠ If main.cpp appends fields to a RESULT_CSV* line, N_CSV* and the *_COLS names must change together.
CSV_HEADER="$SCRIPT_COLS,$CSV1_COLS,$CSV2_COLS,$CSV3_COLS,$EXTRA_COLS"

# nan_list, arg_val, result_csv12, gpu_stats_snap and cool_before_run live in lib/common.sh.

# make_row <id> <run> <rc> <cool> <nwarn> <nerr> <t_pre> <run_dir> <args> -> one results_raw.csv line
make_row() {
  local id="$1" r="$2" rc="$3" cool="$4" nwarn="$5" nerr="$6" t_pre="$7" dir="$8" args="$9"
  local log="$dir/stdout.log" l1="" l2="" l3="" notes=() unreliable=0 backend pace frames extra
  backend=$(arg_val "$args" --display-backend nan); pace=$(arg_val "$args" --display-pace nan)
  (( nwarn > 0 )) && unreliable=1
  result_csv12 "$log" "$id" "$args" nan nan
  if [[ -f "$log" ]]; then
    l3=$(grep '^RESULT_CSV3,' "$log" | tail -1 | sed 's/^RESULT_CSV3,//' || true)
  fi
  if [[ -z "$l3" || "$(awk -F, '{print NF}' <<<"$l3")" != "$N_CSV3" ]]; then
    notes+=(no_result_csv3); l3="$(nan_list $((N_CSV3 - 2)))"; l1="${l1%,*},0"
  else
    # drop label (1) and ok (14); d2s_* (15..18) follow ok
    l3=$(awk -F, 'BEGIN{OFS=","}{ out=""; for(i=2;i<=NF;i++){ if(i==14) continue; out = (out=="" ? $i : out OFS $i) } print out }' <<<"$l3")
  fi
  frames=$(cut -d, -f6 <<<"$l1")
  # Side-file metrics. CLK_TCK ticks -> ms. labwc must be the same pid before and after.
  local tck lpid lt0 lt1 lcpu=nan lv3d=nan yt yv yvo_cpu=nan yvo_v3d=nan wall
  tck=$(getconf CLK_TCK)
  lpid=$(sed -n 's/^labwc_pid //p' "$dir/meta.txt" 2>/dev/null || true)
  lt0=$(stat_ticks "$dir/labwc_stat_pre.txt"); lt1=$(stat_ticks "$dir/labwc_stat_post.txt")
  if [[ -n "$lpid" && "$lpid" != "-" && -n "$lt0" && -n "$lt1" ]] && awk -v n="$frames" 'BEGIN{exit !(n+0>0)}'; then
    lcpu=$(awk -v a="$lt0" -v b="$lt1" -v t="$tck" -v n="$frames" 'BEGIN{printf "%.4f", (b - a) * 1000 / t / n}')
    lv3d=$(awk -v a="$(fdinfo_engine_ns "$dir/labwc_fdinfo_pre.txt")" -v b="$(fdinfo_engine_ns "$dir/labwc_fdinfo_post.txt")" \
               -v n="$frames" 'BEGIN{printf "%.4f", (b - a) / 1e6 / n}')
  fi
  yt=$(stat_ticks "$dir/yvo_stat_last.txt"); yv=$(fdinfo_engine_ns "$dir/yvo_fdinfo_last.txt")
  if [[ -n "$yt" ]] && awk -v n="$frames" 'BEGIN{exit !(n+0>0)}'; then
    yvo_cpu=$(awk -v a="$yt" -v t="$tck" -v n="$frames" 'BEGIN{printf "%.4f", a * 1000 / t / n}')
    yvo_v3d=$(awk -v a="$yv" -v n="$frames" 'BEGIN{printf "%.4f", a / 1e6 / n}')
  fi
  wall=$(sed -n 's/^wall_script_s //p' "$dir/meta.txt" 2>/dev/null || echo nan)
  extra="$(slot_wait_metrics "$log"),$(kms_metrics "$log"),${lpid:--},$lcpu,$lv3d,$yvo_cpu,$yvo_v3d"
  extra+=",$(v3d_delta "$dir/gpu_stats_pre.txt" "$dir/gpu_stats_post.txt" "$frames"),${wall:-nan}"
  extra+=",$(sed -n 's/^session //p' "$dir/meta.txt" 2>/dev/null || echo '?')"
  (( rc != 0 )) && notes+=("rc_$rc")
  local note="-"; (( ${#notes[@]} )) && note=$(IFS=';'; echo "${notes[*]}")
  local row="$id,$backend,$pace,$r,$rc,$cool,$nwarn,$nerr,$unreliable,$t_pre,$note,$l1,$l2,$l3,$extra"
  echo "$row,$(run_checks "$row")"
}

# run_checks <row without checks> -> "pass" or the failed checks joined by ';' (plan §7 + kms summary contract)
run_checks() {
  /usr/bin/python3 - "$CSV_HEADER" "$1" <<'PYEOF'
import sys
hdr = sys.argv[1].split(",")[:-1]
v = dict(zip(hdr, sys.argv[2].split(",")))
bad = []
if v.get("exit_code") != "0": bad.append("exit")
if v.get("display") != v.get("backend"): bad.append("backend")   # e.g. wayland silently became offscreen
if v.get("ok") != "1": bad.append("ok")
if v.get("hailort_warnings") != "0": bad.append("hailort_warn")
# Invalid runs: nothing left after the --display-warmup cut (glass_n 0 -> glass_median nan), or no vblank gap to judge.
if v.get("glass_n") in ("0", "nan", ""): bad.append("no_glass")
if v.get("d2s_n") != v.get("glass_n"): bad.append("d2s_n")   # both series are pushed at the same screen time
if v.get("display") in ("wayland-planes", "kms"):
    if v.get("seq_gaps") in ("0", "nan", "") or v.get("late_frames") == "unverified": bad.append("late_unverified")
    if v.get("plane_color_ok") != "1": bad.append("color_ok")
    if v.get("scanout_zero_copy") != "1": bad.append("zero_copy")
    if v.get("hvs_underrun_delta") != "0": bad.append("underrun")
if v.get("display") == "kms":
    if v.get("kms_flip_timeouts") != "0": bad.append("flip_timeouts")
    if v.get("kms_master_lost") != "no": bad.append("master_lost")
    if v.get("kms_flips") != v.get("presented"): bad.append("flips!=presented")
print("pass" if not bad else ";".join(bad))
PYEOF
}

prev_warn=0

# run_one <id> <args> <run> -> sets rc, nwarn, nerr, run_dir
run_one() {
  local id="$1" args="$2" r="$3" lpid ypid="" t0 t1 tnow mid_done=0 n=0
  run_dir="$OUT_DIR/raw/${id}_run${r}"
  cool_before_run
  rm -rf "$run_dir"; mkdir -p "$run_dir"
  build_cmd "$id" "$args"
  if ! is_kms "$id"; then
    # Display safety right before the run (cool-down may have taken minutes).
    if ! desktop_ready "$DESKTOP_SID"; then
      echo "STOP: $CONNECTOR not Enabled / desktop not active / labwc not master before $id run $r (masters: $(masters))" >&2
      recovery_hint; exit 1
    fi
    wlr-randr > "$run_dir/wlr_randr_pre.txt" 2>&1 || true
  fi
  lpid=$(pgrep -x labwc | head -1 || true)
  echo "[$(date +%T)] run $r $id (hailo ${t_pre} C, cooled ${COOL_WAITED}s, session $(is_kms "$id" && echo tty || echo desktop))"
  clients_snap "$run_dir/dri_clients_pre.txt"
  gpu_stats_snap "$run_dir/gpu_stats_pre.txt"
  if [[ -n "$lpid" ]]; then proc_stat "$lpid" "$run_dir/labwc_stat_pre.txt"; drm_fdinfo "$lpid" "$run_dir/labwc_fdinfo_pre.txt"; fi
  t0=$(date +%s.%N)
  (cd "$run_dir" && exec timeout -s KILL "$TIMEOUT_S" "${cmd[@]}" > stdout.log 2>&1) &
  CHILD_PID=$!
  # Sample the child's /proc while it runs (the last sample is kept: <= SAMPLE_S before exit) + one debugfs state mid-run.
  while kill -0 "$CHILD_PID" 2>/dev/null; do
    [[ -z "$ypid" ]] && ypid=$(pgrep -P "$CHILD_PID" -x yvo_pipeline 2>/dev/null | head -1 || true)
    if [[ -n "$ypid" && -r /proc/$ypid/stat ]]; then
      proc_stat "$ypid" "$run_dir/yvo_stat_last.tmp" && [[ -s "$run_dir/yvo_stat_last.tmp" ]] && mv -f "$run_dir/yvo_stat_last.tmp" "$run_dir/yvo_stat_last.txt"
      if (( n % 4 == 0 )); then   # fdinfo every 4th sample (readlink over all fds)
        drm_fdinfo "$ypid" "$run_dir/yvo_fdinfo_last.tmp"
        [[ -s "$run_dir/yvo_fdinfo_last.tmp" ]] && mv -f "$run_dir/yvo_fdinfo_last.tmp" "$run_dir/yvo_fdinfo_last.txt"
      fi
    fi
    tnow=$(date +%s.%N)
    if (( ! mid_done )) && awk -v a="$t0" -v b="$tnow" -v m="$MID_SNAP_S" 'BEGIN{exit !(b - a >= m)}'; then
      cat "$DRI_DIR/state" > "$run_dir/dri_state_mid.txt" 2>/dev/null || echo unreadable > "$run_dir/dri_state_mid.txt"
      mid_done=1
    fi
    n=$((n + 1))
    sleep "$SAMPLE_S"
  done
  set +e
  wait "$CHILD_PID"
  rc=$?
  set -e
  CHILD_PID=""
  rm -f "$run_dir"/*.tmp
  t1=$(date +%s.%N)
  gpu_stats_snap "$run_dir/gpu_stats_post.txt"
  if [[ -n "$lpid" ]]; then proc_stat "$lpid" "$run_dir/labwc_stat_post.txt"; drm_fdinfo "$lpid" "$run_dir/labwc_fdinfo_post.txt"; fi
  clients_snap "$run_dir/dri_clients_post.txt"
  (( mid_done )) || echo "not taken (run shorter than MID_SNAP_S=$MID_SNAP_S)" > "$run_dir/dri_state_mid.txt"
  {
    echo "wall_script_s $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b - a}')"
    echo "clk_tck $(getconf CLK_TCK)"
    echo "labwc_pid ${lpid:--}"
    echo "labwc_pid_post $(pgrep -x labwc | head -1 || echo -)"
    echo "yvo_pid ${ypid:--}"
    echo "session $(is_kms "$id" && echo "tty:$TTY_SID" || echo "desktop:$DESKTOP_SID")"
    echo "mid_snap_s $MID_SNAP_S"
    echo "slots $SLOTS"
    printf 'cmd timeout -s KILL %s ' "$TIMEOUT_S"; printf '%q ' "${cmd[@]}"; echo
  } > "$run_dir/meta.txt"
  # labwc restarted in between -> its tick delta is meaningless
  if [[ "$(sed -n 's/^labwc_pid_post //p' "$run_dir/meta.txt")" != "${lpid:--}" ]]; then
    sed -i 's/^labwc_pid .*/labwc_pid -/' "$run_dir/meta.txt"
  fi
  read -r nwarn nerr < <(count_hailort_warnings "$run_dir/hailort.log")
  prev_warn=$nwarn
}

RAW="$OUT_DIR/results_raw.csv"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

for ((r = 1; r <= RUNS; r++)); do
  for c in "${selected[@]}"; do
    IFS='|' read -r id args <<<"$c"
    if is_kms "$id" && (( ! ON_TTY )); then
      to_tty || { echo "STOP: labwc still holds DRM master after activating tty $TTY_SID" >&2; to_desktop || recovery_hint; exit 1; }
    elif ! is_kms "$id" && (( ON_TTY )); then
      to_desktop || { echo "STOP: desktop did not come back" >&2; recovery_hint; exit 1; }
    fi
    run_one "$id" "$args" "$r"
    row=$(make_row "$id" "$r" "$rc" "$COOL_WAITED" "$nwarn" "$nerr" "$t_pre" "$run_dir" "$args")
    echo "$row" >> "$RAW"
    checks=${row##*,}
    if [[ "$checks" != "pass" ]]; then echo "  rc=$rc checks=$checks (see stdout.log)"; tail -3 "$run_dir/stdout.log" || true; fi
    echo "  $(grep -E '^wall|^cpu {5}|^slot wait|^glass-to-glass|^decode-to-screen|^display  |^scanout         (zero_copy|late)|^  kms: flips' "$run_dir/stdout.log" | tr -s ' ' | tr '\n' ' ') hailo_warn=$nwarn rc=$rc"
  done
  if (( ON_TTY )); then
    to_desktop || { echo "STOP: desktop did not come back after the kms runs" >&2; recovery_hint; exit 1; }
  fi
done

# Per-config aggregation (NaN-safe medians) + console table.
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'PYEOF'
import csv, math, statistics, sys

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan

def fin(v): return [x for x in v if not math.isnan(x)]
def nmed(v): v = fin(v); return statistics.median(v) if v else math.nan
def nmin(v): v = fin(v); return min(v) if v else math.nan
def nmax(v): v = fin(v); return max(v) if v else math.nan
def fmt(v, d=3): return "nan" if math.isnan(v) else f"{v:.{d}f}"

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
if not rows:
    print("no rows"); sys.exit(0)
order, groups = [], {}
for r in rows:
    if r["id"] not in groups:
        order.append(r["id"]); groups[r["id"]] = []
    groups[r["id"]].append(r)

METRICS = [("fps", 2), ("cpu_ms_per_frame", 3), ("e2e_ms_mean", 2), ("glass_median", 2), ("e2e_glass_p90", 2),
           ("e2e_glass_mean", 2), ("glass_n", 0), ("d2s_median", 2), ("d2s_p90", 2), ("d2s_mean", 2), ("d2s_n", 0),
           ("slot_wait_ms_mean", 3), ("slot_wait_ms_p90", 3), ("present_interval_median", 3), ("present_interval_p90", 3),
           ("presented", 0), ("display_dropped", 0), ("late_frames", 0), ("seq_gaps", 0), ("seq_gap_max", 0), ("seq_breaks", 0),
           ("discarded", 0),
           ("disp_convert_mean", 3), ("disp_overlay_mean", 3), ("held_frames_max", 0), ("overlay_skipped", 0),
           ("frame_cb_timeouts", 0), ("zero_copy_feedback", 0), ("hvs_underrun_delta", 0),
           ("labwc_cpu_ms_per_frame", 4), ("labwc_v3d_ms_per_frame", 4), ("yvo_ticks_ms_per_frame", 3),
           ("yvo_v3d_ms_per_frame", 4), ("v3d_csd_ms_per_frame", 4), ("v3d_tfu_ms_per_frame", 4),
           ("v3d_binrender_ms_per_frame", 4), ("v3d_total_ms_per_frame", 4), ("detections", 0)]
summary = {}
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")             # no CRLF (measure skill §5)
    hdr = ["id", "backend", "pace", "runs", "runs_pass", "exit_codes", "checks", "unreliable_runs", "hailort_warnings",
           "cool_wait_s_total", "hailo_pre_c_max", "plane_color_ok_all", "scanout_zero_copy_all", "kms_flip_timeouts_sum",
           "kms_master_lost_any", "notes"]
    for k, _ in METRICS:
        hdr += [f"{k}_median", f"{k}_min", f"{k}_max"]
    hdr += ["all_pass"]
    w.writerow(hdr)
    for cid in order:
        g = groups[cid]
        npass = sum(x["checks"] == "pass" for x in g)
        stats = {k: (nmed([fnum(x[k]) for x in g]), nmin([fnum(x[k]) for x in g]), nmax([fnum(x[k]) for x in g]))
                 for k, _ in METRICS}
        notes = sorted({n for x in g for n in x["note"].split(";") if n not in ("-", "")})
        colv = {x["plane_color_ok"] for x in g}
        zcv = {x["scanout_zero_copy"] for x in g}
        all_pass = int(npass == len(g))
        summary[cid] = {"stats": stats, "pass": npass, "runs": len(g),
                        "color": "-" if colv == {"-"} else ("1" if colv == {"1"} else "/".join(sorted(colv))),
                        "zc": "-" if zcv == {"-"} else ("1" if zcv == {"1"} else "/".join(sorted(zcv)))}
        fts = [fnum(x["kms_flip_timeouts"]) for x in g]
        row = [cid, g[0]["backend"], g[0]["pace"], len(g), npass, ";".join(x["exit_code"] for x in g),
               ";".join(x["checks"] for x in g), sum(x["unreliable"] == "1" for x in g),
               sum(int(fnum(x["hailort_warnings"])) if not math.isnan(fnum(x["hailort_warnings"])) else 0 for x in g),
               sum(int(fnum(x["cool_wait_s"])) if not math.isnan(fnum(x["cool_wait_s"])) else 0 for x in g),
               fmt(nmax([fnum(x["hailo_pre_c"]) for x in g]), 1), summary[cid]["color"], summary[cid]["zc"],
               "-" if all(math.isnan(t) for t in fts) else int(sum(fin(fts))),
               "-" if all(x["kms_master_lost"] == "-" for x in g) else int(any(x["kms_master_lost"] not in ("no", "-") for x in g)),
               ";".join(notes) or "-"]
        for k, d in METRICS:
            row += [fmt(v, d) for v in stats[k]]
        row += [all_pass]
        w.writerow(row)

def med(c, k, d): return fmt(summary[c]["stats"][k][0], d)
def rng(c, k, d): s = summary[c]["stats"][k]; return f"{fmt(s[0], d)} ({fmt(s[1], d)}..{fmt(s[2], d)})"

lines = []
for c in order:
    s = summary[c]
    lines.append([c, rng(c, "fps", 2), rng(c, "cpu_ms_per_frame", 3), med(c, "glass_median", 2), med(c, "e2e_glass_p90", 2),
                  med(c, "d2s_median", 2), med(c, "d2s_p90", 2), med(c, "slot_wait_ms_mean", 2),
                  med(c, "present_interval_median", 2), med(c, "present_interval_p90", 2), med(c, "presented", 0),
                  med(c, "display_dropped", 0), med(c, "late_frames", 0), med(c, "discarded", 0),
                  med(c, "labwc_cpu_ms_per_frame", 3), med(c, "v3d_csd_ms_per_frame", 3), med(c, "v3d_tfu_ms_per_frame", 3),
                  med(c, "v3d_total_ms_per_frame", 3), s["color"], s["zc"], med(c, "hvs_underrun_delta", 0),
                  f"{s['pass']}/{s['runs']}"])
hdr = ["id", "fps", "cpu_ms/f", "glass_med", "glass_p90", "d2s_med", "d2s_p90", "slotw_ms", "pres_med", "pres_p90", "presented", "dropped", "late",
       "discard", "labwc_ms/f", "csd_ms/f", "tfu_ms/f", "v3d_ms/f", "color", "zc", "underrun", "pass"]
wid = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)]
print("== per config (medians; fps / cpu as median (min..max); glass / d2s / present / late after --display-warmup;"
      " glass = slot acquired -> screen, d2s = decode return -> screen, slotw = slot wait mean) ==")
print(" | ".join(h.ljust(x) for h, x in zip(hdr, wid)))
print("-+-".join("-" * x for x in wid))
for l in lines:
    print(" | ".join(c.ljust(x) for c, x in zip(l, wid)))
bad = [c for c in order if summary[c]["pass"] != summary[c]["runs"]]
print("\nruns failing checks: " + (", ".join(bad) if bad else "none"))
PYEOF

{
  env_snapshot after
  echo "bin md5 (after measurement): $(md5sum "$BIN" | cut -d' ' -f1)"
} >> "$OUT_DIR/env_bench.txt"
