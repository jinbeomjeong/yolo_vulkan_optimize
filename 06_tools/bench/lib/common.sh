# Common functions for the bench scripts. Each script `source`s this file.
#
# Why this exists (2026-09-21):
#   The same functions were copy-pasted into 7 scripts and, **with nothing keeping them in sync, had actually diverged.**
#   - Of the 7 copies of `pcie_link()`, only `backpressure_bench.sh` lacked the Gen5 (32.0 GT/s) branch
#     -> it records a Gen5 link as `Gen?`. A wrong record of measurement conditions is the costliest mistake in this project.
#   - `cool_down()` differed per script in sensor (hailo/soc), threshold and loop structure, and
#     `decode_bench.sh` had it inline without a function, with no `nan` handling and no wait-time record.
#
# Prerequisites: PROJ (repository root) must be defined before sourcing this file, and
#   the following must be defined before calling the functions.
#   OUT_DIR        result folder (hailo_temp runs in $OUT_DIR/raw/_session)
#   COOL_C         cool-down target temperature
#   COOL_MAX_S     cool-down max wait in seconds
#
# Sensor choice: benches that use the NPU use hailo (default); benches that don't (preprocess, decode) use soc.

# ── Paths (defined only once, here) ─────────────────────────────────────────────
# All executables come out in CMAKE_RUNTIME_OUTPUT_DIRECTORY = <build>/bin (2026-09-22).
# This keeps the benches' binary paths from breaking when source folders move.
BUILD_DIR="${BUILD_DIR:-$PROJ/build}"
BIN_DIR="${BIN_DIR:-$BUILD_DIR/bin}"
PY_HAILO="${PY_HAILO:-/home/pi/miniconda3/envs/hailo-yolo/bin/python}"   # conda hailo-yolo Python

# ── Pre-checks ─────────────────────────────────────────────────────────────────
# require_bin <path> <name>: exit if the executable is missing.
require_bin() {
  [[ -x "$1" ]] || { echo "$2 not found: $1 (run cmake --build build first)" >&2; exit 1; }
}

# With CHECK_ONLY=1, only check the binaries and exit 0 **before lock, cool-down and measurement**.
# Use it to verify path settings without running the bench (e.g. after a folder relocation).
check_only_exit() {
  if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    echo "CHECK_ONLY: $(basename "$0") path check passed (BIN_DIR=$BIN_DIR)"
    exit 0
  fi
}

# ── Archived-result protection ─────────────────────────────────────────────────────────────
# If OUT_DIR already exists and is not empty, stop unless FORCE=1.
# 07_results/<experiment>/ is the source of truth for numbers. Re-running a bench with the default OUT_DIR overwrites env_bench.txt/results*.csv
# and rm -rf's raw/<config>_run<N>/, silently destroying the archive (STRUCTURE.md §3.3 content is immutable).
# Appending with APPEND=1 also modifies the existing folder, so it needs FORCE=1 too.
# Call after check_only_exit and before bench_lock_acquire/mkdir (CHECK_ONLY exits without writing anything).
guard_out_dir() {
  local dir="${1:-$OUT_DIR}"
  if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "OUT_DIR is not empty: $dir" >&2
    echo "  Stopping so archived results are not overwritten. Pass a new folder via OUT_DIR=..., or FORCE=1 to overwrite." >&2
    exit 1
  fi
}

# ── Temperature ───────────────────────────────────────────────────────────────────────
# Run from raw/_session/ so the session hailort.log lands there (STRUCTURE.md §3.5).
# Directly under raw/ there are only <config>_run<N>/ and _session/.
hailo_temp() {
  mkdir -p "$OUT_DIR/raw/_session" 2>/dev/null || { echo nan; return; }
  (cd "$OUT_DIR/raw/_session" && "$PY_HAILO" - <<'PYEOF' 2>/dev/null) || echo nan
from hailo_platform import Device
d = Device()
t = d.control.get_chip_temperature()
print(f"{max(t.ts0_temperature, t.ts1_temperature):.1f}")
d.release()
PYEOF
}

soc_temp() { vcgencmd measure_temp 2>/dev/null | sed -E 's/temp=([0-9.]+).*/\1/' || echo nan; }

throttled() { vcgencmd get_throttled 2>/dev/null || echo "throttled=?"; }

# ── PCIe link ──────────────────────────────────────────────────────────────────
# Do not trust values written in docs; read sysfs (measure skill §1).
# lspci -vv's LnkSta: prints nothing unless root, so it is not used.
pcie_link() {
  local d speed width gen
  for d in /sys/bus/pci/devices/*/; do
    [[ "$(cat "$d/vendor" 2>/dev/null)" == "0x1e60" ]] || continue   # Hailo
    speed=$(cat "$d/current_link_speed" 2>/dev/null)
    width=$(cat "$d/current_link_width" 2>/dev/null)
    if [[ -z "$speed" || -z "$width" ]]; then
      echo "pcie: read failed ($d current_link_speed/width)"; return
    fi
    case "${speed%% *}" in
      2.5) gen=Gen1 ;; 5.0) gen=Gen2 ;; 8.0) gen=Gen3 ;; 16.0) gen=Gen4 ;; 32.0) gen=Gen5 ;;
      *)   gen="Gen?" ;;
    esac
    echo "pcie: $speed x$width ($gen, $(basename "$d"))"
    return
  done
  echo "pcie: read failed (vendor 0x1e60 device not found in /sys/bus/pci/devices)"
}

# ── Package versions ───────────────────────────────────────────────────────────
# pkg_versions <pkg>…: prints one "<pkg>: <version>" line per package ("?" when dpkg-query fails).
# Extracted from the loop that was inline in dec_threads_bench.sh (2026-10-08) because three display benches need it.
#
# Why the compositor packages matter: the wayland-planes path depends on the compositor inverting COLOR_RANGE
#   ("fullhack", wayland_plane_scanout.hpp). On 2026-10-07 an apt upgrade moved labwc 0.20.1 -> 0.20.2 and
#   libwlroots-0.20 +rpt3 -> +rpt4 and **no env_bench.txt had recorded either version**, so the change could not be
#   dated from the archived results (07_results/display_vulkan_hdmi/verification/color_range_rpt4/).
#   Benches that pass --display therefore record labwc and libwlroots-0.20 as well.
pkg_versions() {
  local p
  for p in "$@"; do
    echo "$p: $(dpkg-query -W -f='${Version}' "$p" 2>/dev/null || echo '?')"
  done
}

# Default package set for the benches that use the display path (decode + NPU + GPU + compositor).
PKGS_DISPLAY_BENCH=(libavcodec61 hailort mesa-vulkan-drivers labwc libwlroots-0.20)

# ── Cool-down ───────────────────────────────────────────────────────────────────────
# $1 = reason (for the log), $2 = sensor hailo|soc (default hailo)
# Leaves the seconds waited in COOL_WAITED — measure skill §2 requires recording it.
# If the temperature can't be read (nan), **wait** (do not measure prematurely).
cool_down() {
  local reason="${1:-}" sensor="${2:-hailo}" t waited=0
  local read_temp=hailo_temp
  [[ "$sensor" == "soc" ]] && read_temp=soc_temp
  t=$("$read_temp")
  echo "  cooling ($reason): ${sensor} ${t} C -> <= ${COOL_C} C (max ${COOL_MAX_S}s)"
  while awk -v t="$t" -v c="$COOL_C" 'BEGIN{exit !(t=="nan" || t>c)}' && (( waited < COOL_MAX_S )); do
    sleep 5; waited=$((waited + 5)); t=$("$read_temp")
  done
  echo "  cooled ${waited}s, ${sensor} ${t} C"
  COOL_WAITED=$waited
}

# ── hailort.log reliability verdict (measure skill §4) ─────────────────────────────────
# If there is any temperature/throttle/clock warning, that run's performance numbers are not trustworthy.
WARN_RE='[Tt]emperature|[Tt]hrottl|[Oo]verheat|clock'

# $1 = hailort.log path. Prints "warn_count err_count".
count_hailort_warnings() {
  local log="$1" nwarn=0 nerr=0
  if [[ -f "$log" ]]; then
    nwarn=$(grep -E '\[warning\]' "$log" | grep -cE "$WARN_RE" || true)
    nerr=$(grep -c '\[error\]' "$log" || true)
  fi
  echo "$nwarn $nerr"
}

# ── Bench lock ───────────────────────────────────────────────────────────────────
# The npu_guard hook (.claude/hooks/npu_guard.sh) uses the PID in this file to decide "a bench is running".
# Because a build/test sneaking in during measurement silently corrupts CPU-time numbers without any error.
# The hook checks whether the PID is alive, so a lock left by an abnormal exit is automatically stale.
bench_lock_acquire() {
  BENCH_LOCK="${BENCH_LOCK:-$PROJ/build/.bench_running}"
  mkdir -p "$(dirname "$BENCH_LOCK")"
  printf '%s' "$$" > "$BENCH_LOCK"
  trap 'rm -f "$BENCH_LOCK"' EXIT
  trap 'rm -f "$BENCH_LOCK"; exit 130' INT; trap 'rm -f "$BENCH_LOCK"; exit 143' TERM   # stop, don't continue
}

# ── Per-run cool-down gate (2026-10-12) ─────────────────────────────────────────────────────────
# cool_before_run: the per-run cool-down gate of the NPU benches. Sets t_pre (chip temp right before the run) and
# COOL_WAITED. Cools when the previous run's hailort.log had temperature/clock warnings (prev_warn > 0), when
# COOL_EVERY=1 and the chip is above COOL_C (or unreadable), or when the chip is above HOT_C (measure skill rule).
# Needs: prev_warn, COOL_EVERY, COOL_C, HOT_C. Was copied verbatim into dec_threads/nms/display_planes_bench.sh;
# gst_pipeline_bench.sh had the same gate without the COOL_EVERY branch and pins COOL_EVERY=0.
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

# ── env_bench.txt lines shared by several benches (2026-10-12) ───────────────────────────────
# env_git_head: "git HEAD: <sha>" plus " (tracked changes)" when tracked files are modified.
env_git_head() {
  local dirty=""
  [[ -n "$(git -C "$PROJ" status --porcelain --untracked-files=no 2>/dev/null)" ]] && dirty=" (tracked changes)"
  echo "git HEAD: $(git -C "$PROJ" rev-parse HEAD 2>/dev/null || echo '?')${dirty}"
}

# env_bin <path>: "bin: <path> md5 <md5> mtime <mtime>".
env_bin() { echo "bin: $1 md5 $(md5sum "$1" | cut -d' ' -f1) mtime $(stat -c %y "$1")"; }

# gpu_stats_snap <out>: copy of the V3D gpu_stats file ($GPU_STATS, set by the caller); empty file when unreadable.
gpu_stats_snap() { cat "$GPU_STATS" 2>/dev/null > "$1" || : > "$1"; }

# ── results_raw.csv row helpers (dec_threads/nms/display_planes_bench.sh, 2026-10-12) ──────────────
# nan_list <n>: n comma-separated "nan" fields (no newline).
nan_list() { local n="$1" i out=""; for ((i = 0; i < n; i++)); do out+="${out:+,}nan"; done; printf '%s' "$out"; }

arg_val() {   # arg_val <args> <flag> <default>
  local a=() i
  read -ra a <<<"$1"
  for ((i = 0; i + 1 < ${#a[@]}; i++)); do [[ "${a[i]}" == "$2" ]] && { printf '%s' "${a[i+1]}"; return; }; done
  printf '%s' "$3"
}

# result_csv12 <stdout.log> <id> <args> <mode default> <layout default>
#   The RESULT_CSV / RESULT_CSV2 part of make_row(), which differs per bench only in the columns around it.
#   Sets the CALLER's l1 (RESULT_CSV after its prefix, N_CSV1 fields) and l2 (RESULT_CSV2 without label and its
#   duplicate ok, N_CSV2 - 2 fields) and appends no_result_csv / no_result_csv2 to the caller's notes array — the
#   caller declares `local l1 l2 notes=()` (bash dynamic scoping). A missing or wrong-width line becomes a nan row;
#   a missing RESULT_CSV2 also forces ok=0 (last field of l1).
result_csv12() {
  local log="$1" id="$2" args="$3"
  l1=""; l2=""
  if [[ -f "$log" ]]; then
    l1=$(grep '^RESULT_CSV,' "$log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
    l2=$(grep '^RESULT_CSV2,' "$log" | tail -1 | sed 's/^RESULT_CSV2,//' || true)
  fi
  if [[ -z "$l1" || "$(awk -F, '{print NF}' <<<"$l1")" != "$N_CSV1" ]]; then
    notes+=(no_result_csv)
    l1="$id,$(arg_val "$args" --mode "$4"),$(arg_val "$args" --layout "$5"),$(arg_val "$args" --batch nan),$(arg_val "$args" --slots nan),$(nan_list 16),0"
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
}

# ── Compositor (labwc) CPU over a run (2026-10-16) ─────────────────────────────────────────────
# proc_stat / stat_ticks moved here from display_planes_bench.sh so display_bench.sh records the same
# raw/<run>/labwc_stat_{pre,post}.txt. Fields are split on blanks, which is right for comm values without
# spaces ("(labwc)", "(yvo_pipeline)").
# proc_stat <pid> <out>: one-line copy of /proc/<pid>/stat; empty file when the process is gone.
#   cat, not the former printf "$(< ...)": bash 5.2 exits a set -e script when $(< file) fails, even under ||, so a
#   process exiting mid-run killed the bench. For a live process the output is byte-identical (one line + "\n").
proc_stat() { cat /proc/"$1"/stat > "$2" 2>/dev/null || : > "$2"; }
stat_ticks() { awk '{ print $14 + $15 }' "$1" 2>/dev/null || echo ""; }   # utime + stime (clock ticks)

# labwc_pid: pid of the labwc compositor, empty when there is none (headless, kms tty session).
labwc_pid() { pgrep -x labwc | head -1 || true; }

# labwc_cpu_cols <pre stat> <post stat> <frames> -> "labwc_cpu_ms,labwc_cpu_ms_per_frame" (no newline).
#   utime+stime delta of one labwc process between the two /proc/<pid>/stat copies, ticks -> ms (CLK_TCK, 10 ms
#   resolution at 100 Hz). "nan,nan" when a file is missing or empty (no labwc, or it exited during the run) or when
#   pid (field 1) or starttime (field 22) differ (a different process); per-frame is nan when frames is not > 0.
labwc_cpu_cols() {
  [[ -s "$1" && -s "$2" ]] || { printf 'nan,nan'; return; }
  awk -v n="$3" -v t="$(getconf CLK_TCK)" '
    FILENAME == ARGV[1] && FNR == 1 { p = $1; s = $22; a = $14 + $15; h0 = 1 }
    FILENAME == ARGV[2] && FNR == 1 { q = $1; e = $22; b = $14 + $15; h1 = 1 }
    END {
      if (!h0 || !h1 || p != q || s != e || t + 0 <= 0) { printf "nan,nan"; exit }
      ms = (b - a) * 1000 / t
      if (n + 0 > 0) printf "%.1f,%.4f", ms, ms / n; else printf "%.1f,nan", ms
    }' "$1" "$2"
}
