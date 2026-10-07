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
