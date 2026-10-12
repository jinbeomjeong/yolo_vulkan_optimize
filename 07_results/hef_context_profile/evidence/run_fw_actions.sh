#!/usr/bin/env bash
# Per-context NPU timing of 03_models/yolov11m.hef (07_results/hef_context_profile).
# For each repetition (outer loop) and batch size B in {1, 2, 8} (inner loop):
#   hailortcli run2 -t 3 -m raw_async measure-fw-actions --output-path fwactions.json set-net <hef> --batch-size B
# raw_async is required (the default full_sync mode is rejected by measure-fw-actions), and the HEF is passed via
# set-net (passing it right after the subcommand gives "Only one model").
#
# Measure skill rules applied:
#   - environment snapshot -> env_bench.txt, bench lock (npu_guard blocks builds/tests while this runs)
#   - cooling gate before each run (cool_before_run: chip > HOT_C=75 C, or warnings in the previous hailort.log,
#     -> wait until <= COOL_C=65 C, max COOL_MAX_S)
#   - each run executes with raw/b<B>_run<N>/ as CWD so hailort.log lands there (never in the repo root)
#   - hailort.log temperature/clock warnings counted per run (count_hailort_warnings)
# Per-run record: evidence/run_log.csv. Analysis afterwards: 06_tools/python/fw_actions_breakdown.py.
set -euo pipefail
PROJ=$(cd "$(dirname "$0")/../../.." && pwd)
OUT_DIR="$PROJ/07_results/hef_context_profile"
EVID="$OUT_DIR/evidence"
HEF="$PROJ/03_models/yolov11m.hef"
REPS=${REPS:-3}
BATCHES=(${BATCHES:-1 2 8})
RUN_S=${RUN_S:-3}
COOL_C=${COOL_C:-65}
HOT_C=${HOT_C:-75}
COOL_MAX_S=${COOL_MAX_S:-120}
COOL_EVERY=0
# shellcheck source=../../../06_tools/bench/lib/common.sh
source "$PROJ/06_tools/bench/lib/common.sh"

command -v hailortcli >/dev/null || { echo "hailortcli not found" >&2; exit 1; }
[[ -f "$HEF" ]] || { echo "missing $HEF" >&2; exit 1; }
guard_out_dir "$OUT_DIR/raw"     # evidence/ already holds this script; protect the run folders

bench_lock_acquire
mkdir -p "$OUT_DIR/raw/_session" "$EVID"

# ── environment snapshot ──
{
  echo "# hef_context_profile environment snapshot $(date -Iseconds)"
  echo "# measure-fw-actions per-context timing, B = ${BATCHES[*]}, ${REPS} repetitions, run2 -t ${RUN_S}"
  env_git_head
  echo "kernel: $(uname -r)"
  pcie_link
  echo "soc: $(vcgencmd measure_temp) $(throttled) $(vcgencmd measure_clock arm)"
  echo "$(vcgencmd get_config v3d_freq)"
  echo "hailo temp: $(hailo_temp) C"
  echo "hef: 03_models/yolov11m.hef md5 $(md5sum "$HEF" | cut -d' ' -f1) size $(stat -c %s "$HEF")"
  echo "hailortcli: $(hailortcli --version 2>&1 | head -1)"
  echo "hailo_pci driver: $(cat /sys/module/hailo_pci/version 2>/dev/null || echo '?')"
  echo "bench lock: $(cat "$BENCH_LOCK") (this script)"
  echo "/dev/hailo0 users: $(lsof -t /dev/hailo0 2>/dev/null | wc -l)"
  echo "cooling gate: HOT_C=${HOT_C} COOL_C=${COOL_C} COOL_MAX_S=${COOL_MAX_S} (hailo chip, before every run)"
  echo "--- hailortcli fw-control identify"
  (cd "$OUT_DIR/raw/_session" && hailortcli fw-control identify 2>&1)
} > "$OUT_DIR/env_bench.txt"
(cd "$OUT_DIR/raw/_session" && hailortcli parse-hef "$HEF" > "$EVID/parse_hef.txt" 2>&1)

LOG="$EVID/run_log.csv"
echo "rep,batch,run_dir,start,end,rc,t_pre_c,cool_waited_s,t_post_c,fps_stdout,fps_json,hailort_warn,hailort_err" > "$LOG"
prev_warn=0
for ((rep = 1; rep <= REPS; rep++)); do
  for B in "${BATCHES[@]}"; do
    id="b${B}_run${rep}"
    rd="$OUT_DIR/raw/$id"
    mkdir -p "$rd"
    cool_before_run
    start=$(date -Iseconds)
    echo "[$(date +%T)] $id  chip ${t_pre} C (cooled ${COOL_WAITED}s)"
    rc=0
    (cd "$rd" && hailortcli run2 -t "$RUN_S" -m raw_async measure-fw-actions --output-path fwactions.json \
        set-net "$HEF" --batch-size "$B") > "$rd/stdout.log" 2>&1 || rc=$?
    end=$(date -Iseconds)
    t_post=$(hailo_temp)
    read -r nwarn nerr < <(count_hailort_warnings "$rd/hailort.log")
    prev_warn=$nwarn
    fps_out=$(grep -oiE 'fps[^0-9]*[0-9]+(\.[0-9]+)?' "$rd/stdout.log" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?$' || echo nan)
    fps_json=$("$PY_HAILO" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][0]["fps"])' \
        "$rd/fwactions.json" 2>/dev/null || echo nan)
    echo "$rep,$B,raw/$id,$start,$end,$rc,$t_pre,$COOL_WAITED,$t_post,$fps_out,$fps_json,$nwarn,$nerr" >> "$LOG"
    echo "    rc $rc  fps(stdout) $fps_out  fps(json) $fps_json  chip after ${t_post} C  hailort.log warn $nwarn err $nerr"
  done
done
echo "done: $LOG"
