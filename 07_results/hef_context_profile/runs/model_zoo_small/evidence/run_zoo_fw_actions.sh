#!/usr/bin/env bash
# Per-context NPU timing and throughput of the Hailo Model Zoo yolov11 n/s/m HEFs (analysis only; the default
# model 03_models/yolov11m.hef is not replaced). Same method as 07_results/hef_context_profile/evidence/run_fw_actions.sh.
#
# Usage:  run_zoo_fw_actions.sh OUT            measure (OUT must be new or empty; FORCE=1 to overwrite)
#         ANALYZE_ONLY=1 VIEW_DIR=<tmp> run_zoo_fw_actions.sh OUT
#                                              no NPU use: build per-model views under VIEW_DIR (symlinks) and run
#                                              06_tools/python/fw_actions_breakdown.py and hef_context_profile's
#                                              evidence/context_table.py on each model
#   OUT is e.g. 07_results/hef_context_profile/runs/model_zoo_small (planned location).
#
# Loops (repetition outermost, B innermost):
#   for rep in 1..REPS; for model in MODELS:
#     fw-actions: for B in FW_BATCHES:
#       hailortcli run2 -t RUN_S -m raw_async measure-fw-actions --output-path fwactions.json set-net <hef> --batch-size B
#       -> raw/<model>_b<B>_run<rep>/
#     throughput: for B in BENCH_BATCHES:
#       hailortcli benchmark --batch-size B -t BENCH_S --csv benchmark.csv <hef>
#       -> raw/<model>_bench_b<B>_run<rep>/
#
# Measure skill rules applied:
#   - environment snapshot -> OUT/env_bench.txt, bench lock build/.bench_running with this PID (removed by trap)
#   - cooling gate before every run: Hailo chip (cool_before_run: > HOT_C=75 C or warnings in the previous
#     hailort.log -> wait until <= COOL_C=65 C) and SoC (> SOC_MAX_C=75 C -> wait until <= SOC_MAX_C),
#     each at most COOL_MAX_S
#   - every run executes with its raw/<cfg>_run<N>/ folder as CWD so hailort.log lands there
#   - hailort.log temperature/clock warnings and [error] lines counted per run
# Per-run records: OUT/results_raw_fwactions.csv, OUT/results_raw_benchmark.csv.
# Do not quote the JSON fps of fwactions.json; use the stdout fps (fps_stdout column).
set -euo pipefail
PROJ=/home/pi/workspace/yolo_vulkan_optimize
[[ $# -eq 1 ]] || { echo "usage: $0 OUT   (see header)" >&2; exit 2; }
OUT_DIR=$(realpath -m "$1")
MODELS=(${MODELS:-yolov11n yolov11s yolov11m})
REPS=${REPS:-3}
FW_BATCHES=(${FW_BATCHES:-1 2 8})
BENCH_BATCHES=(${BENCH_BATCHES:-1 8})
RUN_S=${RUN_S:-3}
BENCH_S=${BENCH_S:-10}          # per benchmark mode (hw_only, streaming, latency) -> about 3 * BENCH_S per run
COOL_C=${COOL_C:-65}
HOT_C=${HOT_C:-75}
SOC_MAX_C=${SOC_MAX_C:-75}
COOL_MAX_S=${COOL_MAX_S:-180}
COOL_EVERY=0
# shellcheck source=/home/pi/workspace/yolo_vulkan_optimize/06_tools/bench/lib/common.sh
source "$PROJ/06_tools/bench/lib/common.sh"

hef_of() { echo "$PROJ/03_models/$1.hef"; }

# ── analysis only (no NPU) ──
if [[ "${ANALYZE_ONLY:-0}" == "1" ]]; then
  VIEW_DIR=${VIEW_DIR:?set VIEW_DIR to an empty scratch folder}
  CTX_TABLE="$PROJ/07_results/hef_context_profile/evidence/context_table.py"
  BREAKDOWN="$PROJ/06_tools/python/fw_actions_breakdown.py"
  [[ -f "$OUT_DIR/results_raw_fwactions.csv" ]] || { echo "missing $OUT_DIR/results_raw_fwactions.csv" >&2; exit 1; }
  for m in "${MODELS[@]}"; do
    v="$VIEW_DIR/$m"
    rm -rf "$v"; mkdir -p "$v/raw" "$v/evidence"
    # context_table.py expects raw/b<B>_run<N>/fwactions.json and evidence/run_log.csv (columns rep,batch,fps_stdout)
    awk -F, -v m="$m" 'NR == 1 || $1 == m' "$OUT_DIR/results_raw_fwactions.csv" | cut -d, -f2- > "$v/evidence/run_log.csv"
    files=()
    for d in "$OUT_DIR"/raw/"$m"_b*_run*/; do
      [[ -f "$d/fwactions.json" ]] || continue
      id=$(basename "$d"); ln -s "$d" "$v/raw/${id#"${m}_"}"
      files+=("$d/fwactions.json")
    done
    (( ${#files[@]} > 0 )) || { echo "no fwactions.json for $m" >&2; continue; }
    echo "################ $m: context_table.py"
    python3 -I "$CTX_TABLE" "$v"
    echo "################ $m: fw_actions_breakdown.py (${#files[@]} files)"
    python3 -I "$BREAKDOWN" "${files[@]}" --json "$v/breakdown.json"
  done
  exit 0
fi

# ── measurement ──
command -v hailortcli >/dev/null || { echo "hailortcli not found" >&2; exit 1; }
for m in "${MODELS[@]}"; do [[ -f "$(hef_of "$m")" ]] || { echo "missing $(hef_of "$m")" >&2; exit 1; }; done
if [[ -e "$PROJ/build/.bench_running" ]] && kill -0 "$(cat "$PROJ/build/.bench_running" 2>/dev/null)" 2>/dev/null; then
  echo "another bench holds build/.bench_running (PID $(cat "$PROJ/build/.bench_running")); not starting" >&2; exit 1
fi
guard_out_dir "$OUT_DIR"

bench_lock_acquire
mkdir -p "$OUT_DIR/raw/_session" "$OUT_DIR/evidence"
exec > >(tee -a "$OUT_DIR/raw/_session/bench_stdout.log") 2>&1

# SoC gate: wait while the SoC is above SOC_MAX_C (at most COOL_MAX_S). Sets SOC_WAITED and soc_pre.
soc_gate() {
  local t waited=0
  t=$(soc_temp)
  while awk -v t="$t" -v c="$SOC_MAX_C" 'BEGIN{exit !(t=="nan" || t>c)}' && (( waited < COOL_MAX_S )); do
    (( waited == 0 )) && echo "  cooling (soc ${t} C > ${SOC_MAX_C} C)"
    sleep 5; waited=$((waited + 5)); t=$(soc_temp)
  done
  SOC_WAITED=$waited; soc_pre=$t
}

{
  echo "# hef_context_profile/model_zoo_small environment snapshot $(date -Iseconds)"
  echo "# models: ${MODELS[*]}; fw-actions B = ${FW_BATCHES[*]} (run2 -t ${RUN_S}); benchmark B = ${BENCH_BATCHES[*]} (-t ${BENCH_S}); ${REPS} repetitions (rep outer, B inner)"
  env_git_head
  echo "kernel: $(uname -r)"
  pcie_link
  echo "soc: $(vcgencmd measure_temp) $(throttled) $(vcgencmd measure_clock arm)"
  vcgencmd get_config v3d_freq
  cpu_governor
  echo "hailo temp: $(hailo_temp) C"
  for m in "${MODELS[@]}"; do
    h=$(hef_of "$m")
    echo "hef: 03_models/$m.hef sha256 $(sha256sum "$h" | cut -d' ' -f1) size $(stat -c %s "$h")"
  done
  echo "hef source: Hailo Model Zoo v2.19.0 compiled/hailo8 (n/s/m byte-identical to v2.20.0)"
  echo "hailortcli: $(hailortcli --version 2>&1 | head -1)"
  echo "hailo_pci driver: $(cat /sys/module/hailo_pci/version 2>/dev/null || echo '?')"
  echo "bench lock: $(cat "$BENCH_LOCK") (this script)"
  echo "/dev/hailo0 users: $(lsof -t /dev/hailo0 2>/dev/null | wc -l)"
  echo "cooling gate: hailo HOT_C=${HOT_C} COOL_C=${COOL_C}; soc SOC_MAX_C=${SOC_MAX_C}; COOL_MAX_S=${COOL_MAX_S} (before every run)"
  echo "--- hailortcli fw-control identify"
  (cd "$OUT_DIR/raw/_session" && hailortcli fw-control identify 2>&1)
} > "$OUT_DIR/env_bench.txt"
for m in "${MODELS[@]}"; do
  (cd "$OUT_DIR/raw/_session" && hailortcli parse-hef "$(hef_of "$m")" --parse-streams --parse-vstreams \
      > "$OUT_DIR/evidence/parse_hef_$m.txt" 2>&1)
done

FW_LOG="$OUT_DIR/results_raw_fwactions.csv"
BN_LOG="$OUT_DIR/results_raw_benchmark.csv"
echo "model,rep,batch,run_dir,start,end,rc,t_pre_c,cool_waited_s,soc_pre_c,soc_waited_s,t_post_c,fps_stdout,hailort_warn,hailort_err" > "$FW_LOG"
echo "model,rep,batch,run_dir,start,end,rc,t_pre_c,cool_waited_s,soc_pre_c,soc_waited_s,t_post_c,fps_streaming,fps_hw_only,hw_latency_ms,hailort_warn,hailort_err" > "$BN_LOG"

gate() { cool_before_run; soc_gate; }

prev_warn=0
for ((rep = 1; rep <= REPS; rep++)); do
  for m in "${MODELS[@]}"; do
    hef=$(hef_of "$m")
    for B in "${FW_BATCHES[@]}"; do
      id="${m}_b${B}_run${rep}"; rd="$OUT_DIR/raw/$id"; mkdir -p "$rd"
      gate
      start=$(date -Iseconds)
      echo "[$(date +%T)] $id  chip ${t_pre} C (cooled ${COOL_WAITED}s)  soc ${soc_pre} C (waited ${SOC_WAITED}s)"
      rc=0
      (cd "$rd" && hailortcli run2 -t "$RUN_S" -m raw_async measure-fw-actions --output-path fwactions.json \
          set-net "$hef" --batch-size "$B") > "$rd/stdout.log" 2>&1 || rc=$?
      end=$(date -Iseconds)
      t_post=$(hailo_temp)
      read -r nwarn nerr < <(count_hailort_warnings "$rd/hailort.log")
      prev_warn=$nwarn
      fps_out=$(grep -oiE 'fps[^0-9]*[0-9]+(\.[0-9]+)?' "$rd/stdout.log" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?$' || echo nan)
      echo "$m,$rep,$B,raw/$id,$start,$end,$rc,$t_pre,$COOL_WAITED,$soc_pre,$SOC_WAITED,$t_post,$fps_out,$nwarn,$nerr" >> "$FW_LOG"
      echo "    rc $rc  fps(stdout) $fps_out  chip after ${t_post} C  hailort.log warn $nwarn err $nerr"
    done
    for B in "${BENCH_BATCHES[@]}"; do
      id="${m}_bench_b${B}_run${rep}"; rd="$OUT_DIR/raw/$id"; mkdir -p "$rd"
      gate
      start=$(date -Iseconds)
      echo "[$(date +%T)] $id  chip ${t_pre} C (cooled ${COOL_WAITED}s)  soc ${soc_pre} C (waited ${SOC_WAITED}s)"
      rc=0
      (cd "$rd" && hailortcli benchmark --batch-size "$B" -t "$BENCH_S" --csv benchmark.csv "$hef") \
          > "$rd/stdout.log" 2>&1 || rc=$?
      end=$(date -Iseconds)
      t_post=$(hailo_temp)
      read -r nwarn nerr < <(count_hailort_warnings "$rd/hailort.log")
      prev_warn=$nwarn
      # benchmark.csv: net_name,fps,hw_only_fps,num_of_frames,num_of_frames_hw_only,hw_latency,overall_latency,...
      read -r fps hwfps lat < <(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
          NR == 2 { print $c["fps"], $c["hw_only_fps"], $c["hw_latency"] }' "$rd/benchmark.csv" 2>/dev/null \
          || echo "nan nan nan")
      echo "$m,$rep,$B,raw/$id,$start,$end,$rc,$t_pre,$COOL_WAITED,$soc_pre,$SOC_WAITED,$t_post,${fps:-nan},${hwfps:-nan},${lat:-nan},$nwarn,$nerr" >> "$BN_LOG"
      echo "    rc $rc  fps(streaming) ${fps:-nan}  fps(hw_only) ${hwfps:-nan}  hw latency ${lat:-nan} ms  chip after ${t_post} C  hailort.log warn $nwarn err $nerr"
    done
  done
done
echo "done: $FW_LOG $BN_LOG"
