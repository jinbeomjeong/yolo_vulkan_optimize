#!/usr/bin/env bash
# GPU NMS (efficient_nms.comp, --nms gpu) vs CPU suppression (--nms cpu) — pipeline A/B + accuracy check.
# Experiment A of 07_results/display_vulkan_hdmi/notes/formal_measurement_plan.md §4 (A1 perf, A2 accuracy).
#
# Why a separate script:
#   display_bench.sh / dec_threads_bench.sh have no --nms axis, and several A1 metrics exist only in the
#   yvo_pipeline stdout summary (gpu nms median/max, fallback by cause, slot hold, cpu nms median/p90).
#   Cool-down, environment snapshot, loop structure and the RESULT_CSV + RESULT_CSV2 column names follow
#   dec_threads_bench.sh (the 55 app columns are display_bench.sh's names).
#   The verdict is "no regression + GPU cost", not "CPU saving": the CPU suppression that disappears is 3~11 µs/frame
#   (07_results/gpu_postprocess/summary.md), below the run-to-run noise of getrusage CPU (~0.1 ms).
#
# Modes (MODE=...):
#   perf   (default) A1. Display off, --mode dmabuf --layout nhcw --postproc gpu --nms {cpu|gpu},
#          b2s4 (--batch 2 --slots 4) and b8s16 (--batch 8 --slots 16) → b2s4_cpu b2s4_gpu b8s16_cpu b8s16_gpu.
#          FRAMES (900) x RUNS (3), interleaved: repetition outer, config inner.
#   verify A2. One full-video run each (VERIFY_FRAMES, default all = 1816 frames) at b2s4 of
#          verify (--nms verify: GPU and CPU both run, nms_verify_mismatch counts bitwise differences),
#          cpu and gpu, all with --dump-dir; then compare_detections.py cpu vs gpu and a sorted-line diff.
#          **Not a performance run** — nothing goes into results*.csv (verify runs both suppressions).
#          [A2] PASS needs all of: nms_verify_mismatch 0, cpu == gpu detection count, detections.jsonl cpu/gpu and
#          cpu/verify identical (sorted lines), all three runs present in nms_verify.csv with exit 0 and ok 1, and
#          (with BASELINE=) all three detections.jsonl identical to the baseline's.
#          ⚠ verify alone cannot catch decoder-side regressions — use BASELINE= for that. The three runs suppress the
#          SAME yolo_decode.comp candidates, so a broken decoder makes them agree and the mismatch stays 0.
#   Every flag that matters is passed explicitly, although the CLI defaults are the same since 72f88a7
#   (--mode dmabuf --layout nhcw --postproc gpu --nms gpu --batch 8): defaults can move, archived commands must not.
#
# Usage:
#   bash 06_tools/bench/nms_bench.sh                              # A1, 3 runs x 4 configs x 900 frames
#   MODE=verify bash 06_tools/bench/nms_bench.sh                  # A2 into the same OUT_DIR (separate files)
#   ONLY='^b2s4_' RUNS=1 FRAMES=300 bash ...                      # subset (grep -E on config id)
#   DRYRUN=1 bash ... / DRYRUN=1 MODE=verify bash ...             # print every command and exit (no lock, writes nothing)
#   CHECK_ONLY=1 bash ...                                         # binary/tool check only (lib/common.sh)
#   BIN=/path/to/yvo_pipeline bash ...                            # pin the binary
#   APPEND=1 FORCE=1 bash ...                                     # perf: append to an existing results_raw.csv
#   COOL_C=65 COOL_MAX_S=120 HOT_C=75 COOL_EVERY=1 bash ...       # cool-down (see below)
#   VERIFY_FRAMES=0 DUMP_EVERY=100000 bash ...                    # verify: 0 = full video; PNG every K frames
#   MODE=verify VIDEO=.../mot20_05_h265.mp4 BASELINE=07_results/nms_crowd_stress/runs/mot20_05 bash ...
#                                                                 # verify + byte-compare against an archived verify
#   MODE=verify VERDICT_ONLY=1 OUT_DIR=<verify folder> [BASELINE=...] bash ...  # re-judge existing samples/ only
#
# BASELINE=<archived MODE=verify folder> (verify only; the folder is only read):
#   Before the lock and any NPU use, checks that $BASELINE/samples/{cpu,gpu,verify}_b2s4/detections.jsonl exist and
#   that $BASELINE/env/env_verify.txt names the same video and hef (realpath) and the same effective frame count
#   (VERIFY_FRAMES, 0 = the container frame count from ffprobe). Any mismatch exits 1 without running.
#   After the runs, each new detections.jsonl is compared with the baseline's (sorted lines, byte-exact); a pair that
#   differs fails the verdict and gets evidence/compare_baseline_<id>.txt (compare_detections.py baseline vs new).
# VERDICT_ONLY=1 (verify only): no runs, no lock, no NPU. Re-judges OUT_DIR's samples/ + evidence/nms_verify.csv
#   (the run's own values come from OUT_DIR/env/env_verify.txt) and writes the evidence files to VERDICT_OUT
#   (default build/out/nms_bench/verdict_only). A VERDICT_OUT inside 07_results/ needs FORCE=1.
#
# Cool-down: COOL_EVERY=1 (default) cools the Hailo chip to <= COOL_C before every run, because the A/B deltas looked
#   for are smaller than the thermal drift between runs. COOL_EVERY=0 keeps only the measure-skill rule (chip > HOT_C,
#   or the previous run's hailort.log had temperature/clock warnings). The previous-run-warning cool-down applies in both.
# Reliability: a run whose hailort.log has temperature/throttle/clock warnings is marked unreliable=1 (measure skill §4).
# Decoder threads: the caller's YVO_DEC_THREADS is removed for every child (env -u), so runs use the app default (1).
#
# Output (default OUT_DIR = 07_results/gpu_nms_e2e). The two modes write disjoint paths, so either order works:
#   perf:   env_bench.txt, results_raw.csv, results.csv, raw/<id>_run<N>/{stdout.log,hailort.log,gpu_stats_{pre,post}.txt},
#           raw/_session/hailort.log (temperature probe)
#   verify: env/env_verify.txt, samples/{verify,cpu,gpu}_b2s4/{stdout.log,hailort.log,detections.jsonl,frame*.png},
#           evidence/nms_verify.csv, evidence/compare_cpu_vs_gpu.{txt,csv}, evidence/nms_verify_summary.txt,
#           evidence/compare_baseline_<id>.txt (BASELINE= only, for pairs that differ)
# results_raw.csv = 10 script columns + RESULT_CSV (22) + RESULT_CSV2 after label without its duplicate ok (33)
#                   + 12 stdout-only / V3D columns (STDOUT_COLS below) = 77 columns.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, package versions, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/gpu_nms_e2e}"
MODE="${MODE:-perf}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-900}"
VERIFY_FRAMES="${VERIFY_FRAMES:-0}"    # verify mode: 0 = whole video (1816 frames)
DUMP_EVERY="${DUMP_EVERY:-100000}"     # verify mode: --dump-every (frame 0 still gets one PNG)
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
COOL_EVERY="${COOL_EVERY:-1}"
APPEND="${APPEND:-0}"
DRYRUN="${DRYRUN:-0}"
BASELINE="${BASELINE:-}"               # verify mode: archived verify folder to byte-compare against
VERDICT_ONLY="${VERDICT_ONLY:-0}"      # verify mode: re-judge OUT_DIR's existing samples/ (no runs)
VERDICT_OUT="${VERDICT_OUT:-$BUILD_DIR/out/nms_bench/verdict_only}"
COMPARE_PY="$PROJ/06_tools/python/compare_detections.py"
GPU_STATS="/sys/class/drm/renderD128/device/gpu_stats"

[[ "$MODE" == "perf" || "$MODE" == "verify" ]] || { echo "MODE must be perf | verify: $MODE" >&2; exit 2; }
[[ "$FRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "FRAMES must be a positive integer: $FRAMES" >&2; exit 2; }
[[ "$VERIFY_FRAMES" =~ ^[0-9]+$ ]] || { echo "VERIFY_FRAMES must be >= 0: $VERIFY_FRAMES" >&2; exit 2; }
if [[ "$MODE" != "verify" ]] && [[ -n "$BASELINE" || "$VERDICT_ONLY" == "1" ]]; then
  echo "BASELINE= and VERDICT_ONLY=1 apply to MODE=verify only" >&2; exit 2
fi
require_bin "$BIN" yvo_pipeline
[[ -f "$COMPARE_PY" ]] || { echo "compare_detections.py not found: $COMPARE_PY" >&2; exit 1; }
check_only_exit

CALLER_DEC_THREADS="${YVO_DEC_THREADS-<unset>}"

# Shared operating-point arguments (everything explicit; display off).
BASE_ARGS="--mode dmabuf --layout nhcw --postproc gpu --postproc-alloc export --gpu-depth 3 --display off"
# id|args (perf mode)
configs=(
  "b2s4_cpu  |$BASE_ARGS --batch 2 --slots 4 --nms cpu"
  "b2s4_gpu  |$BASE_ARGS --batch 2 --slots 4 --nms gpu"
  "b8s16_cpu |$BASE_ARGS --batch 8 --slots 16 --nms cpu"
  "b8s16_gpu |$BASE_ARGS --batch 8 --slots 16 --nms gpu"
)
# id|args (verify mode). Same b2s4 operating point; the ids are the samples/ folder names.
vconfigs=(
  "verify_b2s4 |$BASE_ARGS --batch 2 --slots 4 --nms verify"
  "cpu_b2s4    |$BASE_ARGS --batch 2 --slots 4 --nms cpu"
  "gpu_b2s4    |$BASE_ARGS --batch 2 --slots 4 --nms gpu"
)

selected=()
if [[ "$MODE" == "perf" ]]; then
  for c in "${configs[@]}"; do
    IFS='|' read -r id args <<<"$c"; id="${id// /}"
    grep -Eq "$ONLY" <<<"$id" && selected+=("$id|$args")
  done
  (( ${#selected[@]} > 0 )) || { echo "ONLY='$ONLY' matches no config id" >&2; exit 1; }
else
  for c in "${vconfigs[@]}"; do IFS='|' read -r id args <<<"$c"; selected+=("${id// /}|$args"); done
fi

# build_cmd <id> <args> <frames|0> [dump_dir] -> fills cmd[]
build_cmd() {
  local id="$1" args="$2" frames="$3" dump="${4:-}" argv=()
  read -ra argv <<<"$args"
  cmd=(env -u YVO_DEC_THREADS "$BIN" --video "$VIDEO" --hef "$HEF" "${argv[@]}")
  (( frames > 0 )) && cmd+=(--frames "$frames")
  [[ -n "$dump" ]] && cmd+=(--dump-dir "$dump" --dump-every "$DUMP_EVERY")
  cmd+=(--quiet --csv-label "$id")
}

run_dir_of() {   # run_dir_of <id> <run>
  if [[ "$MODE" == "perf" ]]; then echo "$OUT_DIR/raw/${1}_run${2}"; else echo "$OUT_DIR/samples/$1"; fi
}

# ── verify: baseline pre-check and verdict ─────────────────────────────────────────
VIDS=(cpu_b2s4 gpu_b2s4 verify_b2s4)
env_field() {   # env_field <env_verify.txt> <key> -> first "<key>: value" (empty if absent)
  sed -n "s/^$2: //p" "$1" 2>/dev/null | head -1
}
effective_frames() {   # effective_frames <VERIFY_FRAMES> <video> -> frames the run decodes (literal if unknown)
  local n="$1" nb
  nb=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nw=1:nk=1 "$2" 2>/dev/null \
       | head -1 || true)
  if [[ "$nb" =~ ^[1-9][0-9]*$ ]] && (( n == 0 || n > nb )); then echo "$nb"; else echo "$n"; fi
}
# baseline_precheck: prints [baseline] lines; returns 1 if the baseline cannot be compared with this run.
#   The run's own video/hef/frames are the current knobs, or OUT_DIR/env/env_verify.txt under VERDICT_ONLY=1.
baseline_precheck() {
  local benv="$BASELINE/env/env_verify.txt" bad=() id f bv bh bf rv rh rf src
  echo "[baseline] $BASELINE"
  [[ -d "$BASELINE" ]] || { echo "[baseline] FAIL: not a directory"; return 1; }
  for id in "${VIDS[@]}"; do
    f="$BASELINE/samples/$id/detections.jsonl"
    [[ -s "$f" ]] || bad+=("missing or empty $f")
  done
  if [[ -f "$benv" ]]; then
    bv=$(env_field "$benv" video); bh=$(env_field "$benv" hef)
    bf=$(env_field "$benv" "verify frames"); bf="${bf%% *}"
  else
    bad+=("missing $benv")
  fi
  if [[ "$VERDICT_ONLY" == "1" ]]; then
    src="$OUT_DIR/env/env_verify.txt"
    rv=$(env_field "$src" video); rh=$(env_field "$src" hef)
    rf=$(env_field "$src" "verify frames"); rf="${rf%% *}"
  else
    src="current VIDEO/HEF/VERIFY_FRAMES"; rv="$VIDEO"; rh="$HEF"; rf="$VERIFY_FRAMES"
  fi
  if [[ -f "$benv" ]]; then
    [[ -n "$bv" && -n "$bh" && "$bf" =~ ^[0-9]+$ ]] || bad+=("$benv lacks video:/hef:/verify frames: lines")
    [[ -n "$rv" && -n "$rh" && "$rf" =~ ^[0-9]+$ ]] || bad+=("$src lacks video/hef/verify frames")
    echo "[baseline] video  baseline $bv | run $rv"
    echo "[baseline] hef    baseline $bh | run $rh"
    if [[ -n "$bv" && -n "$rv" ]]; then
      [[ "$(realpath -m "$bv")" == "$(realpath -m "$rv")" ]] || bad+=("video differs")
    fi
    if [[ -n "$bh" && -n "$rh" ]]; then
      [[ "$(realpath -m "$bh")" == "$(realpath -m "$rh")" ]] || bad+=("hef differs")
    fi
    if [[ "$bf" =~ ^[0-9]+$ && "$rf" =~ ^[0-9]+$ ]]; then
      local ebf erf
      ebf=$(effective_frames "$bf" "$rv"); erf=$(effective_frames "$rf" "$rv")
      echo "[baseline] frames baseline VERIFY_FRAMES=$bf (effective $ebf) | run VERIFY_FRAMES=$rf (effective $erf)"
      [[ "$ebf" == "$erf" ]] || bad+=("effective frame count differs ($ebf vs $erf)")
    fi
  fi
  if (( ${#bad[@]} )); then
    local b; for b in "${bad[@]}"; do echo "[baseline] FAIL: $b"; done
    return 1
  fi
  echo "[baseline] OK: comparable (video, hef, frames match; 3 detections.jsonl present)"
}

# Line order follows NPU completion, so compare sorted lines (one line per frame, keyed by "frame").
same_sorted() { [[ -s "$1" && -s "$2" ]] || { echo missing; return; }
                if diff -q <(sort "$1") <(sort "$2") >/dev/null; then echo identical; else echo differ; fi; }

# verify_verdict <evidence dir>: judges OUT_DIR/samples + OUT_DIR/evidence/nms_verify.csv, writes
#   compare_cpu_vs_gpu.{txt,csv}, compare_baseline_<id>.txt and nms_verify_summary.txt into <evidence dir>,
#   prints the summary and returns 0 only on "[A2] PASS".
verify_verdict() {
  local evid="$1" cj gj vj cmp_status id bl_res=() r
  mkdir -p "$evid"
  rm -f "$evid"/compare_baseline_*.txt
  cj="$OUT_DIR/samples/cpu_b2s4/detections.jsonl"; gj="$OUT_DIR/samples/gpu_b2s4/detections.jsonl"
  vj="$OUT_DIR/samples/verify_b2s4/detections.jsonl"
  cmp_status=missing
  if [[ -s "$cj" && -s "$gj" ]]; then
    "$PY_HAILO" "$COMPARE_PY" "$cj" "$gj" --csv "$evid/compare_cpu_vs_gpu.csv" --label cpu_vs_gpu \
      > "$evid/compare_cpu_vs_gpu.txt" 2>&1 && cmp_status=ok || cmp_status=failed
  fi
  if [[ -n "$BASELINE" ]]; then
    for id in "${VIDS[@]}"; do
      r=$(same_sorted "$BASELINE/samples/$id/detections.jsonl" "$OUT_DIR/samples/$id/detections.jsonl")
      bl_res+=("$r")
      if [[ "$r" == "differ" ]]; then
        "$PY_HAILO" "$COMPARE_PY" "$BASELINE/samples/$id/detections.jsonl" "$OUT_DIR/samples/$id/detections.jsonl" \
          --label "baseline_vs_${id}" > "$evid/compare_baseline_${id}.txt" 2>&1 || true
      fi
    done
  else
    bl_res=(- - -)
  fi
  # Verdict line from the CSV (python, NaN-safe).
  /usr/bin/python3 - "$OUT_DIR/evidence/nms_verify.csv" "$(same_sorted "$cj" "$gj")" "$(same_sorted "$cj" "$vj")" \
      "$cmp_status" "$evid/compare_cpu_vs_gpu.txt" "$OUT_DIR/samples" "$BASELINE" "${bl_res[@]}" "$evid" \
      > "$evid/nms_verify_summary.txt" <<'PYEOF' || true   # a crash leaves no PASS line -> FAIL
import csv, json, os, sys
try:
    rows = {r["id"]: r for r in csv.DictReader(open(sys.argv[1], newline=""))}
except OSError as e:
    print(f"cannot read nms_verify.csv: {e}")
    rows = {}
jsonl_cg, jsonl_cv, cmp_status, cmp_txt, samples, baseline = sys.argv[2:8]
bl = dict(zip(("cpu_b2s4", "gpu_b2s4", "verify_b2s4"), sys.argv[8:11]))
evid = sys.argv[11]
IDS = ("verify_b2s4", "cpu_b2s4", "gpu_b2s4")
def g(i, k): return rows.get(i, {}).get(k, "nan")
def excerpt(path):
    try:
        for line in open(path):
            if line.startswith(("[warn]", "detections", "match @ IoU", "frames")):
                print("  " + line.rstrip())
    except OSError:
        pass
print("# nms_bench MODE=verify (A2) — not a performance result")
print(f"samples: {samples}")
for i in IDS:
    print(f"{i:12s} exit {g(i, 'exit_code')} ok {g(i, 'ok')} frames {g(i, 'frames')} detections {g(i, 'detections')} "
          f"nms {g(i, 'nms')} verify_mismatch {g(i, 'nms_verify_mismatch')} fallback {g(i, 'nms_fallback')} "
          f"(capacity {g(i, 'nms_fb_capacity')}, iou guard {g(i, 'nms_fb_guard')}) hailort_warn {g(i, 'hailort_warnings')}")
print(f"detections.jsonl cpu vs gpu (sorted lines): {jsonl_cg}")
print(f"detections.jsonl cpu vs verify (sorted lines, both CPU suppression — run-to-run determinism): {jsonl_cv}")
print(f"compare_detections.py cpu vs gpu: {cmp_status}")
excerpt(cmp_txt)
if baseline:
    print(f"baseline: {baseline}")
    for i in ("cpu_b2s4", "gpu_b2s4", "verify_b2s4"):
        print(f"detections.jsonl baseline vs new {i} (sorted lines): {bl[i]}")
        if bl[i] == "differ":
            print(f"  compare_detections.py baseline vs new -> {os.path.join(evid, f'compare_baseline_{i}.txt')}")
            excerpt(os.path.join(evid, f"compare_baseline_{i}.txt"))
            try:   # frame-level diff: compare_detections.py is IoU-based and does not show score-only changes
                fr = [{json.loads(l)["frame"]: l.strip() for l in open(f) if l.strip()}
                      for f in (os.path.join(baseline, "samples", i, "detections.jsonl"),
                                os.path.join(samples, i, "detections.jsonl"))]
                dif = sorted(k for k in set(fr[0]) | set(fr[1]) if fr[0].get(k) != fr[1].get(k))
                print(f"  frames whose line differs: {len(dif)} (first: {dif[:5]})")
            except (OSError, ValueError, KeyError) as e:
                print(f"  frame-level diff unavailable: {e}")
else:
    print("baseline: none (decoder-side regressions are NOT checked — pass BASELINE= for that)")
mm = g("verify_b2s4", "nms_verify_mismatch")
det_eq = g("cpu_b2s4", "detections") == g("gpu_b2s4", "detections") and g("cpu_b2s4", "detections") != "nan"
missing = [i for i in IDS if i not in rows]
ok_all = not missing and all(g(i, "exit_code") == "0" and g(i, "ok") == "1" for i in IDS)
jsonl_ok = jsonl_cg == "identical" and jsonl_cv == "identical"
bl_ok = (not baseline) or all(v == "identical" for v in bl.values())
verdict = "PASS" if (mm == "0" and det_eq and ok_all and jsonl_ok and bl_ok) else "FAIL"
bl_txt = ("none" if not baseline else
          "cpu/gpu/verify " + "/".join(bl[i] for i in ("cpu_b2s4", "gpu_b2s4", "verify_b2s4")))
print(f"[A2] {verdict}: nms_verify_mismatch {mm} (want 0), detections cpu == gpu {det_eq}, "
      f"jsonl cpu/gpu {jsonl_cg}, jsonl cpu/verify {jsonl_cv}, "
      f"all 3 runs ok {ok_all}{' (missing rows: ' + ','.join(missing) + ')' if missing else ''}, "
      f"baseline {bl_txt}")
PYEOF
  cat "$evid/nms_verify_summary.txt"
  grep -q '^\[A2\] PASS' "$evid/nms_verify_summary.txt"
}

if [[ -n "$BASELINE" ]]; then
  baseline_precheck || { echo "[A2] FAIL: baseline not comparable — nothing was run (no lock, no NPU)" >&2; exit 1; }
fi

if [[ "$VERDICT_ONLY" == "1" ]]; then
  # Re-judge an existing verify folder: no lock, no NPU, nothing written into OUT_DIR.
  [[ -f "$OUT_DIR/evidence/nms_verify.csv" && -d "$OUT_DIR/samples" ]] || {
    echo "VERDICT_ONLY=1 needs $OUT_DIR/evidence/nms_verify.csv and $OUT_DIR/samples/" >&2; exit 1; }
  case "$(realpath -m "$VERDICT_OUT")/" in
    "$(realpath -m "$PROJ/07_results")"/*)
      [[ "${FORCE:-0}" == "1" ]] || { echo "VERDICT_OUT is inside 07_results/ (archived results): needs FORCE=1" >&2; exit 1; } ;;
  esac
  [[ "$(realpath -m "$VERDICT_OUT")" != "$(realpath -m "$OUT_DIR/evidence")" || "${FORCE:-0}" == "1" ]] || {
    echo "VERDICT_OUT would overwrite $OUT_DIR/evidence: needs FORCE=1" >&2; exit 1; }
  echo "VERDICT_ONLY: judging $OUT_DIR/samples -> $VERDICT_OUT"
  verify_verdict "$VERDICT_OUT"
  exit $?
fi

if [[ "$DRYRUN" == "1" ]]; then
  echo "DRYRUN: $(basename "$0") MODE=$MODE OUT_DIR=$OUT_DIR RUNS=$RUNS FRAMES=$FRAMES VERIFY_FRAMES=$VERIFY_FRAMES ONLY='$ONLY' COOL_EVERY=$COOL_EVERY"
  echo "DRYRUN: caller YVO_DEC_THREADS=$CALLER_DEC_THREADS (removed for every child)"
  nruns=$RUNS; [[ "$MODE" == "verify" ]] && nruns=1
  for ((r = 1; r <= nruns; r++)); do
    for c in "${selected[@]}"; do
      IFS='|' read -r id args <<<"$c"
      d=$(run_dir_of "$id" "$r")
      if [[ "$MODE" == "perf" ]]; then build_cmd "$id" "$args" "$FRAMES"; else build_cmd "$id" "$args" "$VERIFY_FRAMES" "$d"; fi
      printf '[run %d/%d %s] cd %q && ' "$r" "$nruns" "$id" "$d"; printf '%q ' "${cmd[@]}"; printf '> stdout.log 2>&1\n'
    done
  done
  if [[ "$MODE" == "verify" ]]; then
    printf '[compare] %q %q %q %q --csv %q --label cpu_vs_gpu\n' "$PY_HAILO" "$COMPARE_PY" \
      "$OUT_DIR/samples/cpu_b2s4/detections.jsonl" "$OUT_DIR/samples/gpu_b2s4/detections.jsonl" \
      "$OUT_DIR/evidence/compare_cpu_vs_gpu.csv"
    for id in "${VIDS[@]}"; do
      if [[ -n "$BASELINE" ]]; then
        printf '[baseline compare] sorted-line diff %q vs %q (differ -> compare_detections.py -> evidence/compare_baseline_%s.txt)\n' \
          "$BASELINE/samples/$id/detections.jsonl" "$OUT_DIR/samples/$id/detections.jsonl" "$id"
      fi
    done
    [[ -n "$BASELINE" ]] || echo "[baseline compare] none (BASELINE unset: decoder-side regressions are not checked)"
    echo "[verdict] mismatch 0 + det cpu==gpu + jsonl cpu/gpu & cpu/verify identical + 3 rows ok${BASELINE:+ + 3 baseline pairs identical}"
  fi
  exit 0
fi

# Archived-result protection per mode (the two modes write disjoint paths into the same OUT_DIR).
guard_paths() {
  local p hit=()
  for p in "$@"; do
    if [[ -d "$p" && -n "$(ls -A "$p" 2>/dev/null)" ]] || [[ -f "$p" ]]; then hit+=("$p"); fi
  done
  if (( ${#hit[@]} )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "MODE=$MODE output already exists: ${hit[*]}" >&2
    echo "  Stopping so archived results are not overwritten. Pass a new OUT_DIR=..., or FORCE=1 to overwrite." >&2
    exit 1
  fi
}
if [[ "$MODE" == "perf" ]]; then
  # raw/_session/ (temperature probe log) is shared by both modes, so only the per-run folders count.
  if [[ "$APPEND" == "1" ]]; then [[ "${FORCE:-0}" == "1" ]] || { echo "APPEND=1 modifies archived results: needs FORCE=1" >&2; exit 1; }
  else guard_paths "$OUT_DIR/env_bench.txt" "$OUT_DIR/results_raw.csv" "$OUT_DIR/results.csv" "$OUT_DIR"/raw/*_run*; fi
else
  guard_paths "$OUT_DIR/samples" "$OUT_DIR/env/env_verify.txt" "$OUT_DIR/evidence/nms_verify.csv" \
              "$OUT_DIR/evidence/compare_cpu_vs_gpu.txt" "$OUT_DIR/evidence/nms_verify_summary.txt"
fi
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

env_snapshot() {
  local dirty=""
  [[ -n "$(git -C "$PROJ" status --porcelain --untracked-files=no 2>/dev/null)" ]] && dirty=" (tracked changes)"
  echo "# nms_bench MODE=$MODE $1 $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  throttled
  vcgencmd measure_clock arm 2>/dev/null || true
  vcgencmd measure_clock v3d 2>/dev/null || true
  vcgencmd get_config v3d_freq 2>/dev/null || true
  echo "bootloader: $(vcgencmd bootloader_version 2>/dev/null | head -1)"
  pcie_link
  pkg_versions "${PKGS_DISPLAY_BENCH[@]}"
  echo "git HEAD: $(git -C "$PROJ" rev-parse HEAD 2>/dev/null || echo '?')${dirty}"
  echo "bin: $BIN md5 $(md5sum "$BIN" | cut -d' ' -f1) mtime $(stat -c %y "$BIN")"
}

if [[ "$MODE" == "perf" ]]; then ENV_FILE="$OUT_DIR/env_bench.txt"; else mkdir -p "$OUT_DIR/env"; ENV_FILE="$OUT_DIR/env/env_verify.txt"; fi
{
  env_snapshot before
  echo "video: $VIDEO"
  echo "hef: $HEF"
  if [[ "$MODE" == "perf" ]]; then echo "runs: $RUNS frames: $FRAMES only: $ONLY"
  else echo "verify frames: $VERIFY_FRAMES (0 = whole video) dump-every: $DUMP_EVERY"
       echo "baseline: ${BASELINE:-none}"; fi
  echo "cool: COOL_C=$COOL_C COOL_MAX_S=$COOL_MAX_S HOT_C=$HOT_C COOL_EVERY=$COOL_EVERY"
  echo "caller YVO_DEC_THREADS: $CALLER_DEC_THREADS (removed for every child: app default)"
  echo "env: $(env | grep -E '^(VK_|HAILO|AV_|FFREPORT|YVO_)' | paste -sd ' ' || true)"
  echo "configs (id|args):"
  for c in "${selected[@]}"; do echo "  $c"; done
} > "$ENV_FILE"
cat "$ENV_FILE"
if env | grep -qE '^(VK_LOADER_LAYERS_ENABLE|VK_INSTANCE_LAYERS)='; then
  echo "WARNING: VK_*LAYERS* set in the environment (recorded in $(basename "$ENV_FILE"))" >&2
fi

# ── stdout-only metrics ─────────────────────────────────────────────────────────────
# Parsed from the yvo_pipeline summary (01_src/pipeline/main.cpp, end of run()). nan when the line is absent
# (the gpu nms / fallback lines exist only for --nms gpu|verify).
#   gpu nms ms      mean M  median M  p90 M  max M  (efficient_nms.comp)       -> gpu_nms_median, gpu_nms_max
#   nms fallback    N frames (capacity X, iou guard Y)                         -> nms_fb_capacity, nms_fb_guard
#   cpu nms us      mean M  median M  p90 M  max M  (cpu_nms_greedy_by_class, n=K) -> cpu_nms_us_median, cpu_nms_us_p90
#   slot hold ms    mean M  median M  p90 M  max M                             -> slot_hold_mean/median/p90
#   e2e latency ms  mean M  median M  p90 M  max M                             -> e2e_ms_median
# gpu_nms_p90 and nms_fallback (N) are already RESULT_CSV2 columns, so they are not repeated.
# V3D columns: CSD (compute) queue delta of gpu_stats over the run / completed frames. **System-wide** (all clients).
STDOUT_COLS="gpu_nms_median,gpu_nms_max,nms_fb_capacity,nms_fb_guard,cpu_nms_us_median,cpu_nms_us_p90,slot_hold_mean,slot_hold_median,slot_hold_p90,e2e_ms_median,v3d_csd_jobs_per_frame,v3d_csd_ms_per_frame"
N_STDOUT=12
stdout_metrics() {   # stdout_metrics <stdout.log> -> 10 comma-separated values
  awk '
    function f(name) { for (i = 1; i < NF; i++) if ($i == name) return $(i + 1); return "nan" }
    /^gpu nms ms /      { gm = f("median"); gx = f("max") }
    /^nms fallback /    { if (match($0, /capacity [0-9]+/)) fc = substr($0, RSTART + 9, RLENGTH - 9)
                          if (match($0, /iou guard [0-9]+/)) fg = substr($0, RSTART + 10, RLENGTH - 10) }
    /^cpu nms us /      { cm = f("median"); cp = f("p90") }
    /^slot hold ms /    { sm = f("mean"); sd = f("median"); sp = f("p90") }
    /^e2e latency ms /  { em = f("median") }
    END {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
        (gm == "" ? "nan" : gm), (gx == "" ? "nan" : gx), (fc == "" ? "nan" : fc), (fg == "" ? "nan" : fg),
        (cm == "" ? "nan" : cm), (cp == "" ? "nan" : cp), (sm == "" ? "nan" : sm), (sd == "" ? "nan" : sd),
        (sp == "" ? "nan" : sp), (em == "" ? "nan" : em)
    }' "$1" 2>/dev/null || printf 'nan,nan,nan,nan,nan,nan,nan,nan,nan,nan'
}
gpu_stats_snap() { cat "$GPU_STATS" 2>/dev/null > "$1" || : > "$1"; }
v3d_csd_delta() {   # v3d_csd_delta <pre> <post> <frames> -> "jobs_per_frame,ms_per_frame"
  awk -v n="$3" '
    FNR == NR && $1 == "csd" { j0 = $3; r0 = $4; next }
    $1 == "csd" { j1 = $3; r1 = $4 }
    END {
      if (j0 == "" || j1 == "" || n + 0 <= 0) { printf "nan,nan"; exit }
      printf "%.3f,%.4f", (j1 - j0) / n, (r1 - r0) / 1e6 / n
    }' "$1" "$2" 2>/dev/null || printf 'nan,nan'
}

SCRIPT_COLS="id,nms_cfg,run,exit_code,cool_wait_s,hailort_warnings,hailort_errors,unreliable,hailo_pre_c,note"
CSV1_COLS="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok"
CSV2_COLS="postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,display,sync,pace,disp_convert_mean,disp_convert_p90,disp_overlay_mean,present_interval_median,e2e_glass_mean,e2e_glass_p90,presented,display_dropped,recreates,held_frames_max,slot_timeout_count,nms,gpu_nms_mean,gpu_nms_p90,nms_fallback,nms_verify_mismatch,scanout_zero_copy,plane_color_ok,hvs_underrun_delta,overlay_skipped,frame_cb_timeouts,zero_copy_feedback"
N_CSV1=22   # fields after "RESULT_CSV,"  (label ... ok)
N_CSV2=35   # fields after "RESULT_CSV2," (label + 10 postproc incl. ok + 14 display + 5 NMS + 6 scanout)
#   ⚠ If main.cpp appends RESULT_CSV2 fields, N_CSV2 and CSV2_COLS must change together.
CSV_HEADER="$SCRIPT_COLS,$CSV1_COLS,$CSV2_COLS,$STDOUT_COLS"

nan_list() { local n="$1" i out=""; for ((i = 0; i < n; i++)); do out+="${out:+,}nan"; done; printf '%s' "$out"; }
arg_val() {   # arg_val <args> <flag> <default>
  local a=() i
  read -ra a <<<"$1"
  for ((i = 0; i + 1 < ${#a[@]}; i++)); do [[ "${a[i]}" == "$2" ]] && { printf '%s' "${a[i+1]}"; return; }; done
  printf '%s' "$3"
}

# make_row <id> <run> <rc> <cool> <nwarn> <nerr> <t_pre> <run_dir> <args> -> one results_raw.csv line
#   A missing or malformed RESULT_CSV / RESULT_CSV2 line becomes nan fields and forces ok=0.
make_row() {
  local id="$1" r="$2" rc="$3" cool="$4" nwarn="$5" nerr="$6" t_pre="$7" dir="$8" args="$9"
  local log="$dir/stdout.log" l1="" l2="" l3 notes=() unreliable=0 nms frames
  nms=$(arg_val "$args" --nms nan)
  (( nwarn > 0 )) && unreliable=1
  if [[ -f "$log" ]]; then
    l1=$(grep '^RESULT_CSV,' "$log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
    l2=$(grep '^RESULT_CSV2,' "$log" | tail -1 | sed 's/^RESULT_CSV2,//' || true)
  fi
  if [[ -z "$l1" || "$(awk -F, '{print NF}' <<<"$l1")" != "$N_CSV1" ]]; then
    notes+=(no_result_csv)
    l1="$id,$(arg_val "$args" --mode nan),$(arg_val "$args" --layout nan),$(arg_val "$args" --batch nan),$(arg_val "$args" --slots nan),$(nan_list 16),0"
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
  if [[ -f "$log" ]]; then l3=$(stdout_metrics "$log"); else l3=$(nan_list 10); fi
  frames=$(cut -d, -f6 <<<"$l1")
  l3="$l3,$(v3d_csd_delta "$dir/gpu_stats_pre.txt" "$dir/gpu_stats_post.txt" "$frames")"
  (( rc != 0 )) && notes+=("rc_$rc")
  local note="-"; (( ${#notes[@]} )) && note=$(IFS=';'; echo "${notes[*]}")
  echo "$id,$nms,$r,$rc,$cool,$nwarn,$nerr,$unreliable,$t_pre,$note,$l1,$l2,$l3"
}

prev_warn=0
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

# run_one <id> <args> <run> <frames> <dump_dir|""> -> sets rc, nwarn, nerr, run_dir
run_one() {
  local id="$1" args="$2" r="$3" frames="$4" dump="$5"
  run_dir=$(run_dir_of "$id" "$r")
  cool_before_run
  rm -rf "$run_dir"; mkdir -p "$run_dir"
  build_cmd "$id" "$args" "$frames" "$dump"
  echo "[$(date +%T)] $MODE run $r $id (hailo ${t_pre} C, cooled ${COOL_WAITED}s)"
  gpu_stats_snap "$run_dir/gpu_stats_pre.txt"
  set +e
  (cd "$run_dir" && "${cmd[@]}" > stdout.log 2>&1)
  rc=$?
  set -e
  gpu_stats_snap "$run_dir/gpu_stats_post.txt"
  read -r nwarn nerr < <(count_hailort_warnings "$run_dir/hailort.log")
  prev_warn=$nwarn
}

# ════════════════════════════════════════════════════════════════════ MODE=verify (A2)
if [[ "$MODE" == "verify" ]]; then
  mkdir -p "$OUT_DIR/samples" "$OUT_DIR/evidence"
  VCSV="$OUT_DIR/evidence/nms_verify.csv"
  echo "$CSV_HEADER" > "$VCSV"
  for c in "${selected[@]}"; do
    IFS='|' read -r id args <<<"$c"
    d=$(run_dir_of "$id" 1)
    run_one "$id" "$args" 1 "$VERIFY_FRAMES" "$d"
    make_row "$id" 1 "$rc" "$COOL_WAITED" "$nwarn" "$nerr" "$t_pre" "$run_dir" "$args" >> "$VCSV"
    echo "  $(grep -E '^frames|^nms verify|^nms fallback|^detections' "$run_dir/stdout.log" | tr -s ' ' | tr '\n' ' ') hailo_warn=$nwarn rc=$rc"
  done
  { env_snapshot after; echo "bin md5 (after): $(md5sum "$BIN" | cut -d' ' -f1)"; } >> "$ENV_FILE"
  verify_verdict "$OUT_DIR/evidence"
  exit $?
fi

# ════════════════════════════════════════════════════════════════════ MODE=perf (A1)
RAW="$OUT_DIR/results_raw.csv"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1 but the existing $RAW header differs" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

for ((r = 1; r <= RUNS; r++)); do
  for c in "${selected[@]}"; do
    IFS='|' read -r id args <<<"$c"
    run_one "$id" "$args" "$r" "$FRAMES" ""
    row=$(make_row "$id" "$r" "$rc" "$COOL_WAITED" "$nwarn" "$nerr" "$t_pre" "$run_dir" "$args")
    echo "$row" >> "$RAW"
    note=$(cut -d, -f10 <<<"$row")
    if [[ "$note" != "-" ]]; then echo "  rc=$rc note=$note (see stdout.log)"; tail -3 "$run_dir/stdout.log" || true; fi
    echo "  $(grep -E '^wall|^cpu |^gpu nms ms|^nms fallback' "$run_dir/stdout.log" | tr -s ' ' | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc"
  done
done

# Per-config aggregation (NaN-safe medians) + console A/B table.
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
def ival(x): v = fnum(x); return 0 if math.isnan(v) else int(v)
def fmt(v, d=3): return "nan" if math.isnan(v) else f"{v:.{d}f}"

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
if not rows:
    print("no rows"); sys.exit(0)
order, groups = [], {}
for r in rows:
    if r["id"] not in groups:
        order.append(r["id"]); groups[r["id"]] = []
    groups[r["id"]].append(r)

# metric -> printed decimals
METRICS = [("fps", 2), ("cpu_ms_per_frame", 3), ("e2e_ms_mean", 2), ("e2e_ms_median", 2), ("e2e_ms_p90", 2),
           ("npu_ms_mean", 2), ("gpu_ms_mean", 3), ("gpu_post_mean", 4), ("slot_hold_mean", 3), ("slot_hold_p90", 3),
           ("cpu_nms_us_mean", 3), ("cpu_nms_us_median", 3), ("gpu_nms_mean", 4), ("gpu_nms_median", 4),
           ("gpu_nms_p90", 4), ("gpu_nms_max", 4), ("nms_fallback", 0), ("nms_fb_capacity", 0), ("nms_fb_guard", 0),
           ("cand_mean", 2), ("cand_max", 0), ("detections", 0), ("held_frames_max", 0),
           ("v3d_csd_jobs_per_frame", 3), ("v3d_csd_ms_per_frame", 4)]
summary = {}
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")             # no CRLF (measure skill §5)
    hdr = ["id", "nms", "batch", "slots", "frames_median", "runs", "runs_ok", "exit_codes", "unreliable_runs",
           "hailort_warnings", "hailort_errors", "cool_wait_s_total", "hailo_pre_c_max", "notes"]
    for k, _ in METRICS:
        hdr += [f"{k}_median", f"{k}_min", f"{k}_max"]
    hdr += ["all_ok"]
    w.writerow(hdr)
    for cid in order:
        g = groups[cid]
        ok = [x for x in g if x["exit_code"] == "0" and x["ok"] == "1" and x["hailort_warnings"] == "0"
              and x["display"] == "off"]
        stats = {k: (nmed([fnum(x[k]) for x in g]), nmin([fnum(x[k]) for x in g]), nmax([fnum(x[k]) for x in g]))
                 for k, _ in METRICS}
        notes = sorted({n for x in g for n in x["note"].split(";") if n not in ("-", "")})
        all_ok = int(len(ok) == len(g))
        summary[cid] = {"stats": stats, "all_ok": all_ok, "runs_ok": len(ok), "runs": len(g)}
        row = [cid, g[0]["nms_cfg"], g[0]["batch"], g[0]["slots"], fmt(nmed([fnum(x["frames"]) for x in g]), 0),
               len(g), len(ok), ";".join(x["exit_code"] for x in g), sum(x["unreliable"] == "1" for x in g),
               sum(ival(x["hailort_warnings"]) for x in g), sum(ival(x["hailort_errors"]) for x in g),
               sum(ival(x["cool_wait_s"]) for x in g), fmt(nmax([fnum(x["hailo_pre_c"]) for x in g]), 1),
               ";".join(notes) or "-"]
        for k, d in METRICS:
            row += [fmt(v, d) for v in stats[k]]
        row += [all_ok]
        w.writerow(row)

def med(c, k): return summary[c]["stats"][k][0]
def rng(c, k, d): s = summary[c]["stats"][k]; return f"{fmt(s[0], d)} ({fmt(s[1], d)}..{fmt(s[2], d)})"

def table(title, hdr, lines):
    print(title)
    wid = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)] if lines else [len(h) for h in hdr]
    print(" | ".join(h.ljust(x) for h, x in zip(hdr, wid)))
    print("-+-".join("-" * x for x in wid))
    for l in lines:
        print(" | ".join(c.ljust(x) for c, x in zip(l, wid)))
    print()

lines = []
for c in order:
    s = summary[c]
    lines.append([c, rng(c, "fps", 2), rng(c, "cpu_ms_per_frame", 3), fmt(med(c, "e2e_ms_mean"), 2),
                  fmt(med(c, "e2e_ms_p90"), 2), fmt(med(c, "slot_hold_mean"), 3), fmt(med(c, "cpu_nms_us_mean"), 2),
                  fmt(med(c, "gpu_nms_median"), 4), fmt(med(c, "gpu_nms_p90"), 4), fmt(med(c, "nms_fallback"), 0),
                  fmt(med(c, "detections"), 0), f"{s['runs_ok']}/{s['runs']}"])
table("== per config (medians; fps / cpu as median (min..max)) ==",
      ["id", "fps", "cpu_ms/f", "e2e_mean", "e2e_p90", "hold_mean", "cpu_nms_us", "gpu_nms_med", "gpu_nms_p90",
       "fallback", "dets", "ok"], lines)

# A/B: <op>_cpu vs <op>_gpu, delta = gpu - cpu. Range overlap = no-regression hint (fps/cpu/e2e).
AB = [("fps", 2), ("cpu_ms_per_frame", 3), ("e2e_ms_mean", 2), ("e2e_ms_p90", 2), ("slot_hold_mean", 3)]
lines = []
for c in order:
    if not c.endswith("_cpu"):
        continue
    op = c[: -len("_cpu")]
    gcfg = op + "_gpu"
    if gcfg not in summary:
        continue
    line = [op]
    for k, d in AB:
        a, b = summary[c]["stats"][k], summary[gcfg]["stats"][k]
        overlap = not (math.isnan(a[1]) or math.isnan(b[1])) and a[1] <= b[2] and b[1] <= a[2]
        line.append(f"{fmt(a[0], d)}->{fmt(b[0], d)} ({'+' if b[0] - a[0] >= 0 else ''}{fmt(b[0] - a[0], d)}"
                    f"{', overlap' if overlap else ', NO overlap'})")
    dc, dg = med(c, "detections"), med(gcfg, "detections")
    line.append("equal" if dc == dg else f"DIFFER {fmt(dc, 0)} vs {fmt(dg, 0)}")
    line.append(f"{summary[c]['all_ok']}/{summary[gcfg]['all_ok']}")
    lines.append(line)
if lines:
    table("== A/B (gpu - cpu, medians; overlap = min..max ranges of the runs overlap) ==",
          ["op"] + [k for k, _ in AB] + ["detections", "all_ok c/g"], lines)

bad = [c for c in order if not summary[c]["all_ok"]]
print("all_ok=0: " + (", ".join(bad) if bad else "none"))
PYEOF

{
  env_snapshot after
  echo "bin md5 (after measurement): $(md5sum "$BIN" | cut -d' ' -f1)"
} >> "$ENV_FILE"
