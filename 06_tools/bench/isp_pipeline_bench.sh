#!/usr/bin/env bash
# 전체 파이프라인(NPU 포함) 측정: 현행 GPU 셰이더 경로 vs ISP 경로(A안 NHWC / C안 NHCW)
#   b8 운용점만 잰다(batch 8, slots 16). b1/b4는 운용점이 아니라 뺐다.
#
# 사용법:
#   bash 06_tools/bench/isp_pipeline_bench.sh                 # 전체(1816프레임), 설정당 3회
#   RUNS=1 FRAMES=300 bash 06_tools/bench/isp_pipeline_bench.sh
#   ONLY='isp_' bash ...            # 설정 id(grep -E 패턴) 일부만
#   DUMPS=0 bash ...                # 검출 JSONL 덤프 실행 생략
#
# 실행 순서: 반복(run) 바깥 루프, 설정 안쪽 루프(열 상태 편향을 설정 간에 고르게 나눈다).
# 냉각(measure 스킬 §2): 직전 실행의 hailort.log에 온도·스로틀 경고가 있거나 실행 직전 칩 온도가
#   HOT_C 초과면 COOL_C 이하가 될 때까지(최대 COOL_MAX_S초) 대기한다.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 공통 함수(온도·PCIe·냉각·hailort 경고 판정·벤치 락)는 lib/common.sh 하나로 모았다.
# 예전에는 이 함수들이 7개 스크립트에 복붙돼 있었고 실제로 갈라져 있었다(pcie_link 의 Gen5 분기 등).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
bench_lock_acquire
GPU_BIN="${GPU_BIN:-$PROJ/build/01_src/pipeline/yvo_pipeline}"
ISP_BIN="${ISP_BIN:-$PROJ/build/01_src/isp/yvo_isp_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/isp_resize/pipeline}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
DUMPS="${DUMPS:-1}"
APPEND="${APPEND:-0}"
PY_HAILO=/home/pi/miniconda3/envs/hailo-yolo/bin/python

[[ -x "$GPU_BIN" ]] || { echo "yvo_pipeline 없음: $GPU_BIN" >&2; exit 1; }
[[ -x "$ISP_BIN" ]] || { echo "yvo_isp_pipeline 없음: $ISP_BIN" >&2; exit 1; }
mkdir -p "$OUT_DIR/raw"

# id|tool|args
configs=(
  "gpu_nhcw_b8|gpu|--mode dmabuf --layout nhcw --batch 8 --slots 16"
  "isp_A_nhwc_b8|isp|--mode dmabuf --layout nhwc --batch 8 --slots 16"
  "isp_C_nhcw_b8|isp|--mode dmabuf --layout nhcw --repack cpu --batch 8 --slots 16"
)


# 측정 창 오염 확인: 01_src/·02_shaders/ 파일 mtime 스냅샷(다른 세션이 고치는지)
src_snapshot() { find "$PROJ/01_src" "$PROJ/02_shaders" -type f -printf '%T@ %p\n' | sort | md5sum | cut -d' ' -f1; }
src_newest() { find "$PROJ/01_src" "$PROJ/02_shaders" -type f -printf '%TT %p\n' | sort | tail -1; }

SRC_BEFORE=$(src_snapshot)
{
  echo "# isp_pipeline_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "model: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)  (0x80000 = 과거 기록 비트)"
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  echo "hef: $HEF"
  echo "libpisp: $(pkg-config --modversion libpisp 2>/dev/null || echo NA)"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
  echo "src tree md5 (before): $SRC_BEFORE"
  echo "src newest (before): $(src_newest)"
} > "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
CSV_HEADER="label,tool,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,pre_ms_mean,pre_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,run,cool_wait_s,hailort_warnings,hailort_errors"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1인데 기존 헤더가 다르다" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")
prev_warn=0

for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id tool extra <<<"$c"
    echo "$id" | grep -Eq "$ONLY" || continue
    COOL_WAITED=0
    t_now=$(hailo_temp)
    if (( prev_warn > 0 )); then
      cool_down "hailort.log 경고 ${prev_warn}건"
    elif awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then
      cool_down "칩 온도 ${t_now} C > ${HOT_C} C"
    fi
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    BIN=$([[ "$tool" == "gpu" ]] && echo "$GPU_BIN" || echo "$ISP_BIN")
    echo "[$(date +%T)] run $r/$RUNS $id (hailo $(hailo_temp) C)"
    set +e
    # shellcheck disable=SC2086
    (cd "$run_dir" && "$BIN" --video "$VIDEO" --hef "$HEF" $extra "${frames_arg[@]}" \
        --quiet --csv-label "$id" > stdout.log 2>&1)
    rc=$?
    set -e
    nwarn=0; nerr=0
    if [[ -f "$run_dir/hailort.log" ]]; then
      nwarn=$(grep -E '\[warning\]' "$run_dir/hailort.log" | grep -cE "$WARN_RE" || true)
      nerr=$(grep -c '\[error\]' "$run_dir/hailort.log" || true)
    fi
    prev_warn=$nwarn
    line=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | sed 's/^RESULT_CSV,//' || true)
    if [[ -z "$line" ]]; then
      echo "  실패 rc=$rc (stdout.log 참조)"; tail -8 "$run_dir/stdout.log"; continue
    fi
    # RESULT_CSV: label,mode,layout,... → label 뒤에 tool을 끼워 넣는다
    lab="${line%%,*}"; rest="${line#*,}"
    echo "$lab,$tool,$rest,$r,$COOL_WAITED,$nwarn,$nerr" >> "$RAW"
    echo "  $(grep -E '^wall|^cpu |^detections' "$run_dir/stdout.log" | tr '\n' ' ') warn=$nwarn err=$nerr rc=$rc"
  done
done

/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)
num = ["fps", "wall_s", "cpu_ms_per_frame", "decode_ms_mean", "pre_ms_mean", "pre_ms_p90", "npu_ms_mean",
       "npu_ms_p90", "e2e_ms_mean", "e2e_ms_p90", "detections", "hailo_t1", "soc_t1"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "tool", "mode", "layout", "batch", "slots", "runs", "frames"] +
               [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "cpu_ms_min", "cpu_ms_max", "detections_min", "detections_max",
                "hailo_t1_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [statistics.median(float(x[k]) for x in g) for k in num]
        fps = [float(x["fps"]) for x in g]
        cpu = [float(x["cpu_ms_per_frame"]) for x in g]
        det = [int(x["detections"]) for x in g]
        w.writerow([lab, g[0]["tool"], g[0]["mode"], g[0]["layout"], g[0]["batch"], g[0]["slots"], len(g),
                    g[0]["frames"]] + [f"{v:.3f}" for v in med] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}", f"{min(cpu):.3f}", f"{max(cpu):.3f}",
                    min(det), max(det), f"{max(float(x['hailo_t1']) for x in g):.1f}",
                    sum(int(x["hailort_warnings"]) for x in g), sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])
print(open(sys.argv[2]).read())
EOF

# 검출 JSONL 덤프 (측정과 별도 — 타이밍에 넣지 않는다)
if [[ "$DUMPS" == "1" ]]; then
  for c in "${configs[@]}"; do
    IFS='|' read -r id tool extra <<<"$c"
    echo "$id" | grep -Eq "$ONLY" || continue
    d="$OUT_DIR/../detections/$id"
    rm -rf "$d"; mkdir -p "$d"
    BIN=$([[ "$tool" == "gpu" ]] && echo "$GPU_BIN" || echo "$ISP_BIN")
    echo "[dumps] $id -> $d"
    t_now=$(hailo_temp)
    if awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then cool_down "덤프 전 ${t_now} C"; fi
    set +e
    # shellcheck disable=SC2086
    (cd "$d" && "$BIN" --video "$VIDEO" --hef "$HEF" $extra "${frames_arg[@]}" \
        --dump-dir "$d" --dump-every 300 --quiet > stdout.log 2>&1) || echo "  덤프 실패 ($id)"
    set -e
  done
fi

SRC_AFTER=$(src_snapshot)
{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)"
  echo "src tree md5 (after): $SRC_AFTER"
  echo "src newest (after): $(src_newest)"
  if [[ "$SRC_BEFORE" == "$SRC_AFTER" ]]; then
    echo "오염 점검: 01_src/·02_shaders/ 변경 없음 (md5 일치)"
  else
    echo "오염 점검: **측정 창 안에 01_src/ 또는 02_shaders/ 파일이 바뀌었다** (md5 불일치)"
  fi
} >> "$OUT_DIR/env_bench.txt"
