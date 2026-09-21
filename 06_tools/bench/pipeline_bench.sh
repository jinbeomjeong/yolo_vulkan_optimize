#!/usr/bin/env bash
# 통합 파이프라인(yvo_pipeline) 성능 측정
#   {ptr, dmabuf} × NHWC + dmabuf × NHCW, 각각 {batch1/slots2, batch4/slots8, batch8/slots16} × RUNS회
#   NHCW는 HailoRT의 CPU 입력 변환을 없애는 경로이고 dma-buf 입력에서만 동작한다.
#
# 사용법:
#   bash 06_tools/bench/pipeline_bench.sh                    # 전체(1816프레임), 설정당 3회
#   RUNS=1 FRAMES=300 bash 06_tools/bench/pipeline_bench.sh  # 짧게
#   ONLY='dmabuf_b8' bash 06_tools/bench/pipeline_bench.sh   # 설정 id(grep -E 패턴) 일부만
#   ONLY='nhcw' APPEND=1 bash ...                        # 기존 results_raw.csv에 이어붙이기(같은 냉각 규칙일 때만)
#   COOL_C=65 COOL_MAX_S=120 HOT_C=75 bash ...          # 냉각 조건
#   OUT_DIR=/path bash ...                               # 출력 디렉터리
#   SAMPLES=0 bash ...                                   # 샘플 PNG·검출 JSONL 덤프 실행 생략
#
# 실행 순서: 반복(run) 바깥 루프, 설정 안쪽 루프(열 상태 편향을 설정 간에 고르게 나눈다).
# 냉각: 직전 실행의 hailort.log에 온도·스로틀 경고가 있거나, 실행 직전 Hailo 칩 온도가 HOT_C 초과면
#       칩 온도가 COOL_C 이하가 될 때까지(최대 COOL_MAX_S초) 대기한다. 칩 온도는 hailo_platform(Python)으로 읽는다.
# 출력:
#   $OUT_DIR/raw/<id>_run<N>/{stdout.log,hailort.log}  실행별 로그 (hailort.log는 실행 디렉터리에 생긴다)
#   $OUT_DIR/results_raw.csv   실행별 RESULT_CSV 행 + 대기 시간·경고 수
#   $OUT_DIR/results.csv       설정별 중앙값
#   $OUT_DIR/samples/          dmabuf batch8 1회 실행의 detections.jsonl + 박스 PNG (측정과 별도)
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 공통 함수(온도·PCIe·냉각·hailort 경고 판정·벤치 락)는 lib/common.sh 하나로 모았다.
# 예전에는 이 함수들이 7개 스크립트에 복붙돼 있었고 실제로 갈라져 있었다(pcie_link 의 Gen5 분기 등).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/pipeline_ffmpeg_vulkan_hailo}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
SAMPLES="${SAMPLES:-1}"
SAMPLE_LAYOUT="${SAMPLE_LAYOUT:-nhcw}"   # 샘플 덤프에 쓸 레이아웃
APPEND="${APPEND:-0}"                  # 1이면 results_raw.csv를 지우지 않고 이어붙인다

require_bin "$BIN" yvo_pipeline
check_only_exit
guard_out_dir   # OUT_DIR 이 비어 있지 않으면 FORCE=1 없이는 멈춘다(보관 결과 보호)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

# id|mode|layout|batch|slots
configs=(
  "ptr_b1|ptr|nhwc|1|2"
  "ptr_b4|ptr|nhwc|4|8"
  "ptr_b8|ptr|nhwc|8|16"
  "dmabuf_b1|dmabuf|nhwc|1|2"
  "dmabuf_b4|dmabuf|nhwc|4|8"
  "dmabuf_b8|dmabuf|nhwc|8|16"
  "nhcw_b1|dmabuf|nhcw|1|2"
  "nhcw_b4|dmabuf|nhcw|4|8"
  "nhcw_b8|dmabuf|nhcw|8|16"
)


{
  echo "# pipeline_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
} > "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
CSV_HEADER="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,run,cool_wait_s,hailort_warnings,hailort_errors"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1인데 기존 $RAW 헤더가 다르다" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

prev_warn=0
frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id mode layout batch slots <<<"$c"
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
    echo "[$(date +%T)] run $r/$RUNS $id (hailo $(hailo_temp) C)"
    set +e
    (cd "$run_dir" && "$BIN" --video "$VIDEO" --hef "$HEF" --mode "$mode" --layout "$layout" \
        --batch "$batch" --slots "$slots" \
        "${frames_arg[@]}" --quiet --csv-label "${id}" > stdout.log 2>&1)
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
      echo "  실패 rc=$rc (stdout.log 참조)"; tail -5 "$run_dir/stdout.log"
      continue
    fi
    echo "$line,$r,$COOL_WAITED,$nwarn,$nerr" >> "$RAW"
    echo "  $(grep -E '^wall|^cpu ' "$run_dir/stdout.log" | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc"
  done
done

# 설정별 중앙값
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)
num = ["fps", "wall_s", "cpu_ms_per_frame", "decode_ms_mean", "gpu_ms_mean", "gpu_ms_p90", "npu_ms_mean",
       "npu_ms_p90", "e2e_ms_mean", "e2e_ms_p90", "detections", "hailo_t1", "soc_t1"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "mode", "layout", "batch", "slots", "runs", "frames"] + [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "hailo_t1_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [statistics.median(float(x[k]) for x in g) for k in num]
        fps = [float(x["fps"]) for x in g]
        w.writerow([lab, g[0]["mode"], g[0]["layout"], g[0]["batch"], g[0]["slots"], len(g), g[0]["frames"]] +
                   [f"{v:.3f}" for v in med] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}", f"{max(float(x['hailo_t1']) for x in g):.1f}",
                    sum(int(x["hailort_warnings"]) for x in g), sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])
print(open(sys.argv[2]).read())
EOF

if [[ "$SAMPLES" == "1" ]]; then
  echo "[samples] dmabuf/$SAMPLE_LAYOUT batch8 --dump-dir (측정 제외)"
  rm -rf "$OUT_DIR/samples"; mkdir -p "$OUT_DIR/samples"
  (cd "$OUT_DIR/samples" && "$BIN" --video "$VIDEO" --hef "$HEF" --mode dmabuf --layout "$SAMPLE_LAYOUT" --batch 8 "${frames_arg[@]}" \
      --dump-dir "$OUT_DIR/samples" --dump-every 300 > stdout.log 2>&1) || echo "  samples 실행 실패"
fi
{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
} >> "$OUT_DIR/env_bench.txt"
