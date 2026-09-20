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
# npu_guard 훅(.claude/hooks/npu_guard.sh)이 이 파일의 PID 로 "벤치가 돌고 있다"를 판단한다.
# 측정 중에 빌드·테스트가 끼어들면 CPU 시간 측정이 에러 없이 조용히 오염되기 때문이다.
# 훅이 PID 생존 여부를 보므로 비정상 종료로 남은 락은 자동으로 무효가 된다.
BENCH_LOCK="${BENCH_LOCK:-$PROJ/build/.bench_running}"
mkdir -p "$(dirname "$BENCH_LOCK")"
printf '%s' "$$" > "$BENCH_LOCK"
trap 'rm -f "$BENCH_LOCK"' EXIT INT TERM
BIN="${BIN:-$PROJ/build/01_src/pipeline/yvo_pipeline}"
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
PY_HAILO=/home/pi/miniconda3/envs/hailo-yolo/bin/python

[[ -x "$BIN" ]] || { echo "yvo_pipeline 없음: $BIN (먼저 cmake --build build)" >&2; exit 1; }
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

hailo_temp() {  # hailort.log가 raw/에 생기도록 그 디렉터리에서 실행
  (cd "$OUT_DIR/raw" && "$PY_HAILO" - <<'EOF' 2>/dev/null) || echo nan
from hailo_platform import Device
d = Device()
t = d.control.get_chip_temperature()
print(f"{max(t.ts0_temperature, t.ts1_temperature):.1f}")
d.release()
EOF
}
# PCIe 링크 세대·폭을 한 줄로 출력한다. 예: "pcie: 8.0 GT/s PCIe x1 (Gen3, 0001:01:00.0)"
# sysfs 를 쓴다. lspci -vv 의 LnkSta: 는 root 가 아니면 출력되지 않아 조용히 빈 값이 된다
# (과거에 이 때문에 env 스냅샷에 링크 속도가 하나도 기록되지 않았다).
# 읽기에 실패하면 반드시 그 사실을 찍는다. 빈 줄은 "정보 없음"과 구분되지 않는다.
pcie_link() {
  local d speed width gen
  for d in /sys/bus/pci/devices/*/; do
    [[ "$(cat "$d/vendor" 2>/dev/null)" == "0x1e60" ]] || continue   # Hailo
    speed=$(cat "$d/current_link_speed" 2>/dev/null)
    width=$(cat "$d/current_link_width" 2>/dev/null)
    if [[ -z "$speed" || -z "$width" ]]; then
      echo "pcie: 읽기 실패 ($d current_link_speed/width)"; return
    fi
    case "${speed%% *}" in
      2.5) gen=Gen1 ;; 5.0) gen=Gen2 ;; 8.0) gen=Gen3 ;; 16.0) gen=Gen4 ;; 32.0) gen=Gen5 ;;
      *)   gen="Gen?" ;;
    esac
    echo "pcie: $speed x$width ($gen, $(basename "$d"))"
    return
  done
  echo "pcie: 읽기 실패 (vendor 0x1e60 장치를 /sys/bus/pci/devices 에서 찾지 못함)"
}

soc_temp() { vcgencmd measure_temp 2>/dev/null | sed -E 's/temp=([0-9.]+).*/\1/' || echo nan; }

# 경고 패턴: 온도·스로틀·클럭 저하
WARN_RE='[Tt]emperature|[Tt]hrottl|[Oo]verheat|clock'

cool_down() {  # $1 = 이유
  local t waited=0
  t=$(hailo_temp)
  echo "  cooling ($1): hailo ${t} C -> <= ${COOL_C} C (max ${COOL_MAX_S}s)"
  while awk -v t="$t" -v c="$COOL_C" 'BEGIN{exit !(t=="nan" || t>c)}' && (( waited < COOL_MAX_S )); do
    sleep 5; waited=$((waited + 5)); t=$(hailo_temp)
  done
  echo "  cooled ${waited}s, hailo ${t} C"
  COOL_WAITED=$waited
}

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
