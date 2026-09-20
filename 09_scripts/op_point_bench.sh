#!/usr/bin/env bash
# 운용점(operating point) 측정 — batch/slots/gpu_depth/postproc/display 매트릭스
#
# 왜 pipeline_bench.sh 를 쓰지 않는가:
#   pipeline_bench.sh 의 configs 는 `id|mode|layout|batch|slots` 5필드 고정이라
#   --gpu-depth·--postproc·--display* 를 넣을 자리가 없다. 그리고 그 스크립트의 9행은
#   **회귀 기준선 자산**이다(이 스크립트의 대조군이 그것과 비교해 세션 유효성을 판정한다).
#   스키마를 바꾸면 그 자산을 재현할 수 없으므로 건드리지 않고 새로 만든다.
#   configs 스키마는 같은 저장소에 이미 검증된 선례인 isp_pipeline_bench.sh 의 `id|자유형 인자열`을 따른다.
#   냉각·환경 스냅샷·루프 구조는 pipeline_bench.sh 에서 그대로 복사했다(수정하지 않았다).
#
# 사용법:
#   bash 09_scripts/op_point_bench.sh                       # 전체, 설정당 3회
#   RUNS=3 FRAMES=900 ONLY='b2s2|b8s8' bash ...             # 일부만
#   BIN=/path/to/yvo_pipeline bash ...                      # 바이너리 고정(측정 중 재빌드 차단)
#   APPEND=1 bash ...                                       # 기존 results_raw.csv 에 이어붙이기
#
# 실행 순서: 반복(run) 바깥 루프, 설정 안쪽 루프 — 열 상태 편향을 설정 간에 고르게 나눈다.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$PROJ/build/01_src/pipeline/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/op_point}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-900}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
APPEND="${APPEND:-0}"
PY_HAILO=/home/pi/miniconda3/envs/hailo-yolo/bin/python

[[ -x "$BIN" ]] || { echo "yvo_pipeline 없음: $BIN" >&2; exit 1; }
mkdir -p "$OUT_DIR/raw"

# 표시 설정을 돌리려면 Wayland 가 필요하다. 호출자 환경에 의존하지 않고 여기서 찾는다.
# 없으면 --display 가 든 설정을 건너뛰고 그 사실을 화면과 env_bench.txt 에 남긴다.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
  _wl=$(ls -t "$XDG_RUNTIME_DIR"/wayland-[0-9]* 2>/dev/null | head -1 || true)
  [[ -n "$_wl" ]] && export WAYLAND_DISPLAY="$(basename "$_wl")"
fi
HAVE_WAYLAND=0; [[ -n "${WAYLAND_DISPLAY:-}" ]] && HAVE_WAYLAND=1

# id|자유형 인자열
#   단계 1(M1): 배치·슬롯 스윕. 표시 off, postproc host, --gpu-depth 는 기본값(3)을 그대로 둬
#   기존 07_results/pipeline_ffmpeg_vulkan_hailo/results.csv 와 직접 비교되게 한다.
#   b2s2 를 맨 앞에 둔 이유: 30 FPS 를 못 넘길 실위험이 있어 가장 먼저 알아야 한다.
configs=(
  "b2s2      |--mode dmabuf --layout nhcw --batch 2 --slots 2"
  "b2s4      |--mode dmabuf --layout nhcw --batch 2 --slots 4"
  "b4s4      |--mode dmabuf --layout nhcw --batch 4 --slots 4"
  "b8s8      |--mode dmabuf --layout nhcw --batch 8 --slots 8"
  "ctl_b4s8  |--mode dmabuf --layout nhcw --batch 4 --slots 8"    # 대조군: 기존 44.52 FPS
  "ctl_b8s16 |--mode dmabuf --layout nhcw --batch 8 --slots 16"   # 대조군: 기존 51.63 FPS
)

# 지연 최소화 스윕 (ONLY='lat_' 로 고른다). 30 FPS 를 포기하는 대신 지연을 얼마나 줄일 수 있는지 본다.
#   슬롯을 줄이면 큐 대기가 사라지고, gpu_depth 를 줄이면 GPU 단계 버퍼링이 사라진다.
#   s=1 은 파이프라이닝이 아예 없는 완전 직렬 경로다(미측정 구간).
configs_lat=(
  "lat_b1s1d1 |--mode dmabuf --layout nhcw --batch 1 --slots 1 --gpu-depth 1"
  "lat_b1s2d1 |--mode dmabuf --layout nhcw --batch 1 --slots 2 --gpu-depth 1"
  "lat_b2s2d1 |--mode dmabuf --layout nhcw --batch 2 --slots 2 --gpu-depth 1"
  "lat_b2s3d2 |--mode dmabuf --layout nhcw --batch 2 --slots 3 --gpu-depth 2"
  "lat_b2s4d1 |--mode dmabuf --layout nhcw --batch 2 --slots 4 --gpu-depth 1"
)
[[ "${USE_LAT:-0}" == "1" ]] && configs=("${configs_lat[@]}")

# ── 아래 5개 함수는 09_scripts/pipeline_bench.sh 에서 그대로 복사했다 (수정 금지) ──
hailo_temp() {  # hailort.log 가 raw/ 에 생기도록 그 디렉터리에서 실행
  (cd "$OUT_DIR/raw" && "$PY_HAILO" - <<'EOF' 2>/dev/null) || echo nan
from hailo_platform import Device
d = Device()
t = d.control.get_chip_temperature()
print(f"{max(t.ts0_temperature, t.ts1_temperature):.1f}")
d.release()
EOF
}
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
# ── 복사 끝 ──

{
  echo "# op_point_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  echo "bin: $BIN"
  echo "bin md5: $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "bin mtime: $(stat -c %y "$BIN")"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
  if (( HAVE_WAYLAND )); then echo "wayland: $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
  else echo "wayland: 없음 — --display 가 든 설정은 건너뛴다"; fi
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
# yvo_pipeline 의 RESULT_CSV 22필드 + 이 스크립트가 붙이는 4필드.
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
    IFS='|' read -r id args <<<"$c"
    id="${id// /}"
    echo "$id" | grep -Eq "$ONLY" || continue
    if [[ "$args" == *"--display"* ]] && (( ! HAVE_WAYLAND )); then
      echo "[skip] $id — Wayland 없음"; continue
    fi
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
    # shellcheck disable=SC2086  # args 는 의도적으로 단어 분할한다
    (cd "$run_dir" && "$BIN" --video "$VIDEO" --hef "$HEF" $args \
        "${frames_arg[@]}" --quiet --csv-label "$id" > stdout.log 2>&1)
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

# 설정별 중앙값.
# pipeline_bench.sh 와 달리 숫자 열 목록을 하드코딩하지 않는다 — 헤더에서 자동으로 고른다.
# 그래야 yvo_pipeline 이 RESULT_CSV 에 열을 더해도 이 스크립트를 고칠 필요가 없다.
/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("행이 없다"); sys.exit(0)

NON_NUM = {"label", "mode", "layout", "run"}          # 식별자 — 중앙값을 내지 않는다
def numeric(key):
    if key in NON_NUM:
        return False
    for r in rows:                                     # 전 행이 float 로 읽히면 숫자 열이다
        try: float(r[key])
        except (TypeError, ValueError): return False
    return True

num = [k for k in rows[0] if numeric(k)]

order, groups = [], {}
for r in rows:
    if r["label"] not in groups:
        order.append(r["label"]); groups[r["label"]] = []
    groups[r["label"]].append(r)

with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")             # CRLF 금지 (measure 스킬 §5)
    w.writerow(["label", "mode", "layout", "runs"] + [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [statistics.median(float(x[k]) for x in g) for k in num]
        fps = [float(x["fps"]) for x in g]
        w.writerow([lab, g[0]["mode"], g[0]["layout"], len(g)] + [f"{v:.3f}" for v in med] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}",
                    sum(int(x["hailort_warnings"]) for x in g),
                    sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])

# 화면에는 판정에 쓰는 열만 좁게 찍는다(전체는 results.csv 에 있다).
show = ["fps_median", "cpu_ms_per_frame_median", "npu_ms_mean_median",
        "e2e_ms_mean_median", "e2e_ms_p90_median", "detections_median", "hailo_t1_median"]
out = list(csv.DictReader(open(sys.argv[2])))
hdr = ["label"] + [c for c in show if c in out[0]]
wid = [max(len(h), *(len(r[h]) for r in out)) for h in hdr]
print(" | ".join(h.ljust(w) for h, w in zip(hdr, wid)))
print("-+-".join("-" * w for w in wid))
for r in out:
    print(" | ".join(r[h].ljust(w) for h, w in zip(hdr, wid)))
EOF

{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "bin md5 (측정 후): $(md5sum "$BIN" | cut -d' ' -f1)"
  vcgencmd get_throttled 2>/dev/null || true
} >> "$OUT_DIR/env_bench.txt"
