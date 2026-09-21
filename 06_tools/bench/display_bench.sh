#!/usr/bin/env bash
# M4 — HDMI 표시 통합 e2e 측정 (계획서 단계 4~5).
#
# 왜 op_point_bench.sh 를 쓰지 않는가:
#   그 스크립트는 `RESULT_CSV` 22필드만 거둔다. 표시 지표(disp_convert, present 간격,
#   glass-to-glass, held_frames_max …)는 `RESULT_CSV2` 줄의 **뒤에 덧붙인 14필드**에 있다.
#   그래서 두 줄을 합쳐 한 행으로 만드는 이 스크립트를 따로 둔다. 냉각·환경 스냅샷·루프 구조는
#   op_point_bench.sh 에서 그대로 복사했다(수정하지 않았다).
#
# ⚠ 측정은 반드시 --display-fullscreen 이다. 창 모드는 균일 스케일 경로로 넘어가
#   에러도 경고도 없이 표시 변환이 2.6배 느려진다.
# ⚠ 처리량 판정은 pace=asap, 지연·드롭 판정은 pace=source 로 축을 나눈다.
#   pace=source 에서는 FPS 가 소스 30 으로 고정돼 "FPS ±2%" 판정이 무의미해진다.
#
# 사용법:
#   bash 06_tools/bench/display_bench.sh
#   RUNS=3 FRAMES=900 ONLY='disp_on' bash 06_tools/bench/display_bench.sh
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 공통 함수(온도·PCIe·냉각·hailort 경고 판정·벤치 락)는 lib/common.sh 하나로 모았다.
# 예전에는 이 함수들이 7개 스크립트에 복붙돼 있었고 실제로 갈라져 있었다(pcie_link 의 Gen5 분기 등).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
BIN="${BIN:-$BIN_DIR/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/display_e2e}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-900}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-75}"
APPEND="${APPEND:-0}"

require_bin "$BIN" yvo_pipeline
check_only_exit
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
  _wl=$(ls -t "$XDG_RUNTIME_DIR"/wayland-[0-9]* 2>/dev/null | head -1 || true)
  [[ -n "$_wl" ]] && export WAYLAND_DISPLAY="$(basename "$_wl")"
fi
HAVE_WAYLAND=0; [[ -n "${WAYLAND_DISPLAY:-}" ]] && HAVE_WAYLAND=1

# id|자유형 인자열. 운용점은 b2/s4 --postproc gpu 로 고정하고 표시 축만 움직인다.
configs=(
  "reg_host_off |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc host"
  "disp_off     |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu"
  "disp_asap    |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu --display on --display-fullscreen --display-pace asap"
  "disp_source  |--mode dmabuf --layout nhcw --batch 2 --slots 4 --postproc gpu --display on --display-fullscreen --display-pace source"
)

# ── 복사 끝 ──

# M5a — NPU 유휴 GPU 기준선. M4 **직전 같은 열 상태에서** 찍는다. 이게 없으면 M4 의 GPU 시간이
# 경합 때문인지 코드 때문인지 구분할 수 없다. 냉각하지 않는다(의도적).
if [[ "${SKIP_M5A:-0}" != "1" ]] && (( HAVE_WAYLAND )); then
  PROBE="$BIN_DIR/yvo_display_probe"
  if [[ -x "$PROBE" ]]; then
    echo "[M5a] NPU 유휴 GPU 기준선 (yvo_display_probe --probe convert, fullscreen, overlay 5개)"
    mkdir -p "$OUT_DIR/raw/m5a"
    # ⚠ --video 는 절대경로로 넘긴다. 프로브를 raw/m5a 에서 돌리므로 기본 상대경로가 깨진다.
    (cd "$OUT_DIR/raw/m5a" && "$PROBE" --probe convert --video "$VIDEO" --frames 300 --display-fullscreen \
        --with-overlay --overlay-dets 5 > m5a.log 2>&1) || true
    grep -E "display path|on-screen shader|on-screen overlay|present_wait|presented" \
      "$OUT_DIR/raw/m5a/m5a.log" | tee "$OUT_DIR/m5a_baseline.txt"
  fi
fi

{
  echo "# display_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  echo "bin: $BIN"
  echo "bin md5: $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS"
  if (( HAVE_WAYLAND )); then echo "wayland: $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
  else echo "wayland: 없음 — --display 가 든 설정은 건너뛴다"; fi
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
# RESULT_CSV 22필드 + RESULT_CSV2 의 표시 14필드(label 중복 제외) + 스크립트 4필드
CSV_HEADER="label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,display,sync,pace,disp_convert_mean,disp_convert_p90,disp_overlay_mean,present_interval_median,e2e_glass_mean,e2e_glass_p90,presented,display_dropped,recreates,held_frames_max,slot_timeout_count,run,cool_wait_s,hailort_warnings,hailort_errors"
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
    if [[ "$args" == *"--display on"* ]] && (( ! HAVE_WAYLAND )); then
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
    # shellcheck disable=SC2086
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
    l1=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | sed 's/^RESULT_CSV,//' || true)
    # RESULT_CSV2 는 label 로 시작하므로 첫 필드를 버리고 이어 붙인다.
    l2=$(grep '^RESULT_CSV2,' "$run_dir/stdout.log" | cut -d, -f3- || true)
    if [[ -z "$l1" || -z "$l2" ]]; then
      echo "  실패 rc=$rc (stdout.log 참조)"; tail -5 "$run_dir/stdout.log"
      continue
    fi
    # l2 의 9번째 필드(ok)는 l1 의 ok 와 중복이라 뺀다.
    #   l2 = postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,[ok],표시 14개
    l2=$(awk -F, 'BEGIN{OFS=","}{ out=""; for(i=1;i<=NF;i++){ if(i==9) continue;
                                   out = (out=="" ? $i : out OFS $i) } print out }' <<<"$l2")
    echo "$l1,$l2,$r,$COOL_WAITED,$nwarn,$nerr" >> "$RAW"
    echo "  $(grep -E '^wall|^cpu |^held frames' "$run_dir/stdout.log" | tr '\n' ' ') hailo_warn=$nwarn err=$nerr rc=$rc"
  done
done

/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("행이 없다"); sys.exit(0)

NON_NUM = {"label", "mode", "layout", "run", "postproc", "alloc", "display", "sync", "pace"}
def numeric(key):
    if key in NON_NUM:
        return False
    for r in rows:
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
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "postproc", "display", "sync", "pace", "runs"] +
               [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "hailort_warnings", "hailort_errors", "all_ok"])
    for lab in order:
        g = groups[lab]
        med = [statistics.median(float(x[k]) for x in g) for k in num]
        fps = [float(x["fps"]) for x in g]
        w.writerow([lab, g[0]["postproc"], g[0]["display"], g[0]["sync"], g[0]["pace"], len(g)] +
                   [f"{v:.3f}" for v in med] + [f"{min(fps):.2f}", f"{max(fps):.2f}",
                    sum(int(x["hailort_warnings"]) for x in g),
                    sum(int(x["hailort_errors"]) for x in g),
                    int(all(x["ok"] == "1" for x in g))])

show = ["fps_median", "cpu_ms_per_frame_median", "gpu_ms_mean_median", "e2e_ms_mean_median",
        "disp_convert_mean_median", "disp_overlay_mean_median", "present_interval_median_median",
        "e2e_glass_mean_median", "presented_median", "display_dropped_median",
        "held_frames_max_median", "slot_timeout_count_median", "detections_median"]
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
