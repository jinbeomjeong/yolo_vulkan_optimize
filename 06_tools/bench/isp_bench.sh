#!/usr/bin/env bash
# 전처리 단계 성능 측정: PiSP BE(ISP) 경로 vs Vulkan 셰이더 경로 vs 디코드 단독
#   NPU를 쓰지 않는다(전처리 단계만). 전체 파이프라인은 06_tools/bench/pipeline_bench.sh다.
#
# 사용법:
#   bash 06_tools/bench/isp_bench.sh                       # 전체(1816프레임), 설정당 3회
#   RUNS=1 FRAMES=300 bash 06_tools/bench/isp_bench.sh     # 짧게
#   ONLY='isp_' bash 06_tools/bench/isp_bench.sh           # 설정 id(grep -E 패턴) 일부만
#   OUT_DIR=/path bash ...                             # 출력 디렉터리
#
# 시간 축이 두 개다. 섞어 읽으면 안 된다.
#   wall   : 호스트 wall-clock. ISP는 BackendDevice::Run() 직전/직후(동기).
#   device : Vulkan timestamp query가 준 GPU 디바이스 시간. 호스트 제출·대기 비용이 빠져 있다.
# 그래서 GPU는 두 설정으로 잰다: depth 2(파이프라인, 처리량)와 depth 1(동기, wall-clock 비교용).
#
# 실행 순서: 반복(run) 바깥 루프, 설정 안쪽 루프(열 상태 편향을 설정 간에 고르게 나눈다).
# 냉각: 이 단계는 NPU를 쓰지 않으므로 SoC 온도 기준이다. 실행 직전 SoC 온도가 HOT_C 초과면
#       COOL_C 이하가 될 때까지(최대 COOL_MAX_S초) 대기한다. Hailo 칩 온도는 스냅샷용으로만 읽는다.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 공통 함수(온도·PCIe·냉각·hailort 경고 판정·벤치 락)는 lib/common.sh 하나로 모았다.
# 예전에는 이 함수들이 7개 스크립트에 복붙돼 있었고 실제로 갈라져 있었다(pcie_link 의 Gen5 분기 등).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
bench_lock_acquire
ISP_BIN="${ISP_BIN:-$PROJ/build/01_src/isp/yvo_isp_bench}"
GPU_BIN="${GPU_BIN:-$PROJ/build/01_src/vulkan/yvo_preproc_bench}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/isp_resize}"
RUNS="${RUNS:-3}"
FRAMES="${FRAMES:-0}"
WARMUP="${WARMUP:-60}"
SLOTS="${SLOTS:-4}"
ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-66}"   # Pi 5(CM5, 액티브 쿨러)는 유휴에서도 SoC 65~66 C다. 이보다 낮게 잡으면 절대 못 내려간다.
COOL_MAX_S="${COOL_MAX_S:-120}"
HOT_C="${HOT_C:-72}"
APPEND="${APPEND:-0}"
PY_HAILO=/home/pi/miniconda3/envs/hailo-yolo/bin/python

[[ -x "$ISP_BIN" ]] || { echo "yvo_isp_bench 없음: $ISP_BIN (먼저 cmake --build build)" >&2; exit 1; }
[[ -x "$GPU_BIN" ]] || { echo "yvo_preproc_bench 없음: $GPU_BIN" >&2; exit 1; }
mkdir -p "$OUT_DIR/raw" "$OUT_DIR/samples"

# id|tool|args...
configs=(
  "decode_only|isp|--no-isp"
  "gpu_nhcw_d2|gpu|--layout nhcw --depth 2"
  "gpu_nhcw_d1|gpu|--layout nhcw --depth 1"
  "isp_A_nhwc|isp|--repack none"
  "isp_C_repack_cpu|isp|--repack cpu"
  "isp_C_repack_neon|isp|--repack neon"
)


{
  echo "# isp_bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "model: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)  (0x80000 = 과거 기록 비트, 현재 스로틀링 아님)"
  vcgencmd measure_clock arm 2>/dev/null || true
  vcgencmd measure_clock v3d 2>/dev/null || true
  pcie_link
  echo "libpisp: $(pkg-config --modversion libpisp 2>/dev/null || echo NA)"
  echo "video: $VIDEO frames: ${FRAMES:-all} runs: $RUNS warmup: $WARMUP slots: $SLOTS"
  echo "note: 이 측정은 NPU를 쓰지 않는다(전처리 단계 단독). hailort 경고 열은 항상 0이다."
} > "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
CSV_HEADER="label,tool,variant,time_axis,frames,wall_s,fps,cpu_ms_per_frame,pre_ms_median,pre_ms_p90,pre_ms_mean,pre_ms_min,pre_ms_max,host_submit_us_median,host_wait_us_median,repack_ms_median,repack_ms_p90,sync_ms_median,n,ok,run,cool_wait_s,soc_t0,soc_t1,throttled,hailort_warnings,hailort_errors"
if [[ "$APPEND" == "1" && -f "$RAW" ]]; then
  head -1 "$RAW" | grep -qx "$CSV_HEADER" || { echo "APPEND=1인데 기존 $RAW 헤더가 다르다" >&2; exit 1; }
else
  echo "$CSV_HEADER" > "$RAW"
fi

frames_arg=()
[[ "$FRAMES" != "0" ]] && frames_arg=(--frames "$FRAMES")

for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id tool extra <<<"$c"
    echo "$id" | grep -Eq "$ONLY" || continue
    COOL_WAITED=0
    t_now=$(soc_temp)
    if awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then
      cool_down "SoC 온도 ${t_now} C > ${HOT_C} C" soc   # 이 벤치는 NPU 를 쓰지 않는다
    fi
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    rm -rf "$run_dir"; mkdir -p "$run_dir"
    soc_t0=$(soc_temp)
    echo "[$(date +%T)] run $r/$RUNS $id (soc ${soc_t0} C)"
    set +e
    if [[ "$tool" == "isp" ]]; then
      # shellcheck disable=SC2086
      (cd "$run_dir" && "$ISP_BIN" --video "$VIDEO" --slots "$SLOTS" --warmup "$WARMUP" \
          "${frames_arg[@]}" $extra --no-png --csv-label "$id" > stdout.log 2>&1)
    else
      # GPU는 warmup 플래그가 없다. --repeat 2로 돌리고 두 번째(워밍업된) 블록을 쓴다.
      # shellcheck disable=SC2086
      (cd "$run_dir" && "$GPU_BIN" --video "$VIDEO" --slots 8 --repeat 2 \
          "${frames_arg[@]}" $extra --png-dir "$OUT_DIR/samples" --png-frame 100 > stdout.log 2>&1)
    fi
    rc=$?
    set -e
    soc_t1=$(soc_temp)
    thr=$(throttled)
    nwarn=0; nerr=0
    if [[ -f "$run_dir/hailort.log" ]]; then
      nwarn=$(grep -E '\[warning\]' "$run_dir/hailort.log" | grep -cE "$WARN_RE" || true)
      nerr=$(grep -c '\[error\]' "$run_dir/hailort.log" || true)
    fi

    if [[ "$tool" == "isp" ]]; then
      line=$(grep '^RESULT_CSV,' "$run_dir/stdout.log" | tail -1 | sed 's/^RESULT_CSV,//' || true)
      if [[ -z "$line" ]]; then
        echo "  실패 rc=$rc (stdout.log 참조)"; tail -5 "$run_dir/stdout.log"; continue
      fi
      # RESULT_CSV: label,mode,repack,slots,frames,wall_s,fps,cpu,med,p90,mean,min,max,rep_med,rep_p90,sync_med,n,ok
      IFS=',' read -r _lab mode repack _slots nfr wall fps cpu med p90 mean mn mx rmed rp90 smed nn ok <<<"$line"
      variant="$mode/$repack"
      echo "$id,isp,$variant,wall,$nfr,$wall,$fps,$cpu,$med,$p90,$mean,$mn,$mx,,,$rmed,$rp90,$smed,$nn,$ok,$r,$COOL_WAITED,$soc_t0,$soc_t1,$thr,$nwarn,$nerr" >> "$RAW"
      echo "  fps=$fps cpu=$cpu ms/f isp_median=$med ms repack_median=${rmed:-} rc=$rc"
    else
      # preproc_bench의 마지막 [decode+gpu] 블록을 파싱한다(--repeat 2의 두 번째 = 워밍업 후).
      mapfile -t gl < <(grep '^\[decode+gpu\]' "$run_dir/stdout.log")
      if (( ${#gl[@]} < 3 )); then
        echo "  실패 rc=$rc (stdout.log 참조)"; tail -5 "$run_dir/stdout.log"; continue
      fi
      # 마지막 3줄 = 마지막 반복의 frames/gpu/host 줄
      l_fr="${gl[-3]}"; l_gpu="${gl[-2]}"; l_host="${gl[-1]}"
      nfr=$(sed -E 's/.*frames=([0-9]+).*/\1/' <<<"$l_fr")
      wall=$(sed -E 's/.*wall=([0-9.]+) s.*/\1/' <<<"$l_fr")
      fps=$(sed -E 's/.*fps=([0-9.]+).*/\1/' <<<"$l_fr")
      cpu=$(sed -E 's/.*cpu=([0-9.]+) ms\/frame.*/\1/' <<<"$l_fr")
      med=$(sed -E 's/.*median=([0-9.]+).*/\1/' <<<"$l_gpu")
      p90=$(sed -E 's/.*p90=([0-9.]+).*/\1/' <<<"$l_gpu")
      mean=$(sed -E 's/.*mean=([0-9.]+).*/\1/' <<<"$l_gpu")
      mn=$(sed -E 's/.*min=([0-9.]+).*/\1/' <<<"$l_gpu")
      mx=$(sed -E 's/.*max=([0-9.]+) .*/\1/' <<<"$l_gpu")
      nn=$(sed -E 's/.*\(n=([0-9]+)\).*/\1/' <<<"$l_gpu")
      sub=$(sed -E 's/.*submit us: median=([0-9.]+).*/\1/' <<<"$l_host")
      wt=$(sed -E 's/.*wait us: median=([0-9.]+).*/\1/' <<<"$l_host")
      ok=$([[ $rc -eq 0 ]] && echo 1 || echo 0)
      echo "$id,gpu,$extra,device,$nfr,$wall,$fps,$cpu,$med,$p90,$mean,$mn,$mx,$sub,$wt,,,,$nn,$ok,$r,$COOL_WAITED,$soc_t0,$soc_t1,$thr,$nwarn,$nerr" >> "$RAW"
      echo "  fps=$fps cpu=$cpu ms/f gpu_device_median=$med ms submit+wait=$(awk -v a="$sub" -v b="$wt" 'BEGIN{printf "%.3f", (a+b)/1000}') ms rc=$rc"
    fi
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
num = ["fps", "wall_s", "cpu_ms_per_frame", "pre_ms_median", "pre_ms_p90", "pre_ms_mean", "pre_ms_min",
       "pre_ms_max", "host_submit_us_median", "host_wait_us_median", "repack_ms_median", "repack_ms_p90",
       "sync_ms_median", "soc_t1"]
def med(g, k):
    vals = [float(x[k]) for x in g if x[k] not in ("", None)]
    return f"{statistics.median(vals):.3f}" if vals else ""
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "tool", "variant", "time_axis", "runs", "frames"] + [f"{k}_median" for k in num] +
               ["fps_min", "fps_max", "cpu_ms_min", "cpu_ms_max", "hailort_warnings", "all_ok"])
    for lab in order:
        g = groups[lab]
        fps = [float(x["fps"]) for x in g]
        cpu = [float(x["cpu_ms_per_frame"]) for x in g]
        w.writerow([lab, g[0]["tool"], g[0]["variant"], g[0]["time_axis"], len(g), g[0]["frames"]] +
                   [med(g, k) for k in num] +
                   [f"{min(fps):.2f}", f"{max(fps):.2f}", f"{min(cpu):.3f}", f"{max(cpu):.3f}",
                    sum(int(x["hailort_warnings"]) for x in g), int(all(x["ok"] == "1" for x in g))])
print(open(sys.argv[2]).read())
EOF

# 샘플 PNG (측정과 별도)
echo "[samples] ISP letterbox PNG (측정 제외)"
(cd "$OUT_DIR/samples" && "$ISP_BIN" --video "$VIDEO" --frames 200 --warmup 0 --repack none \
    --png-dir "$OUT_DIR/samples" --png-frame 100 > isp_sample.log 2>&1) || echo "  samples 실행 실패"

{
  echo "# after $(date -Is)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  echo "throttled: $(throttled)"
} >> "$OUT_DIR/env_bench.txt"
