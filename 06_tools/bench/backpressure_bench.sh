#!/usr/bin/env bash
# 러너 백프레셔 수정(hailo_raw_runner.cpp: 수동 increase/decrease 제거) 전후 비교 측정.
# host(기준 경로) 와 gpu(원시 러너 경로) 를 **같은 세션에서 인터리브**해 열 상태 편향을 없앤다.
# 냉각·환경 스냅샷·루프 구조는 06_tools/bench/op_point_bench.sh 에서 복사했다.
set -euo pipefail
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 공통 함수(온도·PCIe·냉각·hailort 경고 판정·벤치 락)는 lib/common.sh 하나로 모았다.
# 예전에는 이 함수들이 7개 스크립트에 복붙돼 있었고 실제로 갈라져 있었다(pcie_link 의 Gen5 분기 등).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
bench_lock_acquire
BIN="${BIN:-$PROJ/build/01_src/pipeline/yvo_pipeline}"
VIDEO="${VIDEO:-$PROJ/04_data/videos/video_h265.mp4}"
HEF="${HEF:-$PROJ/03_models/yolov11m.hef}"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/raw_runner_backpressure}"
RUNS="${RUNS:-3}"; FRAMES="${FRAMES:-900}"; ONLY="${ONLY:-.}"
COOL_C="${COOL_C:-65}"; COOL_MAX_S="${COOL_MAX_S:-120}"; HOT_C="${HOT_C:-75}"
PY_HAILO=/home/pi/miniconda3/envs/hailo-yolo/bin/python
mkdir -p "$OUT_DIR/raw"

configs=(
  "host_b2s4 |--mode dmabuf --layout nhcw --batch 2 --slots 4  --postproc host"
  "gpu_b2s4  |--mode dmabuf --layout nhcw --batch 2 --slots 4  --postproc gpu"
  "host_b8s16|--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc host"
  "gpu_b8s16 |--mode dmabuf --layout nhcw --batch 8 --slots 16 --postproc gpu"
)

{
  echo "# raw_runner_backpressure bench $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "soc temp: $(soc_temp) C, hailo temp: $(hailo_temp) C"
  vcgencmd get_throttled 2>/dev/null || true
  vcgencmd measure_clock arm 2>/dev/null || true
  pcie_link
  echo "bin: $BIN"; echo "bin md5: $(md5sum "$BIN" | cut -d' ' -f1)"; echo "bin mtime: $(stat -c %y "$BIN")"
  hailortcli fw-control identify 2>/dev/null | grep -E 'Firmware Version' || true
  echo "video: $VIDEO frames: $FRAMES runs: $RUNS"
} > "$OUT_DIR/env_bench.txt"
cat "$OUT_DIR/env_bench.txt"

RAW="$OUT_DIR/results_raw.csv"
echo "label,postproc,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_ms_mean,gpu_ms_mean,gpu_ms_p90,npu_ms_mean,npu_ms_p90,e2e_ms_mean,e2e_ms_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,run,cool_wait_s,hailort_warnings,hailort_errors,in_flight" > "$RAW"

prev_warn=0
for ((r = 1; r <= RUNS; r++)); do
  for c in "${configs[@]}"; do
    IFS='|' read -r id args <<<"$c"; id="${id// /}"
    echo "$id" | grep -Eq "$ONLY" || continue
    COOL_WAITED=0; t_now=$(hailo_temp)
    if (( prev_warn > 0 )); then cool_down "hailort.log 경고 ${prev_warn}건"
    elif awk -v t="$t_now" -v h="$HOT_C" 'BEGIN{exit !(t!="nan" && t>h)}'; then cool_down "칩 온도 ${t_now} C > ${HOT_C} C"; fi
    run_dir="$OUT_DIR/raw/${id}_run${r}"; rm -rf "$run_dir"; mkdir -p "$run_dir"
    echo "[$(date +%T)] run $r/$RUNS $id (hailo $(hailo_temp) C)"
    set +e
    # shellcheck disable=SC2086
    (cd "$run_dir" && "$BIN" --video "$VIDEO" --hef "$HEF" $args --frames "$FRAMES" --quiet --csv-label "$id" > stdout.log 2>&1)
    rc=$?
    set -e
    nwarn=0; nerr=0
    if [[ -f "$run_dir/hailort.log" ]]; then
      nwarn=$(grep -E '\[warning\]' "$run_dir/hailort.log" | grep -cE "$WARN_RE" || true)
      nerr=$(grep -c '\[error\]' "$run_dir/hailort.log" || true)
    fi
    prev_warn=$nwarn
    l1=$(grep '^RESULT_CSV,'  "$run_dir/stdout.log" | sed 's/^RESULT_CSV,//'  || true)
    l2=$(grep '^RESULT_CSV2,' "$run_dir/stdout.log" | sed 's/^RESULT_CSV2,//' || true)
    if [[ -z "$l1" || -z "$l2" ]]; then echo "  실패 rc=$rc"; tail -5 "$run_dir/stdout.log"; continue; fi
    # RESULT_CSV(22) + RESULT_CSV2 의 후처리 지표 6개 + 메타 + in-flight(= fps x npu_mean/1000)
    /usr/bin/python3 - "$RAW" "$l1" "$l2" "$r" "$COOL_WAITED" "$nwarn" "$nerr" <<'EOF'
import sys
raw, l1, l2, run, cool, nwarn, nerr = sys.argv[1:8]
a = l1.split(","); b = l2.split(",")
fps = float(a[7]); npu = float(a[12])
row = a[:2] + [b[1]] + a[2:22] + b[3:9] + [run, cool, nwarn, nerr, f"{fps*npu/1000.0:.2f}"]
# label,postproc 순서로 맞춘다: a[0]=label, a[1]=mode → label,postproc,mode,...
row = [a[0], b[1]] + a[1:22] + b[3:9] + [run, cool, nwarn, nerr, f"{fps*npu/1000.0:.2f}"]
open(raw, "a").write(",".join(row) + "\n")
EOF
    echo "  $(grep -E '^wall|^cpu ' "$run_dir/stdout.log" | tr '\n' ' ') warn=$nwarn err=$nerr rc=$rc"
  done
done

/usr/bin/python3 - "$RAW" "$OUT_DIR/results.csv" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows: print("행이 없다"); sys.exit(0)
NON_NUM = {"label", "postproc", "mode", "layout", "run"}
def numeric(k):
    if k in NON_NUM: return False
    for r in rows:
        try: float(r[k])
        except (TypeError, ValueError): return False
    return True
num = [k for k in rows[0] if numeric(k)]
order, groups = [], {}
for r in rows:
    lab = r["label"]
    if lab not in groups:
        order.append(lab)
        groups[lab] = []
    groups[lab].append(r)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["label", "postproc", "runs"] + [f"{k}_median" for k in num] + ["fps_min", "fps_max"])
    for lab in order:
        g = groups[lab]
        meds = [f"{statistics.median(float(r[k]) for r in g):.3f}" for k in num]
        fpss = [float(r["fps"]) for r in g]
        w.writerow([lab, g[0]["postproc"], len(g)] + meds + [f"{min(fpss):.2f}", f"{max(fpss):.2f}"])
print(open(sys.argv[2]).read())
EOF
