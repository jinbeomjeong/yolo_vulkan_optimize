# 벤치 스크립트 공통 함수. 각 스크립트가 `source` 해서 쓴다.
#
# 왜 만들었나 (2026-09-21):
#   같은 함수가 7개 스크립트에 복붙돼 있었고 **동기화 장치가 없어서 실제로 갈라져 있었다.**
#   - `pcie_link()` 7벌 중 `backpressure_bench.sh` 만 Gen5(32.0 GT/s) 분기가 빠져 있었다
#     → Gen5 링크를 `Gen?` 로 기록한다. 측정 조건 기록이 틀리는 것은 이 프로젝트에서 가장 비싼 실수다.
#   - `cool_down()` 이 스크립트마다 센서(hailo/soc)·임계·루프 구조가 달랐고,
#     `decode_bench.sh` 는 함수 없이 인라인이라 `nan` 처리와 대기 시간 기록이 없었다.
#
# 전제: 이 파일을 source 하기 전에 아래가 정의돼 있어야 한다.
#   OUT_DIR        결과 폴더 (hailo_temp 가 $OUT_DIR/raw 에서 실행한다)
#   PY_HAILO       conda hailo-yolo 파이썬 경로
#   COOL_C         냉각 목표 온도
#   COOL_MAX_S     냉각 최대 대기 초
#
# 센서 선택: NPU 를 쓰는 벤치는 hailo(기본), NPU 를 안 쓰는 벤치(전처리·디코드)는 soc 를 쓴다.

# ── 온도 ───────────────────────────────────────────────────────────────────────
hailo_temp() {  # hailort.log 가 raw/ 에 생기도록 그 디렉터리에서 실행
  (cd "$OUT_DIR/raw" && "$PY_HAILO" - <<'PYEOF' 2>/dev/null) || echo nan
from hailo_platform import Device
d = Device()
t = d.control.get_chip_temperature()
print(f"{max(t.ts0_temperature, t.ts1_temperature):.1f}")
d.release()
PYEOF
}

soc_temp() { vcgencmd measure_temp 2>/dev/null | sed -E 's/temp=([0-9.]+).*/\1/' || echo nan; }

throttled() { vcgencmd get_throttled 2>/dev/null || echo "throttled=?"; }

# ── PCIe 링크 ──────────────────────────────────────────────────────────────────
# 문서에 적힌 값을 믿지 말고 sysfs 를 읽는다(measure 스킬 §1).
# lspci -vv 의 LnkSta: 는 root 가 아니면 아무것도 출력하지 않으므로 쓰지 않는다.
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

# ── 냉각 ───────────────────────────────────────────────────────────────────────
# $1 = 이유(로그용), $2 = 센서 hailo|soc (기본 hailo)
# 대기한 초를 COOL_WAITED 에 남긴다 — measure 스킬 §2 가 기록을 요구한다.
# 온도를 못 읽어 nan 이면 **기다린다**(성급히 측정하지 않는다).
cool_down() {
  local reason="${1:-}" sensor="${2:-hailo}" t waited=0
  local read_temp=hailo_temp
  [[ "$sensor" == "soc" ]] && read_temp=soc_temp
  t=$("$read_temp")
  echo "  cooling ($reason): ${sensor} ${t} C -> <= ${COOL_C} C (max ${COOL_MAX_S}s)"
  while awk -v t="$t" -v c="$COOL_C" 'BEGIN{exit !(t=="nan" || t>c)}' && (( waited < COOL_MAX_S )); do
    sleep 5; waited=$((waited + 5)); t=$("$read_temp")
  done
  echo "  cooled ${waited}s, ${sensor} ${t} C"
  COOL_WAITED=$waited
}

# ── hailort.log 신뢰성 판정 (measure 스킬 §4) ─────────────────────────────────
# 온도·스로틀·클럭 경고가 하나라도 있으면 그 실행의 성능 수치는 신뢰할 수 없다.
WARN_RE='[Tt]emperature|[Tt]hrottl|[Oo]verheat|clock'

# $1 = hailort.log 경로. "경고수 에러수" 를 출력한다.
count_hailort_warnings() {
  local log="$1" nwarn=0 nerr=0
  if [[ -f "$log" ]]; then
    nwarn=$(grep -E '\[warning\]' "$log" | grep -cE "$WARN_RE" || true)
    nerr=$(grep -c '\[error\]' "$log" || true)
  fi
  echo "$nwarn $nerr"
}

# ── 벤치 락 ───────────────────────────────────────────────────────────────────
# npu_guard 훅(.claude/hooks/npu_guard.sh)이 이 파일의 PID 로 "벤치가 돌고 있다"를 판단한다.
# 측정 중에 빌드·테스트가 끼어들면 CPU 시간 측정이 에러 없이 조용히 오염되기 때문이다.
# 훅이 PID 생존 여부를 보므로 비정상 종료로 남은 락은 자동으로 무효가 된다.
bench_lock_acquire() {
  BENCH_LOCK="${BENCH_LOCK:-$PROJ/build/.bench_running}"
  mkdir -p "$(dirname "$BENCH_LOCK")"
  printf '%s' "$$" > "$BENCH_LOCK"
  trap 'rm -f "$BENCH_LOCK"' EXIT INT TERM
}
