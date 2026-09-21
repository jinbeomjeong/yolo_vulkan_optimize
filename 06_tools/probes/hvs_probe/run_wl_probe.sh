#!/usr/bin/env bash
# hvs_wl_probe 변형 전체를 labwc 위에서 순서대로 돌린다 (데스크톱 세션 안에서, sudo 불필요).
#   결과: docs/hvs_review/evidence/wayland/<label>.log, <label>_frames.csv, <label>_state_f*.txt
# V3D 사용량은 시스템 전체 값(gpu_stats)이므로, 다른 V3D/NPU 작업(yvo_pipeline 등)이 돌고 있으면 끝날 때까지 기다린다.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
BIN=./build/hvs_wl_probe
OUT=docs/hvs_review/evidence/wayland
DETS=${DETS:-docs/hvs_review/evidence/baseline/disp_off_dets/detections.jsonl}
FRAMES=${FRAMES:-300}
mkdir -p "$OUT"

wait_quiet() {
    while pgrep -f 'bin/yvo_pipeline|build/yvo_pipeline' >/dev/null; do
        echo "[wait] 다른 yvo_pipeline 실행 중 — 대기"; sleep 10
    done
}

run() {
    local label=$1; shift
    wait_quiet
    echo "== $label: $*"
    timeout 120 "$BIN" --frames "$FRAMES" --out-dir "$OUT" --label "$label" "$@" > "$OUT/$label.log" 2>&1
    echo "   rc=$? $(grep -E '^RESULT_WL' "$OUT/$label.log")"
    grep -E '^\[state f' "$OUT/$label.log" | sed 's/^/   /'
    sleep 2
}

# 색: labwc 0.20.1 + wlroots 0.20.2+rpt3 는 COLOR_RANGE 를 뒤집어 쓴다(보고서 §3). "_lim" 은 정직한 요청,
#     "_fullhack" 은 range=full 로 요청해 커널에 limited 가 들어가게 만든 우회다.
run wl_fs_video_default      --overlay off
run wl_fs_video_lim          --overlay off --color-repr on --color-range limited
run wl_fs_video_fullhack     --overlay off --color-repr on --color-range full
run wl_fs_overlay_fullhack   --overlay on  --color-repr on --color-range full --dets "$DETS"
run wl_fs_overlay_dumb       --overlay on  --color-repr on --color-range full --dets "$DETS" --overlay-alloc dumb
run wl_fs_vp720_overlay      --overlay on  --color-repr on --color-range full --dets "$DETS" --viewport 1280x720
run wl_fs_vp720_video        --overlay off --color-repr on --color-range full --viewport 1280x720
run wl_win720_overlay        --overlay on  --color-repr on --color-range limited --dets "$DETS" --window 1280x720
run wl_win720_video          --overlay off --color-repr on --color-range limited --window 1280x720
