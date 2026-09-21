#!/usr/bin/env bash
# Run all hvs_wl_probe variants in order on top of labwc (inside the desktop session, no sudo needed).
#   results: docs/hvs_review/evidence/wayland/<label>.log, <label>_frames.csv, <label>_state_f*.txt
# V3D usage is system-wide (gpu_stats), so wait until other V3D/NPU jobs (yvo_pipeline etc.) finish.
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
        echo "[wait] another yvo_pipeline is running — waiting"; sleep 10
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

# color: labwc 0.20.1 + wlroots 0.20.2+rpt3 inverts COLOR_RANGE when writing (report §3). "_lim" is the honest request,
#     "_fullhack" is the workaround that requests range=full so that limited reaches the kernel.
run wl_fs_video_default      --overlay off
run wl_fs_video_lim          --overlay off --color-repr on --color-range limited
run wl_fs_video_fullhack     --overlay off --color-repr on --color-range full
run wl_fs_overlay_fullhack   --overlay on  --color-repr on --color-range full --dets "$DETS"
run wl_fs_overlay_dumb       --overlay on  --color-repr on --color-range full --dets "$DETS" --overlay-alloc dumb
run wl_fs_vp720_overlay      --overlay on  --color-repr on --color-range full --dets "$DETS" --viewport 1280x720
run wl_fs_vp720_video        --overlay off --color-repr on --color-range full --viewport 1280x720
run wl_win720_overlay        --overlay on  --color-repr on --color-range limited --dets "$DETS" --window 1280x720
run wl_win720_video          --overlay off --color-repr on --color-range limited --window 1280x720
