#!/usr/bin/env bash
# hvs_kms_probe run procedure (run by the user directly).
#
# This script does not stop or start the display manager. It does not switch VT either.
# The direct KMS path needs DRM master on card1, which cannot be obtained while labwc (lightdm) is running.
#
# Procedure (do it over SSH from another PC — the screen briefly becomes a console/black screen):
#   1) sudo systemctl stop lightdm          # stop labwc → release card1 master
#   2) bash tools/hvs_probe/run_kms_probe.sh   # as user pi without sudo (video group members can open card1)
#   3) sudo systemctl start lightdm         # restore desktop
#
# Results accumulate in docs/hvs_review/evidence/kms/. If someone is watching the screen,
# check by eye at each step whether video colors/boxes look right and note it in visual_notes.txt for the report.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BIN=./build
OUT=docs/hvs_review/evidence/kms
DETS=${DETS:-docs/hvs_review/evidence/baseline/disp_off_dets/detections.jsonl}
VIDEO=${VIDEO:-04_data/videos/video_h265.mp4}
FRAMES=${FRAMES:-300}
mkdir -p "$OUT"

if pgrep -x labwc >/dev/null || pgrep -x lightdm >/dev/null; then
    echo "labwc/lightdm is running. Run 'sudo systemctl stop lightdm' first, then run this again." >&2
    echo "(this script does not stop the session)" >&2
    exit 3
fi
for b in hvs_kms_probe hvs_caps hvs_psnr; do
    [ -x "$BIN/$b" ] || { echo "$BIN/$b missing: cmake -S . -B build -DYVO_BUILD_HVS_PROBES=ON && cmake --build build" >&2; exit 2; }
done
[ -f "$DETS" ] || { echo "detections.jsonl missing: $DETS (set with DETS=...)" >&2; exit 2; }

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run_kms_probe.log"; }
log "start $(uname -r)"
"$BIN/hvs_caps" > "$OUT/hvs_caps_nomaster_session.txt" 2>&1
cp /sys/kernel/debug/dri/1/state "$OUT/state_before.txt" 2>/dev/null || true

# Run one variant. If TEST_ONLY with video on primary is rejected, retry once with the overlay plane (107).
run_variant() {
    local label=$1; shift
    log "variant $label: $*"
    "$BIN/hvs_kms_probe" --video "$VIDEO" --dets "$DETS" --frames "$FRAMES" --out-dir "$OUT" --label "$label" "$@" \
        > "$OUT/$label.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] && grep -q "TEST_ONLY atomic.*: [^O]" "$OUT/$label.log"; then
        log "  primary plane TEST_ONLY failed → retrying with --video-plane 107 --overlay-plane 118"
        "$BIN/hvs_kms_probe" --video "$VIDEO" --dets "$DETS" --frames "$FRAMES" --out-dir "$OUT" \
            --label "${label}_ovl" --video-plane 107 --overlay-plane 118 "$@" > "$OUT/${label}_ovl.log" 2>&1
        rc=$?
        label="${label}_ovl"
    fi
    log "  rc=$rc  $(grep -E '^RESULT_KMS' "$OUT/$label.log" | tail -1)"
    sleep 2
}

run_variant kms_1to1_video_only   --overlay off
run_variant kms_1to1_overlay      --overlay on
run_variant kms_720_video_only    --overlay off --dst 1280x720
run_variant kms_720_overlay       --overlay on  --dst 1280x720

# Color check: capture HVS output (1:1, overlay off) via Writeback connector and compare with the CPU reference conversion (BT.709 limited)
log "writeback capture (frame 60, overlay off)"
"$BIN/hvs_kms_probe" --video "$VIDEO" --frames 90 --overlay off --writeback-at 60 --out-dir "$OUT" \
    --label kms_wb > "$OUT/kms_wb.log" 2>&1
log "  rc=$? $(grep -E '^\[wb\]' "$OUT/kms_wb.log" | tr '\n' ' ')"
REF=$(ls "$OUT"/kms_wb_f*_ref.ppm 2>/dev/null | head -1)
WB=$(ls "$OUT"/kms_wb_f*_writeback.ppm 2>/dev/null | head -1)
if [ -n "$REF" ] && [ -n "$WB" ]; then
    "$BIN/hvs_psnr" "$REF" "$WB" | tee "$OUT/psnr_writeback.txt"
    # PPMs are 6 MB each, so delete them after comparison (KEEP_PPM=1 to keep)
    [ "${KEEP_PPM:-0}" = 1 ] || rm -f "$REF" "$WB"
else
    log "  writeback capture failed — see kms_wb.log (color verification falls back to color-encoding/range in debugfs state)"
fi
cp /sys/kernel/debug/dri/1/state "$OUT/state_after.txt" 2>/dev/null || true
log "done. Now restore the desktop with 'sudo systemctl start lightdm'."
