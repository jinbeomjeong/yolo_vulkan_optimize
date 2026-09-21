#!/usr/bin/env bash
# hvs_kms_probe 실행 절차 (사용자가 직접 실행한다).
#
# 이 스크립트는 디스플레이 매니저를 멈추거나 시작하지 않는다. VT 도 바꾸지 않는다.
# KMS 직접 경로는 card1 의 DRM master 가 필요한데, labwc(lightdm)가 떠 있으면 master 를 얻을 수 없다.
#
# 절차 (다른 PC 에서 SSH 로 접속해서 한다 — 화면이 잠깐 콘솔/검은 화면이 된다):
#   1) sudo systemctl stop lightdm          # labwc 종료 → card1 master 해제
#   2) bash tools/hvs_probe/run_kms_probe.sh   # sudo 없이 pi 사용자로 (video 그룹이면 card1 을 열 수 있다)
#   3) sudo systemctl start lightdm         # 데스크톱 복구
#
# 결과는 docs/hvs_review/evidence/kms/ 에 쌓인다. 화면을 직접 보는 사람이 있으면
# 각 단계에서 영상 색·박스가 정상인지 눈으로 확인하고 visual_notes.txt 에 적어 두면 보고서에 반영한다.
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
    echo "labwc/lightdm 이 실행 중이다. 먼저 'sudo systemctl stop lightdm' 을 실행한 뒤 다시 실행하라." >&2
    echo "(이 스크립트는 세션을 멈추지 않는다)" >&2
    exit 3
fi
for b in hvs_kms_probe hvs_caps hvs_psnr; do
    [ -x "$BIN/$b" ] || { echo "$BIN/$b 없음: cmake -S . -B build -DYVO_BUILD_HVS_PROBES=ON && cmake --build build" >&2; exit 2; }
done
[ -f "$DETS" ] || { echo "detections.jsonl 없음: $DETS (DETS=... 로 지정)" >&2; exit 2; }

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run_kms_probe.log"; }
log "start $(uname -r)"
"$BIN/hvs_caps" > "$OUT/hvs_caps_nomaster_session.txt" 2>&1
cp /sys/kernel/debug/dri/1/state "$OUT/state_before.txt" 2>/dev/null || true

# 한 변형을 돌린다. 비디오를 primary 에 올리는 TEST_ONLY 가 거부되면 오버레이 평면(107)으로 한 번 더 시도한다.
run_variant() {
    local label=$1; shift
    log "variant $label: $*"
    "$BIN/hvs_kms_probe" --video "$VIDEO" --dets "$DETS" --frames "$FRAMES" --out-dir "$OUT" --label "$label" "$@" \
        > "$OUT/$label.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] && grep -q "TEST_ONLY atomic.*: [^O]" "$OUT/$label.log"; then
        log "  primary 평면 TEST_ONLY 실패 → --video-plane 107 --overlay-plane 118 로 재시도"
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

# 색 확인: Writeback 커넥터로 HVS 출력(1:1, 오버레이 끔)을 캡처해 CPU 기준 변환(BT.709 limited)과 비교
log "writeback capture (frame 60, overlay off)"
"$BIN/hvs_kms_probe" --video "$VIDEO" --frames 90 --overlay off --writeback-at 60 --out-dir "$OUT" \
    --label kms_wb > "$OUT/kms_wb.log" 2>&1
log "  rc=$? $(grep -E '^\[wb\]' "$OUT/kms_wb.log" | tr '\n' ' ')"
REF=$(ls "$OUT"/kms_wb_f*_ref.ppm 2>/dev/null | head -1)
WB=$(ls "$OUT"/kms_wb_f*_writeback.ppm 2>/dev/null | head -1)
if [ -n "$REF" ] && [ -n "$WB" ]; then
    "$BIN/hvs_psnr" "$REF" "$WB" | tee "$OUT/psnr_writeback.txt"
    # PPM 은 6 MB 씩이라 비교 뒤 지운다(필요하면 KEEP_PPM=1)
    [ "${KEEP_PPM:-0}" = 1 ] || rm -f "$REF" "$WB"
else
    log "  writeback 캡처 실패 — kms_wb.log 참고 (색 검증은 debugfs state 의 color-encoding/range 로 대신)"
fi
cp /sys/kernel/debug/dri/1/state "$OUT/state_after.txt" 2>/dev/null || true
log "done. 이제 'sudo systemctl start lightdm' 으로 데스크톱을 복구하라."
