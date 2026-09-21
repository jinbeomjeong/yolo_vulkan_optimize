#!/usr/bin/env bash
# 영상 디코드 벤치마크: H.264 SW vs H.265 HW/SW, S1(디코드만) / S2(디코드 + 640x640 RGB letterbox)
#
# 사용법:
#   bash 06_tools/bench/decode_bench.sh                 # 전체 매트릭스, 조합당 3회
#   RUNS=1 bash 06_tools/bench/decode_bench.sh          # 반복 횟수 변경
#   ONLY='A_t0' bash 06_tools/bench/decode_bench.sh     # combo id에 대한 grep -E 패턴으로 일부만 실행
#   COOL_C=65 COOL_MAX_S=120 bash ...             # 실행 전 냉각 대기 조건
#   OUT_DIR=/path bash 06_tools/bench/decode_bench.sh   # 출력 디렉터리 변경(비어 있지 않으면 FORCE=1 필요)
#   ONLY='^h265_E_' bash 06_tools/bench/decode_bench.sh # ffmpeg HW HEVC 디코드 → 파이프 → GStreamer 조합(E)만 실행
#
# 출력: $OUT_DIR/raw/<combo>_run<N>/run.log (gst-launch -v 로그 중 caps/에러/마지막 줄만 뽑은 발췌라 run.log),
#       $OUT_DIR/results_raw.csv (실행별), $OUT_DIR/results.csv (조합별 중앙값·편차),
#       $OUT_DIR/env_bench.txt (측정 전 스냅샷 + 끝에 "# after" 스냅샷을 덧붙인다)
#   레이아웃 규약은 STRUCTURE.md §3. 2026-09-22 이전 결과는 raw/<combo>_run<N>.log 평면 파일,
#   env_before.txt·env_after.txt 였다(보관본은 raw/<combo>_run<N>/run.log(R1), env/(2026-09-21 표준화)로 옮겨져 있다).
#
# 측정 방식:
# - /usr/bin/time 이 설치되어 있지 않아, 동일한 정보(wait4 rusage: user, sys, maxrss)를
#   파이썬 래퍼가 os.wait4()로 직접 얻는다(= GNU time -v 와 같은 커널 값).
# - 프레임 수: sink를 silent=false 로 두고 gst-launch -v 가 출력하는 last-message(chain) 줄을
#   파이썬이 파이프로 읽어 센다. gst 프로세스 rusage에는 파이썬 쪽 CPU가 포함되지 않는다.
# - 파이프 조합(E, pipeline 필드가 "FFPIPE "로 시작): `bash -c "set -o pipefail; ffmpeg ... | gst-launch-1.0 -v ..."`
#   를 자식으로 실행하고 bash를 os.wait4()로 기다린다. bash가 파이프라인의 두 프로세스를 wait하므로
#   커널이 그 rusage를 bash에 누적해, wait4가 돌려주는 user/sys = bash + ffmpeg + gst-launch 합계가 된다.
#   max_rss_mb는 합계가 아니라 세 프로세스 중 최대값(커널 cmaxrss 규칙)이다. 실측으로 확인함.
#   종료 코드는 pipefail로 두 프로세스 중 하나라도 실패하면 0이 아니다.
#   ffmpeg는 -fps_mode passthrough를 준다. 없으면 rawvideo 출력의 CFR 처리로 프레임 1개가 복제된다(dup=1 → 1817).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 공통 함수(온도·PCIe·냉각·hailort 경고 판정·벤치 락)는 lib/common.sh 하나로 모았다.
# 예전에는 이 함수들이 7개 스크립트에 복붙돼 있었고 실제로 갈라져 있었다(pcie_link 의 Gen5 분기 등).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
VID="$PROJ/04_data/videos"
OUT_DIR="${OUT_DIR:-$PROJ/07_results/decode_benchmark}"
RUNS="${RUNS:-3}"
ONLY="${ONLY:-.}"
PY=/usr/bin/python3
COOL_C="${COOL_C:-65}"        # 각 실행 전 대기 목표 온도(°C)
COOL_MAX_S="${COOL_MAX_S:-120}"
check_only_exit
guard_out_dir   # OUT_DIR 이 비어 있지 않으면 FORCE=1 없이는 멈춘다(보관 결과 보호)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

H264="$VID/video_h264.mp4"
H265="$VID/video_h265.mp4"
URBAN="$VID/urban_street.mp4"
N264=1834
N265=1816
NURB=3000

S2CAPS='video/x-raw,format=RGB,width=640,height=640,pixel-aspect-ratio=1/1'

# combo_id|video|expected_frames|pipeline (sink 부분 포함)
# SINK_FS: fakesink(계획서 기준), SINK_FVS: fakevideosink(VideoMeta 지원, HW 보조 측정)
SINK_FS='fakesink name=sink sync=false silent=false'
SINK_FVS='fakevideosink name=sink sync=false silent=false'

combos=()
add() { combos+=("$1|$2|$3|$4"); }

src264="filesrc location=$H264 ! qtdemux ! h264parse"
src265="filesrc location=$H265 ! qtdemux ! h265parse"
srcurb="filesrc location=$URBAN ! qtdemux ! h264parse"

# --- H.264 1080p: A avdec_h264 (max-threads 0/1/2/4), B openh264dec
for t in 0 1 2 4; do
  add "h264_A_t${t}_S1"     "$H264" $N264 "$src264 ! avdec_h264 max-threads=$t ! $SINK_FS"
  for nt in 1 4; do
    add "h264_A_t${t}_S2_nt${nt}" "$H264" $N264 "$src264 ! avdec_h264 max-threads=$t ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS"
  done
done
add "h264_B_openh264_S1" "$H264" $N264 "$src264 ! openh264dec ! $SINK_FS"
for nt in 1 4; do
  add "h264_B_openh264_S2_nt${nt}" "$H264" $N264 "$src264 ! openh264dec ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS"
done

# --- H.265 1080p: C v4l2slh265dec (HW, 출력 NV12_128C8 SAND를 videoconvertscale가 직접 받음), D avdec_h265
add "h265_C_hw_S1"          "$H265" $N265 "$src265 ! v4l2slh265dec ! $SINK_FS"
add "h265_C_hw_S1_fvsink"   "$H265" $N265 "$src265 ! v4l2slh265dec ! $SINK_FVS"
for nt in 1 4; do
  add "h265_C_hw_S2_nt${nt}" "$H265" $N265 "$src265 ! v4l2slh265dec ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS"
done
add "h265_D_t0_S1" "$H265" $N265 "$src265 ! avdec_h265 max-threads=0 ! $SINK_FS"
for nt in 1 4; do
  add "h265_D_t0_S2_nt${nt}" "$H265" $N265 "$src265 ! avdec_h265 max-threads=0 ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS"
done

# --- urban_street 720p, 앞 3000프레임(sink num-buffers)
add "urban_A_t0_S1" "$URBAN" $NURB "$srcurb ! avdec_h264 max-threads=0 ! $SINK_FS num-buffers=$NURB"
for nt in 1 4; do
  add "urban_A_t0_S2_nt${nt}" "$URBAN" $NURB "$srcurb ! avdec_h264 max-threads=0 ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS num-buffers=$NURB"
done

# --- 참고: GStreamer 밖에서 HW HEVC 디코더 자체 처리량(ffmpeg -hwaccel drm, 프레임을 CPU로 내리지 않음)
add "h265_Cref_ffmpeg_drm_S1" "$H265" $N265 "FFMPEG"

# --- H.265 1080p: E ffmpeg HW 디코드(-hwaccel drm) → I420 rawvideo 파이프 → gst-launch(fdsrc)
#     "FFPIPE " 뒤는 gst 쪽 파이프라인. CPU는 ffmpeg + gst-launch 합산(측정 방식 참조).
ffsrc="FFPIPE fdsrc fd=0 blocksize=3110400 ! rawvideoparse width=1920 height=1080 format=i420 framerate=30/1"
add "h265_E_ffpipe_S1"     "$H265" $N265 "$ffsrc ! $SINK_FS"
add "h265_E_ffpipe_S2_nt4" "$H265" $N265 "$ffsrc ! videoconvertscale n-threads=4 add-borders=true ! $S2CAPS ! $SINK_FS"

env_snapshot() {
  {
    echo "date: $(date -Is)"
    echo "uname: $(uname -a)"
    echo "temp: $(vcgencmd measure_temp 2>&1)"
    echo "throttled: $(vcgencmd get_throttled 2>&1)"
    pcie_link
    echo "arm_clock: $(vcgencmd measure_clock arm 2>&1)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c | tr '\n' ' ')"
    echo "loadavg: $(cat /proc/loadavg)"
    echo "gst: $(gst-launch-1.0 --version | head -2 | tr '\n' ' ')"
    echo -n "hailo0 users (readable /proc only): "
    for p in /proc/[0-9]*; do ls -l "$p/fd" 2>/dev/null | grep -q hailo0 && echo -n "$(cat $p/comm)(${p#/proc/}) "; done; echo
    echo "hailo/vulkan processes: $(pgrep -a -f 'hailortcli|hailonet|vkcube|vulkan' | grep -v pgrep | tr '\n' ';')"
  }
}

# measure 스킬 §5: 새 측정은 env_bench.txt 하나에 스냅샷을 쓴다.
env_snapshot > "$OUT_DIR/env_bench.txt"

RAW_CSV="$OUT_DIR/results_raw.csv"
echo "combo,run,expected_frames,frames,frames_ok,exit_code,wall_s,user_s,sys_s,fps,avg_cores,cpu_pct_4core,max_rss_mb,temp_c_after" > "$RAW_CSV"

for c in "${combos[@]}"; do
  IFS='|' read -r id video expected pipe <<< "$c"
  echo "$id" | grep -qE "$ONLY" || continue
  for r in $(seq 1 "$RUNS"); do
    # 발열 영향을 줄이기 위해 SoC 온도가 COOL_C 이하가 될 때까지 대기.
    # 예전에는 여기에 자체 루프가 있었는데 온도를 못 읽으면 `|| echo 0` 으로 0 이 되어
    # **식지 않았는데도 즉시 통과**했고, 대기 시간도 기록하지 않았다(measure 스킬 §2 위반).
    cool_down "$id run$r" soc
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    mkdir -p "$run_dir"
    log="$run_dir/run.log"   # 발췌 로그(전문이 아니다)라 stdout.log 가 아니라 run.log
    "$PY" - "$id" "$r" "$expected" "$log" "$video" "$pipe" >> "$RAW_CSV" <<'PYEOF'
import os, re, shlex, subprocess, sys, time
cid, run, expected, log, video, pipe = sys.argv[1:7]
expected = int(expected)
if pipe == "FFMPEG":
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-hwaccel", "drm", "-i", video, "-f", "null", "-"]
elif pipe.startswith("FFPIPE "):
    # bash가 두 프로세스를 wait하므로 bash의 wait4 rusage에 ffmpeg + gst-launch CPU가 합산된다
    gst = pipe[len("FFPIPE "):]
    cmd = ["bash", "-c",
           "set -o pipefail; ffmpeg -hide_banner -loglevel error -nostdin -hwaccel drm -i " + shlex.quote(video)
           + " -an -fps_mode passthrough -f rawvideo -pix_fmt yuv420p - | gst-launch-1.0 -v " + gst]
else:
    cmd = ["gst-launch-1.0", "-v"] + pipe.split()
t0 = time.monotonic()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1 << 16)
frames = 0
keep = []          # caps·에러·요약 줄만 로그에 남긴다
tail = []
for raw in p.stdout:
    line = raw.decode("utf-8", "replace").rstrip("\n")
    if pipe == "FFMPEG":
        for seg in line.split("\r"):
            m = re.search(r"frame=\s*(\d+)", seg)
            if m:
                frames = int(m.group(1))
        keep.append(line[-400:])
        continue
    if "last-message = chain" in line:
        frames += 1
        continue
    if "last-message" in line:
        continue
    if "caps =" in line or "ERROR" in line or "WARN" in line or "Execution" in line or "EOS" in line:
        keep.append(line)
    tail.append(line)
    tail = tail[-10:]
_, status, ru = os.wait4(p.pid, 0)
wall = time.monotonic() - t0
rc = os.waitstatus_to_exitcode(status)
user, sys_ = ru.ru_utime, ru.ru_stime
cores = (user + sys_) / wall
fps = frames / wall
try:
    temp = re.search(r"([\d.]+)", subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout).group(1)
except Exception:
    temp = ""
with open(log, "w") as f:
    f.write("cmd: " + " ".join(cmd) + "\n")
    f.write("\n".join(keep) + "\n--- tail ---\n" + "\n".join(tail) + "\n")
    f.write(f"frames={frames} expected={expected} rc={rc} wall={wall:.3f} user={user:.3f} sys={sys_:.3f} maxrss_kb={ru.ru_maxrss}\n")
ok = int(frames == expected and rc == 0)
print(f"{cid},{run},{expected},{frames},{ok},{rc},{wall:.3f},{user:.3f},{sys_:.3f},{fps:.2f},{cores:.3f},{cores/4*100:.1f},{ru.ru_maxrss/1024:.1f},{temp}", flush=True)
PYEOF
    tail -n1 "$RAW_CSV" >&2
  done
done

{ echo; echo "# after"; env_snapshot; } >> "$OUT_DIR/env_bench.txt"

# 조합별 중앙값 요약
"$PY" - "$RAW_CSV" "$OUT_DIR/results.csv" <<'PYEOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    by.setdefault(r["combo"], []).append(r)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["combo", "runs", "expected_frames", "frames_all_ok", "fps_median", "fps_min", "fps_max",
                "fps_spread_pct", "avg_cores_median", "cpu_pct_4core_median", "max_rss_mb_median", "wall_s_median"])
    for cid, rs in by.items():
        fps = [float(r["fps"]) for r in rs]
        med = statistics.median(fps)
        spread = (max(fps) - min(fps)) / med * 100 if med > 0 else 0.0
        w.writerow([cid, len(rs), rs[0]["expected_frames"], int(all(r["frames_ok"] == "1" for r in rs)),
                    f"{med:.2f}", f"{min(fps):.2f}", f"{max(fps):.2f}", f"{spread:.1f}",
                    f"{statistics.median(float(r['avg_cores']) for r in rs):.3f}",
                    f"{statistics.median(float(r['cpu_pct_4core']) for r in rs):.1f}",
                    f"{statistics.median(float(r['max_rss_mb']) for r in rs):.1f}",
                    f"{statistics.median(float(r['wall_s']) for r in rs):.3f}"])
PYEOF
echo "done: $OUT_DIR/results.csv" >&2
