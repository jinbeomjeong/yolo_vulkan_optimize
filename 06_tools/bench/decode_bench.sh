#!/usr/bin/env bash
# Video decode benchmark: H.264 SW vs H.265 HW/SW, S1 (decode only) / S2 (decode + 640x640 RGB letterbox)
#
# Usage:
#   bash 06_tools/bench/decode_bench.sh                 # full matrix, 3 runs per combo
#   RUNS=1 bash 06_tools/bench/decode_bench.sh          # change repetition count
#   ONLY='A_t0' bash 06_tools/bench/decode_bench.sh     # run a subset via a grep -E pattern on combo id
#   COOL_C=65 COOL_MAX_S=120 bash ...             # pre-run cool-down conditions
#   OUT_DIR=/path bash 06_tools/bench/decode_bench.sh   # change output dir (FORCE=1 needed if not empty)
#   ONLY='^h265_E_' bash 06_tools/bench/decode_bench.sh # run only the ffmpeg HW HEVC decode -> pipe -> GStreamer combos (E)
#   ONLY='^h265_Cref_ffmpeg_' bash ...                  # ffmpeg CLI reference: baseline + the 4 output-format (outfmt) combos
#
# Opt-in combos (added with add_optin) are skipped when ONLY is not set, so a default full run keeps
# reproducing the original decode_benchmark matrix. They run only when an explicit ONLY pattern matches them.
#
# Output: $OUT_DIR/raw/<combo>_run<N>/run.log (named run.log since it is an excerpt of caps/error/last lines of the gst-launch -v log),
#       $OUT_DIR/results_raw.csv (per run), $OUT_DIR/results.csv (per-combo median/spread),
#       $OUT_DIR/env_bench.txt (pre-measurement snapshot + an "# after" snapshot appended at the end)
#   Layout convention is STRUCTURE.md §3. Results before 2026-09-22 were flat raw/<combo>_run<N>.log files and
#   env_before.txt/env_after.txt (archives were moved to raw/<combo>_run<N>/run.log (R1) and env/ (2026-09-21 standardization)).
#
# Measurement method:
# - /usr/bin/time is not installed, so the same information (wait4 rusage: user, sys, maxrss)
#   is obtained directly by a Python wrapper via os.wait4() (= same kernel values as GNU time -v).
# - Frame count: with the sink at silent=false, the last-message(chain) lines printed by gst-launch -v
#   are read over a pipe and counted by Python. The gst process rusage excludes Python-side CPU.
# - Pipe combos (E, pipeline field starts with "FFPIPE "): `bash -c "set -o pipefail; ffmpeg ... | gst-launch-1.0 -v ..."`
#   is run as a child and bash is awaited with os.wait4(). Since bash waits on both pipeline processes,
#   the kernel accumulates their rusage into bash, so user/sys from wait4 = bash + ffmpeg + gst-launch total.
#   max_rss_mb is not a sum but the max of the three processes (kernel cmaxrss rule). Verified empirically.
#   With pipefail, the exit code is non-zero if either process fails.
#   ffmpeg gets -fps_mode passthrough. Without it, CFR handling of rawvideo output duplicates one frame (dup=1 -> 1817).
# - ffmpeg CLI reference combos (pipeline field "FFMPEG" or "FFMPEG <pre-input args> [-- <post-input args>]"):
#   argv = ffmpeg -hide_banner -nostdin -hwaccel drm <pre-input args> -i <video> <post-input args> -f null -
#   Both segments are shlex-split; "--" separates input-side options (e.g. -hwaccel_output_format) from
#   output-side options (e.g. -vf). Plain "FFMPEG" is exactly the original reference command.
#   Frames are counted from the last "frame=" progress value on ffmpeg's stderr; run.log records the full argv.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Common functions (temperature, PCIe, cool-down, hailort warning verdict, bench lock) live in lib/common.sh.
# They used to be copy-pasted into 7 scripts and had actually diverged (e.g. pcie_link's Gen5 branch).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
VID="$PROJ/04_data/videos"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PROJ/07_results/decode_benchmark}")"   # absolute: runs cd into raw/<run>/
RUNS="${RUNS:-3}"
ONLY_SET="${ONLY:+1}"         # non-empty only when the caller passed an explicit ONLY pattern (gates opt-in combos)
ONLY="${ONLY:-.}"
PY=/usr/bin/python3
COOL_C="${COOL_C:-65}"        # target temperature before each run (°C)
COOL_MAX_S="${COOL_MAX_S:-120}"
check_only_exit
guard_out_dir   # stop if OUT_DIR is not empty unless FORCE=1 (archived-result protection)
bench_lock_acquire
mkdir -p "$OUT_DIR/raw"

H264="$VID/video_h264.mp4"
H265="$VID/video_h265.mp4"
URBAN="$VID/urban_street.mp4"
N264=1834
N265=1816
NURB=3000

S2CAPS='video/x-raw,format=RGB,width=640,height=640,pixel-aspect-ratio=1/1'

# combo_id|video|expected_frames|pipeline (including the sink part)
# SINK_FS: fakesink (per the plan), SINK_FVS: fakevideosink (VideoMeta support, auxiliary HW measurement)
SINK_FS='fakesink name=sink sync=false silent=false'
SINK_FVS='fakevideosink name=sink sync=false silent=false'

combos=()
optin_ids=" "
add() { combos+=("$1|$2|$3|$4"); }
# Same as add, but the combo runs only when an explicit ONLY pattern matches it (not in the default full run).
add_optin() { add "$@"; optin_ids+="$1 "; }

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

# --- H.265 1080p: C v4l2slh265dec (HW, output NV12_128C8 SAND taken directly by videoconvertscale), D avdec_h265
add "h265_C_hw_S1"          "$H265" $N265 "$src265 ! v4l2slh265dec ! $SINK_FS"
add "h265_C_hw_S1_fvsink"   "$H265" $N265 "$src265 ! v4l2slh265dec ! $SINK_FVS"
for nt in 1 4; do
  add "h265_C_hw_S2_nt${nt}" "$H265" $N265 "$src265 ! v4l2slh265dec ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS"
done
add "h265_D_t0_S1" "$H265" $N265 "$src265 ! avdec_h265 max-threads=0 ! $SINK_FS"
for nt in 1 4; do
  add "h265_D_t0_S2_nt${nt}" "$H265" $N265 "$src265 ! avdec_h265 max-threads=0 ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS"
done

# --- urban_street 720p, first 3000 frames (sink num-buffers)
add "urban_A_t0_S1" "$URBAN" $NURB "$srcurb ! avdec_h264 max-threads=0 ! $SINK_FS num-buffers=$NURB"
for nt in 1 4; do
  add "urban_A_t0_S2_nt${nt}" "$URBAN" $NURB "$srcurb ! avdec_h264 max-threads=0 ! videoconvertscale n-threads=$nt add-borders=true ! $S2CAPS ! $SINK_FS num-buffers=$NURB"
done

# --- Reference: HW HEVC decoder throughput outside GStreamer (ffmpeg -hwaccel drm, -f null).
#     Baseline with the output format left unspecified: ffmpeg then downloads SAND frames to CPU memory as
#     yuv420p (~1.5 of its 2.0 ms/frame CPU; 07_results/decode_benchmark/runs/ffmpeg_outfmt, 2026-09-29).
add "h265_Cref_ffmpeg_drm_S1" "$H265" $N265 "FFMPEG"
#     outfmt (opt-in): drm_prime keeps frames as DRM PRIME (no download); the others add an explicit
#     hwdownload (+ format conversion). The unsand combo fails in rpi-ffmpeg 7.1.5+rpt2 (filter declares 2 inputs).
ffdrm="FFMPEG -hwaccel_output_format drm_prime"
add_optin "h265_Cref_ffmpeg_outfmt_drmprime_S1"    "$H265" $N265 "$ffdrm"
add_optin "h265_Cref_ffmpeg_outfmt_unsand_S1"      "$H265" $N265 "$ffdrm -- -vf hwdownload,format=sand128,unsand,format=yuv420p"
add_optin "h265_Cref_ffmpeg_outfmt_dl_nv12_S1"     "$H265" $N265 "$ffdrm -- -vf hwdownload,format=nv12"
add_optin "h265_Cref_ffmpeg_outfmt_dl_yuv420p_S1"  "$H265" $N265 "$ffdrm -- -vf hwdownload,format=yuv420p"

# --- H.265 1080p: E ffmpeg HW decode (-hwaccel drm) -> I420 rawvideo pipe -> gst-launch (fdsrc)
#     After "FFPIPE " is the gst-side pipeline. CPU is ffmpeg + gst-launch summed (see measurement method).
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

# measure skill §5: new measurements write snapshots to a single env_bench.txt.
env_snapshot > "$OUT_DIR/env_bench.txt"

RAW_CSV="$OUT_DIR/results_raw.csv"
echo "combo,run,expected_frames,frames,frames_ok,exit_code,wall_s,user_s,sys_s,fps,avg_cores,cpu_pct_4core,max_rss_mb,temp_c_after" > "$RAW_CSV"

for c in "${combos[@]}"; do
  IFS='|' read -r id video expected pipe <<< "$c"
  echo "$id" | grep -qE "$ONLY" || continue
  [[ -z "$ONLY_SET" && "$optin_ids" == *" $id "* ]] && continue   # opt-in combos need an explicit ONLY
  for r in $(seq 1 "$RUNS"); do
    # Wait until the SoC temperature is at or below COOL_C to reduce thermal effects.
    # There used to be an inline loop here; if the temperature couldn't be read, `|| echo 0` made it 0, so it
    # **passed immediately without cooling** and didn't record the wait time (violating measure skill §2).
    cool_down "$id run$r" soc
    run_dir="$OUT_DIR/raw/${id}_run${r}"
    mkdir -p "$run_dir"
    log="$run_dir/run.log"   # excerpt log (not the full output), hence run.log rather than stdout.log
    "$PY" - "$id" "$r" "$expected" "$log" "$video" "$pipe" >> "$RAW_CSV" <<'PYEOF'
import os, re, shlex, subprocess, sys, time
cid, run, expected, log, video, pipe = sys.argv[1:7]
expected = int(expected)
is_ffmpeg = pipe == "FFMPEG" or pipe.startswith("FFMPEG ")
if is_ffmpeg:
    # "FFMPEG [<pre-input args>] [-- <post-input args>]"; plain "FFMPEG" gives the original reference argv
    toks = shlex.split(pipe[len("FFMPEG"):])
    pre, post = (toks[:toks.index("--")], toks[toks.index("--") + 1:]) if "--" in toks else (toks, [])
    cmd = (["ffmpeg", "-hide_banner", "-nostdin", "-hwaccel", "drm"] + pre + ["-i", video] + post
           + ["-f", "null", "-"])
elif pipe.startswith("FFPIPE "):
    # bash waits on both processes, so bash's wait4 rusage sums ffmpeg + gst-launch CPU
    gst = pipe[len("FFPIPE "):]
    cmd = ["bash", "-c",
           "set -o pipefail; ffmpeg -hide_banner -loglevel error -nostdin -hwaccel drm -i " + shlex.quote(video)
           + " -an -fps_mode passthrough -f rawvideo -pix_fmt yuv420p - | gst-launch-1.0 -v " + gst]
else:
    cmd = ["gst-launch-1.0", "-v"] + pipe.split()
t0 = time.monotonic()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1 << 16)
frames = 0
keep = []          # keep only caps/error/summary lines in the log
tail = []
for raw in p.stdout:
    line = raw.decode("utf-8", "replace").rstrip("\n")
    if is_ffmpeg:
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

# Per-combo median summary
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
