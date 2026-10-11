#!/usr/bin/env python3
"""PreToolUse(Bash) hook body. Reads the hook input JSON from stdin; exits 2 + stderr only when blocking.

Occupancy checks
  - NPU   : fuser -s /dev/hailo0 (device-based; verified to work without root privileges)
  - bench : whether the PID in <project>/build/.bench_running is alive (created by 06_tools/bench/*_bench.sh)
Do not look up process names with pgrep -f. The pattern string is part of the caller shell's command line,
so it matches itself (verified empirically).
"""
import glob
import json
import os
import re
import subprocess
import sys

# Commands that open the NPU. Only executables that link yvo_hailo (or yvo_stages) in 01_src/tools/CMakeLists.txt,
# 01_src/pipeline or 05_tests and actually open a VDevice are listed. test_postproc opens the NPU via PostprocLive.*.
NPU_CMDS = {"yvo_pipeline", "yvo_hailo_selftest", "yvo_hailo_raw_probe", "yvo_isp_pipeline",
            "hailortcli", "test_detections", "test_gst_detections", "test_postproc", "ctest"}
# yvo_gst_pipeline depends on --sink (gst_pipeline.cpp): hailo (default) opens a VDevice; fake|hold only
# decode + run the Vulkan letterbox, so they are classified as load.
GST_PIPELINE = "yvo_gst_pipeline"
_GST_SINK_RE = re.compile(r"--sink(?:\s+|=)(\S+)")
# yvo_postproc_probe depends on the mode (postproc_probe.cpp main()):
#   live, stale, v3 -> opens a VDevice (exclusive NPU)
#   gpuab, nms, stages, sweep -> uses only fixtures, but measures GPU/CPU time, so it is classified as load
#   fixture (default), heapimport, topk -> no NPU needed
POSTPROC_PROBE = "yvo_postproc_probe"
POSTPROC_NPU_MODES = {"live", "stale", "v3"}
POSTPROC_LOAD_MODES = {"gpuab", "nms", "stages", "sweep"}
_PROBE_MODE_RE = re.compile(r"--probe(?:\s+|=)(\S+)|--(live|stale|fixture|sweep|heapimport)\b")

# The bench list is found by globbing 06_tools/bench/*_bench.sh. If root resolution fails, fall back to the list below.
_BENCH_FALLBACK = ("pipeline_bench.sh", "decode_bench.sh", "op_point_bench.sh",
                   "display_bench.sh", "display_planes_bench.sh", "backpressure_bench.sh",
                   "isp_bench.sh", "isp_pipeline_bench.sh", "gst_pipeline_bench.sh", "dec_threads_bench.sh", "nms_bench.sh")


def _bench_scripts():
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    try:
        found = tuple(sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "06_tools", "bench", "*_bench.sh"))))
    except OSError:
        found = ()
    return found or _BENCH_FALLBACK


BENCH_SCRIPTS = _bench_scripts()

HOW = ("\nWhen it works again: passes as soon as the job above finishes. Check its progress in that job's log."
       "\nIf you must run it now, prefix the command with YVO_HOOK_OFF=1 (measurements become unreliable).")


def leading_commands(shell_cmd):
    """Split a shell string into commands and yield (basename of leading token, segment).
    Leading env assignments, sudo, time and parentheses are stripped. So `grep yvo_pipeline x`
    is not caught, because its leading token is grep."""
    for seg in re.split(r"(?:\|\||&&|[;\n|&])", shell_cmd):
        seg = seg.strip().lstrip("(").lstrip()
        while True:
            m = re.match(r"(?:\w+=\S*|sudo|time|nohup|exec|command)\s+", seg)
            if not m:
                break
            seg = seg[m.end():]
        parts = seg.split()
        if parts:
            yield os.path.basename(parts[0]), seg


def postproc_probe_mode(seg):
    """Effective mode for a yvo_postproc_probe segment. Arguments override left to right, so the last one wins."""
    mode = "fixture"
    for m in _PROBE_MODE_RE.finditer(seg):
        mode = m.group(1) or m.group(2)
    return mode


def classify(shell_cmd):
    """Return (npu_hit, load_hit): the matched command name for each, or None."""
    npu_hit = load_hit = None
    for base, seg in leading_commands(shell_cmd):
        if base in NPU_CMDS:
            npu_hit = base
        elif base == GST_PIPELINE:
            sinks = _GST_SINK_RE.findall(seg)
            sink = sinks[-1] if sinks else "hailo"
            if sink in ("fake", "hold"):
                load_hit = f"{base} --sink {sink}"
            else:
                npu_hit = f"{base} --sink {sink}"
        elif base == POSTPROC_PROBE:
            mode = postproc_probe_mode(seg)
            if mode in POSTPROC_NPU_MODES:
                npu_hit = f"{base} --probe {mode}"
            elif mode in POSTPROC_LOAD_MODES:
                load_hit = f"{base} --probe {mode}"
        elif base == "bash" or base.endswith(".sh"):
            if any(b in seg for b in BENCH_SCRIPTS):
                npu_hit = "bench script"
        elif base in ("python", "python3") and "hailo_platform" in seg:
            npu_hit = base
        elif base == "cmake" and "--build" in seg:
            load_hit = "cmake --build"
        elif base == "ninja":
            load_hit = "ninja"
        elif base == "perf" and re.search(r"\b(record|stat)\b", seg):
            load_hit = "perf"
    return npu_hit, load_hit


def live_bench_pid(marker):
    try:
        pid = int(open(marker).read().strip())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def npu_busy():
    try:
        return subprocess.run(["fuser", "-s", "/dev/hailo0"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except Exception:
        return False   # if undecidable, do not block


def main():
    marker = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    except Exception:
        return 0       # pass on parse failure. The hook must not hold work hostage.
    if not cmd or re.match(r"\s*YVO_HOOK_OFF=1\b", cmd):
        return 0

    npu_hit, load_hit = classify(cmd)
    if not npu_hit and not load_hit:
        return 0

    pid = live_bench_pid(marker)
    if pid:
        what = npu_hit or load_hit
        sys.stderr.write(
            f"[npu_guard] blocked: a benchmark is running (PID {pid}).\n"
            f"The requested command ({what}) uses CPU/NPU and would pollute measurements. "
            f"This project measures cpu_ms_per_frame, so concurrent load means a wrong measurement.{HOW}\n")
        return 2

    if npu_hit and npu_busy():
        sys.stderr.write(
            f"[npu_guard] blocked: another process already has /dev/hailo0 open.\n"
            f"The requested command ({npu_hit}) also tries to open the NPU and would fail with HAILO_DEVICE_IN_USE. "
            f"There is only one Hailo-8 on this board.{HOW}\n")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
