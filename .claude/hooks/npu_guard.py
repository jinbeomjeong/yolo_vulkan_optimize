#!/usr/bin/env python3
"""PreToolUse(Bash) 훅 본체. stdin 으로 훅 입력 JSON 을 받고, 차단할 때만 exit 2 + stderr.

점유 판정
  - NPU : fuser -s /dev/hailo0 (장치 기준. root 권한 없이 동작하는 것을 확인했다)
  - 벤치 : <project>/build/.bench_running 의 PID 생존 여부 (06_tools/bench/*_bench.sh 가 만든다)
pgrep -f 로 프로세스 이름을 찾으면 안 된다. 패턴 문자열이 호출자 셸의 명령줄에 들어 있어
자기 자신을 매칭한다(실측 확인).
"""
import glob
import json
import os
import re
import subprocess
import sys

# NPU 를 여는 명령. 실행파일은 01_src/tools/CMakeLists.txt·01_src/pipeline·05_tests 에서 yvo_hailo(또는
# yvo_stages)를 링크하고 실제로 VDevice 를 여는 것만 넣었다. test_postproc 은 PostprocLive.* 가 NPU 를 연다.
NPU_CMDS = {"yvo_pipeline", "yvo_hailo_selftest", "yvo_hailo_raw_probe", "yvo_isp_pipeline",
            "hailortcli", "test_detections", "test_postproc", "ctest"}
# yvo_postproc_probe 는 모드에 따라 다르다(postproc_probe.cpp main()):
#   live·stale·v3 → VDevice 를 연다(NPU 독점)
#   gpuab         → 픽스처만 쓰지만 GPU/CPU 시간을 재는 측정이라 부하로 분류한다
#   fixture(기본)·sweep·heapimport → NPU 불필요
POSTPROC_PROBE = "yvo_postproc_probe"
POSTPROC_NPU_MODES = {"live", "stale", "v3"}
POSTPROC_LOAD_MODES = {"gpuab"}
_PROBE_MODE_RE = re.compile(r"--probe(?:\s+|=)(\S+)|--(live|stale|fixture|sweep|heapimport)\b")

# 벤치 목록은 06_tools/bench/*_bench.sh 글롭으로 찾는다. 루트 계산이 실패하면 아래 목록으로 폴백한다.
_BENCH_FALLBACK = ("pipeline_bench.sh", "decode_bench.sh", "op_point_bench.sh",
                   "display_bench.sh", "backpressure_bench.sh",
                   "isp_bench.sh", "isp_pipeline_bench.sh")


def _bench_scripts():
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    try:
        found = tuple(sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "06_tools", "bench", "*_bench.sh"))))
    except OSError:
        found = ()
    return found or _BENCH_FALLBACK


BENCH_SCRIPTS = _bench_scripts()

HOW = ("\n다시 되는 시점: 위 작업이 끝나면 바로 통과한다. 진행 상황은 그 작업의 로그로 확인할 것."
       "\n꼭 지금 실행해야 하면 명령 앞에 YVO_HOOK_OFF=1 을 붙인다(측정값은 신뢰할 수 없게 된다).")


def leading_commands(shell_cmd):
    """셸 문자열을 명령 단위로 쪼개고 (선두 토큰의 basename, 조각) 을 돌려준다.
    앞에 붙는 환경변수 대입·sudo·time·괄호는 벗겨 낸다. 그래서 `grep yvo_pipeline x` 는
    선두 토큰이 grep 이라 걸리지 않는다."""
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
    """yvo_postproc_probe 조각에서 실제로 적용될 모드. 인자는 앞에서부터 덮어쓰므로 마지막 것이 이긴다."""
    mode = "fixture"
    for m in _PROBE_MODE_RE.finditer(seg):
        mode = m.group(1) or m.group(2)
    return mode


def classify(shell_cmd):
    """(npu_hit, load_hit) 반환. 각각 걸린 명령 이름 또는 None."""
    npu_hit = load_hit = None
    for base, seg in leading_commands(shell_cmd):
        if base in NPU_CMDS:
            npu_hit = base
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
        return False   # 판정 불가면 막지 않는다


def main():
    marker = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    except Exception:
        return 0       # 파싱 실패 시 통과시킨다. 훅이 작업을 볼모로 잡으면 안 된다.
    if not cmd or re.match(r"\s*YVO_HOOK_OFF=1\b", cmd):
        return 0

    npu_hit, load_hit = classify(cmd)
    if not npu_hit and not load_hit:
        return 0

    pid = live_bench_pid(marker)
    if pid:
        what = npu_hit or load_hit
        sys.stderr.write(
            f"[npu_guard] 차단: 벤치마크가 실행 중이다(PID {pid}).\n"
            f"요청한 명령({what})은 CPU/NPU 를 써서 측정값을 오염시킨다. "
            f"이 프로젝트는 cpu_ms_per_frame 을 재므로 동시 부하가 곧 오측이다.{HOW}\n")
        return 2

    if npu_hit and npu_busy():
        sys.stderr.write(
            f"[npu_guard] 차단: /dev/hailo0 을 다른 프로세스가 이미 열고 있다.\n"
            f"요청한 명령({npu_hit})도 NPU 를 열려 하므로 HAILO_DEVICE_IN_USE 로 실패한다. "
            f"Hailo-8 은 이 보드에 하나뿐이다.{HOW}\n")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
