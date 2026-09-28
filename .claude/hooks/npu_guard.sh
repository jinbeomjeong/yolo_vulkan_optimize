#!/usr/bin/env bash
# PreToolUse(Bash) hook: exclusive Hailo NPU access and measurement protection.
#
# This board has only one Hailo-8, yet CLAUDE.md encourages parallel subagents. So
#   (1) overlapping commands that open the NPU fail with HAILO_DEVICE_IN_USE, and
#   (2) a build or test sneaking in while a benchmark runs silently pollutes CPU-time measurements.
# (2) is more expensive: you only notice the odd values after finishing a 20-minute measurement.
#
# This wrapper only does cheap screening; npu_guard.py makes the decision. The hook is attached to every
# Bash call, so most commands must exit without spawning python.
#
# Bypass: prefix the command with YVO_HOOK_OFF=1.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj="${CLAUDE_PROJECT_DIR:-$(cd "$here/../.." && pwd)}"

input=$(cat)

# If none of the words below appear, pass immediately without JSON parsing.
if ! grep -qE 'yvo_pipeline|yvo_gst_pipeline|yvo_hailo_selftest|yvo_hailo_raw_probe|yvo_isp_pipeline|yvo_postproc_probe|test_detections|test_gst_detections|test_postproc|hailortcli|ctest|_bench\.sh|hailo_platform|--build|ninja|perf' <<<"$input"; then
    exit 0
fi

PY=/usr/bin/python3
[[ -x "$PY" ]] || PY=python3

printf '%s' "$input" | "$PY" "$here/npu_guard.py" "$proj/build/.bench_running"
