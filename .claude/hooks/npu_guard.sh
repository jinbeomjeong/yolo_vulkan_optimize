#!/usr/bin/env bash
# PreToolUse(Bash) 훅: Hailo NPU 배타 접근과 측정 보호.
#
# 이 보드에는 Hailo-8 이 하나뿐인데 CLAUDE.md 는 병렬 서브에이전트를 권장한다. 그래서
#   (1) NPU 를 여는 명령이 겹치면 HAILO_DEVICE_IN_USE 가 나고,
#   (2) 벤치마크가 도는 중에 빌드·테스트가 끼어들면 CPU 시간 측정이 조용히 오염된다.
# (2) 가 더 비싸다. 20 분짜리 측정을 끝내고 나서야 값이 이상한 걸 알게 되기 때문이다.
#
# 이 래퍼는 값싼 선별만 하고, 판단은 npu_guard.py 가 한다. 훅은 모든 Bash 호출에 붙으므로
# 대부분의 명령이 python 을 띄우지 않고 빠져나가야 한다.
#
# 우회: 명령 앞에 YVO_HOOK_OFF=1 을 붙인다.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj="${CLAUDE_PROJECT_DIR:-$(cd "$here/../.." && pwd)}"

input=$(cat)

# 아래 낱말이 하나도 없으면 JSON 파싱 없이 즉시 통과한다.
if ! grep -qE 'yvo_pipeline|yvo_hailo_selftest|yvo_hailo_raw_probe|yvo_isp_pipeline|yvo_postproc_probe|test_detections|test_postproc|hailortcli|ctest|_bench\.sh|hailo_platform|--build|ninja|perf' <<<"$input"; then
    exit 0
fi

PY=/usr/bin/python3
[[ -x "$PY" ]] || PY=python3

printf '%s' "$input" | "$PY" "$here/npu_guard.py" "$proj/build/.bench_running"
