#!/usr/bin/env bash
# PreToolUse(Write) 훅: 새 파일이 STRUCTURE.md 의 배치 규칙을 어기면 만들기 전에 막는다.
#
# 판정은 placement_guard.py 가 06_tools/python/check_structure.py 의 classify_new_path() 로 한다.
# 이 래퍼는 프로젝트 루트만 계산해서 넘긴다(npu_guard.sh 와 같은 방식).
# 어떤 이유로든 판정할 수 없으면 통과시킨다(fail-open). 훅이 작업을 볼모로 잡으면 안 된다.
#
# 우회: YVO_PLACEMENT_OFF=1 로 세션을 시작한다.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj="${CLAUDE_PROJECT_DIR:-$(cd "$here/../.." && pwd)}"

PY=/usr/bin/python3
[[ -x "$PY" ]] || PY=python3

"$PY" "$here/placement_guard.py" "$proj"
rc=$?
# python 자체가 죽은 경우(1 등)는 통과로 바꾼다. 차단은 2 만 의미가 있다.
[[ $rc -eq 2 ]] && exit 2
exit 0
