#!/usr/bin/env bash
# PreToolUse(Write) hook: blocks a new file before creation if it violates the placement rules in STRUCTURE.md.
#
# placement_guard.py decides using classify_new_path() from 06_tools/python/check_structure.py.
# This wrapper only computes the project root and passes it on (same approach as npu_guard.sh).
# If a decision cannot be made for any reason, let it through (fail-open). The hook must not hold work hostage.
#
# Bypass: start the session with YVO_PLACEMENT_OFF=1.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj="${CLAUDE_PROJECT_DIR:-$(cd "$here/../.." && pwd)}"

PY=/usr/bin/python3
[[ -x "$PY" ]] || PY=python3

"$PY" "$here/placement_guard.py" "$proj"
rc=$?
# If python itself died (1, etc.), turn it into a pass. Only 2 means block.
[[ $rc -eq 2 ]] && exit 2
exit 0
