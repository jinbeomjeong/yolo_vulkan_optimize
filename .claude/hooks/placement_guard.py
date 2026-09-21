#!/usr/bin/env python3
"""PreToolUse(Write) 훅 본체. stdin 으로 훅 입력 JSON 을 받고, 차단할 때만 exit 2 + stderr.

새 파일 경로 하나를 06_tools/python/check_structure.py 의 classify_new_path() 로 판정한다.
파일 내용은 보지 않는다. 판정 순서(어느 단계든 예외·판정 불가면 통과 = fail-open):
  1. file_path 파싱 실패            → 통과
  2. 프로젝트 루트 밖 경로           → 통과 (~/.claude/plans, 스크래치패드 등)
  3. 이미 존재하는 파일              → 통과 (덮어쓰기는 편집으로 본다)
  4. build*/ · 11_logs/ 아래         → 통과
  5. YVO_PLACEMENT_OFF=1             → 통과
  6. classify_new_path(relpath)      → import 실패면 통과
  7. 위반이면 stderr 안내 + exit 2
"""
import fnmatch
import json
import os
import sys

UNGUARDED_TOPS = ("build*", "11_logs")


def project_root() -> str:
    if len(sys.argv) > 1 and sys.argv[1]:
        root = sys.argv[1]
    else:
        root = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.join(os.path.dirname(__file__), "..", "..")
    return os.path.realpath(root)


def judge(data: dict, root: str) -> str | None:
    """차단 메시지를 돌려준다. 통과면 None."""
    path = (data.get("tool_input") or {}).get("file_path")
    if not isinstance(path, str) or not path:
        return None                                              # 1
    if not os.path.isabs(path):
        path = os.path.join(data.get("cwd") or os.getcwd(), path)
    full = os.path.realpath(path)
    if os.path.commonpath([full, root]) != root or full == root:
        return None                                              # 2
    if os.path.lexists(path) or os.path.lexists(full):
        return None                                              # 3
    rel = os.path.relpath(full, root).replace(os.sep, "/")
    top = rel.split("/", 1)[0]
    if any(fnmatch.fnmatchcase(top, pat) for pat in UNGUARDED_TOPS):
        return None                                              # 4
    if os.environ.get("YVO_PLACEMENT_OFF") == "1":
        return None                                              # 5

    sys.path.insert(0, os.path.join(root, "06_tools", "python"))
    try:
        from check_structure import classify_new_path            # 6
    except ImportError:
        return None
    v = classify_new_path(rel)
    if v is None:
        return None
    rule_id, why, suggestion = v                                 # 7
    return (f"[placement_guard] 차단: {rel} — {rule_id} {why}. 올바른 위치: {suggestion}. "
            f"정본: STRUCTURE.md. 규칙이 틀렸다고 판단되면 사용자 확인 후 STRUCTURE.md 와 "
            f"06_tools/structure_rules.toml 을 함께 고친다(우회: YVO_PLACEMENT_OFF=1 로 세션 시작).\n")


def main() -> int:
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict):
            return 0
        msg = judge(data, project_root())
    except Exception as e:   # noqa: BLE001 — 판정 불가는 통과시킨다(fail-open). 원인만 남긴다.
        sys.stderr.write(f"[placement_guard] 판정 생략({type(e).__name__}: {e})\n")
        return 0
    if msg:
        sys.stderr.write(msg)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
