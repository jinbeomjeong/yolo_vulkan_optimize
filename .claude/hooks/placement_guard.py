#!/usr/bin/env python3
"""PreToolUse(Write) hook body. Reads the hook input JSON from stdin; exits 2 + stderr only when blocking.

Judges a single new file path with classify_new_path() from 06_tools/python/check_structure.py.
Does not look at file contents. Decision order (an exception or undecidable at any step = pass, fail-open):
  1. file_path parse failure        -> pass
  2. path outside project root      -> pass (~/.claude/plans, scratchpad, etc.)
  3. file already exists            -> pass (overwriting counts as editing)
  4. under build*/ or 11_logs/      -> pass
  5. YVO_PLACEMENT_OFF=1            -> pass
  6. classify_new_path(relpath)     -> pass if import fails
  7. on violation, stderr guidance + exit 2
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
    """Return the block message, or None to pass."""
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
    return (f"[placement_guard] blocked: {rel} — {rule_id} {why}. Correct location: {suggestion}. "
            f"Source of truth: STRUCTURE.md. If you believe the rule is wrong, confirm with the user, then fix STRUCTURE.md and "
            f"06_tools/structure_rules.toml together (bypass: start the session with YVO_PLACEMENT_OFF=1).\n")


def main() -> int:
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict):
            return 0
        msg = judge(data, project_root())
    except Exception as e:   # noqa: BLE001 — undecidable means pass (fail-open). Only record the cause.
        sys.stderr.write(f"[placement_guard] decision skipped ({type(e).__name__}: {e})\n")
        return 0
    if msg:
        sys.stderr.write(msg)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
