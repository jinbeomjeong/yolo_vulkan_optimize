#!/usr/bin/env python3
"""Check that every markdown link in every `.md` in the repository exists.

Run after moving folders or editing docs. Runs on system python3 (no dependencies).

    python3 06_tools/python/check_links.py

Only **markdown links** of the form `](target)` are checked.
Paths inside backticks (prose shorthand like `op_point/summary.md`) use inconsistent bases, so they are not checked —
including them yields over 300 false positives and buries the real breakages.

For the `file:line` form, it also checks that the line number is within the file length.
Exit code is 1 if there is even one broken link.

    python3 06_tools/python/check_links.py --stale STRUCTURE.md

`--stale MAPFILE` checks, instead of links, whether any **old path** from the mapping table (relocate.py format) remains.
Target/exclusion rules and path resolution are the same as relocate.py (regex rows are judged by fullmatch per token, so
old paths are recognized even after the move). Exit code is 1 only if a mapped old path actually remains.
The `guessed` and `manual check` lines are warnings for a human to read: they are hints that often turn out to be
false positives (a history note in a comment, a glob), so they are printed but never affect the exit code.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "build", "build-asan", "__pycache__"}
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
LINE_SUFFIX = re.compile(r"\d+(-\d+)?$")


def check(md: Path, target: str, bad: list) -> bool:
    t = target.split("#", 1)[0]
    if not t or t.startswith(("http://", "https://", "mailto:")):
        return False
    line_no = None
    if ":" in t:
        head, _, tail = t.rpartition(":")
        if LINE_SUFFIX.fullmatch(tail):
            t, line_no = head, int(tail.split("-")[0])
    if not t or "*" in t or "<" in t:
        return False
    p = Path(t) if t.startswith("/") else md.parent / t
    if not p.exists():
        bad.append((md.relative_to(ROOT), target, "missing target"))
        return True
    if line_no is not None and p.is_file():
        try:
            n = sum(1 for _ in p.open("rb"))
        except OSError:
            return True
        if line_no > n:
            bad.append((md.relative_to(ROOT), target, f"line out of range (file has {n} lines)"))
    return True


def stale(map_file: str) -> int:
    """Report old paths from the mapping table that remain in non-excluded files."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import relocate  # same folder. Shares the mapping-table parser and token resolution

    mp = Path(map_file)
    if not mp.is_absolute():
        mp = mp if mp.exists() else ROOT / mp
    if not mp.exists():
        print(f"mapping table not found: {mp}")
        return 2
    try:
        map_rel = str(mp.resolve().relative_to(ROOT))
    except ValueError:
        map_rel = ""
    rows = relocate.parse_map(mp)
    if not rows:
        print("found no old-path table rows in the mapping table")
        return 2
    mapper = relocate.Mapper(rows)
    results, rw = relocate.scan_refs(mapper, map_rel)
    n = 0
    for rel, _, _, edits in results:
        for e in edits:
            print(f"old path remains  {rel}:{e.line} -> {e.old}  (new: {e.new})")
            n += 1
    for g in rw.guess:
        print(f"old path guessed  {g.file}:{g.line} -> {g.token}  ({g.why})")
    for g in rw.manual:
        print(f"manual check      {g.file}:{g.line} -> {g.token}  (glob/placeholder)")
    if rw.guess or rw.manual:
        print("warning: the guessed/manual lines above are hints, not failures - they do not affect the exit code.\n"
              "         Read them and judge by hand (e.g. a history note in a comment is a known false positive).")
    print(f"---\n{len(rows)} mapping rows, {n} old paths remaining, "
          f"{len(rw.guess)} guessed (warning), {len(rw.manual)} manual checks (warning)")
    return 1 if n else 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--stale":
        if len(sys.argv) != 3:
            print("usage: check_links.py --stale MAPFILE")
            return 2
        return stale(sys.argv[2])
    bad: list = []
    checked = 0
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP_DIRS for part in md.relative_to(ROOT).parts):
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in MD_LINK.finditer(text):
            if check(md, m.group(1), bad):
                checked += 1
    for f, t, why in bad:
        print(f"{why:16s} {f} -> {t}")
    print(f"---\nchecked {checked}, broken {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
