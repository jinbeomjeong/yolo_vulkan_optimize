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

    python3 06_tools/python/check_links.py --lines

`--lines` checks that `path:line` / `path:a-b` citations in prose (inside backticks or link text, which the default
mode never sees) still point at what the sentence talks about. For each citation it looks for the backticked
identifiers on the same doc line (e.g. `read_candidates()` -> read_candidates) within +-5 lines of the cited lines.
  - resolution: full path from the repo root or the doc's folder; otherwise the path is matched as a suffix of the
    repo files (a bare file name included) and used only if exactly one file matches. Several matches -> ambiguous.
    No match -> an outside source (HailoRT, GStreamer, ...), skipped silently.
  - drift: none of the identifiers is near the cited lines, or the cited line is past the end of the file.
  - manual: a continuation citation (`:856`, the file is not guessed) or a citation with no identifier on its line.
    A miss on a line that also has a continuation citation is manual too (the identifier may be the latter's).
  - Scope is the living docs only (LINES_DOCS). `07_results/**` is excluded on purpose: a result doc cites the code
    as it was when it was measured (that HEAD is in env_bench.txt), and result files are never edited in this repo,
    so a warning there could never be fixed and would only bury real drift. `.graphml` descriptions are not read.
Every line it prints is a warning: `--lines` always exits 0 (like guessed/manual in `--stale`).
The summary counts every citation found: N = ok + drift + ambiguous + manual + external skipped.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "build", "build-asan", "__pycache__"}
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
LINE_SUFFIX = re.compile(r"\d+(-\d+)?$")

# --lines: living docs (glob patterns relative to ROOT). 07_results is excluded on purpose (see the docstring).
LINES_DOCS = ("08_learn/**/*.md", "README.md", "09_diagrams/**/*.md", "10_paper/**/*.md", "CLAUDE.md",
              "STRUCTURE.md", ".claude/**/*.md", "06_tools/README.md")
CITE_EXTS = ("cpp", "hpp", "cc", "cxx", "c", "h", "hh", "inl", "comp", "glsl", "py", "sh", "md", "txt", "toml",
             "cmake", "json", "yml", "yaml", "graphml", "csv")
CITE = re.compile(r"(?<![\w./+-])((?:[\w.+-]+/)*[\w.+-]*\.(?:" + "|".join(CITE_EXTS) + r")):(\d+)(?:-(\d+))?(?!\d)")
CONT_CITE = re.compile(r"`:(\d+)(?:-(\d+))?`")
BACKTICK = re.compile(r"`([^`]+)`")
IDENT = re.compile(r"[A-Za-z_~]\w*(?:(?:::|\.|->)[A-Za-z_~]\w*)*")
WINDOW = 5


def repo_files() -> list[str]:
    """Repo-relative paths of every file outside build output and VCS folders (for suffix resolution)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith("build")]
        rel = Path(dirpath).relative_to(ROOT)
        out.extend((rel / f).as_posix() for f in filenames)
    return out


def identifiers(line: str) -> list[str]:
    """Backticked identifiers on a doc line. Paths, citations, numbers and tokens of 3 chars or less are not ones."""
    found = []
    for m in BACKTICK.finditer(line):
        tok = m.group(1).strip()
        if "(" in tok and tok.endswith(")"):
            tok = tok[:tok.index("(")]
        if CITE.search(tok) or "/" in tok or not IDENT.fullmatch(tok):
            continue
        if "." in tok and tok.rsplit(".", 1)[1] in CITE_EXTS:  # a file name such as main.cpp
            continue
        names = {tok, re.split(r"::|\.|->", tok)[-1]}  # Class::method also matches a bare `method(` in a header
        found.extend(n for n in names if len(n) > 3 and n not in found)
    return found


def resolve_cite(md: Path, path: str, files: list[str]) -> tuple[str, Path | list[str] | None]:
    """('file', Path) | ('ambiguous', candidates) | ('external', None)."""
    for p in ([Path(path)] if path.startswith("/") else [md.parent / path, ROOT / path]):
        try:
            p = p.resolve()
            p.relative_to(ROOT)
        except (OSError, ValueError):
            continue
        if p.is_file():
            return "file", p
    parts = [s for s in path.split("/") if s not in ("", ".", "..")]
    suffix = "/".join(parts)
    hits = [f for f in files if f == suffix or f.endswith("/" + suffix)]
    if len(hits) == 1:
        return "file", ROOT / hits[0]
    if hits:
        return "ambiguous", hits
    return "external", None


def lines_mode() -> int:
    """Warn about file:line citations in the living docs that no longer point at the identifier they name."""
    files = repo_files()
    docs = sorted({p for pat in LINES_DOCS for p in ROOT.glob(pat) if p.is_file()})
    cache: dict[Path, list[str]] = {}
    n = ok = drift = ambiguous = manual = external = 0
    for md in docs:
        rel_md = md.relative_to(ROOT).as_posix()
        try:
            text = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            cont = len(CONT_CITE.findall(line))
            n += cont
            manual += cont
            idents = identifiers(line)
            resolved, pending = [], []
            for m in CITE.finditer(line):
                a = int(m.group(2))
                b = int(m.group(3)) if m.group(3) else a
                kind, val = resolve_cite(md, m.group(1), files)
                resolved.append((m.group(0), kind, val, a, b))
            seen = set()
            for cite, kind, val, a, b in resolved:
                n += 1
                if kind == "external":
                    external += 1
                    continue
                if kind == "ambiguous":
                    base = cite.split(":")[0].rsplit("/", 1)[-1]
                    # the same citation also written with its full path on this line (link target) is enough
                    if any(k == "file" and v.name == base and (x, y) == (a, b) for _, k, v, x, y in resolved):
                        ok += 1
                        continue
                    ambiguous += 1
                    print(f"ambiguous  {rel_md}:{ln} -> {cite}  ({len(val)} candidates: {', '.join(val[:4])})")
                    continue
                key = (val, a, b)
                if key in seen:  # duplicate on the same line (link text + link target)
                    ok += 1
                    continue
                seen.add(key)
                rel_f = val.relative_to(ROOT).as_posix()
                if val not in cache:
                    try:
                        cache[val] = val.read_text(encoding="utf-8", errors="replace").splitlines()
                    except OSError:
                        cache[val] = []
                src = cache[val]
                if b > len(src):
                    drift += 1
                    print(f"drift  {rel_md}:{ln} -> {rel_f}:{a if a == b else f'{a}-{b}'}  "
                          f"line out of range (file has {len(src)} lines)")
                    continue
                if not idents:
                    manual += 1
                    continue
                window = "\n".join(src[max(0, a - 1 - WINDOW):b + WINDOW])
                if any(re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", window) for t in idents):
                    ok += 1
                    continue
                if cont:  # the identifier may belong to a `:N` continuation on this line, whose file is unknown
                    manual += 1
                    continue
                drift += 1
                shown = "' / '".join(idents)
                print(f"drift  {rel_md}:{ln} -> {rel_f}:{a if a == b else f'{a}-{b}'}  "
                      f"'{shown}' not within ±{WINDOW} lines")
    print(f"--- {n} citations checked, {drift} drift (warning), {ambiguous} ambiguous, {manual} manual, "
          f"{external} external skipped")
    return 0


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
    if len(sys.argv) >= 2 and sys.argv[1] == "--lines":
        if len(sys.argv) != 2:
            print("usage: check_links.py --lines")
            return 2
        return lines_mode()
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
