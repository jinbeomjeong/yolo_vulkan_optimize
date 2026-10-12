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
    so a warning there could never be fixed and would only bury real drift.
  - `.graphml` (LINES_GRAPHML): the text of `<data>` elements whose key is a node/edge `description`, and the visible
    text of `y:NodeLabel` / `y:EdgeLabel`. Each text line is judged like a doc line, reported as `<graphml>:<xml line>`.
    Differences from .md, because diagram text has no backticks: the path may be absolute (/home/.../01_src/x.cpp:12),
    a bare `:N` / `(:N)` after a space or '(' is a continuation, and the identifiers are the code-like words of the
    sentence (with '::', '->', '_', a camelCase hump, or a following '('), not backticked ones. A diagram text line
    is a paragraph, so a line is split into sentences; a miss is drift only when no citation of the sentence found
    any of its identifiers (otherwise the identifier is taken to belong to the other citation -> manual).
    These citations get their own summary line (`--- graphml: ...`), so the .md summary line stays comparable.
Every line it prints is a warning: `--lines` always exits 0 (like guessed/manual in `--stale`).
The summary counts every citation found: N = ok + drift + ambiguous + manual + external skipped.
"""
import os
import re
import sys
import xml.parsers.expat
from dataclasses import dataclass
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

# --lines on .graphml (see the docstring): absolute paths allowed, plain-text continuations, code-like words
LINES_GRAPHML = ("09_diagrams/**/*.graphml",)
CITE_G = re.compile(r"(?<![\w./+-])(/?(?:[\w.+-]+/)*[\w.+-]*\.(?:" + "|".join(CITE_EXTS) + r")):(\d+)(?:-(\d+))?(?!\d)")
CONT_G = re.compile(r"(?<=[\s(]):(\d+)(?:-(\d+))?(?![\d:])")
PATHISH = re.compile(r"\S*/\S*")
SENTENCE_G = re.compile(r"(?<=[.;])\s+")   # a diagram text line is a paragraph; identifiers are taken per sentence
WORD_G = re.compile(r"[A-Za-z_~]\w*(?:(?:::|->|\.)[A-Za-z_~]\w*)*(\()?")
GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
YFILES_LABELS = ("NodeLabel", "EdgeLabel")


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


def plain_identifiers(line: str) -> list[str]:
    """Code-like words of a plain-text line (.graphml). Paths, citations and file names are not ones.

    A word counts when it has '::', '->' or '_', a camelCase hump (VkQueue, FdCloser), or is followed by '('.
    Plain prose words (Vulkan, mutex) are left out: they would make almost every citation look checked.
    """
    text = PATHISH.sub(" ", CITE_G.sub(" ", line))
    found: list[str] = []
    for m in WORD_G.finditer(text):
        tok = m.group(0).rstrip("(")
        if "." in tok and tok.rsplit(".", 1)[1] in CITE_EXTS:  # a file name such as main.cpp
            continue
        if not (m.group(1) or "::" in tok or "->" in tok or "_" in tok or re.search(r"[a-z][A-Z]", tok)):
            continue
        names = {tok, re.split(r"::|\.|->", tok)[-1]}
        found.extend(n for n in sorted(names) if len(n) > 3 and n not in found)
    return found


def graphml_texts(path: Path) -> list[tuple[int, str]]:
    """(xml line, text line) of every node/edge description `<data>` and y:NodeLabel / y:EdgeLabel text.

    expat gives the line of each character-data chunk; a text keeps its newlines, so line k of a text that starts
    on xml line L is on line L + k. Only the element's own text is taken (not that of nested elements).
    """
    desc_keys: set[str] = set()
    out: list[tuple[int, str]] = []
    stack: list[list] = []   # [collect?, start line, chunks]
    parser = xml.parsers.expat.ParserCreate(namespace_separator=" ")

    def start(name: str, attrs: dict) -> None:
        ns, _, local = name.rpartition(" ")
        if ns == GRAPHML_NS and local == "key" and attrs.get("attr.name") == "description":
            desc_keys.add(attrs.get("id", ""))
        want = (ns == GRAPHML_NS and local == "data" and attrs.get("key") in desc_keys) or \
               (ns != GRAPHML_NS and local in YFILES_LABELS)
        stack.append([want, 0, []])

    def end(name: str) -> None:
        want, line, chunks = stack.pop()
        if want and chunks:
            out.extend((line + k, s) for k, s in enumerate("".join(chunks).split("\n")) if s.strip())

    def chars(data: str) -> None:
        if stack and stack[-1][0]:
            if not stack[-1][2]:
                stack[-1][1] = parser.CurrentLineNumber
            stack[-1][2].append(data)

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = chars
    parser.buffer_text = False   # keep CurrentLineNumber at the start of each chunk
    parser.Parse(path.read_bytes(), True)
    return out


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


@dataclass
class Tally:
    """Citation counts of one --lines scope. n = ok + drift + ambiguous + manual + external."""
    n: int = 0
    ok: int = 0
    drift: int = 0
    ambiguous: int = 0
    manual: int = 0
    external: int = 0

    def summary(self) -> str:
        return (f"{self.n} citations checked, {self.drift} drift (warning), {self.ambiguous} ambiguous, "
                f"{self.manual} manual, {self.external} external skipped")


def check_line(doc: Path, rel_doc: str, ln: int, cites: list[re.Match], idents: list[str], cont: int,
               files: list[str], cache: dict[Path, list[str]], t: Tally, shared: bool = False) -> None:
    """Judge the citations found on one doc line (one sentence of a .graphml text) and print the warnings.

    shared: when another citation of the unit found one of the identifiers, a miss is `manual`, not drift
    (the identifiers of a .graphml sentence are shared by all of its citations, like a `:N` continuation in .md).
    """
    t.n += cont
    t.manual += cont
    resolved = []
    for m in cites:
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) else a
        kind, val = resolve_cite(doc, m.group(1), files)
        resolved.append((m.group(0), kind, val, a, b))
    seen = set()
    misses: list[str] = []
    found = False
    for cite, kind, val, a, b in resolved:
        t.n += 1
        if kind == "external":
            t.external += 1
            continue
        if kind == "ambiguous":
            base = cite.split(":")[0].rsplit("/", 1)[-1]
            # the same citation also written with its full path on this line (link target) is enough
            if any(k == "file" and v.name == base and (x, y) == (a, b) for _, k, v, x, y in resolved):
                t.ok += 1
                continue
            t.ambiguous += 1
            print(f"ambiguous  {rel_doc}:{ln} -> {cite}  ({len(val)} candidates: {', '.join(val[:4])})")
            continue
        key = (val, a, b)
        if key in seen:  # duplicate on the same line (link text + link target)
            t.ok += 1
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
            t.drift += 1
            print(f"drift  {rel_doc}:{ln} -> {rel_f}:{a if a == b else f'{a}-{b}'}  "
                  f"line out of range (file has {len(src)} lines)")
            continue
        if not idents:
            t.manual += 1
            continue
        window = "\n".join(src[max(0, a - 1 - WINDOW):b + WINDOW])
        if any(re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", window) for x in idents):
            t.ok += 1
            found = True
            continue
        if cont:  # the identifier may belong to a `:N` continuation on this line, whose file is unknown
            t.manual += 1
            continue
        shown = "' / '".join(idents)
        msg = f"drift  {rel_doc}:{ln} -> {rel_f}:{a if a == b else f'{a}-{b}'}  '{shown}' not within ±{WINDOW} lines"
        if shared:
            misses.append(msg)   # judged once every citation of the unit has been tried
            continue
        t.drift += 1
        print(msg)
    for msg in misses:
        if found:
            t.manual += 1
            continue
        t.drift += 1
        print(msg)


def lines_mode() -> int:
    """Warn about file:line citations in the living docs that no longer point at the identifier they name."""
    files = repo_files()
    cache: dict[Path, list[str]] = {}
    md_tally, g_tally = Tally(), Tally()
    docs = sorted({p for pat in LINES_DOCS for p in ROOT.glob(pat) if p.is_file()})
    for md in docs:
        rel_md = md.relative_to(ROOT).as_posix()
        try:
            text = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            check_line(md, rel_md, ln, list(CITE.finditer(line)), identifiers(line), len(CONT_CITE.findall(line)),
                       files, cache, md_tally)
    graphs = sorted({p for pat in LINES_GRAPHML for p in ROOT.glob(pat) if p.is_file()})
    for g in graphs:
        rel_g = g.relative_to(ROOT).as_posix()
        try:
            texts = graphml_texts(g)
        except (xml.parsers.expat.ExpatError, OSError) as e:
            print(f"unreadable  {rel_g}  ({e})")
            continue
        for ln, line in texts:
            for sent in SENTENCE_G.split(line):
                check_line(g, rel_g, ln, list(CITE_G.finditer(sent)), plain_identifiers(sent),
                           len(CONT_G.findall(sent)), files, cache, g_tally, shared=True)
    print(f"--- {md_tally.summary()}")
    print(f"--- graphml: {g_tally.summary()} ({len(graphs)} files)")
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
            with p.open("rb") as fh:
                n = sum(1 for _ in fh)
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
