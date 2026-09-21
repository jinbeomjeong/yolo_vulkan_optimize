#!/usr/bin/env python3
"""Move files according to a mapping table (old path -> new path) and rewrite references in the repo to the new paths.

Runs on the system python3 (no dependencies). Every operation is idempotent — running it twice gives the same result.

    python3 06_tools/python/relocate.py --dry-run                     # default mapping table STRUCTURE.md
    python3 06_tools/python/relocate.py --map M.md --section 07_results --dry-run
    python3 06_tools/python/relocate.py --apply-moves                 # git mv only (contents unchanged)
    python3 06_tools/python/relocate.py --apply-refs                  # reference rewriting only

Mapping table format (markdown table whose header row first cell contains "옛 경로" = "old path"; the doc stays Korean):

    | 옛 경로 | 새 경로 | 규칙 | 이유 |
    |---|---|---|---|
    | `07_results/op_point/latency_sweep/` | `07_results/op_point/runs/latency_sweep/` | dir | run folder |
    | `07_results/nms_host_cost/raw/(?P<c>[^/]+)_(?P<n>\\d+)\\.log` | `07_results/nms_host_cost/raw/\\g<c>_run\\g<n>/stdout.log` | re | flat raw |

- If the rule column is `re`, the old path is a Python regex (fullmatch against the whole path) and the new path is a `\\g<name>` substitution template.
- Anything else (dir, file, empty) is a prefix mapping. If several rows match: re rows (table order) -> longest prefix.

Reference rewrite rules (target file F, old location Fo, new location Fn)
- Finds markdown links `](...)`, paths inside backticks, and path tokens in plain text.
- Tokens starting with a top-level band like `07_results/` are root-relative, `/home/pi/workspace/yolo_vulkan_optimize/`
  absolute paths stay absolute, everything else is resolved relative to Fo.
- The mapping is applied to the resolved path; if it was relative, it is rewritten relative to Fn
  (if F itself moves, relative links pointing at unmoved targets are rewritten too).
- Tokens containing `{`, `*`, `<…>` are not fixed automatically and go to the "manual" list.
- Non-root-relative prose shorthand (`op_point/latency_gen3_cooled/…`) that matches an old path under `07_results/`
  goes to the "guessed reference" list (not rewritten automatically).

Targets: git-tracked *.md, *.graphml, 06_tools/**/*.sh, 01_src/**/*.{cpp,hpp}, .claude/**, CLAUDE.md, README.md.
Excluded: non-.md files inside 07_results (raw logs are never edited), the "옛 경로" (old path) table of the mapping file,
      and the "지운 파일" (deleted files) column of 07_results/duplicate_png.md.
"""
from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ABS = str(ROOT)
DEFAULT_MAP = "STRUCTURE.md"
DUP_PNG = "07_results/duplicate_png.md"
TOP_NAMES = ("01_src", "02_shaders", "03_models", "04_data", "05_tests", "06_tools", "07_results",
             "08_learn", "09_diagrams", "10_paper", "11_logs", "cmake", ".claude", "build")

# Path token: components made of path characters or `<placeholder>`, joined by '/'
_SEG = r"(?:[\w.\-~+@{},*]|<[^<>\s/`|:=\"']{1,24}>)"
TOKEN_RE = re.compile(rf"{_SEG}*(?:/{_SEG}*)+")
SPECIAL = ("{", "*", "<")


# ─── Mapping table ────────────────────────────────────────────────────────────
@dataclass
class Row:
    old: str
    new: str
    is_re: bool
    reason: str
    line: int
    rx: re.Pattern | None = None

    @property
    def old_root(self) -> str:
        """For a regex, the directory prefix up to the first metacharacter; otherwise the old path itself."""
        if not self.is_re:
            return self.old.rstrip("/")
        lit = []
        for ch in self.old.lstrip("^"):
            if ch in "\\^$*+?{}[]|().":
                break
            lit.append(ch)
        s = "".join(lit)
        return s.rsplit("/", 1)[0] if "/" in s else s


def _strip_cell(c: str) -> str:
    c = c.strip()
    if c.startswith("`") and c.endswith("`") and len(c) >= 2:
        c = c[1:-1]
    return c.replace("\\|", "|").strip()


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into cells. '|' inside backticks or as `\\|` is not a cell separator."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, tick, i = [], [], False, 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("\\|")
            i += 2
            continue
        if ch == "`":
            tick = not tick
        if ch == "|" and not tick:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return cells


def _is_sep(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in cells if c.strip())


def parse_map(map_path: Path, section: str | None = None) -> list[Row]:
    """Read all rows of tables in the mapping file whose header first cell is '옛 경로' (old path).
    If section is given, only look under the heading containing that string (until a heading of the same/higher level)."""
    lines = map_path.read_text(encoding="utf-8").splitlines()
    rows: list[Row] = []
    in_sec = section is None
    sec_level = 0
    header: list[str] | None = None
    prev: list[str] | None = None
    for no, ln in enumerate(lines, 1):
        h = re.match(r"^(#{1,6})\s+(.*)", ln)
        if h:
            level = len(h.group(1))
            if section is not None:
                if in_sec and level <= sec_level:
                    in_sec = False
                if not in_sec and section in h.group(2):
                    in_sec, sec_level = True, level
            header = prev = None
            continue
        if not ln.strip().startswith("|"):
            header = prev = None
            continue
        cells = _split_row(ln)
        if _is_sep(cells):
            header = prev
            continue
        prev = cells
        if not in_sec or header is None or "옛 경로" not in header[0]:  # matches Korean text in STRUCTURE.md
            continue
        if len(cells) < 2:
            continue
        old, new = _strip_cell(cells[0]), _strip_cell(cells[1])
        rule = _strip_cell(cells[2]) if len(cells) > 2 else ""
        reason = _strip_cell(cells[3]) if len(cells) > 3 else ""
        if not old or not new:
            continue
        is_re = rule.lower() == "re"
        row = Row(old, new, is_re, reason, no)
        if is_re:
            try:
                row.rx = re.compile(old.lstrip("^").rstrip("$"))
            except re.error as e:
                raise SystemExit(f"{map_path}:{no}: regex error: {e}")
        rows.append(row)
    return rows


class Mapper:
    def __init__(self, rows: list[Row]):
        self.re_rows = [r for r in rows if r.is_re]
        self.prefix_rows = sorted((r for r in rows if not r.is_re), key=lambda r: -len(r.old.rstrip("/")))
        # Drop depth < 2 (e.g. '07_results' of regex '07_results/(?P<exp>…)') — too broad
        self.roots = sorted({r.old_root for r in rows if r.old_root.count("/") >= 1})

    def map_path(self, p: str) -> str | None:
        """If p (root-relative, no trailing '/') is an old path, return the new path, else None."""
        for r in self.re_rows:
            m = r.rx.fullmatch(p)
            if m:
                return m.expand(r.new).rstrip("/")
        for r in self.prefix_rows:
            o = r.old.rstrip("/")
            if p == o or p.startswith(o + "/"):
                n = r.new.rstrip("/") + p[len(o):]
                # For rows like a/ -> a/b/, do not move again what is already at the new path
                if p.startswith(r.new.rstrip("/") + "/") or p == r.new.rstrip("/"):
                    return None
                return n
        return None

    def inverse(self, p: str) -> str | None:
        """Invert prefix rows only (inversion is undefined for regex rows)."""
        for r in sorted(self.prefix_rows, key=lambda r: -len(r.new.rstrip("/"))):
            n = r.new.rstrip("/")
            if p == n or p.startswith(n + "/"):
                return r.old.rstrip("/") + p[len(n):]
        return None

    def touches(self, literal_dir: str) -> bool:
        """Does the literal directory of a glob token overlap a mapping target (only at depth >= 2)?"""
        d = literal_dir.rstrip("/")
        if d.count("/") < 1:
            return False
        return any(r == d or r.startswith(d + "/") or d.startswith(r + "/") for r in self.roots)


# ─── git ──────────────────────────────────────────────────────────────────────
def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", REPO_ABS, *args], capture_output=True, text=True, check=check).stdout


def tracked() -> list[str]:
    return [p for p in git("ls-files", "-z").split("\0") if p]


def dirs_of(files) -> set[str]:
    out = set()
    for f in files:
        parts = f.split("/")
        for k in range(1, len(parts)):
            out.add("/".join(parts[:k]))
    return out


# ─── Move expansion ───────────────────────────────────────────────────────────
@dataclass
class Plan:
    moves: list[tuple[str, str]] = field(default_factory=list)
    done: int = 0                       # already moved (exists only at the new path)
    conflicts: list[str] = field(default_factory=list)


def expand_moves(mapper: Mapper, files: list[str]) -> Plan:
    plan = Plan()
    fset = set(files)
    targets: dict[str, str] = {}
    inv_candidates = 0
    for f in files:
        n = mapper.map_path(f)
        if n is None or n == f:
            o = mapper.inverse(f)
            if o is not None and o not in fset:
                inv_candidates += 1
            continue
        if n in fset and mapper.map_path(n) is None:
            plan.conflicts.append(f"{f} → {n}: a file already exists at the new path")
            continue
        if n in targets:
            plan.conflicts.append(f"{f} → {n}: goes to the same place as {targets[n]}")
            continue
        if mapper.map_path(n) is not None:
            plan.conflicts.append(f"{f} → {n}: new path is mapped again (chain)")
            continue
        targets[n] = f
        plan.moves.append((f, n))
    # Count as 'already moved' only when there is nothing to move. Before moving, original files like the existing samples/
    # overlap the new-path prefix and the inverse mapping would give false positives.
    plan.done = inv_candidates if not plan.moves else 0
    return plan


# ─── Reference rewriting ──────────────────────────────────────────────────────
def is_target(p: str, map_rel: str) -> bool:
    if p.startswith("07_results/") and not p.endswith(".md"):
        return False
    if p.endswith((".md", ".graphml")):
        return True
    if p.startswith("06_tools/") and p.endswith(".sh"):
        return True
    if p.startswith("01_src/") and p.endswith((".cpp", ".hpp")):
        return True
    return p.startswith(".claude/") or p in ("CLAUDE.md", "README.md", map_rel)


@dataclass
class Edit:
    line: int
    old: str
    new: str


@dataclass
class Note:
    file: str
    line: int
    token: str
    why: str


def _norm(p: str) -> str | None:
    q = posixpath.normpath(p)
    if q.startswith("..") or q.startswith("/") or q == ".":
        return None
    return q


class Rewriter:
    def __init__(self, mapper: Mapper, files: list[str], map_rel: str):
        self.m = mapper
        self.map_rel = map_rel
        self.files = set(files)
        self.known = self.files | dirs_of(files)
        self.new_paths: set[str] = set()   # paths that will exist after the move (filled in dry-run)
        # If any old path is still tracked we are 'before the move'. Then the inverse (Fn -> Fo) is not used.
        self.post_move = not any(mapper.map_path(f) is not None for f in files)
        self.manual: list[Note] = []
        self.guess: list[Note] = []

    def _real(self, old: str, new: str) -> bool:
        """Rewrite only if either the old or the new path exists. History prose pointing at already deleted
        files (e.g. selftest/b4/ deleted by duplicate_png) is left as is."""
        return old in self.known or new in self.known or new in self.new_paths

    def _exists_pre(self, p: str) -> bool:
        return p in self.known or self.m.map_path(p) is not None

    def _valid_post(self, p: str) -> bool:
        """Will this path exist in the tree after the move? After moving, known is the new tree."""
        return (p in self.known and self.m.map_path(p) is None) or p in self.new_paths

    def locate(self, p: str) -> tuple[str, str]:
        """(old location Fo, new location Fn) of currently tracked path p."""
        n = self.m.map_path(p)
        if n is not None:
            return p, n
        o = self.m.inverse(p) if self.post_move else None
        if o is not None and o not in self.files:
            return o, p
        return p, p

    def excluded_spans(self, rel: str, lines: list[str]) -> dict[int, list[tuple[int, int]]]:
        """Line number -> [start, end) spans to exclude (mapping file history table, duplicate_png '지운 파일' (deleted files) column)."""
        out: dict[int, list[tuple[int, int]]] = {}
        if rel not in (self.map_rel, DUP_PNG):
            return out
        header = prev = None
        for i, ln in enumerate(lines):
            if not ln.strip().startswith("|"):
                header = prev = None
                continue
            cells = _split_row(ln)
            if _is_sep(cells):
                header = prev
                continue
            prev = cells
            if header is None:
                continue
            if rel == self.map_rel and "옛 경로" in header[0]:  # matches Korean text in STRUCTURE.md
                out[i] = [(0, len(ln))]
            if rel == DUP_PNG:
                cols = [k for k, h in enumerate(header) if "지운 파일" in h]  # matches Korean text in duplicate_png.md
                if cols:
                    out.setdefault(i, []).extend(_cell_spans(ln, cols))
        return out

    def rewrite_token(self, raw: str, prev_ch: str, fo: str, fn: str, is_src: bool) -> tuple[str | None, str | None]:
        """(new token or None, note) — note is 'manual'/'guess'/None."""
        tok = raw.rstrip(".,;:")
        while tok.endswith("**"):          # markdown bold
            tok = tok[:-2].rstrip(".,;:")
        if tok.startswith("**"):
            return None, None              # the stripped token is examined separately on the next call
        if not tok or "/" not in tok or "//" in tok:
            return None, None
        trail = "/" if tok.endswith("/") else ""
        core = tok.rstrip("/")
        if any(c in core for c in SPECIAL):
            lit = re.split(r"[{*<]", core, maxsplit=1)[0]
            lit_dir = lit.rsplit("/", 1)[0] if "/" in lit else ""
            base = self._resolve_form(lit_dir, prev_ch, fo)
            if base and (base[0] in self.known or self.m.map_path(base[0]) is not None) \
                    and self.m.touches(base[0]):
                return None, "manual"
            return None, None
        form = self._resolve_form(core, prev_ch, fo)
        if form is None:
            return None, None
        r, kind = form
        if kind != "rel":
            n = self.m.map_path(r)
            if n is None or not self._real(r, n):
                return None, None
            return self._emit(n, kind, fn, trail, core, tok, raw), None
        # relative path
        ro = r
        if fo != fn:
            rn = _norm(posixpath.join(posixpath.dirname(fn), core))
            if rn and self._valid_post(rn):
                return None, None
        n = self.m.map_path(ro)
        if n is not None and not self._real(ro, n):
            return None, None
        target = n if n is not None else ro if self._exists_pre(ro) else None
        if target is None and is_src:
            # include-root (01_src) relative path like #include "common/x.hpp"
            inc = _norm("01_src/" + core)
            if inc and self.m.map_path(inc) is not None:
                new = self.m.map_path(inc)[len("01_src/"):] + trail
                return raw.replace(tok, new, 1), None
        if target is None:
            g = _norm("07_results/" + core)
            if g and self.m.map_path(g) is not None:
                return None, "guess"
            return None, None
        if target == ro and fo == fn:
            return None, None
        new = posixpath.relpath(target, posixpath.dirname(fn) or ".")
        if core.startswith("./") and not new.startswith("."):
            new = "./" + new
        if new == core:
            return None, None
        return raw.replace(tok, new + trail, 1), None

    def _resolve_form(self, core: str, prev_ch: str, fo: str) -> tuple[str, str] | None:
        if core.startswith(REPO_ABS + "/"):
            r = _norm(core[len(REPO_ABS) + 1:])
            return (r, "abs") if r else None
        if core.startswith("/"):
            if prev_ch == "}" or prev_ch.isalnum() or prev_ch == "_":   # $PROJ/07_results/…
                r = _norm(core[1:])
                return (r, "rootslash") if r else None
            return None
        head = core.split("/", 1)[0]
        if head in TOP_NAMES or head.startswith("build"):
            r = _norm(core)
            return (r, "root") if r else None
        if not core:
            return None
        r = _norm(posixpath.join(posixpath.dirname(fo), core))
        return (r, "rel") if r else None

    def _emit(self, n: str, kind: str, fn: str, trail: str, core: str, tok: str, raw: str) -> str:
        if kind == "abs":
            new = f"{REPO_ABS}/{n}"
        elif kind == "rootslash":
            new = "/" + n
        else:
            new = n
        return raw.replace(tok, new + trail, 1)

    def process(self, rel: str, text: str) -> tuple[str, list[Edit]]:
        fo, fn = self.locate(rel)
        is_src = rel.endswith((".cpp", ".hpp"))
        lines = text.split("\n")
        excl = self.excluded_spans(rel, lines)
        edits: list[Edit] = []
        for i, ln in enumerate(lines):
            spans = excl.get(i, [])
            pieces, last = [], 0
            for m in TOKEN_RE.finditer(ln):
                s, e = m.span()
                if any(a <= s < b for a, b in spans):
                    continue
                raw = m.group(0)
                if raw.startswith("**"):
                    s, raw = s + 2, raw[2:]
                # a comma is a separator outside braces
                subs = [(s, raw)] if "{" in raw or "," not in raw else _split_commas(s, raw)
                for ss, sub in subs:
                    prev_ch = ln[ss - 1] if ss > 0 else ""
                    new, note = self.rewrite_token(sub, prev_ch, fo, fn, is_src)
                    if note == "manual":
                        self.manual.append(Note(rel, i + 1, sub, "glob/placeholder — fix by hand"))
                    elif note == "guess":
                        self.guess.append(Note(rel, i + 1, sub, "looks like prose shorthand relative to 07_results/"))
                    if new is not None and new != sub:
                        pieces.append(ln[last:ss])
                        pieces.append(new)
                        last = ss + len(sub)
                        edits.append(Edit(i + 1, sub, new))
            if pieces:
                pieces.append(ln[last:])
                lines[i] = "".join(pieces)
        return "\n".join(lines), edits


def _cell_spans(ln: str, cols: list[int]) -> list[tuple[int, int]]:
    """[start, end) spans of cell numbers (cols, per _split_row) in table row ln."""
    pipes, tick = [], False
    for k, ch in enumerate(ln):
        if ch == "`":
            tick = not tick
        elif ch == "|" and not tick and (k == 0 or ln[k - 1] != "\\"):
            pipes.append(k)
    if not ln.lstrip().startswith("|"):
        pipes.insert(0, -1)
    if len(pipes) < 2 or pipes[-1] != len(ln.rstrip()) - 1:
        pipes.append(len(ln))
    return [(pipes[c] + 1, pipes[c + 1]) for c in cols if c + 1 < len(pipes)]


def _split_commas(start: int, raw: str) -> list[tuple[int, str]]:
    out, pos = [], start
    for part in raw.split(","):
        if part:
            out.append((pos, part))
        pos += len(part) + 1
    return out


def scan_refs(mapper: Mapper, map_rel: str, new_paths: set[str] | None = None):
    files = tracked()
    rw = Rewriter(mapper, files, map_rel)
    if new_paths:
        rw.new_paths = new_paths | dirs_of(new_paths)
    results = []
    for rel in sorted(files):
        if not is_target(rel, map_rel):
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text, edits = rw.process(rel, text)
        if edits or new_text != text:
            results.append((rel, rw.locate(rel), new_text, edits))
    return results, rw


# ─── CLI ──────────────────────────────────────────────────────────────────────
def warn_dirty() -> None:
    st = git("status", "--porcelain")
    if st.strip():
        n = len(st.strip().splitlines())
        print(f"warning: working tree is not clean ({n} entries). Take care not to mix move/rewrite commits with other changes.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="git mv according to the mapping table and rewrite references")
    ap.add_argument("--map", default=DEFAULT_MAP, help=f"mapping table markdown (default {DEFAULT_MAP})")
    ap.add_argument("--section", help="read only tables under the heading containing this string (e.g. '07_results')")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="print only the move list and a reference rewrite summary")
    g.add_argument("--apply-moves", action="store_true", help="run git mv only")
    g.add_argument("--apply-refs", action="store_true", help="run reference rewriting only")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every rewrite")
    a = ap.parse_args(argv)

    map_path = Path(a.map)
    if not map_path.is_absolute():
        map_path = (Path.cwd() / map_path) if (Path.cwd() / map_path).exists() else ROOT / map_path
    if not map_path.exists():
        print(f"mapping table not found: {map_path}", file=sys.stderr)
        return 2
    try:
        map_rel = str(map_path.resolve().relative_to(ROOT))
    except ValueError:
        map_rel = ""   # mapping table outside the repo (for testing)
    rows = parse_map(map_path, a.section)
    if not rows:
        print("no '옛 경로' (old path) table rows found in the mapping table", file=sys.stderr)
        return 2
    mapper = Mapper(rows)
    files = tracked()
    plan = expand_moves(mapper, files)
    print(f"mapping rows: {len(rows)} (regex {len(mapper.re_rows)}, prefix {len(mapper.prefix_rows)}) — {map_path}"
          + (f" §'{a.section}'" if a.section else ""))

    if a.apply_moves or a.apply_refs:
        warn_dirty()

    if plan.conflicts:
        print(f"== conflicts: {len(plan.conflicts)} (not moving)")
        for c in plan.conflicts:
            print("  " + c)

    if a.dry_run or a.apply_moves:
        print(f"== moves: {len(plan.moves)}" + (f" (already moved: {plan.done})" if plan.done else ""))
        for o, n in plan.moves:
            print(f"  {o} → {n}")
        unused = [r for r in rows if not any(mapper.map_path(o) and _row_hits(r, o) for o, _ in plan.moves)]
        for r in unused:
            print(f"  (line {r.line}: no matching tracked file — {r.old})")

    if a.apply_moves:
        if plan.conflicts:
            print("Conflicts present; nothing was moved.", file=sys.stderr)
            return 1
        for o, n in plan.moves:
            (ROOT / n).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-C", REPO_ABS, "mv", "-k", o, n], check=True)
        print(f"git mv done: {len(plan.moves)}")
        return 0

    new_paths = {n for _, n in plan.moves}
    results, rw = scan_refs(mapper, map_rel, new_paths)
    total = sum(len(e) for *_, e in results)
    print(f"== reference rewrite: {len(results)} files, {total} replacements")
    for rel, (fo, fn), _, edits in results:
        loc = f"  (file also moves: → {fn})" if fo != fn else ""
        print(f"  {rel}: {len(edits)}{loc}")
        shown = edits if a.verbose else edits[:5]
        for e in shown:
            print(f"      L{e.line}: {e.old}  →  {e.new}")
        if len(edits) > len(shown):
            print(f"      … and {len(edits) - len(shown)} more (-v for all)")
    if rw.manual:
        print(f"== manual: {len(rw.manual)} (glob/{{}}/<placeholder> — not rewritten automatically)")
        for n in rw.manual:
            print(f"  {n.file}:{n.line}  {n.token}")
    if rw.guess:
        print(f"== guessed references: {len(rw.guess)} (prose shorthand relative to 07_results/ — not rewritten automatically)")
        for n in rw.guess:
            print(f"  {n.file}:{n.line}  {n.token}")

    if a.apply_refs:
        for rel, _, new_text, _ in results:
            (ROOT / rel).write_text(new_text, encoding="utf-8")
        print(f"saved {len(results)} files")
    return 0


def _row_hits(row: Row, path: str) -> bool:
    if row.is_re:
        return row.rx.fullmatch(path) is not None
    o = row.old.rstrip("/")
    return path == o or path.startswith(o + "/")


if __name__ == "__main__":
    sys.exit(main())
