#!/usr/bin/env python3
"""Check the repository structure rules (06_tools/structure_rules.toml).

Runs on system python3 (3.11+, tomllib). No external dependencies.

    python3 06_tools/python/check_structure.py                # full check. exit 1 on errors
    python3 06_tools/python/check_structure.py --strict       # warnings also exit 1
    python3 06_tools/python/check_structure.py --report-only  # always exit 0 (for transition periods)
    python3 06_tools/python/check_structure.py --changed      # only new/untracked files from git status
    python3 06_tools/python/check_structure.py --explain 07_results/x/raw/a.log
    python3 06_tools/python/check_structure.py --selftest     # self-test of classify_new_path

One output line = `ID  path  reason → suggested correct location`.

Check IDs
  T1 allowed top-level items       T2 root log leftovers/tracked dumps  T3 tracked though ignored
  R1 07_results folders ↔ index 1:1  R2 07_results root files     R3 summary.md, env_bench.txt
  R4 allowed direct items of exp dir  R5 required items per type   R6 raw/ layout
  R7 raw/_session contents         R8 runs/ depth                  R9 notes/, evidence/ names
  R10 tracked 0-byte hailort*.log
  C1 folders directly under 01_src  C2 name clash with system include dir  C3 .comp location/registration
  C4 include layer DAG             C5 main() location              C6 .cpp CMake registration
  C7 path literals in source, asset macros  C8 forbidden CMake tokens  C9 bench shell paths
  C10 npu_guard BENCH_SCRIPTS      D1 STRUCTURE.md ↔ TOML           E1 unused exceptions

Public API (imported by the Write hook)
  load_rules() -> dict
  classify_new_path(relpath) -> None | (rule_id, why, suggestion)
"""
from __future__ import annotations

import argparse
import ast
import functools
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "06_tools" / "structure_rules.toml"

ERROR = "error"
WARN = "warning"


# ─── Rule loading and globs ───────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def load_rules() -> dict:
    """Read structure_rules.toml (once per process)."""
    with RULES_PATH.open("rb") as f:
        return tomllib.load(f)


@functools.lru_cache(maxsize=512)
def _glob_re(pattern: str) -> re.Pattern:
    """'*' is one component, '**/' is zero or more components, a trailing '**' is everything remaining."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(path: str, pattern: str) -> bool:
    return _glob_re(pattern).match(path) is not None


def any_glob(path: str, patterns) -> bool:
    return any(glob_match(path, p) for p in patterns)


# ─── Violation records ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    why: str
    suggest: str = ""
    level: str = ERROR
    token: str = ""   # exception waive tokens (space-separated, may be several). Also matched by rule ID

    def line(self) -> str:
        s = f"{self.rule:<4} {self.path}  {self.why}"
        if self.suggest:
            s += f" → {self.suggest}"
        return s


@dataclass
class Exceptions:
    items: list = field(default_factory=list)
    used: set = field(default_factory=set)

    @classmethod
    def from_rules(cls, rules: dict) -> "Exceptions":
        return cls(items=list(rules.get("exception", [])))

    def waives(self, v: Violation, mark: bool = True) -> bool:
        for idx, ex in enumerate(self.items):
            toks = set(ex.get("waive", []))
            if v.rule not in toks and not toks.intersection(v.token.split()):
                continue
            if _ancestor_matches(v.path, ex["path"]):
                if mark:
                    self.used.add(idx)
                return True
        return False


def _ancestor_matches(path: str, pattern: str) -> bool:
    """Does path itself or one of its ancestors match pattern (component glob)?"""
    parts = path.rstrip("/").split("/")
    for n in range(len(parts), 0, -1):
        if glob_match("/".join(parts[:n]), pattern):
            return True
    return False


# ─── classify_new_path: judging a new path with barely any filesystem access ───────
def _normalize_rel(relpath: str) -> str | None:
    p = relpath.replace("\\", "/")
    if os.path.isabs(p):
        try:
            p = str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return None   # outside the repository
    p = os.path.normpath(p).replace("\\", "/")
    if p.startswith("../") or p == "..":
        return None
    return p


def classify_new_path(relpath: str) -> None | tuple[str, str, str]:
    """Judge a single, not-yet-existing new file path using the rule table only.

    Returns: None if it passes, (rule ID, reason, suggested correct location) on violation.
    Paths outside the repository and untracked bands (build*, 03_models, 04_data, 11_logs, .git) give None.
    """
    rules = load_rules()
    rel = _normalize_rel(relpath)
    if rel is None or rel in ("", "."):
        return None
    v = _classify(rel, rules, Exceptions.from_rules(rules))
    if v is None:
        return None
    return (v.rule, v.why, v.suggest)


def _classify(rel: str, rules: dict, exc: Exceptions) -> Violation | None:
    top = rules["top_level"]
    parts = rel.split("/")
    head = parts[0]
    untracked = set(top["untracked"])
    if head in untracked or any_glob(head, top.get("untracked_globs", [])):
        return None

    if len(parts) == 1:   # root file
        if head in rules["junk"]["root_logs"]:
            return Violation("T2", rel, "runtime log leftover in root",
                             "11_logs/<date>_<name>.log or 07_results/<experiment>/raw/_session/")
        if head in top["tracked"]:
            return None
        ext = os.path.splitext(head)[1]
        sug = rules.get("placement_root_suggest", {}).get(ext, "one of the bands in STRUCTURE.md §1")
        return Violation("T1", rel, "file in root not on the allow list", sug)

    if head not in top["tracked"]:
        return Violation("T1", rel, f"top-level folder '{head}/' not on the allow list", "one of the bands in STRUCTURE.md §1")

    if parts[-1].endswith(".comp") and head != "02_shaders":
        return Violation("C3", rel, ".comp belongs only in 02_shaders/", "02_shaders/" + parts[-1])

    base = os.path.basename(rel)
    if any_glob(base, rules["junk"]["dump_globs"]):
        return Violation("T2", rel, "dump files are not tracked", "11_logs/ or build/")

    if head == rules["code"]["src_root"]:
        return _classify_src(rel, parts, rules)
    if head == rules["results"]["root"]:
        return _classify_results(rel, parts, rules, exc)

    return _classify_placement(rel, rules)


def _classify_src(rel: str, parts: list[str], rules: dict) -> Violation | None:
    code = rules["code"]
    if len(parts) == 2:
        if parts[1] in code["root_files"]:
            return None
        return Violation("C1", rel, "only CMakeLists.txt goes directly under 01_src", "01_src/<layer>/")
    d = parts[1]
    if d not in code["allowed_dirs"]:
        sug = code.get("suggest", {}).get(d, "01_src/{" + ",".join(code["allowed_dirs"]) + "}/")
        return Violation("C1", rel, f"folder not allowed directly under 01_src: '{d}/'", sug)
    if d in code["library_dirs"] and len(parts) > 3 and parts[2] in code["library_forbidden_subdirs"]:
        return Violation("C1", rel, f"'{parts[2]}/' inside a library folder",
                         f"01_src/tools/{d}/" + "/".join(parts[3:]))
    return None


def _classify_placement(rel: str, rules: dict) -> Violation | None:
    best = None
    for pl in rules.get("placement", []):
        if rel.startswith(pl["prefix"]) and (best is None or len(pl["prefix"]) > len(best["prefix"])):
            best = pl
    if best is None:
        return None
    rest = rel[len(best["prefix"]):]
    # If a sub-rule with a longer prefix exists (06_tools/ vs 06_tools/bench/), it does not get caught here
    if any_glob(rest, best["allow"]):
        return None
    rid = "C9" if best["prefix"].startswith("06_tools/bench") else "T1"
    return Violation(rid, rel, f"file not allowed in {best['prefix']} (allowed: {', '.join(best['allow'])})",
                     best.get("hint", "see STRUCTURE.md"))


def _suggest_exp_dir(name: str, rules: dict) -> str:
    for pat, sug in rules["results"].get("suggest_dirs", {}).items():
        if glob_match(name, pat):
            return sug
    return "allowed folders: " + ", ".join(rules["results"]["dirs"])


_RUN_LOG = re.compile(r"^(?P<cfg>.+?)_run(?P<n>\d+)\.log$")
_NUM_LOG = re.compile(r"^(?P<cfg>.+?)_(?P<n>\d+)\.log$")


def _suggest_raw_file(base: str, name: str, rules: dict) -> str:
    res = rules["results"]
    if name in res["session_files"]:
        return f"{base}/raw/{res['session_dir']}/{name}"
    m = _RUN_LOG.match(name)
    if m:
        return f"{base}/raw/{m['cfg']}_run{m['n']}/run.log"
    m = _NUM_LOG.match(name)
    if m:
        return f"{base}/raw/{m['cfg']}_run{m['n']}/stdout.log"
    return f"{base}/raw/<config>_run<N>/{name}"


def _classify_results(rel: str, parts: list[str], rules: dict, exc: Exceptions,
                      mark: bool = False) -> Violation | None:
    res = rules["results"]
    if len(parts) == 2:
        if parts[1] in res["root_files"]:
            return None
        return Violation("R2", rel, "file not allowed in 07_results root", "07_results/<experiment>/")
    base = "/".join(parts[:2])
    v = _classify_exp_item(base, parts[2:], rules, in_run=False)
    if v is not None and exc.waives(v, mark=mark):
        return None
    return v


def _classify_exp_item(base: str, rest: list[str], rules: dict, in_run: bool) -> Violation | None:
    """base = experiment (or runs/<round>) folder, rest = path components below it."""
    res = rules["results"]
    first = rest[0]
    anchor = f"{base}/{first}"
    if len(rest) == 1:   # direct file
        if any_glob(first, res["files"]):
            return None
        tok = "sweep_csv layout" if first.endswith(".csv") else "layout"
        return Violation("R4", anchor, "file not allowed directly in experiment folder",
                         "allowed: " + ", ".join(res["files"]) + " — anything else goes to evidence/ or raw/", token=tok)
    if first not in res["dirs"]:
        return Violation("R4", anchor + "/", "folder not allowed directly in experiment folder",
                         f"{base}/{_suggest_exp_dir(first, rules)}", token="layout")
    if first == "raw":
        second = rest[1]
        if len(rest) == 2:
            return Violation("R6", anchor + "/" + second, "file directly in raw/ (put it in a run folder)",
                             _suggest_raw_file(base, second, rules), token="layout")
        if second == res["session_dir"]:
            if len(rest) == 3 and rest[2] in res["session_files"]:
                return None
            return Violation("R7", f"{anchor}/{second}/" + "/".join(rest[2:]),
                             "item not allowed in _session/ (allowed: " + ", ".join(res["session_files"]) + ")",
                             f"{base}/evidence/ or {base}/raw/<config>_run<N>/", token="layout")
        if not re.match(res["run_dir_re"], second):
            return Violation("R6", f"{anchor}/{second}/", "folder name under raw/ is not of the form <config>_run<N>",
                             f"{base}/raw/{second}_run1/", token="layout")
        if len(rest) > 3:
            return Violation("R6", f"{anchor}/{second}/{rest[2]}/", "subfolder inside a run folder",
                             f"flat in {anchor}/{second}/", token="layout")
        return None
    if first == "runs":
        if in_run:
            return Violation("R8", anchor + "/", "runs/ goes only one level deep", base.rsplit("/runs/", 1)[0] + "/runs/<round>/",
                             token="layout")
        if len(rest) == 2:
            return Violation("R8", anchor + "/" + rest[1], "file directly in runs/", f"{anchor}/<round>/{rest[1]}",
                             token="layout")
        return _classify_exp_item(f"{anchor}/{rest[1]}", rest[2:], rules, in_run=True)
    if first == "notes":
        sub = "/".join(rest[1:])
        if not any_glob(sub, res["notes_globs"]):
            return Violation("R9", f"{anchor}/{sub}", "notes/ holds only .md", f"{base}/evidence/", token="notes")
    return None


# ─── git helpers ──────────────────────────────────────────────────────────────
def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


@functools.lru_cache(maxsize=1)
def tracked_files() -> tuple[str, ...]:
    return tuple(_git("ls-files", "-z"))


@functools.lru_cache(maxsize=1)
def untracked_files() -> tuple[str, ...]:
    return tuple(_git("ls-files", "-z", "-o", "--exclude-standard"))


def _read(rel: str) -> str:
    try:
        return (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ─── Source scanners ──────────────────────────────────────────────────────────
def scan_cpp(text: str) -> tuple[str, list[tuple[int, str]]]:
    """From C/C++ text, build (code with comments/strings blanked out, [(line, string literal content)]).
    Newlines are preserved, so line numbers in the code text match the original."""
    out: list[str] = []
    lits: list[tuple[int, str]] = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            seg = text[i:j]
            out.append(re.sub(r"[^\n]", " ", seg))
            line += seg.count("\n")
            i = j
        elif c == "R" and text.startswith('R"', i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'R"([^()\\ ]{0,16})\(', text[i:])
            if not m:
                out.append(c)
                i += 1
                continue
            end = text.find(")" + m.group(1) + '"', i + m.end())
            end = n if end < 0 else end + len(m.group(1)) + 2
            seg = text[i:end]
            lits.append((line, text[i + m.end():max(i + m.end(), end - len(m.group(1)) - 2)]))
            out.append(re.sub(r"[^\n]", " ", seg))
            line += seg.count("\n")
            i = end
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            if c == '"':
                lits.append((line, text[i + 1:j - 1]))
            out.append(" " * (j - i))
            i = j
        else:
            if c == "\n":
                line += 1
            out.append(c)
            i += 1
    return "".join(out), lits


def strip_cmake_comments(text: str) -> str:
    """Strip CMake '#' comments (preserving quoted text). Line count preserved."""
    lines = []
    for ln in text.splitlines():
        q = False
        cut = len(ln)
        for k, ch in enumerate(ln):
            if ch == '"':
                q = not q
            elif ch == "#" and not q:
                cut = k
                break
        lines.append(ln[:cut])
    return "\n".join(lines)


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


# ─── Checker body ─────────────────────────────────────────────────────────────
class Checker:
    def __init__(self, rules: dict):
        self.rules = rules
        self.exc = Exceptions.from_rules(rules)
        self.out: list[Violation] = []
        self.tracked = tracked_files()
        self.untracked = untracked_files()
        self.all_files = tuple(sorted(set(self.tracked) | set(self.untracked)))

    def add(self, v: Violation) -> None:
        if not self.exc.waives(v):
            self.out.append(v)

    # T ────────────────────────────────────────────────────────────────────────
    def check_top(self) -> None:
        top = self.rules["top_level"]
        allowed_t = set(top["tracked"])
        allowed_u = set(top["untracked"])
        junk = self.rules["junk"]
        heads = sorted({p.split("/", 1)[0] for p in self.tracked})
        for h in heads:
            if h not in allowed_t:
                self.add(Violation("T1", h, "tracked top-level item not on the allow list", "STRUCTURE.md §1"))
        for h in sorted(os.listdir(ROOT)):
            if h in allowed_t or h in allowed_u or any_glob(h, top.get("untracked_globs", [])):
                continue
            if h in junk["root_logs"] or any_glob(h, junk["dump_globs"]):
                continue   # T2 checks these separately
            self.add(Violation("T1", h, "untracked top-level item not on the allow list", "delete it or move it to a STRUCTURE.md §1 band", level=WARN))
        tracked_set = set(self.tracked)
        for name in junk["root_logs"]:
            if name in tracked_set:
                self.add(Violation("T2", name, "root runtime log is tracked", "git rm --cached, then 11_logs/"))
            elif (ROOT / name).exists():
                self.add(Violation("T2", name, "runtime log leftover in root",
                                   "move it to 11_logs/<date>_<name>.log or delete it", level=WARN))
        for p in self.tracked:
            if any_glob(os.path.basename(p), junk["dump_globs"]):
                self.add(Violation("T2", p, "dump file is tracked", "git rm --cached"))
        for h in os.listdir(ROOT):
            if (ROOT / h).is_file() and h not in tracked_set and any_glob(h, junk["dump_globs"]):
                self.add(Violation("T2", h, "dump file leftover in root", "delete it or 11_logs/", level=WARN))
        for p in _git("ls-files", "-z", "-ci", "--exclude-standard"):
            self.add(Violation("T3", p, "tracked though matched by .gitignore", "git rm --cached, or add a .gitignore exception (!)"))

    # R ────────────────────────────────────────────────────────────────────────
    def parse_index(self) -> tuple[dict[str, str | None], bool, list[str]]:
        """From the index table: {folder: type or None}, whether a type column exists, list of duplicate folders."""
        res = self.rules["results"]
        text = _read(res["index"])
        rows: dict[str, str | None] = {}
        dups: list[str] = []
        has_type = False
        header: list[str] | None = None
        prev_cells: list[str] | None = None
        link = re.compile(r"\]\(([^)\s]+)\)")
        for ln in text.splitlines():
            s = ln.strip()
            if not s.startswith("|"):
                header = prev_cells = None
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                header = prev_cells
                continue
            prev_cells = cells
            if header is None:
                continue
            m = link.search(cells[0])
            if not m:
                continue
            tgt = m.group(1).split("#", 1)[0].lstrip("./")
            folder = tgt.split("/", 1)[0]
            if not folder or folder.startswith(".."):
                continue
            typ = None
            if "유형" in header:  # matches the Korean "type" column header in 07_results/README.md
                has_type = True
                k = header.index("유형")  # matches the Korean "type" column header in 07_results/README.md
                typ = cells[k].strip("` *") if k < len(cells) else None
            if folder in rows:
                dups.append(folder)
            rows[folder] = typ
        return rows, has_type, dups

    def check_results(self) -> None:
        res = self.rules["results"]
        root = res["root"]
        files = [p for p in self.all_files if p.startswith(root + "/")]
        exp_dirs = sorted({p.split("/")[1] for p in files if p.count("/") >= 2})

        # R1
        rows, has_type, dups = self.parse_index()
        for d in exp_dirs:
            if d not in rows:
                self.add(Violation("R1", f"{root}/{d}/", "no row in the index (07_results/README.md) table", "add one row to the index table"))
        for d in rows:
            if d not in exp_dirs:
                self.add(Violation("R1", f"{root}/{d}/", "folder pointed to by index row does not exist", "delete the index row or fix the path"))
        for d in dups:
            self.add(Violation("R1", f"{root}/{d}/", "duplicate rows in the index table", "merge into one row"))

        # R2, R4, R6~R9 (per file, deduplicated by anchor)
        seen: set[tuple[str, str]] = set()
        for p in files:
            v = _classify_results(p, p.split("/"), self.rules, self.exc, mark=True)
            if v is not None and (v.rule, v.path) not in seen:
                seen.add((v.rule, v.path))
                self.out.append(v)

        # R3: experiment folders and runs/<round>/
        fileset = set(files)
        dirset = {os.path.dirname(p) for p in files}
        dirset |= {"/".join(p.split("/")[:k]) for p in files for k in range(2, p.count("/") + 1)}
        units: list[tuple[str, list[str]]] = [(f"{root}/{d}", res["required"]) for d in exp_dirs]
        for d in sorted(dirset):
            parts = d.split("/")
            if len(parts) == 4 and parts[2] == "runs":
                units.append((d, res["run_required"]))
        for unit, req in units:
            for r in req:
                if f"{unit}/{r}" not in fileset:
                    tok = "summary" if r == "summary.md" else "env_bench" if r == "env_bench.txt" else ""
                    self.add(Violation("R3", unit + "/", f"required file {r} missing", f"write {unit}/{r}", token=tok))

        # R5
        if not has_type:
            self.add(Violation("R5", res["index"], "index has no type column — skipping the per-type required items (R5) check",
                               "add a type (" + "/".join(res["type_required"]) + ") column to the index table", level=WARN))
        else:
            treq = res["type_required"]
            for d, typ in rows.items():
                if d not in exp_dirs:
                    continue
                if typ not in treq:
                    self.add(Violation("R5", f"{root}/{d}/", f"unknown type '{typ}'",
                                       "type must be " + "/".join(treq), token="type"))
                    continue
                for r in treq[typ]:
                    ok = f"{root}/{d}/{r}".rstrip("/") in (dirset if r.endswith("/") else fileset)
                    if not ok:
                        self.add(Violation("R5", f"{root}/{d}/", f"required item {r} for type {typ} missing",
                                           f"{root}/{d}/{r}", token="type"))

        # R7 warning: session file outside _session (bench_stdout.log)
        # R9 warning: verification log name in evidence/
        for p in files:
            parts = p.split("/")
            name = parts[-1]
            if name == "bench_stdout.log" and not (len(parts) >= 3 and parts[-2] == res["session_dir"]):
                self.add(Violation("R7", p, "session log is outside raw/_session/",
                                   "/".join(parts[:2]) + f"/raw/{res['session_dir']}/{name}", level=WARN, token="layout"))
            if "evidence" in parts[2:-1] and any_glob(name, res["evidence_warn_globs"]):
                ev = parts.index("evidence", 2)
                self.add(Violation("R9", p, "verification-gate log name is in evidence/",
                                   "/".join(parts[:ev]) + "/verification/" + name, level=WARN, token="evidence_names"))

        # R10
        for p in self.tracked:
            if p.startswith(root + "/") and glob_match(os.path.basename(p), "hailort*.log"):
                try:
                    if (ROOT / p).stat().st_size == 0:
                        self.add(Violation("R10", p, "tracked hailort log is 0 bytes (the reliability evidence is empty)",
                                           "cross-check with the measurement records", level=WARN))
                except OSError:
                    pass

    # C ────────────────────────────────────────────────────────────────────────
    def check_code(self) -> None:
        code = self.rules["code"]
        src = code["src_root"]
        tests = code["tests_dir"]
        files = [p for p in self.all_files if p.startswith(src + "/")]

        # C1
        seen_dirs: set[str] = set()
        for p in files:
            v = _classify_src(p, p.split("/"), self.rules)
            if v is None:
                continue
            parts = p.split("/")
            key = "/".join(parts[:2]) if v.why.startswith("folder not allowed directly under 01_src") else \
                "/".join(parts[:3]) if "inside a library folder" in v.why else p
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            anchor = key + "/" if key != p else p
            sug = v.suggest
            if "inside a library folder" in v.why:
                sug = f"01_src/tools/{parts[1]}/"
            self.add(Violation("C1", anchor, v.why, sug))

        # C2
        top_dirs = sorted({p.split("/")[1] for p in files if p.count("/") >= 2})
        sysnames: dict[str, str] = {}
        for r in code["system_include_roots"]:
            try:
                for e in os.scandir(r):
                    if e.is_dir():
                        sysnames.setdefault(e.name, r)
            except OSError:
                pass
        for d in top_dirs:
            if d in sysnames and d not in code["system_name_ok"]:
                self.add(Violation("C2", f"{src}/{d}/", f"name clashes with system include folder {sysnames[d]}/{d}/",
                                   code.get("suggest", {}).get(d, "a different name")))

        # C3
        comps = [p for p in self.all_files if p.endswith(".comp")]
        shader_cmake = "\n".join(strip_cmake_comments(_read(p)) for p in self.all_files
                                 if p.startswith("02_shaders/") and p.endswith("CMakeLists.txt"))
        for p in comps:
            if not p.startswith("02_shaders/"):
                self.add(Violation("C3", p, ".comp is outside 02_shaders/", "02_shaders/" + os.path.basename(p)))
                continue
            relp = p[len("02_shaders/"):]
            # Also accept registration by bare filename in the same folder's CMakeLists.txt (02_shaders/diag/CMakeLists.txt)
            own_cmake = os.path.dirname(p) + "/CMakeLists.txt"
            own_text = strip_cmake_comments(_read(own_cmake)) if own_cmake in self.all_files else ""
            pat = lambda s: r"(?<![\w/.-])" + re.escape(s) + r"(?![\w.-])"
            if not (re.search(pat(relp), shader_cmake) or re.search(pat(os.path.basename(p)), own_text)):
                self.add(Violation("C3", p, "not registered in a CMakeLists.txt under 02_shaders/", f"yvo_add_shaders(... {relp})"))

        # C4, C5, C7
        deps = code["include_deps"]
        layers = set(deps)
        unpref = code.get("include_unprefixed", {}).get("globs", [])
        file_exc = {e["file"]: set(e["includes"]) for e in code.get("include_file_exception", [])}
        lib_dirs = set(code["library_dirs"])
        lib_excl = set(code.get("library_exclude", []))
        exts = tuple(code["source_exts"])
        srcs = [p for p in self.all_files if (p.startswith(src + "/") or p.startswith(tests + "/")) and p.endswith(exts)]
        asset_re = re.compile(r"\b(YVO_REPO_ROOT|YVO_DEFAULT_\w+)\b")
        for p in srcs:
            parts = p.split("/")
            layer = tests if parts[0] == tests else parts[1] if len(parts) > 2 else None
            text = _read(p)
            codetext, lits = scan_cpp(text)
            # C4
            if layer in layers:
                # library_exclude files (app main.cpp etc.) are executables, so they follow the same rules as tools
                allowed = deps["tools"] if p in lib_excl else deps[layer]
                for m in INCLUDE_RE.finditer(text):
                    inc = m.group(1)
                    ln = text.count("\n", 0, m.start()) + 1
                    where = f"{p}:{ln}"
                    if inc in file_exc.get(p, ()):
                        continue
                    if "../" in inc or inc.startswith("./"):
                        self.add(Violation("C4", where, f'relative include "{inc}" is forbidden', '"<layer>/<file>" form'))
                        continue
                    if "/" not in inc:
                        if not any_glob(inc, unpref):
                            self.add(Violation("C4", where, f'unprefixed include "{inc}"', '"<layer>/<file>" form'))
                        continue
                    first = inc.split("/", 1)[0]
                    if first == layer:
                        continue
                    if first not in layers or first == tests:
                        self.add(Violation("C4", where, f'layer \'{first}\' of include "{inc}" is not in the rule table',
                                           code.get("suggest", {}).get(first, "STRUCTURE.md §2")))
                    elif "*" not in allowed and first not in allowed:
                        self.add(Violation("C4", where, f"{layer} → {first} include forbidden (allowed: {', '.join(allowed) or 'none'})",
                                           "reverse the dependency or move the shared part to core/ref"))
            # C5
            mains = [codetext.count("\n", 0, m.start()) + 1 for m in re.finditer(r"\bint\s+main\s*\(", codetext)]
            if mains:
                ok = any_glob(p, code["main_allowed"])
                if not ok:
                    self.add(Violation("C5", f"{p}:{mains[0]}", "main() only in 01_src/tools/** and pipeline/main.cpp",
                                       f"01_src/tools/<domain>/{parts[-1]}"))
                elif len(mains) > 1:
                    self.add(Violation("C5", f"{p}:{mains[1]}", "one main() per file", "split the tool"))
            # C7
            for ln, lit in lits:
                if re.match(r"0[3-7]_", lit):
                    self.add(Violation("C7", f"{p}:{ln}", f'path literal "{lit}" (CWD-dependent)',
                                       "YVO_DEFAULT_*/YVO_REPO_ROOT macros from yvo_assets"))
            if layer in lib_dirs and p not in lib_excl:
                for m in asset_re.finditer(codetext):
                    ln = codetext.count("\n", 0, m.start()) + 1
                    self.add(Violation("C7", f"{p}:{ln}", f"{m.group(1)} used in a library folder",
                                       "the caller (tool/app/test) passes the path"))

        # C6
        cmake_files = [p for p in self.all_files if os.path.basename(p) == "CMakeLists.txt"
                       or (p.startswith("cmake/") and p.endswith(".cmake"))]
        cmake_text = "\n".join(strip_cmake_comments(_read(p)) for p in cmake_files)
        for p in self.all_files:
            if p.endswith(".cpp") and (p.startswith(src + "/") or p.startswith(tests + "/")):
                b = os.path.basename(p)
                if not re.search(r"(?<![\w.-])" + re.escape(b) + r"(?![\w.-])", cmake_text):
                    self.add(Violation("C6", p, ".cpp is not registered in any CMakeLists.txt",
                                       os.path.dirname(p) + "/CMakeLists.txt"))

        # C8
        cm = self.rules["cmake"]
        exists_re = re.compile(r"\bif\s*\(\s*(?:NOT\s+)?EXISTS\s+\"?\$\{(?:PROJECT_SOURCE_DIR|CMAKE_CURRENT_SOURCE_DIR|CMAKE_SOURCE_DIR)\}")
        for p in cmake_files:
            txt = strip_cmake_comments(_read(p)).splitlines()
            outside = not p.startswith("cmake/")
            for i, ln in enumerate(txt, 1):
                if outside:
                    for tok in cm["forbidden_outside_cmake_dir"]:
                        if tok in ln:
                            self.add(Violation("C8", f"{p}:{i}", f"'{tok}' is not used outside cmake/",
                                               "cmake/YvoTargets.cmake (yvo_target_defaults / yvo_opencv)"))
                for tok in cm["forbidden_anywhere"]:
                    if tok in ln:
                        self.add(Violation("C8", f"{p}:{i}", f"'{tok}' forbidden (silently skipped if the folder is missing)",
                                           "add_subdirectory() + if(TARGET)"))
                if exists_re.search(ln):
                    self.add(Violation("C8", f"{p}:{i}", "if(EXISTS) on a source folder is forbidden", "if(TARGET <target>)"))

        # C9
        bench = self.rules["bench"]
        shs = sorted(p for p in self.all_files if p.startswith(bench["dir"] + "/") and p.endswith(".sh"))
        for p in shs:
            for i, ln in enumerate(_read(p).splitlines(), 1):
                for tok in bench["forbidden"]:
                    if tok in ln:
                        self.add(Violation("C9", f"{p}:{i}", f"'{tok}' forbidden (binary path tied to source layout)", "$BIN_DIR/<executable>"))
                if p != bench["py_hailo_owner"] and re.match(r"^\s*(?:export\s+|readonly\s+|local\s+)?PY_HAILO=", ln):
                    self.add(Violation("C9", f"{p}:{i}", "PY_HAILO defined here", f"only in one place: {bench['py_hailo_owner']}"))

        # C10
        guard = bench["npu_guard"]
        gtext = _read(guard)
        if gtext:
            try:
                tree = ast.parse(gtext)
            except SyntaxError as e:
                self.add(Violation("C10", guard, f"parse failed: {e}", "", level=WARN))
                tree = None
            for node in ast.walk(tree) if tree else ():
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BENCH_SCRIPTS" for t in node.targets):
                    try:
                        val = ast.literal_eval(node.value)
                    except ValueError:
                        break   # dynamically computed (glob etc.), so the list matches automatically
                    want = {os.path.basename(p) for p in shs if os.path.dirname(p) == bench["dir"]}
                    have = set(val)
                    for x in sorted(want - have):
                        self.add(Violation("C10", f"{guard}:{node.lineno}", f"{x} missing from BENCH_SCRIPTS", "switch to a glob or add it"))
                    for x in sorted(have - want):
                        self.add(Violation("C10", f"{guard}:{node.lineno}", f"{x} in BENCH_SCRIPTS is not in 06_tools/bench/", "remove it from the list"))
                    break

    # D ────────────────────────────────────────────────────────────────────────
    def check_docs(self) -> None:
        sm = self.rules["docs"]["structure_md"]
        text = _read(sm)
        if not text:
            self.add(Violation("D1", sm, "STRUCTURE.md missing — skipping comparison with the TOML", "write STRUCTURE.md (plan step 2)", level=WARN))
            return
        items = _first_table_col0(text, "최상위")  # matches the Korean "top-level" heading (§1) in STRUCTURE.md
        top = self.rules["top_level"]
        want = set(top["tracked"]) | set(top["untracked"])
        if items is None:
            self.add(Violation("D1", sm, "table under the top-level heading not found", "§1 top-level band table"))
            return
        for x in sorted(want - items):
            self.add(Violation("D1", sm, f"'{x}' missing from the table (present in TOML top_level)", "add a table row or fix the TOML"))
        for x in sorted(items - want):
            self.add(Violation("D1", sm, f"'{x}' in the table is not in TOML top_level", "add it to the TOML or remove it from the table"))

    # E ────────────────────────────────────────────────────────────────────────
    def check_exceptions(self) -> None:
        for idx, ex in enumerate(self.exc.items):
            if not ex.get("reason"):
                self.out.append(Violation("E1", ex.get("path", "?"), "exception has no reason", "write a reason"))
            if idx not in self.exc.used:
                self.out.append(Violation("E1", ex.get("path", "?"),
                                          f"unused exception (waive={ex.get('waive')})", "delete it or check the path", level=WARN))

    def run(self) -> list[Violation]:
        self.check_top()
        self.check_results()
        self.check_code()
        self.check_docs()
        self.check_exceptions()
        return self.out


def _first_table_col0(text: str, heading_word: str) -> set[str] | None:
    """Collect backticked items from the first column of the first table under the heading containing heading_word. None if no table."""
    in_sec = False
    rows: list[list[str]] | None = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            if rows:
                break
            in_sec = heading_word in s
            continue
        if not in_sec:
            continue
        if s.startswith("|"):
            rows = rows or []
            rows.append([c.strip() for c in s.strip("|").split("|")])
        elif rows:
            break
    if not rows or len(rows) < 2:
        return None
    body = [r for r in rows[1:] if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
    return {t.rstrip("/") for r in body for t in re.findall(r"`([^`]+)`", r[0])}


# ─── CLI ──────────────────────────────────────────────────────────────────────
ORDER = ["T1", "T2", "T3", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10",
         "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "D1", "E1"]


def _print(vs: list[Violation]) -> tuple[int, int]:
    key = lambda v: (ORDER.index(v.rule) if v.rule in ORDER else 99, v.path)
    errs = sorted((v for v in vs if v.level == ERROR), key=key)
    warns = sorted((v for v in vs if v.level == WARN), key=key)
    if errs:
        print(f"== {ERROR} {len(errs)}")
        for v in errs:
            print(v.line())
    if warns:
        print(f"== {WARN} {len(warns)}")
        for v in warns:
            print(v.line())
    counts: dict[str, list[int]] = {}
    for v in vs:
        c = counts.setdefault(v.rule, [0, 0])
        c[0 if v.level == ERROR else 1] += 1
    summary = ", ".join(f"{r} {c[0]}" + (f"+{c[1]}w" if c[1] else "")
                        for r, c in sorted(counts.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))
    print(f"---\nerrors {len(errs)}, warnings {len(warns)}" + (f"  [{summary}]" if summary else ""))
    return len(errs), len(warns)


def changed_paths() -> list[str]:
    """Paths newly added (A), untracked (??) or rename/copy targets (R/C) per git status."""
    out = []
    ents = _git("status", "--porcelain=v1", "-z", "-uall")
    i = 0
    while i < len(ents):
        e = ents[i]
        st, path = e[:2], e[3:]
        if st[0] in "RC":
            i += 1   # the next entry is the old path
        if st == "??" or st[0] in "ARC":
            out.append(path)
        i += 1
    return out


def explain(path: str) -> int:
    rules = load_rules()
    rel = _normalize_rel(path)
    print(f"path: {rel}")
    if rel is None:
        print("path outside the repository — not checked")
        return 0
    exists = (ROOT / rel).exists()
    print(f"exists: {'yes' if exists else 'no'}")
    r = classify_new_path(rel)
    print("new-file verdict (classify_new_path): " + ("pass" if r is None else f"{r[0]}  {r[1]} → {r[2]}"))
    exc = [e for e in rules.get("exception", []) if _ancestor_matches(rel, e["path"])]
    for e in exc:
        print(f"applied exception: {e['path']} waive={e['waive']} — {e['reason']}")
    hits = [v for v in Checker(rules).run() if v.path.rstrip("/").split(":")[0] == rel.rstrip("/")
            or v.path.startswith(rel.rstrip("/") + "/") or rel.startswith(v.path.rstrip("/") + "/")]
    if hits:
        print("items flagged for this path in the full check:")
        for v in hits:
            print(f"  [{v.level}] {v.line()}")
    else:
        print("items flagged for this path in the full check: none")
    return 0


# Self-test: (path, expected rule ID or None)
SELFTEST_CASES = [
    ("newfile.txt", "T1"),
    ("notes.md", "T1"),
    ("hailort.log", "T2"),
    ("07_results/x/raw/a.log", "R6"),
    ("07_results/x/raw/hailort.log", "R6"),
    ("07_results/x/raw/c_run1/sub/x.log", "R6"),
    ("07_results/x/raw/badname/stdout.log", "R6"),
    ("07_results/x/raw/_session/other.log", "R7"),
    ("01_src/foo.comp", "C3"),
    ("06_tools/python/x.comp", "C3"),
    ("01_src/common/x.hpp", "C1"),
    ("01_src/utils/x.hpp", "C1"),
    ("01_src/vulkan/tools/x.cpp", "C1"),
    ("01_src/stray.cpp", "C1"),
    ("07_results/x/foo.txt", "R4"),
    ("07_results/x/detections/a.jsonl", "R4"),
    ("07_results/x/runs/r1/runs/r2/env_bench.txt", "R8"),
    ("07_results/x/runs/stray.txt", "R8"),
    ("07_results/x/notes/a.txt", "R9"),
    ("07_results/stray.md", "R2"),
    ("06_tools/bench/README.md", "C9"),
    ("06_tools/bench/foo.py", "C9"),
    ("06_tools/python/data.csv", "T1"),
    ("06_tools/python/README.md", "T1"),
    ("99_misc/x.md", "T1"),
    ("cmake/foo.txt", "T1"),
    # pass
    ("07_results/x/raw/c_run1/stdout.log", None),
    ("07_results/x/raw/_session/hailort.log", None),
    ("07_results/x/raw/_session/bench_stdout.log", None),
    ("07_results/x/summary.md", None),
    ("07_results/x/results_raw_part2.csv", None),
    ("07_results/x/evidence/anything.txt", None),
    ("07_results/x/verification/asan_x.log", None),
    ("07_results/x/runs/r1/raw/b8_run2/stdout.log", None),
    ("07_results/x/runs/r1/env_bench.txt", None),
    ("07_results/x/notes/memo.md", None),
    ("07_results/npu_benchmark/runs/gen2/benchmark_b1.csv", None),   # exception: layout
    ("07_results/README.md", None),
    ("01_src/vulkan/foo.cpp", None),
    ("01_src/tools/vulkan/foo_probe.cpp", None),
    ("01_src/CMakeLists.txt", None),
    ("02_shaders/diag/new.comp", None),
    ("02_shaders/new.comp", None),
    ("05_tests/test_new.cpp", None),
    ("06_tools/bench/new_bench.sh", None),
    ("06_tools/bench/lib/extra.sh", None),
    ("06_tools/python/new_tool.py", None),
    ("08_learn/11_new.md", None),
    ("09_diagrams/new.graphml", None),
    ("11_logs/2026-09-22_hailort.log", None),
    ("build/anything/x.o", None),
    ("build-display/x", None),
    ("04_data/videos/new.mp4", None),
    ("STRUCTURE.md", None),
    (".claude/hooks/placement_guard.py", None),
    ("/tmp/outside.txt", None),
]


def selftest() -> int:
    bad = 0
    for path, want in SELFTEST_CASES:
        got = classify_new_path(path)
        gid = got[0] if got else None
        ok = gid == want
        bad += not ok
        detail = f"{got[1]} → {got[2]}" if got else "pass"
        print(f"{'OK ' if ok else 'BAD'}  want {str(want):<5} got {str(gid):<5} {path}  ({detail})")
    print(f"---\n{bad} failed out of {len(SELFTEST_CASES)}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repository structure rule check (06_tools/structure_rules.toml)")
    ap.add_argument("--strict", action="store_true", help="warnings also exit 1")
    ap.add_argument("--report-only", action="store_true", help="always exit 0 (transition period)")
    ap.add_argument("--changed", action="store_true", help="check only new/untracked files from git status with the new-file rules")
    ap.add_argument("--explain", metavar="PATH", help="explain the verdict for one path")
    ap.add_argument("--selftest", action="store_true", help="self-test of classify_new_path")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.explain:
        return explain(a.explain)

    rules = load_rules()
    if a.changed:
        vs = []
        for p in changed_paths():
            r = classify_new_path(p)
            if r:
                vs.append(Violation(r[0], p, r[1], r[2]))
        print(f"checked {len(changed_paths())} new/untracked paths")
    else:
        vs = Checker(rules).run()
    n_err, n_warn = _print(vs)
    if a.report_only:
        return 0
    return 1 if n_err or (a.strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
