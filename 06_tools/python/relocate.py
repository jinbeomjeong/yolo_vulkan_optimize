#!/usr/bin/env python3
"""매핑표(옛 경로 → 새 경로)대로 파일을 옮기고, 저장소 안의 참조를 새 경로로 고친다.

시스템 python3 로 돈다(의존성 없음). 모든 동작은 멱등이다 — 두 번 돌려도 결과가 같다.

    python3 06_tools/python/relocate.py --dry-run                     # 기본 매핑표 STRUCTURE.md
    python3 06_tools/python/relocate.py --map M.md --section 07_results --dry-run
    python3 06_tools/python/relocate.py --apply-moves                 # git mv 만 (내용 불변)
    python3 06_tools/python/relocate.py --apply-refs                  # 참조 치환만

매핑표 형식 (마크다운 표, 머리행 첫 칸에 "옛 경로"):

    | 옛 경로 | 새 경로 | 규칙 | 이유 |
    |---|---|---|---|
    | `07_results/op_point/latency_sweep/` | `07_results/op_point/runs/latency_sweep/` | dir | 회차 폴더 |
    | `07_results/nms_host_cost/raw/(?P<c>[^/]+)_(?P<n>\\d+)\\.log` | `07_results/nms_host_cost/raw/\\g<c>_run\\g<n>/stdout.log` | re | raw 평면 |

- 규칙 열이 `re` 이면 옛 경로는 파이썬 정규식(경로 전체와 fullmatch), 새 경로는 `\\g<이름>` 치환 템플릿.
- 그 밖(dir·file·빈칸)은 접두사 매핑이다. 여러 행이 맞으면 re 행(표 순서) → 가장 긴 접두사 순.

참조 치환 규칙 (대상 파일 F 의 옛 위치 Fo, 새 위치 Fn)
- 마크다운 링크 `](...)`, 백틱 안 경로, 일반 텍스트의 경로 토큰을 찾는다.
- `07_results/` 처럼 최상위 대역으로 시작하면 루트 기준, `/home/pi/workspace/yolo_vulkan_optimize/`
  절대경로는 절대경로로, 그 밖은 Fo 기준 상대경로로 해석한다.
- 해석한 경로에 매핑을 적용하고, 상대경로였으면 Fn 기준 상대경로로 다시 쓴다
  (F 자신이 옮겨지면 옮겨지지 않은 대상을 가리키는 상대 링크도 다시 쓴다).
- `{`, `*`, `<…>` 가 든 토큰은 자동으로 고치지 않고 "수동 처리" 목록에 낸다.
- 루트 기준이 아닌 산문 축약(`op_point/latency_gen3_cooled/…`)이 `07_results/` 아래 옛 경로와
  맞으면 "추정 참조" 목록에 낸다(자동 치환 안 함).

대상: git 추적 파일 중 *.md, *.graphml, 06_tools/**/*.sh, 01_src/**/*.{cpp,hpp}, .claude/**, CLAUDE.md, README.md.
제외: 07_results 안의 .md 아닌 파일(원시 로그는 내용을 고치지 않는다), 매핑표 파일의 "옛 경로" 표,
      07_results/duplicate_png.md 의 "지운 파일" 열.
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

# 경로 토큰: 경로 문자 또는 `<자리표시자>` 로 이루어진 성분들을 '/' 로 이은 것
_SEG = r"(?:[\w.\-~+@{},*]|<[^<>\s/`|:=\"']{1,24}>)"
TOKEN_RE = re.compile(rf"{_SEG}*(?:/{_SEG}*)+")
SPECIAL = ("{", "*", "<")


# ─── 매핑표 ────────────────────────────────────────────────────────────────────
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
        """정규식이면 메타문자 앞까지의 디렉터리 접두사, 아니면 옛 경로 자체."""
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
    """마크다운 표 행을 칸으로 나눈다. 백틱 안과 `\\|` 의 '|' 는 칸 구분자가 아니다."""
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
    """매핑표 파일에서 머리행 첫 칸이 '옛 경로' 인 표의 행을 모두 읽는다.
    section 이 있으면 그 문자열이 든 헤딩 아래(같은/상위 레벨 헤딩 전까지)만 본다."""
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
        if not in_sec or header is None or "옛 경로" not in header[0]:
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
                raise SystemExit(f"{map_path}:{no}: 정규식 오류: {e}")
        rows.append(row)
    return rows


class Mapper:
    def __init__(self, rows: list[Row]):
        self.re_rows = [r for r in rows if r.is_re]
        self.prefix_rows = sorted((r for r in rows if not r.is_re), key=lambda r: -len(r.old.rstrip("/")))
        # 깊이 2 미만(예: 정규식 '07_results/(?P<exp>…)' 의 '07_results')은 너무 넓어 뺀다
        self.roots = sorted({r.old_root for r in rows if r.old_root.count("/") >= 1})

    def map_path(self, p: str) -> str | None:
        """p(루트 기준, 끝 '/' 없음)가 옛 경로면 새 경로, 아니면 None."""
        for r in self.re_rows:
            m = r.rx.fullmatch(p)
            if m:
                return m.expand(r.new).rstrip("/")
        for r in self.prefix_rows:
            o = r.old.rstrip("/")
            if p == o or p.startswith(o + "/"):
                n = r.new.rstrip("/") + p[len(o):]
                # a/ → a/b/ 같은 행에서 이미 새 경로인 것은 다시 옮기지 않는다
                if p.startswith(r.new.rstrip("/") + "/") or p == r.new.rstrip("/"):
                    return None
                return n
        return None

    def inverse(self, p: str) -> str | None:
        """접두사 행만 역변환한다(정규식 행은 역변환이 정의되지 않는다)."""
        for r in sorted(self.prefix_rows, key=lambda r: -len(r.new.rstrip("/"))):
            n = r.new.rstrip("/")
            if p == n or p.startswith(n + "/"):
                return r.old.rstrip("/") + p[len(n):]
        return None

    def touches(self, literal_dir: str) -> bool:
        """글롭 토큰의 리터럴 디렉터리가 매핑 대상과 겹치는가(깊이 2 이상일 때만)."""
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


# ─── 이동 전개 ─────────────────────────────────────────────────────────────────
@dataclass
class Plan:
    moves: list[tuple[str, str]] = field(default_factory=list)
    done: int = 0                       # 이미 옮겨진 것(새 경로에만 있음)
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
            plan.conflicts.append(f"{f} → {n}: 새 경로에 이미 파일이 있음")
            continue
        if n in targets:
            plan.conflicts.append(f"{f} → {n}: {targets[n]} 와 같은 곳으로 감")
            continue
        if mapper.map_path(n) is not None:
            plan.conflicts.append(f"{f} → {n}: 새 경로가 다시 매핑됨(연쇄)")
            continue
        targets[n] = f
        plan.moves.append((f, n))
    # 옮길 것이 하나도 없을 때만 '이미 이동됨' 으로 센다. 이동 전에는 기존 samples/ 처럼
    # 새 경로 접두사와 겹치는 원래 파일이 있어 역변환이 거짓 양성을 낸다.
    plan.done = inv_candidates if not plan.moves else 0
    return plan


# ─── 참조 치환 ─────────────────────────────────────────────────────────────────
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
        self.new_paths: set[str] = set()   # 이동 후 존재할 경로 (dry-run 에서 채운다)
        # 옛 경로가 하나라도 추적 중이면 '이동 전' 상태다. 이때는 역변환(Fn → Fo)을 쓰지 않는다.
        self.post_move = not any(mapper.map_path(f) is not None for f in files)
        self.manual: list[Note] = []
        self.guess: list[Note] = []

    def _real(self, old: str, new: str) -> bool:
        """옛 경로나 새 경로 중 하나가 실재해야 고친다. 이미 지운 파일을 가리키는
        이력 서술(예: duplicate_png 로 지운 selftest/b4/)은 그대로 둔다."""
        return old in self.known or new in self.known or new in self.new_paths

    def _exists_pre(self, p: str) -> bool:
        return p in self.known or self.m.map_path(p) is not None

    def _valid_post(self, p: str) -> bool:
        """이동이 끝난 트리에서 실재할 경로인가. 이동 뒤에는 known 이 곧 새 트리다."""
        return (p in self.known and self.m.map_path(p) is None) or p in self.new_paths

    def locate(self, p: str) -> tuple[str, str]:
        """현재 추적 경로 p 의 (옛 위치 Fo, 새 위치 Fn)."""
        n = self.m.map_path(p)
        if n is not None:
            return p, n
        o = self.m.inverse(p) if self.post_move else None
        if o is not None and o not in self.files:
            return o, p
        return p, p

    def excluded_spans(self, rel: str, lines: list[str]) -> dict[int, list[tuple[int, int]]]:
        """행 번호 → 제외할 [start, end) 구간들 (매핑표의 이력 표, duplicate_png 의 '지운 파일' 열)."""
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
            if rel == self.map_rel and "옛 경로" in header[0]:
                out[i] = [(0, len(ln))]
            if rel == DUP_PNG:
                cols = [k for k, h in enumerate(header) if "지운 파일" in h]
                if cols:
                    out.setdefault(i, []).extend(_cell_spans(ln, cols))
        return out

    def rewrite_token(self, raw: str, prev_ch: str, fo: str, fn: str, is_src: bool) -> tuple[str | None, str | None]:
        """(새 토큰 또는 None, 메모) — 메모는 'manual'/'guess'/None."""
        tok = raw.rstrip(".,;:")
        while tok.endswith("**"):          # 마크다운 굵게
            tok = tok[:-2].rstrip(".,;:")
        if tok.startswith("**"):
            return None, None              # 다음 호출에서 벗긴 토큰을 따로 본다
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
        # 상대 경로
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
            # #include "common/x.hpp" 처럼 include 루트(01_src) 기준 경로
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
                # 쉼표는 중괄호 밖이면 구분자다
                subs = [(s, raw)] if "{" in raw or "," not in raw else _split_commas(s, raw)
                for ss, sub in subs:
                    prev_ch = ln[ss - 1] if ss > 0 else ""
                    new, note = self.rewrite_token(sub, prev_ch, fo, fn, is_src)
                    if note == "manual":
                        self.manual.append(Note(rel, i + 1, sub, "글롭·자리표시자 — 사람이 고친다"))
                    elif note == "guess":
                        self.guess.append(Note(rel, i + 1, sub, "07_results/ 기준 산문 축약으로 보임"))
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
    """표 행 ln 에서 칸 번호(cols, _split_row 기준)의 [start, end) 구간."""
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
        print(f"경고: 작업트리가 깨끗하지 않다({n}건). 이동·치환 커밋을 다른 변경과 섞지 않도록 주의.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="매핑표대로 git mv 하고 참조를 고친다")
    ap.add_argument("--map", default=DEFAULT_MAP, help=f"매핑표 마크다운 (기본 {DEFAULT_MAP})")
    ap.add_argument("--section", help="이 문자열이 든 헤딩 아래 표만 읽는다 (예: '07_results')")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="이동 목록과 참조 치환 요약만 출력")
    g.add_argument("--apply-moves", action="store_true", help="git mv 만 실행")
    g.add_argument("--apply-refs", action="store_true", help="참조 치환만 실행")
    ap.add_argument("-v", "--verbose", action="store_true", help="치환 내역을 전부 출력")
    a = ap.parse_args(argv)

    map_path = Path(a.map)
    if not map_path.is_absolute():
        map_path = (Path.cwd() / map_path) if (Path.cwd() / map_path).exists() else ROOT / map_path
    if not map_path.exists():
        print(f"매핑표 없음: {map_path}", file=sys.stderr)
        return 2
    try:
        map_rel = str(map_path.resolve().relative_to(ROOT))
    except ValueError:
        map_rel = ""   # 저장소 밖 매핑표(시험용)
    rows = parse_map(map_path, a.section)
    if not rows:
        print("매핑표에서 '옛 경로' 표 행을 찾지 못했다", file=sys.stderr)
        return 2
    mapper = Mapper(rows)
    files = tracked()
    plan = expand_moves(mapper, files)
    print(f"매핑 행 {len(rows)}개 (정규식 {len(mapper.re_rows)}, 접두사 {len(mapper.prefix_rows)}) — {map_path}"
          + (f" §'{a.section}'" if a.section else ""))

    if a.apply_moves or a.apply_refs:
        warn_dirty()

    if plan.conflicts:
        print(f"== 충돌 {len(plan.conflicts)}건 (이동하지 않는다)")
        for c in plan.conflicts:
            print("  " + c)

    if a.dry_run or a.apply_moves:
        print(f"== 이동 {len(plan.moves)}건" + (f" (이미 이동됨 {plan.done}건)" if plan.done else ""))
        for o, n in plan.moves:
            print(f"  {o} → {n}")
        unused = [r for r in rows if not any(mapper.map_path(o) and _row_hits(r, o) for o, _ in plan.moves)]
        for r in unused:
            print(f"  (행 {r.line}: 일치하는 추적 파일 없음 — {r.old})")

    if a.apply_moves:
        if plan.conflicts:
            print("충돌이 있어 이동하지 않았다.", file=sys.stderr)
            return 1
        for o, n in plan.moves:
            (ROOT / n).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-C", REPO_ABS, "mv", "-k", o, n], check=True)
        print(f"git mv {len(plan.moves)}건 완료")
        return 0

    new_paths = {n for _, n in plan.moves}
    results, rw = scan_refs(mapper, map_rel, new_paths)
    total = sum(len(e) for *_, e in results)
    print(f"== 참조 치환: 파일 {len(results)}개, 치환 {total}건")
    for rel, (fo, fn), _, edits in results:
        loc = f"  (파일도 이동: → {fn})" if fo != fn else ""
        print(f"  {rel}: {len(edits)}건{loc}")
        shown = edits if a.verbose else edits[:5]
        for e in shown:
            print(f"      L{e.line}: {e.old}  →  {e.new}")
        if len(edits) > len(shown):
            print(f"      … 외 {len(edits) - len(shown)}건 (-v 로 전부)")
    if rw.manual:
        print(f"== 수동 처리 {len(rw.manual)}건 (글롭·{{}}·<자리표시자> — 자동 치환 안 함)")
        for n in rw.manual:
            print(f"  {n.file}:{n.line}  {n.token}")
    if rw.guess:
        print(f"== 추정 참조 {len(rw.guess)}건 (07_results/ 기준 산문 축약 — 자동 치환 안 함)")
        for n in rw.guess:
            print(f"  {n.file}:{n.line}  {n.token}")

    if a.apply_refs:
        for rel, _, new_text, _ in results:
            (ROOT / rel).write_text(new_text, encoding="utf-8")
        print(f"파일 {len(results)}개 저장")
    return 0


def _row_hits(row: Row, path: str) -> bool:
    if row.is_re:
        return row.rx.fullmatch(path) is not None
    o = row.old.rstrip("/")
    return path == o or path.startswith(o + "/")


if __name__ == "__main__":
    sys.exit(main())
