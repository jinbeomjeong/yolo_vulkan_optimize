#!/usr/bin/env python3
"""저장소 안 모든 `.md` 의 마크다운 링크가 실재하는지 검사한다.

폴더를 옮기거나 문서를 고친 뒤에 돌린다. 시스템 python3 로 돈다(의존성 없음).

    python3 06_tools/python/check_links.py

검사 대상은 `](target)` 형태의 **마크다운 링크**뿐이다.
백틱 안의 경로(`op_point/summary.md` 같은 산문 축약)는 기준이 제각각이라 검사하지 않는다 —
넣으면 오탐이 300건 넘게 나와 진짜 깨짐이 묻힌다.

`파일:행` 형태는 행 번호가 파일 길이 이내인지도 본다.
깨진 링크가 하나라도 있으면 종료 코드 1이다.

    python3 06_tools/python/check_links.py --stale STRUCTURE.md

`--stale MAPFILE` 은 링크 검사 대신, 매핑표(relocate.py 형식)의 **옛 경로**가 아직 남아 있는지 본다.
대상·제외 규칙과 경로 해석은 relocate.py 와 같다(정규식 행은 토큰마다 fullmatch 로 판정하므로
이동 뒤에도 옛 경로를 알아본다). 옛 경로가 하나라도 남아 있으면 종료 코드 1이다.
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
        bad.append((md.relative_to(ROOT), target, "대상 없음"))
        return True
    if line_no is not None and p.is_file():
        try:
            n = sum(1 for _ in p.open("rb"))
        except OSError:
            return True
        if line_no > n:
            bad.append((md.relative_to(ROOT), target, f"행 초과(파일 {n}행)"))
    return True


def stale(map_file: str) -> int:
    """매핑표의 옛 경로가 제외 대상 밖 파일에 남아 있으면 보고한다."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import relocate  # 같은 폴더. 매핑표 파서·토큰 해석을 공유한다

    mp = Path(map_file)
    if not mp.is_absolute():
        mp = mp if mp.exists() else ROOT / mp
    if not mp.exists():
        print(f"매핑표 없음: {mp}")
        return 2
    try:
        map_rel = str(mp.resolve().relative_to(ROOT))
    except ValueError:
        map_rel = ""
    rows = relocate.parse_map(mp)
    if not rows:
        print("매핑표에서 '옛 경로' 표 행을 찾지 못했다")
        return 2
    mapper = relocate.Mapper(rows)
    results, rw = relocate.scan_refs(mapper, map_rel)
    n = 0
    for rel, _, _, edits in results:
        for e in edits:
            print(f"옛 경로 잔존    {rel}:{e.line} -> {e.old}  (새: {e.new})")
            n += 1
    for g in rw.guess:
        print(f"옛 경로 추정    {g.file}:{g.line} -> {g.token}  ({g.why})")
    for g in rw.manual:
        print(f"수동 확인       {g.file}:{g.line} -> {g.token}  (글롭·자리표시자)")
    print(f"---\n매핑 행 {len(rows)}개, 옛 경로 잔존 {n}건, 추정 {len(rw.guess)}건, 수동 확인 {len(rw.manual)}건")
    return 1 if n or rw.guess else 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--stale":
        if len(sys.argv) != 3:
            print("사용법: check_links.py --stale MAPFILE")
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
    print(f"---\n검사 {checked}건, 깨짐 {len(bad)}건")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
