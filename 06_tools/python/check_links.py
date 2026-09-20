#!/usr/bin/env python3
"""저장소 안 모든 `.md` 의 마크다운 링크가 실재하는지 검사한다.

폴더를 옮기거나 문서를 고친 뒤에 돌린다. 시스템 python3 로 돈다(의존성 없음).

    python3 06_tools/python/check_links.py

검사 대상은 `](target)` 형태의 **마크다운 링크**뿐이다.
백틱 안의 경로(`op_point/summary.md` 같은 산문 축약)는 기준이 제각각이라 검사하지 않는다 —
넣으면 오탐이 300건 넘게 나와 진짜 깨짐이 묻힌다.

`파일:행` 형태는 행 번호가 파일 길이 이내인지도 본다.
깨진 링크가 하나라도 있으면 종료 코드 1이다.
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


def main() -> int:
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
