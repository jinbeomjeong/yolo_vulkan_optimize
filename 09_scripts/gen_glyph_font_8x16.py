#!/usr/bin/env python3
"""01_src/common/glyph_font_8x16.hpp 를 생성한다 (ASCII 32~126, 8x16 비트맵).

원본: /usr/share/consolefonts/Lat15-VGA16.psf.gz (console-setup 패키지의 IBM VGA 8x16 비트맵).
비트맵 글리프 형상 자체는 공개 도메인이다(미국 저작권법상 서체 비트맵은 보호 대상이 아니다).

이 스크립트는 **빌드에 관여하지 않는다.** 최종 산출물은 헤더에 박힌 상수이고, 이 파일은 그 상수를 어떻게
얻었는지 재현하기 위한 기록이다. 폰트를 바꾸려면 다시 돌려 헤더를 덮어쓰고 커밋한다.

사용:
  python3 09_scripts/gen_glyph_font_8x16.py [--psf PATH.psf(.gz)] [--out 01_src/common/glyph_font_8x16.hpp]
  python3 09_scripts/gen_glyph_font_8x16.py --dump ABC   # 글리프를 아스키 아트로 확인만 한다
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import struct
import sys

FIRST_CH = 32
LAST_CH = 126
NUM_GLYPHS = LAST_CH - FIRST_CH + 1  # 95
ROWS = 16

DEFAULT_PSF = "/usr/share/consolefonts/Lat15-VGA16.psf.gz"
DEFAULT_OUT = "01_src/common/glyph_font_8x16.hpp"


def read_psf(path: pathlib.Path) -> bytes:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def parse_psf1(raw: bytes) -> dict[int, bytes]:
    """PSF1 을 파싱해 {unicode 코드포인트: 16바이트 글리프} 를 돌려준다."""
    if raw[0] != 0x36 or raw[1] != 0x04:
        raise SystemExit("PSF1 매직이 아니다 (PSF2 는 아직 지원하지 않는다)")
    mode, charsize = raw[2], raw[3]
    if charsize != ROWS:
        raise SystemExit(f"charsize {charsize} != {ROWS}: 8x16 폰트가 아니다")
    count = 512 if (mode & 0x01) else 256
    glyphs = [raw[4 + i * charsize : 4 + (i + 1) * charsize] for i in range(count)]

    if not (mode & 0x02 or mode & 0x04):
        # 유니코드 표가 없으면 글리프 인덱스 = 코드포인트로 본다(CP437 가정).
        return {i: g for i, g in enumerate(glyphs)}

    tab = raw[4 + count * charsize :]
    vals = struct.unpack(f"<{len(tab) // 2}H", tab[: len(tab) // 2 * 2])
    out: dict[int, bytes] = {}
    gi, cur = 0, []
    for v in vals:
        if v == 0xFFFF:  # 글리프 항목 끝
            for u in cur:
                out.setdefault(u, glyphs[gi])
            cur = []
            gi += 1
        elif v == 0xFFFE:  # 합성 문자열 시작 — 쓰지 않는다
            cur = []
        else:
            cur.append(v)
    return out


def ascii_art(glyph: bytes) -> str:
    return "\n".join("".join("#" if (b >> (7 - x)) & 1 else "." for x in range(8)) for b in glyph)


HEADER_TOP = '''// 8x16 비트맵 폰트 (ASCII 32~126, 95 글리프 x 16 B = 1520 B). GPU 오버레이 셰이더와
// CPU 기준 구현이 **같은 표**를 쓰기 위한 헤더다. 런타임 베이크도, OpenCV 의존도 없다.
//
// 데이터 출처: /usr/share/consolefonts/Lat15-VGA16.psf.gz (IBM VGA 8x16 비트맵, 공개 도메인).
// 생성 스크립트: 09_scripts/gen_glyph_font_8x16.py — 빌드에 관여하지 않는다. 아래 상수가 최종 산출물이다.
//
// ======================== 렌더링 규약 (이 주석이 유일한 기준이다) ========================
// GPU 컴퓨트 셰이더와 CPU 기준 구현은 이 주석만 보고 **비트 단위로 같은 결과**를 내야 한다.
// (05_tests 의 오버레이 비트 일치 테스트가 이 규약을 근거로 판정한다.)
//
//  1) 글리프 인덱스
//       g = ch - 32,  유효 범위 0 <= g < 95.
//       ch < 32 이거나 ch > 126 이면 g = 0 (공백)으로 취급한다 — 즉 아무 픽셀도 찍지 않는다.
//
//  2) 저장 구조 (kGlyph8x16: uint32 4개 = 16 B 가 글리프 하나)
//       행 row(0..15)의 8비트 비트맵은
//         row_bits = (kGlyph8x16[g * 4 + (row >> 2)] >> ((row & 3) * 8)) & 0xFFu
//       즉 uint32 하나에 4행이 들어가고, **최하위 바이트가 더 위쪽 행**이다(리틀엔디언 바이트 순서).
//       word 0 = 행 0,1,2,3 / word 1 = 행 4,5,6,7 / word 2 = 행 8..11 / word 3 = 행 12..15.
//
//  3) 행 순서 : row = 0 이 글리프의 맨 위, row = 15 가 맨 아래.
//
//  4) 비트 순서 : 바이트의 **최상위 비트(0x80)가 가장 왼쪽 열(col = 0)**, 최하위 비트(0x01)가 col = 7.
//         set(col, row) = ((row_bits >> (7 - col)) & 1u) != 0
//
//  5) 배경 : set()==false 인 픽셀은 **건드리지 않는다**(투명). 배경 상자가 필요하면 호출자가 따로 채운다.
//
//  6) 정수 배율 s (>= 1) : 글리프 원점 (x0, y0)에 대해 화면 픽셀은
//         px = x0 + col * s + i,  py = y0 + row * s + j   (0 <= i, j < s)
//       정수 배율이므로 안티에일리어싱도 필터링도 없다 → 비트 일치 판정이 유지된다.
//
//  7) 문자 진행폭 : 8 * s 픽셀. 글리프 사이에 추가 여백을 넣지 않는다.
//     (VGA 8x16 글리프는 대부분 열 7 이 비어 있어 그 자체가 자연스러운 자간이 된다.)
//
//  8) 줄 높이 : 16 * s 픽셀.
// =======================================================================================
#pragma once

#include <cstdint>

namespace yvo {

inline constexpr int kGlyphWidth = 8;
inline constexpr int kGlyphHeight = 16;
inline constexpr int kGlyphFirstChar = 32;   // ' '
inline constexpr int kGlyphLastChar = 126;   // '~'
inline constexpr int kGlyphCount = 95;

// 95 글리프 x (16행 / 4행per워드) = 380 워드 = 1520 B
inline constexpr std::uint32_t kGlyph8x16[95 * 4] = {
'''

HEADER_BOTTOM = '''};

// 규약 1): 표현할 수 없는 문자는 공백(g = 0)으로 접는다.
inline constexpr int glyph_index(int ch) {
    return (ch >= kGlyphFirstChar && ch <= kGlyphLastChar) ? (ch - kGlyphFirstChar) : 0;
}

// 규약 2)~4): 글리프 g 의 row 행 비트맵 8비트. 최상위 비트가 col = 0.
inline constexpr std::uint32_t glyph_row_bits(int g, int row) {
    return (kGlyph8x16[static_cast<unsigned>(g) * 4u + (static_cast<unsigned>(row) >> 2)] >>
            ((static_cast<unsigned>(row) & 3u) * 8u)) &
           0xFFu;
}

// 규약 4): 글리프 g 의 (col, row) 픽셀이 켜져 있는가.
inline constexpr bool glyph_pixel(int g, int col, int row) {
    return ((glyph_row_bits(g, row) >> (7u - static_cast<unsigned>(col))) & 1u) != 0u;
}

// ── 규약 자기검증 ──  표를 다시 생성했을 때 행/비트 순서가 뒤집히면 여기서 컴파일이 깨진다.
// 'A' 는 VGA 8x16 에서 행 2 가 "...#...."(0x10), 행 7 이 "#######."(0xFE) 이다.
static_assert(glyph_row_bits(glyph_index('A'), 2) == 0x10u, "행 순서 또는 워드 패킹이 규약과 다르다");
static_assert(glyph_row_bits(glyph_index('A'), 7) == 0xFEu, "행 순서 또는 워드 패킹이 규약과 다르다");
static_assert(glyph_pixel(glyph_index('A'), 3, 2), "비트 순서가 규약과 다르다 (MSB = col 0)");
static_assert(!glyph_pixel(glyph_index('A'), 0, 2), "비트 순서가 규약과 다르다 (MSB = col 0)");
static_assert(glyph_index(31) == 0 && glyph_index(127) == 0, "범위 밖 문자는 공백으로 접어야 한다");
static_assert(glyph_row_bits(glyph_index(' '), 8) == 0u, "공백 글리프는 전부 0 이어야 한다");

} // namespace yvo
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", default=DEFAULT_PSF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dump", default=None, help="글리프를 아스키 아트로 출력만 하고 끝낸다")
    a = ap.parse_args()

    table = parse_psf1(read_psf(pathlib.Path(a.psf)))
    missing = [c for c in range(FIRST_CH, LAST_CH + 1) if c not in table]
    if missing:
        raise SystemExit(f"글리프 누락: {missing}")

    if a.dump:
        for ch in a.dump:
            print(f"--- {ch!r} (0x{ord(ch):02x}) ---")
            print(ascii_art(table[ord(ch)]))
        return 0

    lines = []
    for g in range(NUM_GLYPHS):
        ch = FIRST_CH + g
        rows = table[ch]
        words = [int.from_bytes(rows[w * 4 : (w + 1) * 4], "little") for w in range(4)]
        label = "space" if ch == 32 else chr(ch)
        lines.append(
            "    " + ", ".join(f"0x{w:08x}u" for w in words) + f",  // {g:2d} '{label}'"
        )

    out = pathlib.Path(a.out)
    out.write_text(HEADER_TOP + "\n".join(lines) + "\n" + HEADER_BOTTOM, encoding="utf-8")
    print(f"wrote {out} ({NUM_GLYPHS} glyphs, {NUM_GLYPHS * ROWS} B of bitmap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
