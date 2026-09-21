// 8x16 비트맵 폰트 (ASCII 32~126, 95 글리프 x 16 B = 1520 B). GPU 오버레이 셰이더와
// CPU 기준 구현이 **같은 표**를 쓰기 위한 헤더다. 런타임 베이크도, OpenCV 의존도 없다.
//
// 데이터 출처: /usr/share/consolefonts/Lat15-VGA16.psf.gz (IBM VGA 8x16 비트맵, 공개 도메인).
// 생성 스크립트: 06_tools/python/gen_glyph_font_8x16.py — 빌드에 관여하지 않는다. 아래 상수가 최종 산출물이다.
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
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,  //  0 'space'
    0x3c180000u, 0x18183c3cu, 0x18180018u, 0x00000000u,  //  1 '!'
    0x66666600u, 0x00000024u, 0x00000000u, 0x00000000u,  //  2 '"'
    0x6c000000u, 0x6c6cfe6cu, 0x6c6cfe6cu, 0x00000000u,  //  3 '#'
    0xc67c1818u, 0x067cc0c2u, 0x7cc68606u, 0x00001818u,  //  4 '$'
    0x00000000u, 0x180cc6c2u, 0x86c66030u, 0x00000000u,  //  5 '%'
    0x6c380000u, 0xdc76386cu, 0x76ccccccu, 0x00000000u,  //  6 '&'
    0x30303000u, 0x00000020u, 0x00000000u, 0x00000000u,  //  7 '''
    0x180c0000u, 0x30303030u, 0x0c183030u, 0x00000000u,  //  8 '('
    0x18300000u, 0x0c0c0c0cu, 0x30180c0cu, 0x00000000u,  //  9 ')'
    0x00000000u, 0xff3c6600u, 0x0000663cu, 0x00000000u,  // 10 '*'
    0x00000000u, 0x7e181800u, 0x00001818u, 0x00000000u,  // 11 '+'
    0x00000000u, 0x00000000u, 0x18181800u, 0x00000030u,  // 12 ','
    0x00000000u, 0xfe000000u, 0x00000000u, 0x00000000u,  // 13 '-'
    0x00000000u, 0x00000000u, 0x18180000u, 0x00000000u,  // 14 '.'
    0x00000000u, 0x180c0602u, 0x80c06030u, 0x00000000u,  // 15 '/'
    0x6c380000u, 0xd6d6c6c6u, 0x386cc6c6u, 0x00000000u,  // 16 '0'
    0x38180000u, 0x18181878u, 0x7e181818u, 0x00000000u,  // 17 '1'
    0xc67c0000u, 0x30180c06u, 0xfec6c060u, 0x00000000u,  // 18 '2'
    0xc67c0000u, 0x063c0606u, 0x7cc60606u, 0x00000000u,  // 19 '3'
    0x1c0c0000u, 0xfecc6c3cu, 0x1e0c0c0cu, 0x00000000u,  // 20 '4'
    0xc0fe0000u, 0x06fcc0c0u, 0x7cc60606u, 0x00000000u,  // 21 '5'
    0x60380000u, 0xc6fcc0c0u, 0x7cc6c6c6u, 0x00000000u,  // 22 '6'
    0xc6fe0000u, 0x180c0606u, 0x30303030u, 0x00000000u,  // 23 '7'
    0xc67c0000u, 0xc67cc6c6u, 0x7cc6c6c6u, 0x00000000u,  // 24 '8'
    0xc67c0000u, 0x067ec6c6u, 0x780c0606u, 0x00000000u,  // 25 '9'
    0x00000000u, 0x00001818u, 0x00181800u, 0x00000000u,  // 26 ':'
    0x00000000u, 0x00001818u, 0x30181800u, 0x00000000u,  // 27 ';'
    0x06000000u, 0x6030180cu, 0x060c1830u, 0x00000000u,  // 28 '<'
    0x00000000u, 0x00007e00u, 0x0000007eu, 0x00000000u,  // 29 '='
    0x60000000u, 0x060c1830u, 0x6030180cu, 0x00000000u,  // 30 '>'
    0xc67c0000u, 0x18180cc6u, 0x18180018u, 0x00000000u,  // 31 '?'
    0x7c000000u, 0xdedec6c6u, 0x7cc0dcdeu, 0x00000000u,  // 32 '@'
    0x38100000u, 0xfec6c66cu, 0xc6c6c6c6u, 0x00000000u,  // 33 'A'
    0x66fc0000u, 0x667c6666u, 0xfc666666u, 0x00000000u,  // 34 'B'
    0x663c0000u, 0xc0c0c0c2u, 0x3c66c2c0u, 0x00000000u,  // 35 'C'
    0x6cf80000u, 0x66666666u, 0xf86c6666u, 0x00000000u,  // 36 'D'
    0x66fe0000u, 0x68786862u, 0xfe666260u, 0x00000000u,  // 37 'E'
    0x66fe0000u, 0x68786862u, 0xf0606060u, 0x00000000u,  // 38 'F'
    0x663c0000u, 0xdec0c0c2u, 0x3a66c6c6u, 0x00000000u,  // 39 'G'
    0xc6c60000u, 0xc6fec6c6u, 0xc6c6c6c6u, 0x00000000u,  // 40 'H'
    0x183c0000u, 0x18181818u, 0x3c181818u, 0x00000000u,  // 41 'I'
    0x0c1e0000u, 0x0c0c0c0cu, 0x78ccccccu, 0x00000000u,  // 42 'J'
    0x66e60000u, 0x78786c66u, 0xe666666cu, 0x00000000u,  // 43 'K'
    0x60f00000u, 0x60606060u, 0xfe666260u, 0x00000000u,  // 44 'L'
    0xeec60000u, 0xc6d6fefeu, 0xc6c6c6c6u, 0x00000000u,  // 45 'M'
    0xe6c60000u, 0xcedefef6u, 0xc6c6c6c6u, 0x00000000u,  // 46 'N'
    0xc67c0000u, 0xc6c6c6c6u, 0x7cc6c6c6u, 0x00000000u,  // 47 'O'
    0x66fc0000u, 0x607c6666u, 0xf0606060u, 0x00000000u,  // 48 'P'
    0xc67c0000u, 0xc6c6c6c6u, 0x7cded6c6u, 0x00000e0cu,  // 49 'Q'
    0x66fc0000u, 0x6c7c6666u, 0xe6666666u, 0x00000000u,  // 50 'R'
    0xc67c0000u, 0x0c3860c6u, 0x7cc6c606u, 0x00000000u,  // 51 'S'
    0x7e7e0000u, 0x1818185au, 0x3c181818u, 0x00000000u,  // 52 'T'
    0xc6c60000u, 0xc6c6c6c6u, 0x7cc6c6c6u, 0x00000000u,  // 53 'U'
    0xc6c60000u, 0xc6c6c6c6u, 0x10386cc6u, 0x00000000u,  // 54 'V'
    0xc6c60000u, 0xd6d6c6c6u, 0x6ceefed6u, 0x00000000u,  // 55 'W'
    0xc6c60000u, 0x38387c6cu, 0xc6c66c7cu, 0x00000000u,  // 56 'X'
    0x66660000u, 0x183c6666u, 0x3c181818u, 0x00000000u,  // 57 'Y'
    0xc6fe0000u, 0x30180c86u, 0xfec6c260u, 0x00000000u,  // 58 'Z'
    0x303c0000u, 0x30303030u, 0x3c303030u, 0x00000000u,  // 59 '['
    0x80000000u, 0x3870e0c0u, 0x02060e1cu, 0x00000000u,  // 60 '\'
    0x0c3c0000u, 0x0c0c0c0cu, 0x3c0c0c0cu, 0x00000000u,  // 61 ']'
    0xc66c3810u, 0x00000000u, 0x00000000u, 0x00000000u,  // 62 '^'
    0x00000000u, 0x00000000u, 0x00000000u, 0x0000ff00u,  // 63 '_'
    0x00183030u, 0x00000000u, 0x00000000u, 0x00000000u,  // 64 '`'
    0x00000000u, 0x7c0c7800u, 0x76ccccccu, 0x00000000u,  // 65 'a'
    0x60e00000u, 0x666c7860u, 0x7c666666u, 0x00000000u,  // 66 'b'
    0x00000000u, 0xc0c67c00u, 0x7cc6c0c0u, 0x00000000u,  // 67 'c'
    0x0c1c0000u, 0xcc6c3c0cu, 0x76ccccccu, 0x00000000u,  // 68 'd'
    0x00000000u, 0xfec67c00u, 0x7cc6c0c0u, 0x00000000u,  // 69 'e'
    0x6c380000u, 0x60f06064u, 0xf0606060u, 0x00000000u,  // 70 'f'
    0x00000000u, 0xcccc7600u, 0x7cccccccu, 0x0078cc0cu,  // 71 'g'
    0x60e00000u, 0x66766c60u, 0xe6666666u, 0x00000000u,  // 72 'h'
    0x18180000u, 0x18183800u, 0x3c181818u, 0x00000000u,  // 73 'i'
    0x06060000u, 0x06060e00u, 0x06060606u, 0x003c6666u,  // 74 'j'
    0x60e00000u, 0x786c6660u, 0xe6666c78u, 0x00000000u,  // 75 'k'
    0x18380000u, 0x18181818u, 0x3c181818u, 0x00000000u,  // 76 'l'
    0x00000000u, 0xd6feec00u, 0xc6d6d6d6u, 0x00000000u,  // 77 'm'
    0x00000000u, 0x6666dc00u, 0x66666666u, 0x00000000u,  // 78 'n'
    0x00000000u, 0xc6c67c00u, 0x7cc6c6c6u, 0x00000000u,  // 79 'o'
    0x00000000u, 0x6666dc00u, 0x7c666666u, 0x00f06060u,  // 80 'p'
    0x00000000u, 0xcccc7600u, 0x7cccccccu, 0x001e0c0cu,  // 81 'q'
    0x00000000u, 0x6676dc00u, 0xf0606060u, 0x00000000u,  // 82 'r'
    0x00000000u, 0x60c67c00u, 0x7cc60c38u, 0x00000000u,  // 83 's'
    0x30100000u, 0x3030fc30u, 0x1c363030u, 0x00000000u,  // 84 't'
    0x00000000u, 0xcccccc00u, 0x76ccccccu, 0x00000000u,  // 85 'u'
    0x00000000u, 0x66666600u, 0x183c6666u, 0x00000000u,  // 86 'v'
    0x00000000u, 0xd6c6c600u, 0x6cfed6d6u, 0x00000000u,  // 87 'w'
    0x00000000u, 0x386cc600u, 0xc66c3838u, 0x00000000u,  // 88 'x'
    0x00000000u, 0xc6c6c600u, 0x7ec6c6c6u, 0x00f80c06u,  // 89 'y'
    0x00000000u, 0x18ccfe00u, 0xfec66030u, 0x00000000u,  // 90 'z'
    0x180e0000u, 0x18701818u, 0x0e181818u, 0x00000000u,  // 91 '{'
    0x18180000u, 0x18181818u, 0x18181818u, 0x00000000u,  // 92 '|'
    0x18700000u, 0x180e1818u, 0x70181818u, 0x00000000u,  // 93 '}'
    0xdc760000u, 0x00000000u, 0x00000000u, 0x00000000u,  // 94 '~'
};

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
