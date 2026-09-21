// 8x16 bitmap font (ASCII 32~126, 95 glyphs x 16 B = 1520 B). A header so that the GPU overlay shader and
// the CPU reference implementation use **the same table**. No runtime baking, no OpenCV dependency.
//
// Data source: /usr/share/consolefonts/Lat15-VGA16.psf.gz (IBM VGA 8x16 bitmap, public domain).
// Generator script: 06_tools/python/gen_glyph_font_8x16.py — not part of the build. The constants below are the final artifact.
//
// ======================== Rendering contract (this comment is the only reference) ========================
// The GPU compute shader and the CPU reference implementation must produce **bit-identical results** from this comment alone.
// (The overlay bit-exact tests in 05_tests judge against this contract.)
//
//  1) Glyph index
//       g = ch - 32,  valid range 0 <= g < 95.
//       If ch < 32 or ch > 126, treat as g = 0 (space) — i.e. no pixel is drawn.
//
//  2) Storage layout (kGlyph8x16: 4 uint32 = 16 B per glyph)
//       The 8-bit bitmap of row row(0..15) is
//         row_bits = (kGlyph8x16[g * 4 + (row >> 2)] >> ((row & 3) * 8)) & 0xFFu
//       i.e. one uint32 holds 4 rows, and **the least significant byte is the upper row** (little-endian byte order).
//       word 0 = rows 0,1,2,3 / word 1 = rows 4,5,6,7 / word 2 = rows 8..11 / word 3 = rows 12..15.
//
//  3) Row order : row = 0 is the top of the glyph, row = 15 the bottom.
//
//  4) Bit order : the byte's **most significant bit (0x80) is the leftmost column (col = 0)**, the least significant bit (0x01) is col = 7.
//         set(col, row) = ((row_bits >> (7 - col)) & 1u) != 0
//
//  5) Background : pixels with set()==false are **left untouched** (transparent). If a background box is needed, the caller fills it separately.
//
//  6) Integer scale s (>= 1) : for glyph origin (x0, y0), the screen pixels are
//         px = x0 + col * s + i,  py = y0 + row * s + j   (0 <= i, j < s)
//       Integer scaling means no anti-aliasing and no filtering -> bit-exact comparison holds.
//
//  7) Character advance : 8 * s pixels. No extra spacing between glyphs.
//     (Most VGA 8x16 glyphs have column 7 empty, which itself gives natural letter spacing.)
//
//  8) Line height : 16 * s pixels.
// =======================================================================================
#pragma once

#include <cstdint>

namespace yvo {

inline constexpr int kGlyphWidth = 8;
inline constexpr int kGlyphHeight = 16;
inline constexpr int kGlyphFirstChar = 32;   // ' '
inline constexpr int kGlyphLastChar = 126;   // '~'
inline constexpr int kGlyphCount = 95;

// 95 glyphs x (16 rows / 4 rows per word) = 380 words = 1520 B
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

// Contract 1): characters that cannot be represented fold to space (g = 0).
inline constexpr int glyph_index(int ch) {
    return (ch >= kGlyphFirstChar && ch <= kGlyphLastChar) ? (ch - kGlyphFirstChar) : 0;
}

// Contract 2)~4): 8-bit bitmap of row `row` of glyph g. The most significant bit is col = 0.
inline constexpr std::uint32_t glyph_row_bits(int g, int row) {
    return (kGlyph8x16[static_cast<unsigned>(g) * 4u + (static_cast<unsigned>(row) >> 2)] >>
            ((static_cast<unsigned>(row) & 3u) * 8u)) &
           0xFFu;
}

// Contract 4): is pixel (col, row) of glyph g set?
inline constexpr bool glyph_pixel(int g, int col, int row) {
    return ((glyph_row_bits(g, row) >> (7u - static_cast<unsigned>(col))) & 1u) != 0u;
}

// ── Contract self-check ──  If a regenerated table flips row/bit order, compilation breaks here.
// In VGA 8x16, 'A' has row 2 = "...#...."(0x10) and row 7 = "#######."(0xFE).
static_assert(glyph_row_bits(glyph_index('A'), 2) == 0x10u, "row order or word packing differs from the contract");
static_assert(glyph_row_bits(glyph_index('A'), 7) == 0xFEu, "row order or word packing differs from the contract");
static_assert(glyph_pixel(glyph_index('A'), 3, 2), "bit order differs from the contract (MSB = col 0)");
static_assert(!glyph_pixel(glyph_index('A'), 0, 2), "bit order differs from the contract (MSB = col 0)");
static_assert(glyph_index(31) == 0 && glyph_index(127) == 0, "out-of-range characters must fold to space");
static_assert(glyph_row_bits(glyph_index(' '), 8) == 0u, "the space glyph must be all zeros");

} // namespace yvo
