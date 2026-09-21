#!/usr/bin/env python3
"""Generates 01_src/ref/glyph_font_8x16.hpp (ASCII 32~126, 8x16 bitmap).

Source: /usr/share/consolefonts/Lat15-VGA16.psf.gz (IBM VGA 8x16 bitmap from the console-setup package).
The bitmap glyph shapes themselves are public domain (typeface bitmaps are not protected under US copyright law).

This script **is not part of the build.** The final artifact is the constants baked into the header; this file is a record
of how those constants were obtained. To change the font, rerun it to overwrite the header and commit.

Usage:
  python3 06_tools/python/gen_glyph_font_8x16.py [--psf PATH.psf(.gz)] [--out 01_src/ref/glyph_font_8x16.hpp]
  python3 06_tools/python/gen_glyph_font_8x16.py --dump ABC   # only inspect glyphs as ASCII art
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
DEFAULT_OUT = "01_src/ref/glyph_font_8x16.hpp"


def read_psf(path: pathlib.Path) -> bytes:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def parse_psf1(raw: bytes) -> dict[int, bytes]:
    """Parse a PSF1 file and return {unicode code point: 16-byte glyph}."""
    if raw[0] != 0x36 or raw[1] != 0x04:
        raise SystemExit("not a PSF1 magic (PSF2 is not supported yet)")
    mode, charsize = raw[2], raw[3]
    if charsize != ROWS:
        raise SystemExit(f"charsize {charsize} != {ROWS}: not an 8x16 font")
    count = 512 if (mode & 0x01) else 256
    glyphs = [raw[4 + i * charsize : 4 + (i + 1) * charsize] for i in range(count)]

    if not (mode & 0x02 or mode & 0x04):
        # Without a unicode table, treat glyph index = code point (assume CP437).
        return {i: g for i, g in enumerate(glyphs)}

    tab = raw[4 + count * charsize :]
    vals = struct.unpack(f"<{len(tab) // 2}H", tab[: len(tab) // 2 * 2])
    out: dict[int, bytes] = {}
    gi, cur = 0, []
    for v in vals:
        if v == 0xFFFF:  # end of glyph entry
            for u in cur:
                out.setdefault(u, glyphs[gi])
            cur = []
            gi += 1
        elif v == 0xFFFE:  # start of a combining sequence — not used
            cur = []
        else:
            cur.append(v)
    return out


def ascii_art(glyph: bytes) -> str:
    return "\n".join("".join("#" if (b >> (7 - x)) & 1 else "." for x in range(8)) for b in glyph)


HEADER_TOP = '''// 8x16 bitmap font (ASCII 32~126, 95 glyphs x 16 B = 1520 B). A header so that the GPU overlay shader and
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
'''

HEADER_BOTTOM = '''};

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
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", default=DEFAULT_PSF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dump", default=None, help="only print glyphs as ASCII art and exit")
    a = ap.parse_args()

    table = parse_psf1(read_psf(pathlib.Path(a.psf)))
    missing = [c for c in range(FIRST_CH, LAST_CH + 1) if c not in table]
    if missing:
        raise SystemExit(f"missing glyphs: {missing}")

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
