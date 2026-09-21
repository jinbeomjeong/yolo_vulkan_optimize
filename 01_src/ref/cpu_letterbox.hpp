// CPU reference letterbox implementation (header-only, no OpenCV needed). Used as the baseline for GPU preprocessing comparisons (GTest).
//
// Input: linear NV12 (Y plane + interleaved UV plane), BT.709 limited range.
// Output: dst_w × dst_h × 3 packed RGB888 (same as HailoRT UINT8 NHWC input).
//
// Formulas (the GPU shader must follow the same rules):
//   1) Content area: new_w = int(src_w*scale + 0.5), new_h = int(src_h*scale + 0.5) (same as make_letterbox)
//      Pixels outside dx ∈ [pad_x, pad_x+new_w), dy ∈ [pad_y, pad_y+new_h) are R=G=B=pad_value(114).
//   2) Source sample position (pixel-center aligned, same half-pixel rule as cv::resize INTER_LINEAR):
//        sx = (dx - pad_x + 0.5) / scale - 0.5,  sy = (dy - pad_y + 0.5) / scale - 0.5
//   3) Y: bilinear at (sx, sy). Coordinates clamped to [0, w-1]×[0, h-1] (edge replication).
//   4) UV: chroma location left (HEVC default, video metadata chroma_location=left):
//        horizontally at the same position as even luma columns, vertically midway between two luma rows.
//        cx = sx / 2,  cy = (sy + 0.5) / 2 - 0.5, clamped to [0, w/2-1]×[0, h/2-1], then bilinear for U and V separately.
//   5) BT.709 limited → RGB (float):
//        Y' = (Y - 16) * 255/219,  U' = (U - 128) * 255/224,  V' = (V - 128) * 255/224
//        R = Y' + 1.5748 V',  G = Y' - 0.187324 U' - 0.468124 V',  B = Y' + 1.8556 U'
//      Clamp to 0~255, then round (add 0.5 and truncate).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "core/frame_types.hpp"

namespace yvo {

namespace detail {

inline float bilinear_u8(const std::uint8_t* plane, std::size_t pitch, int w, int h, float x, float y,
                         int stride = 1, int offset = 0) {
    x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const std::uint8_t* r0 = plane + static_cast<std::size_t>(y0) * pitch;
    const std::uint8_t* r1 = plane + static_cast<std::size_t>(y1) * pitch;
    const float a = r0[x0 * stride + offset], b = r0[x1 * stride + offset];
    const float c = r1[x0 * stride + offset], d = r1[x1 * stride + offset];
    const float top = a + (b - a) * fx;
    const float bot = c + (d - c) * fx;
    return top + (bot - top) * fy;
}

inline std::uint8_t to_u8(float v) {
    v = std::clamp(v, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(v + 0.5f);
}

} // namespace detail

// NV12 (BT.709 limited) → letterbox RGB888. dst is lb.dst_w*lb.dst_h*3 bytes.
inline void cpu_letterbox_nv12_bt709(const std::uint8_t* y_plane, std::size_t y_pitch, const std::uint8_t* uv_plane,
                                     std::size_t uv_pitch, int src_w, int src_h, const LetterboxParams& lb,
                                     std::uint8_t* dst) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;

    for (int dy = 0; dy < lb.dst_h; ++dy) {
        std::uint8_t* row = dst + static_cast<std::size_t>(dy) * lb.dst_w * 3;
        const bool row_in = dy >= lb.pad_y && dy < lb.pad_y + new_h;
        const float sy = (static_cast<float>(dy - lb.pad_y) + 0.5f) * inv - 0.5f;
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        for (int dx = 0; dx < lb.dst_w; ++dx) {
            std::uint8_t* px = row + dx * 3;
            if (!row_in || dx < lb.pad_x || dx >= lb.pad_x + new_w) {
                px[0] = px[1] = px[2] = lb.pad_value;
                continue;
            }
            const float sx = (static_cast<float>(dx - lb.pad_x) + 0.5f) * inv - 0.5f;
            const float cx = sx * 0.5f;
            const float Y = detail::bilinear_u8(y_plane, y_pitch, src_w, src_h, sx, sy);
            const float U = detail::bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 0);
            const float V = detail::bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 1);
            const float yf = (Y - 16.0f) * (255.0f / 219.0f);
            const float uf = (U - 128.0f) * (255.0f / 224.0f);
            const float vf = (V - 128.0f) * (255.0f / 224.0f);
            px[0] = detail::to_u8(yf + 1.5748f * vf);
            px[1] = detail::to_u8(yf - 0.187324f * uf - 0.468124f * vf);
            px[2] = detail::to_u8(yf + 1.8556f * uf);
        }
    }
}

// packed RGB888 (NHWC) ↔ NHCW transpose. The two buffers must not overlap and their size is w*h*3.
// The only color conversion reference is cpu_letterbox_nv12_bt709(); NHCW reference values are made by transposing its result.
//   NHWC idx = (y * w + x) * 3 + c      NHCW idx = (y * 3 + c) * w + x
inline void rgb_packed_to_nhcw(const std::uint8_t* src, int w, int h, std::uint8_t* dst) {
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* row = src + static_cast<std::size_t>(y) * w * 3;
        std::uint8_t* r = dst + static_cast<std::size_t>(y) * 3 * w;
        std::uint8_t* g = r + w;
        std::uint8_t* b = g + w;
        for (int x = 0; x < w; ++x) {
            r[x] = row[x * 3 + 0];
            g[x] = row[x * 3 + 1];
            b[x] = row[x * 3 + 2];
        }
    }
}

// NHCW → packed RGB888 (inverse transform for visualization, e.g. saving PNGs)
inline void nhcw_to_rgb_packed(const std::uint8_t* src, int w, int h, std::uint8_t* dst) {
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* r = src + static_cast<std::size_t>(y) * 3 * w;
        const std::uint8_t* g = r + w;
        const std::uint8_t* b = g + w;
        std::uint8_t* row = dst + static_cast<std::size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            row[x * 3 + 0] = r[x];
            row[x * 3 + 1] = g[x];
            row[x * 3 + 2] = b[x];
        }
    }
}

} // namespace yvo
