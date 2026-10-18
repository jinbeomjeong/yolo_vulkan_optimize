// Column LUT for the SAND128 bilinear shader (sand_letterbox.comp).
// Header-only. Used by SandLetterbox (vulkan/).
//
// Per output column dx the host precomputes the horizontal sample positions and weights with the same float operations
// as cpu_letterbox_nv12_bt709() (ref/cpu_letterbox.hpp: letterbox_src_coord + bilinear_tap), so the shader only adds
// the row component. Entry layout (8 words = 2 x uvec4 per column, see the sand_letterbox.comp header comment):
//   [0..3] column words of Y x0, Y x1, UV cx0, UV cx1:  ((x >> 7) * col_stride + (x & 127)) >> 2
//   [4]    byte shift pack: y0 | y1 << 8 | uv0 << 16 | uv1 << 24 (each 0/8/16/24)
//   [5]    floatBits(fx)   [6] floatBits(fcx)   [7] 1 if padding column else 0
// Padding columns copy the sample positions of the nearest content column (reads stay valid) and set [7] = 1.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ref/cpu_letterbox.hpp"

namespace yvo::vk {

inline constexpr std::size_t kSandLutWordsPerColumn = 8;

// Inputs that determine the LUT contents (the LUT is rebuilt when any of them changes).
struct SandColumnLutKey {
    int src_w = 0, src_h = 0, dst_w = 0, dst_h = 0, pad_x = 0, pad_y = 0;
    float scale = 0.0f;
    std::uint32_t y_col_stride = 0, uv_col_stride = 0;
    bool operator==(const SandColumnLutKey&) const = default;
};

// Writes cols entries (cols * kSandLutWordsPerColumn words) to lut. cols may exceed k.dst_w (a caller may round it
// up to a multiple of 4 so the last 4-pixel group reads valid entries); such columns are padding columns.
inline void build_sand_column_lut(const SandColumnLutKey& k, int cols, std::uint32_t* lut) {
    const int new_w = static_cast<int>(static_cast<float>(k.src_w) * k.scale + 0.5f);
    const float inv = 1.0f / k.scale;
    const int cw = k.src_w / 2;
    auto y_col = [&](int x) {
        return static_cast<std::uint32_t>(x >> 7) * k.y_col_stride + static_cast<std::uint32_t>(x & 127);
    };
    auto uv_col = [&](int cx) {
        const int x = cx * 2;
        return static_cast<std::uint32_t>(x >> 7) * k.uv_col_stride + static_cast<std::uint32_t>(x & 127);
    };
    for (int dx = 0; dx < cols; ++dx) {
        std::uint32_t* e = lut + static_cast<std::size_t>(dx) * kSandLutWordsPerColumn;
        const bool pad = dx < k.pad_x || dx >= k.pad_x + new_w;
        const int dxs = std::clamp(dx, k.pad_x, k.pad_x + new_w - 1);
        const float sx = letterbox_src_coord(dxs, k.pad_x, inv);
        const BilinearTap tx = bilinear_tap(sx, k.src_w);
        const BilinearTap tc = bilinear_tap(sx * 0.5f, cw); // chroma_location = left: cx = sx / 2
        const std::uint32_t a[4] = {y_col(tx.i0), y_col(tx.i1), uv_col(tc.i0), uv_col(tc.i1)};
        std::uint32_t shifts = 0;
        for (int i = 0; i < 4; ++i) {
            e[i] = a[i] >> 2;
            shifts |= ((a[i] & 3u) * 8u) << (8 * i);
        }
        e[4] = shifts;
        std::memcpy(&e[5], &tx.f, 4);
        std::memcpy(&e[6], &tc.f, 4);
        e[7] = pad ? 1u : 0u;
    }
}

} // namespace yvo::vk
