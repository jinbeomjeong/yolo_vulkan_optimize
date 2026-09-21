// Broadcom SAND128 NV12 (8bit) → linear NV12 CPU conversion (reference implementation, header-only)
//
// Layout (confirmed on Pi 5 HEVC decoder output with yvo_drm_dump, byte-identical to av_hwframe_transfer_data):
//   - Each plane is split into columns 128 bytes wide, and the columns are laid out contiguously in memory.
//   - One column = 128 bytes x column height (row count). Column stride (col_stride) = 128 * column height bytes.
//   - Y and UV each have their own column height (Pi 5 HEVC: Y 1088, UV 544, separate dma-buf objects).
//   - Y(x, y)   = y_base [y_offset  + (x / 128) * y_col_stride  + y  * 128 + (x % 128)]
//   - UV(x, cy) = uv_base[uv_offset + (x / 128) * uv_col_stride + cy * 128 + (x % 128)]
//     (x is the UV interleaved byte position = luma x coordinate, even U / odd V)
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "core/frame_types.hpp"

namespace yvo {

inline constexpr std::size_t kSandColWidth = 128; // SAND128 column width (bytes)

struct SandNv12Layout {
    std::size_t y_offset = 0;      // Y plane start offset (within the Y buffer, bytes)
    std::size_t y_col_stride = 0;  // stride between Y column starts (bytes) = 128 * Y column height
    std::size_t uv_offset = 0;     // UV plane start offset (within the UV buffer, bytes)
    std::size_t uv_col_stride = 0; // stride between UV column starts (bytes) = 128 * UV column height
    int width = 0;                 // width to convert (pixels, even)
    int height = 0;                // height to convert (rows, even)
};

inline SandNv12Layout make_sand_layout(const DrmFrame& f) {
    SandNv12Layout l;
    l.y_offset = f.y_offset;
    l.y_col_stride = kSandColWidth * static_cast<std::size_t>(f.sand_col_height);
    l.uv_offset = f.uv_offset;
    l.uv_col_stride = kSandColWidth * static_cast<std::size_t>(f.uv_sand_col_height);
    l.width = f.width;
    l.height = f.height;
    return l;
}

// SAND128 NV12 → linear NV12.
// y_src/uv_src: start address and size of the buffer (mapping) holding Y and UV. Pass the same pointer if it is one buffer.
// dst_y: height rows x width bytes (stride dst_y_stride), dst_uv: height/2 rows x width bytes.
// For a layout that would cause out-of-bounds access, writes nothing and returns false.
inline bool desand_nv12(const std::uint8_t* y_src, std::size_t y_src_size,
                        const std::uint8_t* uv_src, std::size_t uv_src_size, const SandNv12Layout& l,
                        std::uint8_t* dst_y, std::size_t dst_y_stride,
                        std::uint8_t* dst_uv, std::size_t dst_uv_stride) {
    if (y_src == nullptr || uv_src == nullptr || dst_y == nullptr || dst_uv == nullptr) return false;
    if (l.width <= 0 || l.height <= 0 || (l.width % 2) != 0 || (l.height % 2) != 0) return false;
    const std::size_t w = static_cast<std::size_t>(l.width);
    const std::size_t h = static_cast<std::size_t>(l.height);
    const std::size_t ch = h / 2;
    if (dst_y_stride < w || dst_uv_stride < w) return false;
    // all plane rows must fit in one column
    if (l.y_col_stride < kSandColWidth * h || l.uv_col_stride < kSandColWidth * ch) return false;

    // bounds check using the maximum byte position read in the last column
    const std::size_t last_col = (w - 1) / kSandColWidth;
    const std::size_t last_x = (w - 1) % kSandColWidth;
    const std::size_t y_end = l.y_offset + last_col * l.y_col_stride + (h - 1) * kSandColWidth + last_x + 1;
    const std::size_t uv_end = l.uv_offset + last_col * l.uv_col_stride + (ch - 1) * kSandColWidth + last_x + 1;
    if (y_end > y_src_size || uv_end > uv_src_size) return false;

    const std::size_t ncols = last_col + 1;
    for (std::size_t c = 0; c < ncols; ++c) {
        const std::size_t x0 = c * kSandColWidth;
        const std::size_t n = std::min(kSandColWidth, w - x0);
        const std::uint8_t* ycol = y_src + l.y_offset + c * l.y_col_stride;
        for (std::size_t y = 0; y < h; ++y)
            std::memcpy(dst_y + y * dst_y_stride + x0, ycol + y * kSandColWidth, n);
        const std::uint8_t* uvcol = uv_src + l.uv_offset + c * l.uv_col_stride;
        for (std::size_t y = 0; y < ch; ++y)
            std::memcpy(dst_uv + y * dst_uv_stride + x0, uvcol + y * kSandColWidth, n);
    }
    return true;
}

} // namespace yvo
