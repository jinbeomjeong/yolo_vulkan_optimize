// Broadcom SAND128 NV12(8bit) → 선형 NV12 CPU 변환 (기준 구현, 헤더 전용)
//
// 레이아웃 (yvo_drm_dump로 Pi 5 HEVC 디코더 출력에서 확인, av_hwframe_transfer_data와 바이트 일치):
//   - plane마다 이미지를 폭 128바이트 열(column)로 나누고, 열을 메모리에 연속으로 둔다.
//   - 열 하나 = 128바이트 x 열 높이(행 수). 열 간격(col_stride) = 128 * 열 높이 바이트.
//   - Y와 UV는 각자 열 높이를 가진다(Pi 5 HEVC: Y 1088, UV 544, 서로 다른 dma-buf 객체).
//   - Y(x, y)   = y_base [y_offset  + (x / 128) * y_col_stride  + y  * 128 + (x % 128)]
//   - UV(x, cy) = uv_base[uv_offset + (x / 128) * uv_col_stride + cy * 128 + (x % 128)]
//     (x는 UV 인터리브 바이트 위치 = 루마 x 좌표, 짝수 U / 홀수 V)
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "common/frame_types.hpp"

namespace yvo {

inline constexpr std::size_t kSandColWidth = 128; // SAND128 열 폭(바이트)

struct SandNv12Layout {
    std::size_t y_offset = 0;      // Y plane 시작 오프셋(Y 버퍼 안, 바이트)
    std::size_t y_col_stride = 0;  // Y 열 시작 간 간격(바이트) = 128 * Y 열 높이
    std::size_t uv_offset = 0;     // UV plane 시작 오프셋(UV 버퍼 안, 바이트)
    std::size_t uv_col_stride = 0; // UV 열 시작 간 간격(바이트) = 128 * UV 열 높이
    int width = 0;                 // 변환할 폭(픽셀, 짝수)
    int height = 0;                // 변환할 높이(행, 짝수)
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

// SAND128 NV12 → 선형 NV12.
// y_src/uv_src: Y·UV가 들어 있는 버퍼(매핑) 시작 주소와 크기. 한 버퍼면 같은 포인터를 넘긴다.
// dst_y: height 행 x width 바이트(stride dst_y_stride), dst_uv: height/2 행 x width 바이트.
// 경계 밖 접근이 생기는 레이아웃이면 아무것도 쓰지 않고 false를 반환한다.
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
    // 열 하나에 plane 행이 다 들어가야 함
    if (l.y_col_stride < kSandColWidth * h || l.uv_col_stride < kSandColWidth * ch) return false;

    // 마지막 열에서 읽는 최대 바이트 위치로 경계 검사
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
