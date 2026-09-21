#include "isp/nhcw_repack.hpp"

#include <cstddef>

#include <arm_neon.h>

#include "ref/cpu_letterbox.hpp"

namespace yvo::isp {

const char* repack_name(RepackMode m) {
    switch (m) {
    case RepackMode::None: return "none";
    case RepackMode::Cpu: return "cpu";
    case RepackMode::Neon: return "neon";
    }
    return "?";
}

bool parse_repack(const std::string& s, RepackMode& out) {
    if (s == "none") out = RepackMode::None;
    else if (s == "cpu") out = RepackMode::Cpu;
    else if (s == "neon") out = RepackMode::Neon;
    else return false;
    return true;
}

void rgb_packed_to_nhcw_scalar(const std::uint8_t* src, int w, int h, std::uint8_t* dst) {
    yvo::rgb_packed_to_nhcw(src, w, h, dst);
}

// vld3q_u8이 16픽셀치 RGBRGB...를 R/G/B 레지스터 3개로 디인터리브한다. 행마다 R·G·B가
// 연속 구간이므로 그대로 vst1q_u8 3번이면 끝이다. 640 = 40 × 16이라 1080p→640 경로에는 꼬리가 없다.
void rgb_packed_to_nhcw_neon(const std::uint8_t* src, int w, int h, std::uint8_t* dst) {
    const int vec = w & ~15; // 16의 배수까지
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* row = src + static_cast<std::size_t>(y) * w * 3;
        std::uint8_t* r = dst + static_cast<std::size_t>(y) * 3 * w;
        std::uint8_t* g = r + w;
        std::uint8_t* b = g + w;
        int x = 0;
        for (; x < vec; x += 16) {
            const uint8x16x3_t v = vld3q_u8(row + static_cast<std::size_t>(x) * 3);
            vst1q_u8(r + x, v.val[0]);
            vst1q_u8(g + x, v.val[1]);
            vst1q_u8(b + x, v.val[2]);
        }
        for (; x < w; ++x) { // 꼬리 (w가 16의 배수가 아닐 때만)
            r[x] = row[x * 3 + 0];
            g[x] = row[x * 3 + 1];
            b[x] = row[x * 3 + 2];
        }
    }
}

void rgb_packed_to_nhcw(RepackMode mode, const std::uint8_t* src, int w, int h, std::uint8_t* dst) {
    switch (mode) {
    case RepackMode::None: return;
    case RepackMode::Cpu: rgb_packed_to_nhcw_scalar(src, w, h, dst); return;
    case RepackMode::Neon: rgb_packed_to_nhcw_neon(src, w, h, dst); return;
    }
}

} // namespace yvo::isp
