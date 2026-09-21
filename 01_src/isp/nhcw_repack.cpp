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

// vld3q_u8 deinterleaves 16 pixels of RGBRGB... into 3 registers R/G/B. Since R, G, B are
// contiguous runs per row, 3 vst1q_u8 stores finish the job. 640 = 40 × 16, so the 1080p→640 path has no tail.
void rgb_packed_to_nhcw_neon(const std::uint8_t* src, int w, int h, std::uint8_t* dst) {
    const int vec = w & ~15; // up to a multiple of 16
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
        for (; x < w; ++x) { // tail (only when w is not a multiple of 16)
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
