// ISP output (packed RGB888, NHWC) → NHCW transpose
//
// The ISP can only produce packed RGB. The device input of yolov11m.hef is NHCW (per row R×w, G×w, B×w), so
// someone has to transpose to pass NHCW. Doing it ourselves instead of leaving it to HailoRT's internal CPU conversion
// lets us measure and control the cost. This file is the scalar and NEON implementation of that transpose.
//
//   NHWC idx = (y * w + x) * 3 + c      NHCW idx = (y * 3 + c) * w + x
//
// The only reference implementation of the color conversion is ref/cpu_letterbox.hpp; the scalar version calls its
// rgb_packed_to_nhcw() directly. The NEON version must produce the same result (bit-exact).
//
// If the input is a dma-buf (cached mapping), wrap the call in DMA_BUF_IOCTL_SYNC (IspResizer::OutputRead).
#pragma once

#include <cstdint>
#include <string>

namespace yvo::isp {

enum class RepackMode {
    None, // no transpose (option A: pass NHWC to HailoRT as-is)
    Cpu,  // scalar
    Neon, // NEON vld3q_u8 deinterleave
};

const char* repack_name(RepackMode m);
bool parse_repack(const std::string& s, RepackMode& out);

// src and dst must not overlap; both are w*h*3 bytes.
void rgb_packed_to_nhcw_scalar(const std::uint8_t* src, int w, int h, std::uint8_t* dst);
void rgb_packed_to_nhcw_neon(const std::uint8_t* src, int w, int h, std::uint8_t* dst);
// Does nothing if mode == None.
void rgb_packed_to_nhcw(RepackMode mode, const std::uint8_t* src, int w, int h, std::uint8_t* dst);

} // namespace yvo::isp
