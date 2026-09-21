// ISP 출력(packed RGB888, NHWC) → NHCW 전치
//
// ISP는 packed RGB만 낼 수 있다. yolov11m.hef의 장치 입력은 NHCW(행마다 R×w, G×w, B×w)이므로,
// NHCW로 넘기려면 누군가 전치를 해야 한다. 그 일을 HailoRT의 내부 CPU 변환에 맡기지 않고
// 직접 하면 비용을 측정·제어할 수 있다. 이 파일이 그 전치의 스칼라·NEON 구현이다.
//
//   NHWC idx = (y * w + x) * 3 + c      NHCW idx = (y * 3 + c) * w + x
//
// 색 변환 기준 구현은 ref/cpu_letterbox.hpp 하나뿐이고, 스칼라 버전은 그 안의
// rgb_packed_to_nhcw()를 그대로 부른다. NEON 버전은 같은 결과를 내야 한다(비트 일치).
//
// 입력이 dma-buf(캐시 매핑)면 호출 전에 DMA_BUF_IOCTL_SYNC로 감쌀 것(IspResizer::OutputRead).
#pragma once

#include <cstdint>
#include <string>

namespace yvo::isp {

enum class RepackMode {
    None, // 전치하지 않음 (NHWC 그대로 HailoRT에 넘기는 A안)
    Cpu,  // 스칼라
    Neon, // NEON vld3q_u8 디인터리브
};

const char* repack_name(RepackMode m);
bool parse_repack(const std::string& s, RepackMode& out);

// src와 dst는 겹치면 안 되고 둘 다 w*h*3 바이트다.
void rgb_packed_to_nhcw_scalar(const std::uint8_t* src, int w, int h, std::uint8_t* dst);
void rgb_packed_to_nhcw_neon(const std::uint8_t* src, int w, int h, std::uint8_t* dst);
// mode == None이면 아무것도 하지 않는다.
void rgb_packed_to_nhcw(RepackMode mode, const std::uint8_t* src, int w, int h, std::uint8_t* dst);

} // namespace yvo::isp
