#pragma once
// 진단 도구의 샘플 이미지 저장. OpenCV 가 있으면 PNG, 없으면 PPM 으로 쓴다.
//
// 2026-09-21 이전에는 tools/isp/isp_bench.cpp 와 tools/vulkan/preproc_bench.cpp 에 바이트 동일한 사본이
// 있었다. tool_util.hpp 에 넣지 않은 이유: OpenCV 헤더를 모든 도구에 끌어오지 않으려고.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>

#ifdef YVO_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace yvo::tool {

// packed RGB888 w×h 를 path_noext + ".png"(또는 ".ppm") 로 저장한다.
inline bool save_rgb(const std::string& path_noext, const std::uint8_t* rgb, int w, int h) {
#ifdef YVO_HAVE_OPENCV
    cv::Mat m(h, w, CV_8UC3, const_cast<std::uint8_t*>(rgb));
    cv::Mat bgr;
    cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);
    const std::string p = path_noext + ".png";
    const bool ok = cv::imwrite(p, bgr);
#else
    const std::string p = path_noext + ".ppm";
    std::ofstream o(p, std::ios::binary);
    o << "P6\n" << w << " " << h << "\n255\n";
    o.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(w) * h * 3);
    const bool ok = static_cast<bool>(o);
#endif
    if (ok) std::printf("saved sample: %s\n", p.c_str());
    return ok;
}

}  // namespace yvo::tool
