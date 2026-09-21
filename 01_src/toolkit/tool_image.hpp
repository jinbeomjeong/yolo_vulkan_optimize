#pragma once
// Sample image saving for diagnostic tools. Writes PNG if OpenCV is available, PPM otherwise.
//
// Before 2026-09-21, byte-identical copies lived in tools/isp/isp_bench.cpp and tools/vulkan/preproc_bench.cpp.
// Not put in tool_util.hpp to avoid pulling OpenCV headers into every tool.
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

// Save packed RGB888 w×h to path_noext + ".png" (or ".ppm").
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
