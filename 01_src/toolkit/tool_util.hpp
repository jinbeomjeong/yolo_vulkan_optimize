#pragma once
// Small helpers shared by diagnostic tools (01_src/tools/, pipeline/main.cpp).
//
// Before 2026-09-21, the functions below were copied into each tool (with nothing keeping them in sync):
//   soc_temperature()    pipeline/main.cpp ↔ tools/isp/isp_pipeline.cpp          (byte-identical)
//   fmt(), count_open_fds() tools/hailo/hailo_raw_probe.cpp ↔ tools/vulkan/postproc_probe.cpp
//   Nv12Frame, read_nv12_frames() tools/hailo/hailo_selftest.cpp ↔ hailo_raw_probe.cpp (fields differed)
//   cpu_ms()             pipeline/main.cpp · tools/decode/drm_dump.cpp · tools/isp/{isp_bench,isp_pipeline}.cpp ·
//                        tools/vulkan/preproc_bench.cpp — 5 copies. This is the function that measures the pipeline's headline metric (CPU ms/frame)
//
// Library code does not include this header — it is for tools only.
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <sys/resource.h>

namespace yvo::tool {

// Cumulative CPU time (user + sys) of this process [ms]. The difference between two points divided by the frame count is CPU ms/frame.
inline double cpu_ms() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e3 +
           static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e3;
}

// SoC temperature [°C]. NaN if it cannot be read.
inline double soc_temperature() {
    FILE* p = popen("vcgencmd measure_temp 2>/dev/null", "r");
    if (!p) return std::nan("");
    char buf[128] = {0};
    const bool got = std::fgets(buf, sizeof(buf), p) != nullptr;
    pclose(p);
    if (!got) return std::nan("");
    const char* eq = std::strchr(buf, '=');
    return eq ? std::atof(eq + 1) : std::nan("");
}

inline std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

// Number of fds opened by this process. Used for counting fd leaks.
inline long count_open_fds() {
    long n = 0;
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/proc/self/fd", ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec))
        ++n;
    return n;
}

struct Nv12Frame {
    int w = 0, h = 0;
    std::vector<std::uint8_t> data;  // Y(w*h) + UV(w*h/2)
};

// Read the first n frames as raw NV12 bytes with ffmpeg (CPU decode — diagnostic tools only).
inline std::vector<Nv12Frame> read_nv12_frames(const std::string& video, int n, int w, int h) {
    const std::string cmd = "ffmpeg -nostdin -v error -i '" + video + "' -frames:v " + std::to_string(n) +
                            " -f rawvideo -pix_fmt nv12 -";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("popen(ffmpeg) failed");
    std::vector<Nv12Frame> out;
    const std::size_t fsz = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3 / 2;
    for (int i = 0; i < n; ++i) {
        Nv12Frame f;
        f.w = w;
        f.h = h;
        f.data.resize(fsz);
        if (std::fread(f.data.data(), 1, fsz, p) != fsz) break;
        out.push_back(std::move(f));
    }
    pclose(p);
    return out;
}

}  // namespace yvo::tool
