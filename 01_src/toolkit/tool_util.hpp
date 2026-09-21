#pragma once
// 진단 도구(01_src/tools/, pipeline/main.cpp)가 공유하는 작은 헬퍼.
//
// 2026-09-21 이전에는 아래 함수들이 도구마다 복사돼 있었다(동기화 장치 없음):
//   soc_temperature()    pipeline/main.cpp ↔ tools/isp/isp_pipeline.cpp          (바이트 동일)
//   fmt(), count_open_fds() tools/hailo/hailo_raw_probe.cpp ↔ tools/vulkan/postproc_probe.cpp
//   Nv12Frame, read_nv12_frames() tools/hailo/hailo_selftest.cpp ↔ hailo_raw_probe.cpp (필드가 달랐다)
//   cpu_ms()             pipeline/main.cpp · tools/decode/drm_dump.cpp · tools/isp/{isp_bench,isp_pipeline}.cpp ·
//                        tools/vulkan/preproc_bench.cpp — 5벌. 파이프라인의 대표 지표(CPU ms/frame)를 재는 함수다
//
// 라이브러리 코드는 이 헤더를 include 하지 않는다 — 도구 전용이다.
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

// 이 프로세스의 누적 CPU 시간(user + sys) [ms]. 두 시점의 차를 프레임 수로 나눈 것이 CPU ms/frame 이다.
inline double cpu_ms() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e3 +
           static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e3;
}

// SoC 온도 [°C]. 읽지 못하면 NaN.
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

// 이 프로세스가 연 fd 수. fd 누수 계수에 쓴다.
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

// ffmpeg 로 앞 n 프레임을 NV12 원시 바이트로 읽는다(CPU 디코드 — 진단 도구 전용).
inline std::vector<Nv12Frame> read_nv12_frames(const std::string& video, int n, int w, int h) {
    const std::string cmd = "ffmpeg -nostdin -v error -i '" + video + "' -frames:v " + std::to_string(n) +
                            " -f rawvideo -pix_fmt nv12 -";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("popen(ffmpeg) 실패");
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
