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
// perf_window() (added 2026-09-29) brackets the same CPU window as cpu_ms() for `perf -D -1 --control fifo:...`.
// Callers: pipeline/main.cpp (c0..c1) · tools/gst/gst_pipeline.cpp (c0..cpu_total).
//
// Library code does not include this header — it is for tools only.
#include <cerrno>
#include <chrono>
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

#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <unistd.h>

namespace yvo::tool {

// Cumulative CPU time (user + sys) of this process [ms]. The difference between two points divided by the frame count is CPU ms/frame.
inline double cpu_ms() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e3 +
           static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e3;
}

// Open/close perf's event window at the tool's CPU-window boundaries, so that
//   perf stat|record -D -1 --control fifo:<ctl>[,<ack>] -- env YVO_PERF_CTL=<ctl> [YVO_PERF_ACK=<ack>] <tool> ...
// counts exactly the interval the tool reports as cpu_ms_per_frame. Call perf_window(true) right after the
// start cpu_ms() sample and perf_window(false) right before the end sample.
//  - YVO_PERF_CTL unset: no-op (getenv runs once, no other syscalls).
//  - The FIFOs are opened once, on the first call. perf opens both FIFOs itself (O_RDWR | O_NONBLOCK) before
//    starting the workload, so open(ctl, O_WRONLY) and open(ack, O_RDONLY) find the other end already open and
//    return at once. Run outside perf with YVO_PERF_CTL pointing at a FIFO nobody reads, open() blocks forever.
//  - With YVO_PERF_ACK, each command waits (poll, 1000 ms) for perf's "ack\n"; on timeout one warning is printed
//    and the tool continues. Without it the command is fire-and-forget (perf may switch a few us late).
//  - Never throws. An open/write failure prints one warning and disables all further calls.
//  - Not thread-safe: call from one thread (the tools' main loops do).
inline void perf_window(bool on) {
    struct State {
        int ctl = -1;
        int ack = -1;
        bool active = false;
    };
    static State s = [] {
        State st;
        const char* ctl = std::getenv("YVO_PERF_CTL");
        if (!ctl || !*ctl) return st;
        st.ctl = ::open(ctl, O_WRONLY | O_CLOEXEC);
        if (st.ctl < 0) {
            std::fprintf(stderr, "warning: perf_window: open YVO_PERF_CTL '%s' failed: %s; perf window disabled\n",
                         ctl, std::strerror(errno));
            return st;
        }
        const char* ack = std::getenv("YVO_PERF_ACK");
        if (ack && *ack) {
            st.ack = ::open(ack, O_RDONLY | O_CLOEXEC);
            if (st.ack < 0) {
                std::fprintf(stderr, "warning: perf_window: open YVO_PERF_ACK '%s' failed: %s; perf window disabled\n",
                             ack, std::strerror(errno));
                ::close(st.ctl);
                st.ctl = -1;
                return st;
            }
        }
        st.active = true;
        return st;
    }();
    if (!s.active) return;

    const auto disable_all = [](const char* what) {
        std::fprintf(stderr, "warning: perf_window: %s failed: %s; perf window disabled\n", what, std::strerror(errno));
        ::close(s.ctl);
        if (s.ack >= 0) ::close(s.ack);
        s = State{};
    };
    char buf[64];
    if (s.ack >= 0) {  // drop a late ack left over from an earlier timeout
        pollfd p{s.ack, POLLIN, 0};
        while (::poll(&p, 1, 0) > 0 && (p.revents & POLLIN) && ::read(s.ack, buf, sizeof(buf)) > 0) {}
    }
    const char* cmd = on ? "enable\n" : "disable\n";
    const std::size_t len = std::strlen(cmd);
    ssize_t w;
    do w = ::write(s.ctl, cmd, len);
    while (w < 0 && errno == EINTR);
    if (w != static_cast<ssize_t>(len)) {
        if (w >= 0) errno = EIO;
        disable_all("write to YVO_PERF_CTL");
        return;
    }
    if (s.ack < 0) return;

    // Wait for "ack\n". Reads happen only after POLLIN, so the blocking fd never blocks here.
    using clk = std::chrono::steady_clock;
    const auto deadline = clk::now() + std::chrono::milliseconds(1000);
    std::string got;
    for (;;) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - clk::now()).count();
        pollfd p{s.ack, POLLIN, 0};
        const int r = left > 0 ? ::poll(&p, 1, static_cast<int>(left)) : 0;
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) {
            disable_all("poll on YVO_PERF_ACK");
            return;
        }
        if (r == 0 || !(p.revents & POLLIN)) break;  // timeout, or POLLHUP (perf went away)
        const ssize_t n = ::read(s.ack, buf, sizeof(buf));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        got.append(buf, static_cast<std::size_t>(n));
        if (got.find("ack\n") != std::string::npos) return;
    }
    std::fprintf(stderr, "warning: perf_window: no ack from perf within 1000 ms after '%s'; continuing\n",
                 on ? "enable" : "disable");
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
