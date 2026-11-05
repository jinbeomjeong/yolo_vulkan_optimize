// core_only: the same letterbox kernels with every buffer small enough for L1 (32x18 source, 64x36 content), so data
// cache misses are removed and only the core's throughput on each code shape remains. Content rows only. Diagnostic only.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "ref/cpu_letterbox.hpp"
#include "toolkit/cpu_preproc.hpp"
#include "variants.hpp"

using Clock = std::chrono::steady_clock;

struct Pmu {
    std::vector<int> fds;
    std::vector<std::string> names;
    void add(const char* name, std::uint32_t type, std::uint64_t config) {
        perf_event_attr a{};
        a.size = sizeof(a);
        a.type = type;
        a.config = config;
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        a.read_format = PERF_FORMAT_GROUP;
        a.disabled = fds.empty() ? 1 : 0;
        a.pinned = fds.empty() ? 1 : 0;
        const int fd = static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, fds.empty() ? -1 : fds[0], 0));
        if (fd < 0) throw std::runtime_error(std::string("perf_event_open failed: ") + name);
        fds.push_back(fd);
        names.emplace_back(name);
    }
    void read_all(std::uint64_t* out) const {
        std::uint64_t buf[16];
        if (read(fds[0], buf, sizeof(buf)) < 8) throw std::runtime_error("perf read failed");
        std::copy(buf + 1, buf + 1 + fds.size(), out);
    }
};

int main(int argc, char** argv) {
    const std::string v = argc > 1 ? argv[1] : "ref";
    const int sw = 32, sh = 18;
    // bytes in the video's usual range; pseudo-random so no sample pattern is special
    std::vector<std::uint8_t> Y(sw * sh), UV(sw * sh / 2);
    std::uint32_t s = 12345;
    const auto rnd = [&s] {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    };
    for (auto& b : Y) b = static_cast<std::uint8_t>(16 + rnd() % 220);
    for (auto& b : UV) b = static_cast<std::uint8_t>(90 + rnd() % 76);

    const yvo::LetterboxParams full = yvo::make_letterbox(sw, sh, 64, 64);
    yvo::LetterboxParams lb = full; // content rows only, as ref_letterbox_rows does for a band
    const int new_w = static_cast<int>(sw * lb.scale + 0.5f), new_h = static_cast<int>(sh * lb.scale + 0.5f);
    lb.dst_h = new_h;
    lb.pad_y = 0;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3);
    const std::size_t np = static_cast<std::size_t>(new_w) * new_h;
    std::vector<float> py(np), pu(np), pv(np);
    const std::size_t pitch = sw;

    void (*fn)(const std::uint8_t*, std::size_t, const std::uint8_t*, std::size_t, int, int, const yvo::LetterboxParams&,
               std::uint8_t*) = nullptr;
    if (v == "ref") fn = yvo::cpu_letterbox_nv12_bt709;
    else if (v == "local_params") fn = rc::local_params;
    else if (v == "hoist_y") fn = rc::hoist_y;
    else if (v == "hoist_all") fn = rc::hoist_all;
    else if (v == "local_hoist") fn = rc::local_hoist;
    else if (v == "rowsplit") fn = rc::rowsplit;
    else if (v == "ref_notrap") fn = rc::ref_notrap;
    else if (v == "nobranch") fn = rc::nobranch_content;
    else if (v == "nobranch_nohoist") fn = rc::nobranch_nohoist_content;
    else if (v == "rowbuf") fn = rc::rowbuf_content;
    else if (v != "split" && v != "split_nohoist") throw std::runtime_error("unknown variant");

    Pmu pmu;
    pmu.add("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    pmu.add("inst_retired", PERF_TYPE_RAW, 0x08);
    pmu.add("stall_frontend", PERF_TYPE_RAW, 0x23);
    pmu.add("stall_backend", PERF_TYPE_RAW, 0x24);
    pmu.add("l1d_cache_refill", PERF_TYPE_RAW, 0x03);
    pmu.add("br_mis_pred_retired", PERF_TYPE_RAW, 0x22);
    pmu.add("l1i_cache_refill", PERF_TYPE_RAW, 0x01);
    ioctl(pmu.fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(pmu.fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

    const int reps = 100, outer = 400; // reps * 64*36 = 230,400 px per "frame", same pixel count as the real content
    std::vector<double> ms;
    std::vector<std::array<std::uint64_t, 7>> cnt;
    std::uint64_t c0[7], c1[7];
    for (int o = 0; o < outer; ++o) {
        pmu.read_all(c0);
        const auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (fn) {
                fn(Y.data(), pitch, UV.data(), pitch, sw, sh, lb, rgb.data());
            } else {
                if (v == "split") yvo::tool::split_resize(Y.data(), pitch, UV.data(), pitch, sw, sh, lb, py.data(), pu.data(), pv.data());
                else rc::split_nohoist_resize(Y.data(), pitch, UV.data(), pitch, sw, sh, lb, py.data(), pu.data(), pv.data());
                yvo::tool::split_color(py.data(), pu.data(), pv.data(), new_w, new_h, lb, rgb.data());
            }
        }
        const auto t1 = Clock::now();
        pmu.read_all(c1);
        if (o < 20) continue;
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        std::array<std::uint64_t, 7> d{};
        for (int k = 0; k < 7; ++k) d[static_cast<std::size_t>(k)] = c1[k] - c0[k];
        cnt.push_back(d);
    }
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size() / 2];
    std::uint64_t h = yvo::tool::fnv1a64(rgb.data(), rgb.size());
    const double PX = 230400.0;
    std::printf("[CORE] variant=%s med_ms=%.3f hash=%016llx", v.c_str(), med, static_cast<unsigned long long>(h));
    for (std::size_t k = 0; k < pmu.names.size(); ++k) {
        std::vector<double> x;
        for (const auto& c : cnt) x.push_back(static_cast<double>(c[k]));
        std::sort(x.begin(), x.end());
        std::printf(" %s=%.2f", pmu.names[k].c_str(), x[x.size() / 2] / PX);
    }
    std::printf("\n");
    return 0;
}
