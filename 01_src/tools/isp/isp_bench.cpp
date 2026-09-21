// yvo_isp_bench: HEVC HW decode (DRM_PRIME) → PiSP BE (ISP) resize preprocessing performance measurement
//
// Flag names match yvo_preproc_bench (GPU path). A tool for comparing the two paths under the same conditions.
// Output: FPS, ISP interval wall-clock (median/p90/mean/min/max), CPU ms/frame (getrusage),
//       NHCW transpose cost (scalar/NEON), dma-buf sync cost, output memcpy cost, sample PNG.
//
// Time-axis caution: the ISP's BackendDevice::Run() is synchronous, so everything measured here is wall-clock.
// The GPU shader's 2.1 ms is device time from a Vulkan timestamp query, so do not compare them directly.
// The GPU-side wall-clock is measured separately as the submit+wait median of yvo_preproc_bench --depth 1.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <sys/resource.h>

#ifdef YVO_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "ref/cpu_letterbox.hpp"
#include "core/dmabuf_map.hpp"
#include "ref/sand.hpp"
#include "isp/isp_resizer.hpp"
#include "isp/nhcw_repack.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"
#include "toolkit/tool_image.hpp"

using yvo::Stats;
using yvo::stats;

namespace {

using yvo::tool::save_rgb;
using yvo::tool::cpu_ms;
struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string media = "/dev/media0";
    std::string filter;  // resample filter (empty = smart selection)
    std::string png_dir = YVO_REPO_ROOT "/build/out/yvo_isp_bench";
    std::string csv_label;
    int png_frame = 100;
    int frames = 0;   // 0 = all
    int slots = yvo::kNumSlots;
    int buffers = 12;
    unsigned int branch = 1;
    int repeat = 1;
    int warmup = 60;  // first N frames are excluded from statistics (DVFS, cache warm-up)
    yvo::isp::RepackMode repack = yvo::isp::RepackMode::None;
    bool no_isp = false;       // decode only (same loop, baseline)
    bool cpu_baseline = false; // time of the CPU reference path (mmap → de-SAND → cpu_letterbox)
    bool no_png = false;
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--frames N] [--slots N] [--repeat N] [--warmup N]\n"
                 "          [--repack none|cpu|neon] [--png-dir DIR] [--png-frame K] [--no-png]\n"
                 "          [--no-isp] [--cpu-baseline] [--media /dev/mediaN] [--branch N]\n"
                 "          [--buffers N] [--filter NAME] [--csv-label LABEL]\n",
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--media" && (v = next())) a.media = v;
        else if (s == "--filter" && (v = next())) a.filter = v;
        else if (s == "--png-dir" && (v = next())) a.png_dir = v;
        else if (s == "--png-frame" && (v = next())) a.png_frame = std::atoi(v);
        else if (s == "--csv-label" && (v = next())) a.csv_label = v;
        else if (s == "--frames" && (v = next())) a.frames = std::atoi(v);
        else if (s == "--slots" && (v = next())) a.slots = std::atoi(v);
        else if (s == "--buffers" && (v = next())) a.buffers = std::atoi(v);
        else if (s == "--branch" && (v = next())) a.branch = static_cast<unsigned int>(std::atoi(v));
        else if (s == "--repeat" && (v = next())) a.repeat = std::atoi(v);
        else if (s == "--warmup" && (v = next())) a.warmup = std::atoi(v);
        else if (s == "--repack" && (v = next())) {
            if (!yvo::isp::parse_repack(v, a.repack)) return false;
        }
        else if (s == "--no-isp") a.no_isp = true;
        else if (s == "--cpu-baseline") a.cpu_baseline = true;
        else if (s == "--no-png") a.no_png = true;
        else return false;
    }
    return a.slots > 0 && a.buffers >= a.slots && a.repeat > 0 && a.warmup >= 0;
}

struct RunResult {
    long frames = 0;        // frames counted in statistics (excluding warm-up)
    long total_frames = 0;
    double wall_s = 0, cpu_ms = 0;
    std::vector<double> isp_ms, repack_ms, sync_ms;
    bool ok = true;
};

RunResult run_once(const Args& a, yvo::isp::IspResizer* isp, bool save_png) {
    RunResult r;
    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.extra_hw_frames = 2 * yvo::kNumSlots;
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        r.ok = false;
        return r;
    }
    const int out_w = isp ? isp->out_width() : 0;
    const int out_h = isp ? isp->out_height() : 0;
    std::vector<std::uint8_t> nhcw(isp ? isp->output_size() : 0);
    std::vector<std::uint8_t> png_copy;
    long png_pts = -1;

    // Re-take the clock and rusage when the warm-up interval ends (FPS and CPU count only the measured interval).
    double c0 = cpu_ms();
    auto t0 = std::chrono::steady_clock::now();
    long n = 0;
    while (a.frames <= 0 || n < a.frames) {
        auto f = dec.next();
        if (!f) break;
        const bool measured = n >= a.warmup;
        if (n == a.warmup) {
            c0 = cpu_ms();
            t0 = std::chrono::steady_clock::now();
        }
        if (isp) {
            const int slot = static_cast<int>(n % isp->num_slots());
            const yvo::PreprocOutput out = isp->process(*f, slot);
            if (measured) r.isp_ms.push_back(isp->last_isp_ms());
            const bool want_png = save_png && out.pts == a.png_frame;
            if (a.repack != yvo::isp::RepackMode::None || want_png) {
                double repack_ms = 0, sync_ms = 0;
                const auto s0 = std::chrono::steady_clock::now();
                std::chrono::steady_clock::time_point s3;
                {
                    // The CPU reads a cache-mapped dma-heap buffer, so wrap it in DMA_BUF_IOCTL_SYNC.
                    const yvo::isp::IspResizer::OutputRead lock(*isp, slot);
                    const auto s1 = std::chrono::steady_clock::now();
                    yvo::isp::rgb_packed_to_nhcw(a.repack, lock.data(), out_w, out_h, nhcw.data());
                    const auto s2 = std::chrono::steady_clock::now();
                    if (want_png) {
                        png_copy.assign(lock.data(), lock.data() + out.size);
                        png_pts = static_cast<long>(out.pts);
                    }
                    repack_ms = std::chrono::duration<double, std::milli>(s2 - s1).count();
                    sync_ms = std::chrono::duration<double, std::milli>(s1 - s0).count(); // SYNC_START
                    s3 = std::chrono::steady_clock::now();
                } // lock destroyed = SYNC_END
                sync_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s3).count();
                if (measured && a.repack != yvo::isp::RepackMode::None) {
                    r.repack_ms.push_back(repack_ms);
                    r.sync_ms.push_back(sync_ms);
                }
            }
        }
        ++n;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double c1 = cpu_ms();
    // Clear the input cache before closing the decoder (the cache holds dup fds of decoder dma-bufs).
    if (isp) isp->clear_input_cache();
    if (!dec.error().empty()) {
        std::fprintf(stderr, "decode error after %ld frames: %s\n", n, dec.error().c_str());
        r.ok = false;
    }
    r.total_frames = n;
    r.frames = n > a.warmup ? n - a.warmup : 0;
    r.wall_s = std::chrono::duration<double>(t1 - t0).count();
    r.cpu_ms = c1 - c0;
    if (save_png && png_pts >= 0 && !a.no_png) {
        std::error_code ec;
        std::filesystem::create_directories(a.png_dir, ec);
        save_rgb(a.png_dir + "/isp_letterbox_frame" + std::to_string(png_pts), png_copy.data(), out_w, out_h);
    }
    return r;
}

// CPU reference path: per frame dma-buf mmap → desand_nv12 → cpu_letterbox_nv12_bt709 (single thread, scalar)
int run_cpu_baseline(const Args& a) {
    yvo::FfmpegDrmDecoder dec;
    if (!dec.open(a.video)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }
    std::vector<double> desand_ms, lb_ms;
    std::vector<std::uint8_t> y, uv, rgb;
    const long limit = a.frames > 0 ? a.frames : 300;
    for (long n = 0; n < limit; ++n) {
        auto f = dec.next();
        if (!f) break;
        const std::size_t w = static_cast<std::size_t>(f->width), h = static_cast<std::size_t>(f->height);
        y.resize(w * h);
        uv.resize(w * h / 2);
        const yvo::LetterboxParams lb = yvo::make_letterbox(f->width, f->height);
        rgb.resize(static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3);
        const auto t0 = std::chrono::steady_clock::now();
        {
            yvo::DmaBufReadMap my(f->fd, f->size), muv(f->uv_fd, f->uv_size);
            if (!my.data() || !muv.data() ||
                !yvo::desand_nv12(my.data(), f->size, muv.data(), f->uv_size, yvo::make_sand_layout(*f), y.data(), w,
                                  uv.data(), w)) {
                std::fprintf(stderr, "cpu baseline: desand failed\n");
                return 1;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        yvo::cpu_letterbox_nv12_bt709(y.data(), w, uv.data(), w, f->width, f->height, lb, rgb.data());
        const auto t2 = std::chrono::steady_clock::now();
        desand_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        lb_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    const Stats d = stats(desand_ms), l = stats(lb_ms);
    std::printf("[cpu-baseline] frames=%zu mmap+desand ms: median=%.3f p90=%.3f | letterbox ms: median=%.3f p90=%.3f\n",
                desand_ms.size(), d.median, d.p90, l.median, l.p90);
    return 0;
}

void print_result(const Args& a, const char* tag, const RunResult& r) {
    const Stats i = stats(r.isp_ms), rp = stats(r.repack_ms), sy = stats(r.sync_ms);
    std::printf("[%s] frames=%ld (of %ld, warmup %d) wall=%.3f s fps=%.2f cpu=%.3f ms/frame (total %.0f ms)\n", tag,
                r.frames, r.total_frames, a.warmup, r.wall_s,
                r.wall_s > 0 ? static_cast<double>(r.frames) / r.wall_s : 0.0,
                r.frames ? r.cpu_ms / static_cast<double>(r.frames) : 0.0, r.cpu_ms);
    if (i.n)
        std::printf("[%s] isp ms (wall): median=%.3f p90=%.3f mean=%.3f min=%.3f max=%.3f (n=%zu)\n", tag, i.median,
                    i.p90, i.mean, i.min, i.max, i.n);
    if (rp.n)
        std::printf("[%s] repack(%s) ms: median=%.3f p90=%.3f mean=%.3f min=%.3f max=%.3f (n=%zu) | "
                    "dmabuf sync ms: median=%.3f p90=%.3f\n",
                    tag, yvo::isp::repack_name(a.repack), rp.median, rp.p90, rp.mean, rp.min, rp.max, rp.n, sy.median,
                    sy.p90);
    if (!a.csv_label.empty())
        std::printf("RESULT_CSV,%s,%s,%s,%d,%ld,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%zu,%d\n",
                    a.csv_label.c_str(), r.isp_ms.empty() ? "decode_only" : "isp", yvo::isp::repack_name(a.repack),
                    a.slots, r.frames, r.wall_s, r.wall_s > 0 ? static_cast<double>(r.frames) / r.wall_s : 0.0,
                    r.frames ? r.cpu_ms / static_cast<double>(r.frames) : 0.0, i.median, i.p90, i.mean, i.min, i.max,
                    rp.median, rp.p90, sy.median, i.n, r.ok ? 1 : 0);
}

// Auxiliary measurement: cost of the CPU reading ISP output (cache-mapped dma-heap). The 14 ms CPU of the old NHWC path was not the transpose
// itself but the cost of the CPU reading V3D memory. This measures how much the same work costs on a cache-mapped buffer.
void micro_host_read(const yvo::isp::IspResizer& isp) {
    const std::size_t n = isp.output_size();
    std::vector<std::uint8_t> dst(n), src_copy(n), nhcw(n);
    for (const yvo::isp::RepackMode mode : {yvo::isp::RepackMode::Cpu, yvo::isp::RepackMode::Neon}) {
        std::vector<double> memcpy_ms, sync_ms, repack_dma_ms, repack_heap_ms;
        for (int i = 0; i < 40; ++i) {
            const int slot = i % isp.num_slots();
            const auto s0 = std::chrono::steady_clock::now();
            {
                const yvo::isp::IspResizer::OutputRead lock(isp, slot);
                const auto s1 = std::chrono::steady_clock::now();
                std::memcpy(dst.data(), lock.data(), n);
                const auto s2 = std::chrono::steady_clock::now();
                yvo::isp::rgb_packed_to_nhcw(mode, lock.data(), isp.out_width(), isp.out_height(), nhcw.data());
                const auto s3 = std::chrono::steady_clock::now();
                memcpy_ms.push_back(std::chrono::duration<double, std::milli>(s2 - s1).count());
                sync_ms.push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());
                repack_dma_ms.push_back(std::chrono::duration<double, std::milli>(s3 - s2).count());
            }
            src_copy = dst;
            const auto h0 = std::chrono::steady_clock::now();
            yvo::isp::rgb_packed_to_nhcw(mode, src_copy.data(), isp.out_width(), isp.out_height(), nhcw.data());
            repack_heap_ms.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count());
        }
        const Stats m = stats(memcpy_ms), s = stats(sync_ms), rd = stats(repack_dma_ms), rh = stats(repack_heap_ms);
        if (mode == yvo::isp::RepackMode::Cpu)
            std::printf("host read of ISP output (%zu B, cached dma-heap): memcpy median=%.3f ms p90=%.3f | "
                        "dma-buf sync median=%.3f ms\n",
                        n, m.median, m.p90, s.median);
        std::printf("repack(%-4s): from dma-buf median=%.3f ms | from warm malloc copy median=%.3f ms\n",
                    yvo::isp::repack_name(mode), rd.median, rh.median);
    }
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }
    try {
        if (a.cpu_baseline) return run_cpu_baseline(a);
        if (a.no_isp) {
            for (int i = 0; i < a.repeat; ++i) print_result(a, "decode-only", run_once(a, nullptr, false));
            return 0;
        }

        // Input geometry is read from the video. The ISP config is fixed from the first frame's SAND column height.
        yvo::isp::IspResizerConfig cfg;
        cfg.media_device = a.media;
        cfg.branch = a.branch;
        cfg.num_slots = a.slots;
        cfg.buffer_count = a.buffers;
        cfg.resample_filter = a.filter;
        {
            yvo::FfmpegDrmDecoder probe;
            if (!probe.open(a.video)) {
                std::fprintf(stderr, "decoder open failed: %s\n", probe.error().c_str());
                return 1;
            }
            const auto first = probe.next();
            if (!first) {
                std::fprintf(stderr, "no frames in %s\n", a.video.c_str());
                return 1;
            }
            cfg.src_w = first->width;
            cfg.src_h = first->height;
            cfg.coded_height = first->coded_height;
            cfg.sand_col_height = static_cast<int>(first->sand_col_height);
            cfg.uv_sand_col_height = static_cast<int>(first->uv_sand_col_height);
        } // release frame references before closing the decoder
        yvo::isp::IspResizer isp(cfg);

        std::printf("config: slots=%d buffers=%d repack=%s warmup=%d video=%s\n", a.slots, a.buffers,
                    yvo::isp::repack_name(a.repack), a.warmup, a.video.c_str());
        std::printf("%s", isp.describe().c_str());

        for (int i = 0; i < a.repeat; ++i) {
            const RunResult r = run_once(a, &isp, i == 0);
            print_result(a, "decode+isp", r);
            if (!r.ok) return 1;
        }
        micro_host_read(isp);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
