// yvo_preproc_bench: HEVC HW decode (DRM_PRIME) → Vulkan SAND128 letterbox preprocessing performance measurement
//
// Output: FPS, GPU shader time (timestamp median/p90), host submit/wait time, CPU ms/frame (getrusage),
//       import cache statistics, validation layer error/warning counts, one sample letterbox PNG.
// Validation layer: enable with VK_LOADER_LAYERS_ENABLE='*validation' or --validation.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef YVO_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "ref/cpu_letterbox.hpp"
#include "core/dmabuf_map.hpp"
#include "ref/sand.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"
#include "toolkit/cli_parse.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"
#include "toolkit/tool_image.hpp"
#include "toolkit/cpu_preproc.hpp"
#include "core/dma_heap_alloc.hpp"
#ifdef YVO_HAVE_SWSCALE
#include "toolkit/sws_preproc.hpp"
#endif

using yvo::Stats;
using yvo::stats;

namespace {

using yvo::tool::save_rgb;
using yvo::tool::cpu_ms;
struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    int frames = 0;       // 0 = all
    int slots = 8;
    int depth = 2;        // number of frames kept in flight on the GPU at once
    std::uint32_t lx = 32, ly = 8;
    yvo::TensorLayout layout = yvo::TensorLayout::Nhwc; // output tensor layout (--layout)
    std::string png_dir = YVO_REPO_ROOT "/build/out/yvo_preproc_bench";
    int png_frame = 100;
    bool validation = false;
    bool sync_validation = false;
    bool no_gpu = false;  // decode only (same loop, baseline)
    bool sweep = false;   // workgroup size sweep
    bool export_test = false;
    bool cpu_baseline = false; // time the CPU reference path (dma-buf mmap → de-SAND → cpu_letterbox)
    int repeat = 1;
    // --impl: same-condition CPU vs GPU preprocessing comparison (frame -> 640x640 NHCW uint8, decode excluded)
    std::string impl;               // cpuref | sws-rgb24 | sws-gbrp | gpu (empty = legacy modes)
    int threads = 1;                // CPU worker threads (cpuref bands, sws slice threads + band helpers)
    int mem_threads = 0;            // threads for the memory-bound stages (flatten, transpose); 0 = same as --threads
    int warmup = 60;                // frames excluded from the statistics
    std::string cpu_out = "heap";   // CPU output buffers: heap (cached dma-heap, NPU input kind) | malloc
    std::string cpu_map = "cached"; // decoder dma-buf mapping: cached (persistent mmap) | perframe (mmap per frame)
    std::string sws_flags = "bilinear";
    bool accuracy = false;          // untimed pass: compare every frame with the CPU reference (cpuref, 1 thread)
    std::string frame_csv;          // per-frame timings (optional)
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--frames N] [--slots N] [--depth N] [--local X Y] [--png-dir DIR]\n"
                 "          [--png-frame K] [--layout nhwc|nhcw] [--validation] [--sync-validation] [--no-gpu]\n"
                 "          [--sweep] [--export-test] [--repeat N] [--cpu-baseline]\n"
                 "       %s --impl cpuref|sws-rgb24|sws-gbrp|gpu [--threads N] [--mem-threads N] [--frames N] [--warmup N]\n"
                 "          [--cpu-out heap|malloc] [--cpu-map cached|perframe] [--sws-flags F] [--accuracy]\n"
                 "          [--frame-csv PATH] [--validation]\n",
                 a0,
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--frames" && (v = next())) a.frames = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--slots" && (v = next())) a.slots = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--depth" && (v = next())) a.depth = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--local" && i + 2 < argc) {
            a.lx = static_cast<std::uint32_t>(yvo::tool::arg_number<int>(argv[0], s, argv[++i]));
            a.ly = static_cast<std::uint32_t>(yvo::tool::arg_number<int>(argv[0], s, argv[++i]));
        } else if (s == "--png-dir" && (v = next())) a.png_dir = v;
        else if (s == "--png-frame" && (v = next())) a.png_frame = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--repeat" && (v = next())) a.repeat = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--layout" && (v = next())) {
            const std::string l = v;
            if (l == "nhwc") a.layout = yvo::TensorLayout::Nhwc;
            else if (l == "nhcw") a.layout = yvo::TensorLayout::Nhcw;
            else return false;
        }
        else if (s == "--validation") a.validation = true;
        else if (s == "--sync-validation") a.validation = a.sync_validation = true;
        else if (s == "--no-gpu") a.no_gpu = true;
        else if (s == "--sweep") a.sweep = true;
        else if (s == "--export-test") a.export_test = true;
        else if (s == "--cpu-baseline") a.cpu_baseline = true;
        else if (s == "--impl" && (v = next())) a.impl = v;
        else if (s == "--threads" && (v = next())) a.threads = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--mem-threads" && (v = next())) a.mem_threads = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--warmup" && (v = next())) a.warmup = yvo::tool::arg_number<int>(argv[0], s, v);
        else if (s == "--cpu-out" && (v = next())) a.cpu_out = v;
        else if (s == "--cpu-map" && (v = next())) a.cpu_map = v;
        else if (s == "--sws-flags" && (v = next())) a.sws_flags = v;
        else if (s == "--accuracy") a.accuracy = true;
        else if (s == "--frame-csv" && (v = next())) a.frame_csv = v;
        else return false;
    }
    if (!a.impl.empty() && a.impl != "cpuref" && a.impl != "sws-rgb24" && a.impl != "sws-gbrp" && a.impl != "gpu")
        return false;
    if (a.cpu_out != "heap" && a.cpu_out != "malloc") return false;
    if (a.cpu_map != "cached" && a.cpu_map != "perframe") return false;
    return a.slots > 0 && a.depth > 0 && a.depth <= a.slots && a.threads >= 1 && a.threads <= 16 && a.mem_threads >= 0 &&
           a.mem_threads <= a.threads && a.warmup >= 0;
}

struct RunResult {
    long frames = 0;
    double wall_s = 0, cpu_ms = 0;
    std::vector<double> gpu_ms, submit_us, wait_us;
    bool ok = true;
};

RunResult run_once(const Args& a, yvo::vk::SandLetterbox* pre, bool save_png) {
    RunResult r;
    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.extra_hw_frames = std::max(2 * yvo::kNumSlots, a.depth + 4);
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        r.ok = false;
        return r;
    }
    std::deque<int> inflight;
    std::vector<std::uint8_t> png_copy;
    long png_pts = -1;
    auto finish = [&](int slot) {
        const auto t0 = std::chrono::steady_clock::now();
        const yvo::PreprocOutput out = pre->wait(slot);
        const auto t1 = std::chrono::steady_clock::now();
        r.wait_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (pre->last_gpu_ms() >= 0) r.gpu_ms.push_back(pre->last_gpu_ms());
        if (save_png && out.pts == a.png_frame) {
            png_copy.assign(out.host_ptr, out.host_ptr + out.size);
            png_pts = static_cast<long>(out.pts);
        }
    };

    const double c0 = cpu_ms();
    const auto t0 = std::chrono::steady_clock::now();
    long n = 0;
    while (a.frames <= 0 || n < a.frames) {
        auto f = dec.next();
        if (!f) break;
        if (pre) {
            const int slot = static_cast<int>(n % pre->num_slots());
            while (!inflight.empty() &&
                   (static_cast<int>(inflight.size()) >= a.depth || pre->pending(slot))) {
                finish(inflight.front());
                inflight.pop_front();
            }
            const auto s0 = std::chrono::steady_clock::now();
            pre->submit(*f, slot);
            const auto s1 = std::chrono::steady_clock::now();
            r.submit_us.push_back(std::chrono::duration<double, std::micro>(s1 - s0).count());
            inflight.push_back(slot);
        }
        ++n;
    }
    while (pre && !inflight.empty()) {
        finish(inflight.front());
        inflight.pop_front();
    }
    // Empty the import cache before closing the decoder (the cache holds dma-buf references, so decoder buffers would not be freed)
    if (pre) pre->clear_import_cache();
    const auto t1 = std::chrono::steady_clock::now();
    const double c1 = cpu_ms();
    if (!dec.error().empty()) {
        std::fprintf(stderr, "decode error after %ld frames: %s\n", n, dec.error().c_str());
        r.ok = false;
    }
    r.frames = n;
    r.wall_s = std::chrono::duration<double>(t1 - t0).count();
    r.cpu_ms = c1 - c0;
    if (save_png && png_pts >= 0) {
        std::error_code ec;
        std::filesystem::create_directories(a.png_dir, ec);
        const auto& lb = yvo::make_letterbox(1920, 1080);
        std::string name = "/letterbox_frame" + std::to_string(png_pts);
        if (pre && pre->config().out_layout == yvo::TensorLayout::Nhcw) { // convert back to packed before saving
            std::vector<std::uint8_t> packed(png_copy.size());
            yvo::nhcw_to_rgb_packed(png_copy.data(), lb.dst_w, lb.dst_h, packed.data());
            png_copy.swap(packed);
            name += "_nhcw";
        }
        save_rgb(a.png_dir + name, png_copy.data(), lb.dst_w, lb.dst_h);
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

// ── --impl: CPU vs GPU preprocessing under the same conditions (2026-10-27) ─────────────────────────────────
// Every implementation turns the same decoded DRM_PRIME frame into the same 640x640 NHCW uint8 tensor. Decoding is
// outside the timed window (decoder frame threading off, so no decoder thread runs inside it). Per frame:
//   wall = steady_clock around the whole preprocessing, cpu = CLOCK_PROCESS_CPUTIME_ID delta (all threads),
//   user/sys = getrusage(RUSAGE_SELF) deltas. Stages (ms): map (decoder dma-buf mmap + DMA_BUF_SYNC READ start/end),
//   flatten (SAND -> planar), resize (letterbox resize + color conversion), pad, transpose (packed -> NHCW),
//   sync (output DMA_BUF_SYNC WRITE start/end). GPU: submit / wait (fence) and the shader timestamp.
namespace impl {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double proc_cpu_ms() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) / 1e6;
}

void user_sys_ms(double& u, double& s) {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    u = static_cast<double>(ru.ru_utime.tv_sec) * 1e3 + static_cast<double>(ru.ru_utime.tv_usec) / 1e3;
    s = static_cast<double>(ru.ru_stime.tv_sec) * 1e3 + static_cast<double>(ru.ru_stime.tv_usec) / 1e3;
}

struct StageMs {
    double map = 0, flatten = 0, resize = 0, pad = 0, transpose = 0, sync = 0;
};

// CPU output ring: cached dma-heap buffers (the kind HailoRT gets as NHCW input) or malloc.
class OutRing {
public:
    OutRing(bool heap, std::size_t size, int n) : heap_(heap), size_(size) {
        if (heap_) {
            yvo::DmaHeapConfig cfg;
            cfg.name = "yvo_preproc_bench_cpu_out";
            pool_ = std::make_unique<yvo::DmaHeapPool>(cfg);
            for (int i = 0; i < n; ++i) pool_->allocate(size);
        } else {
            const std::size_t len = (size + 16383) / 16384 * 16384;
            for (int i = 0; i < n; ++i) {
                auto* p = static_cast<std::uint8_t*>(std::aligned_alloc(16384, len));
                if (!p) throw std::runtime_error("aligned_alloc failed");
                std::memset(p, 0, len); // fault the pages in before timing
                mem_.emplace_back(p, &std::free);
            }
        }
    }
    int count() const { return heap_ ? static_cast<int>(pool_->count()) : static_cast<int>(mem_.size()); }
    std::uint8_t* ptr(int i) const { return heap_ ? pool_->at(i).ptr : mem_.at(static_cast<std::size_t>(i)).get(); }
    void begin(int i) const {
        if (heap_) yvo::dma_sync_begin(pool_->at(i).fd, yvo::DmaSyncDir::Write);
    }
    void end(int i) const {
        if (heap_) yvo::dma_sync_end(pool_->at(i).fd, yvo::DmaSyncDir::Write);
    }
    std::string describe() const {
        return heap_ ? "dma-heap " + pool_->heap() + " (exp_name " + yvo::tool::dmabuf_exporter(pool_->at(0).fd) + ")"
                     : std::string("malloc (aligned_alloc 16384)");
    }
    std::size_t size() const { return size_; }

private:
    bool heap_;
    std::size_t size_;
    std::unique_ptr<yvo::DmaHeapPool> pool_;
    std::vector<std::unique_ptr<std::uint8_t, void (*)(void*)>> mem_;
};

// Decoder dma-buf CPU access: mapping (cached or per frame) + DMA_BUF_SYNC START/END READ.
struct InputView {
    const std::uint8_t* y = nullptr;
    const std::uint8_t* uv = nullptr;
};

class InputMapper {
public:
    explicit InputMapper(bool cached) : cached_(cached) {}
    InputView open(const yvo::DrmFrame& f) {
        const bool two = f.uv_fd >= 0 && f.uv_fd != f.fd;
        InputView v;
        if (cached_) {
            v.y = cache_.map(f.fd, f.size);
            v.uv = two ? cache_.map(f.uv_fd, f.uv_size) : v.y;
        } else {
            v.y = map_one(f.fd, f.size);
            v.uv = two ? map_one(f.uv_fd, f.uv_size) : v.y;
        }
        yvo::dma_sync_begin(f.fd, yvo::DmaSyncDir::Read);
        if (two) yvo::dma_sync_begin(f.uv_fd, yvo::DmaSyncDir::Read);
        return v;
    }
    void close(const yvo::DrmFrame& f, const InputView& v) {
        const bool two = f.uv_fd >= 0 && f.uv_fd != f.fd;
        yvo::dma_sync_end(f.fd, yvo::DmaSyncDir::Read);
        if (two) yvo::dma_sync_end(f.uv_fd, yvo::DmaSyncDir::Read);
        if (!cached_) {
            ::munmap(const_cast<std::uint8_t*>(v.y), f.size);
            if (two) ::munmap(const_cast<std::uint8_t*>(v.uv), f.uv_size);
        }
    }
    void clear() { cache_.clear(); }
    std::string stats() const {
        return cached_ ? "cached entries=" + std::to_string(cache_.entries()) + " mmaps=" +
                             std::to_string(cache_.maps_made()) + " hits=" + std::to_string(cache_.hits())
                       : std::string("perframe");
    }

private:
    static const std::uint8_t* map_one(int fd, std::size_t size) {
        void* p = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("input mmap failed");
        return static_cast<const std::uint8_t*>(p);
    }
    bool cached_;
    yvo::tool::DmaBufMapCache cache_;
};

// fn(i) for i in [0, k) using the first k participants of the pool (k == 1: on the caller, no wake-up).
void run_k(yvo::tool::WorkerPool& pool, int k, const std::function<void(int)>& fn) {
    if (k <= 1) {
        fn(0);
        return;
    }
    if (k >= pool.size()) {
        pool.run(fn);
        return;
    }
    pool.run([&](int i) {
        if (i < k) fn(i);
    });
}

class CpuImpl {
public:
    virtual ~CpuImpl() = default;
    // Writes the NHCW tensor of frame f into out slot `slot`.
    virtual void process(const yvo::DrmFrame& f, const OutRing& out, int slot, StageMs& st) = 0;
    virtual std::string describe() const = 0;
    virtual void clear() = 0;
};

// CPU-ref: ref/sand.hpp desand_nv12 + ref/cpu_letterbox.hpp cpu_letterbox_nv12_bt709 + rgb_packed_to_nhcw.
// threads > 1 splits de-SAND by luma rows, letterbox by content rows and the transpose by destination rows
// (byte-identical to threads = 1, checked by --accuracy).
class RefImpl final : public CpuImpl {
public:
    RefImpl(int threads, int mem_threads, bool cached_map)
        : pool_(threads), mem_(mem_threads > 0 ? mem_threads : threads), in_(cached_map) {}
    void process(const yvo::DrmFrame& f, const OutRing& out, int slot, StageMs& st) override {
        const int w = f.width, h = f.height;
        const std::size_t sw = static_cast<std::size_t>(w);
        y_.resize(sw * static_cast<std::size_t>(h));
        uv_.resize(sw * static_cast<std::size_t>(h) / 2);
        const yvo::LetterboxParams lb = yvo::make_letterbox(w, h);
        rgb_.resize(static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3);
        const yvo::SandNv12Layout l = yvo::make_sand_layout(f);
        const int new_h = static_cast<int>(h * lb.scale + 0.5f);
        const int T = pool_.size();
        ok_.assign(static_cast<std::size_t>(mem_), 1);

        const auto t0 = Clock::now();
        const InputView v = in_.open(f);
        const auto t1 = Clock::now();
        run_k(pool_, mem_, [&](int i) {
            const auto [r0, r1] = yvo::tool::band_range(h, mem_, i, 2);
            ok_[static_cast<std::size_t>(i)] = yvo::tool::ref_desand_rows(
                v.y, f.size, v.uv, f.uv_fd >= 0 && f.uv_fd != f.fd ? f.uv_size : f.size, l, r0, r1, y_.data(),
                uv_.data(), sw);
        });
        const auto t2 = Clock::now();
        in_.close(f, v);
        const auto t3 = Clock::now();
        for (char o : ok_)
            if (!o) throw std::runtime_error("desand_nv12: layout out of bounds");
        pool_.run([&](int i) {
            const auto [r0, r1] = yvo::tool::letterbox_row_band(lb, new_h, T, i);
            yvo::tool::ref_letterbox_rows(y_.data(), sw, uv_.data(), sw, w, h, lb, r0, r1, rgb_.data());
        });
        const auto t4 = Clock::now();
        out.begin(slot);
        const auto t5 = Clock::now();
        std::uint8_t* dst = out.ptr(slot);
        run_k(pool_, mem_, [&](int i) {
            const auto [r0, r1] = yvo::tool::band_range(lb.dst_h, mem_, i);
            yvo::tool::ref_nhcw_rows(rgb_.data(), lb.dst_w, r0, r1, dst);
        });
        const auto t6 = Clock::now();
        out.end(slot);
        const auto t7 = Clock::now();
        st.map = ms_between(t0, t1) + ms_between(t2, t3);
        st.flatten = ms_between(t1, t2);
        st.resize = ms_between(t3, t4); // includes the 114 padding (cpu_letterbox_nv12_bt709 writes it)
        st.pad = 0;
        st.transpose = ms_between(t5, t6);
        st.sync = ms_between(t4, t5) + ms_between(t6, t7);
    }
    const std::vector<std::uint8_t>& nv12_y() const { return y_; }
    const std::vector<std::uint8_t>& nv12_uv() const { return uv_; }
    std::string describe() const override {
        return "cpuref threads=" + std::to_string(pool_.size()) + " mem_threads=" + std::to_string(mem_) +
               " map=" + in_.stats();
    }
    void clear() override { in_.clear(); }

private:
    yvo::tool::WorkerPool pool_;
    int mem_;
    InputMapper in_;
    std::vector<std::uint8_t> y_, uv_, rgb_;
    std::vector<char> ok_;
};

#ifdef YVO_HAVE_SWSCALE
// CPU-swscale: rpi-ffmpeg av_rpi_sand_to_planar_{y8,c8} (SAND -> YUV420P) + libswscale resize/color conversion at the
// content size (640x360) + pad + NHCW. rgb24: packed RGB24 temp -> NEON transpose. gbrp: planar G/B/R written straight
// into the NHCW tensor (plane pointers on the R/G/B row segments, linesize 3*dst_w) -> no transpose.
// threads > 1: swscale slice threads (sws_scale_frame, "threads" option) + row bands for SAND -> planar and transpose.
class SwsImpl final : public CpuImpl {
public:
    SwsImpl(bool gbrp, int threads, int mem_threads, bool cached_map, const std::string& flags)
        : gbrp_(gbrp), threads_(threads), flags_(flags), pool_(threads), mem_(mem_threads > 0 ? mem_threads : threads),
          in_(cached_map) {}
    void process(const yvo::DrmFrame& f, const OutRing& out, int slot, StageMs& st) override {
        const int w = f.width, h = f.height;
        const yvo::LetterboxParams lb = yvo::make_letterbox(w, h);
        const int new_w = static_cast<int>(w * lb.scale + 0.5f), new_h = static_cast<int>(h * lb.scale + 0.5f);
        if (!sws_ || w != w_ || h != h_) {
            sws_ = std::make_unique<yvo::tool::SwsResize>(w, h, new_w, new_h,
                                                          gbrp_ ? AV_PIX_FMT_GBRP : AV_PIX_FMT_RGB24, flags_, threads_);
            w_ = w;
            h_ = h;
            const std::size_t sw = static_cast<std::size_t>(w), sh = static_cast<std::size_t>(h);
            y_.assign(sw * sh, 0);
            u_.assign(sw * sh / 4, 0);
            v_.assign(sw * sh / 4, 0);
            rgb_.assign(static_cast<std::size_t>(new_w) * new_h * 3, 0);
        }
        const std::uint8_t* src[3] = {y_.data(), u_.data(), v_.data()};
        const int src_stride[3] = {w, w / 2, w / 2};

        const auto t0 = Clock::now();
        const InputView v = in_.open(f);
        const auto t1 = Clock::now();
        run_k(pool_, mem_, [&](int i) {
            const auto [r0, r1] = yvo::tool::band_range(h, mem_, i, 2);
            yvo::tool::sand_to_yuv420p_rows(v.y + f.y_offset, f.sand_col_height, v.uv + f.uv_offset,
                                            f.uv_sand_col_height, w, r0, r1, y_.data(), u_.data(), v_.data());
        });
        const auto t2 = Clock::now();
        in_.close(f, v);
        const auto t3 = Clock::now();
        std::uint8_t* dst = out.ptr(slot);
        const std::size_t row = static_cast<std::size_t>(lb.dst_w) * 3;
        if (gbrp_) {
            out.begin(slot);
            const auto t4 = Clock::now();
            std::uint8_t* base = dst + static_cast<std::size_t>(lb.pad_y) * row + static_cast<std::size_t>(lb.pad_x);
            std::uint8_t* planes[3] = {base + lb.dst_w, base + 2 * lb.dst_w, base}; // GBRP order: G, B, R
            const int strides[3] = {static_cast<int>(row), static_cast<int>(row), static_cast<int>(row)};
            sws_->scale(src, src_stride, planes, strides);
            const auto t5 = Clock::now();
            yvo::tool::nhcw_fill_pad(dst, lb, new_w, new_h);
            const auto t6 = Clock::now();
            out.end(slot);
            const auto t7 = Clock::now();
            st.resize = ms_between(t4, t5);
            st.pad = ms_between(t5, t6);
            st.transpose = 0;
            st.sync = ms_between(t3, t4) + ms_between(t6, t7);
        } else {
            std::uint8_t* planes[3] = {rgb_.data(), nullptr, nullptr};
            const int strides[3] = {new_w * 3, 0, 0};
            sws_->scale(src, src_stride, planes, strides);
            const auto t4 = Clock::now();
            out.begin(slot);
            const auto t5 = Clock::now();
            yvo::tool::nhcw_fill_pad(dst, lb, new_w, new_h);
            const auto t6 = Clock::now();
            run_k(pool_, mem_, [&](int i) {
                const auto [r0, r1] = yvo::tool::band_range(new_h, mem_, i);
                yvo::tool::packed_to_nhcw_rows(rgb_.data() + static_cast<std::size_t>(r0) * new_w * 3,
                                               static_cast<std::size_t>(new_w) * 3, new_w, r1 - r0, dst, lb.dst_w,
                                               lb.pad_y + r0, lb.pad_x);
            });
            const auto t7 = Clock::now();
            out.end(slot);
            const auto t8 = Clock::now();
            st.resize = ms_between(t3, t4);
            st.pad = ms_between(t5, t6);
            st.transpose = ms_between(t6, t7);
            st.sync = ms_between(t4, t5) + ms_between(t7, t8);
        }
        st.map = ms_between(t0, t1) + ms_between(t2, t3);
        st.flatten = ms_between(t1, t2);
    }
    // Planar result of the last SAND -> YUV420P step (accuracy check against the ref de-SAND)
    const std::vector<std::uint8_t>& plane(int i) const { return i == 0 ? y_ : i == 1 ? u_ : v_; }
    std::string describe() const override {
        return std::string(gbrp_ ? "sws-gbrp" : "sws-rgb24") + " threads=" + std::to_string(threads_) +
               " mem_threads=" + std::to_string(mem_) +
               " sws_threads=" + (sws_ ? std::to_string(sws_->threads()) : std::string("?")) +
               " sws_flags=" + (sws_ ? sws_->flags() : flags_) + " libswscale=" AV_STRINGIFY(LIBSWSCALE_VERSION) +
               " map=" + in_.stats();
    }
    void clear() override { in_.clear(); }

private:
    bool gbrp_;
    int threads_;
    std::string flags_;
    yvo::tool::WorkerPool pool_;
    int mem_;
    InputMapper in_;
    std::unique_ptr<yvo::tool::SwsResize> sws_;
    int w_ = 0, h_ = 0;
    std::vector<std::uint8_t> y_, u_, v_, rgb_;
};
#endif

// NHCW content rectangle compare (rows pad_y..pad_y+new_h, per channel columns pad_x..pad_x+new_w) + whole tensor.
void compare_nhcw(const std::uint8_t* a, const std::uint8_t* b, const yvo::LetterboxParams& lb, int new_w, int new_h,
                  yvo::tool::DiffStats& content, yvo::tool::DiffStats& whole) {
    whole.add(a, b, static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3);
    for (int y = lb.pad_y; y < lb.pad_y + new_h; ++y)
        for (int c = 0; c < 3; ++c) {
            const std::size_t o = (static_cast<std::size_t>(y) * 3 + static_cast<std::size_t>(c)) * lb.dst_w +
                                  static_cast<std::size_t>(lb.pad_x);
            content.add(a + o, b + o, static_cast<std::size_t>(new_w));
        }
}

void print_diff(const char* tag, const yvo::tool::DiffStats& d) {
    std::printf(" %s: bytes=%llu identical=%.6f max_abs=%d mean_abs=%.4f psnr_db=%.2f", tag,
                static_cast<unsigned long long>(d.n), d.identical_frac(), d.max_abs, d.mean_abs(), d.psnr());
}

std::string affinity_list() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return "?";
    std::string s;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) s += (s.empty() ? "" : ",") + std::to_string(c);
    return s;
}

} // namespace impl

int run_impl(const Args& a) {
    using namespace impl;
    const bool gpu = a.impl == "gpu";
#ifndef YVO_HAVE_SWSCALE
    if (a.impl == "sws-rgb24" || a.impl == "sws-gbrp") {
        std::fprintf(stderr, "--impl %s: built without libswscale (optional dependency)\n", a.impl.c_str());
        return 2;
    }
#endif
    const long frames = a.frames > 0 ? a.frames : 600;
    const long total = a.accuracy ? frames : frames + a.warmup;

    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.threads = 1; // frame threading off: no decoder thread runs inside the timed window
    dopt.extra_hw_frames = std::max(2 * yvo::kNumSlots, a.slots + 4);
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }

    std::unique_ptr<yvo::vk::VulkanContext> ctx;
    std::unique_ptr<yvo::vk::SandLetterbox> pre;
    std::unique_ptr<CpuImpl> cpu;
    std::unique_ptr<OutRing> out;
    const yvo::LetterboxParams lb0 = yvo::make_letterbox(dec.width(), dec.height());
    const std::size_t out_size = static_cast<std::size_t>(lb0.dst_w) * lb0.dst_h * 3;
    if (gpu) {
        yvo::vk::VulkanContextOptions copt;
        copt.enable_validation = a.validation;
        copt.validation_sync = a.sync_validation;
        copt.verbose = true;
        ctx = std::make_unique<yvo::vk::VulkanContext>(copt);
        yvo::vk::SandLetterboxConfig cfg;
        cfg.num_slots = a.slots;
        cfg.out_layout = yvo::TensorLayout::Nhcw;
        cfg.export_dmabuf = true; // same output memory kind as yvo_pipeline --mode dmabuf (V3D memory, exported)
        cfg.local_size_x = a.lx;
        cfg.local_size_y = a.ly;
        pre = std::make_unique<yvo::vk::SandLetterbox>(*ctx, cfg);
    } else {
        out = std::make_unique<OutRing>(a.cpu_out == "heap", out_size, 4);
        const bool cached = a.cpu_map == "cached";
        if (a.impl == "cpuref") cpu = std::make_unique<RefImpl>(a.threads, a.mem_threads, cached);
#ifdef YVO_HAVE_SWSCALE
        else cpu = std::make_unique<SwsImpl>(a.impl == "sws-gbrp", a.threads, a.mem_threads, cached, a.sws_flags);
#endif
    }
    // Accuracy reference: CPU-ref, 1 thread, malloc output
    std::unique_ptr<RefImpl> ref;
    std::unique_ptr<OutRing> ref_out;
    if (a.accuracy) {
        ref = std::make_unique<RefImpl>(1, 1, true);
        ref_out = std::make_unique<OutRing>(false, out_size, 1);
    }

    std::FILE* csv = nullptr;
    if (!a.frame_csv.empty()) {
        csv = std::fopen(a.frame_csv.c_str(), "w");
        if (!csv) {
            std::fprintf(stderr, "cannot open --frame-csv %s\n", a.frame_csv.c_str());
            return 1;
        }
        std::fprintf(csv, "frame,wall_ms,cpu_ms,user_ms,sys_ms,map_ms,flatten_ms,resize_ms,pad_ms,transpose_ms,sync_ms,"
                          "submit_ms,wait_ms,gpu_ms\n");
    }

    std::vector<double> wall, cpums, mapv, flat, rsz, padv, trn, syn, sub, wai, gms;
    double user_sum = 0, sys_sum = 0;
    yvo::tool::DiffStats d_content, d_whole;
    long identical_frames = 0;
    std::uint64_t flat_diff = 0; // sws: bytes where SAND -> planar differs from the ref de-SAND (Y + U + V)
    bool flat_checked = false;
    std::uint64_t hash = 1469598103934665603ULL;
    std::vector<std::uint8_t> gpu_copy;
    std::string in_desc;
    long n = 0;
    for (; n < total; ++n) {
        auto f = dec.next();
        if (!f) break;
        if (n == 0) {
            in_desc = "input " + std::to_string(f->width) + "x" + std::to_string(f->height) + " y_fd exp_name=" +
                      yvo::tool::dmabuf_exporter(f->fd) + " size=" + std::to_string(f->size) +
                      " uv_fd exp_name=" + yvo::tool::dmabuf_exporter(f->uv_fd) + " size=" + std::to_string(f->uv_size) +
                      " sand_col_h=" + std::to_string(f->sand_col_height) + "/" + std::to_string(f->uv_sand_col_height);
            std::printf("%s\n", in_desc.c_str());
        }
        const yvo::LetterboxParams lb = yvo::make_letterbox(f->width, f->height);
        StageMs st;
        double s_ms = 0, w_ms = 0, g_ms = -1;
        const std::uint8_t* result = nullptr;
        const int slot = static_cast<int>(n % (gpu ? a.slots : 4));

        double u0, s0, u1, s1;
        user_sys_ms(u0, s0);
        const double c0 = proc_cpu_ms();
        const auto t0 = Clock::now();
        if (gpu) {
            pre->submit(*f, slot);
            const auto ts = Clock::now();
            const yvo::PreprocOutput po = pre->wait(slot);
            const auto tw = Clock::now();
            s_ms = ms_between(t0, ts);
            w_ms = ms_between(ts, tw);
            g_ms = pre->last_gpu_ms();
            result = po.host_ptr;
        } else {
            cpu->process(*f, *out, slot, st);
            result = out->ptr(slot);
        }
        const auto t1 = Clock::now();
        const double c1 = proc_cpu_ms();
        user_sys_ms(u1, s1);

        if (a.accuracy) {
            StageMs rs;
            ref->process(*f, *ref_out, 0, rs);
            const std::uint8_t* got = result;
            if (gpu) { // V3D memory: one memcpy instead of uncached byte loads
                gpu_copy.assign(result, result + out_size);
                got = gpu_copy.data();
            }
            const int new_w = static_cast<int>(f->width * lb.scale + 0.5f);
            const int new_h = static_cast<int>(f->height * lb.scale + 0.5f);
            yvo::tool::DiffStats fc, fw;
            compare_nhcw(got, ref_out->ptr(0), lb, new_w, new_h, fc, fw);
            if (fw.identical == fw.n) {
                ++identical_frames;
            } else if (a.impl == "gpu" || a.impl == "cpuref") { // expected identical: list where it is not
                const std::uint8_t* rp = ref_out->ptr(0);
                long shown = 0;
                for (std::size_t i = 0; i < out_size && shown < 8; ++i)
                    if (got[i] != rp[i]) {
                        const std::size_t row = i / (static_cast<std::size_t>(lb.dst_w) * 3);
                        const std::size_t ch = (i / static_cast<std::size_t>(lb.dst_w)) % 3;
                        const std::size_t x = i % static_cast<std::size_t>(lb.dst_w);
                        std::printf("[mismatch] frame=%ld bytes=%llu row=%zu ch=%zu x=%zu got=%d ref=%d\n", n,
                                    static_cast<unsigned long long>(fw.n - fw.identical), row, ch, x, got[i], rp[i]);
                        ++shown;
                    }
            }
            d_content.merge(fc);
            d_whole.merge(fw);
            hash = yvo::tool::fnv1a64(got, out_size, hash);
#ifdef YVO_HAVE_SWSCALE
            if (const auto* sw = dynamic_cast<const SwsImpl*>(cpu.get())) { // flatten == ref de-SAND, byte for byte
                const std::vector<std::uint8_t>& ry = ref->nv12_y();
                const std::vector<std::uint8_t>& ruv = ref->nv12_uv();
                for (std::size_t i = 0; i < ry.size(); ++i) flat_diff += ry[i] != sw->plane(0)[i];
                for (std::size_t i = 0; i < ruv.size() / 2; ++i)
                    flat_diff += (ruv[2 * i] != sw->plane(1)[i]) + (ruv[2 * i + 1] != sw->plane(2)[i]);
                flat_checked = true;
            }
#endif
            if (n == a.png_frame && !a.png_dir.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(a.png_dir, ec);
                std::vector<std::uint8_t> packed(out_size);
                yvo::nhcw_to_rgb_packed(got, lb.dst_w, lb.dst_h, packed.data());
                const std::string tag = a.impl.rfind("sws", 0) == 0 ? a.impl + "-" + a.sws_flags : a.impl;
                save_rgb(a.png_dir + "/impl_" + tag + "_t" + std::to_string(a.threads) + "_frame" + std::to_string(n),
                         packed.data(), lb.dst_w, lb.dst_h);
            }
            continue;
        }
        if (n < a.warmup) continue;
        const double wms = ms_between(t0, t1), cms = c1 - c0;
        wall.push_back(wms);
        cpums.push_back(cms);
        user_sum += u1 - u0;
        sys_sum += s1 - s0;
        if (gpu) {
            sub.push_back(s_ms);
            wai.push_back(w_ms);
            if (g_ms >= 0) gms.push_back(g_ms);
        } else {
            mapv.push_back(st.map);
            flat.push_back(st.flatten);
            rsz.push_back(st.resize);
            padv.push_back(st.pad);
            trn.push_back(st.transpose);
            syn.push_back(st.sync);
        }
        if (csv)
            std::fprintf(csv, "%ld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", n, wms, cms,
                         u1 - u0, s1 - s0, st.map, st.flatten, st.resize, st.pad, st.transpose, st.sync, s_ms, w_ms,
                         g_ms);
    }
    if (csv) std::fclose(csv);
    if (pre) pre->clear_import_cache();
    if (ref) ref->clear();
    const std::string desc = gpu ? "gpu sand_letterbox.comp local=" + std::to_string(a.lx) + "x" +
                                       std::to_string(a.ly) + " slots=" + std::to_string(a.slots) +
                                       " out=V3D exported dma-buf (NHCW)"
                                 : cpu->describe() + " out=" + out->describe();
    if (cpu) cpu->clear();
    if (!dec.error().empty()) {
        std::fprintf(stderr, "decode error after %ld frames: %s\n", n, dec.error().c_str());
        return 1;
    }
    std::printf("[impl] %s\n[impl] affinity=%s video=%s\n", desc.c_str(), affinity_list().c_str(), a.video.c_str());
    int rc = 0;
    if (a.accuracy) {
        std::printf("[ACCURACY] impl=%s threads=%d frames=%ld identical_frames=%ld", a.impl.c_str(), a.threads, n,
                    identical_frames);
        print_diff("content", d_content);
        print_diff("whole", d_whole);
        std::printf(" hash=%016llx", static_cast<unsigned long long>(hash));
        if (flat_checked) std::printf(" flatten_vs_ref_desand_diff_bytes=%llu", static_cast<unsigned long long>(flat_diff));
        std::printf("\n");
    } else {
        const auto m = [](const std::vector<double>& v) { return stats(v); };
        const Stats W = m(wall), C = m(cpums);
        const double nf = static_cast<double>(wall.size());
        std::printf("[RESULT] impl=%s threads=%d mem_threads=%d cpu_out=%s cpu_map=%s frames=%zu warmup=%d wall_med=%.4f "
                    "wall_p90=%.4f wall_mean=%.4f cpu_med=%.4f cpu_p90=%.4f cpu_mean=%.4f user_mean=%.4f "
                    "sys_mean=%.4f",
                    a.impl.c_str(), a.threads, a.mem_threads > 0 ? a.mem_threads : a.threads, gpu ? "v3d" : a.cpu_out.c_str(), gpu ? "-" : a.cpu_map.c_str(),
                    wall.size(), a.warmup, W.median, W.p90, W.mean, C.median, C.p90, C.mean,
                    nf > 0 ? user_sum / nf : 0.0, nf > 0 ? sys_sum / nf : 0.0);
        const auto pr = [&](const char* k, const std::vector<double>& v) {
            const Stats s = m(v);
            std::printf(" %s_med=%.4f %s_p90=%.4f", k, s.median, k, s.p90);
        };
        if (gpu) {
            pr("submit", sub);
            pr("wait", wai);
            pr("gpu", gms);
        } else {
            pr("map", mapv);
            pr("flatten", flat);
            pr("resize", rsz);
            pr("pad", padv);
            pr("transpose", trn);
            pr("sync", syn);
        }
        std::printf("\n");
        if (wall.empty()) rc = 1;
    }
    if (ctx) {
        std::printf("validation: layer=%s messenger=%s errors=%llu warnings=%llu\n",
                    ctx->validation_layer_active() ? "on" : "off", ctx->debug_messenger_active() ? "on" : "off",
                    static_cast<unsigned long long>(ctx->validation_errors()),
                    static_cast<unsigned long long>(ctx->validation_warnings()));
        if (ctx->validation_errors() != 0) rc = 3;
    }
    return rc;
}

void print_result(const char* tag, const RunResult& r) {
    const Stats g = stats(r.gpu_ms), su = stats(r.submit_us), wu = stats(r.wait_us);
    std::printf("[%s] frames=%ld wall=%.3f s fps=%.1f cpu=%.3f ms/frame (total %.0f ms)\n", tag, r.frames, r.wall_s,
                static_cast<double>(r.frames) / r.wall_s, r.cpu_ms / static_cast<double>(r.frames), r.cpu_ms);
    if (!r.gpu_ms.empty())
        std::printf("[%s] gpu shader ms: median=%.3f p90=%.3f mean=%.3f min=%.3f max=%.3f (n=%zu)\n", tag, g.median,
                    g.p90, g.mean, g.min, g.max, r.gpu_ms.size());
    if (!r.submit_us.empty())
        std::printf("[%s] host submit us: median=%.1f p90=%.1f | wait us: median=%.1f p90=%.1f\n", tag, su.median,
                    su.p90, wu.median, wu.p90);
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }
    try {
        if (!a.impl.empty()) return run_impl(a);
        if (a.cpu_baseline) return run_cpu_baseline(a);
        if (a.no_gpu) {
            for (int i = 0; i < a.repeat; ++i) print_result("decode-only", run_once(a, nullptr, false));
            return 0;
        }
        yvo::vk::VulkanContextOptions copt;
        copt.enable_validation = a.validation;
        copt.validation_sync = a.sync_validation;
        copt.verbose = true;
        yvo::vk::VulkanContext ctx(copt);

        if (a.export_test) {
            yvo::vk::SandLetterboxConfig cfg;
            cfg.num_slots = a.slots;
            cfg.out_layout = a.layout;
            cfg.export_dmabuf = true;
            yvo::vk::SandLetterbox pre(ctx, cfg);
            for (int s = 0; s < pre.num_slots(); ++s) {
                const int fd = pre.export_dmabuf_fd(s);
                const off_t sz = lseek(fd, 0, SEEK_END);
                std::printf("export slot %d: dma-buf fd=%d size=%lld host_ptr=%p\n", s, fd, static_cast<long long>(sz),
                            static_cast<void*>(pre.host_ptr(s)));
                close(fd);
            }
            Args b = a;
            b.frames = a.frames > 0 ? a.frames : 200;
            print_result("export-mode", run_once(b, &pre, false));
        }

        if (a.sweep) {
            const std::uint32_t cand[][2] = {{8, 8}, {16, 4}, {16, 8}, {32, 2}, {32, 4}, {32, 8}, {64, 1},
                                             {64, 2}, {64, 4}, {16, 16}, {160, 1}, {128, 2}};
            Args b = a;
            b.frames = a.frames > 0 ? a.frames : 300;
            for (const auto& c : cand) {
                yvo::vk::SandLetterboxConfig cfg;
                cfg.num_slots = a.slots;
                cfg.out_layout = a.layout;
                cfg.local_size_x = c[0];
                cfg.local_size_y = c[1];
                yvo::vk::SandLetterbox pre(ctx, cfg);
                const RunResult r = run_once(b, &pre, false);
                const Stats g = stats(r.gpu_ms);
                std::printf("sweep local=%ux%u (%u inv) layout=%s: gpu median=%.3f p90=%.3f ms, fps=%.1f\n", c[0],
                            c[1], c[0] * c[1], a.layout == yvo::TensorLayout::Nhcw ? "nhcw" : "nhwc", g.median, g.p90,
                            static_cast<double>(r.frames) / r.wall_s);
            }
        } else {
            yvo::vk::SandLetterboxConfig cfg;
            cfg.num_slots = a.slots;
            cfg.out_layout = a.layout;
            cfg.local_size_x = a.lx;
            cfg.local_size_y = a.ly;
            yvo::vk::SandLetterbox pre(ctx, cfg);
            std::printf("config: slots=%d depth=%d local=%ux%u layout=%s video=%s\n", a.slots, a.depth, a.lx, a.ly,
                        a.layout == yvo::TensorLayout::Nhcw ? "nhcw" : "nhwc", a.video.c_str());
            for (int i = 0; i < a.repeat; ++i) {
                const RunResult r = run_once(a, &pre, i == 0);
                print_result("decode+gpu", r);
                if (!r.ok) return 1;
            }
            {
                // Cost of the CPU reading output memory (V3D: no HOST_CACHED = uncached CPU mapping). Reference value for
                // the case where HailoRT copies from a host pointer.
                std::vector<std::uint8_t> dst(pre.output_size());
                std::vector<double> ms;
                for (int i = 0; i < 20; ++i) {
                    const auto t0 = std::chrono::steady_clock::now();
                    std::memcpy(dst.data(), pre.host_ptr(i % pre.num_slots()), dst.size());
                    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                const Stats m = stats(ms);
                std::printf("host read of output (memcpy %zu B from mapped slot): median=%.3f ms p90=%.3f ms\n",
                            dst.size(), m.median, m.p90);
            }
            std::printf("import cache: entries=%zu imports=%llu hits=%llu\n", pre.import_cache().entries(),
                        static_cast<unsigned long long>(pre.import_cache().imports()),
                        static_cast<unsigned long long>(pre.import_cache().hits()));
        }
        std::printf("validation: layer=%s messenger=%s errors=%llu warnings=%llu\n",
                    ctx.validation_layer_active() ? "on" : "off", ctx.debug_messenger_active() ? "on" : "off",
                    static_cast<unsigned long long>(ctx.validation_errors()),
                    static_cast<unsigned long long>(ctx.validation_warnings()));
        return ctx.validation_errors() == 0 ? 0 : 3;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
