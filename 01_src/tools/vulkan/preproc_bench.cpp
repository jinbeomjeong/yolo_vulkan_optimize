// yvo_preproc_bench: HEVC HW 디코드(DRM_PRIME) → Vulkan SAND128 letterbox 전처리 성능 측정
//
// 출력: FPS, GPU 셰이더 시간(timestamp 중앙값/p90), 호스트 submit/wait 시간, CPU ms/frame(getrusage),
//       import 캐시 통계, 검증 레이어 에러·경고 수, 샘플 letterbox PNG 1장.
// 검증 레이어: VK_LOADER_LAYERS_ENABLE='*validation' 또는 --validation 으로 켠다.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"
#include "toolkit/tool_image.hpp"

using yvo::Stats;
using yvo::stats;

namespace {

using yvo::tool::save_rgb;
using yvo::tool::cpu_ms;
struct Args {
    std::string video = "04_data/videos/video_h265.mp4";
    int frames = 0;       // 0 = 전체
    int slots = 8;
    int depth = 2;        // 동시에 GPU에 올려 둘 프레임 수
    std::uint32_t lx = 32, ly = 8;
    yvo::TensorLayout layout = yvo::TensorLayout::Nhwc; // 출력 텐서 레이아웃 (--layout)
    std::string png_dir = "07_results/pipeline_ffmpeg_vulkan_hailo/preproc";
    int png_frame = 100;
    bool validation = false;
    bool sync_validation = false;
    bool no_gpu = false;  // 디코드만 (같은 루프, 기준선)
    bool sweep = false;   // 워크그룹 크기 스윕
    bool export_test = false;
    bool cpu_baseline = false; // CPU 기준 경로(dma-buf mmap → de-SAND → cpu_letterbox) 시간 측정
    int repeat = 1;
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--frames N] [--slots N] [--depth N] [--local X Y] [--png-dir DIR]\n"
                 "          [--png-frame K] [--layout nhwc|nhcw] [--validation] [--sync-validation] [--no-gpu]\n"
                 "          [--sweep] [--export-test] [--repeat N] [--cpu-baseline]\n",
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--frames" && (v = next())) a.frames = std::atoi(v);
        else if (s == "--slots" && (v = next())) a.slots = std::atoi(v);
        else if (s == "--depth" && (v = next())) a.depth = std::atoi(v);
        else if (s == "--local" && i + 2 < argc) {
            a.lx = static_cast<std::uint32_t>(std::atoi(argv[++i]));
            a.ly = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (s == "--png-dir" && (v = next())) a.png_dir = v;
        else if (s == "--png-frame" && (v = next())) a.png_frame = std::atoi(v);
        else if (s == "--repeat" && (v = next())) a.repeat = std::atoi(v);
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
        else return false;
    }
    return a.slots > 0 && a.depth > 0 && a.depth <= a.slots;
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
    // 디코더를 닫기 전에 import 캐시를 비운다(캐시가 dma-buf 참조를 들고 있어 디코더 버퍼가 해제되지 않음)
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
        if (pre && pre->config().out_layout == yvo::TensorLayout::Nhcw) { // 저장 전에 packed로 되돌린다
            std::vector<std::uint8_t> packed(png_copy.size());
            yvo::nhcw_to_rgb_packed(png_copy.data(), lb.dst_w, lb.dst_h, packed.data());
            png_copy.swap(packed);
            name += "_nhcw";
        }
        save_rgb(a.png_dir + name, png_copy.data(), lb.dst_w, lb.dst_h);
    }
    return r;
}

// CPU 기준 경로: 프레임마다 dma-buf mmap → desand_nv12 → cpu_letterbox_nv12_bt709 (단일 스레드, 스칼라)
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
                // 출력 메모리(V3D: HOST_CACHED 없음 = CPU 비캐시 매핑)를 CPU가 읽는 비용. HailoRT가 host 포인터를
                // 복사하는 경우의 참고값.
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
