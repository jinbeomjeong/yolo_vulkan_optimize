// 추론 결과 일치: CPU 전처리 → HailoRT vs GPU 전처리(ptr/dmabuf, NHWC/NHCW) → HailoRT
//
// 같은 DRM_PRIME 프레임을 네 경로로 NPU에 넣는다.
//   CPU    : dma-buf mmap → desand_nv12 → cpu_letterbox_nv12_bt709 → 페이지 정렬 malloc 버퍼 → set_buffer
//   GPU ptr: SandLetterbox(export_dmabuf=true) 출력의 host 매핑 포인터 → set_buffer
//   GPU dmabuf: 같은 출력 메모리를 export한 dma-buf fd(map_dmabuf 사전 매핑) → set_dma_buffer
//   GPU dmabuf NHCW: 셰이더가 NHCW로 직접 쓴 출력을 dma-buf로 → NHCW 러너(set_dma_buffer)
// 앞 세 경로는 NHWC 러너 하나로 한 번에 돌리고, NHCW는 러너 설정이 달라 두 번째 패스로 따로 돌린다
// (Hailo 장치가 하나뿐이라 러너를 동시에 두 개 만들 수 없다).
// 기준: 같은 클래스·IoU >= 0.9로 매칭된 검출 수 / max(CPU 검출 수, GPU 검출 수) >= 0.95 (프레임 합계).
// 프레임 수: 환경변수 YVO_DET_FRAMES(기본 100). 영상: YVO_TEST_VIDEO(기본 04_data/videos/video_h265.mp4).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "ref/detection_match.hpp"

#include "ref/cpu_letterbox.hpp"
#include "core/dmabuf_map.hpp"
#include "ref/sand.hpp"
#include "hailo/hailo_runner.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

namespace {

using yvo::detmatch::kMinMatchRatio;
using yvo::detmatch::match;
using yvo::detmatch::MatchResult;

enum Path { kCpu = 0, kGpuPtr = 1, kGpuDmabuf = 2, kGpuDmabufNhcw = 3, kNumPaths = 4 };
const char* kPathName[kNumPaths] = {"cpu", "gpu_ptr", "gpu_dmabuf", "gpu_dmabuf_nhcw"};

bool cpu_letterbox(const yvo::DrmFrame& f, const yvo::LetterboxParams& lb, std::uint8_t* out) {
    const std::size_t w = static_cast<std::size_t>(f.width), h = static_cast<std::size_t>(f.height);
    std::vector<std::uint8_t> y(w * h), uv(w * h / 2);
    {
        yvo::DmaBufReadMap my(f.fd, f.size);
        std::unique_ptr<yvo::DmaBufReadMap> muv_own;
        const yvo::DmaBufReadMap* muv = &my;
        const bool two = f.uv_fd >= 0 && f.uv_fd != f.fd;
        if (two) {
            muv_own = std::make_unique<yvo::DmaBufReadMap>(f.uv_fd, f.uv_size);
            muv = muv_own.get();
        }
        if (!my.data() || !muv->data()) return false;
        if (!yvo::desand_nv12(my.data(), f.size, muv->data(), two ? f.uv_size : f.size, yvo::make_sand_layout(f),
                              y.data(), w, uv.data(), w))
            return false;
    }
    yvo::cpu_letterbox_nv12_bt709(y.data(), w, uv.data(), w, f.width, f.height, lb, out);
    return true;
}

struct PageBuf {
    std::uint8_t* p = nullptr;
    explicit PageBuf(std::size_t size) {
        const std::size_t rounded = (size + 4095) / 4096 * 4096;
        p = static_cast<std::uint8_t*>(std::aligned_alloc(4096, rounded));
    }
    ~PageBuf() { std::free(p); }
    PageBuf(const PageBuf&) = delete;
    PageBuf& operator=(const PageBuf&) = delete;
};

class DetectionsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const char* env = std::getenv("YVO_TEST_VIDEO");
        const std::string video = env && *env ? env : YVO_TEST_VIDEO;
        const char* nenv = std::getenv("YVO_DET_FRAMES");
        const long nframes = nenv && *nenv ? std::max(1L, std::atol(nenv)) : 100L;
        if (!std::filesystem::exists(video)) {
            setup_error_ = "video not found: " + video;
            return;
        }
        int export_fd = -1;
        try {
            yvo::vk::VulkanContext ctx;
            yvo::vk::SandLetterboxConfig pcfg;
            pcfg.num_slots = 1;
            pcfg.export_dmabuf = ctx.dmabuf_export_supported();
            yvo::vk::SandLetterbox pre(ctx, pcfg);

            yvo::HailoRunnerConfig hcfg;
            hcfg.hef_path = YVO_TEST_HEF;
            hcfg.batch_size = 1;
            hcfg.num_slots = kNumPaths;
            auto runner = std::make_unique<yvo::HailoRunner>(hcfg);
            if (runner->input_frame_size() != pre.output_size()) {
                setup_error_ = "input size mismatch";
                return;
            }
            dmabuf_ok_ = pcfg.export_dmabuf;
            if (dmabuf_ok_) {
                export_fd = pre.export_dmabuf_fd(0);
                runner->map_dmabuf(export_fd, pre.output_size());
            }

            yvo::FfmpegDrmDecoder dec;
            if (!dec.open(video)) {
                setup_error_ = "decoder open failed: " + dec.error();
                return;
            }
            PageBuf cpu_buf(pre.output_size());
            std::mutex m;
            results_.assign(kNumPaths, {});
            for (auto& r : results_) r.resize(static_cast<std::size_t>(nframes));
            for (long n = 0; n < nframes; ++n) {
                auto f = dec.next();
                if (!f) break;
                f->pts = n;
                const yvo::LetterboxParams lb = yvo::make_letterbox(f->width, f->height);
                if (!cpu_letterbox(*f, lb, cpu_buf.p)) {
                    setup_error_ = "cpu letterbox failed at frame " + std::to_string(n);
                    break;
                }
                yvo::PreprocOutput gpu = pre.process(*f, 0);
                f.reset();
                if (std::memcmp(gpu.host_ptr, cpu_buf.p, gpu.size) == 0) ++preproc_identical_;

                auto submit = [&](Path path, yvo::PreprocOutput in) {
                    in.slot = path;
                    in.pts = n;
                    runner->submit(in, [&, path, n](yvo::InferResult&& r) {
                        std::lock_guard<std::mutex> lk(m);
                        if (r.status != HAILO_SUCCESS) ++infer_errors_;
                        results_[path][static_cast<std::size_t>(n)] = std::move(r.detections);
                    });
                };
                yvo::PreprocOutput cpu_in = gpu;
                cpu_in.host_ptr = cpu_buf.p;
                cpu_in.dmabuf_fd = -1;
                submit(kCpu, cpu_in);
                submit(kGpuPtr, gpu);
                if (dmabuf_ok_) {
                    yvo::PreprocOutput d = gpu;
                    d.dmabuf_fd = export_fd;
                    submit(kGpuDmabuf, d);
                }
                runner->drain();  // CPU 버퍼·GPU 출력을 다음 프레임에서 덮어쓰기 전에 완료 대기
                ++frames_;
            }
            pre.clear_import_cache();
            dec.close();
            runner.reset();  // dma-buf unmap
        } catch (const std::exception& e) {
            setup_error_ = std::string("exception: ") + e.what();
        }
        if (export_fd >= 0) ::close(export_fd);
        if (setup_error_.empty() && dmabuf_ok_ && frames_ > 0) run_nhcw_pass(video);
    }

    // 두 번째 패스: 셰이더가 NHCW로 직접 쓴 출력을 NHCW 러너에 dma-buf로 넣는다.
    // 첫 패스의 러너를 완전히 해제한 뒤에 호출해야 한다(Hailo 장치 1개).
    static void run_nhcw_pass(const std::string& video) {
        int export_fd = -1;
        try {
            yvo::vk::VulkanContext ctx;
            yvo::vk::SandLetterboxConfig pcfg;
            pcfg.num_slots = 1;
            pcfg.export_dmabuf = ctx.dmabuf_export_supported();
            pcfg.out_layout = yvo::TensorLayout::Nhcw;
            if (!pcfg.export_dmabuf) {
                nhcw_error_ = "dma-buf export 미지원";
                return;
            }
            yvo::vk::SandLetterbox pre(ctx, pcfg);

            yvo::HailoRunnerConfig hcfg;
            hcfg.hef_path = YVO_TEST_HEF;
            hcfg.input_layout = yvo::TensorLayout::Nhcw;
            hcfg.batch_size = 1;
            hcfg.num_slots = kNumPaths;
            auto runner = std::make_unique<yvo::HailoRunner>(hcfg);
            if (runner->input_frame_size() != pre.output_size()) {
                nhcw_error_ = "input size mismatch";
                return;
            }
            export_fd = pre.export_dmabuf_fd(0);
            runner->map_dmabuf(export_fd, pre.output_size());

            yvo::FfmpegDrmDecoder dec;
            if (!dec.open(video)) {
                nhcw_error_ = "decoder open failed: " + dec.error();
                return;
            }
            PageBuf cpu_buf(pre.output_size());
            std::vector<std::uint8_t> unpacked(pre.output_size());
            std::mutex m;
            for (long n = 0; n < frames_; ++n) {
                auto f = dec.next();
                if (!f) {
                    nhcw_error_ = "프레임 부족: " + std::to_string(n) + " / " + std::to_string(frames_);
                    break;
                }
                f->pts = n;
                const yvo::LetterboxParams lb = yvo::make_letterbox(f->width, f->height);
                if (!cpu_letterbox(*f, lb, cpu_buf.p)) {
                    nhcw_error_ = "cpu letterbox failed at frame " + std::to_string(n);
                    break;
                }
                yvo::PreprocOutput gpu = pre.process(*f, 0);
                f.reset();
                if (gpu.layout != yvo::TensorLayout::Nhcw) {
                    nhcw_error_ = "전처리 출력이 NHCW가 아니다";
                    break;
                }
                // NHCW → packed로 되돌려 CPU 기준값과 비교(셰이더의 NHCW 배치 검증)
                yvo::nhcw_to_rgb_packed(gpu.host_ptr, gpu.lb.dst_w, gpu.lb.dst_h, unpacked.data());
                if (std::memcmp(unpacked.data(), cpu_buf.p, gpu.size) == 0) ++nhcw_identical_;
                gpu.slot = kGpuDmabufNhcw;
                gpu.pts = n;
                gpu.dmabuf_fd = export_fd;
                runner->submit(gpu, [&, n](yvo::InferResult&& r) {
                    std::lock_guard<std::mutex> lk(m);
                    if (r.status != HAILO_SUCCESS) ++infer_errors_;
                    results_[kGpuDmabufNhcw][static_cast<std::size_t>(n)] = std::move(r.detections);
                });
                runner->drain();  // GPU 출력을 다음 프레임에서 덮어쓰기 전에 완료 대기
            }
            pre.clear_import_cache();
            dec.close();
            runner.reset();  // dma-buf unmap
        } catch (const std::exception& e) {
            nhcw_error_ = std::string("exception: ") + e.what();
        }
        if (export_fd >= 0) ::close(export_fd);
    }

    void SetUp() override {
        if (!setup_error_.empty()) FAIL() << setup_error_;
    }

    static void check_path(Path path) {
        ASSERT_GT(frames_, 0);
        MatchResult total;
        int printed = 0;
        for (long n = 0; n < frames_; ++n) {
            std::vector<std::string> detail;
            const auto m = match(results_[kCpu][static_cast<std::size_t>(n)],
                                 results_[path][static_cast<std::size_t>(n)], &detail);
            total.ref += m.ref;
            total.test += m.test;
            total.matched += m.matched;
            if (!detail.empty() && printed < 20) {
                std::printf("[%s] frame %ld: cpu %ld, %s %ld, matched %ld\n", kPathName[path], n, m.ref,
                            kPathName[path], m.test, m.matched);
                for (const auto& s : detail) std::printf("    %s\n", s.c_str());
                ++printed;
            }
        }
        const long denom = std::max(total.ref, total.test);
        const double ratio = denom > 0 ? static_cast<double>(total.matched) / static_cast<double>(denom) : 1.0;
        // NHCW 경로는 두 번째 패스에서 따로 센 bit-identical 수를 쓴다.
        const long identical = path == kGpuDmabufNhcw ? nhcw_identical_ : preproc_identical_;
        std::printf("[%s] frames %ld, detections cpu %ld / %s %ld, matched %ld, ratio %.4f, preproc bit-identical %ld/%ld\n",
                    kPathName[path], frames_, total.ref, kPathName[path], total.test, total.matched, ratio,
                    identical, frames_);
        EXPECT_GT(total.ref, 0) << "CPU 경로 검출이 0개";
        EXPECT_GE(ratio, kMinMatchRatio);
        EXPECT_EQ(infer_errors_, 0);
    }

    static inline std::string setup_error_;
    static inline std::string nhcw_error_;
    static inline long frames_ = 0;
    static inline long preproc_identical_ = 0;
    static inline long nhcw_identical_ = 0;
    static inline long infer_errors_ = 0;
    static inline bool dmabuf_ok_ = false;
    static inline std::vector<std::vector<std::vector<yvo::Detection>>> results_;  // [path][frame]
};

TEST_F(DetectionsTest, GpuPtrMatchesCpu) { check_path(kGpuPtr); }

TEST_F(DetectionsTest, GpuDmabufMatchesCpu) {
    if (!dmabuf_ok_) GTEST_SKIP() << "Vulkan dma-buf export 미지원";
    check_path(kGpuDmabuf);
}

TEST_F(DetectionsTest, GpuDmabufNhcwMatchesCpu) {
    if (!dmabuf_ok_) GTEST_SKIP() << "Vulkan dma-buf export 미지원";
    ASSERT_TRUE(nhcw_error_.empty()) << nhcw_error_;
    check_path(kGpuDmabufNhcw);
}

} // namespace
