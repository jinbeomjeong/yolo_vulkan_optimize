// ISP 전처리(IspResizer) vs CPU 기준(de-SAND → cpu_letterbox_nv12_bt709) 화질 비교
//
// GPU 경로(test_preproc.cpp)와 달리 **비트 일치를 요구하지 않는다.** 리샘플 커널이 다르기 때문이다:
//   - 셰이더: bilinear 4-tap
//   - ISP:   libpisp smart selection → 3배 축소 구간은 michel-netravali(Mitchell-Netravali)
// 색 변환(BT.709 limited YUV→RGB)은 같은 수식이지만 ISP는 1024 스케일 고정소수점이라 반올림이 다르다.
//
// 합격선(요청받은 게이트):
//   패딩 영역      max|err| == 0                      (하드 게이트: letterbox가 ISP 쓰기 밖인지)
//   콘텐츠 영역    |채널별 평균 부호 오차| <= 1.0      (CSC 오류·range 혼동·R↔B 스왑 탐지기)
//   콘텐츠 영역    MAE <= 3.0
//   PSNR, max|err|, |err|>16 비율은 보고만 한다(불합격 근거로 쓰지 않는다).
//
// 교차 검산: (ISP vs GPU셰이더)도 같이 계산한다. GPU는 CPU와 비트 일치이므로 두 지표가 같아야 한다.
// 실패하면 build/05_tests/isp_preproc_diff/ 에 ISP·CPU·차이(x8) 이미지를 저장한다.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifdef YVO_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "common/cpu_letterbox.hpp"
#include "common/dmabuf_map.hpp"
#include "common/sand.hpp"
#include "isp/isp_resizer.hpp"
#include "isp/nhcw_repack.hpp"
#include "pipeline/ffmpeg_drm_decoder.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

namespace {

const std::vector<int> kTestFrames = {0, 100, 500, 900, 1815};

using yvo::DmaBufReadMap;

bool cpu_reference(const yvo::DrmFrame& f, const yvo::LetterboxParams& lb, std::vector<std::uint8_t>& out) {
    const std::size_t w = static_cast<std::size_t>(f.width), h = static_cast<std::size_t>(f.height);
    std::vector<std::uint8_t> y(w * h), uv(w * h / 2);
    {
        DmaBufReadMap my(f.fd, f.size);
        std::unique_ptr<DmaBufReadMap> muv_own;
        const DmaBufReadMap* muv = &my;
        const bool two = f.uv_fd >= 0 && f.uv_fd != f.fd;
        if (two) {
            muv_own = std::make_unique<DmaBufReadMap>(f.uv_fd, f.uv_size);
            muv = muv_own.get();
        }
        if (!my.data() || !muv->data()) return false;
        if (!yvo::desand_nv12(my.data(), f.size, muv->data(), two ? f.uv_size : f.size, yvo::make_sand_layout(f),
                              y.data(), w, uv.data(), w))
            return false;
    }
    out.resize(static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3);
    yvo::cpu_letterbox_nv12_bt709(y.data(), w, uv.data(), w, f.width, f.height, lb, out.data());
    return true;
}

struct FrameData {
    std::int64_t pts = -1;
    int src_w = 0, src_h = 0;
    yvo::LetterboxParams lb;
    std::vector<std::uint8_t> isp, cpu, gpu; // packed RGB888 (NHWC)
    double isp_ms = -1;
};

// 콘텐츠 영역 비교 지표
struct Metrics {
    double mean_signed[3] = {0, 0, 0}; // 채널별 mean(a - b)
    double mae = 0;
    double psnr = 0;
    int max_abs = 0;
    double frac_gt16 = 0;
    std::size_t n = 0;
    std::size_t pad_bad = 0;
    int pad_max_abs = 0;
};

Metrics compare(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b,
                const yvo::LetterboxParams& lb, int new_w, int new_h) {
    Metrics m;
    double sum_signed[3] = {0, 0, 0};
    double sum_abs = 0, sum_sq = 0;
    std::size_t gt16 = 0, per_ch = 0;
    for (int y = 0; y < lb.dst_h; ++y) {
        const bool row_in = y >= lb.pad_y && y < lb.pad_y + new_h;
        for (int x = 0; x < lb.dst_w; ++x) {
            const bool content = row_in && x >= lb.pad_x && x < lb.pad_x + new_w;
            for (int c = 0; c < 3; ++c) {
                const std::size_t i = (static_cast<std::size_t>(y) * lb.dst_w + x) * 3 + c;
                if (!content) {
                    const int e = std::abs(int(a[i]) - int(lb.pad_value));
                    if (e) {
                        ++m.pad_bad;
                        m.pad_max_abs = std::max(m.pad_max_abs, e);
                    }
                    continue;
                }
                const int d = int(a[i]) - int(b[i]);
                sum_signed[c] += d;
                sum_abs += std::abs(d);
                sum_sq += double(d) * d;
                if (std::abs(d) > 16) ++gt16;
                m.max_abs = std::max(m.max_abs, std::abs(d));
                ++m.n;
            }
        }
    }
    per_ch = m.n / 3;
    if (per_ch) for (int c = 0; c < 3; ++c) m.mean_signed[c] = sum_signed[c] / static_cast<double>(per_ch);
    if (m.n) {
        m.mae = sum_abs / static_cast<double>(m.n);
        const double mse = sum_sq / static_cast<double>(m.n);
        m.psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
        m.frac_gt16 = static_cast<double>(gt16) / static_cast<double>(m.n);
    }
    return m;
}

// 전체 스위트에서 한 번만 디코드: 대상 프레임의 ISP·GPU·CPU 결과를 모은다
class IspPreprocTest : public ::testing::TestWithParam<int> {
protected:
    static void SetUpTestSuite() {
        const char* env = std::getenv("YVO_TEST_VIDEO");
        video_ = env && *env ? env : YVO_TEST_VIDEO;
        if (!std::filesystem::exists(video_)) {
            setup_error_ = "video not found: " + video_;
            return;
        }
        try {
            ctx_ = std::make_unique<yvo::vk::VulkanContext>();
            yvo::vk::SandLetterboxConfig gcfg;
            gcfg.num_slots = 2;
            gpu_ = std::make_unique<yvo::vk::SandLetterbox>(*ctx_, gcfg);

            yvo::FfmpegDrmDecoder dec;
            if (!dec.open(video_)) {
                setup_error_ = "decoder open failed: " + dec.error();
                return;
            }
            const int last = *std::max_element(kTestFrames.begin(), kTestFrames.end());
            int slot = 0;
            for (int i = 0; i <= last; ++i) {
                auto f = dec.next();
                if (!f) {
                    setup_error_ = "decoder stopped at frame " + std::to_string(i) + ": " + dec.error();
                    return;
                }
                if (std::find(kTestFrames.begin(), kTestFrames.end(), i) == kTestFrames.end()) continue;
                if (!isp_) { // 첫 대상 프레임의 기하로 ISP 설정을 확정한다
                    yvo::isp::IspResizerConfig icfg;
                    icfg.num_slots = 2;
                    icfg.buffer_count = 12;
                    icfg.src_w = f->width;
                    icfg.src_h = f->height;
                    icfg.coded_height = f->coded_height;
                    icfg.sand_col_height = static_cast<int>(f->sand_col_height);
                    icfg.uv_sand_col_height = static_cast<int>(f->uv_sand_col_height);
                    isp_ = std::make_unique<yvo::isp::IspResizer>(icfg);
                }
                FrameData d;
                d.pts = f->pts;
                d.src_w = f->width;
                d.src_h = f->height;
                const yvo::PreprocOutput out = isp_->process(*f, slot);
                d.isp_ms = isp_->last_isp_ms();
                d.lb = out.lb;
                if (out.layout != yvo::TensorLayout::Nhwc) {
                    setup_error_ = "isp reported wrong layout at frame " + std::to_string(i);
                    return;
                }
                {
                    const yvo::isp::IspResizer::OutputRead lock(*isp_, slot);
                    d.isp.assign(lock.data(), lock.data() + out.size);
                }
                const yvo::PreprocOutput g = gpu_->process(*f, slot);
                d.gpu.assign(g.host_ptr, g.host_ptr + g.size);
                if (!cpu_reference(*f, out.lb, d.cpu)) {
                    setup_error_ = "cpu reference failed at frame " + std::to_string(i);
                    return;
                }
                frames_[i] = std::move(d);
                slot ^= 1;
            }
            isp_->clear_input_cache();
            gpu_->clear_import_cache();
        } catch (const std::exception& e) {
            setup_error_ = std::string("exception: ") + e.what();
        }
    }
    static void TearDownTestSuite() {
        isp_.reset();
        gpu_.reset();
        ctx_.reset();
        frames_.clear();
    }
    void SetUp() override {
        if (!std::filesystem::exists(video_)) GTEST_SKIP() << setup_error_;
        ASSERT_TRUE(setup_error_.empty()) << setup_error_;
    }

    static std::string video_;
    static std::string setup_error_;
    static std::unique_ptr<yvo::vk::VulkanContext> ctx_;
    static std::unique_ptr<yvo::vk::SandLetterbox> gpu_;
    static std::unique_ptr<yvo::isp::IspResizer> isp_;
    static std::map<int, FrameData> frames_;
};

std::string IspPreprocTest::video_;
std::string IspPreprocTest::setup_error_;
std::unique_ptr<yvo::vk::VulkanContext> IspPreprocTest::ctx_;
std::unique_ptr<yvo::vk::SandLetterbox> IspPreprocTest::gpu_;
std::unique_ptr<yvo::isp::IspResizer> IspPreprocTest::isp_;
std::map<int, FrameData> IspPreprocTest::frames_;

void save_images(int idx, const FrameData& d) {
    const std::string dir = "isp_preproc_diff";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const int w = d.lb.dst_w, h = d.lb.dst_h;
    std::vector<std::uint8_t> diff(d.isp.size());
    for (std::size_t i = 0; i < d.isp.size(); ++i)
        diff[i] = static_cast<std::uint8_t>(std::min(255, 8 * std::abs(int(d.isp[i]) - int(d.cpu[i]))));
    const std::string base = dir + "/frame" + std::to_string(idx);
#ifdef YVO_HAVE_OPENCV
    auto save = [&](const std::string& p, const std::vector<std::uint8_t>& rgb) {
        cv::Mat m(h, w, CV_8UC3, const_cast<std::uint8_t*>(rgb.data())), bgr;
        cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(p + ".png", bgr);
    };
#else
    auto save = [&](const std::string& p, const std::vector<std::uint8_t>& rgb) {
        std::ofstream o(p + ".ppm", std::ios::binary);
        o << "P6\n" << w << " " << h << "\n255\n";
        o.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    };
#endif
    save(base + "_isp", d.isp);
    save(base + "_cpu", d.cpu);
    save(base + "_absdiff_x8", diff);
    std::fprintf(stderr, "saved diff images: %s_{isp,cpu,absdiff_x8}\n", base.c_str());
}

TEST_P(IspPreprocTest, MatchesCpuReference) {
    const int idx = GetParam();
    auto it = frames_.find(idx);
    ASSERT_NE(it, frames_.end()) << "frame " << idx << " not collected";
    const FrameData& d = it->second;
    const yvo::LetterboxParams& lb = d.lb;
    const std::size_t n = static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3;
    ASSERT_EQ(d.isp.size(), n);
    ASSERT_EQ(d.cpu.size(), n);
    ASSERT_EQ(d.gpu.size(), n);
    const int new_w = static_cast<int>(d.src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(d.src_h * lb.scale + 0.5f);

    // GPU는 CPU와 비트 일치여야 한다(test_preproc의 게이트). 여기서는 교차 검산용 전제 확인.
    EXPECT_EQ(d.gpu, d.cpu) << "GPU shader is expected to be bit-identical to the CPU reference";

    const Metrics m = compare(d.isp, d.cpu, lb, new_w, new_h);
    const Metrics g = compare(d.isp, d.gpu, lb, new_w, new_h);

    std::printf("frame %4d (pts %lld): content %dx%d at (%d,%d), isp=%.3f ms\n"
                "  vs CPU : mean_signed R=%+.4f G=%+.4f B=%+.4f | MAE=%.4f | PSNR=%.2f dB | "
                "max|err|=%d | frac(|err|>16)=%.5f%% | pad_bad=%zu (max %d)\n"
                "  vs GPU : mean_signed R=%+.4f G=%+.4f B=%+.4f | MAE=%.4f | PSNR=%.2f dB | max|err|=%d\n",
                idx, static_cast<long long>(d.pts), new_w, new_h, lb.pad_x, lb.pad_y, d.isp_ms,
                m.mean_signed[0], m.mean_signed[1], m.mean_signed[2], m.mae, m.psnr, m.max_abs,
                100.0 * m.frac_gt16, m.pad_bad, m.pad_max_abs,
                g.mean_signed[0], g.mean_signed[1], g.mean_signed[2], g.mae, g.psnr, g.max_abs);

    const bool ok = m.pad_max_abs == 0 && m.mae <= 3.0 && std::abs(m.mean_signed[0]) <= 1.0 &&
                    std::abs(m.mean_signed[1]) <= 1.0 && std::abs(m.mean_signed[2]) <= 1.0;
    if (!ok) save_images(idx, d);

    // 하드 게이트: letterbox 패딩은 ISP가 물리적으로 건드리지 않아야 한다.
    EXPECT_EQ(m.pad_max_abs, 0) << "padding bytes != " << int(lb.pad_value) << " (" << m.pad_bad << " bytes)";
    // CSC 오류·range 혼동·R↔B 스왑 탐지기
    EXPECT_LE(std::abs(m.mean_signed[0]), 1.0) << "R channel mean signed error";
    EXPECT_LE(std::abs(m.mean_signed[1]), 1.0) << "G channel mean signed error";
    EXPECT_LE(std::abs(m.mean_signed[2]), 1.0) << "B channel mean signed error";
    EXPECT_LE(m.mae, 3.0);
    // GPU가 CPU와 비트 일치이므로 두 비교 결과는 같아야 한다.
    EXPECT_DOUBLE_EQ(m.mae, g.mae);
    EXPECT_EQ(m.max_abs, g.max_abs);
}

// NEON 전치는 스칼라 전치와 바이트 단위로 같아야 한다.
TEST_P(IspPreprocTest, NeonRepackMatchesScalar) {
    const int idx = GetParam();
    auto it = frames_.find(idx);
    ASSERT_NE(it, frames_.end()) << "frame " << idx << " not collected";
    const FrameData& d = it->second;
    const int w = d.lb.dst_w, h = d.lb.dst_h;
    std::vector<std::uint8_t> a(d.isp.size()), b(d.isp.size());
    yvo::isp::rgb_packed_to_nhcw_scalar(d.isp.data(), w, h, a.data());
    yvo::isp::rgb_packed_to_nhcw_neon(d.isp.data(), w, h, b.data());
    EXPECT_EQ(a, b) << "NEON repack differs from the scalar repack";
    // 되돌리면 원본과 같아야 한다(전치가 정보를 잃지 않음).
    std::vector<std::uint8_t> back(d.isp.size());
    yvo::nhcw_to_rgb_packed(b.data(), w, h, back.data());
    EXPECT_EQ(back, d.isp);
}

INSTANTIATE_TEST_SUITE_P(Frames, IspPreprocTest, ::testing::ValuesIn(kTestFrames),
                         [](const ::testing::TestParamInfo<int>& i) { return "frame" + std::to_string(i.param); });

} // namespace
