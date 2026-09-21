// GPU preprocessing (SandLetterbox) vs CPU reference (de-SAND → cpu_letterbox_nv12_bt709) comparison
//
// Converts the same DRM_PRIME frame via two paths.
//   GPU: dma-buf import → sand_letterbox.comp  (both the NHWC pipeline and the NHCW pipeline)
//   CPU: dma-buf mmap(DMA_BUF_IOCTL_SYNC) → desand_nv12(sand.hpp) → cpu_letterbox_nv12_bt709(cpu_letterbox.hpp)
// Criteria: content region max error <= 2, mean error <= 0.5, padding region exactly 114.
// The NHCW case must be byte-for-byte identical (zero error) to the CPU reference transposed with rgb_packed_to_nhcw().
// On failure, GPU / CPU / diff (x32) images are saved to build/05_tests/preproc_diff/ (NHCW is converted back to packed before saving).
#include <algorithm>
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

#include "ref/cpu_letterbox.hpp"
#include "core/dmabuf_map.hpp"
#include "ref/sand.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
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
    std::vector<std::uint8_t> gpu, cpu;    // packed RGB888 (NHWC)
    std::vector<std::uint8_t> gpu_nhcw;    // NHCW pipeline output as is
    double gpu_ms = -1, gpu_nhcw_ms = -1;
};

// Decode only once for the whole suite: collect GPU / CPU results for the target frames
class PreprocTest : public ::testing::TestWithParam<int> {
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
            yvo::vk::SandLetterboxConfig cfg;
            cfg.num_slots = 2;
            pre_ = std::make_unique<yvo::vk::SandLetterbox>(*ctx_, cfg);
            cfg.out_layout = yvo::TensorLayout::Nhcw;
            pre_nhcw_ = std::make_unique<yvo::vk::SandLetterbox>(*ctx_, cfg);

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
                FrameData d;
                d.pts = f->pts;
                d.src_w = f->width;
                d.src_h = f->height;
                const yvo::PreprocOutput out = pre_->process(*f, slot);
                d.gpu_ms = pre_->last_gpu_ms();
                d.lb = out.lb;
                d.gpu.assign(out.host_ptr, out.host_ptr + out.size);
                if (out.layout != yvo::TensorLayout::Nhwc) {
                    setup_error_ = "nhwc pipeline reported wrong layout at frame " + std::to_string(i);
                    return;
                }
                const yvo::PreprocOutput out_n = pre_nhcw_->process(*f, slot);
                d.gpu_nhcw_ms = pre_nhcw_->last_gpu_ms();
                d.gpu_nhcw.assign(out_n.host_ptr, out_n.host_ptr + out_n.size);
                if (out_n.layout != yvo::TensorLayout::Nhcw) {
                    setup_error_ = "nhcw pipeline reported wrong layout at frame " + std::to_string(i);
                    return;
                }
                if (!cpu_reference(*f, out.lb, d.cpu)) {
                    setup_error_ = "cpu reference failed at frame " + std::to_string(i);
                    return;
                }
                frames_[i] = std::move(d);
                slot ^= 1; // alternate between the two slots
            }
        } catch (const std::exception& e) {
            setup_error_ = std::string("exception: ") + e.what();
        }
    }
    static void TearDownTestSuite() {
        pre_nhcw_.reset();
        pre_.reset();
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
    static std::unique_ptr<yvo::vk::SandLetterbox> pre_;
    static std::unique_ptr<yvo::vk::SandLetterbox> pre_nhcw_;
    static std::map<int, FrameData> frames_;
};

std::string PreprocTest::video_;
std::string PreprocTest::setup_error_;
std::unique_ptr<yvo::vk::VulkanContext> PreprocTest::ctx_;
std::unique_ptr<yvo::vk::SandLetterbox> PreprocTest::pre_;
std::unique_ptr<yvo::vk::SandLetterbox> PreprocTest::pre_nhcw_;
std::map<int, FrameData> PreprocTest::frames_;

void save_images(int idx, const FrameData& d) {
    const std::string dir = "preproc_diff";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const int w = d.lb.dst_w, h = d.lb.dst_h;
    std::vector<std::uint8_t> diff(d.gpu.size());
    for (std::size_t i = 0; i < d.gpu.size(); ++i)
        diff[i] = static_cast<std::uint8_t>(std::min(255, 32 * std::abs(int(d.gpu[i]) - int(d.cpu[i]))));
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
    save(base + "_gpu", d.gpu);
    save(base + "_cpu", d.cpu);
    save(base + "_absdiff_x32", diff);
    std::fprintf(stderr, "saved diff images: %s_{gpu,cpu,absdiff_x32}\n", base.c_str());
}

TEST_P(PreprocTest, MatchesCpuReference) {
    const int idx = GetParam();
    auto it = frames_.find(idx);
    ASSERT_NE(it, frames_.end()) << "frame " << idx << " not collected";
    const FrameData& d = it->second;
    const yvo::LetterboxParams& lb = d.lb;
    ASSERT_EQ(d.gpu.size(), static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3);
    ASSERT_EQ(d.gpu.size(), d.cpu.size());
    const int new_w = static_cast<int>(d.src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(d.src_h * lb.scale + 0.5f);

    int max_err = 0;
    double sum_err = 0;
    std::size_t n_content = 0, pad_bad = 0, hist[4] = {0, 0, 0, 0};
    int worst_x = -1, worst_y = -1;
    for (int y = 0; y < lb.dst_h; ++y) {
        for (int x = 0; x < lb.dst_w; ++x) {
            const bool content = y >= lb.pad_y && y < lb.pad_y + new_h && x >= lb.pad_x && x < lb.pad_x + new_w;
            for (int c = 0; c < 3; ++c) {
                const std::size_t i = (static_cast<std::size_t>(y) * lb.dst_w + x) * 3 + c;
                if (!content) {
                    if (d.gpu[i] != lb.pad_value) ++pad_bad;
                    continue;
                }
                const int e = std::abs(int(d.gpu[i]) - int(d.cpu[i]));
                ++hist[std::min(e, 3)];
                sum_err += e;
                ++n_content;
                if (e > max_err) {
                    max_err = e;
                    worst_x = x;
                    worst_y = y;
                }
            }
        }
    }
    const double mean_err = n_content ? sum_err / static_cast<double>(n_content) : 0.0;
    std::printf("frame %4d (pts %lld): content %dx%d at (%d,%d), max_err=%d mean_err=%.5f "
                "hist[0]=%zu [1]=%zu [2]=%zu [>=3]=%zu pad_mismatch=%zu gpu=%.3f ms\n",
                idx, static_cast<long long>(d.pts), new_w, new_h, lb.pad_x, lb.pad_y, max_err, mean_err, hist[0],
                hist[1], hist[2], hist[3], pad_bad, d.gpu_ms);

    const bool ok = max_err <= 2 && mean_err <= 0.5 && pad_bad == 0;
    if (!ok) save_images(idx, d);
    EXPECT_LE(max_err, 2) << "worst pixel at (" << worst_x << "," << worst_y << ")";
    EXPECT_LE(mean_err, 0.5);
    EXPECT_EQ(pad_bad, 0u) << "padding bytes != " << int(lb.pad_value);
    // In a run with validation layers on (VK_LOADER_LAYERS_ENABLE='*validation'), the whole suite must have 0 errors
    EXPECT_EQ(ctx_->validation_errors(), 0u);
    EXPECT_NE(std::string(ctx_->device_name()).find("V3D"), std::string::npos) << "selected " << ctx_->device_name();
}

// NHCW output: must be byte-for-byte identical to the CPU reference (packed) transposed with rgb_packed_to_nhcw().
TEST_P(PreprocTest, NhcwMatchesCpuReference) {
    const int idx = GetParam();
    auto it = frames_.find(idx);
    ASSERT_NE(it, frames_.end()) << "frame " << idx << " not collected";
    const FrameData& d = it->second;
    const yvo::LetterboxParams& lb = d.lb;
    const std::size_t n = static_cast<std::size_t>(lb.dst_w) * lb.dst_h * 3;
    ASSERT_EQ(d.gpu_nhcw.size(), n);
    ASSERT_EQ(d.cpu.size(), n);

    std::vector<std::uint8_t> cpu_nhcw(n);
    yvo::rgb_packed_to_nhcw(d.cpu.data(), lb.dst_w, lb.dst_h, cpu_nhcw.data());

    int max_err = 0;
    std::size_t mismatch = 0, worst = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int e = std::abs(int(d.gpu_nhcw[i]) - int(cpu_nhcw[i]));
        if (e) {
            ++mismatch;
            if (e > max_err) {
                max_err = e;
                worst = i;
            }
        }
    }
    // Only the layout differs; pixel values must equal the NHWC path (transposing back gives a byte-exact match).
    std::vector<std::uint8_t> back(n);
    yvo::nhcw_to_rgb_packed(d.gpu_nhcw.data(), lb.dst_w, lb.dst_h, back.data());
    const bool same_as_nhwc = back == d.gpu;
    std::printf("frame %4d (pts %lld) NHCW: bytes=%zu mismatch=%zu max_err=%d (worst idx %zu: row %d ch %d col %d) "
                "same_as_nhwc=%d gpu=%.3f ms\n",
                idx, static_cast<long long>(d.pts), n, mismatch, max_err, worst,
                static_cast<int>(worst) / (3 * lb.dst_w), (static_cast<int>(worst) / lb.dst_w) % 3,
                static_cast<int>(worst) % lb.dst_w, same_as_nhwc ? 1 : 0, d.gpu_nhcw_ms);

    if (mismatch != 0 || !same_as_nhwc) {
        FrameData packed = d;   // for visual inspection, convert back to packed and reuse the existing utility as is
        packed.gpu = back;
        save_images(idx, packed);
    }
    EXPECT_EQ(mismatch, 0u) << "max_err=" << max_err;
    EXPECT_TRUE(same_as_nhwc) << "NHCW output differs from transposed NHWC output";
    EXPECT_EQ(ctx_->validation_errors(), 0u);
}

INSTANTIATE_TEST_SUITE_P(Frames, PreprocTest, ::testing::ValuesIn(kTestFrames),
                         [](const ::testing::TestParamInfo<int>& i) { return "frame" + std::to_string(i.param); });

} // namespace
