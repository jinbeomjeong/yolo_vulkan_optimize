// GPU preprocessing (SandLetterbox) vs CPU reference (de-SAND → cpu_letterbox_nv12_bt709) comparison
//
// Converts the same DRM_PRIME frame via two paths.
//   GPU: dma-buf import → sand_letterbox.comp  (both the NHWC pipeline and the NHCW pipeline)
//   CPU: dma-buf mmap(DMA_BUF_IOCTL_SYNC) → desand_nv12(sand.hpp) → cpu_letterbox_nv12_bt709(cpu_letterbox.hpp)
// Criteria: content region max error <= 2, mean error <= 0.5, padding region exactly 114.
// The NHCW case must be byte-for-byte identical (zero error) to the CPU reference transposed with rgb_packed_to_nhcw().
// On failure, GPU / CPU / diff (x32) images are saved to build/05_tests/preproc_diff/ (NHCW is converted back to packed before saving).
// SandRefVsFfmpeg.* (no GPU): desand_nv12 (ref/sand.hpp) vs rpi-ffmpeg's own SAND→NV12 (av_hwframe_transfer_data),
// byte-identical over the visible frame. Closes the chain GPU shader ↔ ref/ ↔ rpi-ffmpeg (promoted from yvo_drm_dump --desand).
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

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

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

// ── SandRefVsFfmpeg: ref/sand.hpp de-SAND vs rpi-ffmpeg av_hwframe_transfer_data(NV12) ─────────────────────────
// Same comparison as yvo_drm_dump --desand (one frame), promoted to a ctest gate over kTestFrames. No GPU needed.
// Only the visible width x height (Y) and width x height/2 (UV) are compared. The SAND column padding rows
// (coded_height 1088 > 1080, i.e. rows 1080..1087 of Y / 540..543 of UV) are not part of the transfer output and are
// skipped, exactly as yvo_drm_dump does. 1920 is a multiple of the 128-byte column width, so there is no column padding.

struct PlaneDiff {
    std::size_t count = 0;  // differing bytes
    long first_row = -1;    // first mismatch (row-major), -1 if none
    long first_col = -1;
    int ref_val = -1;       // desand_nv12 byte at the first mismatch
    int ffmpeg_val = -1;    // av_hwframe_transfer_data byte at the first mismatch
};

struct SandRefResult {
    int width = 0, height = 0;       // DrmFrame visible size
    int sw_width = 0, sw_height = 0; // transfer output size
    PlaneDiff y, uv;
    std::string error;               // non-empty = comparison could not be made
};

PlaneDiff compare_plane(const std::uint8_t* ref, std::size_t ref_stride, const std::uint8_t* ff, std::size_t ff_stride,
                        std::size_t cols, std::size_t rows) {
    PlaneDiff d;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            const std::uint8_t a = ref[r * ref_stride + c], b = ff[r * ff_stride + c];
            if (a == b) continue;
            if (d.count == 0) {
                d.first_row = static_cast<long>(r);
                d.first_col = static_cast<long>(c);
                d.ref_val = a;
                d.ffmpeg_val = b;
            }
            ++d.count;
        }
    return d;
}

// f must still hold its AVFrame (keepalive) — the transfer reads the same DRM_PRIME frame that was de-SANDed.
SandRefResult sand_ref_vs_ffmpeg(const yvo::DrmFrame& f) {
    SandRefResult res;
    res.width = f.width;
    res.height = f.height;
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
        if (!my.data() || !muv->data()) {
            res.error = "dma-buf mmap failed";
            return res;
        }
        if (!yvo::desand_nv12(my.data(), f.size, muv->data(), two ? f.uv_size : f.size, yvo::make_sand_layout(f),
                              y.data(), w, uv.data(), w)) {
            res.error = "desand_nv12: layout out of bounds";
            return res;
        }
    }

    AVFrame* hw = yvo::FfmpegDrmDecoder::av_frame(f);
    if (!hw || !hw->hw_frames_ctx) {
        res.error = "DrmFrame has no hw AVFrame / hw_frames_ctx";
        return res;
    }
    // NV12 must be among the download formats (rpi-ffmpeg lists "yuv420p nv12" for SAND128 frames)
    AVPixelFormat* fmts = nullptr;
    bool nv12_ok = false;
    std::string listed;
    if (av_hwframe_transfer_get_formats(hw->hw_frames_ctx, AV_HWFRAME_TRANSFER_DIRECTION_FROM, &fmts, 0) >= 0 && fmts) {
        for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p) {
            if (*p == AV_PIX_FMT_NV12) nv12_ok = true;
            const char* n = av_get_pix_fmt_name(*p);
            listed += std::string(" ") + (n ? n : "?");
        }
        av_free(fmts);
    }
    if (!nv12_ok) {
        res.error = "av_hwframe_transfer_get_formats does not list nv12 (got:" + listed + ")";
        return res;
    }
    std::unique_ptr<AVFrame, void (*)(AVFrame*)> sw(av_frame_alloc(), [](AVFrame* p) { av_frame_free(&p); });
    if (!sw) {
        res.error = "av_frame_alloc failed";
        return res;
    }
    sw->format = AV_PIX_FMT_NV12;
    const int ret = av_hwframe_transfer_data(sw.get(), hw, 0);
    if (ret < 0) {
        char e[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror(ret, e, sizeof(e));
        res.error = std::string("av_hwframe_transfer_data(NV12) failed: ") + e;
        return res;
    }
    res.sw_width = sw->width;
    res.sw_height = sw->height;
    if (sw->format != AV_PIX_FMT_NV12 || sw->width < f.width || sw->height < f.height || !sw->data[0] ||
        !sw->data[1] || sw->linesize[0] < f.width || sw->linesize[1] < f.width) {
        const char* n = av_get_pix_fmt_name(static_cast<AVPixelFormat>(sw->format));
        res.error = std::string("unexpected transfer result: ") + (n ? n : "?") + " " + std::to_string(sw->width) +
                    "x" + std::to_string(sw->height) + " linesize=" + std::to_string(sw->linesize[0]) + "/" +
                    std::to_string(sw->linesize[1]);
        return res;
    }
    res.y = compare_plane(y.data(), w, sw->data[0], static_cast<std::size_t>(sw->linesize[0]), w, h);
    res.uv = compare_plane(uv.data(), w, sw->data[1], static_cast<std::size_t>(sw->linesize[1]), w, h / 2);
    return res;
}

// Decode once per process up to the last target frame; compare while each target frame is still alive.
// Skipped when the video is missing or the HW HEVC decoder cannot be opened / produces no DRM_PRIME frame.
class SandRefVsFfmpeg : public ::testing::TestWithParam<int> {
protected:
    static void SetUpTestSuite() {
        const char* env = std::getenv("YVO_TEST_VIDEO");
        video_ = env && *env ? env : YVO_TEST_VIDEO;
        if (!std::filesystem::exists(video_)) {
            skip_reason_ = "video not found: " + video_;
            return;
        }
        yvo::FfmpegDrmDecoder dec;
        if (!dec.open(video_)) {
            skip_reason_ = "HW HEVC decoder unavailable (open failed): " + dec.error();
            return;
        }
        const int last = *std::max_element(kTestFrames.begin(), kTestFrames.end());
        for (int i = 0; i <= last; ++i) {
            auto f = dec.next();
            if (!f) {
                if (i == 0)
                    skip_reason_ = "HW HEVC decoder produced no DRM_PRIME frame: " + dec.error();
                else
                    setup_error_ = "decoder stopped at frame " + std::to_string(i) + ": " + dec.error();
                return;
            }
            if (std::find(kTestFrames.begin(), kTestFrames.end(), i) == kTestFrames.end()) continue;
            results_[i] = sand_ref_vs_ffmpeg(*f);
        }
    }
    static void TearDownTestSuite() { results_.clear(); }
    void SetUp() override {
        if (!skip_reason_.empty()) GTEST_SKIP() << skip_reason_;
        ASSERT_TRUE(setup_error_.empty()) << setup_error_;
    }

    static std::string video_;
    static std::string skip_reason_;
    static std::string setup_error_;
    static std::map<int, SandRefResult> results_;
};

std::string SandRefVsFfmpeg::video_;
std::string SandRefVsFfmpeg::skip_reason_;
std::string SandRefVsFfmpeg::setup_error_;
std::map<int, SandRefResult> SandRefVsFfmpeg::results_;

std::string describe_first(const char* plane, const PlaneDiff& d) {
    return std::string(plane) + " first mismatch at row " + std::to_string(d.first_row) + " col(byte) " +
           std::to_string(d.first_col) + ": desand_nv12=" + std::to_string(d.ref_val) +
           " av_hwframe_transfer_data=" + std::to_string(d.ffmpeg_val);
}

TEST_P(SandRefVsFfmpeg, ByteIdentical) {
    const int idx = GetParam();
    const auto it = results_.find(idx);
    ASSERT_NE(it, results_.end()) << "frame " << idx << " was not decoded";
    const SandRefResult& r = it->second;
    ASSERT_TRUE(r.error.empty()) << "frame " << idx << ": " << r.error;
    const std::size_t w = static_cast<std::size_t>(r.width), h = static_cast<std::size_t>(r.height);
    std::printf("[frame %d] %dx%d (transfer %dx%d)  Y diff %zu/%zu  UV diff %zu/%zu\n", idx, r.width, r.height,
                r.sw_width, r.sw_height, r.y.count, w * h, r.uv.count, w * h / 2);
    EXPECT_EQ(r.y.count, 0u) << describe_first("Y", r.y);
    EXPECT_EQ(r.uv.count, 0u) << describe_first("UV", r.uv);
}

INSTANTIATE_TEST_SUITE_P(Frames, SandRefVsFfmpeg, ::testing::ValuesIn(kTestFrames),
                         [](const ::testing::TestParamInfo<int>& i) { return "frame" + std::to_string(i.param); });

} // namespace
