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
// I420ToSand.* (no device): the --decoder sw copy (decode/sw_sand_frames.hpp) against explicit SAND128 addresses and desand_nv12.
// SandFramePool.* (dma-heap only, skipped without it): fd reuse, exhaustion, frames outliving the pool owner.
// SwDecode.* (HW HEVC decoder for the comparison, no GPU): SW frames vs HW frames, same layout fields and same visible bytes.
// DecodeCopy.* (dma-heap; the second one needs the HW HEVC decoder, no GPU): the --decode-copy copy (pipeline/decode_copy.hpp),
// whole SAND128 planes byte-identical to the source, layout fields kept, decoder frame released, pool exhaustion is an error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

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
#include "decode/sw_sand_frames.hpp"
#include "pipeline/decode_copy.hpp"
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

// ── I420ToSand: yvo::i420_to_sand128 ──────────────────────────────────────────────────────────────────────────────
// Every source byte is a function of plane and position, so a byte in the wrong place is caught. Strides have padding.
struct I420Image {
    int w, h;
    std::size_t y_stride, c_stride;
    std::vector<std::uint8_t> y, u, v;
    static std::uint8_t y_at(std::size_t x, std::size_t r) { return static_cast<std::uint8_t>(x * 3 + r * 7 + 1); }
    static std::uint8_t u_at(std::size_t x, std::size_t r) { return static_cast<std::uint8_t>(x * 5 + r * 11 + 2); }
    static std::uint8_t v_at(std::size_t x, std::size_t r) { return static_cast<std::uint8_t>(x * 13 + r * 17 + 3); }
    I420Image(int width, int height)
        : w(width), h(height), y_stride(static_cast<std::size_t>(width) + 32), c_stride(static_cast<std::size_t>(width) / 2 + 16) {
        const std::size_t rows = static_cast<std::size_t>(h), crows = rows / 2;
        y.assign(y_stride * rows, 0xAA);
        u.assign(c_stride * crows, 0xAA);
        v.assign(c_stride * crows, 0xAA);
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t x = 0; x < static_cast<std::size_t>(w); ++x) y[r * y_stride + x] = y_at(x, r);
        for (std::size_t r = 0; r < crows; ++r)
            for (std::size_t x = 0; x < static_cast<std::size_t>(w) / 2; ++x) {
                u[r * c_stride + x] = u_at(x, r);
                v[r * c_stride + x] = v_at(x, r);
            }
    }
};

constexpr std::uint8_t kUntouched = 0xEE;

struct SandDst {
    std::size_t y_col, uv_col;
    std::vector<std::uint8_t> y, uv;
    SandDst(int w, int h)
        : y_col(yvo::sand_y_col_height(h)), uv_col(yvo::sand_uv_col_height(h)), y(yvo::sand_plane_bytes(w, y_col), kUntouched),
          uv(yvo::sand_plane_bytes(w, uv_col), kUntouched) {}
    std::uint8_t y_byte(std::size_t x, std::size_t r) const { return y.at((x / 128) * 128 * y_col + r * 128 + x % 128); }
    std::uint8_t uv_byte(std::size_t x, std::size_t r) const { return uv.at((x / 128) * 128 * uv_col + r * 128 + x % 128); }
};

bool convert(const I420Image& s, SandDst& d) {
    return yvo::i420_to_sand128(s.y.data(), s.y_stride, s.u.data(), s.c_stride, s.v.data(), s.c_stride, s.w, s.h, d.y.data(), d.y.size(),
                                d.y_col, d.uv.data(), d.uv.size(), d.uv_col);
}

TEST(I420ToSand, ColumnHeightsMatchTheHwDecoder) {
    EXPECT_EQ(yvo::sand_y_col_height(1080), 1088u);    // ffmpeg_drm_decoder.cpp measured layout: pitch Y=1088 / UV=544
    EXPECT_EQ(yvo::sand_uv_col_height(1080), 544u);
    EXPECT_EQ(yvo::sand_y_col_height(1088), 1088u);
    EXPECT_EQ(yvo::sand_plane_bytes(1920, 1088), 15u * 128u * 1088u);
    EXPECT_EQ(yvo::sand_plane_bytes(1921, 1088), 16u * 128u * 1088u);
    EXPECT_EQ(yvo::sand_plane_bytes(1920, 544), 15u * 128u * 544u);
}

TEST(I420ToSand, ColumnBoundaryAndOddColumns) {
    const I420Image s(512, 8);   // 4 full columns
    SandDst d(512, 8);
    ASSERT_TRUE(convert(s, d));
    for (std::size_t r = 0; r < 8; ++r)
        for (const std::size_t x : {0u, 127u, 128u, 255u, 256u, 383u, 384u, 511u})
            EXPECT_EQ(d.y_byte(x, r), I420Image::y_at(x, r)) << "x " << x << " row " << r;
    // Column 1 starts one column stride after column 0, not one row after it.
    EXPECT_EQ(d.y[128 * d.y_col], I420Image::y_at(128, 0));
    EXPECT_EQ(d.uv[128 * d.uv_col], I420Image::u_at(64, 0));
    EXPECT_EQ(d.uv[128 * d.uv_col + 1], I420Image::v_at(64, 0));
}

TEST(I420ToSand, UvIsInterleavedUFirst) {
    const I420Image s(512, 8);
    SandDst d(512, 8);
    ASSERT_TRUE(convert(s, d));
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t cx = 0; cx < 256; ++cx) {
            ASSERT_EQ(d.uv_byte(cx * 2, r), I420Image::u_at(cx, r)) << "U cx " << cx << " row " << r;
            ASSERT_EQ(d.uv_byte(cx * 2 + 1, r), I420Image::v_at(cx, r)) << "V cx " << cx << " row " << r;
        }
}

TEST(I420ToSand, LastColumnEndsInsideTheColumn) {
    const I420Image s(300, 8);   // columns 0, 1 full; column 2 holds 44 bytes
    SandDst d(300, 8);
    ASSERT_TRUE(convert(s, d));
    for (std::size_t r = 0; r < 8; ++r) {
        EXPECT_EQ(d.y_byte(299, r), I420Image::y_at(299, r));
        EXPECT_EQ(d.y_byte(300, r), kUntouched) << "past the width, row " << r;
        EXPECT_EQ(d.y_byte(383, r), kUntouched) << "end of the last column, row " << r;
    }
    for (std::size_t r = 0; r < 4; ++r) {
        EXPECT_EQ(d.uv_byte(298, r), I420Image::u_at(149, r));
        EXPECT_EQ(d.uv_byte(299, r), I420Image::v_at(149, r));
        EXPECT_EQ(d.uv_byte(300, r), kUntouched) << "past the width, row " << r;
    }
}

TEST(I420ToSand, RowsBeyondTheHeightStayUntouched) {
    const I420Image s(256, 18);   // column height 32 / 16, so 14 Y rows and 7 UV rows stay out of the copy
    SandDst d(256, 18);
    ASSERT_EQ(d.y_col, 32u);
    ASSERT_EQ(d.uv_col, 16u);
    ASSERT_TRUE(convert(s, d));
    for (const std::size_t x : {0u, 127u, 128u, 255u}) {
        EXPECT_EQ(d.y_byte(x, 17), I420Image::y_at(x, 17));
        EXPECT_EQ(d.y_byte(x, 18), kUntouched);
        EXPECT_EQ(d.y_byte(x, 31), kUntouched);
        EXPECT_EQ(d.uv_byte(x, 9), kUntouched);
        EXPECT_EQ(d.uv_byte(x, 15), kUntouched);
    }
    EXPECT_EQ(d.uv_byte(128, 8), I420Image::u_at(64, 8));
}

TEST(I420ToSand, RoundTripThroughTheReferenceDesand) {
    for (const auto& [w, h] : std::vector<std::pair<int, int>>{{1920, 1080}, {1280, 720}, {300, 34}}) {
        const I420Image s(w, h);
        SandDst d(w, h);
        ASSERT_TRUE(convert(s, d));
        yvo::SandNv12Layout l;
        l.y_col_stride = 128 * d.y_col;
        l.uv_col_stride = 128 * d.uv_col;
        l.width = w;
        l.height = h;
        const std::size_t uw = static_cast<std::size_t>(w), uh = static_cast<std::size_t>(h);
        std::vector<std::uint8_t> ny(uw * uh), nuv(uw * uh / 2);
        ASSERT_TRUE(yvo::desand_nv12(d.y.data(), d.y.size(), d.uv.data(), d.uv.size(), l, ny.data(), uw, nuv.data(), uw));
        for (std::size_t r = 0; r < uh; ++r)
            for (std::size_t x = 0; x < uw; ++x) ASSERT_EQ(ny[r * uw + x], I420Image::y_at(x, r)) << w << "x" << h << " x " << x << " row " << r;
        for (std::size_t r = 0; r < uh / 2; ++r)
            for (std::size_t cx = 0; cx < uw / 2; ++cx) {
                ASSERT_EQ(nuv[r * uw + cx * 2], I420Image::u_at(cx, r)) << w << "x" << h;
                ASSERT_EQ(nuv[r * uw + cx * 2 + 1], I420Image::v_at(cx, r)) << w << "x" << h;
            }
    }
}

TEST(I420ToSand, RejectsWhatDoesNotFitAndWritesNothing) {
    const I420Image s(256, 16);   // height = column height: the last rows need the last bytes
    SandDst d(256, 16);
    const auto call = [&](int w, int h, std::size_t y_size, std::size_t uv_size, std::size_t y_col, std::size_t uv_col, std::size_t y_stride) {
        return yvo::i420_to_sand128(s.y.data(), y_stride, s.u.data(), s.c_stride, s.v.data(), s.c_stride, w, h, d.y.data(), y_size, y_col,
                                    d.uv.data(), uv_size, uv_col);
    };
    EXPECT_FALSE(call(255, 16, d.y.size(), d.uv.size(), d.y_col, d.uv_col, s.y_stride));      // odd width
    EXPECT_FALSE(call(256, 15, d.y.size(), d.uv.size(), d.y_col, d.uv_col, s.y_stride));      // odd height
    EXPECT_FALSE(call(0, 16, d.y.size(), d.uv.size(), d.y_col, d.uv_col, s.y_stride));
    EXPECT_FALSE(call(256, 16, d.y.size() - 1, d.uv.size(), d.y_col, d.uv_col, s.y_stride));  // Y does not fit
    EXPECT_FALSE(call(256, 16, d.y.size(), d.uv.size() - 1, d.y_col, d.uv_col, s.y_stride));  // UV does not fit
    EXPECT_FALSE(call(256, 16, d.y.size(), d.uv.size(), 4, d.uv_col, s.y_stride));            // column shorter than the plane
    EXPECT_FALSE(call(256, 16, d.y.size(), d.uv.size(), d.y_col, 2, s.y_stride));
    EXPECT_FALSE(call(256, 16, d.y.size(), d.uv.size(), d.y_col, d.uv_col, 255));             // stride shorter than the width
    for (const std::uint8_t b : d.y) ASSERT_EQ(b, kUntouched);
    for (const std::uint8_t b : d.uv) ASSERT_EQ(b, kUntouched);
}

// ── SandFramePool ─────────────────────────────────────────────────────────────────────────────────────────────────
std::shared_ptr<yvo::SandFramePool> make_pool_or_null(int count) {
    yvo::SandFramePool::Config cfg;
    cfg.width = 256;
    cfg.height = 16;
    cfg.count = count;
    try {
        return std::make_shared<yvo::SandFramePool>(cfg);
    } catch (const std::runtime_error&) {
        return nullptr;   // no dma-heap node on this machine
    }
}

TEST(SandFramePool, ReusedBufferKeepsItsFds) {
    const auto pool = make_pool_or_null(2);
    if (!pool) GTEST_SKIP() << "dma-heap not available";
    auto a = pool->acquire();
    ASSERT_TRUE(a.has_value());
    const int y_fd = a->y_fd, uv_fd = a->uv_fd;
    EXPECT_NE(y_fd, uv_fd);
    EXPECT_GE(a->y_size, 2u * 128u * 16u);
    EXPECT_GE(a->uv_size, 2u * 128u * 8u);
    EXPECT_EQ(pool->in_use(), 1);
    a.reset();   // Lease destroyed -> keepalive released
    EXPECT_EQ(pool->in_use(), 0);
    auto b = pool->acquire();
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(b->y_fd, y_fd);   // the same fd reaches DmaBufImportCache, which hits on fstat(st_dev, st_ino) + size
    EXPECT_EQ(b->uv_fd, uv_fd);
    EXPECT_TRUE(static_cast<bool>(b->keepalive));
}

TEST(SandFramePool, ExhaustionReturnsNulloptWithoutBlocking) {
    const auto pool = make_pool_or_null(2);
    if (!pool) GTEST_SKIP() << "dma-heap not available";
    auto a = pool->acquire();
    auto b = pool->acquire();
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_NE(a->y_fd, b->y_fd);
    EXPECT_FALSE(pool->acquire().has_value());
    EXPECT_EQ(pool->peak_in_use(), 2);
    const std::shared_ptr<void> held = a->keepalive;   // a copy keeps the buffer out of the free list
    a.reset();
    EXPECT_FALSE(pool->acquire().has_value());
}

TEST(SandFramePool, FrameOutlivesTheOwnerOfThePool) {
    auto pool = make_pool_or_null(1);
    if (!pool) GTEST_SKIP() << "dma-heap not available";
    auto a = pool->acquire();
    ASSERT_TRUE(a.has_value());
    const std::weak_ptr<yvo::SandFramePool> watch = pool;
    pool.reset();                 // the decoder closed
    ASSERT_FALSE(watch.expired());
    a->y[0] = 7;                  // mapping and fd are still valid
    EXPECT_GE(::lseek(a->y_fd, 0, SEEK_END), 0);
    a.reset();                    // last frame released: the pool goes with it
    EXPECT_TRUE(watch.expired());
}

// ── SwDecode: --decoder sw frames vs HW frames ────────────────────────────────────────────────────────────────────
struct VisiblePlanes {
    std::vector<std::uint8_t> y, uv;
};

bool read_visible(const yvo::DrmFrame& f, VisiblePlanes& out) {
    const std::size_t w = static_cast<std::size_t>(f.width), h = static_cast<std::size_t>(f.height);
    out.y.resize(w * h);
    out.uv.resize(w * h / 2);
    const DmaBufReadMap my(f.fd, f.size), muv(f.uv_fd, f.uv_size);
    if (!my.data() || !muv.data()) return false;
    return yvo::desand_nv12(my.data(), f.size, muv.data(), f.uv_size, yvo::make_sand_layout(f), out.y.data(), w, out.uv.data(), w);
}

std::uint64_t fnv1a(const std::vector<std::uint8_t>& v) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const std::uint8_t b : v) h = (h ^ b) * 0x100000001b3ull;
    return h;
}

TEST(SwDecode, SameLayoutAndVisibleBytesAsHwFrames) {
    const std::string video = YVO_TEST_VIDEO;
    if (!std::filesystem::exists(video)) GTEST_SKIP() << "video not found: " << video;
    yvo::FfmpegDrmDecoder hw, sw;
    yvo::FfmpegDrmDecoder::Options hopt, sopt;
    hopt.threads = 1;
    sopt.threads = 1;
    sopt.software = true;
    if (!hw.open(video, hopt)) GTEST_SKIP() << "HW HEVC decoder unavailable (open failed): " << hw.error();
    ASSERT_TRUE(sw.open(video, sopt)) << sw.error();
    constexpr int kFrames = 8;
    for (int i = 0; i < kFrames; ++i) {
        auto a = hw.next();
        if (!a) GTEST_SKIP() << "HW HEVC decoder produced no frame: " << hw.error();
        auto b = sw.next();
        ASSERT_TRUE(b.has_value()) << sw.error();
        EXPECT_EQ(b->size, a->size);
        EXPECT_EQ(b->uv_size, a->uv_size);
        EXPECT_EQ(b->modifier, a->modifier);
        EXPECT_EQ(b->drm_format, a->drm_format);
        EXPECT_EQ(b->width, a->width);
        EXPECT_EQ(b->height, a->height);
        EXPECT_EQ(b->coded_height, a->coded_height);
        EXPECT_EQ(b->y_offset, a->y_offset);
        EXPECT_EQ(b->uv_offset, a->uv_offset);
        EXPECT_EQ(b->y_pitch, a->y_pitch);
        EXPECT_EQ(b->uv_pitch, a->uv_pitch);
        EXPECT_EQ(b->sand_col_height, a->sand_col_height);
        EXPECT_EQ(b->uv_sand_col_height, a->uv_sand_col_height);
        EXPECT_EQ(b->pts, a->pts);
        EXPECT_NE(b->fd, b->uv_fd);
        VisiblePlanes pa, pb;
        ASSERT_TRUE(read_visible(*a, pa));
        ASSERT_TRUE(read_visible(*b, pb));
        std::printf("[frame %d] hw Y %016llx UV %016llx  sw Y %016llx UV %016llx\n", i, static_cast<unsigned long long>(fnv1a(pa.y)),
                    static_cast<unsigned long long>(fnv1a(pa.uv)), static_cast<unsigned long long>(fnv1a(pb.y)),
                    static_cast<unsigned long long>(fnv1a(pb.uv)));
        EXPECT_TRUE(pa.y == pb.y) << "Y differs at frame " << i;
        EXPECT_TRUE(pa.uv == pb.uv) << "UV differs at frame " << i;
    }
    const auto& st = sw.sw_stats();
    EXPECT_EQ(st.frames, kFrames);
    EXPECT_GT(st.memcpy_ms, 0.0);
    EXPECT_GT(st.sync_ms, 0.0);
}

TEST(SwDecode, PoolExhaustionIsAnErrorNotAStall) {
    const std::string video = YVO_TEST_VIDEO;
    if (!std::filesystem::exists(video)) GTEST_SKIP() << "video not found: " << video;
    yvo::FfmpegDrmDecoder sw;
    yvo::FfmpegDrmDecoder::Options opt;
    opt.threads = 1;
    opt.software = true;
    opt.extra_hw_frames = 0;   // pool of 4
    if (!sw.open(video, opt)) GTEST_SKIP() << "dma-heap not available: " << sw.error();
    std::vector<yvo::DrmFrame> held;
    for (int i = 0; i < 4; ++i) {
        auto f = sw.next();
        ASSERT_TRUE(f.has_value()) << sw.error();
        held.push_back(std::move(*f));
    }
    EXPECT_FALSE(sw.next().has_value());
    EXPECT_NE(sw.error().find("frame pool exhausted"), std::string::npos) << sw.error();
    EXPECT_FALSE(sw.eof());
}

TEST(SwDecode, FramesStayReadableAfterTheDecoderCloses) {
    const std::string video = YVO_TEST_VIDEO;
    if (!std::filesystem::exists(video)) GTEST_SKIP() << "video not found: " << video;
    yvo::FfmpegDrmDecoder sw;
    yvo::FfmpegDrmDecoder::Options opt;
    opt.threads = 1;
    opt.software = true;
    if (!sw.open(video, opt)) GTEST_SKIP() << "dma-heap not available: " << sw.error();
    auto f = sw.next();
    ASSERT_TRUE(f.has_value()) << sw.error();
    VisiblePlanes before, after;
    ASSERT_TRUE(read_visible(*f, before));
    sw.close();   // main.cpp's shutdown order can release frames after this point
    ASSERT_TRUE(read_visible(*f, after));
    EXPECT_TRUE(before.y == after.y);
    EXPECT_TRUE(before.uv == after.uv);
    EXPECT_EQ(sw.sw_stats().frames, 1);   // the stats stay readable after close()
}

// ── DecodeCopy: --decode-copy (pipeline/decode_copy.hpp) ──────────────────────────────────────────────────────────────
// Whole SAND128 planes of a frame (column height padding included), read through DmaBufReadMap.
bool read_planes(const yvo::DrmFrame& f, VisiblePlanes& out) {
    const std::size_t yb = yvo::sand_plane_bytes(f.width, f.sand_col_height), uvb = yvo::sand_plane_bytes(f.width, f.uv_sand_col_height);
    if (f.y_offset + yb > f.size || f.uv_offset + uvb > f.uv_size) return false;
    const DmaBufReadMap my(f.fd, f.size), muv(f.uv_fd, f.uv_size);
    if (!my.data() || !muv.data()) return false;
    out.y.assign(my.data() + f.y_offset, my.data() + f.y_offset + yb);
    out.uv.assign(muv.data() + f.uv_offset, muv.data() + f.uv_offset + uvb);
    return true;
}

void expect_same_layout(const yvo::DrmFrame& c, const yvo::DrmFrame& s) {
    EXPECT_EQ(c.modifier, s.modifier);
    EXPECT_EQ(c.drm_format, s.drm_format);
    EXPECT_EQ(c.width, s.width);
    EXPECT_EQ(c.height, s.height);
    EXPECT_EQ(c.coded_height, s.coded_height);
    EXPECT_EQ(c.y_pitch, s.y_pitch);
    EXPECT_EQ(c.uv_pitch, s.uv_pitch);
    EXPECT_EQ(c.sand_col_height, s.sand_col_height);
    EXPECT_EQ(c.uv_sand_col_height, s.uv_sand_col_height);
    EXPECT_EQ(c.pts, s.pts);
    EXPECT_EQ(c.y_offset, 0u);
    EXPECT_EQ(c.uv_offset, 0u);
}

// Synthetic source (no decoder): dma-heap planes with offsets and random bytes, padding rows included.
TEST(DecodeCopy, WholePlanesAreByteIdenticalAndTheSourceIsReleased) {
    constexpr int kW = 300, kH = 20;   // 3 columns, the last one partly used; column height 32 (rows 20..31 are padding)
    std::unique_ptr<yvo::DmaHeapPool> src_pool;
    std::optional<yvo::pipeline::DecodeCopy> dc;
    try {
        src_pool = std::make_unique<yvo::DmaHeapPool>();
        dc.emplace(kW, kH, 2);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "dma-heap not available";
    }
    const std::size_t y_col = yvo::sand_y_col_height(kH), uv_col = yvo::sand_uv_col_height(kH);
    const std::size_t yb = yvo::sand_plane_bytes(kW, y_col), uvb = yvo::sand_plane_bytes(kW, uv_col);
    constexpr std::size_t kYOff = 4096, kUvOff = 128;
    const int yi = src_pool->allocate(kYOff + yb), uvi = src_pool->allocate(kUvOff + uvb);
    const yvo::DmaHeapBuf& by = src_pool->at(yi);
    const yvo::DmaHeapBuf& buv = src_pool->at(uvi);
    {
        yvo::DmaCpuAccess wy(by.fd, yvo::DmaSyncDir::Write), wuv(buv.fd, yvo::DmaSyncDir::Write);
        std::uint32_t r = 12345;
        for (std::size_t i = 0; i < kYOff + yb; ++i) by.ptr[i] = static_cast<std::uint8_t>((r = r * 1664525u + 1013904223u) >> 24);
        for (std::size_t i = 0; i < kUvOff + uvb; ++i) buv.ptr[i] = static_cast<std::uint8_t>((r = r * 1664525u + 1013904223u) >> 24);
    }
    yvo::DrmFrame src;
    src.fd = by.fd;
    src.size = by.dmabuf_size;
    src.uv_fd = buv.fd;
    src.uv_size = buv.dmabuf_size;
    src.y_offset = kYOff;
    src.uv_offset = kUvOff;
    src.width = kW;
    src.height = kH;
    src.coded_height = static_cast<int>(y_col);
    src.modifier = 0x0700000000000004ull;
    src.y_pitch = y_col;
    src.uv_pitch = uv_col;
    src.sand_col_height = static_cast<std::uint32_t>(y_col);
    src.uv_sand_col_height = static_cast<std::uint32_t>(uv_col);
    src.pts = 7;
    auto token = std::make_shared<int>(0);
    const std::weak_ptr<int> watch = token;
    src.keepalive = std::move(token);   // stands in for the decoder's AVFrame

    VisiblePlanes ps, pc;
    ASSERT_TRUE(read_planes(src, ps));
    yvo::DrmFrame f = src;
    src.keepalive.reset();
    std::string why;
    ASSERT_TRUE(dc->copy(f, why)) << why;
    EXPECT_TRUE(watch.expired()) << "the source frame must be released right after the copy";
    EXPECT_NE(f.fd, by.fd);
    EXPECT_NE(f.uv_fd, buv.fd);
    expect_same_layout(f, src);
    ASSERT_TRUE(read_planes(f, pc));
    EXPECT_TRUE(ps.y == pc.y) << "Y plane differs";
    EXPECT_TRUE(ps.uv == pc.uv) << "UV plane differs";
    EXPECT_EQ(dc->total_ms().size(), 1u);
    EXPECT_EQ(dc->mappings(), 2u);

    // Same source again: the mapping cache hits. Then the pool (2) runs out: an error, not a stall.
    yvo::DrmFrame g = src;
    ASSERT_TRUE(dc->copy(g, why)) << why;
    EXPECT_EQ(dc->mappings(), 2u);
    yvo::DrmFrame h = src;
    EXPECT_FALSE(dc->copy(h, why));
    EXPECT_NE(why.find("frame pool exhausted"), std::string::npos) << why;
    EXPECT_EQ(dc->peak_in_use(), 2);

    // A frame of another size is rejected without touching the pool.
    yvo::DrmFrame bad = src;
    bad.width = 320;
    EXPECT_FALSE(dc->copy(bad, why));
    EXPECT_NE(why.find("does not fit"), std::string::npos) << why;
}

// Real HW decoder frames: the copy path (frame filter) against a plain decoder, whole planes byte-identical.
TEST(DecodeCopy, HwDecoderFramesThroughTheFilterAreByteIdentical) {
    const std::string video = YVO_TEST_VIDEO;
    if (!std::filesystem::exists(video)) GTEST_SKIP() << "video not found: " << video;
    yvo::FfmpegDrmDecoder plain, copied;
    yvo::FfmpegDrmDecoder::Options opt;
    opt.threads = 1;
    if (!plain.open(video, opt)) GTEST_SKIP() << "HW HEVC decoder unavailable (open failed): " << plain.error();
    ASSERT_TRUE(copied.open(video, opt)) << copied.error();
    std::optional<yvo::pipeline::DecodeCopy> dc;
    try {
        dc.emplace(copied.width(), copied.height(), opt.extra_hw_frames + 4);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "dma-heap not available";
    }
    copied.set_frame_filter(dc->filter());
    constexpr int kFrames = 8;
    std::vector<yvo::DrmFrame> held;   // copies outlive the decoder frames they came from
    for (int i = 0; i < kFrames; ++i) {
        auto a = plain.next();
        if (!a) GTEST_SKIP() << "HW HEVC decoder produced no frame: " << plain.error();
        auto b = copied.next();
        ASSERT_TRUE(b.has_value()) << copied.error();
        expect_same_layout(*b, *a);
        VisiblePlanes pa, pb;
        ASSERT_TRUE(read_planes(*a, pa));
        ASSERT_TRUE(read_planes(*b, pb));
        std::printf("[frame %d] hw Y %016llx UV %016llx  copy Y %016llx UV %016llx\n", i, static_cast<unsigned long long>(fnv1a(pa.y)),
                    static_cast<unsigned long long>(fnv1a(pa.uv)), static_cast<unsigned long long>(fnv1a(pb.y)),
                    static_cast<unsigned long long>(fnv1a(pb.uv)));
        EXPECT_TRUE(pa.y == pb.y) << "Y differs at frame " << i;
        EXPECT_TRUE(pa.uv == pb.uv) << "UV differs at frame " << i;
        held.push_back(std::move(*b));
    }
    EXPECT_EQ(dc->total_ms().size(), static_cast<std::size_t>(kFrames));
    EXPECT_GT(dc->mappings(), 0u);
    copied.close();   // clears the filter; held copies stay valid (they hold the pool)
    VisiblePlanes after;
    EXPECT_TRUE(read_planes(held.front(), after));
}

} // namespace
