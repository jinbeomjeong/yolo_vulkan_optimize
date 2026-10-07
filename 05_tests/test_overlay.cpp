// GPU overlay (overlay_boxes.comp) vs CPU reference (overlay_reference.hpp) — **bit-exact** verdict
//
// Follows the table in plan §7. NPU and display are **not needed** (offscreen R8G8B8A8_UNORM image), so
// it runs as is under ctest.
//
//   OverlayGeometry     1920x1080 (same resolution as the real path), text off, line width 1/2/3.
//                       Edge contact, zero area, partly/fully off-screen, class 0/79, 80 detections. **max error 0**
//   OverlayText         Labels + HUD. text_scale 1 and 2. **max error 0**
//                       (OpenCV is not used as the oracle — the CPU reference using the same kGlyph8x16 table is the oracle)
//   OverlayDeterminism  Non-overlapping input repeated 10 times is **fully identical**
//   OverlayPlane        dma-buf ring for the HVS overlay plane (overlay_plane.hpp, buffer variant of overlay_boxes).
//                       Reads the export fd via mmap (DMA_BUF_IOCTL_SYNC) and **bit-compares** with the CPU reference (outside boxes = (0,0,0,0)),
//                       zero ghosting on ring entry reuse (both Full and Erase clear modes), pitch/size/ring state checks, clear GPU time.
//   DrmState            debugfs state parser (display/drm_state.hpp) for verifying plane scanout (wayland-planes). No GPU needed.
//                       Pins measured state excerpts (hvs_wl_probe fullhack f225, labwc desktop fbcon/XR24) as string literals.
//   KmsScanout          --display-backend kms transport layer (display/kms_scanout.hpp). Skipped if card1 is absent.
//                       **Never takes master and never changes the screen**: runs only when the desktop (labwc) is master.
//                       If nobody is master (= open would automatically receive master), skip without attempting a modeset.
//                       Device/connector/CRTC/plane discovery (NV12 SAND128 / AB24 LINEAR), box and video AddFB2, whether a commit without master
//                       ends with a clear error instead of hanging, and whether the KmsMasterError message contains the master owner.
//   OverlayVsOpenCV     (not a gate) loose geometric comparison with draw_and_save() at 640x640.
//                       FONT_HERSHEY_SIMPLEX is a vector font with different metrics, so bit-exactness is impossible in principle.
//
// Intersection pixels of overlapping boxes are **order-dependent** (workgroup execution order is undefined. plan §7, §9).
// Therefore all test inputs are built **not to overlap each other**, and the HUD region (4,4)~(4+8*s*len, 4+16*s) is left empty too
// (on the GPU, workgroup 0 draws the HUD together with detection 0, so boxes overlapping the HUD are order-dependent as well).
//
// On failure, GPU / CPU / diff images are saved to build/05_tests/overlay_diff/ (test_preproc's convention).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef YVO_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "ref/overlay_reference.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "display/drm_state.hpp"
#include "display/kms_scanout.hpp"
#include "display/plane_display.hpp"
#include "display/overlay_plane.hpp"
#include "display/overlay_renderer.hpp"
#include "testkit/offscreen_target.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace {

using yvo::Detection;
using yvo::display::DetGpu;
using yvo::display::make_overlay_affine;
using yvo::display::OverlayAffine;
using yvo::display::OverlayConfig;
using yvo::display::OverlayRenderer;
using yvo::display::overlay_cpu_render;

constexpr std::array<std::uint8_t, 4> kBg{32, 32, 32, 255};

// Use a single VulkanContext for the whole process (device creation is the most expensive part).
yvo::vk::VulkanContext& ctx() {
    static std::unique_ptr<yvo::vk::VulkanContext> c = [] {
        yvo::vk::VulkanContextOptions o;
        const char* v = std::getenv("YVO_VK_VALIDATION");
        o.enable_validation = v && *v == '1';
        return std::make_unique<yvo::vk::VulkanContext>(o);
    }();
    return *c;
}

// ── Offscreen render harness ────────────────────────────────────────────────────────────────────
// STORAGE|TRANSFER_SRC|TRANSFER_DST image + readback buffer. Fills the background exactly with vkCmdClearColorImage
// and then records the overlay dispatch (same format and same GENERAL layout as the swapchain path).
// The image, readback buffer and barriers use OffscreenTarget from testkit/offscreen_target.hpp
// (before 2026-09-21 this harness had its own copy of the same code as display_probe's OffscreenTarget).
class OverlayHarness {
public:
    OverlayHarness(int w, int h)
        : w_(w), h_(h),
          t_(ctx(), static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), true,
             VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
        VkDevice dev = ctx().device();
        VkCommandPoolCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = ctx().queue_family();
        YVO_VK_CHECK(vkCreateCommandPool(dev, &cp, nullptr, &pool_));
        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        YVO_VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd_));
    }
    ~OverlayHarness() {
        VkDevice dev = ctx().device();
        vkDeviceWaitIdle(dev);
        if (pool_) vkDestroyCommandPool(dev, pool_, nullptr);
    }
    OverlayHarness(const OverlayHarness&) = delete;
    OverlayHarness& operator=(const OverlayHarness&) = delete;

    // Fill the background with kBg, draw the overlay, and return RGBA.
    std::vector<std::uint8_t> render(OverlayRenderer& ov, int slot) {
        using yvo::display::barrier_image;
        VkDevice dev = ctx().device();
        YVO_VK_CHECK(vkResetFences(dev, 1, &t_.fence));
        YVO_VK_CHECK(vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        YVO_VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));

        barrier_image(cmd_, t_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue cc{};
        for (int i = 0; i < 4; ++i) cc.float32[i] = static_cast<float>(kBg[static_cast<std::size_t>(i)]) / 255.0f;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, t_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1, &range);
        barrier_image(cmd_, t_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        // No preceding compute write (the clear barrier is already there), so barrier_prior_compute_write = false
        ov.record(slot, cmd_, t_.view, {t_.w, t_.h}, affine_, false);
        t_.record_readback(cmd_);
        YVO_VK_CHECK(vkEndCommandBuffer(cmd_));

        VkCommandBufferSubmitInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cbi.commandBuffer = cmd_;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        YVO_VK_CHECK(vkQueueSubmit2(ctx().queue(), 1, &si, t_.fence));
        YVO_VK_CHECK(vkWaitForFences(dev, 1, &t_.fence, VK_TRUE, 5'000'000'000ull));
        ov.collect(slot);
        return std::vector<std::uint8_t>(t_.host, t_.host + static_cast<std::size_t>(w_) * h_ * 4);
    }

    void set_affine(const OverlayAffine& a) { affine_ = a; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    int w_ = 0, h_ = 0;
    yvo::display::OffscreenTarget t_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    OverlayAffine affine_{};
};

std::vector<std::uint8_t> background(int w, int h) {
    std::vector<std::uint8_t> v(static_cast<std::size_t>(w) * h * 4);
    for (std::size_t i = 0; i < v.size(); i += 4)
        for (int k = 0; k < 4; ++k) v[i + static_cast<std::size_t>(k)] = kBg[static_cast<std::size_t>(k)];
    return v;
}

struct Diff {
    int max_err = 0;
    long long diff_px = 0;
    long long changed_gpu = 0;   // number of GPU pixels that differ from the background
    long long changed_cpu = 0;
    int first_x = -1, first_y = -1;
    int gpu[4]{}, cpu[4]{};
};

Diff compare(const std::vector<std::uint8_t>& g, const std::vector<std::uint8_t>& c, int w, int h,
             const std::array<std::uint8_t, 4>& bg = kBg) {  // bg: background (OverlayPlane uses transparent (0,0,0,0))
    Diff d;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                                   static_cast<std::size_t>(x)) * 4;
            bool cg = false, cc = false, differ = false;
            int e = 0;
            for (int k = 0; k < 4; ++k) {
                if (g[o + static_cast<std::size_t>(k)] != bg[static_cast<std::size_t>(k)]) cg = true;
                if (c[o + static_cast<std::size_t>(k)] != bg[static_cast<std::size_t>(k)]) cc = true;
                const int dd = std::abs(static_cast<int>(g[o + static_cast<std::size_t>(k)]) -
                                        static_cast<int>(c[o + static_cast<std::size_t>(k)]));
                if (dd) differ = true;
                e = std::max(e, dd);
            }
            if (cg) ++d.changed_gpu;
            if (cc) ++d.changed_cpu;
            if (differ) {
                ++d.diff_px;
                if (d.first_x < 0) {
                    d.first_x = x;
                    d.first_y = y;
                    for (int k = 0; k < 4; ++k) {
                        d.gpu[k] = g[o + static_cast<std::size_t>(k)];
                        d.cpu[k] = c[o + static_cast<std::size_t>(k)];
                    }
                }
                d.max_err = std::max(d.max_err, e);
            }
        }
    }
    return d;
}

// Save on failure (test_preproc's preproc_diff/ convention). Falls back to PPM without OpenCV.
void save_diff(const std::string& tag, const std::vector<std::uint8_t>& g, const std::vector<std::uint8_t>& c,
               int w, int h) {
    const std::filesystem::path dir = "overlay_diff";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    auto to_rgb = [&](const std::vector<std::uint8_t>& src) {
        std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * h * 3);
        for (std::size_t i = 0, j = 0; j < out.size(); i += 4, j += 3) {
            out[j] = src[i];
            out[j + 1] = src[i + 1];
            out[j + 2] = src[i + 2];
        }
        return out;
    };
    const std::vector<std::uint8_t> gr = to_rgb(g), cr = to_rgb(c);
    std::vector<std::uint8_t> dr(gr.size());
    for (std::size_t i = 0; i < dr.size(); ++i)
        dr[i] = static_cast<std::uint8_t>(std::min(255, std::abs(static_cast<int>(gr[i]) - static_cast<int>(cr[i])) * 32));
#ifdef YVO_HAVE_OPENCV
    auto write = [&](const char* name, const std::vector<std::uint8_t>& rgb) {
        cv::Mat m(h, w, CV_8UC3, const_cast<std::uint8_t*>(rgb.data()));
        cv::Mat bgr;
        cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);
        cv::imwrite((dir / (tag + "_" + name + ".png")).string(), bgr);
    };
    write("gpu", gr);
    write("cpu", cr);
    write("diff", dr);
#else
    auto write = [&](const char* name, const std::vector<std::uint8_t>& rgb) {
        std::ofstream f(dir / (tag + "_" + name + ".ppm"), std::ios::binary);
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    };
    write("gpu", gr);
    write("cpu", cr);
    write("diff", dr);
#endif
    std::fprintf(stderr, "[diff] %s → %s/\n", tag.c_str(), std::filesystem::absolute(dir).string().c_str());
}

Detection det(int cls, float score, float x1, float y1, float x2, float y2) {
    Detection d;
    d.class_id = cls;
    d.score = score;
    d.x1 = x1;
    d.y1 = y1;
    d.x2 = x2;
    d.y2 = y2;
    return d;
}

// Collection of edge cases. **They do not overlap each other** (overlap makes intersection pixels order-dependent).
std::vector<Detection> boundary_cases() {
    return {
        det(0, 0.90f, 0, 0, 100, 100),                 // touches the top-left screen edge
        det(79, 0.12f, 1819, 979, 1919, 1079),         // touches the bottom-right edge (includes the last pixel)
        det(1, 0.50f, 0, 300, 40, 400),                // left edge
        det(2, 1.00f, 1879, 300, 1919, 400),           // right edge, score 1.00
        det(3, 0.00f, 300, 0, 400, 30),                // top edge, score 0.00 (label does not fit above)
        det(4, 0.33f, 300, 1049, 400, 1079),           // bottom edge
        det(5, 0.44f, 500, 200, 500, 200),             // zero area (1 px dot)
        det(6, 0.55f, 600, 200, 600, 300),             // zero width (vertical 1 px line)
        det(7, 0.66f, 700, 200, 800, 200),             // zero height (horizontal 1 px line)
        det(8, 0.77f, -50, 500, 50, 600),              // partly off-screen (left)
        det(9, 0.88f, 1890, 700, 2000, 800),           // partly off-screen (right)
        det(10, 0.99f, 900, -40, 1000, 60),            // partly off-screen (top)
        det(11, 0.20f, 900, 1040, 1000, 1140),         // partly off-screen (bottom)
        det(12, 0.30f, -500, -500, -400, -400),        // fully off-screen
        det(13, 0.40f, 2000, 1200, 2100, 1300),        // fully off-screen
        det(14, 0.51f, 1100.5f, 200.5f, 1200.49f, 300.51f), // R1 rounding boundary (.5 rounds up)
        det(15, 0.49f, 1300.4f, 200.6f, 1400.6f, 300.4f),   // R1 rounding
        det(16, 0.61f, 1500, 200, 1400, 300),          // x1 > x2 (flipped input → min/max normalization)
        det(17, 0.71f, 600, 500, 700, 450),            // y1 > y2
    };
}

// Non-overlapping grid of n boxes (for testing the 80-detection cap). **Borders do not overlap each other.**
// Labels (text) can overlap horizontally with labels in the neighboring column, so this input is used only with draw_text = false.
std::vector<Detection> grid_cases(int n, int w, int h) {
    std::vector<Detection> v;
    const int cols = 10;
    const int cw = w / cols, ch = h / ((n + cols - 1) / cols);
    for (int i = 0; i < n; ++i) {
        const int cx = (i % cols) * cw, cy = (i / cols) * ch;
        v.push_back(det(i % 80, 0.1f + 0.01f * static_cast<float>(i % 80), static_cast<float>(cx + 4),
                        static_cast<float>(cy + 20), static_cast<float>(cx + cw - 6),
                        static_cast<float>(cy + ch - 6)));
    }
    return v;
}

// Input where **nothing overlaps** even with text on (also avoids the HUD region y < 36).
// Labels are drawn 16*s (up to 32) px above the box, so y1 >= 200 leaves margin.
std::vector<Detection> separated_text_cases() {
    return {
        det(0, 0.87f, 100, 200, 400, 500),      // person 0.87
        det(9, 0.05f, 600, 200, 900, 500),      // traffic light 0.05 (longest name)
        det(79, 1.00f, 1100, 200, 1400, 500),   // toothbrush 1.00
        det(2, 0.50f, 100, 700, 400, 1000),     // car 0.50
        det(5, 0.995f, 600, 700, 900, 1000),    // bus 1.00 (rounding)
        det(3, 0.004f, 1100, 700, 1400, 1000),  // motorcycle 0.00 (rounding)
        det(7, 0.62f, 1500, 20, 1700, 120),     // not enough space above → label goes inside the box
        det(1, 0.42f, 1750, 700, 1919, 800),    // label overflows to the right → screen clipping
    };
}

struct RunResult {
    Diff diff;
    std::vector<std::uint8_t> gpu, cpu;
};

RunResult run_case(OverlayHarness& hz, const OverlayConfig& cfg, const std::vector<Detection>& dets,
                   const std::string& hud) {
    OverlayRenderer ov(ctx(), cfg);
    const OverlayAffine a = make_overlay_affine(hz.width(), hz.height(), hz.width(), hz.height());
    hz.set_affine(a);
    ov.set_frame(0, dets, hud);
    RunResult r;
    r.gpu = hz.render(ov, 0);
    r.cpu = background(hz.width(), hz.height());
    overlay_cpu_render(r.cpu.data(), hz.width(), hz.height(), dets, hud, a, ov.style(),
                       static_cast<std::size_t>(cfg.max_dets));
    r.diff = compare(r.gpu, r.cpu, hz.width(), hz.height());
    return r;
}

void expect_exact(const std::string& tag, const RunResult& r, int w, int h) {
    if (r.diff.max_err != 0 || r.diff.diff_px != 0) {
        save_diff(tag, r.gpu, r.cpu, w, h);
        std::fprintf(stderr,
                     "[%s] max_err=%d diff_px=%lld  first mismatch (%d,%d) GPU(%d,%d,%d,%d) CPU(%d,%d,%d,%d)\n",
                     tag.c_str(), r.diff.max_err, r.diff.diff_px, r.diff.first_x, r.diff.first_y, r.diff.gpu[0],
                     r.diff.gpu[1], r.diff.gpu[2], r.diff.gpu[3], r.diff.cpu[0], r.diff.cpu[1], r.diff.cpu[2],
                     r.diff.cpu[3]);
    }
    EXPECT_EQ(r.diff.max_err, 0) << tag << ": GPU and CPU reference must be bit-exact";
    EXPECT_EQ(r.diff.diff_px, 0) << tag;
    EXPECT_GT(r.diff.changed_gpu, 0) << tag << ": nothing was drawn";
    EXPECT_EQ(r.diff.changed_gpu, r.diff.changed_cpu) << tag;
}

// ─────────────────────────── OverlayGeometry (1920x1080, text off) ───────────────────────────

class OverlayGeometry : public ::testing::TestWithParam<int> {}; // line width 1/2/3

TEST_P(OverlayGeometry, BoundaryCasesMatchCpu) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = false;
    cfg.line_width = GetParam();
    const RunResult r = run_case(hz, cfg, boundary_cases(), "");
    expect_exact("geometry_boundary_lw" + std::to_string(GetParam()), r, 1920, 1080);
    std::printf("[geometry lw=%d] changed pixels %lld (GPU) / %lld (CPU)\n", GetParam(), r.diff.changed_gpu,
                r.diff.changed_cpu);
}

TEST_P(OverlayGeometry, EightyDetectionsMatchCpu) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = false;
    cfg.line_width = GetParam();
    const RunResult r = run_case(hz, cfg, grid_cases(80, 1920, 1080), "");
    expect_exact("geometry_80_lw" + std::to_string(GetParam()), r, 1920, 1080);
}

INSTANTIATE_TEST_SUITE_P(LineWidth, OverlayGeometry, ::testing::Values(1, 2, 3),
                         [](const ::testing::TestParamInfo<int>& i) { return "lw" + std::to_string(i.param); });

// Over the cap: given 81, only 80 are drawn and overflow is set
TEST(OverlayOverflow, TruncatesAndFlags) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = false;
    cfg.line_width = 2;
    OverlayRenderer ov(ctx(), cfg);
    const std::vector<Detection> dets = grid_cases(90, 1920, 1080);
    EXPECT_EQ(ov.set_frame(0, dets, ""), 80);
    EXPECT_EQ(ov.overflow(0), 1u);
    const OverlayAffine a{};
    hz.set_affine(a);
    const std::vector<std::uint8_t> gpu = hz.render(ov, 0);
    std::vector<std::uint8_t> cpu = background(1920, 1080);
    overlay_cpu_render(cpu.data(), 1920, 1080, dets, "", a, ov.style(), 80);
    const Diff d = compare(gpu, cpu, 1920, 1080);
    if (d.max_err) save_diff("overflow", gpu, cpu, 1920, 1080);
    EXPECT_EQ(d.max_err, 0);
    EXPECT_EQ(d.diff_px, 0);
}

// ─────────────────────────── OverlayText (labels + HUD) ───────────────────────────

class OverlayText : public ::testing::TestWithParam<int> {}; // text_scale 1/2

TEST_P(OverlayText, LabelsAndHudMatchCpu) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = true;
    cfg.text_scale = GetParam();
    cfg.line_width = 2;
    const RunResult r = run_case(hz, cfg, separated_text_cases(), "FPS 51.6 DET 8 GPU 4.23ms");
    expect_exact("text_s" + std::to_string(GetParam()), r, 1920, 1080);
    std::printf("[text s=%d] changed pixels %lld\n", GetParam(), r.diff.changed_gpu);
}

// 0 detections + HUD only: dispatch.x = max(count,1) = 1, so the HUD must appear
TEST_P(OverlayText, HudOnlyWithNoDetections) {
    OverlayHarness hz(640, 480);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = true;
    cfg.text_scale = GetParam();
    const RunResult r = run_case(hz, cfg, {}, "no detections");
    expect_exact("hud_only_s" + std::to_string(GetParam()), r, 640, 480);
}

// Full ASCII 32~126 range + unrepresentable characters (rule 1: fold to space)
TEST_P(OverlayText, FullAsciiRangeInHud) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = true;
    cfg.text_scale = GetParam();
    std::string hud;
    for (int c = 32; c <= 126; ++c) hud.push_back(static_cast<char>(c));
    hud.push_back(static_cast<char>(9));   // tab: out of range → space
    hud.push_back(static_cast<char>(200)); // non-ASCII → space
    const RunResult r = run_case(hz, cfg, {}, hud);
    expect_exact("ascii_s" + std::to_string(GetParam()), r, 1920, 1080);
}

INSTANTIATE_TEST_SUITE_P(TextScale, OverlayText, ::testing::Values(1, 2),
                         [](const ::testing::TestParamInfo<int>& i) { return "s" + std::to_string(i.param); });

// ─────────────────────────── affine: when the canvas is smaller than the source ───────────────────────────

TEST(OverlayAffineScale, UniformScaleMatchesCpu) {
    OverlayHarness hz(1280, 720);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = true;
    cfg.text_scale = 1;
    cfg.line_width = 2;
    OverlayRenderer ov(ctx(), cfg);
    // source 1920x1080 coordinates → 1280x720 canvas (uniform scale 2/3, padding 0)
    const OverlayAffine a = make_overlay_affine(1920, 1080, 1280, 720);
    EXPECT_NEAR(a.sx, 2.0f / 3.0f, 1e-6f);
    hz.set_affine(a);
    const std::vector<Detection> dets = {det(0, 0.87f, 150, 300, 600, 750), det(5, 0.31f, 900, 300, 1500, 750)};
    ov.set_frame(0, dets, "scaled");
    const std::vector<std::uint8_t> gpu = hz.render(ov, 0);
    std::vector<std::uint8_t> cpu = background(1280, 720);
    overlay_cpu_render(cpu.data(), 1280, 720, dets, "scaled", a, ov.style(), 80);
    const Diff d = compare(gpu, cpu, 1280, 720);
    if (d.max_err) save_diff("affine_scale", gpu, cpu, 1280, 720);
    EXPECT_EQ(d.max_err, 0);
    EXPECT_EQ(d.diff_px, 0);
    EXPECT_GT(d.changed_gpu, 0);
}

// ─────────────────────────── OverlayDeterminism (non-overlapping, 10 runs) ───────────────────────────

// Input is restricted to **non-overlapping** (plan §7). Intersection pixels of overlapping boxes depend on workgroup execution order, so
// determinism cannot be required — this is a documented limitation (overlay_reference.hpp R8).
void determinism_case(const std::vector<Detection>& dets, const std::string& hud, bool draw_text,
                      const char* tag) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 2; // alternating slots must give the same result
    cfg.draw_text = draw_text;
    cfg.text_scale = 2;
    cfg.line_width = 2;
    OverlayRenderer ov(ctx(), cfg);
    const OverlayAffine a{};
    hz.set_affine(a);
    std::vector<std::uint8_t> first;
    for (int i = 0; i < 10; ++i) {
        const int slot = i % cfg.num_slots;
        ov.set_frame(slot, dets, hud);
        const std::vector<std::uint8_t> out = hz.render(ov, slot);
        if (i == 0)
            first = out;
        else
            ASSERT_EQ(out, first) << tag << ": result changed at repeat " << i << "";
    }
    std::vector<std::uint8_t> cpu = background(1920, 1080);
    overlay_cpu_render(cpu.data(), 1920, 1080, dets, hud, a, ov.style(), 80);
    const Diff d = compare(first, cpu, 1920, 1080);
    if (d.max_err) save_diff(std::string("determinism_") + tag, first, cpu, 1920, 1080);
    EXPECT_EQ(d.max_err, 0) << tag;
    EXPECT_EQ(d.diff_px, 0) << tag;
    EXPECT_GT(d.changed_gpu, 0) << tag;
}

// Borders only: 40-box grid (borders non-overlapping)
TEST(OverlayDeterminism, TenRepeatsIdenticalBoxes) {
    determinism_case(grid_cases(40, 1920, 1080), "", false, "boxes40");
}

// Up to labels + HUD: 8 boxes with nothing overlapping
TEST(OverlayDeterminism, TenRepeatsIdenticalWithText) {
    determinism_case(separated_text_cases(), "deterministic", true, "text8");
}

// ─────────────────────────── OverlayVsOpenCV (reference only, not a gate) ───────────────────────────
#ifdef YVO_HAVE_OPENCV
// Compares with a 640x640 canvas drawn by OpenCV in the same way as draw_and_save() (main.cpp:450-470).
// **Bit-exactness is impossible in principle**: FONT_HERSHEY_SIMPLEX is a vector font with different metrics, and
// cv::rectangle uses a different pixel-center convention. Only the loose geometric criteria of plan §7 are checked.
TEST(OverlayVsOpenCV, GeometryWithinOnePixel) {
    const int W = 640, H = 640;
    OverlayHarness hz(W, H);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = false; // fonts differ, so compare borders only
    cfg.line_width = 1;    // same thickness as draw_and_save()
    OverlayRenderer ov(ctx(), cfg);
    const OverlayAffine a{}; // already given in 640-canvas coordinates
    hz.set_affine(a);
    const std::vector<Detection> dets = {det(0, 0.9f, 50, 60, 200, 300), det(7, 0.4f, 300, 100, 500, 400),
                                         det(20, 0.6f, 100, 400, 250, 600)};
    ov.set_frame(0, dets, "");
    const std::vector<std::uint8_t> gpu = hz.render(ov, 0);

    cv::Mat cvimg(H, W, CV_8UC3, cv::Scalar(kBg[2], kBg[1], kBg[0])); // BGR
    for (const auto& d : dets) {
        const auto rgbc = yvo::class_color(d.class_id);
        cv::rectangle(cvimg, cv::Point(static_cast<int>(std::lround(d.x1)), static_cast<int>(std::lround(d.y1))),
                      cv::Point(static_cast<int>(std::lround(d.x2)), static_cast<int>(std::lround(d.y2))),
                      cv::Scalar(rgbc[2], rgbc[1], rgbc[0]), 1);
    }

    // ① changed-mask IoU ≥ 0.6  ② no changed pixel more than 20 px away from every box
    long long inter = 0, uni = 0, far_px = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t o = (static_cast<std::size_t>(y) * W + static_cast<std::size_t>(x)) * 4;
            const bool g = gpu[o] != kBg[0] || gpu[o + 1] != kBg[1] || gpu[o + 2] != kBg[2];
            const cv::Vec3b p = cvimg.at<cv::Vec3b>(y, x);
            const bool c = p[0] != kBg[2] || p[1] != kBg[1] || p[2] != kBg[0];
            if (g && c) ++inter;
            if (g || c) ++uni;
            if (!g) continue;
            double best = 1e9;
            for (const auto& d : dets) {
                const double dx = std::max({static_cast<double>(d.x1) - x, x - static_cast<double>(d.x2), 0.0});
                const double dy = std::max({static_cast<double>(d.y1) - y, y - static_cast<double>(d.y2), 0.0});
                best = std::min(best, std::max(dx, dy));
            }
            if (best > 20.0) ++far_px;
        }
    }
    const double iou = uni ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
    std::printf("[vs OpenCV] mask IoU %.4f  inter %lld  union %lld  far(>20px) %lld\n", iou, inter, uni, far_px);
    EXPECT_GE(iou, 0.6);
    EXPECT_EQ(far_px, 0);

    // ③ is there a non-background pixel within ±1 px of each of the box's 4 edges
    auto nonbg = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= W || y >= H) return false;
        const std::size_t o = (static_cast<std::size_t>(y) * W + static_cast<std::size_t>(x)) * 4;
        return gpu[o] != kBg[0] || gpu[o + 1] != kBg[1] || gpu[o + 2] != kBg[2];
    };
    for (const auto& d : dets) {
        const int x1 = static_cast<int>(std::lround(d.x1)), y1 = static_cast<int>(std::lround(d.y1));
        const int x2 = static_cast<int>(std::lround(d.x2)), y2 = static_cast<int>(std::lround(d.y2));
        const int mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        bool top = false, bot = false, left = false, right = false;
        for (int k = -1; k <= 1; ++k) {
            top = top || nonbg(mx, y1 + k);
            bot = bot || nonbg(mx, y2 + k);
            left = left || nonbg(x1 + k, my);
            right = right || nonbg(x2 + k, my);
        }
        EXPECT_TRUE(top && bot && left && right) << "class " << d.class_id << ": 4 edges are not within ±1 px";
    }
}
#endif


// ─────────────────────────── OverlayPlane (dma-buf box plane) ───────────────────────────

using yvo::display::OverlayClearMode;
using yvo::display::OverlayPlane;
using yvo::display::OverlayPlaneBuffer;
using yvo::display::OverlayPlaneConfig;

constexpr std::array<std::uint8_t, 4> kClear{0, 0, 0, 0};

// Reads the export fd via mmap (same path as the consumer). Strips the pitch and returns w*h*4, and
// pad_nonzero counts the nonzero bytes among the row-end padding bytes.
std::vector<std::uint8_t> read_plane(const OverlayPlaneBuffer& b, long long* pad_nonzero = nullptr) {
    void* m = ::mmap(nullptr, b.size, PROT_READ, MAP_SHARED, b.fd, 0);
    if (m == MAP_FAILED) {
        ADD_FAILURE() << "mmap(dma-buf) failed errno=" << errno;
        return {};
    }
    dma_buf_sync sync{};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    EXPECT_EQ(::ioctl(b.fd, DMA_BUF_IOCTL_SYNC, &sync), 0) << "DMA_BUF_IOCTL_SYNC START";
    const auto* src = static_cast<const std::uint8_t*>(m);
    const std::size_t row = static_cast<std::size_t>(b.width) * 4;
    std::vector<std::uint8_t> out(row * b.height);
    long long pad = 0;
    for (std::uint32_t y = 0; y < b.height; ++y) {
        const std::uint8_t* r = src + static_cast<std::size_t>(y) * b.pitch;
        std::copy(r, r + row, out.begin() + static_cast<std::ptrdiff_t>(row * y));
        for (std::size_t k = row; k < b.pitch; ++k) pad += r[k] != 0;
    }
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    EXPECT_EQ(::ioctl(b.fd, DMA_BUF_IOCTL_SYNC, &sync), 0) << "DMA_BUF_IOCTL_SYNC END";
    ::munmap(m, b.size);
    if (pad_nonzero) *pad_nonzero = pad;
    return out;
}

OverlayPlaneConfig plane_cfg(int w, int h, OverlayClearMode mode, bool text, int ring = 3) {
    OverlayPlaneConfig c;
    c.ring = ring;
    c.width = w;
    c.height = h;
    c.src_w = w;
    c.src_h = h;
    c.clear = mode;
    c.overlay.draw_text = text;
    c.overlay.text_scale = 2;
    c.overlay.line_width = 2;
    return c;
}

// One frame: acquire → draw → wait → mmap read → compare with CPU reference (transparent background) → release
RunResult plane_frame(OverlayPlane& pl, const std::vector<Detection>& dets, const std::string& hud, int* used_idx,
                      long long* pad_nonzero = nullptr) {
    const int idx = pl.acquire();
    EXPECT_GE(idx, 0) << "no Free entry";
    if (used_idx) *used_idx = idx;
    RunResult r;
    if (idx < 0) return r;
    pl.draw(idx, dets, hud);
    pl.wait(idx);
    const OverlayPlaneBuffer& b = pl.buffer(idx);
    const int w = static_cast<int>(b.width), h = static_cast<int>(b.height);
    r.gpu = read_plane(b, pad_nonzero);
    r.cpu.assign(static_cast<std::size_t>(w) * h * 4, 0);
    overlay_cpu_render(r.cpu.data(), w, h, dets, hud, pl.affine(), pl.style());
    r.diff = compare(r.gpu, r.cpu, w, h, kClear);
    pl.release(idx);
    return r;
}

void expect_plane_exact(const std::string& tag, const RunResult& r, int w, int h, bool expect_drawn = true) {
    if (r.diff.max_err != 0 || r.diff.diff_px != 0) {
        save_diff(tag, r.gpu, r.cpu, w, h);
        std::fprintf(stderr, "[%s] max_err=%d diff_px=%lld first mismatch (%d,%d) GPU(%d,%d,%d,%d) CPU(%d,%d,%d,%d)\n",
                     tag.c_str(), r.diff.max_err, r.diff.diff_px, r.diff.first_x, r.diff.first_y, r.diff.gpu[0],
                     r.diff.gpu[1], r.diff.gpu[2], r.diff.gpu[3], r.diff.cpu[0], r.diff.cpu[1], r.diff.cpu[2],
                     r.diff.cpu[3]);
    }
    EXPECT_EQ(r.diff.max_err, 0) << tag << ": dma-buf contents must be bit-exact with the CPU reference (transparent background)";
    EXPECT_EQ(r.diff.diff_px, 0) << tag;
    EXPECT_EQ(r.diff.changed_gpu, r.diff.changed_cpu) << tag;
    if (expect_drawn) {
        EXPECT_GT(r.diff.changed_gpu, 0) << tag << ": nothing was drawn";
    }
}

// (c) Descriptor: AB24 / LINEAR / pitch 64 B alignment / size
TEST(OverlayPlane, BufferDescriptor) {
    OverlayPlane pl(ctx(), plane_cfg(1920, 1080, OverlayClearMode::Auto, true));
    ASSERT_EQ(pl.ring_size(), 3);
    for (int i = 0; i < pl.ring_size(); ++i) {
        const OverlayPlaneBuffer& b = pl.buffer(i);
        std::printf("[plane] #%d fd=%d fourcc=0x%08x(AB24) mod=0x%llx %ux%u pitch=%u size=%zu dmabuf=%zu clear=%s\n", i,
                    b.fd, b.fourcc, static_cast<unsigned long long>(b.modifier), b.width, b.height, b.pitch, b.size,
                    b.dmabuf_size, yvo::display::clear_mode_name(pl.clear_mode()));
        EXPECT_GE(b.fd, 0);
        EXPECT_EQ(b.fourcc, 0x34324241u);   // 'A','B','2','4' = DRM_FORMAT_ABGR8888
        EXPECT_EQ(b.modifier, 0u);          // DRM_FORMAT_MOD_LINEAR
        EXPECT_EQ(b.offset, 0u);
        EXPECT_EQ(b.width, 1920u);
        EXPECT_EQ(b.height, 1080u);
        EXPECT_EQ(b.pitch % 64u, 0u);
        EXPECT_EQ(b.pitch, 1920u * 4u);     // 7680 is already a multiple of 64 → no padding
        EXPECT_EQ(b.size, static_cast<std::size_t>(b.pitch) * b.height);
        EXPECT_GE(b.dmabuf_size, b.size);
        const int fd2 = pl.dup_fd(i);
        EXPECT_GE(fd2, 0);
        EXPECT_NE(fd2, b.fd);
        if (fd2 >= 0) ::close(fd2);
    }
    // Right after creation every entry must be transparent (initial fill)
    long long pad = -1;
    const std::vector<std::uint8_t> v = read_plane(pl.buffer(2), &pad);
    EXPECT_TRUE(std::all_of(v.begin(), v.end(), [](std::uint8_t x) { return x == 0; }));
}

// (c) Width not a multiple of 64: pitch padding appears, and the shader does not touch the padding bytes
TEST(OverlayPlane, PaddedPitchMatchesCpu) {
    OverlayPlane pl(ctx(), plane_cfg(1000, 600, OverlayClearMode::Auto, true));
    const OverlayPlaneBuffer& b = pl.buffer(0);
    EXPECT_EQ(b.pitch, 4032u);   // align_up(4000, 64)
    const std::vector<Detection> d = {det(0, 0.9f, 10, 100, 300, 400), det(5, 0.5f, 700, 100, 999, 599),
                                      det(9, 0.7f, 950, 300, 1200, 350)};   // overflows off the right of the screen
    long long pad = -1;
    const RunResult r = plane_frame(pl, d, "HUD pad", nullptr, &pad);
    expect_plane_exact("plane_pitch_pad", r, 1000, 600);
    EXPECT_EQ(pad, 0) << "must not write to the row-end padding bytes";
}

class OverlayPlaneMode : public ::testing::TestWithParam<OverlayClearMode> {};

// (a) First frame: edge cases (text off) and labels+HUD (text on)
TEST_P(OverlayPlaneMode, DrawMatchesCpu) {
    {
        OverlayPlane pl(ctx(), plane_cfg(1920, 1080, GetParam(), false));
        expect_plane_exact("plane_boundary", plane_frame(pl, boundary_cases(), "", nullptr), 1920, 1080);
    }
    {
        OverlayPlane pl(ctx(), plane_cfg(1920, 1080, GetParam(), true));
        expect_plane_exact("plane_text", plane_frame(pl, separated_text_cases(), "FPS 30.0 dets 8", nullptr), 1920,
                           1080);
    }
}

// (b) Reuse: with ring=1, keep reusing the same entry while drawing different detections/HUD → zero ghosting every frame.
//     Same with ring=3 over 9 frames (each entry's previous contents differ).
TEST_P(OverlayPlaneMode, ReuseLeavesNoGhost) {
    const std::vector<Detection> a = separated_text_cases();
    const std::vector<Detection> b = {det(20, 0.31f, 200, 250, 700, 650), det(40, 0.93f, 1000, 600, 1800, 1000)};
    std::vector<Detection> c;   // moved boxes (a shifted right-down by 37,23 px)
    for (Detection d : a) {
        d.x1 += 37;
        d.x2 += 37;
        d.y1 += 23;
        d.y2 += 23;
        c.push_back(d);
    }
    const std::vector<std::vector<Detection>> seq = {a, b, {}, c, a, b, c, {}, a};
    const std::vector<std::string> huds = {"frame 0 long hud text", "f1", "", "frame 3", "f4 xx", "", "f6", "f7", "f8"};
    for (int ring : {1, 3}) {
        OverlayPlane pl(ctx(), plane_cfg(1920, 1080, GetParam(), true, ring));
        for (std::size_t i = 0; i < seq.size(); ++i) {
            int idx = -1;
            const RunResult r = plane_frame(pl, seq[i], huds[i], &idx);
            const bool drawn = !seq[i].empty() || !huds[i].empty();
            expect_plane_exact("plane_reuse_r" + std::to_string(ring) + "_f" + std::to_string(i), r, 1920, 1080,
                               drawn);
            if (!drawn) {
                EXPECT_EQ(r.diff.changed_gpu, 0) << "an empty frame must be fully transparent";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Clear, OverlayPlaneMode, ::testing::Values(OverlayClearMode::Full, OverlayClearMode::Erase),
                         [](const ::testing::TestParamInfo<OverlayClearMode>& i) {
                             return std::string(yvo::display::clear_mode_name(i.param));
                         });

// Ring state: reused only after GPU completion + consumer release (wl_buffer.release)
TEST(OverlayPlane, RingStateMachine) {
    OverlayPlane pl(ctx(), plane_cfg(640, 360, OverlayClearMode::Auto, true));
    const int i0 = pl.acquire(), i1 = pl.acquire(), i2 = pl.acquire();
    ASSERT_GE(i0, 0);
    ASSERT_GE(i1, 0);
    ASSERT_GE(i2, 0);
    EXPECT_EQ(pl.acquire(), -1) << "-1 when all are Claimed";
    const std::vector<Detection> d = {det(1, 0.5f, 10, 60, 100, 200)};
    pl.draw(i0, d, "x");
    EXPECT_THROW(pl.draw(i0, d, "x"), std::logic_error) << "draw again on an entry with a pending submit";
    pl.wait(i0);
    EXPECT_TRUE(pl.ready(i0));
    pl.mark_held(i0);                    // handed to the consumer (wl_surface.commit)
    EXPECT_TRUE(pl.held(i0));
    pl.release(i1);                      // entry decided not to be displayed (drop)
    pl.release(i2);
    // i0, which is Held, does not come out
    const int a = pl.acquire(), b = pl.acquire();
    EXPECT_NE(a, i0);
    EXPECT_NE(b, i0);
    EXPECT_EQ(pl.acquire(), -1) << "Held entries are not reused until release";
    pl.release(i0);                      // wl_buffer.release
    EXPECT_EQ(pl.acquire(), i0);
    EXPECT_THROW(pl.mark_held(-1), std::out_of_range);
    pl.release(a);
    pl.release(b);
    pl.release(i0);
    EXPECT_THROW(pl.release(i0), std::logic_error);
}

// Clear cost: GPU time at 1920x1080 of Full (vkCmdFillBuffer over everything) vs Erase (redraw only the previous detections writing 0).
// Not a gate (measurement has its own 07_results conventions); leaves the rationale for the choice in the log.
TEST(OverlayPlane, ClearCost) {
    if (!ctx().timestamps_supported()) GTEST_SKIP() << "timestamp not supported";
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? -1.0 : v[v.size() / 2];
    };
    for (int ndet : {8, 80}) {
        std::vector<Detection> d = ndet == 8 ? separated_text_cases() : grid_cases(80, 1920, 1080);
        for (OverlayClearMode m : {OverlayClearMode::Full, OverlayClearMode::Erase}) {
            OverlayPlane pl(ctx(), plane_cfg(1920, 1080, m, ndet == 8));
            std::vector<double> clr, ovl;
            for (int f = 0; f < 60; ++f) {
                const int idx = pl.acquire();
                ASSERT_GE(idx, 0);
                pl.draw(idx, d, "FPS 30.0 clear cost");
                pl.wait(idx);
                pl.release(idx);
                if (f >= 10) {   // after warm-up + every entry drawn once
                    clr.push_back(pl.last_clear_ms());
                    ovl.push_back(pl.last_overlay_ms());
                }
            }
            std::printf("[clear-cost] dets=%d text=%d mode=%s clear median %.3f ms  overlay median %.3f ms (n=%zu)\n",
                        ndet, ndet == 8 ? 1 : 0, yvo::display::clear_mode_name(m), median(clr), median(ovl),
                        clr.size());
            EXPECT_GE(median(clr), 0.0);
        }
    }
}

// ── DrmState: debugfs /sys/kernel/debug/dri/1/state parser (display/drm_state.hpp) ─────────────────
// Fixtures are excerpts of measured state (plane block structure, indentation and the obj block's "size=<bytes>" line are kept as is).
//   kStatePlanesOk : hvs_wl_probe --overlay on fullhack f225 (2026-09-22, labwc 0.20.1) — NV12 SAND video plane[83]
//                    zpos 0 + AB24 box plane[107] zpos 1 + labwc cursor AR24 plane[657] zpos 2, 1 inactive plane.
//   kStateDesktop  : desktop (a single labwc XR24 fb) — no video plane = GL composition / not displayed.
constexpr const char* kStatePlanesOk = R"(plane[48]: plane-0
	crtc=(null)
	fb=0
	crtc-pos=1920x1080+0+0
	src-pos=1920.000000x1080.000000+0.000000+0.000000
	rotation=1
	normalized-zpos=0
	color-encoding=ITU-R BT.709 YCbCr
	color-range=YCbCr limited range
	color_mgmt_changed=0
plane[83]: plane-2
	crtc=crtc-2
	fb=697
		allocated by = labwc
		refcount=3
		format=NV12 little-endian (0x3231564e)
		modifier=0x700000000000004
		size=1920x1080
		layers:
			size[0]=1920x1080
			pitch[0]=1088
			offset[0]=0
			obj[0]:
				name=0
				refcount=2
				start=0004182b
				size=2097152
				imported=yes
			size[1]=960x540
			pitch[1]=544
			offset[1]=0
	crtc-pos=1920x1080+0+0
	src-pos=1920.000000x1080.000000+0.000000+0.000000
	rotation=1
	normalized-zpos=0
	color-encoding=ITU-R BT.709 YCbCr
	color-range=YCbCr limited range
	color_mgmt_changed=0
plane[107]: plane-4
	crtc=crtc-2
	fb=691
		allocated by = labwc
		refcount=3
		format=AB24 little-endian (0x34324241)
		modifier=0x0
		size=1920x1080
		layers:
			size[0]=1920x1080
			pitch[0]=7680
			offset[0]=0
			obj[0]:
				name=0
				size=8306688
	crtc-pos=1920x1080+0+0
	src-pos=1920.000000x1080.000000+0.000000+0.000000
	rotation=1
	normalized-zpos=1
	color-encoding=ITU-R BT.709 YCbCr
	color-range=YCbCr limited range
	color_mgmt_changed=0
plane[657]: plane-54
	crtc=crtc-2
	fb=681
		allocated by = labwc
		format=AR24 little-endian (0x34325241)
		modifier=0x0
		size=64x64
	crtc-pos=64x64+1887+1
	src-pos=64.000000x64.000000+0.000000+0.000000
	rotation=1
	normalized-zpos=2
	color-encoding=ITU-R BT.709 YCbCr
	color-range=YCbCr limited range
	color_mgmt_changed=0
crtc[94]: crtc-2
	enable=1
	active=1
	plane_mask=0x7
connector[99]: HDMI-A-1
	crtc=crtc-2
)";

constexpr const char* kStateDesktop = R"(plane[83]: plane-2
	crtc=crtc-2
	fb=700
		allocated by = labwc
		refcount=2
		format=XR24 little-endian (0x34325258)
		modifier=0x0
		size=1920x1080
	crtc-pos=1920x1080+0+0
	src-pos=1920.000000x1080.000000+0.000000+0.000000
	rotation=1
	normalized-zpos=0
	color-encoding=ITU-R BT.601 YCbCr
	color-range=YCbCr full range
	color_mgmt_changed=0
crtc[94]: crtc-2
	enable=1
)";

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (std::size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size())) s.replace(p, from.size(), to);
    return s;
}

TEST(DrmState, ParsesActivePlanesOnly) {
    const auto planes = yvo::display::parse_drm_active_planes(kStatePlanesOk);
    ASSERT_EQ(planes.size(), 3u);   // plane[48] has crtc=(null) and is excluded
    EXPECT_EQ(planes[0].id, 83);
    EXPECT_EQ(planes[0].name, "plane-2");
    EXPECT_EQ(planes[0].crtc, "crtc-2");
    EXPECT_EQ(planes[0].alloc, "labwc");
    EXPECT_EQ(planes[0].fourcc, 0x3231564eu);
    EXPECT_EQ(planes[0].modifier, 0x0700000000000004ull);
    EXPECT_EQ(planes[0].size, "1920x1080");   // not the obj block's size=2097152
    EXPECT_EQ(planes[0].zpos, 0);
    EXPECT_EQ(planes[1].id, 107);
    EXPECT_EQ(planes[1].fourcc, 0x34324241u);
    EXPECT_EQ(planes[1].modifier, 0u);
    EXPECT_EQ(planes[1].zpos, 1);
    EXPECT_EQ(planes[2].id, 657);
    EXPECT_EQ(planes[2].size, "64x64");
    EXPECT_TRUE(yvo::display::is_nv12_sand128(planes[0]));
    EXPECT_TRUE(yvo::display::is_ab24_linear(planes[1]));
    EXPECT_FALSE(yvo::display::is_ab24_linear(planes[2]));   // the cursor is AR24
}

TEST(DrmState, VideoPlaneWithBoxesAboveIsZeroCopyAndColorOk) {
    const auto v = yvo::display::judge_plane_scanout(yvo::display::parse_drm_active_planes(kStatePlanesOk));
    EXPECT_TRUE(v.video_on_plane);
    EXPECT_EQ(v.video_plane, 83);
    EXPECT_EQ(v.overlay_plane, 107);   // the AB24 right above the video, not the cursor (zpos 2)
    EXPECT_TRUE(v.overlay_above);
    EXPECT_TRUE(v.encoding_bt709);
    EXPECT_TRUE(v.range_limited);
    EXPECT_TRUE(v.zero_copy(true));
    EXPECT_TRUE(v.color_ok());
}

TEST(DrmState, SandModifierWithColumnHeightParamStillSand128) {
    // Carrying the column height as a modifier parameter (DRM_FORMAT_MOD_BROADCOM_SAND128_COL_HEIGHT(1088)) is also SAND128.
    const std::string st = replace_all(kStatePlanesOk, "modifier=0x700000000000004", "modifier=0x700000000044004");
    const auto v = yvo::display::judge_plane_scanout(yvo::display::parse_drm_active_planes(st));
    EXPECT_TRUE(v.video_on_plane);
    // SAND256 (kind 5) is not
    const std::string st2 = replace_all(kStatePlanesOk, "modifier=0x700000000000004", "modifier=0x700000000000005");
    EXPECT_FALSE(yvo::display::judge_plane_scanout(yvo::display::parse_drm_active_planes(st2)).video_on_plane);
}

TEST(DrmState, FullRangeOrBt601IsColorNotOk) {
    // Change only the video plane's (plane[83]) range to full — the first "limited" belongs to plane[48] (inactive), so
    // find and replace the line inside the plane[83] block.
    std::string st = kStatePlanesOk;
    const auto b = st.find("plane[83]");
    const auto r = st.find("color-range=YCbCr limited range", b);
    st.replace(r, std::string("color-range=YCbCr limited range").size(), "color-range=YCbCr full range");
    auto v = yvo::display::judge_plane_scanout(yvo::display::parse_drm_active_planes(st));
    EXPECT_TRUE(v.video_on_plane);
    EXPECT_FALSE(v.range_limited);
    EXPECT_FALSE(v.color_ok());
    EXPECT_TRUE(v.zero_copy(true));   // the placement is still zero-copy

    std::string st601 = kStatePlanesOk;
    const auto e = st601.find("color-encoding=ITU-R BT.709 YCbCr", st601.find("plane[83]"));
    st601.replace(e, std::string("color-encoding=ITU-R BT.709 YCbCr").size(), "color-encoding=ITU-R BT.601 YCbCr");
    v = yvo::display::judge_plane_scanout(yvo::display::parse_drm_active_planes(st601));
    EXPECT_FALSE(v.encoding_bt709);
    EXPECT_FALSE(v.color_ok());
}

TEST(DrmState, BoxesBelowVideoAreNotAbove) {
    std::string st = kStatePlanesOk;
    // video zpos 0 → 3: the box (zpos 1) goes below the video
    const auto v83 = st.find("normalized-zpos=0", st.find("plane[83]"));
    st.replace(v83, std::string("normalized-zpos=0").size(), "normalized-zpos=3");
    const auto v = yvo::display::judge_plane_scanout(yvo::display::parse_drm_active_planes(st));
    EXPECT_TRUE(v.video_on_plane);
    EXPECT_FALSE(v.overlay_above);
    EXPECT_FALSE(v.zero_copy(true));   // the box was committed but is not above the video
    EXPECT_TRUE(v.zero_copy(false));   // if the box was not committed, only the video is checked
}

TEST(DrmState, DesktopIsNotZeroCopy) {
    const auto planes = yvo::display::parse_drm_active_planes(kStateDesktop);
    ASSERT_EQ(planes.size(), 1u);
    const auto v = yvo::display::judge_plane_scanout(planes);
    EXPECT_FALSE(v.video_on_plane);
    EXPECT_FALSE(v.zero_copy(false));
    EXPECT_FALSE(v.color_ok());
    EXPECT_NE(v.text.find("no NV12/SAND128 plane"), std::string::npos);
}

TEST(DrmState, UnreadableIsUnverified) {
    const auto snap = yvo::display::take_drm_state_snapshot("/nonexistent/dri/1");
    EXPECT_FALSE(snap.readable);
    EXPECT_FALSE(snap.error.empty());
    EXPECT_EQ(snap.hvs_underrun, -1);
    EXPECT_TRUE(yvo::display::parse_drm_active_planes("").empty());
}

TEST(DrmState, HvsUnderrunCounter) {
    EXPECT_EQ(yvo::display::parse_hvs_underrun("0\n"), 0);
    EXPECT_EQ(yvo::display::parse_hvs_underrun(" 17\n"), 17);
    EXPECT_EQ(yvo::display::parse_hvs_underrun(""), -1);
    EXPECT_EQ(yvo::display::parse_hvs_underrun("n/a"), -1);
}

// ── KmsScanout (--display-backend kms transport layer) ─────────────────────────────────────────────────
// Safety rule: never make calls that take master or change the screen. Every KmsScanout is created with acquire_master=false
// (the only exception is MasterErrorNamesHolder, which exercises the master acquisition path — it runs only when another process is master, and then drmSetMaster is
// rejected by the kernel), and commits are rejected before the ioctl because there is no master.
using yvo::display::KmsScanout;
using yvo::display::KmsScanoutOptions;

constexpr const char* kCard = "/dev/dri/card1";

// true only when another process is master. Otherwise fills in the skip reason.
bool kms_guard(std::string* why) {
    switch (KmsScanout::probe_master(kCard)) {
    case KmsScanout::MasterProbe::Unavailable:
        *why = std::string(kCard) + " cannot be opened (no device / permission)";
        return false;
    case KmsScanout::MasterProbe::NoMaster:
        *why = "nobody is DRM master (desktop on an inactive VT?) — skip, risk of changing the screen";
        return false;
    case KmsScanout::MasterProbe::OtherMaster: return true;
    }
    return false;
}

KmsScanoutOptions kms_test_opts() {
    KmsScanoutOptions o;
    o.card = kCard;
    o.acquire_master = false;
    o.verbose = true;
    return o;
}

// First decoder frame (NV12 SAND128 dma-buf)
struct KmsTestFrame {
    yvo::FfmpegDrmDecoder dec;
    std::optional<yvo::DrmFrame> f;
    bool open() {
        if (!dec.open(YVO_DEFAULT_VIDEO)) return false;
        f = dec.next();
        return f.has_value();
    }
};

TEST(KmsScanout, DiscoversConnectorCrtcAndPlanes) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    std::printf("[kms] master holders: %s\n", KmsScanout::find_master_holders(yvo::display::kDefaultDriDebugDir).c_str());
    KmsScanout k(kms_test_opts());
    EXPECT_FALSE(k.is_master());
    EXPECT_EQ(k.connector_name().rfind("HDMI-A-", 0), 0u) << k.connector_name();
    EXPECT_NE(k.connector_id(), 0u);
    EXPECT_NE(k.crtc_id(), 0u);
    EXPECT_NE(k.video_plane_id(), 0u);
    EXPECT_NE(k.overlay_plane_id(), 0u);
    EXPECT_NE(k.video_plane_id(), k.overlay_plane_id());
    EXPECT_TRUE(k.video_plane_nv12_sand128());
    EXPECT_TRUE(k.overlay_plane_ab24_linear());
    EXPECT_TRUE(k.timestamp_monotonic());
    const VkExtent2D e = k.output_extent();
    EXPECT_GT(e.width, 0u);
    EXPECT_GT(e.height, 0u);
    EXPECT_STREQ(k.name(), "kms");
    EXPECT_TRUE(k.reports_presentation());
    EXPECT_FALSE(k.quit_requested());
    EXPECT_FALSE(k.should_close());
    const std::string d = k.describe();
    std::printf("[kms] %s\n", d.c_str());
    EXPECT_NE(d.find("color=BT.709/limited"), std::string::npos) << d;   // auto = limited
    EXPECT_NE(d.find(" modeset:"), std::string::npos) << d;             // color properties to set at modeset (B11)
}

TEST(KmsScanout, UnknownConnectorIsClearError) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    KmsScanoutOptions o = kms_test_opts();
    o.connector = "HDMI-A-99";
    try {
        KmsScanout k(o);
        FAIL() << "no exception for a nonexistent connector";
    } catch (const std::runtime_error& ex) {
        std::printf("[kms] %s\n", ex.what());
        EXPECT_NE(std::string(ex.what()).find("HDMI-A-99"), std::string::npos);
    }
}

TEST(KmsScanout, OverlayBuffersAddFB2) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    KmsScanout k(kms_test_opts());
    const VkExtent2D e = k.output_extent();
    OverlayPlane pl(ctx(), plane_cfg(static_cast<int>(e.width), static_cast<int>(e.height), OverlayClearMode::Auto, true));
    ASSERT_NO_THROW(k.register_overlay(pl));
    std::vector<std::uint32_t> fbs;
    for (int i = 0; i < pl.ring_size(); ++i) {
        fbs.push_back(k.overlay_fb(i));
        EXPECT_NE(fbs.back(), 0u) << "#" << i;
    }
    std::sort(fbs.begin(), fbs.end());
    EXPECT_EQ(std::unique(fbs.begin(), fbs.end()), fbs.end()) << "FB ids collide";
    EXPECT_EQ(k.overlay_fb(pl.ring_size()), 0u);
    ASSERT_NO_THROW(k.register_overlay(pl));   // re-registering cleans up the previous FBs and creates new ones
    EXPECT_NE(k.overlay_fb(0), 0u);
}

TEST(KmsScanout, VideoFbFromDecoderFrame) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    KmsTestFrame tf;
    if (!tf.open()) GTEST_SKIP() << "no HEVC decoder/video: " << YVO_DEFAULT_VIDEO;
    KmsScanout k(kms_test_opts());
    const std::uint32_t fb = k.video_fb_for(*tf.f);
    ASSERT_NE(fb, 0u) << k.last_error();
    EXPECT_EQ(k.video_fb_for(*tf.f), fb);   // (dev, ino) cache
    EXPECT_EQ(k.video_fb_count(), 1u);
    k.clear_video_cache();
    EXPECT_EQ(k.video_fb_count(), 0u);
    yvo::DrmFrame bad = *tf.f;
    bad.fd = -1;
    EXPECT_EQ(k.video_fb_for(bad), 0u);
    EXPECT_FALSE(k.last_error().empty());
}

TEST(KmsScanout, CommitWithoutMasterFailsFastWithoutTouchingScreen) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    KmsTestFrame tf;
    if (!tf.open()) GTEST_SKIP() << "no HEVC decoder/video: " << YVO_DEFAULT_VIDEO;
    KmsScanout k(kms_test_opts());
    const VkExtent2D e = k.output_extent();
    OverlayPlane pl(ctx(), plane_cfg(static_cast<int>(e.width), static_cast<int>(e.height), OverlayClearMode::Auto, true));
    k.register_overlay(pl);
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_FALSE(k.commit(*tf.f, 1, 0));
    EXPECT_FALSE(k.commit(*tf.f, 2, -1));
    EXPECT_TRUE(k.dispatch(0));
    EXPECT_TRUE(k.dispatch(20));      // no events, so it just returns after 20 ms
    k.shutdown(100);
    EXPECT_FALSE(k.commit(*tf.f, 3, 0));   // commit after shutdown is rejected
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[kms] no-master commit error: %s (%.1f ms)\n", k.last_error().c_str(), ms);
    EXPECT_NE(k.last_error().find("master"), std::string::npos) << k.last_error();
    EXPECT_EQ(k.commit_failures(), 2u);
    EXPECT_LT(ms, 1000.0);
}

// Full PlaneDisplay flow: even if every commit is rejected for lack of master, display_done is set and drain finishes in finite time.
TEST(KmsScanout, PlaneDisplayWithoutMasterDropsAndDrains) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    KmsTestFrame tf;
    if (!tf.open()) GTEST_SKIP() << "no HEVC decoder/video: " << YVO_DEFAULT_VIDEO;
    yvo::display::DisplayPipelineConfig cfg;
    cfg.backend = yvo::display::DisplayBackendKind::Kms;
    cfg.pace = yvo::display::DisplayPace::None;
    cfg.num_slots = 2;
    cfg.src_w = tf.f->width;
    cfg.src_h = tf.f->height;
    std::vector<int> done;
    const auto t0 = std::chrono::steady_clock::now();
    {
        yvo::display::PlaneDisplay pd(ctx(), std::make_unique<KmsScanout>(kms_test_opts()), cfg);
        pd.set_display_done_callback([&](int s) { done.push_back(s); });
        EXPECT_STREQ(pd.backend_name(), "kms");
        pd.retain(0, *tf.f, std::chrono::steady_clock::now());
        pd.show(0, {Detection{0, 0.9f, 100.f, 100.f, 300.f, 300.f}});
        pd.pump();
        pd.retain(1, *tf.f, std::chrono::steady_clock::now());
        pd.show(1, {});
        pd.pump();
        pd.drain();
        EXPECT_EQ(pd.presented(), 0u);
        EXPECT_EQ(pd.display_dropped(), 2u);
        EXPECT_EQ(pd.frames_held(), 0);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_EQ(done, (std::vector<int>{0, 1}));
    EXPECT_LT(ms, 3000.0);
}

// Save step of shutdown restore (A1): read and serialize the current KMS state without master (the restore commit itself needs master and is checked on a real display).
TEST(KmsScanout, CapturesCurrentStateWithoutMaster) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    KmsScanout k(kms_test_opts());
    EXPECT_TRUE(k.saved_state().empty());
    k.capture_state();
    const std::string t = k.saved_state_text();
    std::printf("[kms] saved state (%zu props):\n%s", k.saved_state().size(), t.c_str());
    ASSERT_FALSE(k.saved_state().empty());
    const std::string crtc = "crtc[" + std::to_string(k.crtc_id()) + "].";
    const std::string conn = "connector[" + std::to_string(k.connector_id()) + "].";
    const std::string vp = "plane[" + std::to_string(k.video_plane_id()) + "].";
    const std::string op = "plane[" + std::to_string(k.overlay_plane_id()) + "].";
    for (const std::string& key : {crtc + "MODE_ID=", crtc + "ACTIVE=", conn + "CRTC_ID=", vp + "FB_ID=",
                                   vp + "CRTC_ID=", vp + "SRC_W=", vp + "CRTC_W=", op + "FB_ID=", op + "CRTC_ID="})
        EXPECT_NE(t.find(key), std::string::npos) << key;
    // immutable properties (type, IN_FORMATS) are not restore targets
    EXPECT_EQ(t.find(".type="), std::string::npos);
    EXPECT_EQ(t.find(".IN_FORMATS="), std::string::npos);
    // saved planes are either our two planes or planes currently attached to our CRTC
    std::map<std::uint32_t, std::uint64_t> plane_crtc;
    for (const auto& sp : k.saved_state())
        if (sp.kind == "plane" && sp.name == "CRTC_ID") plane_crtc[sp.obj] = sp.value;
    for (const auto& [id, c] : plane_crtc)
        if (id != k.video_plane_id() && id != k.overlay_plane_id()) {
            EXPECT_EQ(c, k.crtc_id()) << "plane " << id;
        }
    // shutdown before modeset: ends without a restore commit (screen and master unchanged)
    k.shutdown(0);
    EXPECT_EQ(k.shutdown_result().rfind("not-needed", 0), 0u) << k.shutdown_result();
    bool has_line = false;
    for (const auto& l : k.report_lines()) {
        std::printf("[kms] report: %s\n", l.c_str());
        if (l.find("flip_timeouts 0") != std::string::npos && l.find("master_lost no") != std::string::npos) has_line = true;
    }
    EXPECT_TRUE(has_line);
    EXPECT_TRUE(k.fatal_error().empty());
    EXPECT_FALSE(k.should_close());
    EXPECT_EQ(KmsScanout::probe_master(kCard), KmsScanout::MasterProbe::OtherMaster);
}

// Path used by main: acquire_master=true but another process is master → KmsMasterError (exit code 2), including the owner name.
// The kernel rejects drmSetMaster while a master exists, so the screen and master stay unchanged.
TEST(KmsScanout, MasterErrorNamesHolder) {
    std::string why;
    if (!kms_guard(&why)) GTEST_SKIP() << why;
    KmsScanoutOptions o = kms_test_opts();
    o.acquire_master = true;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        KmsScanout k(o);
        // Reaching here is a race (master became free between the guard and construction). It is before the first commit, so the screen is unchanged,
        // and the destructor releases master.
        ADD_FAILURE() << "acquired master even though another process is master";
    } catch (const yvo::display::KmsMasterError& ex) {
        const std::string m = ex.what();
        std::printf("[kms] %s\n", m.c_str());
        EXPECT_NE(m.find("master"), std::string::npos);
        const std::string holders = KmsScanout::find_master_holders(yvo::display::kDefaultDriDebugDir);
        if (!holders.empty() && holders != "(none)") {
            EXPECT_NE(m.find(holders), std::string::npos) << holders;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(ms, 1000.0);
    EXPECT_EQ(KmsScanout::probe_master(kCard), KmsScanout::MasterProbe::OtherMaster);   // master unchanged
}

// Validation layer counters (meaningful only when enabled)
TEST(OverlayValidation, NoErrors) {
    std::printf("[validation] errors=%llu warnings=%llu layer=%s\n",
                static_cast<unsigned long long>(ctx().validation_errors()),
                static_cast<unsigned long long>(ctx().validation_warnings()),
                ctx().validation_layer_active() ? "on" : "off");
    EXPECT_EQ(ctx().validation_errors(), 0u);
}

} // namespace
