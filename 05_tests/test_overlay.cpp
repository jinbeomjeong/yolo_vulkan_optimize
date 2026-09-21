// GPU 오버레이(overlay_boxes.comp) vs CPU 기준(overlay_reference.hpp) — **비트 일치** 판정
//
// 계획서 §7 의 표를 따른다. NPU·디스플레이 **불필요**(오프스크린 R8G8B8A8_UNORM 이미지)라
// ctest 에서 그대로 돌아간다.
//
//   OverlayGeometry     1920x1080(실제 경로와 같은 해상도), 텍스트 끔, 선 두께 1/2/3.
//                       경계 접촉·면적 0·일부/전부 화면 밖·class 0/79·검출 80개. **max 오차 0**
//   OverlayText         라벨 + HUD. text_scale 1 과 2. **max 오차 0**
//                       (OpenCV 를 오라클로 쓰지 않는다 — 같은 kGlyph8x16 표를 쓰는 CPU 기준이 오라클이다)
//   OverlayDeterminism  비겹침 입력 10회 반복 **완전 동일**
//   OverlayVsOpenCV     (게이트 아님) draw_and_save() 와 640x640 에서 느슨한 기하 비교.
//                       FONT_HERSHEY_SIMPLEX 는 다른 메트릭의 벡터 폰트라 비트 일치는 원리적으로 불가능하다.
//
// 겹치는 박스의 교차 픽셀은 **순서 의존**이다(워크그룹 실행 순서가 정의되지 않음. 계획서 §7, §9).
// 따라서 모든 테스트 입력은 **서로 겹치지 않게** 만들고, HUD 영역(4,4)~(4+8*s*len, 4+16*s)도 비워 둔다
// (GPU 는 워크그룹 0 이 HUD 와 검출 0 을 함께 그리므로 HUD 와 겹치는 박스도 순서 의존이다).
//
// 실패하면 build/05_tests/overlay_diff/ 에 GPU·CPU·차이 이미지를 저장한다(test_preproc 의 관례).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifdef YVO_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "ref/overlay_reference.hpp"
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

// 프로세스 전체에서 VulkanContext 하나만 쓴다(장치 생성이 가장 비싸다).
yvo::vk::VulkanContext& ctx() {
    static std::unique_ptr<yvo::vk::VulkanContext> c = [] {
        yvo::vk::VulkanContextOptions o;
        const char* v = std::getenv("YVO_VK_VALIDATION");
        o.enable_validation = v && *v == '1';
        return std::make_unique<yvo::vk::VulkanContext>(o);
    }();
    return *c;
}

// ── 오프스크린 렌더 하네스 ────────────────────────────────────────────────────────────────────
// STORAGE|TRANSFER_SRC|TRANSFER_DST 이미지 + 리드백 버퍼. 배경을 vkCmdClearColorImage 로 정확히
// 채운 뒤 오버레이 디스패치를 기록한다(스왑체인 경로와 같은 포맷·같은 GENERAL 레이아웃).
// 이미지·되읽기 버퍼·배리어는 testkit/offscreen_target.hpp 의 OffscreenTarget 을 쓴다
// (2026-09-21 이전에는 이 하네스가 display_probe 의 OffscreenTarget 과 같은 코드를 따로 갖고 있었다).
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

    // 배경을 kBg 로 채운 뒤 오버레이를 그리고 RGBA 를 돌려준다.
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

        // 앞선 컴퓨트 쓰기가 없으므로(클리어 배리어가 이미 있다) barrier_prior_compute_write = false
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
    long long changed_gpu = 0;   // 배경과 달라진 GPU 픽셀 수
    long long changed_cpu = 0;
    int first_x = -1, first_y = -1;
    int gpu[4]{}, cpu[4]{};
};

Diff compare(const std::vector<std::uint8_t>& g, const std::vector<std::uint8_t>& c, int w, int h) {
    Diff d;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                                   static_cast<std::size_t>(x)) * 4;
            bool cg = false, cc = false, differ = false;
            int e = 0;
            for (int k = 0; k < 4; ++k) {
                if (g[o + static_cast<std::size_t>(k)] != kBg[static_cast<std::size_t>(k)]) cg = true;
                if (c[o + static_cast<std::size_t>(k)] != kBg[static_cast<std::size_t>(k)]) cc = true;
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

// 실패 시 저장 (test_preproc 의 preproc_diff/ 관례). OpenCV 가 없으면 PPM 으로 떨어진다.
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

// 경계 케이스 모음. **서로 겹치지 않는다**(겹치면 교차 픽셀이 순서 의존이 된다).
std::vector<Detection> boundary_cases() {
    return {
        det(0, 0.90f, 0, 0, 100, 100),                 // 좌상 화면 경계에 접함
        det(79, 0.12f, 1819, 979, 1919, 1079),         // 우하 경계에 접함 (마지막 픽셀 포함)
        det(1, 0.50f, 0, 300, 40, 400),                // 왼쪽 경계
        det(2, 1.00f, 1879, 300, 1919, 400),           // 오른쪽 경계, score 1.00
        det(3, 0.00f, 300, 0, 400, 30),                // 위쪽 경계, score 0.00 (라벨이 위에 못 들어감)
        det(4, 0.33f, 300, 1049, 400, 1079),           // 아래쪽 경계
        det(5, 0.44f, 500, 200, 500, 200),             // 면적 0 (1 px 점)
        det(6, 0.55f, 600, 200, 600, 300),             // 폭 0 (세로 1 px 선)
        det(7, 0.66f, 700, 200, 800, 200),             // 높이 0 (가로 1 px 선)
        det(8, 0.77f, -50, 500, 50, 600),              // 일부 화면 밖 (왼쪽)
        det(9, 0.88f, 1890, 700, 2000, 800),           // 일부 화면 밖 (오른쪽)
        det(10, 0.99f, 900, -40, 1000, 60),            // 일부 화면 밖 (위)
        det(11, 0.20f, 900, 1040, 1000, 1140),         // 일부 화면 밖 (아래)
        det(12, 0.30f, -500, -500, -400, -400),        // 전부 화면 밖
        det(13, 0.40f, 2000, 1200, 2100, 1300),        // 전부 화면 밖
        det(14, 0.51f, 1100.5f, 200.5f, 1200.49f, 300.51f), // R1 반올림 경계 (.5 는 올림)
        det(15, 0.49f, 1300.4f, 200.6f, 1400.6f, 300.4f),   // R1 반올림
        det(16, 0.61f, 1500, 200, 1400, 300),          // x1 > x2 (뒤집힌 입력 → min/max 정규화)
        det(17, 0.71f, 600, 500, 700, 450),            // y1 > y2
    };
}

// 겹치지 않는 격자 n개 (검출 80개 상한 시험용). **테두리끼리 겹치지 않는다.**
// 라벨(텍스트)은 이웃 열의 라벨과 가로로 겹칠 수 있으므로 이 입력은 draw_text = false 로만 쓴다.
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

// 텍스트까지 켜도 **아무것도 겹치지 않는** 입력 (HUD 영역 y < 36 도 피한다).
// 라벨은 박스 위 16*s(최대 32) px 에 그려지므로 y1 >= 200 으로 여유를 둔다.
std::vector<Detection> separated_text_cases() {
    return {
        det(0, 0.87f, 100, 200, 400, 500),      // person 0.87
        det(9, 0.05f, 600, 200, 900, 500),      // traffic light 0.05 (가장 긴 이름)
        det(79, 1.00f, 1100, 200, 1400, 500),   // toothbrush 1.00
        det(2, 0.50f, 100, 700, 400, 1000),     // car 0.50
        det(5, 0.995f, 600, 700, 900, 1000),    // bus 1.00 (반올림)
        det(3, 0.004f, 1100, 700, 1400, 1000),  // motorcycle 0.00 (반올림)
        det(7, 0.62f, 1500, 20, 1700, 120),     // 위쪽 공간 부족 → 라벨이 박스 안쪽으로
        det(1, 0.42f, 1750, 700, 1919, 800),    // 오른쪽으로 라벨이 넘침 → 화면 클리핑
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
                     "[%s] max_err=%d diff_px=%lld  첫 불일치 (%d,%d) GPU(%d,%d,%d,%d) CPU(%d,%d,%d,%d)\n",
                     tag.c_str(), r.diff.max_err, r.diff.diff_px, r.diff.first_x, r.diff.first_y, r.diff.gpu[0],
                     r.diff.gpu[1], r.diff.gpu[2], r.diff.gpu[3], r.diff.cpu[0], r.diff.cpu[1], r.diff.cpu[2],
                     r.diff.cpu[3]);
    }
    EXPECT_EQ(r.diff.max_err, 0) << tag << ": GPU 와 CPU 기준이 비트 일치해야 한다";
    EXPECT_EQ(r.diff.diff_px, 0) << tag;
    EXPECT_GT(r.diff.changed_gpu, 0) << tag << ": 아무것도 그리지 않았다";
    EXPECT_EQ(r.diff.changed_gpu, r.diff.changed_cpu) << tag;
}

// ─────────────────────────── OverlayGeometry (1920x1080, 텍스트 끔) ───────────────────────────

class OverlayGeometry : public ::testing::TestWithParam<int> {}; // 선 두께 1/2/3

TEST_P(OverlayGeometry, BoundaryCasesMatchCpu) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = false;
    cfg.line_width = GetParam();
    const RunResult r = run_case(hz, cfg, boundary_cases(), "");
    expect_exact("geometry_boundary_lw" + std::to_string(GetParam()), r, 1920, 1080);
    std::printf("[geometry lw=%d] 변경 픽셀 %lld (GPU) / %lld (CPU)\n", GetParam(), r.diff.changed_gpu,
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

// 상한 초과: 81개를 주면 80개만 그리고 overflow 가 선다
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

// ─────────────────────────── OverlayText (라벨 + HUD) ───────────────────────────

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
    std::printf("[text s=%d] 변경 픽셀 %lld\n", GetParam(), r.diff.changed_gpu);
}

// 검출 0개 + HUD 만: dispatch.x = max(count,1) = 1 이라 HUD 가 나와야 한다
TEST_P(OverlayText, HudOnlyWithNoDetections) {
    OverlayHarness hz(640, 480);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = true;
    cfg.text_scale = GetParam();
    const RunResult r = run_case(hz, cfg, {}, "no detections");
    expect_exact("hud_only_s" + std::to_string(GetParam()), r, 640, 480);
}

// ASCII 32~126 전 범위 + 표현 불가 문자(규약 1: 공백으로 접는다)
TEST_P(OverlayText, FullAsciiRangeInHud) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = true;
    cfg.text_scale = GetParam();
    std::string hud;
    for (int c = 32; c <= 126; ++c) hud.push_back(static_cast<char>(c));
    hud.push_back(static_cast<char>(9));   // tab: 범위 밖 → 공백
    hud.push_back(static_cast<char>(200)); // 비 ASCII → 공백
    const RunResult r = run_case(hz, cfg, {}, hud);
    expect_exact("ascii_s" + std::to_string(GetParam()), r, 1920, 1080);
}

INSTANTIATE_TEST_SUITE_P(TextScale, OverlayText, ::testing::Values(1, 2),
                         [](const ::testing::TestParamInfo<int>& i) { return "s" + std::to_string(i.param); });

// ─────────────────────────── affine: 캔버스가 원본보다 작을 때 ───────────────────────────

TEST(OverlayAffineScale, UniformScaleMatchesCpu) {
    OverlayHarness hz(1280, 720);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = true;
    cfg.text_scale = 1;
    cfg.line_width = 2;
    OverlayRenderer ov(ctx(), cfg);
    // 원본 1920x1080 좌표 → 1280x720 캔버스 (균일 스케일 2/3, 패딩 0)
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

// ─────────────────────────── OverlayDeterminism (비겹침 10회) ───────────────────────────

// 입력은 **비겹침**으로 한정한다(계획서 §7). 겹치는 박스의 교차 픽셀은 워크그룹 실행 순서에 의존하므로
// 결정성을 요구할 수 없다 — 허용 한계로 문서화된 사항이다(overlay_reference.hpp R8).
void determinism_case(const std::vector<Detection>& dets, const std::string& hud, bool draw_text,
                      const char* tag) {
    OverlayHarness hz(1920, 1080);
    OverlayConfig cfg;
    cfg.num_slots = 2; // 슬롯을 번갈아 써도 같은 결과여야 한다
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
            ASSERT_EQ(out, first) << tag << ": 반복 " << i << " 에서 결과가 달라졌다";
    }
    std::vector<std::uint8_t> cpu = background(1920, 1080);
    overlay_cpu_render(cpu.data(), 1920, 1080, dets, hud, a, ov.style(), 80);
    const Diff d = compare(first, cpu, 1920, 1080);
    if (d.max_err) save_diff(std::string("determinism_") + tag, first, cpu, 1920, 1080);
    EXPECT_EQ(d.max_err, 0) << tag;
    EXPECT_EQ(d.diff_px, 0) << tag;
    EXPECT_GT(d.changed_gpu, 0) << tag;
}

// 테두리만: 격자 40개(테두리 비겹침)
TEST(OverlayDeterminism, TenRepeatsIdenticalBoxes) {
    determinism_case(grid_cases(40, 1920, 1080), "", false, "boxes40");
}

// 라벨 + HUD 까지: 아무것도 겹치지 않는 8개
TEST(OverlayDeterminism, TenRepeatsIdenticalWithText) {
    determinism_case(separated_text_cases(), "deterministic", true, "text8");
}

// ─────────────────────────── OverlayVsOpenCV (참고용, 게이트 아님) ───────────────────────────
#ifdef YVO_HAVE_OPENCV
// draw_and_save()(main.cpp:220-240)와 같은 방식으로 OpenCV 가 그린 640x640 캔버스와 비교한다.
// **비트 일치는 원리적으로 불가능하다**: FONT_HERSHEY_SIMPLEX 는 다른 메트릭의 벡터 폰트이고
// cv::rectangle 의 픽셀 중심 규약이 다르다. 계획서 §7 의 느슨한 기하 기준만 본다.
TEST(OverlayVsOpenCV, GeometryWithinOnePixel) {
    const int W = 640, H = 640;
    OverlayHarness hz(W, H);
    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.draw_text = false; // 폰트가 다르므로 테두리만 비교한다
    cfg.line_width = 1;    // draw_and_save() 와 같은 두께
    OverlayRenderer ov(ctx(), cfg);
    const OverlayAffine a{}; // 이미 640 캔버스 좌표로 준다
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

    // ① 변경 마스크 IoU ≥ 0.6  ② 어떤 박스에서도 20 px 넘게 떨어진 변경 픽셀 없음
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

    // ③ 각 박스의 4변 ±1 px 안에 비배경 픽셀이 있는가
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
        EXPECT_TRUE(top && bot && left && right) << "class " << d.class_id << " 의 4변이 ±1 px 안에 없다";
    }
}
#endif

// 검증 레이어 카운터 (켜져 있을 때만 의미가 있다)
TEST(OverlayValidation, NoErrors) {
    std::printf("[validation] errors=%llu warnings=%llu layer=%s\n",
                static_cast<unsigned long long>(ctx().validation_errors()),
                static_cast<unsigned long long>(ctx().validation_warnings()),
                ctx().validation_layer_active() ? "on" : "off");
    EXPECT_EQ(ctx().validation_errors(), 0u);
}

} // namespace
