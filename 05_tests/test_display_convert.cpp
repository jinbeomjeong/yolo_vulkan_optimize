// display_convert.comp (display conversion, 1:1 path) vs CPU reference implementation — bit-exact gate.
//
// Why this test exists (2026-09-21):
//   The BT.709 limited → RGB coefficients exist in three copies in the repository.
//     ① ref/cpu_letterbox.hpp  (CPU reference, the original formula)
//     ② 02_shaders/sand_letterbox.comp  (preprocessing)  — test_preproc enforces bit-exactness
//     ③ 02_shaders/display_convert.comp (display)    — **had no ctest** (only manual probe P8)
//   Changing ③ made no test fail. This test closes that hole.
//   The verdict logic lives only in testkit/display_convert_check.hpp, and display_probe (P8) uses the same function.
//
// Requires: V3D GPU + HEVC HW decoder. No NPU or display needed (offscreen).
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "testkit/display_convert_check.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "vulkan/vulkan_context.hpp"

#ifndef YVO_TEST_VIDEO
#error "YVO_TEST_VIDEO must be defined (05_tests/CMakeLists.txt)"
#endif

namespace {

yvo::vk::VulkanContext& ctx() {
    static std::unique_ptr<yvo::vk::VulkanContext> c = [] {
        yvo::vk::VulkanContextOptions o;
        const char* v = std::getenv("YVO_VK_VALIDATION");
        o.enable_validation = v && *v == '1';
        return std::make_unique<yvo::vk::VulkanContext>(o);
    }();
    return *c;
}

void expect_exact(int frame, const yvo::display::ConvertAccuracy& r) {
    EXPECT_EQ(r.max_err, 0) << "frame " << frame << ": first mismatching pixel (" << r.first_x << "," << r.first_y
                            << ") gpu=(" << r.gpu_rgb[0] << "," << r.gpu_rgb[1] << "," << r.gpu_rgb[2]
                            << ") cpu=(" << r.cpu_rgb[0] << "," << r.cpu_rgb[1] << "," << r.cpu_rgb[2]
                            << "), differing pixels " << r.diff_px << " / " << r.total_px;
    EXPECT_EQ(r.alpha_bad, 0) << "frame " << frame << ": there are pixels with alpha != 255";
    EXPECT_TRUE(r.ok);
    std::printf("[display_convert] frame %d: max_err %d, diff %lld / %lld px, alpha_bad %d\n", frame, r.max_err,
                r.diff_px, r.total_px, r.alpha_bad);
}

// Three frames far apart. The scenes (brightness, color) must differ for coefficient errors to show.
TEST(DisplayConvert, OneToOneMatchesCpuReference) {
    yvo::FfmpegDrmDecoder dec;
    ASSERT_TRUE(dec.open(YVO_TEST_VIDEO)) << dec.error();
    const std::vector<int> targets{0, 100, 450};
    int idx = 0;
    for (int want : targets) {
        std::optional<yvo::DrmFrame> f;
        while (idx <= want) {
            f = dec.next();
            ASSERT_TRUE(f.has_value()) << "frame " << idx << " decode failed";
            ++idx;
        }
        expect_exact(want, yvo::display::check_display_convert_1to1(ctx(), *f));
    }
}

TEST(DisplayConvert, ValidationNoErrors) {
    std::printf("[validation] errors=%llu warnings=%llu layer=%s\n",
                static_cast<unsigned long long>(ctx().validation_errors()),
                static_cast<unsigned long long>(ctx().validation_warnings()),
                ctx().validation_layer_active() ? "on" : "off");
    EXPECT_EQ(ctx().validation_errors(), 0u);
}

}  // namespace
