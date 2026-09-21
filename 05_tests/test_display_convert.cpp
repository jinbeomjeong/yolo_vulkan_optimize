// display_convert.comp(표시 변환, 1:1 경로) vs CPU 기준 구현 — 비트 일치 게이트.
//
// 왜 이 테스트가 있나 (2026-09-21):
//   BT.709 limited → RGB 계수가 저장소에 세 벌 있다.
//     ① ref/cpu_letterbox.hpp  (CPU 기준, 수식의 원문)
//     ② 02_shaders/sand_letterbox.comp  (전처리)  — test_preproc 이 비트 일치를 강제한다
//     ③ 02_shaders/display_convert.comp (표시)    — **ctest 가 없었다**(수동 프로브 P8 뿐)
//   ③ 을 고쳐도 아무 테스트도 실패하지 않는 상태였다. 이 테스트가 그 구멍을 막는다.
//   판정 로직은 testkit/display_convert_check.hpp 하나뿐이고 display_probe(P8)도 같은 함수를 쓴다.
//
// 필요한 것: V3D GPU + HEVC HW 디코더. NPU·디스플레이는 필요 없다(오프스크린).
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
#error "YVO_TEST_VIDEO 가 정의돼야 한다 (05_tests/CMakeLists.txt)"
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
    EXPECT_EQ(r.max_err, 0) << "frame " << frame << ": 처음 어긋난 픽셀 (" << r.first_x << "," << r.first_y
                            << ") gpu=(" << r.gpu_rgb[0] << "," << r.gpu_rgb[1] << "," << r.gpu_rgb[2]
                            << ") cpu=(" << r.cpu_rgb[0] << "," << r.cpu_rgb[1] << "," << r.cpu_rgb[2]
                            << "), 다른 픽셀 " << r.diff_px << " / " << r.total_px;
    EXPECT_EQ(r.alpha_bad, 0) << "frame " << frame << ": alpha != 255 인 픽셀이 있다";
    EXPECT_TRUE(r.ok);
    std::printf("[display_convert] frame %d: max_err %d, diff %lld / %lld px, alpha_bad %d\n", frame, r.max_err,
                r.diff_px, r.total_px, r.alpha_bad);
}

// 서로 떨어진 프레임 셋. 장면(밝기·색)이 달라야 계수 오류가 드러난다.
TEST(DisplayConvert, OneToOneMatchesCpuReference) {
    yvo::FfmpegDrmDecoder dec;
    ASSERT_TRUE(dec.open(YVO_TEST_VIDEO)) << dec.error();
    const std::vector<int> targets{0, 100, 450};
    int idx = 0;
    for (int want : targets) {
        std::optional<yvo::DrmFrame> f;
        while (idx <= want) {
            f = dec.next();
            ASSERT_TRUE(f.has_value()) << "프레임 " << idx << " 디코드 실패";
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
