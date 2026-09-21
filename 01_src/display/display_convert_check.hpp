#pragma once
// display_convert.comp(1:1 경로)를 CPU 기준 구현과 **비트 단위로** 대조한다.
//
// 쓰는 곳: 05_tests/test_display_convert.cpp (ctest 게이트), display/tools/display_probe.cpp (P8).
//
// 왜 필요한가 (2026-09-21):
//   BT.709 limited → RGB 계수가 세 벌 있다 — ref/cpu_letterbox.hpp(CPU 기준),
//   sand_letterbox.comp(전처리), display_convert.comp(표시). 앞의 두 벌은 test_preproc 이
//   비트 일치를 강제하는데 **표시 쪽 세 번째 벌은 ctest 가 없고 수동 프로브뿐**이었다.
//   "CPU 기준 구현 + 비트 일치 테스트"가 이 프로젝트의 핵심 검증 장치인데 여기만 빠져 있었다.
//   판정 로직을 여기 하나에 두고 프로브와 테스트가 같이 쓴다(판정이 두 벌로 갈라지지 않게).
//
// CPU 기준: dma-buf mmap → de-SAND → cpu_letterbox_nv12_bt709(scale = 1, 패딩 없음).
// 합격 조건: RGB 최대 오차 0 이고 alpha 가 전부 255.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "ref/cpu_letterbox.hpp"
#include "core/dmabuf_map.hpp"
#include "core/frame_types.hpp"
#include "ref/sand.hpp"
#include "display/display_convert.hpp"
#include "display/offscreen_target.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::display {

struct ConvertAccuracy {
    bool ok = false;
    int max_err = -1;
    long long diff_px = 0;
    long long total_px = 0;
    int alpha_bad = 0;
    int first_x = -1, first_y = -1;       // 처음 어긋난 픽셀
    int gpu_rgb[3]{}, cpu_rgb[3]{};
};

// f 한 장을 display_convert 1:1 경로로 변환해 CPU 기준과 대조한다.
// lx, ly: 워크그룹 크기(0 이면 DisplayConvertConfig 기본값).
inline ConvertAccuracy check_display_convert_1to1(yvo::vk::VulkanContext& ctx, const yvo::DrmFrame& f,
                                                  std::uint32_t lx = 0, std::uint32_t ly = 0) {
    ConvertAccuracy r;
    const int w = f.width, h = f.height;

    DisplayConvertConfig cc;
    cc.num_slots = 1;
    if (lx) cc.local_size_x = lx;
    if (ly) cc.local_size_y = ly;
    cc.uniform_scale = false;  // 1:1 경로를 검증한다
    cc.max_dst_w = std::max(w, 1920);
    DisplayConvert conv(ctx, cc);
    OffscreenTarget t(ctx, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), true);

    YVO_VK_CHECK(vkResetFences(ctx.device(), 1, &t.fence));
    VkCommandBuffer cb = conv.begin(0);
    barrier_image(cb, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                  VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    conv.record(0, f, t.view, {t.w, t.h});
    t.record_readback(cb);
    conv.end_and_submit(0, VK_NULL_HANDLE, VK_NULL_HANDLE, t.fence);
    conv.finish(0);

    std::vector<std::uint8_t> y(static_cast<std::size_t>(w) * h), uv(static_cast<std::size_t>(w) * h / 2);
    {
        yvo::DmaBufReadMap my(f.fd, f.size), muv(f.uv_fd, f.uv_size);
        if (!my.data() || !muv.data() ||
            !yvo::desand_nv12(my.data(), f.size, muv.data(), f.uv_size, yvo::make_sand_layout(f), y.data(),
                              static_cast<std::size_t>(w), uv.data(), static_cast<std::size_t>(w)))
            throw std::runtime_error("display_convert check: de-SAND 실패");
    }
    yvo::LetterboxParams lb;
    lb.dst_w = w;
    lb.dst_h = h;
    lb.scale = 1.0f;
    lb.pad_x = 0;
    lb.pad_y = 0;
    std::vector<std::uint8_t> ref(static_cast<std::size_t>(w) * h * 3);
    yvo::cpu_letterbox_nv12_bt709(y.data(), static_cast<std::size_t>(w), uv.data(), static_cast<std::size_t>(w), w, h,
                                  lb, ref.data());

    r.total_px = static_cast<long long>(w) * h;
    r.max_err = 0;
    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            const std::uint8_t* g = t.host + (static_cast<std::size_t>(py) * w + px) * 4;
            const std::uint8_t* c = ref.data() + (static_cast<std::size_t>(py) * w + px) * 3;
            if (g[3] != 255) ++r.alpha_bad;
            int e = 0;
            for (int k = 0; k < 3; ++k) e = std::max(e, std::abs(static_cast<int>(g[k]) - static_cast<int>(c[k])));
            if (e > 0) {
                ++r.diff_px;
                if (r.first_x < 0) {
                    r.first_x = px;
                    r.first_y = py;
                    for (int k = 0; k < 3; ++k) {
                        r.gpu_rgb[k] = g[k];
                        r.cpu_rgb[k] = c[k];
                    }
                }
                r.max_err = std::max(r.max_err, e);
            }
        }
    }
    r.ok = r.max_err == 0 && r.alpha_bad == 0;
    return r;
}

}  // namespace yvo::display
