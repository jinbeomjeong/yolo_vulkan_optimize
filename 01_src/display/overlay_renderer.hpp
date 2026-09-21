// OverlayRenderer: 검출 박스·라벨·HUD 를 표시용 RGBA storage image 에 그리는 컴퓨트 패스
//                  (02_shaders/overlay_boxes.comp). 계획서 §5.4.
//
// ── 이 클래스가 하지 않는 것 ──────────────────────────────────────────────────────────────────
// **커맨드 버퍼를 만들지 않고 제출하지 않는다.** `DisplayConvert` 가 만든 **같은 커맨드 버퍼에 이어
// 기록**한다(표시 트랙 2단계 보고의 권고). 따라서 present 당 submit 은 여전히 1회다.
//   DisplayConvert::begin(slot) → [스왑체인 acquire 전환] → DisplayConvert::record(...)
//     → OverlayRenderer::record(slot, cb, ...)        ← 여기
//     → [스왑체인 present 전환] → DisplayConvert::end_and_submit(...) → finish(slot)
//     → OverlayRenderer::collect(slot)                 ← 펜스 완료 후 timestamp 읽기
// 슬롯 번호는 `DisplayConvert` 와 같은 것(= `SwapchainFrame::sync_index`)을 쓴다.
//
// ── 검출 업로드: staging 도 transfer 큐도 없다 ────────────────────────────────────────────────
// 검출 80개 x 32 B + 헤더 152 B ≈ 2.7 KB 이고 CPU→GPU **쓰기**는 5 µs 다(계획서 §5.4 —
// REPORT.md §4.3 이 경고한 10배 페널티는 CPU 가 **읽는** 방향이다). 슬롯별 DetBuf 를
// `HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL` 로 잡아 **영구 매핑**하고 CPU 가 직접 쓴 뒤
// `HOST_WRITE → SHADER_READ | INDIRECT_COMMAND_READ` 배리어 하나만 넣는다.
//
// ── 두 디스패치 사이 배리어 (필요한가? → 필요하다) ────────────────────────────────────────────
// `display_convert.comp` 와 `overlay_boxes.comp` 는 **같은 이미지의 같은 픽셀에 쓴다**(WAW).
// 같은 커맨드 버퍼의 연속한 두 디스패치는 배리어가 없으면 겹쳐 실행될 수 있으므로
// 오버레이가 변환보다 먼저 그려질 수 있다 → `SHADER_STORAGE_WRITE → SHADER_STORAGE_WRITE`
// 이미지 메모리 배리어(레이아웃은 GENERAL 유지)를 넣는다. 오버레이는 이미지를 **읽지 않으므로**
// dstAccessMask 에 READ 는 넣지 않는다(알파 블렌딩을 하지 않는다 — R7 이 알파를 항상 255 로 쓴다).
// 이 배리어는 `record(..., barrier_prior_compute_write = true)` 가 기록한다. 오버레이를 단독으로
// (앞선 컴퓨트 쓰기 없이) 돌리는 테스트·프로브는 false 를 준다.
//
// ── 디스크립터·디스패치 ───────────────────────────────────────────────────────────────────────
// set 0 = {0: 스왑체인 storage image(rgba8), 1: 슬롯별 DetBuf, 2: 글리프+이름 SSBO}
//   → V3D 한도(storage buffers 8, storage images 4, descriptorIndexing=false) 안에 넉넉하다.
// **검출 하나당 워크그룹 하나**이고 `vkCmdDispatchIndirect(cb, det_buf, 0)` 가 DetBuf 맨 앞의
// uvec3 를 읽는다. CPU 가 채웠을 때도, 나중에 GPU NMS 셰이더가 채웠을 때도 똑같이 동작한다.
//
// ── 경계 규약 ─────────────────────────────────────────────────────────────────────────────────
// `overlay_reference.hpp` 상단 R0~R8 이 유일한 기준이다. 셰이더와 CPU 기준 구현이 같은 규약을 따르고
// `05_tests/test_overlay.cpp` 가 **max 오차 0** 으로 판정한다. 알려진 허용 한계:
// **겹치는 박스의 교차 픽셀 색은 순서 의존적이다**(워크그룹 실행 순서가 정의되지 않음).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "common/frame_types.hpp"
#include "display/overlay_reference.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

struct OverlayConfig {
    int num_slots = 3;            // 인플라이트 수 (= 스왑체인 이미지 개수, DisplayConvert 와 같게)
    int max_dets = 80;            // DetBuf 의 dets[] 상한 (계획서 §5.4)
    // 워크그룹 크기(spec constant). 검출 하나당 워크그룹 하나이므로 x 만 쓴다.
    // 실측 근거 (yvo_display_probe --probe overlay --overlay-local N, 1920x1080 오프스크린 연속 200프레임,
    //   검출 80개 + 라벨(text_scale 2), median ms — 최종 셰이더 기준):
    //   16: 7.27 | 32: 3.96 | 64: 2.09 | **128: 1.03 ← 채택** | 256: 1.03
    // "invocation 이 클수록 좋다"는 sand_letterbox 의 규율이 여기서도 128 까지는 그대로 통한다.
    // 256 은 더 나아지지 않는다(검출당 작업량이 작아 워크그룹이 다 차지 않는다).
    std::uint32_t local_size_x = 128;
    bool draw_text = true;        // spec constant kDrawText
    int text_scale = 2;           // spec constant kTextScale. 1080p 기본 2 (8x16 → 16x32)
    int line_width = 2;           // push constant. 1080p 기본 2
    bool timestamps = true;
};

// 셰이더 push constant (std430, overlay_boxes.comp 의 Params 와 같은 순서·크기)
struct OverlayPush {
    float sx, sy, tx, ty;
    std::int32_t dst_w, dst_h;
    std::int32_t line_width;
    std::uint32_t max_dets;
};
static_assert(sizeof(OverlayPush) == 32, "push constant layout must match shader");

// DetBuf 헤더 (std430). 셰이더의 DetBuf 앞부분과 바이트 단위로 같아야 한다.
//   dispatch 0..11 / count 12 / overflow 16 / text_bytes 20 / text[32] 24..151 / dets[] 152..
struct OverlayDetBufHeader {
    std::uint32_t dispatch[3];    // VkDispatchIndirectCommand 호환 (offset 0)
    std::uint32_t count;
    std::uint32_t overflow;
    std::uint32_t text_bytes;
    std::uint32_t text[32];       // HUD ASCII
};
static_assert(sizeof(OverlayDetBufHeader) == 152, "DetBuf header layout must match shader (std430)");
static_assert(offsetof(OverlayDetBufHeader, count) == 12, "dispatch must be uvec3 at offset 0");
static_assert(offsetof(OverlayDetBufHeader, overflow) == 16, "overflow offset must match shader");
static_assert(offsetof(OverlayDetBufHeader, text_bytes) == 20, "text_bytes offset must match shader");
static_assert(offsetof(OverlayDetBufHeader, text) == 24, "text offset must match shader");

// 32 B. 후처리 트랙(계획서 §4.3)의 Rec 과 같은 레이아웃이다.
struct DetGpu {
    float x1, y1, x2, y2;
    float score;
    std::uint32_t class_id;
    std::uint32_t anchor_key;
    std::uint32_t keep;
};
static_assert(sizeof(DetGpu) == 32, "DetGpu must be 32 B (postproc 트랙과 공유하는 계약)");
static_assert(offsetof(DetGpu, score) == 16, "DetGpu.score offset must match shader");
static_assert(offsetof(DetGpu, class_id) == 20, "DetGpu.class_id offset must match shader");
static_assert(offsetof(DetGpu, anchor_key) == 24, "DetGpu.anchor_key offset must match shader");
static_assert(offsetof(DetGpu, keep) == 28, "DetGpu.keep offset must match shader");

class OverlayRenderer {
public:
    OverlayRenderer(yvo::vk::VulkanContext& ctx, const OverlayConfig& cfg = {});
    ~OverlayRenderer();
    OverlayRenderer(const OverlayRenderer&) = delete;
    OverlayRenderer& operator=(const OverlayRenderer&) = delete;

    // 슬롯의 DetBuf 를 호스트에서 직접 채운다(staging 없음). record() 보다 먼저 부른다.
    // dets 가 max_dets 를 넘으면 앞에서부터 max_dets 개만 쓰고 overflow 를 1 로 세운다.
    // hud 는 128 B 로 잘린다. 반환값 = 실제로 기록한 검출 수.
    int set_frame(int slot, const std::vector<Detection>& dets, std::string_view hud = {});

    // 같은 커맨드 버퍼에 이어 기록한다. affine 은 make_overlay_affine(src, dst) 결과다.
    void record(int slot, VkCommandBuffer cb, VkImageView dst_view, VkExtent2D dst_extent,
                const OverlayAffine& affine, bool barrier_prior_compute_write = true);

    // 제출 펜스가 끝난 뒤 호출 — timestamp 를 읽는다(대기하지 않는다).
    void collect(int slot);
    // 마지막 collect() 의 오버레이 디스패치 GPU 시간(ms). 측정 불가면 < 0.
    double last_shader_ms() const { return last_shader_ms_; }

    std::uint32_t overflow(int slot) const;     // 이 슬롯에서 검출이 상한을 넘었는가
    int num_slots() const { return static_cast<int>(slots_.size()); }
    int max_dets() const { return cfg_.max_dets; }
    const OverlayConfig& config() const { return cfg_; }
    // CPU 기준 구현에 넘길 스타일 (GPU spec constant/push constant 와 같은 값)
    OverlayStyle style() const;

private:
    struct Slot {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        std::uint8_t* host = nullptr;   // 영구 매핑
        bool recorded = false;
    };

    void create_pipeline();
    void create_font_buffer();
    void create_slots();
    void destroy();
    OverlayDetBufHeader* header(int slot);
    DetGpu* dets(int slot);

    yvo::vk::VulkanContext& ctx_;
    OverlayConfig cfg_;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    VkBuffer font_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory font_mem_ = VK_NULL_HANDLE;
    VkDeviceSize font_size_ = 0;
    VkDeviceSize det_size_ = 0;
    std::vector<Slot> slots_;
    double last_shader_ms_ = -1.0;
};

} // namespace yvo::display
