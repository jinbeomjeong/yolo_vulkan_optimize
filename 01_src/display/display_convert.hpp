// DisplayConvert: 디코더 SAND128 NV12 dma-buf → 표시용 RGBA storage image (02_shaders/display_convert.comp)
//
// 계획서 §5.3 안 (I): **변환을 present 시점에 하고 스왑체인 이미지에 직접 쓴다** → 링 버퍼도 copy 도 없다.
// 실행 경로는 SandLetterbox 의 템플릿(슬롯·배리어·spec constant·timestamp)을 그대로 따르되,
// 출력이 VkBuffer 가 아니라 **스왑체인이 준 VkImageView(rgba8 storage image)** 라는 점만 다르다.
// SandLetterbox 자체는 튜닝된 2.13 ms 경로라 손대지 않는다 — 이 클래스는 새로 썼다.
//
// 슬롯 규약: 슬롯 번호는 **VulkanSwapchain 이 준 SwapchainFrame::sync_index** 를 그대로 쓴다.
//   begin(slot) → [스왑체인 acquire 전환] → record(slot, ...) → [스왑체인 present 전환] →
//   end_and_submit(slot, wait, signal, fence) → (펜스 완료 후) finish(slot)
//   finish() 에서 import pin 을 풀고 DrmFrame keepalive 를 놓는다(디코더 풀로 반납).
//
// 동기화: 입력 dma-buf 는 external_qf(FOREIGN_EXT) 소유이므로 커맨드 버퍼 앞뒤에 acquire/release 배리어를
//   sand_letterbox.cpp:363-387 / :401-445 패턴 그대로 넣는다(전처리와 표시 둘 다 **읽기만** 한다).
//   출력 이미지의 레이아웃 전환은 VulkanSwapchain 이 기록한다(UNDEFINED → GENERAL → PRESENT_SRC).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "common/frame_types.hpp"
#include "vulkan/dmabuf_import.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

struct DisplayConvertConfig {
    int num_slots = 3;                 // 인플라이트 수 (= 스왑체인 이미지 개수)
    // 워크그룹 크기. x 는 **4픽셀 묶음** 단위다. x*y <= 256 (V3D maxComputeWorkGroupInvocations).
    // 기본 3x85 = 255 invocation, 즉 **12 x 85 픽셀의 세로로 긴 타일**이다.
    // 실측 근거 (yvo_display_probe --probe convert --sweep, 1920x1080 1:1, 오프스크린 200프레임, V3D 7.1.
    //   3회 인터리브 반복으로 DVFS 드리프트를 배제했다 — 재현성 ±0.02 ms):
    //   8x8(64)   15.01 | 4x16(64)  13.12 | 16x8(128) 10.07 | 8x16(128)  9.94 | 3x48(144)  7.74
    //   16x16(256) 7.74 | 8x32(256)  7.78 | 32x8(256)  7.80 | 128x2(256) 8.09 | 64x4(256)  8.05
    //   2x64(128)  7.59 | 3x64(192)  6.70 | 5x36(180)  6.63 | 1x256(256) 6.78 | 4x64(256)  6.62
    //   7x36(252)  6.14 | 6x42(252)  6.15 | 5x51(255)  5.84 | 3x72(216)  5.59 | 2x108(216) 5.28
    //   2x128(256) 6.21 | 3x84(252)  4.55 | **3x85(255) 4.23  ← 채택**
    // 두 가지가 같이 작용한다:
    //   ① invocation 이 클수록 좋다는 sand_letterbox 의 규율은 그대로다(64 inv 는 15 ms, 256 inv 는 4~8 ms).
    //   ② 같은 invocation 수 안에서는 **세로로 긴 타일이 결정적으로 빠르다** — 64x16 px 7.74 → 12x85 px 4.23,
    //      1.83배. 읽기(SAND 열은 가로 128 B)만 보면 좁은 타일이 캐시 라인을 낭비하므로 불리해야 하는데도
    //      더 빠르다 → 병목은 **타일 배치(UIF)된 RGBA 이미지 쓰기** 쪽이고 세로로 긴 영역이 이미지 타일에
    //      더 잘 맞는 것으로 본다. **이 해석은 미확정 가설이고, 수치만이 근거다.**
    // 해상도가 바뀌면 최적값도 바뀔 수 있다. spec constant 이므로 --probe convert --sweep 으로 다시 잰다.
    std::uint32_t local_size_x = 3;
    std::uint32_t local_size_y = 85;
    // 출력 크기가 원본과 다르면 true(이때만 bilinear + 열 LUT). 호스트가 판단해 넘긴다.
    bool uniform_scale = false;
    bool timestamps = true;
    std::uint8_t pad_value = 0;        // 균일 스케일에서 letterbox 여백 색 (표시에서는 검정)
    int max_dst_w = 1920;              // 열 LUT 상한. 재생성으로 창이 커져도 이 값을 넘지 않아야 한다.
    std::size_t import_cache_entries = 64;
};

// 셰이더 push constant (std430, 02_shaders/display_convert.comp 의 Params 와 같은 순서·크기)
struct DisplayConvertPush {
    std::uint32_t y_offset;
    std::uint32_t y_col_stride;
    std::uint32_t uv_offset;
    std::uint32_t uv_col_stride;
    std::int32_t src_w;
    std::int32_t src_h;
    std::int32_t dst_w;
    std::int32_t dst_h;
    std::int32_t pad_x;
    std::int32_t pad_y;
    std::int32_t new_w;
    std::int32_t new_h;
    float inv_scale;
    std::uint32_t pad_value;
};
static_assert(sizeof(DisplayConvertPush) == 56, "push constant layout must match shader");

class DisplayConvert {
public:
    DisplayConvert(yvo::vk::VulkanContext& ctx, const DisplayConvertConfig& cfg = {});
    ~DisplayConvert();
    DisplayConvert(const DisplayConvert&) = delete;
    DisplayConvert& operator=(const DisplayConvert&) = delete;

    // 커맨드 버퍼를 리셋·시작하고 query pool 을 리셋한다. 반환된 cb 에 스왑체인 배리어를 함께 기록한다.
    VkCommandBuffer begin(int slot);
    // 입력 dma-buf import(캐시) + 디스크립터 갱신 + 배리어 + 디스패치 기록.
    void record(int slot, const DrmFrame& frame, VkImageView dst_view, VkExtent2D dst_extent);
    // 커맨드 버퍼를 닫고 제출한다. wait/signal 은 스왑체인 세마포어, fence 는 스왑체인 submit 펜스다.
    void end_and_submit(int slot, VkSemaphore wait, VkSemaphore signal, VkFence fence);

    bool pending(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).pending; }
    bool ready(int slot) const;      // 펜스가 끝났는가 (대기하지 않는다)
    // 펜스를 기다린 뒤 timestamp 를 읽고 import pin·keepalive 를 놓는다.
    void finish(int slot, std::uint64_t timeout_ns = UINT64_MAX);

    // 마지막 finish() 의 GPU 시간(ms). 측정 불가면 < 0.
    double last_shader_ms() const { return last_shader_ms_; }   // 디스패치 구간만 (2.13 ms 와 같은 기준)
    double last_total_ms() const { return last_total_ms_; }     // 커맨드 버퍼 전체 (blit 폴백 비용 포함)

    int num_slots() const { return static_cast<int>(slots_.size()); }
    const DisplayConvertConfig& config() const { return cfg_; }
    void set_uniform_scale(bool on);   // 창 크기가 바뀌어 경로가 달라질 때 (파이프라인을 다시 만든다)
    void clear_import_cache();
    const yvo::vk::DmaBufImportCache& import_cache() const { return cache_; }

private:
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;   // 소유하지 않는다(스왑체인 소유). 제출 시 받아 둔다.
        bool pending = false;
        bool recording = false;
        DrmFrame frame;
        yvo::vk::ImportedDmaBuf* y = nullptr;
        yvo::vk::ImportedDmaBuf* uv = nullptr;
    };
    struct LutKey {
        int src_w = 0, src_h = 0, dst_w = 0, dst_h = 0, pad_x = 0, pad_y = 0;
        float scale = 0.0f;
        std::uint32_t y_col_stride = 0, uv_col_stride = 0;
        bool operator==(const LutKey&) const = default;
    };

    void create_pipeline();
    void destroy_pipeline();
    void create_slots();
    void create_lut();
    void update_lut(const LutKey& key);
    void destroy();

    yvo::vk::VulkanContext& ctx_;
    DisplayConvertConfig cfg_;
    yvo::vk::DmaBufImportCache cache_;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    VkBuffer lut_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory lut_mem_ = VK_NULL_HANDLE;
    std::uint32_t* lut_ptr_ = nullptr;
    VkDeviceSize lut_size_ = 0;
    LutKey lut_key_{};
    bool lut_valid_ = false;
    std::vector<Slot> slots_;
    double last_shader_ms_ = -1.0;
    double last_total_ms_ = -1.0;
};

} // namespace yvo::display
