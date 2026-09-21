// SAND128 NV12 dma-buf → 640x640 packed RGB888 letterbox (Vulkan 컴퓨트, 1단계: host-visible 출력)
//
// 셰이더(02_shaders/sand_letterbox.comp)가 한 번의 디스패치로 de-SAND + bilinear 축소 + BT.709 limited → RGB
// + 패딩 114를 처리한다. 수식은 ref/cpu_letterbox.hpp(cpu_letterbox_nv12_bt709)와 같다.
//
// 슬롯 규약 (HailoRunner와 같은 슬롯 번호를 쓴다):
//   submit(frame, slot) → [GPU 실행] → wait(slot)이 PreprocOutput을 돌려준다.
//   - submit은 해당 슬롯이 대기 중(pending)이면 예외. wait 이후 슬롯 출력 버퍼는 호출자(HailoRT) 소유이며,
//     호출자가 다 쓴 뒤(HailoRT 완료 콜백 이후) 같은 슬롯을 다시 submit한다.
//   - DrmFrame.keepalive는 submit에서 복사해 두었다가 wait(펜스 완료 확인) 시점에 놓는다.
//     디코더는 동시에 약 12~13프레임 이상 들고 있으면 멈추므로, pending 슬롯 수를 작게(2~4) 유지할 것.
//   - 스레드 안전하지 않다. 한 스레드에서 submit/wait를 호출한다.
// 동기화:
//   - 입력 dma-buf: 디코더가 프레임을 반환한 시점에 디코드 완료로 본다(V4L2 dequeue). 커맨드 버퍼 앞에서
//     외부 큐 패밀리(FOREIGN_EXT)→컴퓨트 큐 acquire 배리어, 뒤에서 release 배리어로 소유권을 돌려준다.
//   - 출력: 셰이더 쓰기 → HOST_READ 배리어 후 펜스. wait()가 돌아오면 host_ptr을 바로 읽을 수 있다(HOST_COHERENT).
//   - 열 LUT: 입력 크기·SAND 열 간격·letterbox가 바뀔 때만 호스트가 다시 쓴다(그 전에 대기 중인 작업을 기다림).
// 제약: y_offset, uv_offset은 4의 배수여야 한다(행 주소를 32bit 워드 단위로 계산).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "vulkan/dmabuf_import.hpp"

namespace yvo::vk {

class VulkanContext;

struct SandLetterboxConfig {
    int num_slots = 8;               // 출력 링 슬롯 수 (HailoRunnerConfig::num_slots와 같게)
    int dst_w = 640;                 // 4의 배수
    int dst_h = 640;
    // 워크그룹 크기 (x: 4픽셀 묶음, y: 행). x*y <= maxComputeWorkGroupInvocations(V3D 256).
    // V3D 7.1 측정(yvo_preproc_bench --sweep, 1080p→640): 64 inv 3.65 ms, 128 inv 2.45 ms, 256 inv 2.13 ms.
    std::uint32_t local_size_x = 32;
    std::uint32_t local_size_y = 8;
    // 출력 텐서 레이아웃. Nhcw면 HailoRT의 CPU 입력 변환(약 14 ms/frame)이 필요 없다(frame_types.hpp 참고).
    TensorLayout out_layout = TensorLayout::Nhwc;
    bool export_dmabuf = false;      // 2단계 준비: 출력 메모리를 dma-buf로 export 가능하게 할당
    bool timestamps = true;          // 셰이더 구간 timestamp 측정
    std::size_t import_cache_entries = 64;
};

// 셰이더 push constant (std430, 02_shaders/sand_letterbox.comp의 Params와 같은 순서·크기)
struct SandLetterboxPush {
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
static_assert(sizeof(SandLetterboxPush) == 56, "push constant layout must match shader");

class SandLetterbox {
public:
    SandLetterbox(VulkanContext& ctx, const SandLetterboxConfig& cfg = {});
    ~SandLetterbox();
    SandLetterbox(const SandLetterbox&) = delete;
    SandLetterbox& operator=(const SandLetterbox&) = delete;

    // 비동기: 커맨드 기록 + vkQueueSubmit. slot은 [0, num_slots).
    void submit(const DrmFrame& frame, int slot);
    // slot의 펜스를 기다리고 결과를 돌려준다. keepalive(AVFrame)를 여기서 놓는다.
    PreprocOutput wait(int slot, std::uint64_t timeout_ns = UINT64_MAX);
    // 펜스가 이미 끝났는지 (대기 안 함). pending이 아니면 true.
    bool ready(int slot) const;
    bool pending(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).pending; }
    // 동기 편의 함수: submit + wait
    PreprocOutput process(const DrmFrame& frame, int slot);

    int num_slots() const { return static_cast<int>(slots_.size()); }
    std::size_t output_size() const { return out_size_; }
    std::uint8_t* host_ptr(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).mapped; }
    // 2단계용: 출력 메모리를 dma-buf fd로 export (config.export_dmabuf 필요). 새 fd를 반환하며 호출자가 close.
    int export_dmabuf_fd(int slot) const;

    // 마지막 wait된 slot의 셰이더 구간 GPU 시간(ms). 측정 불가면 < 0.
    double last_gpu_ms() const { return last_gpu_ms_; }
    const DmaBufImportCache& import_cache() const { return cache_; }
    // 디코더를 닫거나 다시 열 때 호출 (대기 중인 슬롯이 없어야 한다)
    void clear_import_cache();
    const SandLetterboxConfig& config() const { return cfg_; }

private:
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkBuffer out_buf = VK_NULL_HANDLE;
        VkDeviceMemory out_mem = VK_NULL_HANDLE;
        std::uint8_t* mapped = nullptr;
        bool pending = false;
        DrmFrame frame;                    // keepalive 보관
        ImportedDmaBuf* y = nullptr;       // pin된 import 항목
        ImportedDmaBuf* uv = nullptr;
        LetterboxParams lb;
    };

    struct LutKey {
        int src_w = 0, src_h = 0, dst_w = 0, dst_h = 0, pad_x = 0, pad_y = 0;
        float scale = 0.0f;
        std::uint32_t y_col_stride = 0, uv_col_stride = 0;
        bool operator==(const LutKey&) const = default;
    };

    void create_pipeline();
    void create_slots();
    void create_lut();
    // 열 LUT(셰이더 binding 3)를 key에 맞게 갱신. 바뀌었으면 대기 중인 GPU 작업이 끝난 뒤 다시 쓴다.
    void update_lut(const LutKey& key);
    void destroy();
    void record(Slot& s, int slot, const ImportedDmaBuf& y, const ImportedDmaBuf& uv, const SandLetterboxPush& pc);

    VulkanContext& ctx_;
    SandLetterboxConfig cfg_;
    DmaBufImportCache cache_;
    std::size_t out_size_ = 0;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    VkBuffer lut_buf_ = VK_NULL_HANDLE;       // 열 LUT: dst_w × 2 × uvec4 (HOST_VISIBLE|COHERENT)
    VkDeviceMemory lut_mem_ = VK_NULL_HANDLE;
    std::uint32_t* lut_ptr_ = nullptr;
    VkDeviceSize lut_size_ = 0;
    LutKey lut_key_{};
    bool lut_valid_ = false;
    std::vector<Slot> slots_;
    double last_gpu_ms_ = -1.0;
};

} // namespace yvo::vk
