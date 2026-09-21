// SAND128 NV12 dma-buf → 640x640 packed RGB888 letterbox (Vulkan compute, stage 1: host-visible output)
//
// The shader (02_shaders/sand_letterbox.comp) handles de-SAND + bilinear downscale + BT.709 limited → RGB
// + padding 114 in a single dispatch. The formulas match ref/cpu_letterbox.hpp (cpu_letterbox_nv12_bt709).
//
// Slot convention (uses the same slot numbers as HailoRunner):
//   submit(frame, slot) → [GPU execution] → wait(slot) returns the PreprocOutput.
//   - submit throws if that slot is pending. After wait, the slot's output buffer is owned by the caller (HailoRT),
//     and the caller submits the same slot again once it is done with it (after the HailoRT completion callback).
//   - DrmFrame.keepalive is copied at submit and released at wait (when fence completion is confirmed).
//     The decoder stalls if roughly 12–13 or more frames are held at once, so keep the number of pending slots small (2–4).
//   - Not thread-safe. Call submit/wait from a single thread.
// Synchronization:
//   - Input dma-buf: decoding is considered complete when the decoder returns the frame (V4L2 dequeue). At the start of the command buffer an
//     external queue family (FOREIGN_EXT)→compute queue acquire barrier, and at the end a release barrier returns ownership.
//   - Output: shader write → HOST_READ barrier, then fence. Once wait() returns, host_ptr can be read immediately (HOST_COHERENT).
//   - Column LUT: rewritten by the host only when the input size, SAND column stride or letterbox changes (after waiting for pending work).
// Constraint: y_offset and uv_offset must be multiples of 4 (row addresses are computed in 32-bit word units).
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
    int num_slots = 8;               // number of output ring slots (same as HailoRunnerConfig::num_slots)
    int dst_w = 640;                 // multiple of 4
    int dst_h = 640;
    // Workgroup size (x: group of 4 pixels, y: rows). x*y <= maxComputeWorkGroupInvocations (V3D 256).
    // V3D 7.1 measurement (yvo_preproc_bench --sweep, 1080p→640): 64 inv 3.65 ms, 128 inv 2.45 ms, 256 inv 2.13 ms.
    std::uint32_t local_size_x = 32;
    std::uint32_t local_size_y = 8;
    // Output tensor layout. With Nhcw, HailoRT's CPU input conversion (about 14 ms/frame) is not needed (see frame_types.hpp).
    TensorLayout out_layout = TensorLayout::Nhwc;
    bool export_dmabuf = false;      // stage 2 preparation: allocate output memory so it can be exported as a dma-buf
    bool timestamps = true;          // timestamp measurement of the shader section
    std::size_t import_cache_entries = 64;
};

// Shader push constants (std430, same order and size as Params in 02_shaders/sand_letterbox.comp)
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

    // Asynchronous: command recording + vkQueueSubmit. slot is in [0, num_slots).
    void submit(const DrmFrame& frame, int slot);
    // Waits on the slot's fence and returns the result. Releases the keepalive (AVFrame) here.
    PreprocOutput wait(int slot, std::uint64_t timeout_ns = UINT64_MAX);
    // Whether the fence has already signaled (does not wait). true if not pending.
    bool ready(int slot) const;
    bool pending(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).pending; }
    // Synchronous convenience function: submit + wait
    PreprocOutput process(const DrmFrame& frame, int slot);

    int num_slots() const { return static_cast<int>(slots_.size()); }
    std::size_t output_size() const { return out_size_; }
    std::uint8_t* host_ptr(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).mapped; }
    // For stage 2: export output memory as a dma-buf fd (requires config.export_dmabuf). Returns a new fd; the caller closes it.
    int export_dmabuf_fd(int slot) const;

    // GPU time (ms) of the shader section for the last waited slot. < 0 if not measurable.
    double last_gpu_ms() const { return last_gpu_ms_; }
    const DmaBufImportCache& import_cache() const { return cache_; }
    // Call when closing or reopening the decoder (there must be no pending slots)
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
        DrmFrame frame;                    // holds keepalive
        ImportedDmaBuf* y = nullptr;       // pinned import entry
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
    // Updates the column LUT (shader binding 3) to match key. If changed, rewrites it after pending GPU work finishes.
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
    VkBuffer lut_buf_ = VK_NULL_HANDLE;       // column LUT: dst_w × 2 × uvec4 (HOST_VISIBLE|COHERENT)
    VkDeviceMemory lut_mem_ = VK_NULL_HANDLE;
    std::uint32_t* lut_ptr_ = nullptr;
    VkDeviceSize lut_size_ = 0;
    LutKey lut_key_{};
    bool lut_valid_ = false;
    std::vector<Slot> slots_;
    double last_gpu_ms_ = -1.0;
};

} // namespace yvo::vk
