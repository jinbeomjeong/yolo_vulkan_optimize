// DisplayConvert: decoder SAND128 NV12 dma-buf → RGBA storage image for display (02_shaders/display_convert.comp)
//
// plan §5.3 option (I): **convert at present time and write directly into the swapchain image** → no ring buffer and no copy.
// The execution path follows SandLetterbox's template (slots, barriers, spec constants, timestamps) as is,
// except that the output is **a VkImageView given by the swapchain (rgba8 storage image)** rather than a VkBuffer.
// SandLetterbox itself is the tuned 2.13 ms path and is left untouched — this class was written from scratch.
//
// Slot convention: the slot number is **SwapchainFrame::sync_index given by VulkanSwapchain**, used as is.
//   begin(slot) → [swapchain acquire transition] → record(slot, ...) → [swapchain present transition] →
//   end_and_submit(slot, wait, signal, fence) → (after the fence completes) finish(slot)
//   finish() releases the import pin and drops the DrmFrame keepalive (returned to the decoder pool).
//
// Synchronization: the input dma-buf is owned by external_qf (FOREIGN_EXT), so acquire/release barriers are placed before and after the command buffer
//   exactly following the sand_letterbox.cpp:274-298 / :312-356 pattern (both preprocessing and display **only read** it).
//   The output image layout transitions are recorded by VulkanSwapchain (UNDEFINED → GENERAL → PRESENT_SRC).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "vulkan/dmabuf_import.hpp"
#include "vulkan/sand_column_lut.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

struct DisplayConvertConfig {
    int num_slots = 3;                 // number in flight (= number of swapchain images)
    // Workgroup size. x is in units of **4-pixel groups**. x*y <= 256 (V3D maxComputeWorkGroupInvocations).
    // Default 3x85 = 255 invocations, i.e. a **tall 12 x 85 pixel tile**.
    // Measurement basis (yvo_display_probe --probe convert --sweep, 1920x1080 1:1, offscreen 200 frames, V3D 7.1.
    //   DVFS drift was excluded by 3 interleaved repetitions — reproducibility ±0.02 ms):
    //   8x8(64)   15.01 | 4x16(64)  13.12 | 16x8(128) 10.07 | 8x16(128)  9.94 | 3x48(144)  7.74
    //   16x16(256) 7.74 | 8x32(256)  7.78 | 32x8(256)  7.80 | 128x2(256) 8.09 | 64x4(256)  8.05
    //   2x64(128)  7.59 | 3x64(192)  6.70 | 5x36(180)  6.63 | 1x256(256) 6.78 | 4x64(256)  6.62
    //   7x36(252)  6.14 | 6x42(252)  6.15 | 5x51(255)  5.84 | 3x72(216)  5.59 | 2x108(216) 5.28
    //   2x128(256) 6.21 | 3x84(252)  4.55 | **3x85(255) 4.23  ← chosen**
    // Two effects act together:
    //   ① sand_letterbox's rule that more invocations are better still holds (64 inv is 15 ms, 256 inv is 4~8 ms).
    //   ② Within the same invocation count, **tall tiles are decisively faster** — 64x16 px 7.74 → 12x85 px 4.23,
    //      1.83x. Looking only at reads (SAND columns are 128 B wide), narrow tiles waste cache lines and should be worse, yet
    //      they are faster → the bottleneck is on the **tiled (UIF) RGBA image write** side, and tall regions are
    //      believed to fit the image tiles better. **This interpretation is an unconfirmed hypothesis; only the numbers are evidence.**
    // The optimum may change if the resolution changes. It is a spec constant, so re-measure with --probe convert --sweep.
    std::uint32_t local_size_x = 3;
    std::uint32_t local_size_y = 85;
    // true if the output size differs from the source (only then bilinear + column LUT). The host decides and passes it.
    bool uniform_scale = false;
    bool timestamps = true;
    std::uint8_t pad_value = 0;        // letterbox margin color for uniform scale (black for display)
    int max_dst_w = 1920;              // column LUT upper bound. Must not be exceeded even if the window grows on recreate.
    std::size_t import_cache_entries = 64;
};

// Shader push constant (std430, same order and size as Params in 02_shaders/display_convert.comp)
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

    // Resets and begins the command buffer and resets the query pool. Record the swapchain barriers into the returned cb as well.
    VkCommandBuffer begin(int slot);
    // Input dma-buf import (cached) + descriptor update + barriers + dispatch recording.
    void record(int slot, const DrmFrame& frame, VkImageView dst_view, VkExtent2D dst_extent);
    // Closes and submits the command buffer. wait/signal are the swapchain semaphores, fence is the swapchain submit fence.
    void end_and_submit(int slot, VkSemaphore wait, VkSemaphore signal, VkFence fence);

    bool pending(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).pending; }
    bool ready(int slot) const;      // has the fence finished (does not wait)
    // Waits for the fence, then reads timestamps and releases the import pin and keepalive.
    void finish(int slot, std::uint64_t timeout_ns = UINT64_MAX);

    // GPU time (ms) of the last finish(). < 0 if not measurable.
    double last_shader_ms() const { return last_shader_ms_; }   // dispatch interval only (same basis as 2.13 ms)
    double last_total_ms() const { return last_total_ms_; }     // whole command buffer (including blit fallback cost)

    int num_slots() const { return static_cast<int>(slots_.size()); }
    const DisplayConvertConfig& config() const { return cfg_; }
    void set_uniform_scale(bool on);   // when the window size changes and the path changes (recreates the pipeline)
    void clear_import_cache();
    const yvo::vk::DmaBufImportCache& import_cache() const { return cache_; }

private:
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;   // not owned (owned by the swapchain). Stored at submit time.
        bool pending = false;
        bool recording = false;
        DrmFrame frame;
        yvo::vk::ImportedDmaBuf* y = nullptr;
        yvo::vk::ImportedDmaBuf* uv = nullptr;
    };
    using LutKey = yvo::vk::SandColumnLutKey;

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
