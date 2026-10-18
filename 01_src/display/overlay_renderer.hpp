// OverlayRenderer: compute pass that draws detection boxes, labels, and HUD into the display RGBA storage image
//                  (02_shaders/overlay_boxes.comp). Plan §5.4.
//
// ── What this class does not do ──────────────────────────────────────────────────────────────────
// **It does not create or submit command buffers.** It **continues recording into the same command buffer**
// created by `DisplayConvert` (recommendation of the display track stage 2 report). So there is still 1 submit per present.
//   DisplayConvert::begin(slot) → [swapchain acquire transition] → DisplayConvert::record(...)
//     → OverlayRenderer::record(slot, cb, ...)        ← here
//     → [swapchain present transition] → DisplayConvert::end_and_submit(...) → finish(slot)
//     → OverlayRenderer::collect(slot)                 ← read timestamp after the fence completes
// It uses the same slot number as `DisplayConvert` (= `SwapchainFrame::sync_index`).
//
// ── Detection upload: no staging and no transfer queue ────────────────────────────────────────────────
// 80 detections x 32 B + 152 B header ≈ 2.7 KB, and a CPU→GPU **write** takes 5 µs (plan §5.4 —
// the 10x penalty warned about in REPORT.md §4.3 is in the direction of the CPU **reading**). The per-slot DetBuf is
// allocated as `HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL` and **persistently mapped**; the CPU writes it directly and
// only a single `HOST_WRITE → SHADER_READ | INDIRECT_COMMAND_READ` barrier is inserted.
//
// ── Barrier between the two dispatches (is it needed? → yes) ────────────────────────────────────────────
// `display_convert.comp` and `overlay_boxes.comp` **write the same pixels of the same image** (WAW).
// Two consecutive dispatches in the same command buffer may overlap without a barrier, so
// the overlay could be drawn before the conversion → insert a `SHADER_STORAGE_WRITE → SHADER_STORAGE_WRITE`
// image memory barrier (layout stays GENERAL). The overlay **does not read** the image, so
// READ is not put in dstAccessMask (no alpha blending — R7 always writes alpha as 255).
// This barrier is recorded by `record(..., barrier_prior_compute_write = true)`. Tests and probes that run the overlay alone
// (without a preceding compute write) pass false.
//
// ── Descriptors and dispatch ───────────────────────────────────────────────────────────────────────
// set 0 = {0: swapchain storage image(rgba8), 1: per-slot DetBuf, 2: glyph+name SSBO}
//   → comfortably within V3D limits (storage buffers 8, storage images 4, descriptorIndexing=false).
// **One workgroup per detection**, and `vkCmdDispatchIndirect(cb, det_buf, 0)` reads the uvec3 at the front
// of the DetBuf. It works the same whether the CPU filled it or, later, a GPU NMS shader filled it.
//
// ── Boundary conventions ─────────────────────────────────────────────────────────────────────────────────
// R0~R8 at the top of `overlay_reference.hpp` are the sole reference. The shader and the CPU reference implementation follow the same conventions and
// `05_tests/test_overlay.cpp` judges with **max error 0**. Known tolerated limitation:
// **the color of intersecting pixels of overlapping boxes is order-dependent** (workgroup execution order is undefined).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "ref/overlay_reference.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

struct OverlayConfig {
    int num_slots = 3;            // in-flight count (= swapchain image count, same as DisplayConvert)
    int max_dets = 80;            // upper bound of DetBuf dets[] (plan §5.4)
    // Workgroup size (spec constant). One workgroup per detection, so only x is used.
    // Measured basis (yvo_display_probe --probe overlay --overlay-local N, 1920x1080 offscreen, 200 consecutive frames,
    //   80 detections + labels (text_scale 2), median ms — final shader):
    //   16: 7.27 | 32: 3.96 | 64: 2.09 | **128: 1.03 ← adopted** | 256: 1.03
    // The sand_letterbox rule "the more invocations the better" still holds here up to 128.
    // 256 does not improve further (work per detection is small so the workgroup is not filled).
    std::uint32_t local_size_x = 128;
    bool draw_text = true;        // spec constant kDrawText
    int text_scale = 2;           // spec constant kTextScale. 1080p default 2 (8x16 → 16x32)
    int line_width = 2;           // push constant. 1080p default 2
    bool timestamps = true;
};

// Shader push constant (std430, same order and size as Params in overlay_boxes.comp)
struct OverlayPush {
    float sx, sy, tx, ty;
    std::int32_t dst_w, dst_h;
    std::int32_t line_width;
    std::uint32_t max_dets;
};
static_assert(sizeof(OverlayPush) == 32, "push constant layout must match shader");

// DetBuf header (std430). Must match the front of the shader's DetBuf byte for byte.
//   dispatch 0..11 / count 12 / overflow 16 / text_bytes 20 / text[32] 24..151 / dets[] 152..
struct OverlayDetBufHeader {
    std::uint32_t dispatch[3];    // VkDispatchIndirectCommand compatible (offset 0)
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

// 32 B. Same layout as Rec in the postproc track (plan §4.3).
struct DetGpu {
    float x1, y1, x2, y2;
    float score;
    std::uint32_t class_id;
    std::uint32_t anchor_key;
    std::uint32_t keep;
};
static_assert(sizeof(DetGpu) == 32, "DetGpu must be 32 B (contract shared with the postproc track)");
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

    // Fills the slot's DetBuf directly from the host (no staging). Call before record().
    // If dets exceeds max_dets, only the first max_dets are written and overflow is set to 1.
    // hud is truncated to 128 B. Return value = number of detections actually recorded.
    int set_frame(int slot, const std::vector<Detection>& dets, std::string_view hud = {});

    // Continues recording into the same command buffer. affine is the result of make_overlay_affine(src, dst).
    void record(int slot, VkCommandBuffer cb, VkImageView dst_view, VkExtent2D dst_extent,
                const OverlayAffine& affine, bool barrier_prior_compute_write = true);

    // Call after the submit fence completes — reads the timestamp (does not wait).
    void collect(int slot);
    // GPU time (ms) of the overlay dispatch of the last collect(). < 0 if not measurable.
    double last_shader_ms() const { return last_shader_ms_; }

    std::uint32_t overflow(int slot) const;     // whether detections exceeded the limit in this slot
    int num_slots() const { return static_cast<int>(slots_.size()); }
    int max_dets() const { return cfg_.max_dets; }
    const OverlayConfig& config() const { return cfg_; }
    // Style to pass to the CPU reference implementation (same values as GPU spec constant/push constant)
    OverlayStyle style() const;

private:
    struct Slot {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        std::uint8_t* host = nullptr;   // persistent mapping
        bool recorded = false;
    };

    void create_pipeline();
    // DetBuf barrier (+ optional WAW barrier) → bind → push → timestamp → dispatch
    void record_dispatch(int slot, VkCommandBuffer cb, const void* push, std::uint32_t push_size,
                         bool barrier_prior_compute_write);
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
