// OverlayPlane: a ring that draws boxes, labels, and HUD with the GPU onto an **exportable (dma-buf) transparent RGBA plane**.
//
// This is the "box plane" of the HVS two-plane display backend (plan atomic-munching-kite §2). The video puts the decoder NV12 SAND128
// dma-buf as-is onto the HVS video plane, and the AB24 dma-buf made by this class goes onto the plane above it (higher zpos),
// and the HVS blends them during scanout. Transport (Wayland linux-dmabuf subsurface, WaylandPlaneScanout) is outside this class.
//
// ── Why a buffer rather than an image (V3D measurement + spec) ─────────────────────────────────────────────
// The plan was to export a LINEAR RGBA8 storage image, but that is impossible on V3D 7.1 (Mesa V3DV).
//   · vkGetPhysicalDeviceFormatProperties2(R8G8B8A8_UNORM): linearTilingFeatures = 0xc980
//     (no STORAGE_IMAGE_BIT). optimalTilingFeatures = 0xdd83 (has it).
//   · VkDrmFormatModifierPropertiesListEXT: MOD_LINEAR(0x0) = 0xc980 (no STORAGE),
//     BROADCOM_UIF(0x0700000000000006) = 0xdd83 (has STORAGE — but it is not for the AB24 LINEAR plane).
//   · vkGetPhysicalDeviceImageFormatProperties2(LINEAR or DRM_FORMAT_MODIFIER(LINEAR), STORAGE|TRANSFER_DST,
//     + VkPhysicalDeviceExternalImageFormatInfo(DMA_BUF)) = VK_ERROR_FORMAT_NOT_SUPPORTED.
//   → creating a VkImage with that combination violates VUID-VkImageCreateInfo-imageCreateMaxMipLevels-02251, and creating a view
//     violates VUID-VkImageViewCreateInfo-usage-02275 ("if usage has STORAGE, the view format features must have STORAGE_IMAGE_BIT").
//     So overlay_boxes.comp writes packed RGBA8 **directly into a storage buffer** with a pitch, and the buffer is
//     allocated/exported by DmaBufExporter in vulkan/dmabuf_export.hpp.
//
// ── Pixel format ─────────────────────────────────────────────────────────────────────────────────
// Memory byte order R,G,B,A = DRM_FORMAT_ABGR8888 ("AB24", little-endian 32-bit A:B:G:R), modifier LINEAR.
// Box/label/HUD pixels have alpha 255 (R7), everything else is (0,0,0,0). Since pixels are either opaque or fully transparent,
// the result is the same whether the HVS blend mode is Pre-multiplied or Coverage.
//
// ── Clearing (ring entries are reused, so the previous content must be cleared) ──────────────────────────────────
//   · Full  : clear everything with vkCmdFillBuffer(0).
//   · Erase : **re-run the detections/HUD last drawn into this entry with the same shader, but write (0,0,0,0)** (push erase=1).
//             With the same input and same affine the pixel set is deterministic, so exactly those pixels are cleared (box outlines,
//             labels, HUD only — the box interior area is not touched). Each entry alternates between two OverlayRenderer
//             slots (drawing slot / slot holding the previous content), so there is no DetBuf copy.
//   Auto is Erase: the V3D GPU time of a full 1920x1080 fill far exceeded the target (0.3 ms).
//   Measured (05_tests/test_overlay.cpp OverlayPlane.ClearCost, timestamp median, 50 frames, 2026-09-22):
//     Full  vkCmdFillBuffer 8.3 MB           : 2.80 ms (independent of detection count)
//     Erase 8 detections + labels + HUD      : 0.44 ms (= same as that frame's overlay, 0.43 ms)
//     Erase 80 detections, text off          : 1.12 ms (overlay 1.11 ms)
//   Erase cost equals "the previous frame's overlay cost". It exceeds 0.3 ms even on a typical frame, so to reduce it further
//   a dedicated clear shader that fills label rectangles instead of label glyphs is a candidate (open task).
//
// ── Synchronization ────────────────────────────────────────────────────────────────────────────────────
// One frame = 1 (clear + overlay) command buffer, 1 submit, 1 fence. `ready(idx)` is non-blocking polling.
// Scanout (HVS/compositor) is a consumer outside Vulkan, so we do an **external queue family ownership transfer** (spec synchronization
// chapter "Queue Family Ownership Transfer": resources shared via external memory require an explicit ownership transfer between local and external queues
// regardless of VkSharingMode). At the end of a frame, release our queue → VK_QUEUE_FAMILY_FOREIGN_EXT (EXTERNAL if the extension is absent),
// and at the start of the next reuse, acquire FOREIGN → our queue. It is a buffer, so there is no layout.
//
// ── Ring entry lifetime (waits until Wayland wl_buffer.release) ─────────────────────────────────────
//   Free --acquire()--> Claimed --draw()--> (GPU in progress) --ready()==true--> Drawn
//        --mark_held()--> Held (consumer is scanning out) --release()--> Free
// If it is decided not to display after drawing (drop), call release() without mark_held. acquire() only hands out Free entries.
// In other words, reuse needs both "GPU fence complete" **and** "consumer release (wl_buffer.release)".
//
// Threads: all on the main thread (one VkQueue).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "display/overlay_renderer.hpp"
#include "ref/overlay_reference.hpp"

namespace yvo::vk {
class VulkanContext;
class DmaBufExporter;
}

namespace yvo::display {

// DRM fourcc/modifier (only the values, to avoid pulling in drm_fourcc.h)
inline constexpr std::uint32_t kDrmFormatAbgr8888 =
    static_cast<std::uint32_t>('A') | (static_cast<std::uint32_t>('B') << 8) |
    (static_cast<std::uint32_t>('2') << 16) | (static_cast<std::uint32_t>('4') << 24);
inline constexpr std::uint64_t kDrmFormatModLinear = 0;

enum class OverlayClearMode { Auto, Full, Erase };

struct OverlayPlaneConfig {
    int ring = 3;                      // number of ring entries (on-screen + pending + drawing)
    int width = 1920;                  // plane size (usually screen = source size)
    int height = 1080;
    std::uint32_t pitch_align = 64;    // byte pitch alignment (stride passed to DRM/HVS)
    int src_w = 1920;                  // detection coordinate space (source of make_overlay_affine). Identity if 1:1
    int src_h = 1080;
    OverlayClearMode clear = OverlayClearMode::Auto;
    // Overlay style and limits. num_slots is overridden by OverlayPlane (2*ring).
    OverlayConfig overlay{};
    bool timestamps = true;
};

// Description of one entry to pass to the consumer (Wayland zwp_linux_buffer_params_v1)
struct OverlayPlaneBuffer {
    int fd = -1;                       // owned by OverlayPlane (use dup_fd() when handing it over)
    std::uint32_t fourcc = kDrmFormatAbgr8888;
    std::uint64_t modifier = kDrmFormatModLinear;
    std::uint32_t offset = 0;          // plane 0 offset (always 0)
    std::uint32_t pitch = 0;           // byte stride (multiple of pitch_align)
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t size = 0;              // pitch * height (valid bytes)
    std::size_t dmabuf_size = 0;       // dma-buf object size measured with lseek (>= size)
};

class OverlayPlane {
public:
    OverlayPlane(yvo::vk::VulkanContext& ctx, const OverlayPlaneConfig& cfg = {});
    ~OverlayPlane();
    OverlayPlane(const OverlayPlane&) = delete;
    OverlayPlane& operator=(const OverlayPlane&) = delete;

    // Detection coordinate space (source size). The affine changes from the next draw().
    void set_source_size(int w, int h);

    // Claims one Free entry. -1 if none (all are in progress on the GPU or held by the consumer).
    int acquire();

    // Records and submits clear + overlay into the claimed entry (1 submit + fence). Return value = number of detections actually recorded.
    int draw(int idx, const std::vector<Detection>& dets, std::string_view hud = {});

    // Non-blocking: has this entry's GPU work finished? If so, collects the timestamp. true if nothing was submitted.
    bool ready(int idx);
    // Blocking wait (for tests and shutdown).
    void wait(int idx);

    // The consumer has taken this entry for scanout (right after wl_surface.commit).
    void mark_held(int idx);
    // The consumer released it (wl_buffer.release) or it was decided not to display it → Free.
    void release(int idx);

    int ring_size() const { return static_cast<int>(entries_.size()); }
    const OverlayPlaneBuffer& buffer(int idx) const;
    int dup_fd(int idx) const;          // the caller closes it
    OverlayClearMode clear_mode() const { return clear_mode_; }
    bool held(int idx) const;
    bool claimed(int idx) const;

    // GPU time (ms) last collected by ready(). < 0 if not measurable.
    double last_clear_ms() const { return last_clear_ms_; }
    double last_overlay_ms() const { return last_overlay_ms_; }
    std::uint64_t overlay_overflows() const { return overflows_; }
    OverlayStyle style() const { return ov_->style(); }
    OverlayAffine affine() const { return affine_; }

private:
    enum class State { Free, Claimed, Held };
    struct Entry {
        OverlayPlaneBuffer desc;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        State state = State::Free;
        bool submitted = false;        // there is a submission with a fence (not yet collected)
        bool released_foreign = false; // the last submission released to FOREIGN → acquire needed next time
        int draw_slot = 0;             // OverlayRenderer slot (2*idx or 2*idx+1)
        bool has_prev = false;         // there is previously drawn content (the target for Erase mode to clear)
        OverlayAffine prev_affine{};
        bool pending_erase = false;    // this submission contains an erase pass (for timestamp collection)
        bool pending_fill = false;
        std::uint64_t seq = 0;         // acquire order (entry idle longest goes first)
    };

    void create();
    void destroy();
    void initial_clear();              // all zeros right after creation (once, blocking)
    void collect(Entry& e, int idx);
    Entry& entry(int idx);
    const Entry& entry(int idx) const;

    yvo::vk::VulkanContext& ctx_;
    OverlayPlaneConfig cfg_;
    OverlayClearMode clear_mode_ = OverlayClearMode::Erase;
    OverlayAffine affine_{};
    std::uint32_t pitch_px_ = 0;
    std::uint32_t foreign_qf_ = VK_QUEUE_FAMILY_EXTERNAL;

    std::unique_ptr<yvo::vk::DmaBufExporter> exporter_;
    std::unique_ptr<OverlayRenderer> ov_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkQueryPool fill_query_ = VK_NULL_HANDLE;   // Full-mode fill interval (2 per entry)
    std::vector<Entry> entries_;
    std::uint64_t seq_ = 0;

    double last_clear_ms_ = -1.0;
    double last_overlay_ms_ = -1.0;
    std::uint64_t overflows_ = 0;
};

const char* clear_mode_name(OverlayClearMode m);

} // namespace yvo::display
