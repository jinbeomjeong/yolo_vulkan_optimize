// VulkanSwapchain: Wayland surface + swapchain + present. Implements plan §5.6 as written.
//
// **VulkanContext never knows about VkSurfaceKHR** — this class creates the surface (plan §5.2).
// Creation order: WaylandWindow::open() → VulkanContext(wayland_display=native_display()) → VulkanSwapchain.
//
// ── Decisions and rationale (plan §5.6) ──────────────────────────────────────────────────────────
//  · Format R8G8B8A8_UNORM + SRGB_NONLINEAR **fixed**. Three reasons point to the same answer:
//      ① In Vulkan an _SRGB format cannot be a storage image
//      ② shaderStorageImageWriteWithoutFormat=false, so a GLSL format qualifier is mandatory, and GLSL has no bgra8,
//         so the only direct compute write is rgba8 = R8G8B8A8_UNORM
//      ③ The values the preprocessing shader (bt709.glsl in sand_letterbox.comp) produces are already gamma-encoded sRGB 0~255 →
//         with UNORM they pass through with no conversion
//  · usage = STORAGE | COLOR_ATTACHMENT | TRANSFER_DST.
//    Even if supportedUsageFlags reports STORAGE, **per-format optimalTilingFeatures is a separate query**, so
//    we check at runtime with vkGetPhysicalDeviceImageFormatProperties2(STORAGE), and if not possible
//    we fall back to **our own image + vkCmdBlitImage**. The blit fallback can be forced with force_blit
//    (probe --display-blit), so it is always tested.
//  · Image count is **queried per present mode** via VK_EXT_surface_maintenance1
//    (measured: default caps 4, but per mode FIFO 3 / MAILBOX 4 / IMMEDIATE 3).
//  · currentExtent = 0xFFFFFFFF, supportedPresentScaling = None → **we make image size = window size ourselves.**
//
// ── acquire rule: never block ────────────────────────────────────────────────────────────────────
//   acquire(timeout = 0). On VK_NOT_READY **this frame's display is dropped** (display_dropped++).
//   Decoding must not be blocked by display. Images acquired but not drawn are
//   returned with vkReleaseSwapchainImagesEXT from VK_EXT_swapchain_maintenance1
//   (without the extension they are kept in a stash for the next acquire to reuse — a safety net not used on this hardware).
//
// ── Recreation: without vkDeviceWaitIdle ─────────────────────────────────────────────────────────
//   Triggered by OUT_OF_DATE / SUBOPTIMAL or WaylandWindow::take_pending_size().
//   Every present attaches a fence via VkSwapchainPresentFenceInfoEXT, so **we know exactly when a submitted image
//   becomes reusable** → the old swapchain is destroyed once all those fences are signaled.
//   Neither the common per-image-fence hack nor vkDeviceWaitIdle is used. Semaphores and fences are **owned separately
//   per swapchain instance** — so the new swapchain does not overwrite fences held by the old swapchain.
//
// ── Synchronization ──────────────────────────────────────────────────────────────────────────────
//   Per in-flight slot (= image count): 2 binary semaphores (acquire, render-finished) + 2 fences (submit, present).
//   Timeline semaphores cannot be used for WSI acquire/present.
//
// ── Latency measurement ──────────────────────────────────────────────────────────────────────────
//   VK_KHR_present_id + VK_KHR_present_wait. vkWaitForPresentKHR(presentId) gives the **actual time it reached the screen**
//   (not GPU time). A wp_presentation fallback is **not built** — the stage-1 probe confirmed present_id=1,
//   present_wait=1, and we do not keep two mechanisms with overlapping purposes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

struct SwapchainOptions {
    void* wayland_display = nullptr;   // struct wl_display*  (WaylandWindow::native_display())
    void* wayland_surface = nullptr;   // struct wl_surface*  (WaylandWindow::native_surface())
    int width = 1920;                  // image size = window size (the compositor does not scale for us)
    int height = 1080;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    bool force_blit = false;           // --display-blit: use the blit fallback even if STORAGE is available
    std::uint32_t extra_images = 0;    // margin added to minImageCount (default 0 = minimum)
    // ⚠ Whether to actually use the per-present-mode minImageCount (FIFO 3). **Default is false.**
    //   Measured (V3D, validation layer 1.4.309): default caps is 4, per FIFO mode it is 3. But creating with 3
    //   triggers VUID-VkSwapchainCreateInfoKHR-presentMode-02839 as an error —
    //   "minImageCount must be greater than or equal to the minImageCount of vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
    //   applies unconditionally, and adding VK_EXT_swapchain_maintenance1's VkSwapchainPresentModesCreateInfoEXT to
    //   the chain does not make the layer relax it (VK_KHR_surface_maintenance1 is not in this driver/header).
    //   "Validation layer errors 0" is this project's hard gate, so **the default is the safe 4**, and
    //   the choice to save one image (1080p RGBA 8.29 MB) is left as an explicit opt-in via this flag.
    bool lean_images = false;
    bool verbose = false;
};

// Image count per present mode (VK_EXT_surface_maintenance1)
struct SurfaceModeCount {
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    std::uint32_t min_image_count = 0;
    std::uint32_t max_image_count = 0;
};

// Surface query results that P4 (--probe surface) dumps as-is
struct SurfaceInfo {
    VkBool32 wayland_presentation_support = VK_FALSE;  // vkGetPhysicalDeviceWaylandPresentationSupportKHR
    VkBool32 surface_support = VK_FALSE;               // vkGetPhysicalDeviceSurfaceSupportKHR
    std::uint32_t queue_family = 0;
    VkSurfaceCapabilitiesKHR caps{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
    std::vector<SurfaceModeCount> mode_counts;   // empty if surface_maintenance1 is absent
    bool format_available = false;               // is R8G8B8A8_UNORM + SRGB_NONLINEAR in the list
    bool usage_storage = false;                  // does supportedUsageFlags include STORAGE
    bool fifo_available = false;
    // vkGetPhysicalDeviceImageFormatProperties2(R8G8B8A8_UNORM, OPTIMAL, STORAGE|COLOR_ATTACHMENT|TRANSFER_DST)
    VkResult storage_image_result = VK_ERROR_FORMAT_NOT_SUPPORTED;
    VkImageFormatProperties storage_image_props{};
    bool blit_src_supported = false;             // whether the own image → swapchain blit fallback is possible
    bool blit_dst_supported = false;
};

// Display target for one frame. Filled by acquire().
struct SwapchainFrame {
    bool valid = false;
    std::uint32_t image_index = 0;
    std::uint32_t sync_index = 0;          // the caller also picks its per-slot resources (command buffers, descriptors) by this index
    VkImage swapchain_image = VK_NULL_HANDLE;
    // The target compute actually writes. Swapchain image in storage mode, own image in blit mode.
    VkImage target_image = VK_NULL_HANDLE;
    VkImageView target_view = VK_NULL_HANDLE;   // rgba8 storage image view
    VkExtent2D extent{};
    VkSemaphore wait_semaphore = VK_NULL_HANDLE;    // acquire semaphore the submit waits on
    VkSemaphore signal_semaphore = VK_NULL_HANDLE;  // render-finished semaphore the submit signals
    VkFence submit_fence = VK_NULL_HANDLE;          // fence to attach to the submit (owned by the swapchain, unsignaled)
    std::uint64_t present_id = 0;                   // id to wait for with present_wait (0 = present_id unused)
};

class VulkanSwapchain {
public:
    // Creates the surface and queries caps, formats, modes, image counts, and STORAGE support. **Does not create the swapchain**
    // (P4 only needs this much). Throws on failure.
    VulkanSwapchain(yvo::vk::VulkanContext& ctx, const SwapchainOptions& opt);
    ~VulkanSwapchain();
    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

    const SurfaceInfo& surface_info() const { return info_; }
    VkSurfaceKHR surface() const { return surface_; }

    // Creates the swapchain (or recreates it at a new size). Does not use vkDeviceWaitIdle.
    void create(int width, int height);
    // Did we hit OUT_OF_DATE/SUBOPTIMAL or did the window size change
    bool needs_recreate() const { return needs_recreate_; }
    void request_recreate(int width, int height);

    bool created() const { return cur_.handle != VK_NULL_HANDLE; }
    VkExtent2D extent() const { return cur_.extent; }
    std::uint32_t image_count() const { return static_cast<std::uint32_t>(cur_.images.size()); }
    bool blit_mode() const { return blit_; }
    VkFormat format() const { return format_.format; }

    // Acquires with timeout = 0. Returns false if no image is ready (display_dropped++) — **never blocks.**
    bool acquire(SwapchainFrame* out);
    // Acquired but decided not to draw. Returns the image (or keeps it for the next acquire to reuse).
    // The release (vkReleaseSwapchainImagesEXT) is **deferred** until the fence of an empty submit that consumed the semaphore is signaled
    // — because the spec requires "all uses of the image must have completed execution", and to avoid blocking the host.
    void discard(const SwapchainFrame& f);

    // Before compute write: transitions the target image UNDEFINED → GENERAL.
    // (Swapchain image contents may be undefined after acquire and the shader overwrites every pixel, so UNDEFINED is cheapest)
    void record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const;
    // After compute write: (in blit mode, after copying own image → swapchain image) transitions to PRESENT_SRC.
    void record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const;

    // Call after the submit. Attaches present_id + present fence. Schedules recreation on OUT_OF_DATE/SUBOPTIMAL.
    VkResult present(const SwapchainFrame& f);
    // vkWaitForPresentKHR: waits until presentId has **actually gone out to the screen**. false if present_wait is absent.
    bool wait_present(std::uint64_t present_id, std::uint64_t timeout_ns);
    bool present_wait_available() const;

    // Destroys old swapchains waiting for reuse if they can be destroyed (fence polling, no blocking). Call every frame.
    void collect_retired();

    // ── Statistics ──
    std::uint64_t presented() const { return presented_; }
    std::uint64_t display_dropped() const { return dropped_; }
    std::uint64_t acquire_not_ready() const { return not_ready_; }
    std::uint64_t sync_busy() const { return sync_busy_; }          // count dropped because in-flight slots were full
    std::uint64_t swapchain_recreates() const { return recreates_; }
    std::uint64_t retired_alive() const { return retired_.size(); }
    std::uint64_t images_released() const { return released_; }        // count returned via vkReleaseSwapchainImagesEXT
    std::size_t pending_releases() const { return pending_release_.size(); }

private:
    struct SyncSet {
        VkSemaphore acquire = VK_NULL_HANDLE;
        VkSemaphore render = VK_NULL_HANDLE;
        VkFence submit = VK_NULL_HANDLE;
        VkFence present = VK_NULL_HANDLE;
        bool in_flight = false;      // are the submit/present fences in use
        bool present_fenced = false; // was a present actually attached to the present fence
    };
    struct Instance {
        VkSwapchainKHR handle = VK_NULL_HANDLE;
        VkExtent2D extent{};
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        std::vector<VkImage> images;
        std::vector<VkImageView> views;       // storage mode: rgba8 views of swapchain images
        std::vector<VkImage> own_images;      // blit mode: own images written by compute
        std::vector<VkDeviceMemory> own_mems;
        std::vector<VkImageView> own_views;
        std::vector<SyncSet> sync;
        std::uint64_t next_present_id = 1;
    };

    void query_surface();
    void load_entry_points();
    void destroy_instance(Instance& in);
    bool instance_idle(const Instance& in) const;
    void create_own_images(Instance& in);
    void flush_pending_releases();
    void create_sync(Instance& in, std::uint32_t n);
    std::uint32_t choose_image_count(VkPresentModeKHR mode) const;

    yvo::vk::VulkanContext& ctx_;
    SwapchainOptions opt_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    SurfaceInfo info_{};
    VkSurfaceFormatKHR format_{VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    bool blit_ = false;
    bool blit_use_copy_ = false;   // falls back to vkCmdCopyImage if blit is not supported
    bool needs_recreate_ = false;
    int pending_w_ = 0, pending_h_ = 0;

    Instance cur_;
    std::vector<Instance> retired_;
    std::uint32_t next_sync_ = 0;

    // stash: safety net used only when swapchain_maintenance1 is absent (images acquired but not drawn)
    SwapchainFrame stash_{};
    // Images to return via discard: released once the fence is signaled (see the discard() comment above)
    struct PendingRelease {
        std::uint32_t image_index = 0;
        VkFence fence = VK_NULL_HANDLE;   // not owned (sync slot fence of the current instance)
    };
    std::vector<PendingRelease> pending_release_;

    std::uint64_t presented_ = 0, dropped_ = 0, not_ready_ = 0, sync_busy_ = 0, recreates_ = 0, released_ = 0;

    // Extension entry points (does not rely on the loader's static exports)
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR p_get_caps2_ = nullptr;
    PFN_vkReleaseSwapchainImagesEXT p_release_images_ = nullptr;
    PFN_vkWaitForPresentKHR p_wait_present_ = nullptr;
};

const char* present_mode_name(VkPresentModeKHR m);
bool parse_present_mode(const std::string& s, VkPresentModeKHR* out);
const char* format_name(VkFormat f);
const char* color_space_name(VkColorSpaceKHR c);

} // namespace yvo::display
