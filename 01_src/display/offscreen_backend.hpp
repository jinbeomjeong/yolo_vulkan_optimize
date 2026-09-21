// DisplayBackend / OffscreenBackend: the "display target image provider" abstraction seen by DisplayPipeline.
//
// Why abstract it:
//   `DisplayPipeline`'s frame loop (acquire → begin → barrier → convert → overlay → barrier →
//   submit → present → fence polling) is **exactly the same** whether or not there is a screen. The only differences are where
//   the image comes from and where the result goes. So only those two are pulled out as virtual functions and there is a single loop.
//     · `SwapchainBackend` (inside display_pipeline.cpp) — `VulkanSwapchain` adapter. The real HDMI path.
//     · `OffscreenBackend`  (this file)                  — headless fallback when there is no Wayland.
//
// What OffscreenBackend does:
//   Instead of swapchain images it cycles **its own N R8G8B8A8_UNORM STORAGE images + N fences**.
//   It creates no semaphores — when `DisplayConvert::end_and_submit` receives VK_NULL_HANDLE it
//   sets waitSemaphoreInfoCount/signalSemaphoreInfoCount to 0 (verified in display_convert.cpp).
//   present sends nothing anywhere and only counts presented_. Hence there is no present_wait either
//   (`present_wait_available()==false` → empty glass-to-glass statistics are normal).
//   The image layout is the same as the swapchain, UNDEFINED → GENERAL → GENERAL kept (instead of PRESENT_SRC).
//   It must not be sent to PRESENT_SRC — there is no reason to use that layout for a non-swapchain image,
//   and the next frame's acquire transition starts from UNDEFINED anyway.
#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "display/vk_swapchain.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

// The minimal surface used by DisplayPipeline. Names and meanings match a subset of VulkanSwapchain.
class DisplayBackend {
public:
    virtual ~DisplayBackend() = default;

    // Never blocks. false if there is no image to write to (display drop).
    virtual bool acquire(SwapchainFrame* out) = 0;
    virtual void record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const = 0;
    virtual void record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const = 0;
    virtual void present(const SwapchainFrame& f) = 0;

    virtual bool needs_recreate() const { return false; }
    virtual void request_recreate(int /*w*/, int /*h*/) {}
    virtual void recreate(int /*w*/, int /*h*/) {}
    virtual void collect_retired() {}

    virtual VkExtent2D extent() const = 0;
    virtual std::uint32_t image_count() const = 0;
    virtual bool blit_mode() const { return false; }

    virtual bool present_wait_available() const { return false; }
    virtual bool wait_present(std::uint64_t /*id*/, std::uint64_t /*timeout_ns*/) { return false; }

    virtual std::uint64_t presented() const = 0;
    virtual std::uint64_t display_dropped() const = 0;
    virtual std::uint64_t recreates() const { return 0; }
    virtual const char* name() const = 0;
};

class OffscreenBackend final : public DisplayBackend {
public:
    OffscreenBackend(yvo::vk::VulkanContext& ctx, int width, int height, std::uint32_t images = 3);
    ~OffscreenBackend() override;
    OffscreenBackend(const OffscreenBackend&) = delete;
    OffscreenBackend& operator=(const OffscreenBackend&) = delete;

    bool acquire(SwapchainFrame* out) override;
    void record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const override;
    void record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const override;
    void present(const SwapchainFrame& f) override;

    VkExtent2D extent() const override { return extent_; }
    std::uint32_t image_count() const override { return static_cast<std::uint32_t>(images_.size()); }

    std::uint64_t presented() const override { return presented_; }
    std::uint64_t display_dropped() const override { return dropped_; }
    const char* name() const override { return "offscreen"; }

private:
    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool in_flight = false;
    };
    void destroy();

    yvo::vk::VulkanContext& ctx_;
    VkExtent2D extent_{};
    std::vector<Image> images_;
    std::uint32_t next_ = 0;
    std::uint64_t presented_ = 0, dropped_ = 0;
};

} // namespace yvo::display
