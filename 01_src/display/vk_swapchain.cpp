// vulkan_wayland.h is included only in this .cpp (the header handles wl_display* only as void*).
// yvo_display already depends on PkgConfig::YVO_WAYLAND, so no new dependency is introduced.
#define VK_USE_PLATFORM_WAYLAND_KHR 1
#include <vulkan/vulkan.h>

#include "display/vk_swapchain.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::display {

namespace {

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkColorSpaceKHR kColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
constexpr VkImageUsageFlags kStorageUsage =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageUsageFlags kBlitUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

// Wait stage for the acquire semaphore. ALL_COMMANDS so the layout transition (barrier) is guaranteed to come
// after the semaphore wait — the cost on V3D is negligible, and no ambiguity remains for synchronization validation.
constexpr VkPipelineStageFlags2 kAcquireWaitStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout old_layout, VkImageLayout new_layout,
                   VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                   VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

} // namespace

const char* present_mode_name(VkPresentModeKHR m) {
    switch (m) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR: return "IMMEDIATE";
    case VK_PRESENT_MODE_MAILBOX_KHR: return "MAILBOX";
    case VK_PRESENT_MODE_FIFO_KHR: return "FIFO";
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
    default: return "OTHER";
    }
}

bool parse_present_mode(const std::string& s, VkPresentModeKHR* out) {
    if (s == "fifo") *out = VK_PRESENT_MODE_FIFO_KHR;
    else if (s == "mailbox") *out = VK_PRESENT_MODE_MAILBOX_KHR;
    else if (s == "immediate") *out = VK_PRESENT_MODE_IMMEDIATE_KHR;
    else return false;
    return true;
}

const char* format_name(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB: return "R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB: return "B8G8R8A8_SRGB";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "A2B10G10R10_UNORM_PACK32";
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return "A2R10G10B10_UNORM_PACK32";
    case VK_FORMAT_R5G6B5_UNORM_PACK16: return "R5G6B5_UNORM_PACK16";
    default: return "OTHER";
    }
}

const char* color_space_name(VkColorSpaceKHR c) {
    return c == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ? "SRGB_NONLINEAR" : "OTHER";
}

VulkanSwapchain::VulkanSwapchain(yvo::vk::VulkanContext& ctx, const SwapchainOptions& opt) : ctx_(ctx), opt_(opt) {
    if (!ctx_.present_enabled())
        throw std::runtime_error("VulkanSwapchain: VulkanContext present is not enabled (" +
                                 ctx_.present_unavailable_reason() + ")");
    if (opt_.wayland_display == nullptr || opt_.wayland_surface == nullptr)
        throw std::invalid_argument("VulkanSwapchain: wayland_display/wayland_surface is required");
    load_entry_points();
    try {
        query_surface();
    } catch (...) {
        if (surface_) vkDestroySurfaceKHR(ctx_.instance(), surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
        throw;
    }
}

VulkanSwapchain::~VulkanSwapchain() {
    VkDevice dev = ctx_.device();
    if (dev != VK_NULL_HANDLE) {
        // vkDeviceWaitIdle is used only on the shutdown path (never in the frame loop).
        vkDeviceWaitIdle(dev);
        destroy_instance(cur_);
        for (Instance& in : retired_) destroy_instance(in);
        retired_.clear();
    }
    if (surface_) vkDestroySurfaceKHR(ctx_.instance(), surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
}

void VulkanSwapchain::load_entry_points() {
    VkInstance inst = ctx_.instance();
    p_get_caps2_ = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceSurfaceCapabilities2KHR"));
    if (ctx_.has_swapchain_maintenance1())
        p_release_images_ = reinterpret_cast<PFN_vkReleaseSwapchainImagesEXT>(
            vkGetDeviceProcAddr(ctx_.device(), "vkReleaseSwapchainImagesEXT"));
    if (ctx_.has_present_wait())
        p_wait_present_ =
            reinterpret_cast<PFN_vkWaitForPresentKHR>(vkGetDeviceProcAddr(ctx_.device(), "vkWaitForPresentKHR"));
}

void VulkanSwapchain::query_surface() {
    VkInstance inst = ctx_.instance();
    VkPhysicalDevice phys = ctx_.physical_device();
    const std::uint32_t qf = ctx_.queue_family();

    auto p_create = reinterpret_cast<PFN_vkCreateWaylandSurfaceKHR>(
        vkGetInstanceProcAddr(inst, "vkCreateWaylandSurfaceKHR"));
    auto p_support = reinterpret_cast<PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR>(
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceWaylandPresentationSupportKHR"));
    if (!p_create) throw std::runtime_error("VulkanSwapchain: failed to get the vkCreateWaylandSurfaceKHR entry point");

    VkWaylandSurfaceCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
    sci.display = static_cast<struct wl_display*>(opt_.wayland_display);
    sci.surface = static_cast<struct wl_surface*>(opt_.wayland_surface);
    YVO_VK_CHECK(p_create(inst, &sci, nullptr, &surface_));

    info_.queue_family = qf;
    if (p_support)
        info_.wayland_presentation_support =
            p_support(phys, qf, static_cast<struct wl_display*>(opt_.wayland_display));
    YVO_VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(phys, qf, surface_, &info_.surface_support));
    YVO_VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface_, &info_.caps));
    info_.usage_storage = (info_.caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) != 0;

    std::uint32_t n = 0;
    YVO_VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &n, nullptr));
    info_.formats.resize(n);
    if (n) YVO_VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &n, info_.formats.data()));
    for (const auto& f : info_.formats)
        if (f.format == kFormat && f.colorSpace == kColorSpace) info_.format_available = true;

    YVO_VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface_, &n, nullptr));
    info_.present_modes.resize(n);
    if (n) YVO_VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface_, &n, info_.present_modes.data()));
    for (VkPresentModeKHR m : info_.present_modes)
        if (m == VK_PRESENT_MODE_FIFO_KHR) info_.fifo_available = true;

    // minImageCount per present mode (VK_EXT_surface_maintenance1 + VK_KHR_get_surface_capabilities2)
    if (ctx_.has_surface_maintenance1() && p_get_caps2_) {
        for (VkPresentModeKHR m : info_.present_modes) {
            VkSurfacePresentModeEXT pm{};
            pm.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT;
            pm.presentMode = m;
            VkPhysicalDeviceSurfaceInfo2KHR si{};
            si.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR;
            si.pNext = &pm;
            si.surface = surface_;
            VkSurfaceCapabilities2KHR c2{};
            c2.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR;
            if (p_get_caps2_(phys, &si, &c2) == VK_SUCCESS)
                info_.mode_counts.push_back({m, c2.surfaceCapabilities.minImageCount,
                                             c2.surfaceCapabilities.maxImageCount});
        }
    }

    // ★ Even if supportedUsageFlags reports STORAGE, per-format optimalTilingFeatures is a separate query.
    VkPhysicalDeviceImageFormatInfo2 ifi{};
    ifi.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    ifi.format = kFormat;
    ifi.type = VK_IMAGE_TYPE_2D;
    ifi.tiling = VK_IMAGE_TILING_OPTIMAL;
    ifi.usage = kStorageUsage;
    VkImageFormatProperties2 ifp{};
    ifp.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    info_.storage_image_result = vkGetPhysicalDeviceImageFormatProperties2(phys, &ifi, &ifp);
    info_.storage_image_props = ifp.imageFormatProperties;

    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(phys, kFormat, &fp);
    info_.blit_src_supported = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
    info_.blit_dst_supported = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;

    // Blit fallback decision: forced (--display-blit) or STORAGE not possible.
    const bool storage_ok = info_.usage_storage && info_.storage_image_result == VK_SUCCESS;
    blit_ = opt_.force_blit || !storage_ok;
    blit_use_copy_ = !(info_.blit_src_supported && info_.blit_dst_supported);
    if (opt_.verbose)
        std::printf("[swapchain] storage=%s blit_mode=%s(%s) format=%s modes=%zu mode_counts=%zu\n",
                    storage_ok ? "ok" : yvo::vk::result_str(info_.storage_image_result), blit_ ? "yes" : "no",
                    blit_ ? (blit_use_copy_ ? "vkCmdCopyImage" : "vkCmdBlitImage") : "-", format_name(kFormat),
                    info_.present_modes.size(), info_.mode_counts.size());
}

std::uint32_t VulkanSwapchain::choose_image_count(VkPresentModeKHR mode) const {
    std::uint32_t mn = info_.caps.minImageCount;
    std::uint32_t mx = info_.caps.maxImageCount;
    // Per-mode minImageCount is used **only when opt_.lean_images** — see the warning in the header
    // (the validation layer enforces the default caps value via VUID-...-presentMode-02839).
    if (opt_.lean_images && ctx_.has_swapchain_maintenance1()) {
        for (const SurfaceModeCount& c : info_.mode_counts)
            if (c.mode == mode) {
                mn = c.min_image_count;
                if (c.max_image_count) mx = c.max_image_count;
                break;
            }
    }
    std::uint32_t n = mn + opt_.extra_images;
    if (mx != 0 && n > mx) n = mx;
    return n;
}

void VulkanSwapchain::create(int width, int height) {
    if (!info_.format_available)
        throw std::runtime_error("VulkanSwapchain: surface does not support R8G8B8A8_UNORM + SRGB_NONLINEAR");
    bool mode_ok = false;
    for (VkPresentModeKHR m : info_.present_modes)
        if (m == opt_.present_mode) mode_ok = true;
    if (!mode_ok)
        throw std::runtime_error(std::string("VulkanSwapchain: present mode ") + present_mode_name(opt_.present_mode) +
                                " is not supported by the surface");

    VkDevice dev = ctx_.device();
    // If the window size changes, min/maxImageExtent may change too, so re-read caps.
    YVO_VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.physical_device(), surface_, &info_.caps));
    // Acquired images and the pending-release list are dropped along with the old swapchain (those images belong to it).
    stash_ = SwapchainFrame{};
    pending_release_.clear();

    Instance in;
    in.mode = opt_.present_mode;
    // currentExtent == 0xFFFFFFFF → we decide. Clamp to min/max.
    const VkExtent2D mn = info_.caps.minImageExtent, mx = info_.caps.maxImageExtent;
    in.extent.width = std::clamp(static_cast<std::uint32_t>(std::max(width, 1)), mn.width, mx.width);
    in.extent.height = std::clamp(static_cast<std::uint32_t>(std::max(height, 1)), mn.height, mx.height);
    if (info_.caps.currentExtent.width != 0xFFFFFFFFu) in.extent = info_.caps.currentExtent;

    const std::uint32_t want_images = choose_image_count(in.mode);

    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(info_.caps.supportedCompositeAlpha & alpha)) alpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;

    VkSwapchainPresentModesCreateInfoEXT pmci{};
    pmci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT;
    pmci.presentModeCount = 1;
    pmci.pPresentModes = &in.mode;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    // To use the per-mode (reduced) minImageCount this struct must be in the chain (VK_EXT_swapchain_maintenance1).
    // Not attached unless lean_images — there is no reason to, and present mode changes are not used either.
    if (opt_.lean_images && ctx_.has_swapchain_maintenance1()) ci.pNext = &pmci;
    ci.surface = surface_;
    ci.minImageCount = want_images;
    ci.imageFormat = kFormat;
    ci.imageColorSpace = kColorSpace;
    ci.imageExtent = in.extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = blit_ ? kBlitUsage : kStorageUsage;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = info_.caps.currentTransform;
    ci.compositeAlpha = alpha;
    ci.presentMode = in.mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = cur_.handle;

    const VkResult r = vkCreateSwapchainKHR(dev, &ci, nullptr, &in.handle);
    if (r != VK_SUCCESS) throw yvo::vk::VulkanError("vkCreateSwapchainKHR", r);

    try {
        std::uint32_t n = 0;
        YVO_VK_CHECK(vkGetSwapchainImagesKHR(dev, in.handle, &n, nullptr));
        in.images.resize(n);
        YVO_VK_CHECK(vkGetSwapchainImagesKHR(dev, in.handle, &n, in.images.data()));

        if (!blit_) {
            in.views.resize(n, VK_NULL_HANDLE);
            for (std::uint32_t i = 0; i < n; ++i) {
                VkImageViewCreateInfo vi{};
                vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vi.image = in.images[i];
                vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vi.format = kFormat;
                vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                YVO_VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &in.views[i]));
            }
        } else {
            create_own_images(in);
        }
        create_sync(in, n);
    } catch (...) {
        destroy_instance(in);
        throw;
    }

    // The old swapchain is destroyed **when all its present fences are signaled** (no vkDeviceWaitIdle).
    if (cur_.handle != VK_NULL_HANDLE) {
        retired_.push_back(cur_);
        cur_ = Instance{};
        ++recreates_;
        // Safety net: keep handles from piling up indefinitely even if recreation runs away.
        if (retired_.size() > 8) {
            std::fprintf(stderr, "[swapchain] %zu old swapchains piled up — cleaning up once with idle\n",
                         retired_.size());
            vkDeviceWaitIdle(dev);
            for (Instance& old : retired_) destroy_instance(old);
            retired_.clear();
        }
    }
    cur_ = in;
    next_sync_ = 0;
    needs_recreate_ = false;
    if (opt_.verbose)
        std::printf("[swapchain] created %ux%u images=%u(min %u) mode=%s %s\n", cur_.extent.width, cur_.extent.height,
                    static_cast<std::uint32_t>(cur_.images.size()), want_images, present_mode_name(cur_.mode),
                    blit_ ? "(blit fallback)" : "(compute -> swapchain storage image)");
    collect_retired();
}

void VulkanSwapchain::create_own_images(Instance& in) {
    VkDevice dev = ctx_.device();
    const std::size_t n = in.images.size();
    in.own_images.resize(n, VK_NULL_HANDLE);
    in.own_mems.resize(n, VK_NULL_HANDLE);
    in.own_views.resize(n, VK_NULL_HANDLE);
    for (std::size_t i = 0; i < n; ++i) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = kFormat;
        ici.extent = {in.extent.width, in.extent.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        YVO_VK_CHECK(vkCreateImage(dev, &ici, nullptr, &in.own_images[i]));

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(dev, in.own_images[i], &req);
        const std::uint32_t type = ctx_.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX) throw std::runtime_error("VulkanSwapchain: no DEVICE_LOCAL memory type");
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &in.own_mems[i]));
        YVO_VK_CHECK(vkBindImageMemory(dev, in.own_images[i], in.own_mems[i], 0));

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = in.own_images[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = kFormat;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        YVO_VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &in.own_views[i]));
    }
}

void VulkanSwapchain::create_sync(Instance& in, std::uint32_t n) {
    VkDevice dev = ctx_.device();
    in.sync.resize(n);
    for (SyncSet& s : in.sync) {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateSemaphore(dev, &si, nullptr, &s.acquire));
        YVO_VK_CHECK(vkCreateSemaphore(dev, &si, nullptr, &s.render));
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &s.submit));
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &s.present));
    }
}

bool VulkanSwapchain::instance_idle(const Instance& in) const {
    VkDevice dev = ctx_.device();
    for (const SyncSet& s : in.sync) {
        if (!s.in_flight) continue;
        if (vkGetFenceStatus(dev, s.submit) != VK_SUCCESS) return false;
        if (s.present_fenced && vkGetFenceStatus(dev, s.present) != VK_SUCCESS) return false;
    }
    return true;
}

void VulkanSwapchain::collect_retired() {
    flush_pending_releases();
    if (retired_.empty()) return;
    auto it = retired_.begin();
    while (it != retired_.end()) {
        if (instance_idle(*it)) {
            destroy_instance(*it);
            it = retired_.erase(it);
        } else {
            ++it;
        }
    }
}

void VulkanSwapchain::destroy_instance(Instance& in) {
    VkDevice dev = ctx_.device();
    if (dev == VK_NULL_HANDLE) return;
    for (SyncSet& s : in.sync) {
        if (s.acquire) vkDestroySemaphore(dev, s.acquire, nullptr);
        if (s.render) vkDestroySemaphore(dev, s.render, nullptr);
        if (s.submit) vkDestroyFence(dev, s.submit, nullptr);
        if (s.present) vkDestroyFence(dev, s.present, nullptr);
    }
    in.sync.clear();
    for (VkImageView v : in.views)
        if (v) vkDestroyImageView(dev, v, nullptr);
    for (VkImageView v : in.own_views)
        if (v) vkDestroyImageView(dev, v, nullptr);
    for (VkImage i : in.own_images)
        if (i) vkDestroyImage(dev, i, nullptr);
    for (VkDeviceMemory m : in.own_mems)
        if (m) vkFreeMemory(dev, m, nullptr);
    in.views.clear();
    in.own_views.clear();
    in.own_images.clear();
    in.own_mems.clear();
    in.images.clear();
    if (in.handle) vkDestroySwapchainKHR(dev, in.handle, nullptr);
    in.handle = VK_NULL_HANDLE;
}

void VulkanSwapchain::request_recreate(int width, int height) {
    needs_recreate_ = true;
    pending_w_ = width;
    pending_h_ = height;
    if (pending_w_ > 0 && pending_h_ > 0) {
        opt_.width = pending_w_;
        opt_.height = pending_h_;
    }
}

bool VulkanSwapchain::acquire(SwapchainFrame* out) {
    *out = SwapchainFrame{};
    if (!created()) return false;
    collect_retired();
    flush_pending_releases();
    // Do not acquire while recreation is scheduled. In particular, if present fails with OUT_OF_DATE,
    // that frame's render-finished semaphore **remains signaled** — reusing the same slot would
    // signal an already-signaled binary semaphore again, which is illegal.
    // Semaphores are owned by the swapchain instance, so the problem goes away when create() makes new ones.
    if (needs_recreate_) {
        ++dropped_;
        return false;
    }

    // Safety net when swapchain_maintenance1 is absent: use the image acquired last time but not drawn first.
    if (stash_.valid) {
        *out = stash_;
        stash_ = SwapchainFrame{};
        return true;
    }

    VkDevice dev = ctx_.device();
    const std::uint32_t n = static_cast<std::uint32_t>(cur_.sync.size());
    const std::uint32_t i = next_sync_ % n;
    SyncSet& s = cur_.sync[i];
    // ★ If the slot is still in use, **drop this frame's display without waiting.**
    if (s.in_flight) {
        if (vkGetFenceStatus(dev, s.submit) != VK_SUCCESS ||
            (s.present_fenced && vkGetFenceStatus(dev, s.present) != VK_SUCCESS)) {
            ++dropped_;
            ++sync_busy_;
            return false;
        }
        s.in_flight = false;
        s.present_fenced = false;
    }

    std::uint32_t index = 0;
    const VkResult r = vkAcquireNextImageKHR(dev, cur_.handle, 0, s.acquire, VK_NULL_HANDLE, &index);
    if (r == VK_NOT_READY || r == VK_TIMEOUT) {
        ++dropped_;
        ++not_ready_;
        return false;
    }
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        needs_recreate_ = true;
        ++dropped_;
        return false;
    }
    if (r == VK_SUBOPTIMAL_KHR) {
        needs_recreate_ = true; // the image was acquired — draw this frame and recreate after present
    } else if (r != VK_SUCCESS) {
        throw yvo::vk::VulkanError("vkAcquireNextImageKHR", r);
    }

    YVO_VK_CHECK(vkResetFences(dev, 1, &s.submit));
    YVO_VK_CHECK(vkResetFences(dev, 1, &s.present));
    ++next_sync_;

    out->valid = true;
    out->image_index = index;
    out->sync_index = i;
    out->swapchain_image = cur_.images[index];
    out->target_image = blit_ ? cur_.own_images[index] : cur_.images[index];
    out->target_view = blit_ ? cur_.own_views[index] : cur_.views[index];
    out->extent = cur_.extent;
    out->wait_semaphore = s.acquire;
    out->signal_semaphore = s.render;
    out->submit_fence = s.submit;
    out->present_id = ctx_.has_present_id() ? cur_.next_present_id : 0;
    return true;
}

void VulkanSwapchain::discard(const SwapchainFrame& f) {
    if (!f.valid) return;
    ++dropped_;
    SyncSet& s = cur_.sync[f.sync_index];
    if (p_release_images_) {
        // The acquire semaphore is already signaled. **Consume** it with an empty submit that does nothing —
        // so the semaphore state is unambiguous and it can be reused on the next acquire.
        VkSemaphoreSubmitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        wi.semaphore = f.wait_semaphore;
        wi.stageMask = kAcquireWaitStage;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.waitSemaphoreInfoCount = 1;
        si.pWaitSemaphoreInfos = &wi;
        YVO_VK_CHECK(vkQueueSubmit2(ctx_.queue(), 1, &si, s.submit));
        s.in_flight = true;
        s.present_fenced = false;
        // ★ Do not call vkReleaseSwapchainImagesEXT right here:
        //   VUID-VkReleaseSwapchainImagesInfoEXT-pImageIndices-07786 requires "all uses of the image must have **completed
        //   execution**", but the submit above is still in progress (synchronization validation actually caught it:
        //   "One or more of the images in this swapchain is still in use").
        //   To avoid blocking the host, **defer the release until the fence is signaled** (handled by collect_retired).
        pending_release_.push_back({f.image_index, s.submit});
        return;
    }
    // Without the extension there is no way to return it → the next acquire uses this image as-is.
    stash_ = f;
}

// Returns discard images whose fence has been signaled to the swapchain. Does not block.
void VulkanSwapchain::flush_pending_releases() {
    if (pending_release_.empty() || !p_release_images_) return;
    VkDevice dev = ctx_.device();
    std::vector<std::uint32_t> ready;
    auto it = pending_release_.begin();
    while (it != pending_release_.end()) {
        if (vkGetFenceStatus(dev, it->fence) == VK_SUCCESS) {
            ready.push_back(it->image_index);
            it = pending_release_.erase(it);
        } else {
            ++it;
        }
    }
    if (ready.empty()) return;
    VkReleaseSwapchainImagesInfoEXT ri{};
    ri.sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT;
    ri.swapchain = cur_.handle;
    ri.imageIndexCount = static_cast<std::uint32_t>(ready.size());
    ri.pImageIndices = ready.data();
    const VkResult r = p_release_images_(dev, &ri);
    if (r != VK_SUCCESS) throw yvo::vk::VulkanError("vkReleaseSwapchainImagesEXT", r);
    released_ += ready.size();
}

void VulkanSwapchain::record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const {
    // Swapchain image contents may be undefined right after acquire and the shader overwrites every pixel, so
    // oldLayout = UNDEFINED is always valid and cheapest (no tiler load).
    //
    // Why srcStageMask differs per path:
    //  · storage mode: the previous user of this swapchain image is the **presentation engine**, and the acquire semaphore
    //    guarantees it has finished → no additional dependency is needed inside the command buffer (NONE).
    //  · blit mode: the target is **our own image**, which the presentation engine never touches. The previous user is
    //    the **vkCmdBlitImage read of an earlier frame** that used the same image index. acquire checks fences per sync slot,
    //    but the image index is chosen by the presentation engine, so the two do not correspond
    //    → the layout transition (a write) conflicts with that read as WRITE_AFTER_READ (synchronization validation actually caught it).
    //    The barrier's first synchronization scope includes **all commands earlier in submission order**, so putting
    //    BLIT|COPY in srcStageMask also covers the blit read of the previous submit. Since the cause is a read, an execution dependency alone suffices, so
    //    srcAccessMask is left as NONE.
    const VkPipelineStageFlags2 src =
        blit_ ? (VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT) : VK_PIPELINE_STAGE_2_NONE;
    image_barrier(cb, f.target_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, src, VK_ACCESS_2_NONE,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void VulkanSwapchain::record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const {
    if (!blit_) {
        image_barrier(cb, f.swapchain_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
        return;
    }
    // Blit fallback: own image (GENERAL) → swapchain image (TRANSFER_DST) → PRESENT_SRC
    image_barrier(cb, f.target_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    image_barrier(cb, f.swapchain_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                  VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const VkImageSubresourceLayers sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    if (!blit_use_copy_) {
        VkImageBlit region{};
        region.srcSubresource = sub;
        region.dstSubresource = sub;
        region.srcOffsets[1] = {static_cast<std::int32_t>(f.extent.width), static_cast<std::int32_t>(f.extent.height),
                                1};
        region.dstOffsets[1] = region.srcOffsets[1];
        vkCmdBlitImage(cb, f.target_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, f.swapchain_image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
    } else {
        VkImageCopy region{};
        region.srcSubresource = sub;
        region.dstSubresource = sub;
        region.extent = {f.extent.width, f.extent.height, 1};
        vkCmdCopyImage(cb, f.target_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, f.swapchain_image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    image_barrier(cb, f.swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
}

VkResult VulkanSwapchain::present(const SwapchainFrame& f) {
    SyncSet& s = cur_.sync[f.sync_index];
    s.in_flight = true;

    VkPresentIdKHR pid{};
    pid.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR;
    pid.swapchainCount = 1;
    pid.pPresentIds = &f.present_id;

    VkSwapchainPresentFenceInfoEXT pf{};
    pf.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT;
    pf.swapchainCount = 1;
    pf.pFences = &s.present;

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &f.signal_semaphore;
    pi.swapchainCount = 1;
    pi.pSwapchains = &cur_.handle;
    pi.pImageIndices = &f.image_index;
    const void** tail = &pi.pNext;
    if (f.present_id != 0) {
        *tail = &pid;
        tail = &pid.pNext;
    }
    if (ctx_.has_swapchain_maintenance1()) {
        *tail = &pf;
        s.present_fenced = true;
    }

    const VkResult r = vkQueuePresentKHR(ctx_.queue(), &pi);
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
        ++presented_;
        if (f.present_id != 0) cur_.next_present_id = f.present_id + 1;
        if (r == VK_SUBOPTIMAL_KHR) needs_recreate_ = true;
    } else if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        needs_recreate_ = true;
        s.present_fenced = false; // present was not executed, so the fence will not be signaled either
    } else {
        throw yvo::vk::VulkanError("vkQueuePresentKHR", r);
    }
    return r;
}

bool VulkanSwapchain::present_wait_available() const { return p_wait_present_ != nullptr && ctx_.has_present_id(); }

bool VulkanSwapchain::wait_present(std::uint64_t present_id, std::uint64_t timeout_ns) {
    if (!present_wait_available() || present_id == 0 || !created()) return false;
    const VkResult r = p_wait_present_(ctx_.device(), cur_.handle, present_id, timeout_ns);
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) return true;
    if (r == VK_TIMEOUT || r == VK_ERROR_OUT_OF_DATE_KHR) return false;
    throw yvo::vk::VulkanError("vkWaitForPresentKHR", r);
}

} // namespace yvo::display
