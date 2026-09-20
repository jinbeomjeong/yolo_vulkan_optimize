// vulkan_wayland.h 는 이 .cpp 에서만 포함한다(헤더는 wl_display* 를 void* 로만 다룬다).
// yvo_display 는 이미 PkgConfig::YVO_WAYLAND 에 의존하므로 새 의존성이 생기지 않는다.
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

// acquire 세마포어의 wait 스테이지. 레이아웃 전환(barrier)이 세마포어 대기 뒤에 확실히 오도록
// ALL_COMMANDS 로 둔다 — V3D 에서 비용은 무시할 수 있고, 동기화 검증에서 모호함이 남지 않는다.
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
        throw std::runtime_error("VulkanSwapchain: VulkanContext present가 켜져 있지 않다 (" +
                                 ctx_.present_unavailable_reason() + ")");
    if (opt_.wayland_display == nullptr || opt_.wayland_surface == nullptr)
        throw std::invalid_argument("VulkanSwapchain: wayland_display/wayland_surface가 필요하다");
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
        // 종료 경로에서만 vkDeviceWaitIdle 을 쓴다(프레임 루프에서는 절대 쓰지 않는다).
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
    if (!p_create) throw std::runtime_error("VulkanSwapchain: vkCreateWaylandSurfaceKHR 진입점을 얻지 못했다");

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

    // present mode 별 minImageCount (VK_EXT_surface_maintenance1 + VK_KHR_get_surface_capabilities2)
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

    // ★ supportedUsageFlags 가 STORAGE 를 보고해도 포맷별 optimalTilingFeatures 는 별개 질의다.
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

    // blit 폴백 판정: 강제(--display-blit) 또는 STORAGE 불가.
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
    // 모드별 minImageCount 는 **opt_.lean_images 일 때만** 쓴다 — 헤더의 경고 참고
    // (검증 레이어가 VUID-...-presentMode-02839 로 기본 caps 값을 강제한다).
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
        throw std::runtime_error("VulkanSwapchain: R8G8B8A8_UNORM + SRGB_NONLINEAR 를 surface 가 지원하지 않는다");
    bool mode_ok = false;
    for (VkPresentModeKHR m : info_.present_modes)
        if (m == opt_.present_mode) mode_ok = true;
    if (!mode_ok)
        throw std::runtime_error(std::string("VulkanSwapchain: present mode ") + present_mode_name(opt_.present_mode) +
                                " 를 surface 가 지원하지 않는다");

    VkDevice dev = ctx_.device();
    // 창 크기가 바뀌면 min/maxImageExtent 도 바뀔 수 있으므로 caps 를 다시 읽는다.
    YVO_VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.physical_device(), surface_, &info_.caps));
    // 획득해 둔 이미지와 반납 대기 목록은 구 스왑체인과 함께 버린다(그 이미지들은 구 스왑체인의 것이다).
    stash_ = SwapchainFrame{};
    pending_release_.clear();

    Instance in;
    in.mode = opt_.present_mode;
    // currentExtent == 0xFFFFFFFF → 우리가 정한다. min/max 로 클램프.
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
    // 모드별(축소된) minImageCount 를 쓰려면 이 구조체가 체인에 있어야 한다(VK_EXT_swapchain_maintenance1).
    // lean_images 가 아니면 붙이지 않는다 — 붙일 이유가 없고 present mode 변경도 쓰지 않는다.
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

    // 구 스왑체인은 **자기 present 펜스들이 모두 signaled 될 때** 폐기한다(vkDeviceWaitIdle 없음).
    if (cur_.handle != VK_NULL_HANDLE) {
        retired_.push_back(cur_);
        cur_ = Instance{};
        ++recreates_;
        // 안전망: 재생성이 폭주해도 핸들이 무한히 쌓이지 않게 한다.
        if (retired_.size() > 8) {
            std::fprintf(stderr, "[swapchain] 구 스왑체인이 %zu개 쌓였다 — 한 번 idle 로 정리한다\n",
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
    // 재생성이 예약된 상태에서는 획득하지 않는다. 특히 present 가 OUT_OF_DATE 로 실패하면
    // 그 프레임의 render-finished 세마포어가 **signaled 상태로 남는다** — 같은 슬롯을 다시 쓰면
    // 이미 signaled 인 binary 세마포어를 또 신호하게 되어 불법이다.
    // 세마포어는 스왑체인 인스턴스가 소유하므로 create() 가 새로 만들면서 문제가 사라진다.
    if (needs_recreate_) {
        ++dropped_;
        return false;
    }

    // swapchain_maintenance1 이 없을 때의 안전망: 지난번 획득해 두고 안 그린 이미지를 먼저 쓴다.
    if (stash_.valid) {
        *out = stash_;
        stash_ = SwapchainFrame{};
        return true;
    }

    VkDevice dev = ctx_.device();
    const std::uint32_t n = static_cast<std::uint32_t>(cur_.sync.size());
    const std::uint32_t i = next_sync_ % n;
    SyncSet& s = cur_.sync[i];
    // ★ 슬롯이 아직 사용 중이면 **기다리지 않고 이 프레임 표시를 버린다.**
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
        needs_recreate_ = true; // 이미지는 획득됐다 — 이번 프레임은 그리고 present 뒤에 재생성한다
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
        // acquire 세마포어는 이미 signaled 다. 아무 일도 하지 않는 빈 제출로 그것을 **소비**한다 —
        // 그래야 세마포어 상태가 모호하지 않고 다음 acquire 에서 재사용할 수 있다.
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
        // ★ vkReleaseSwapchainImagesEXT 를 여기서 바로 부르면 안 된다:
        //   VUID-VkReleaseSwapchainImagesInfoEXT-pImageIndices-07786 이 "이미지의 모든 사용이 **실행을
        //   완료**해야 한다"고 요구하는데 위 제출은 아직 진행 중이다(동기화 검증이 실제로 잡았다:
        //   "One or more of the images in this swapchain is still in use").
        //   호스트를 블로킹하지 않기 위해 **펜스가 신호되면 반납하도록 미뤄 둔다**(collect_retired 가 처리).
        pending_release_.push_back({f.image_index, s.submit});
        return;
    }
    // 확장이 없으면 되돌릴 수단이 없다 → 다음 acquire 가 이 이미지를 그대로 쓴다.
    stash_ = f;
}

// 펜스가 신호된 discard 이미지를 스왑체인에 반납한다. 블로킹하지 않는다.
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
    // 스왑체인 이미지는 acquire 직후 내용이 정의돼 있지 않아도 되고 셰이더가 전 픽셀을 덮어쓰므로
    // oldLayout = UNDEFINED 가 항상 유효하고 가장 싸다(타일러 load 가 없다).
    //
    // srcStageMask 가 경로마다 다른 이유:
    //  · storage 모드: 이 스왑체인 이미지의 직전 사용자는 **프레젠테이션 엔진**이고, acquire 세마포어가
    //    그것이 끝났음을 보장한다 → 커맨드 버퍼 안에서 추가 의존성이 필요 없다(NONE).
    //  · blit 모드: 대상은 우리가 만든 **자체 이미지**이고 프레젠테이션 엔진이 건드리지 않는다. 직전 사용자는
    //    같은 이미지 인덱스를 썼던 **이전 프레임의 vkCmdBlitImage 읽기**다. acquire 는 sync 슬롯 단위로
    //    펜스를 확인하지만 이미지 인덱스는 프레젠테이션 엔진이 고르므로 둘이 대응하지 않는다
    //    → 레이아웃 전환(쓰기)이 그 읽기와 WRITE_AFTER_READ 로 충돌한다(동기화 검증이 실제로 잡았다).
    //    배리어의 첫 동기화 범위는 **submission order 로 앞선 모든 커맨드**를 포함하므로, srcStageMask 에
    //    BLIT|COPY 를 넣으면 이전 제출의 blit 읽기까지 덮인다. 읽기가 원인이라 실행 의존성만으로 충분해서
    //    srcAccessMask 는 NONE 으로 둔다.
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
    // blit 폴백: 자체 이미지(GENERAL) → 스왑체인 이미지(TRANSFER_DST) → PRESENT_SRC
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
        s.present_fenced = false; // present 가 실행되지 않았으므로 펜스도 신호되지 않는다
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
