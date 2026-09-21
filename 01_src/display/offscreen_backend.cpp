#include "display/offscreen_backend.hpp"

#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::display {
namespace {

void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                   VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = ss;
    b.srcAccessMask = sa;
    b.dstStageMask = ds;
    b.dstAccessMask = da;
    b.oldLayout = from;
    b.newLayout = to;
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

OffscreenBackend::OffscreenBackend(yvo::vk::VulkanContext& ctx, int width, int height, std::uint32_t images)
    : ctx_(ctx) {
    extent_ = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
    VkDevice dev = ctx_.device();
    images_.resize(images);
    for (Image& im : images_) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;   // same format as the swapchain — matches the shader rgba8 qualifier
        ici.extent = {extent_.width, extent_.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        YVO_VK_CHECK(vkCreateImage(dev, &ici, nullptr, &im.image));

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(dev, im.image, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = ctx_.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &im.mem));
        YVO_VK_CHECK(vkBindImageMemory(dev, im.image, im.mem, 0));

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = im.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        YVO_VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &im.view));

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &im.fence));
    }
}

OffscreenBackend::~OffscreenBackend() { destroy(); }

void OffscreenBackend::destroy() {
    VkDevice dev = ctx_.device();
    if (dev == VK_NULL_HANDLE) return;
    for (Image& im : images_) {
        if (im.fence) vkDestroyFence(dev, im.fence, nullptr);
        if (im.view) vkDestroyImageView(dev, im.view, nullptr);
        if (im.image) vkDestroyImage(dev, im.image, nullptr);
        if (im.mem) vkFreeMemory(dev, im.mem, nullptr);
        im = Image{};
    }
    images_.clear();
}

bool OffscreenBackend::acquire(SwapchainFrame* out) {
    *out = SwapchainFrame{};
    if (images_.empty()) return false;
    const std::uint32_t i = next_ % static_cast<std::uint32_t>(images_.size());
    Image& im = images_[i];
    // Same rule as the swapchain: if it is still in use, **drop this frame's display without waiting.**
    if (im.in_flight) {
        if (vkGetFenceStatus(ctx_.device(), im.fence) != VK_SUCCESS) {
            ++dropped_;
            return false;
        }
        im.in_flight = false;
    }
    YVO_VK_CHECK(vkResetFences(ctx_.device(), 1, &im.fence));
    ++next_;

    out->valid = true;
    out->image_index = i;
    out->sync_index = i;
    out->swapchain_image = im.image;
    out->target_image = im.image;
    out->target_view = im.view;
    out->extent = extent_;
    out->wait_semaphore = VK_NULL_HANDLE;    // end_and_submit treats this as count 0
    out->signal_semaphore = VK_NULL_HANDLE;
    out->submit_fence = im.fence;
    out->present_id = 0;                     // no present_wait
    return true;
}

void OffscreenBackend::record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const {
    // The previous user is the **previous frame's compute write** to the same image (there is no presentation engine).
    // Every pixel is overwritten, so oldLayout is left as UNDEFINED.
    image_barrier(cb, f.target_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void OffscreenBackend::record_present_transition(VkCommandBuffer /*cb*/, const SwapchainFrame& /*f*/) const {
    // Not sent to the screen, so there is no PRESENT_SRC transition. The next acquire transition starts from UNDEFINED.
}

void OffscreenBackend::present(const SwapchainFrame& f) {
    images_[f.sync_index].in_flight = true;
    ++presented_;
}

} // namespace yvo::display
