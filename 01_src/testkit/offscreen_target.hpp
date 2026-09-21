#pragma once
// Offscreen render target — STORAGE image + (optional) host readback buffer + fence.
// Used for accuracy checks that compare compute shader output **bit for bit** against the CPU reference implementation.
//
// Used by: tools/display/display_probe.cpp (P7 overlay · P8 convert),
//          testkit/display_convert_check.hpp (→ 05_tests/test_display_convert.cpp),
//          05_tests/test_overlay.cpp.
//
// Before 2026-09-21, display_probe's OffscreenTarget and test_overlay's OverlayHarness
// each had their own image/buffer/barrier/readback code (no synchronization mechanism). The readback block
// (GENERAL → TRANSFER_SRC barrier, image→buffer copy, host visibility barrier) was copied in three places.
//
// Header-only. Uses no display-path symbols other than VulkanContext.
#include <cstdint>

#include <vulkan/vulkan.h>

#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::display {

inline void barrier_image(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
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

// R8G8B8A8_UNORM, STORAGE|TRANSFER_SRC(+extra_usage) image. If readback=true, a host buffer is kept.
// Adding TRANSFER_DST via extra_usage lets vkCmdClearColorImage fill the background.
struct OffscreenTarget {
    yvo::vk::VulkanContext& ctx;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory buf_mem = VK_NULL_HANDLE;
    std::uint8_t* host = nullptr;
    VkFence fence = VK_NULL_HANDLE;
    std::uint32_t w = 0, h = 0;

    OffscreenTarget(yvo::vk::VulkanContext& c, std::uint32_t width, std::uint32_t height, bool readback,
                    VkImageUsageFlags extra_usage = 0)
        : ctx(c), w(width), h(height) {
        VkDevice dev = ctx.device();
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | extra_usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        YVO_VK_CHECK(vkCreateImage(dev, &ici, nullptr, &image));
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(dev, image, &req);
        std::uint32_t type = ctx.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        YVO_VK_CHECK(vkBindImageMemory(dev, image, mem, 0));
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        YVO_VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &view));

        if (readback) {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = static_cast<VkDeviceSize>(w) * h * 4;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
            vkGetBufferMemoryRequirements(dev, buf, &req);
            type = ctx.find_memory_type(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = type;
            YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &buf_mem));
            YVO_VK_CHECK(vkBindBufferMemory(dev, buf, buf_mem, 0));
            void* p = nullptr;
            YVO_VK_CHECK(vkMapMemory(dev, buf_mem, 0, VK_WHOLE_SIZE, 0, &p));
            host = static_cast<std::uint8_t*>(p);
        }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &fence));
    }
    ~OffscreenTarget() {
        VkDevice dev = ctx.device();
        if (dev == VK_NULL_HANDLE) return;
        if (fence) vkDestroyFence(dev, fence, nullptr);
        if (host) vkUnmapMemory(dev, buf_mem);
        if (buf) vkDestroyBuffer(dev, buf, nullptr);
        if (buf_mem) vkFreeMemory(dev, buf_mem, nullptr);
        if (view) vkDestroyImageView(dev, view, nullptr);
        if (image) vkDestroyImage(dev, image, nullptr);
        if (mem) vkFreeMemory(dev, mem, nullptr);
    }
    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    // Call after compute has written to the image in GENERAL layout. After submit and fence wait, the host can read RGBA.
    //   GENERAL → TRANSFER_SRC barrier · image→buffer copy · host visibility barrier.
    void record_readback(VkCommandBuffer cb) const {
        barrier_image(cb, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);
        VkBufferMemoryBarrier2 bb{};
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        bb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        bb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = buf;
        bb.size = VK_WHOLE_SIZE;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &bb;
        vkCmdPipelineBarrier2(cb, &di);
    }
};

}  // namespace yvo::display
