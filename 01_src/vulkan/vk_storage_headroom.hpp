// STORAGE_BUFFER headroom query for dma-buf import (header-only) — plan Appendix B.12
//
// Problem: V3DV adds slack for TMU prefetch to the storage buffer memory requirement size (measured +256 B, size-independent).
// The import memory size is fixed to the dma-buf size, so if the dma-buf exactly fits the requested size,
// `DmaBufImportCache::get()` copes by **shrinking** the VkBuffer range by that amount → the shader cannot read the last
// bytes. In P0 (Appendix B.17), `conv71_178` (409,600 = 16384×25, slack 0) was exactly in this state:
// `range 409,344 < hw 409,600`.
//
// Solution: when allocating from the dma-heap, allocate **with the headroom added in advance**.
//   1) create a dummy VkBuffer (size = requested, usage = STORAGE_BUFFER,
//                          pNext = VkExternalMemoryBufferCreateInfo{DMA_BUF})
//   2) vkGetBufferMemoryRequirements2() → req.size
//   3) headroom = req.size - requested          (V3D measured: always 256, size-independent)
//   4) destroy the dummy
// → when the caller puts it into `yvo::DmaHeapConfig::device_headroom`,
//   `DmaHeapPool::allocate()` allocates `align_up(requested + device_headroom, align)`.
//
// **Do not hard-code the constant 256** — it is driver-dependent. Query results are memoized per size.
//
// Does over-allocation violate the spec: `VUID-VkMemoryDedicatedAllocateInfo-buffer-02965`
// requires "if buffer is not VK_NULL_HANDLE, allocationSize must equal that buffer's VkMemoryRequirements::size",
// but this applies **only when a dedicated allocation (VkMemoryDedicatedAllocateInfo) is attached**.
// In P0 measurements V3D reported `requiresDedicatedAllocation = prefersDedicatedAllocation = 0` for 6/6, so
// `dmabuf_import.cpp` does not attach that structure → the over-allocation does not trigger it.
// (validusage.json, Vulkan 1.4.309 / commit 43d153dd)
#pragma once

#include <cstddef>
#include <unordered_map>

#include <vulkan/vulkan.h>

#include "vulkan/vk_check.hpp"

namespace yvo::vk {

// Queries the memory requirements of an external-memory (dma-buf) STORAGE_BUFFER of requested bytes. Does not allocate.
// If ded is non-null, VkMemoryDedicatedRequirements is filled in as well.
inline VkMemoryRequirements query_dmabuf_storage_requirements(VkDevice dev, VkDeviceSize requested,
                                                              VkMemoryDedicatedRequirements* ded = nullptr) {
    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext;
    bci.size = requested;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer dummy = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &dummy));

    VkMemoryDedicatedRequirements local{};
    local.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    VkMemoryRequirements2 req2{};
    req2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    req2.pNext = &local;
    VkBufferMemoryRequirementsInfo2 ri{};
    ri.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    ri.buffer = dummy;
    vkGetBufferMemoryRequirements2(dev, &ri, &req2);
    vkDestroyBuffer(dev, dummy, nullptr);

    if (ded) *ded = local;
    return req2.memoryRequirements;
}

// Per-size headroom cache. Never asks for the same size twice (Appendix B.12: "do not re-query per tensor").
class StorageHeadroomQuery {
public:
    explicit StorageHeadroomQuery(VkDevice dev) : dev_(dev) {}

    // Number of extra bytes a dma-buf holding requested bytes must have.
    // 0 if the driver requires no slack.
    std::size_t headroom(std::size_t requested) {
        auto it = memo_.find(requested);
        if (it != memo_.end()) return it->second;
        const VkMemoryRequirements req = query_dmabuf_storage_requirements(dev_, requested);
        const std::size_t h = req.size > requested ? static_cast<std::size_t>(req.size - requested) : 0;
        last_alignment_ = req.alignment;
        memo_.emplace(requested, h);
        return h;
    }

    // Maximum headroom across several sizes. `DmaHeapConfig::device_headroom` is one value per pool, so
    // a pool holding several sizes gets the maximum. **Over-allocation is safe** (see the VUID-02965 explanation in the comment above).
    template <typename It>
    std::size_t max_headroom(It first, It last) {
        std::size_t m = 0;
        for (It i = first; i != last; ++i) {
            const std::size_t h = headroom(static_cast<std::size_t>(*i));
            if (h > m) m = h;
        }
        return m;
    }

    VkDeviceSize last_alignment() const { return last_alignment_; }
    std::size_t queries() const { return memo_.size(); }

private:
    VkDevice dev_;
    std::unordered_map<std::size_t, std::size_t> memo_;
    VkDeviceSize last_alignment_ = 0;
};

} // namespace yvo::vk
