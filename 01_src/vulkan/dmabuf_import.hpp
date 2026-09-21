// dma-buf fd → VkBuffer(STORAGE_BUFFER) import and cache
//
// Import procedure:
//   1) dup(fd) (if vkAllocateMemory succeeds, the driver takes ownership of the dup'ed fd. On failure we close it)
//   2) check importable memoryTypeBits with vkGetMemoryFdPropertiesKHR(DMA_BUF)
//   3) create the VkBuffer with VkExternalMemoryBufferCreateInfo(DMA_BUF) attached, pick a memory type in the intersection with its requirements
//   4) vkAllocateMemory with VkImportMemoryFdInfoKHR(+ VkMemoryDedicatedAllocateInfo if needed) → bind
// Cache key: fd number. The decoder buffer pool's fds are fixed, but numbers can be reused when the decoder is reopened, so
//   fstat(st_dev, st_ino) and size are used to confirm it is the same dma-buf; if different, it is imported anew.
// Lifetime: cache entries hold a dma-buf reference (dup fd). Empty with clear() after closing the decoder.
//   Entries in use (referenced by commands pending on the GPU) are marked by the caller with pin/unpin and are excluded from eviction and clear.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <sys/types.h>
#include <vulkan/vulkan.h>

namespace yvo::vk {

class VulkanContext;

struct ImportedDmaBuf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;     // dma-buf size (= import allocationSize)
    VkDeviceSize range = 0;    // VkBuffer size. May be smaller than size because of the driver's slack requirement (V3DV: -256 B)
    dev_t st_dev = 0;
    ino_t st_ino = 0;
    int pins = 0;              // number of in-use references
    std::uint64_t last_use = 0;
};

class DmaBufImportCache {
public:
    // max_entries: cache limit (comfortably more than decoder pool size × number of planes). When exceeded, the oldest entry not in use is evicted.
    explicit DmaBufImportCache(VulkanContext& ctx, std::size_t max_entries = 64);
    ~DmaBufImportCache();
    DmaBufImportCache(const DmaBufImportCache&) = delete;
    DmaBufImportCache& operator=(const DmaBufImportCache&) = delete;

    // Entry importing fd (size bytes). Reused if already cached. Throws on failure.
    //
    // If min_range > 0, **enforced check**: throws if the resulting range is smaller than min_range (plan Appendix B.12).
    // V3DV adds slack (+256 B) to the storage buffer requirement size and shrinks the VkBuffer range by that much, so
    // if the dma-buf exactly fits the real data size, the shader cannot read the last bytes. Silently letting it pass
    // would make the shader read wrong data, so we stop here. The caller passes the real data byte count.
    // To prevent it, headroom must be added at allocation time (vulkan/vk_storage_headroom.hpp → DmaHeapConfig::device_headroom).
    ImportedDmaBuf& get(int fd, std::size_t size, std::size_t min_range = 0);
    void pin(ImportedDmaBuf& e) { ++e.pins; }
    void unpin(ImportedDmaBuf& e) { --e.pins; }

    // Release all entries not in use (call when closing or reopening the decoder)
    void clear();

    std::size_t entries() const { return map_.size(); }
    std::uint64_t imports() const { return imports_; }   // actual import count
    std::uint64_t hits() const { return hits_; }         // cache hit count

private:
    void release(ImportedDmaBuf& e);
    void evict_if_needed();

    VulkanContext& ctx_;
    std::size_t max_entries_;
    std::unordered_map<int, ImportedDmaBuf> map_;
    std::uint64_t clock_ = 0;
    std::uint64_t imports_ = 0;
    std::uint64_t hits_ = 0;
};

} // namespace yvo::vk
