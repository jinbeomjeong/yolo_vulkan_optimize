#include "vulkan/dmabuf_import.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::vk {

DmaBufImportCache::DmaBufImportCache(VulkanContext& ctx, std::size_t max_entries)
    : ctx_(ctx), max_entries_(max_entries) {}

DmaBufImportCache::~DmaBufImportCache() {
    for (auto& [fd, e] : map_) release(e);
    map_.clear();
}

void DmaBufImportCache::release(ImportedDmaBuf& e) {
    VkDevice dev = ctx_.device();
    if (e.buffer != VK_NULL_HANDLE) vkDestroyBuffer(dev, e.buffer, nullptr);
    if (e.memory != VK_NULL_HANDLE) vkFreeMemory(dev, e.memory, nullptr); // the imported fd is closed as well
    e.buffer = VK_NULL_HANDLE;
    e.memory = VK_NULL_HANDLE;
}

void DmaBufImportCache::clear() {
    for (auto it = map_.begin(); it != map_.end();) {
        if (it->second.pins == 0) {
            release(it->second);
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

void DmaBufImportCache::evict_if_needed() {
    while (map_.size() >= max_entries_) {
        auto victim = map_.end();
        for (auto it = map_.begin(); it != map_.end(); ++it)
            if (it->second.pins == 0 && (victim == map_.end() || it->second.last_use < victim->second.last_use))
                victim = it;
        if (victim == map_.end()) return; // all in use
        release(victim->second);
        map_.erase(victim);
    }
}

namespace {

// Appendix B.12 enforced check. If range < min_range the shader reads truncated real data → exception.
void check_min_range(const ImportedDmaBuf& e, std::size_t min_range) {
    if (min_range == 0 || e.range >= min_range) return;
    throw std::runtime_error("DmaBufImportCache: VkBuffer range " + std::to_string(e.range) +
                             " B < requested data " + std::to_string(min_range) + " B (dma-buf " +
                             std::to_string(e.size) + " B). Driver headroom must be added at allocation time "
                             "(vulkan/vk_storage_headroom.hpp → DmaHeapConfig::device_headroom)");
}

} // namespace

ImportedDmaBuf& DmaBufImportCache::get(int fd, std::size_t size, std::size_t min_range) {
    if (fd < 0 || size == 0) throw std::invalid_argument("DmaBufImportCache::get: invalid fd/size");
    struct stat st{};
    if (fstat(fd, &st) != 0)
        throw std::runtime_error(std::string("DmaBufImportCache: fstat failed: ") + std::strerror(errno));

    auto it = map_.find(fd);
    if (it != map_.end()) {
        ImportedDmaBuf& e = it->second;
        if (e.st_dev == st.st_dev && e.st_ino == st.st_ino && e.size == size) {
            check_min_range(e, min_range);   // validate on a cache hit too
            ++hits_;
            e.last_use = ++clock_;
            return e;
        }
        // fd number reused for a different dma-buf → release the stale import
        if (e.pins != 0)
            throw std::runtime_error("DmaBufImportCache: fd reused while previous import is still in use");
        release(e);
        map_.erase(it);
    }
    evict_if_needed();

    VkDevice dev = ctx_.device();
    ImportedDmaBuf e;
    e.size = size;
    e.st_dev = st.st_dev;
    e.st_ino = st.st_ino;

    // 1) memory type that can import this fd
    VkMemoryFdPropertiesKHR fdp{};
    fdp.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    YVO_VK_CHECK(ctx_.vkGetMemoryFdPropertiesKHR(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdp));

    // 2) buffer for external memory.
    //    V3DV adds slack for TMU prefetch (256 B) to the storage buffer requirement size. The import memory size is fixed
    //    to the dma-buf size, so if the requirement exceeds the dma-buf, recreate the buffer smaller by that amount (the last 256 B are unused by the shader).
    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    try {
        VkMemoryDedicatedRequirements dreq{};
        dreq.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
        VkMemoryRequirements2 req2{};
        req2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        for (int attempt = 0;; ++attempt) {
            YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &e.buffer));
            req2.pNext = &dreq;
            VkBufferMemoryRequirementsInfo2 ri{};
            ri.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
            ri.buffer = e.buffer;
            vkGetBufferMemoryRequirements2(dev, &ri, &req2);
            if (req2.memoryRequirements.size <= size) break;
            const VkDeviceSize excess = req2.memoryRequirements.size - size;
            vkDestroyBuffer(dev, e.buffer, nullptr);
            e.buffer = VK_NULL_HANDLE;
            if (attempt >= 2 || excess >= bci.size)
                throw std::runtime_error("DmaBufImportCache: buffer requirement " +
                                         std::to_string(req2.memoryRequirements.size) + " B exceeds dma-buf size " +
                                         std::to_string(size) + " B");
            bci.size = (bci.size - excess) & ~VkDeviceSize{3};
        }
        e.range = bci.size;
        const VkMemoryRequirements& req = req2.memoryRequirements;
        const std::uint32_t type = ctx_.find_memory_type(req.memoryTypeBits & fdp.memoryTypeBits, 0,
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX)
            throw std::runtime_error("DmaBufImportCache: no compatible memory type (buffer bits 0x" +
                                     std::to_string(req.memoryTypeBits) + ", fd bits 0x" +
                                     std::to_string(fdp.memoryTypeBits) + ")");

        // 3) import. On success, ownership of the dup fd passes to Vulkan.
        const int dfd = dup(fd);
        if (dfd < 0) throw std::runtime_error(std::string("DmaBufImportCache: dup failed: ") + std::strerror(errno));
        VkImportMemoryFdInfoKHR imp{};
        imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        imp.fd = dfd;
        VkMemoryDedicatedAllocateInfo ded{};
        ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        ded.buffer = e.buffer;
        if (dreq.requiresDedicatedAllocation || dreq.prefersDedicatedAllocation || ctx_.dmabuf_import_dedicated_only())
            imp.pNext = &ded;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &imp;
        mai.allocationSize = size;
        mai.memoryTypeIndex = type;
        const VkResult r = vkAllocateMemory(dev, &mai, nullptr, &e.memory);
        if (r != VK_SUCCESS) {
            close(dfd);
            YVO_VK_CHECK(r);
        }
        YVO_VK_CHECK(vkBindBufferMemory(dev, e.buffer, e.memory, 0));
        check_min_range(e, min_range);
    } catch (...) {
        release(e);
        throw;
    }
    ++imports_;
    e.last_use = ++clock_;
    return map_.emplace(fd, e).first->second;
}

} // namespace yvo::vk
