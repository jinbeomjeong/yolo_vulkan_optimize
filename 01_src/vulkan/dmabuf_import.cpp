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
    if (e.memory != VK_NULL_HANDLE) vkFreeMemory(dev, e.memory, nullptr); // import한 fd도 함께 닫힌다
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
        if (victim == map_.end()) return; // 전부 사용 중
        release(victim->second);
        map_.erase(victim);
    }
}

namespace {

// 부록 B.12 강제 검증. range < min_range 면 셰이더가 실데이터를 잘라 읽는다 → 예외.
void check_min_range(const ImportedDmaBuf& e, std::size_t min_range) {
    if (min_range == 0 || e.range >= min_range) return;
    throw std::runtime_error("DmaBufImportCache: VkBuffer range " + std::to_string(e.range) +
                             " B < 요청 데이터 " + std::to_string(min_range) + " B (dma-buf " +
                             std::to_string(e.size) + " B). 할당 때 드라이버 헤드룸을 더해야 한다 "
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
            check_min_range(e, min_range);   // 캐시 적중에서도 검증한다
            ++hits_;
            e.last_use = ++clock_;
            return e;
        }
        // fd 번호가 다른 dma-buf에 재사용됨 → 오래된 import 해제
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

    // 1) 이 fd를 import할 수 있는 메모리 타입
    VkMemoryFdPropertiesKHR fdp{};
    fdp.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    YVO_VK_CHECK(ctx_.vkGetMemoryFdPropertiesKHR(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdp));

    // 2) 외부 메모리용 버퍼.
    //    V3DV는 스토리지 버퍼 요구 크기에 TMU 선읽기용 여유(256 B)를 더한다. import 메모리 크기는 dma-buf 크기로
    //    고정이므로, 요구 크기가 dma-buf를 넘으면 버퍼 크기를 그만큼 줄여 다시 만든다(끝 256 B는 셰이더에서 안 씀).
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

        // 3) import. 성공하면 dup fd의 소유권은 Vulkan으로 넘어간다.
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
