#include "vulkan/dmabuf_export.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::vk {

namespace {

std::size_t align_up(std::size_t v, std::size_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

} // namespace

DmaBufExporter::DmaBufExporter(VulkanContext& ctx, const DmaBufExporterConfig& cfg) : ctx_(ctx), cfg_(cfg) {
    if (!ctx_.dmabuf_export_supported())
        throw std::runtime_error("DmaBufExporter: dma-buf export not supported by device");
    if (ctx_.vkGetMemoryFdKHR == nullptr)
        throw std::runtime_error("DmaBufExporter: vkGetMemoryFdKHR entry point missing");
    if (cfg_.align == 0) cfg_.align = 1;
}

DmaBufExporter::~DmaBufExporter() { destroy(); }

const ExportedDmaBuf& DmaBufExporter::at(int index) const { return bufs_.at(static_cast<std::size_t>(index)); }

int DmaBufExporter::dup_fd(int index) const {
    const ExportedDmaBuf& e = at(index);
    const int fd = ::dup(e.fd);
    if (fd < 0) throw std::runtime_error("DmaBufExporter::dup_fd: dup() failed");
    return fd;
}

ExportedDmaBuf DmaBufExporter::allocate_once(std::size_t requested, VkDeviceSize min_alloc) {
    VkDevice dev = ctx_.device();
    ExportedDmaBuf e;
    e.requested_size = requested;

    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext;
    bci.size = min_alloc;
    bci.usage = cfg_.usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &e.buffer));

    try {
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, e.buffer, &req);
        const std::uint32_t type =
            ctx_.find_memory_type(req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX) throw std::runtime_error("DmaBufExporter: no HOST_VISIBLE|HOST_COHERENT memory type");

        // One allocation = one buffer = one dma-buf. The buffer is bound at offset 0, so dma-buf start = buffer start.
        //
        // ⚠ **If VkMemoryDedicatedAllocateInfo is attached, allocationSize must equal req.size exactly**
        //   (VUID-VkMemoryDedicatedAllocateInfo-buffer-02965). But V3DV reports the buffer memory requirement size as
        //   buffer size + 256 B (the flip side of the import-side -256 B rule), so req.size is not a multiple of align.
        //   → **To round the allocation size up to a multiple of align, a dedicated allocation cannot be used.** Fortunately this device does not
        //   require dedicated allocation for dma-buf export (dmabuf_export_dedicated_only() == false, measured). So we go
        //   dedicated only when the driver requires it, and then allocationSize cannot be raised, so alloc_size may not
        //   be a multiple of align — that fact is exposed through the dedicated flag (the caller decides).
        const bool need_dedicated = ctx_.dmabuf_export_dedicated_only();
        VkMemoryDedicatedAllocateInfo ded{};
        ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        ded.buffer = e.buffer;
        VkExportMemoryAllocateInfo exp{};
        exp.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        if (need_dedicated) exp.pNext = &ded;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &exp;
        const VkDeviceSize need = req.size > min_alloc ? req.size : min_alloc;
        mai.allocationSize = need_dedicated ? req.size : align_up(need, cfg_.align);
        mai.memoryTypeIndex = type;
        e.alloc_size = mai.allocationSize;
        e.dedicated = need_dedicated;
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &e.memory));
        YVO_VK_CHECK(vkBindBufferMemory(dev, e.buffer, e.memory, 0));

        if (cfg_.map) {
            void* p = nullptr;
            YVO_VK_CHECK(vkMapMemory(dev, e.memory, 0, VK_WHOLE_SIZE, 0, &p));
            e.host_ptr = static_cast<std::uint8_t*>(p);
            const auto addr = reinterpret_cast<std::uintptr_t>(p);
            const std::size_t rem = static_cast<std::size_t>(addr % cfg_.align);
            e.align_offset = rem == 0 ? 0 : cfg_.align - rem;
        }

        VkMemoryGetFdInfoKHR gi{};
        gi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        gi.memory = e.memory;
        gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        YVO_VK_CHECK(ctx_.vkGetMemoryFdKHR(dev, &gi, &e.fd));
        const off_t sz = ::lseek(e.fd, 0, SEEK_END);
        e.dmabuf_size = sz > 0 ? static_cast<std::size_t>(sz) : 0;
    } catch (...) {
        release(e);
        throw;
    }
    return e;
}

int DmaBufExporter::allocate(std::size_t size) {
    if (size == 0) throw std::invalid_argument("DmaBufExporter::allocate: size must be > 0");
    const VkDeviceSize want = align_up(size, cfg_.align);

    ExportedDmaBuf e = allocate_once(size, want);
    // If the mapping start is not a multiple of align, reallocate once with align bytes of slack added.
    // (This is the only way to guarantee that size bytes are writable from an aligned start.)
    if (cfg_.map && e.align_offset != 0) {
        if (cfg_.verbose)
            std::fprintf(stderr, "[DmaBufExporter] mapped %p not %zu-aligned — reallocating with %zu B of slack\n",
                         static_cast<void*>(e.host_ptr), cfg_.align, cfg_.align);
        release(e);
        e = allocate_once(size, want + cfg_.align);
        if (e.align_offset + size > e.alloc_size) {
            release(e);
            throw std::runtime_error("DmaBufExporter: cannot obtain an aligned host mapping");
        }
    }
    if (cfg_.verbose)
        std::fprintf(stderr,
                     "[DmaBufExporter] #%zu requested=%zu alloc=%llu dmabuf=%zu fd=%d host=%p aligned=%p(+%zu)\n",
                     bufs_.size(), e.requested_size, static_cast<unsigned long long>(e.alloc_size), e.dmabuf_size,
                     e.fd, static_cast<void*>(e.host_ptr), static_cast<void*>(e.aligned_host_ptr()), e.align_offset);
    bufs_.push_back(e);
    return static_cast<int>(bufs_.size()) - 1;
}

void DmaBufExporter::release(ExportedDmaBuf& e) {
    VkDevice dev = ctx_.device();
    if (e.fd >= 0) ::close(e.fd);
    if (e.host_ptr && e.memory) vkUnmapMemory(dev, e.memory);
    if (e.buffer) vkDestroyBuffer(dev, e.buffer, nullptr);
    if (e.memory) vkFreeMemory(dev, e.memory, nullptr);
    e = ExportedDmaBuf{};
}

void DmaBufExporter::destroy() {
    if (ctx_.device() == VK_NULL_HANDLE) return;
    for (ExportedDmaBuf& e : bufs_) release(e);
    bufs_.clear();
}

} // namespace yvo::vk
