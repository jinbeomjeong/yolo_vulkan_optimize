// Reusable helper that exports VkBuffer + VkDeviceMemory as a dma-buf (used by postprocessing track P2)
//
// Extracts the output slot allocation + export_dmabuf_fd() path of sand_letterbox.cpp into a class
// (the original is the tuned 2.13 ms path, so it is left untouched — this is newly written here).
//
// Allocation procedure:
//   1) Create the VkBuffer with VkExternalMemoryBufferCreateInfo(DMA_BUF) attached
//   2) vkAllocateMemory with VkExportMemoryAllocateInfo(DMA_BUF) (one allocation = one buffer = one dma-buf.
//      The buffer is bound at offset 0, so the dma-buf start address is the buffer start.
//      VkMemoryDedicatedAllocateInfo is attached only when the driver requires it — see the alignment comment in the .cpp)
//   3) vkBindBufferMemory → vkMapMemory (persistent mapping) → export the fd with vkGetMemoryFdKHR
//
// ── Alignment (align) ────────────────────────────────────────────────────────────────────────
// **HailoRT read_async requires 16384 B alignment** (measured on the postprocessing track: with 4096, vdma_stream.cpp:379
// emits an "unaligned buffer causes performance degradation" warning per frame x stream).
// So the default align is 16384, and both of the following are satisfied:
//   · **Allocation size**: align_up(VkMemoryRequirements::size, align). V3DV requires buffer size + 256 B, so
//     with a dedicated allocation this value cannot be raised (the spec enforces allocationSize == req.size) → a dedicated allocation is
//     used only when the driver requires it. V3D does not require it (measured).
//   · **Start address**: on the dma-buf path the offset is 0, so it is aligned automatically. On the host pointer path the address returned by
//     vkMapMemory has mmap alignment (usually 4096, larger for large allocations) and is not guaranteed, so if it is misaligned
//     **it reallocates once with align bytes of extra room** and exposes the aligned start via aligned_host_ptr().
//     (The first attempt almost always succeeds, so the usual waste is 0.)
//
// ── Why there are two sizes ───────────────────────────────────────────────────────────────────
// The size V3DV requires via vkGetBufferMemoryRequirements may differ from the actual dma-buf size
// (for the same reason, on the import side dmabuf_import.hpp shrinks the range by -256 B).
// And whether to pass the **requested size** or the **actual dma-buf size** to HailoRT's dma_map_dmabuf(fd, size, D2H) / read_async(fd, size)
// is not yet settled, so both values are exposed so the caller can try either:
//   requested_size : number of bytes the caller passed to allocate() (tensor size)
//   alloc_size     : size passed to vkAllocateMemory (multiple of align)
//   dmabuf_size    : dma-buf object size measured with lseek(fd, 0, SEEK_END)
// Measured on V3D (yvo_display_probe --probe export): alloc_size == dmabuf_size and both are multiples of 16384, but
// **always larger than requested_size** (e.g. 1,228,800 requested → 1,245,184). This held for all 7 sizes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>
#include "core/hailo_align.hpp"

namespace yvo::vk {

class VulkanContext;

// Alignment required by HailoRT read_async. Value measured by the postprocessing track.
using yvo::kHailoAsyncAlign;  // the only definition is in core/hailo_align.hpp

struct ExportedDmaBuf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    int fd = -1;                        // owned by DmaBufExporter. Use dup_fd() when handing it over.
    std::size_t requested_size = 0;     // number of bytes requested by the caller
    VkDeviceSize alloc_size = 0;        // actual vkAllocateMemory size (multiple of align)
    std::size_t dmabuf_size = 0;        // dma-buf size measured with lseek (0 means measurement failed)
    std::uint8_t* host_ptr = nullptr;   // persistent mapping (mapping start)
    std::size_t align_offset = 0;       // bytes from host_ptr to the align boundary (usually 0)
    // Whether a dedicated allocation (VkMemoryDedicatedAllocateInfo) was used. If true, the spec enforces allocationSize == req.size,
    // so alloc_size may not be a multiple of align (V3D is false, so this case does not occur).
    bool dedicated = false;
    // align-aligned start address. Same as host_ptr if align_offset is 0.
    std::uint8_t* aligned_host_ptr() const { return host_ptr + align_offset; }
};

struct DmaBufExporterConfig {
    // Buffer usage. The postprocessing track uses STORAGE_BUFFER (read by the shader).
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    std::size_t align = kHailoAsyncAlign;   // alignment of allocation size and start address
    bool map = true;                        // persistent mapping (host_ptr). nullptr if false
    bool verbose = false;
};

class DmaBufExporter {
public:
    explicit DmaBufExporter(VulkanContext& ctx, const DmaBufExporterConfig& cfg = {});
    ~DmaBufExporter();
    DmaBufExporter(const DmaBufExporter&) = delete;
    DmaBufExporter& operator=(const DmaBufExporter&) = delete;

    // Allocates a new buffer holding size bytes and exports it as a dma-buf. Returns the index. Throws on failure.
    int allocate(std::size_t size);

    std::size_t count() const { return bufs_.size(); }
    const ExportedDmaBuf& at(int index) const;
    // A new fd to hand to the caller (the caller closes it). This object keeps holding the original fd.
    int dup_fd(int index) const;

    const DmaBufExporterConfig& config() const { return cfg_; }

private:
    void destroy();
    // Tries one allocation. Also checks the mapping address alignment and fills align_offset.
    ExportedDmaBuf allocate_once(std::size_t requested, VkDeviceSize min_alloc);
    void release(ExportedDmaBuf& e);

    VulkanContext& ctx_;
    DmaBufExporterConfig cfg_;
    std::vector<ExportedDmaBuf> bufs_;
};

} // namespace yvo::vk
