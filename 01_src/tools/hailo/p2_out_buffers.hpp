// P2 (plan §6.2) three raw output buffer pools: malloc / dma-heap / Vulkan export dma-buf
//
// Header used by yvo_hailo_raw_probe --alloc {malloc,heap,vulkan}. It is probe-only, so it is not put into the yvo_hailo
// library — doing so would make yvo_hailo depend on yvo_vulkan and break plan §5.2's
// **link-level isolation** (the link graph of test_preproc / yvo_preproc_bench would change).
//
// ── Differences among the three ───────────────────────────────────────────────────────────────
//   malloc : std::aligned_alloc(16384). No fd, so the read_async(void*, size) path. This is the baseline.
//   heap   : /dev/dma_heap/system (DMA_HEAP_IOCTL_ALLOC). **Cached** CPU mapping, so CPU reads are cheap.
//            In exchange, PCIe DMA write ↔ CPU cache coherency must be managed manually with DMA_BUF_IOCTL_SYNC.
//   vulkan : VulkanContext + DmaBufExporter. V3D memory, so shaders can read it directly (plan §4.2 adopted option).
//            V3D has only one memory type, DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT, with no HOST_CACHED
//            → **CPU reads are expensive** (§2). So no sync ioctl is needed, but a full 1.22 MB CPU read is slow.
//
// ── Size contract ─────────────────────────────────────────────────────────────────────────────
// HailoRT read_async requires 16384 B alignment (appendix A.4). So both heap and vulkan
// request align_up(hw_frame_size, 16384) → **actual dma-buf size > hw_frame_size**, and
// it becomes unclear which size to pass to dma_map_dmabuf / read_async. The probe judges those 4 combinations.
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/dma_heap_alloc.hpp"
#include "vulkan/dmabuf_export.hpp"
#include "vulkan/vulkan_context.hpp"
#include "core/hailo_align.hpp"

namespace yvo::p2 {

using yvo::kHailoAsyncAlign;  // Defined only in core/hailo_align.hpp

inline std::size_t align_up(std::size_t v, std::size_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

// Value type pointing to one buffer. The pool holds its lifetime.
struct OutBufRef {
    std::uint8_t* ptr = nullptr;    // CPU-readable start address (16384-aligned)
    int fd = -1;                    // dma-buf fd. -1 in malloc mode
    std::size_t dmabuf_size = 0;    // dma-buf size measured with lseek. 0 if fd < 0
    bool needs_sync = false;        // If true, DMA_BUF_IOCTL_SYNC is needed before/after CPU reads (heap)
};

// Opens/closes a CPU read section. The ioctl is actually issued only for heap.
// The actual work is delegated to dma_sync_* in core/dma_heap_alloc.hpp (appendix B.11 unification).
// **strict=true** — the previous implementation discarded the return value (appendix B.10 R5).
inline void sync_cpu_read_begin(const OutBufRef& b) {
    if (!b.needs_sync || b.fd < 0) return;
    yvo::dma_sync_begin(b.fd, yvo::DmaSyncDir::Read, /*strict=*/true);
}
inline void sync_cpu_read_end(const OutBufRef& b) {
    if (!b.needs_sync || b.fd < 0) return;
    yvo::dma_sync_end(b.fd, yvo::DmaSyncDir::Read, /*strict=*/true);
}

class OutBufPool {
public:
    virtual ~OutBufPool() = default;
    // Allocates one more buffer able to hold hw_frame_size bytes and returns it. Throws on failure.
    virtual OutBufRef acquire(std::size_t hw_frame_size) = 0;
    virtual const char* name() const = 0;
    // Number of buffers allocated so far
    virtual std::size_t count() const = 0;
};

// ── malloc ───────────────────────────────────────────────────────────────────────────────────
class MallocOutPool final : public OutBufPool {
public:
    OutBufRef acquire(std::size_t hw_frame_size) override {
        const std::size_t rounded = align_up(hw_frame_size, kHailoAsyncAlign);
        void* p = std::aligned_alloc(kHailoAsyncAlign, rounded);
        if (!p) throw std::bad_alloc();
        std::memset(p, 0, rounded);
        bufs_.emplace_back(static_cast<std::uint8_t*>(p), [](std::uint8_t* q) { std::free(q); });
        OutBufRef r;
        r.ptr = bufs_.back().get();
        return r;
    }
    const char* name() const override { return "malloc"; }
    std::size_t count() const override { return bufs_.size(); }

private:
    std::vector<std::shared_ptr<std::uint8_t>> bufs_;
};

// ── /dev/dma_heap (= vidbuf_cached → system symlink) ─────────────────────────────────────────
// The device node has video-group permissions and user pi is in the video group (measured).
//
// **A thin adapter over `yvo::DmaHeapPool` (core/dma_heap_alloc.hpp)** (appendix B.11 unification).
// Three differences from the former hand-written implementation:
//   · `memset` is **wrapped in a WRITE sync** — the defect in appendix B.6(c) is fixed automatically here
//   · A 16384 misalignment **does not throw** — it is fixed by reserving and remapping with MAP_FIXED
//   · All return values of `DMA_HEAP_IOCTL_ALLOC` / `DMA_BUF_IOCTL_SYNC` are checked (B.10 R5)
// `device_headroom` is 0 — buffers from this pool go only to HailoRT and are not imported into Vulkan.
// (Paths that import must query the headroom and add it — appendix B.12, vulkan/vk_storage_headroom.hpp)
class HeapOutPool final : public OutBufPool {
public:
    explicit HeapOutPool(const std::string& heap = "/dev/dma_heap/system", std::size_t device_headroom = 0)
        : pool_(make_cfg(heap, device_headroom)) {}

    OutBufRef acquire(std::size_t hw_frame_size) override {
        const int i = pool_.allocate(hw_frame_size);
        const yvo::DmaHeapBuf& b = pool_.at(i);
        OutBufRef r;
        r.ptr = b.ptr;
        r.fd = b.fd;              // Ownership stays with DmaHeapPool. This fd is passed to HailoRT as is.
        r.dmabuf_size = b.dmabuf_size;
        r.needs_sync = true;
        return r;
    }
    const char* name() const override { return "heap"; }
    std::size_t count() const override { return pool_.count(); }

private:
    static yvo::DmaHeapConfig make_cfg(const std::string& heap, std::size_t device_headroom) {
        yvo::DmaHeapConfig c;
        c.heaps = {heap};
        c.align = kHailoAsyncAlign;
        c.device_headroom = device_headroom;
        c.map = true;
        c.strict_sync = true;
        c.zero_init = true;
        c.name = "yvo_p2_out";
        return c;
    }
    yvo::DmaHeapPool pool_;
};

// ── Vulkan export dma-buf (plan §4.2 adopted option) ──────────────────────────────────────────
// VulkanContext is created with present=off (unrelated to display; with off the extension list is the same as today — §5.2).
class VulkanOutPool final : public OutBufPool {
public:
    VulkanOutPool(bool verbose = false) {
        yvo::vk::VulkanContextOptions co;
        co.verbose = verbose;
        co.enable_present = false;
        ctx_ = std::make_unique<yvo::vk::VulkanContext>(co);
        yvo::vk::DmaBufExporterConfig ec;
        ec.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;   // Read by yolo_decode.comp in the next stage
        ec.align = kHailoAsyncAlign;
        ec.map = true;
        ec.verbose = verbose;
        exp_ = std::make_unique<yvo::vk::DmaBufExporter>(*ctx_, ec);
    }

    OutBufRef acquire(std::size_t hw_frame_size) override {
        const int idx = exp_->allocate(hw_frame_size);
        const auto& e = exp_->at(idx);
        std::memset(e.aligned_host_ptr(), 0, hw_frame_size);
        OutBufRef r;
        r.ptr = e.aligned_host_ptr();
        r.fd = e.fd;             // Ownership stays with DmaBufExporter. This fd is passed to HailoRT as is.
        r.dmabuf_size = e.dmabuf_size;
        r.needs_sync = false;    // HOST_COHERENT
        return r;
    }
    const char* name() const override { return "vulkan"; }
    std::size_t count() const override { return exp_->count(); }

    const yvo::vk::VulkanContext& ctx() const { return *ctx_; }

private:
    std::unique_ptr<yvo::vk::VulkanContext> ctx_;
    std::unique_ptr<yvo::vk::DmaBufExporter> exp_;
};

} // namespace yvo::p2
