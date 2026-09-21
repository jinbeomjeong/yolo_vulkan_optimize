// dma-buf CPU read mapping (header-only). Issues DMA_BUF_SYNC_START|READ on construction and END|READ on destruction to keep CPU caches coherent.
// Used when the CPU reference path (tests, benchmarks) reads a decoder dma-buf.
//
// **A separate purpose: mapping someone else's fd read-only** — it does not allocate. The side that allocates directly
// is DmaHeapPool in core/dma_heap_alloc.hpp (Appendix B.11 consolidation).
// The two ioctls are delegated to that header's dma_sync_begin/end. They are called with strict=false,
// **keeping the previous behavior (ioctl failures ignored) unchanged** — this class's public API and behavior did not change.
#pragma once

#include <cstddef>
#include <cstdint>

#include <sys/mman.h>

#include "core/dma_heap_alloc.hpp"

namespace yvo {

class DmaBufReadMap {
public:
    DmaBufReadMap(int fd, std::size_t size) : fd_(fd), size_(size) {
        void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) return;
        ptr_ = static_cast<const std::uint8_t*>(p);
        dma_sync_begin(fd_, DmaSyncDir::Read, /*strict=*/false);
    }
    ~DmaBufReadMap() {
        if (!ptr_) return;
        dma_sync_end(fd_, DmaSyncDir::Read, /*strict=*/false);
        munmap(const_cast<std::uint8_t*>(ptr_), size_);
    }
    DmaBufReadMap(const DmaBufReadMap&) = delete;
    DmaBufReadMap& operator=(const DmaBufReadMap&) = delete;
    const std::uint8_t* data() const { return ptr_; }

private:
    int fd_;
    std::size_t size_;
    const std::uint8_t* ptr_ = nullptr;
};

} // namespace yvo
