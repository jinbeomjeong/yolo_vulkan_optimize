// dma-heap buffer allocation, mapping and CPU cache synchronization (header-only, pure POSIX)
//
// This is the consolidation target of plan Appendix B.11. What had been implemented three times is gathered here in one place:
//   · core/dmabuf_map.hpp  DmaBufReadMap   — maps someone else's fd read-only (no allocation). **It stays as is.**
//                                              Only its internal ioctls delegate to this header's dma_sync_*.
//   · tools/hailo/p2_out_buffers.hpp HeapOutPool — reduced to a thin adapter on top of DmaHeapPool.
//   · (absorbed 2026-09-21) isp/dma_heap_buffer.hpp DmaHeapBuffers was removed and moved onto this pool.
//
// **Uses no Vulkan symbols.** yvo_core is an INTERFACE library, so adding this header does not add a single edge
// to the link graph, and the link isolation that yvo_hailo does not depend on yvo_vulkan (plan §5.2) stays intact.
// The Vulkan import glue (headroom query) lives separately in 01_src/vulkan/vk_storage_headroom.hpp.
//
// ── Fills in everything each of the three implementations was missing ─────────────────────────────────────────────────
//   ① READ/WRITE/RW bidirectional sync  (DmaBufReadMap and HeapOutPool did READ only, DmaHeapBuffers did WRITE only)
//   ② ioctl return value checks (strict by default. Two implementations discarded the return value — Appendix B.10 R5)
//   ③ size alignment + **start address alignment**. mmap only guarantees page alignment. HeapOutPool threw when misaligned, but
//      that kills the process during normal operation → reserve address space, then remap with MAP_FIXED at the aligned point.
//      **The offset inside the dma-buf is always 0** — HailoRT read_async has no offset argument (plan §4.1).
//   ④ wraps memset(zero_init) in a WRITE sync — the current defect pointed out by Appendix B.6(c).
//      Without the wrap, dirty cache lines get evicted later and **overwrite data written by the device**.
//   ⑤ DMA_BUF_SET_NAME (name for debugging)
//   ⑥ device_headroom (Appendix B.12) — prevents a buffer targeted for Vulkan import from being truncated by the driver's
//      extra margin (V3DV: +256 B). No hard-coded constant. The query is done by vk_storage_headroom.hpp.
//
// ── Three-party coherence rules (Appendix B.10) ───────────────────────────────────────────────────────────────
//   R1 device↔device (Hailo DMA write → V3D read): no sync ioctl. Vulkan barriers only.
//   R2 CPU read: **after** the completion signal, START|READ → read → END|READ.
//   R3 CPU write: START|WRITE … END|WRITE. **The initialization memset is no exception.**
//   R4 read and write → READ|WRITE. Do not skimp on direction bits.
//   R5 in strict mode an ioctl failure is an exception.
//   R6 cost: about 0.057 ms for START+END combined per 1.2 MB. The fixed cost is large, so it is a loss for small buffers.
// DMA_BUF_IOCTL_SYNC deals **only with CPU caches** and does not create ordering. Vulkan barriers deal **only with V3D caches**
// and do not touch CPU caches. One is not a substitute for the other.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "core/hailo_align.hpp"

namespace yvo {

// ── Sync direction ───────────────────────────────────────────────────────────────────────────────
enum class DmaSyncDir { Read, Write, ReadWrite };

inline std::uint64_t dma_sync_dir_bits(DmaSyncDir d) {
    switch (d) {
        case DmaSyncDir::Read: return DMA_BUF_SYNC_READ;
        case DmaSyncDir::Write: return DMA_BUF_SYNC_WRITE;
        case DmaSyncDir::ReadWrite: return DMA_BUF_SYNC_RW;
    }
    return DMA_BUF_SYNC_RW;
}

inline const char* dma_sync_dir_name(DmaSyncDir d) {
    switch (d) {
        case DmaSyncDir::Read: return "READ";
        case DmaSyncDir::Write: return "WRITE";
        case DmaSyncDir::ReadWrite: return "READ|WRITE";
    }
    return "?";
}

namespace detail {

// One DMA_BUF_IOCTL_SYNC. Returns 0 on success, errno on failure (the return value is not discarded — R5).
inline int dma_sync_raw(int fd, std::uint64_t flags) {
    if (fd < 0) return EBADF;
    dma_buf_sync s{};
    s.flags = flags;
    return ::ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) != 0 ? (errno != 0 ? errno : EIO) : 0;
}

inline void dma_sync_checked(int fd, std::uint64_t flags, DmaSyncDir dir, const char* phase, bool strict) {
    const int e = dma_sync_raw(fd, flags);
    if (e != 0 && strict)
        throw std::runtime_error(std::string("DMA_BUF_IOCTL_SYNC ") + phase + "|" + dma_sync_dir_name(dir) +
                                 " failed (fd " + std::to_string(fd) + "): " + std::strerror(e));
}

inline std::size_t align_up(std::size_t v, std::size_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

} // namespace detail

// Opens/closes a CPU access section. In strict mode a failure is an exception (R5).
inline void dma_sync_begin(int fd, DmaSyncDir dir, bool strict = true) {
    detail::dma_sync_checked(fd, DMA_BUF_SYNC_START | dma_sync_dir_bits(dir), dir, "START", strict);
}
inline void dma_sync_end(int fd, DmaSyncDir dir, bool strict = true) {
    detail::dma_sync_checked(fd, DMA_BUF_SYNC_END | dma_sync_dir_bits(dir), dir, "END", strict);
}

// ── RAII CPU access section ────────────────────────────────────────────────────────────────────────
// START on construction, END on destruction. **Do not keep a sync section open across device access** (R2).
// The destructor does not throw — to check END directly, call finish() explicitly.
class DmaCpuAccess {
public:
    DmaCpuAccess(int fd, DmaSyncDir dir, bool strict = true) : fd_(fd), dir_(dir), strict_(strict) {
        dma_sync_begin(fd_, dir_, strict_);
        open_ = true;
    }
    ~DmaCpuAccess() {
        try {
            finish();
        } catch (...) {
            // Throwing from a destructor means std::terminate. If a check is needed, call finish() first.
        }
    }
    DmaCpuAccess(const DmaCpuAccess&) = delete;
    DmaCpuAccess& operator=(const DmaCpuAccess&) = delete;

    // Sends END now. Safe to call twice. In strict mode a failure is an exception.
    void finish() {
        if (!open_) return;
        open_ = false;
        dma_sync_end(fd_, dir_, strict_);
    }

    int fd() const { return fd_; }
    DmaSyncDir dir() const { return dir_; }

private:
    int fd_;
    DmaSyncDir dir_;
    bool strict_;
    bool open_ = false;
};

// ── Configuration ──────────────────────────────────────────────────────────────────────────────────────
struct DmaHeapConfig {
    // Tried from the front; the first success is used. vidbuf_cached is a **symbolic link** to system on Pi 5 (measured)
    // — the two are the same heap. The name is kept because they may differ on other devices.
    std::vector<std::string> heaps{"/dev/dma_heap/vidbuf_cached", "/dev/dma_heap/system"};
    // Allocation size and start address alignment. 16384 = HailoRT read_async requirement (Appendix A.4, vulkan/dmabuf_export.hpp kHailoAsyncAlign)
    std::size_t align = kHailoAsyncAlign;
    // Extra bytes the driver requires on top of the requested size (Appendix B.12). Must be > 0 for a Vulkan import target.
    // Put the value obtained by the vk::storage_dmabuf_headroom() query here. **No hard-coded constant.**
    std::size_t device_headroom = 0;
    bool map = true;          // if false, ptr == nullptr (when only the fd is needed)
    bool strict_sync = true;  // if false, ioctl failures are ignored (compatible with DmaBufReadMap's previous behavior)
    bool zero_init = true;    // fill with zeros. **Wrapped in a WRITE sync** (R3, B.6(c))
    std::string name;         // DMA_BUF_SET_NAME. Not set if empty
};

// ── One buffer ─────────────────────────────────────────────────────────────────────────────────
// Why there are two sizes (Appendix A.13 size contract): pass **dmabuf_size** to HailoRT dma_map_dmabuf,
// and **requested_size** to read_async. Vulkan import also gets dmabuf_size.
struct DmaHeapBuf {
    int fd = -1;                    // dma-buf fd. Owned by DmaHeapPool (use dup_fd() to hand it out).
    std::uint8_t* ptr = nullptr;    // start of the CPU mapping (align-aligned guaranteed). nullptr if map=false
    std::size_t requested_size = 0; // byte count the caller passed to allocate() (tensor size)
    std::size_t dmabuf_size = 0;    // dma-buf object size measured via lseek(fd, 0, SEEK_END)
};

// ── Pool ────────────────────────────────────────────────────────────────────────────────────────
class DmaHeapPool {
public:
    explicit DmaHeapPool(const DmaHeapConfig& cfg = {}) : cfg_(cfg) {
        if (cfg_.align == 0) cfg_.align = 1;
        std::string errs;
        for (const std::string& h : cfg_.heaps) {
            heap_fd_ = ::open(h.c_str(), O_RDWR | O_CLOEXEC);
            if (heap_fd_ >= 0) {
                heap_ = h;
                break;
            }
            errs += h + "(" + std::strerror(errno) + ") ";
        }
        if (heap_fd_ < 0) throw std::runtime_error("DmaHeapPool: failed to open dma-heap: " + errs);
    }
    ~DmaHeapPool() {
        for (Entry& e : bufs_) {
            if (e.buf.ptr) ::munmap(e.buf.ptr, e.map_len);
            if (e.buf.fd >= 0) ::close(e.buf.fd);
        }
        if (heap_fd_ >= 0) ::close(heap_fd_);
    }
    DmaHeapPool(const DmaHeapPool&) = delete;
    DmaHeapPool& operator=(const DmaHeapPool&) = delete;

    // Allocates one more buffer that holds `requested` bytes. The actual allocation is
    //   align_up(requested + device_headroom, align)
    // (Appendix B.12). Returns the index. Throws on failure.
    int allocate(std::size_t requested) {
        if (requested == 0) throw std::invalid_argument("DmaHeapPool::allocate: requested == 0");
        const std::size_t want = detail::align_up(requested + cfg_.device_headroom, cfg_.align);

        dma_heap_allocation_data d{};
        d.len = want;
        d.fd_flags = O_RDWR | O_CLOEXEC;
        if (::ioctl(heap_fd_, DMA_HEAP_IOCTL_ALLOC, &d) != 0)
            throw std::runtime_error("DmaHeapPool: DMA_HEAP_IOCTL_ALLOC(" + std::to_string(want) +
                                     ") failed: " + std::strerror(errno));

        Entry e;
        e.buf.fd = static_cast<int>(d.fd);
        e.buf.requested_size = requested;
        const off_t sz = ::lseek(e.buf.fd, 0, SEEK_END);
        e.buf.dmabuf_size = sz > 0 ? static_cast<std::size_t>(sz) : want;
        e.map_len = e.buf.dmabuf_size;

        try {
            if (!cfg_.name.empty()) {
                // Failure is not fatal (debugging name). Some kernels block the O_CLOEXEC dup.
                ::ioctl(e.buf.fd, DMA_BUF_SET_NAME, cfg_.name.c_str());
            }
            if (cfg_.map) e.buf.ptr = map_aligned(e.buf.fd, e.map_len, cfg_.align);
            if (cfg_.map && cfg_.zero_init) {
                // R3 / B.6(c): open the WRITE section before dirtying the cached mapping.
                DmaCpuAccess acc(e.buf.fd, DmaSyncDir::Write, cfg_.strict_sync);
                std::memset(e.buf.ptr, 0, e.map_len);
                acc.finish();   // do not let the destructor swallow a failure
            }
        } catch (...) {
            if (e.buf.ptr) ::munmap(e.buf.ptr, e.map_len);
            ::close(e.buf.fd);
            throw;
        }
        bufs_.push_back(e);
        return static_cast<int>(bufs_.size()) - 1;
    }

    std::size_t count() const { return bufs_.size(); }
    const std::string& heap() const { return heap_; }
    const DmaHeapConfig& config() const { return cfg_; }

    const DmaHeapBuf& at(int i) const { return bufs_.at(check(i)).buf; }

    // A new fd to hand to the caller (the caller closes it). The original fd stays held by this pool.
    int dup_fd(int i) const {
        const int nfd = ::dup(at(i).fd);
        if (nfd < 0) throw std::runtime_error(std::string("DmaHeapPool::dup_fd failed: ") + std::strerror(errno));
        return nfd;
    }

    // Convenience wrapper — a CPU access section wrapped according to the rules
    DmaCpuAccess cpu_access(int i, DmaSyncDir dir) const {
        return DmaCpuAccess(at(i).fd, dir, cfg_.strict_sync);
    }

private:
    struct Entry {
        DmaHeapBuf buf;
        std::size_t map_len = 0;   // length to pass to munmap (= dmabuf_size)
    };

    std::size_t check(int i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= bufs_.size())
            throw std::out_of_range("DmaHeapPool: index " + std::to_string(i) + " out of range");
        return static_cast<std::size_t>(i);
    }

    // Maps the dma-buf at an align-aligned address.
    //   1) **reserve address space only**, len+align bytes, with a PROT_NONE anonymous mapping
    //   2) overlay the dma-buf with MAP_FIXED|MAP_SHARED at the align-aligned point inside the reservation
    //   3) munmap the leading and trailing slack
    // The offset inside the dma-buf is **always 0** — alignment only moves the virtual address.
    // (This device has a 16384 page size so leading is always 0, but it also works correctly on 4 KB page kernels.)
    static std::uint8_t* map_aligned(int fd, std::size_t len, std::size_t align) {
        void* base = ::mmap(nullptr, len + align, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (base == MAP_FAILED)
            throw std::runtime_error(std::string("DmaHeapPool: address space reservation (mmap PROT_NONE) failed: ") +
                                     std::strerror(errno));
        const auto b = reinterpret_cast<std::uintptr_t>(base);
        const std::uintptr_t a = detail::align_up(b, align);
        const std::size_t lead = static_cast<std::size_t>(a - b);
        const std::size_t tail = align - lead;   // (len+align) - lead - len

        void* p = ::mmap(reinterpret_cast<void*>(a), len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
        if (p == MAP_FAILED) {
            const int err = errno;
            ::munmap(base, len + align);
            throw std::runtime_error(std::string("DmaHeapPool: mmap(MAP_FIXED) failed: ") + std::strerror(err));
        }
        if (lead > 0) ::munmap(base, lead);
        if (tail > 0) ::munmap(reinterpret_cast<void*>(a + len), tail);
        return static_cast<std::uint8_t*>(p);
    }

    DmaHeapConfig cfg_;
    int heap_fd_ = -1;
    std::string heap_;
    std::vector<Entry> bufs_;
};

} // namespace yvo
