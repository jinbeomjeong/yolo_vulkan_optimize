// Software decode support (FfmpegDrmDecoder Options::software, yvo_pipeline --decoder sw): I420 → SAND128 NV12 copy and
// the dma-heap buffer pool behind it. The output has the layout of the HW decoder's frames (DrmFrame in core/frame_types.hpp),
// so everything after the decoder runs unchanged.
//
// Heap: the system heap (DmaHeapConfig default). V3D maps imported dma-bufs through its own MMU, so physically contiguous memory
// is not needed, and the 64 MB CMA area stays free for the display. V3D reads it no slower than CMA (gpu shader ms, 300 frames: 1 thread
// 1.85 against 1.91, auto 2.31 against 2.53). These buffers go CPU → V3D only (the path of PostprocFixture.ImportHeapMatchesPlainBitExact).
//
// CPU cache: the mapping is cached and V3D does not see CPU caches, so every copy runs inside a WRITE sync section
// (dma_heap_alloc.hpp R3). The heap's END only cleans through attachments that are already mapped; before the first import there
// is none, and then the import's own dma_map cleans the cache. Both cases reach V3D with clean data.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/dma_heap_alloc.hpp"

namespace yvo {

inline constexpr std::size_t kSandColBytes = 128;  // SAND128 column width (bytes)

// Column height of the Y plane. Same value as the HW decoder's measured output: 1080 → 1088 (UV: half of it, 544).
inline std::size_t sand_y_col_height(int height) { return (static_cast<std::size_t>(height) + 15) / 16 * 16; }
inline std::size_t sand_uv_col_height(int height) { return sand_y_col_height(height) / 2; }

// Bytes of a plane made of whole columns. The last column is complete even when the width ends inside it.
inline std::size_t sand_plane_bytes(int width, std::size_t col_height) {
    const std::size_t cols = (static_cast<std::size_t>(width) + kSandColBytes - 1) / kSandColBytes;
    return cols * kSandColBytes * col_height;
}

// I420 (8 bit) → SAND128 NV12, visible area only. Rows and columns beyond it keep their old bytes: every consumer clamps to
// src_w/src_h (sand_letterbox.comp, display_convert.comp).
// Address: Y(x, y) = dst_y[(x / 128) * 128 * y_col_h + y * 128 + x % 128], UV likewise with the interleaved byte position (U at even
// x, V at odd x). One column holds 64 chroma pairs, so a pair never crosses a column. The loops run column by column, the order
// the destination is laid out in (microbenchmark, hot source: Y 0.25 ms against 0.39 ms row by row). The plain interleave loop is auto-vectorized
// and beat vst2q_u8 (0.07 ms against 0.13 ms), so no intrinsics.
// For a bad size or a layout that does not fit, writes nothing and returns false.
inline bool i420_to_sand128(const std::uint8_t* src_y, std::size_t y_stride, const std::uint8_t* src_u, std::size_t u_stride,
                            const std::uint8_t* src_v, std::size_t v_stride, int width, int height, std::uint8_t* dst_y,
                            std::size_t dst_y_size, std::size_t y_col_h, std::uint8_t* dst_uv, std::size_t dst_uv_size,
                            std::size_t uv_col_h) {
    if (!src_y || !src_u || !src_v || !dst_y || !dst_uv) return false;
    if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0) return false;
    const std::size_t w = static_cast<std::size_t>(width), h = static_cast<std::size_t>(height), ch = h / 2;
    if (y_stride < w || u_stride < w / 2 || v_stride < w / 2) return false;
    if (y_col_h < h || uv_col_h < ch) return false;
    const std::size_t ncols = (w - 1) / kSandColBytes + 1;
    const std::size_t y_col_stride = kSandColBytes * y_col_h, uv_col_stride = kSandColBytes * uv_col_h;
    const std::size_t last_n = w - (ncols - 1) * kSandColBytes;
    if ((ncols - 1) * y_col_stride + (h - 1) * kSandColBytes + last_n > dst_y_size) return false;
    if ((ncols - 1) * uv_col_stride + (ch - 1) * kSandColBytes + last_n > dst_uv_size) return false;

    for (std::size_t c = 0; c < ncols; ++c) {
        const std::size_t x0 = c * kSandColBytes;
        const std::size_t n = std::min(kSandColBytes, w - x0);
        const std::uint8_t* s = src_y + x0;
        std::uint8_t* d = dst_y + c * y_col_stride;
        if (n == kSandColBytes) {
            for (std::size_t y = 0; y < h; ++y, s += y_stride, d += kSandColBytes) std::memcpy(d, s, kSandColBytes);
        } else {
            for (std::size_t y = 0; y < h; ++y, s += y_stride, d += kSandColBytes) std::memcpy(d, s, n);
        }
    }
    for (std::size_t c = 0; c < ncols; ++c) {
        const std::size_t cx0 = c * (kSandColBytes / 2);
        const std::size_t pairs = std::min(kSandColBytes / 2, w / 2 - cx0);
        const std::uint8_t* u = src_u + cx0;
        const std::uint8_t* v = src_v + cx0;
        std::uint8_t* d = dst_uv + c * uv_col_stride;
        for (std::size_t y = 0; y < ch; ++y, u += u_stride, v += v_stride, d += kSandColBytes)
            for (std::size_t i = 0; i < pairs; ++i) {
                d[2 * i] = u[i];
                d[2 * i + 1] = v[i];
            }
    }
    return true;
}

// Fixed set of Y + UV dma-buf pairs. All of it is allocated and first touched in the constructor (before the run's CPU window).
//
// Lifetime: Lease::keepalive plays the role of the HW path's AVFrame reference. Releasing its last reference returns the pair to the
//   free list, from any thread. It also holds a reference to the pool, so frames may outlive the decoder (main.cpp clears the
//   import cache and closes the decoder while the display may still release frames).
// Reuse: a pair keeps its fds for the pool's lifetime, so DmaBufImportCache (keyed on fstat st_dev/st_ino + size) hits. The free
//   list is LIFO: buffers that are not needed never get imported.
// Exhaustion: acquire() returns nullopt and never blocks. The releases can depend on the very thread that decodes, so waiting
//   would be the same silent stall the HW decoder has when too many frames are held.
class SandFramePool : public std::enable_shared_from_this<SandFramePool> {
public:
    struct Config {
        int width = 0;
        int height = 0;
        int count = 0;                   // Y + UV pairs
        std::vector<std::string> heaps;  // empty = DmaHeapConfig default (vidbuf_cached, system)
    };
    struct Lease {
        std::uint8_t* y = nullptr;
        std::uint8_t* uv = nullptr;
        int y_fd = -1;            // owned by the pool: valid while keepalive is alive
        int uv_fd = -1;
        std::size_t y_size = 0;   // dma-buf object sizes
        std::size_t uv_size = 0;
        std::shared_ptr<void> keepalive;
    };

    // Throws std::invalid_argument for a bad config and std::runtime_error when the dma-heap fails.
    explicit SandFramePool(const Config& cfg) : count_(cfg.count), heap_pool_(make_heap_config(cfg)) {
        if (cfg.width <= 0 || cfg.height <= 0 || (cfg.width % 2) != 0 || (cfg.height % 2) != 0 || cfg.count <= 0)
            throw std::invalid_argument("SandFramePool: bad frame size or buffer count");
        const std::size_t y_bytes = sand_plane_bytes(cfg.width, sand_y_col_height(cfg.height));
        const std::size_t uv_bytes = sand_plane_bytes(cfg.width, sand_uv_col_height(cfg.height));
        free_.reserve(static_cast<std::size_t>(cfg.count));
        for (int i = 0; i < cfg.count; ++i) {
            const int y = heap_pool_.allocate(y_bytes);
            const int uv = heap_pool_.allocate(uv_bytes);
            pairs_.push_back({y, uv});
            free_.push_back(cfg.count - 1 - i);  // buffer 0 goes out first
        }
    }
    SandFramePool(const SandFramePool&) = delete;
    SandFramePool& operator=(const SandFramePool&) = delete;

    // nullopt when every pair is held. The pool must be owned by a shared_ptr.
    std::optional<Lease> acquire() {
        int idx = -1;
        {
            const std::lock_guard<std::mutex> lock(mu_);
            if (free_.empty()) return std::nullopt;
            idx = free_.back();
            free_.pop_back();
            peak_in_use_ = std::max(peak_in_use_, count_ - static_cast<int>(free_.size()));
        }
        const DmaHeapBuf& y = heap_pool_.at(pairs_[static_cast<std::size_t>(idx)].y);
        const DmaHeapBuf& uv = heap_pool_.at(pairs_[static_cast<std::size_t>(idx)].uv);
        Lease l;
        l.y = y.ptr;
        l.uv = uv.ptr;
        l.y_fd = y.fd;
        l.uv_fd = uv.fd;
        l.y_size = y.dmabuf_size;
        l.uv_size = uv.dmabuf_size;
        // The stored pointer is non-null on purpose: consumers treat a null keepalive as "no frame".
        l.keepalive = std::shared_ptr<void>(this, [self = shared_from_this(), idx](void*) { self->release(idx); });
        return l;
    }

    int count() const { return count_; }
    int in_use() const {
        const std::lock_guard<std::mutex> lock(mu_);
        return count_ - static_cast<int>(free_.size());
    }
    int peak_in_use() const {
        const std::lock_guard<std::mutex> lock(mu_);
        return peak_in_use_;
    }
    const std::string& heap() const { return heap_pool_.heap(); }

private:
    struct Pair {
        int y;
        int uv;
    };

    static DmaHeapConfig make_heap_config(const Config& cfg) {
        DmaHeapConfig hc;
        if (!cfg.heaps.empty()) hc.heaps = cfg.heaps;
        hc.name = "yvo_sw_sand";
        return hc;
    }

    void release(int idx) {
        const std::lock_guard<std::mutex> lock(mu_);
        free_.push_back(idx);
    }

    const int count_;
    DmaHeapPool heap_pool_;
    std::vector<Pair> pairs_;
    mutable std::mutex mu_;
    std::vector<int> free_;
    int peak_in_use_ = 0;
};

} // namespace yvo
