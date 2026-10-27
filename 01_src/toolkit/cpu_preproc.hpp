#pragma once
// CPU preprocessing helpers for the CPU-vs-GPU preprocessing comparison (yvo_preproc_bench --impl, 2026-10-27).
// Tools only (header-only, no FFmpeg): libraries do not include it.
//
//   WorkerPool          fork-join pool (caller + N-1 persistent workers) used for row/column band parallelism
//   band_range()        splits [0, total) into near-equal bands, optionally aligned
//   ref_*_band()        the ref/ CPU reference functions applied to one band. The band versions call the unchanged
//                       ref functions with shifted pointers/params, so the result is byte-identical to one full call
//                       (checked by yvo_preproc_bench --impl cpuref --threads 4 --accuracy)
//   packed_to_nhcw_rows() NEON (vld3q_u8) packed RGB -> NHCW transpose of content rows with a column offset
//   DmaBufMapCache      persistent read-only mmap of decoder dma-bufs, keyed by inode (the Vulkan path uses an import
//                       cache too, so per-frame mmap is not charged to the optimized CPU paths)
//   dmabuf_exporter()   exporter name from /proc/self/fdinfo (which allocator backs a dma-buf)
//   DiffStats           byte comparison metrics (identical fraction, max/mean abs error, PSNR)
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "core/frame_types.hpp"
#include "ref/cpu_letterbox.hpp"
#include "ref/sand.hpp"

namespace yvo::tool {

// ── Fork-join pool ─────────────────────────────────────────────────────────────────────────────
// run(fn) calls fn(i) for i in [0, size()) — i == 0 on the calling thread — and returns when all calls are done.
// Workers sleep on a condition variable between runs (no spinning, so idle workers cost no CPU time).
class WorkerPool {
public:
    explicit WorkerPool(int n) : n_(std::max(1, n)) {
        for (int i = 1; i < n_; ++i) threads_.emplace_back([this, i] { loop(i); });
    }
    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (std::thread& t : threads_) t.join();
    }
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    int size() const { return n_; }

    void run(const std::function<void(int)>& fn) {
        if (n_ == 1) {
            fn(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            remaining_ = n_ - 1;
            ++gen_;
        }
        cv_.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return remaining_ == 0; });
        fn_ = nullptr;
    }

private:
    void loop(int idx) {
        std::uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)>* fn = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                fn = fn_;
            }
            (*fn)(idx);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--remaining_ == 0) done_cv_.notify_one();
            }
        }
    }

    int n_;
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int)>* fn_ = nullptr;
    int remaining_ = 0;
    std::uint64_t gen_ = 0;
    bool stop_ = false;
};

// Band i of `parts` over [0, total). Band edges are multiples of `align` (except the end).
inline std::pair<int, int> band_range(int total, int parts, int i, int align = 1) {
    const int units = (total + align - 1) / align;
    const int b = static_cast<int>(static_cast<long>(units) * i / parts) * align;
    const int e = static_cast<int>(static_cast<long>(units) * (i + 1) / parts) * align;
    return {std::min(b, total), std::min(e, total)};
}

// ── ref/ functions on one band (byte-identical to the full call) ───────────────────────────────
// de-SAND of luma rows [r0, r1) (r0, r1 even) over all columns: each band writes whole contiguous output rows and
// reads one contiguous (r1 - r0) * 128 byte run per column (better than column bands, which interleave the writers
// inside every output row). Byte-identical to one full call.
inline bool ref_desand_rows(const std::uint8_t* y_src, std::size_t y_size, const std::uint8_t* uv_src,
                            std::size_t uv_size, const SandNv12Layout& l, int r0, int r1, std::uint8_t* dst_y,
                            std::uint8_t* dst_uv, std::size_t dst_stride) {
    if (r0 >= r1) return true;
    if ((r0 % 2) != 0 || (r1 % 2) != 0) return false;
    SandNv12Layout b = l;
    b.y_offset = l.y_offset + static_cast<std::size_t>(r0) * kSandColWidth;
    b.uv_offset = l.uv_offset + static_cast<std::size_t>(r0 / 2) * kSandColWidth;
    b.height = r1 - r0;
    return desand_nv12(y_src, y_size, uv_src, uv_size, b, dst_y + static_cast<std::size_t>(r0) * dst_stride,
                       dst_stride, dst_uv + static_cast<std::size_t>(r0 / 2) * dst_stride, dst_stride);
}

// Destination row band i of `parts` for the letterbox: the content rows [pad_y, pad_y + new_h) are split evenly
// (they carry the work); band 0 also takes the top pad rows and the last band the bottom pad rows.
inline std::pair<int, int> letterbox_row_band(const LetterboxParams& lb, int new_h, int parts, int i) {
    const auto [b, e] = band_range(new_h, parts, i);
    return {i == 0 ? 0 : lb.pad_y + b, i == parts - 1 ? lb.dst_h : lb.pad_y + e};
}

// Letterbox of destination rows [r0, r1): same formulas because only (dy - pad_y) enters them.
inline void ref_letterbox_rows(const std::uint8_t* y, std::size_t y_pitch, const std::uint8_t* uv, std::size_t uv_pitch,
                               int src_w, int src_h, const LetterboxParams& lb, int r0, int r1, std::uint8_t* dst) {
    if (r0 >= r1) return;
    LetterboxParams b = lb;
    b.dst_h = r1 - r0;
    b.pad_y = lb.pad_y - r0;
    cpu_letterbox_nv12_bt709(y, y_pitch, uv, uv_pitch, src_w, src_h, b,
                             dst + static_cast<std::size_t>(r0) * lb.dst_w * 3);
}

// packed -> NHCW transpose of rows [r0, r1) (ref function).
inline void ref_nhcw_rows(const std::uint8_t* src, int w, int r0, int r1, std::uint8_t* dst) {
    if (r0 >= r1) return;
    rgb_packed_to_nhcw(src + static_cast<std::size_t>(r0) * w * 3, w, r1 - r0,
                       dst + static_cast<std::size_t>(r0) * 3 * w);
}

// ── Optimized transpose (content rows only) ─────────────────────────────────────────────────────
// src: `rows` rows of `w` packed RGB pixels (stride src_stride bytes). dst: NHCW tensor of width dst_w; writes row
// dst_row0 + r, columns [x_off, x_off + w) of each channel. Same byte mapping as rgb_packed_to_nhcw.
inline void packed_to_nhcw_rows(const std::uint8_t* src, std::size_t src_stride, int w, int rows, std::uint8_t* dst,
                                int dst_w, int dst_row0, int x_off) {
    for (int r = 0; r < rows; ++r) {
        const std::uint8_t* s = src + static_cast<std::size_t>(r) * src_stride;
        std::uint8_t* R = dst + static_cast<std::size_t>(dst_row0 + r) * 3 * dst_w + x_off;
        std::uint8_t* G = R + dst_w;
        std::uint8_t* B = G + dst_w;
        int x = 0;
#if defined(__ARM_NEON)
        for (; x + 16 <= w; x += 16) {
            const uint8x16x3_t v = vld3q_u8(s + x * 3);
            vst1q_u8(R + x, v.val[0]);
            vst1q_u8(G + x, v.val[1]);
            vst1q_u8(B + x, v.val[2]);
        }
#endif
        for (; x < w; ++x) {
            R[x] = s[x * 3 + 0];
            G[x] = s[x * 3 + 1];
            B[x] = s[x * 3 + 2];
        }
    }
}

// Fill everything outside the content rectangle of an NHCW tensor with lb.pad_value (all channels equal).
// Top/bottom pad rows are contiguous byte ranges; side pads (pad_x > 0) are filled per channel row.
inline void nhcw_fill_pad(std::uint8_t* dst, const LetterboxParams& lb, int new_w, int new_h) {
    const std::size_t row = static_cast<std::size_t>(lb.dst_w) * 3;
    std::memset(dst, lb.pad_value, static_cast<std::size_t>(lb.pad_y) * row);
    const int bot0 = lb.pad_y + new_h;
    std::memset(dst + static_cast<std::size_t>(bot0) * row, lb.pad_value,
                static_cast<std::size_t>(lb.dst_h - bot0) * row);
    const int right = lb.dst_w - lb.pad_x - new_w;
    if (lb.pad_x == 0 && right == 0) return;
    for (int y = lb.pad_y; y < bot0; ++y)
        for (int c = 0; c < 3; ++c) {
            std::uint8_t* p = dst + static_cast<std::size_t>(y) * row + static_cast<std::size_t>(c) * lb.dst_w;
            std::memset(p, lb.pad_value, static_cast<std::size_t>(lb.pad_x));
            std::memset(p + lb.pad_x + new_w, lb.pad_value, static_cast<std::size_t>(right));
        }
}

// ── Decoder dma-buf mappings ────────────────────────────────────────────────────────────────────
// Read-only MAP_SHARED mapping per dma-buf object (key: st_dev/st_ino of the fd). The mapping keeps the object alive,
// so clear() must run before the decoder is closed if its memory should be freed. CPU cache sync is the caller's job.
class DmaBufMapCache {
public:
    DmaBufMapCache() = default;
    ~DmaBufMapCache() { clear(); }
    DmaBufMapCache(const DmaBufMapCache&) = delete;
    DmaBufMapCache& operator=(const DmaBufMapCache&) = delete;

    const std::uint8_t* map(int fd, std::size_t size) {
        struct stat st{};
        if (::fstat(fd, &st) != 0) throw std::runtime_error("DmaBufMapCache: fstat failed");
        const Key k{static_cast<std::uint64_t>(st.st_dev), static_cast<std::uint64_t>(st.st_ino)};
        auto it = maps_.find(k);
        if (it != maps_.end() && it->second.second >= size) {
            ++hits_;
            return it->second.first;
        }
        void* p = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("DmaBufMapCache: mmap failed");
        if (it != maps_.end()) {
            ::munmap(const_cast<std::uint8_t*>(it->second.first), it->second.second);
            maps_.erase(it);
        }
        maps_[k] = {static_cast<const std::uint8_t*>(p), size};
        ++maps_made_;
        return static_cast<const std::uint8_t*>(p);
    }
    void clear() {
        for (auto& [k, v] : maps_) ::munmap(const_cast<std::uint8_t*>(v.first), v.second);
        maps_.clear();
    }
    std::size_t entries() const { return maps_.size(); }
    std::uint64_t hits() const { return hits_; }
    std::uint64_t maps_made() const { return maps_made_; }

private:
    using Key = std::pair<std::uint64_t, std::uint64_t>;
    std::map<Key, std::pair<const std::uint8_t*, std::size_t>> maps_;
    std::uint64_t hits_ = 0, maps_made_ = 0;
};

// "exp_name" line of /proc/self/fdinfo/<fd> (e.g. "system", "linux,cma"); "?" if unreadable.
inline std::string dmabuf_exporter(int fd) {
    std::ifstream in("/proc/self/fdinfo/" + std::to_string(fd));
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("exp_name:", 0) == 0) {
            std::string v = line.substr(9);
            v.erase(0, v.find_first_not_of(" \t"));
            return v;
        }
    return "?";
}

// ── Byte comparison metrics ─────────────────────────────────────────────────────────────────────
struct DiffStats {
    std::uint64_t n = 0, identical = 0, sum_abs = 0, sum_sq = 0;
    int max_abs = 0;

    void add(const std::uint8_t* a, const std::uint8_t* b, std::size_t len) {
        for (std::size_t i = 0; i < len; ++i) {
            const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
            identical += d == 0;
            sum_abs += static_cast<std::uint64_t>(d);
            sum_sq += static_cast<std::uint64_t>(d * d);
            max_abs = std::max(max_abs, d);
        }
        n += len;
    }
    void merge(const DiffStats& o) {
        n += o.n;
        identical += o.identical;
        sum_abs += o.sum_abs;
        sum_sq += o.sum_sq;
        max_abs = std::max(max_abs, o.max_abs);
    }
    double mean_abs() const { return n ? static_cast<double>(sum_abs) / static_cast<double>(n) : 0.0; }
    double identical_frac() const { return n ? static_cast<double>(identical) / static_cast<double>(n) : 1.0; }
    // PSNR in dB over the compared bytes; +inf when identical.
    double psnr() const {
        if (sum_sq == 0) return INFINITY;
        const double mse = static_cast<double>(sum_sq) / static_cast<double>(n);
        return 10.0 * std::log10(255.0 * 255.0 / mse);
    }
};

// FNV-1a 64 over a buffer (cheap output fingerprint for thread-count comparisons).
inline std::uint64_t fnv1a64(const std::uint8_t* p, std::size_t len, std::uint64_t h = 1469598103934665603ULL) {
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace yvo::tool
