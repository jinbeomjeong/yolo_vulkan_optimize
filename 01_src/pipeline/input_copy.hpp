// InputCopy: yvo_pipeline --input-copy. A comparison path that puts one CPU copy between the GPU and the NPU, to measure what
// the zero-copy hand-off (the NPU reading the V3D output dma-buf directly) saves. Not an operating path.
//
//   GPU fence done (SandLetterbox::wait) → CPU memcpy V3D output → cached dma-heap buffer of the same slot → NPU reads that buffer
//
// Source: the slot's V3D mapping (HOST_COHERENT, already readable when wait() returns; sand_letterbox.hpp).
// Destination: one buffer per slot from the system heap (DmaHeapConfig default, cached CPU mapping, HailoRT alignment).
// Lifetime: the slot number is the ownership token. A slot's buffer is written only between its GPU fence and its NPU submit, and the
//   slot returns only after the NPU completion (and display_done), so a copy never overlaps an NPU read of the same buffer.
//   The pool must outlive the stage: the runner holds the fds mapped until it is destroyed (main.cpp declares this first).
// CPU cache: every copy runs in a WRITE sync section (dma_heap_alloc.hpp R3) because the NPU does not see CPU caches. The stage's bind_input_dmabuf()
//   has already mapped the attachment, so the heap's END cleans through it. Before that, the zero-init is cleaned by the mapping itself.
// Single-threaded: main thread only (finish_gpu in main.cpp).
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/dma_heap_alloc.hpp"

namespace yvo::pipeline {

class InputCopy {
public:
    InputCopy(int slots, std::size_t frame_size) : pool_(pool_config()) {
        for (int s = 0; s < slots; ++s) pool_.allocate(frame_size);
    }

    int fd(int slot) const { return pool_.at(slot).fd; }   // owned by the pool: the caller does not close it
    int buffers() const { return static_cast<int>(pool_.count()); }
    const std::string& heap() const { return pool_.heap(); }

    // Call after the GPU fence and before the NPU submit of this slot. Throws on a bad size or a failed sync ioctl.
    void copy(int slot, const std::uint8_t* src, std::size_t size) {
        const DmaHeapBuf& b = pool_.at(slot);
        if (!src || size != b.requested_size)
            throw std::runtime_error("InputCopy: slot " + std::to_string(slot) + " got " + std::to_string(size) + " B, expected " +
                                     std::to_string(b.requested_size) + " B");
        const auto t0 = Clock::now();
        DmaCpuAccess acc(b.fd, DmaSyncDir::Write);
        const auto t1 = Clock::now();
        std::memcpy(b.ptr, src, size);
        const auto t2 = Clock::now();
        acc.finish();
        const auto t3 = Clock::now();
        memcpy_ms_.push_back(ms_of(t2 - t1));
        sync_ms_.push_back(ms_of((t1 - t0) + (t3 - t2)));
        total_ms_.push_back(ms_of(t3 - t0));
    }

    // Per-frame samples in ms. total = memcpy + sync (START + END).
    const std::vector<double>& total_ms() const { return total_ms_; }
    const std::vector<double>& memcpy_ms() const { return memcpy_ms_; }
    const std::vector<double>& sync_ms() const { return sync_ms_; }

private:
    using Clock = std::chrono::steady_clock;

    static double ms_of(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
    static DmaHeapConfig pool_config() {
        DmaHeapConfig c;
        c.name = "yvo_input_copy";
        return c;
    }

    DmaHeapPool pool_;
    std::vector<double> total_ms_, memcpy_ms_, sync_ms_;
};

} // namespace yvo::pipeline
