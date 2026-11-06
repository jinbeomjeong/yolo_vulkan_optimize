// OutputCopy: yvo_pipeline --output-copy. A comparison path that puts one CPU copy between the NPU and the GPU post-processing, to
// measure what the zero-copy hand-off (Hailo DMA writing the 6 raw output tensors straight into the buffers V3D reads) saves.
// Not an operating path.
//
//   NPU completion → CPU memcpy of the 6 raw tensors from cached dma-heap buffers of the slot → the slot's YoloPostproc raw buffers
//   (V3D side) → GPU post-processing is submitted (yolo_decode.comp + efficient_nms.comp, unchanged)
//
// Source: 6 buffers per slot from the system heap (DmaHeapConfig default, cached CPU mapping, HailoRT alignment 16384). The runner
//   maps them D2H in place of the YoloPostproc buffers (GpuPostprocStage::set_output_copy). Buffer index = slot * 6 + tensor.
// Destination: the YoloPostproc raw buffer of the same slot and tensor (raw_host_ptr). ExportDmaBuf: a vkMapMemory pointer to
//   HOST_VISIBLE|HOST_COHERENT memory (dmabuf_export.cpp), no flush needed — vkQueueSubmit makes host writes to coherent memory
//   visible to the device. ImportHeap: the cached heap mmap, so the write runs inside raw_cpu_sync_begin/end(Write) (R3: V3D does
//   not see CPU caches). The stage passes those two calls as dst_begin/dst_end; they are no-ops for ExportDmaBuf.
// CPU cache of the source: the NPU does not see CPU caches either, so every read runs in a READ sync section (dma_heap_alloc.hpp R2)
//   opened after the completion callback. The pool zero-fills inside a WRITE section, so no dirty line can overwrite NPU data, and
//   the CPU only reads these buffers afterwards.
// Lifetime: the slot number is the ownership token. The NPU writes a slot's buffers only between its submit and its completion
//   callback; the copy runs after that callback and before pp_->submit of the same slot; the slot returns (and can be resubmitted)
//   only after the GPU fence of that submit. So a copy never overlaps an NPU write of the source or a GPU read of the destination.
//   The pool must outlive the stage: the runner holds the fds mapped until it is destroyed (main.cpp declares this first).
// Single-threaded: main thread only (GpuPostprocStage::push_ready_to_gpu, inside pump / drain_one / drain). The copy is in the
//   process CPU time like the rest of the main thread.
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/dma_heap_alloc.hpp"

namespace yvo::pipeline {

class OutputCopy {
public:
    static constexpr int kTensors = 6;   // = yvo::vk::kYoloNumTensors (this header does not include Vulkan headers)
    using Sizes = std::array<std::size_t, kTensors>;

    // sizes[t]: transfer size of tensor t (hw_frame_size, YoloPostproc::raw_size). Throws on a bad count or a dma-heap failure.
    OutputCopy(int slots, const Sizes& sizes) : slots_(slots), sizes_(sizes), pool_(pool_config()) {
        if (slots < 1) throw std::invalid_argument("OutputCopy: slots must be >= 1");
        for (std::size_t s : sizes_)
            if (s == 0) throw std::invalid_argument("OutputCopy: tensor size 0");
        for (int s = 0; s < slots; ++s)
            for (int t = 0; t < kTensors; ++t) pool_.allocate(sizes_[static_cast<std::size_t>(t)]);
    }

    int slots() const { return slots_; }
    int buffers() const { return static_cast<int>(pool_.count()); }
    const std::string& heap() const { return pool_.heap(); }
    std::size_t size(int tensor) const { return sizes_.at(static_cast<std::size_t>(tensor)); }
    int fd(int slot, int tensor) const { return buf(slot, tensor).fd; }   // owned by the pool: the caller does not close it
    std::size_t map_size(int slot, int tensor) const { return buf(slot, tensor).dmabuf_size; }   // for dma_map_dmabuf (lseek size)
    const std::uint8_t* data(int slot, int tensor) const { return buf(slot, tensor).ptr; }
    std::uint8_t* data(int slot, int tensor) { return buf(slot, tensor).ptr; }   // tests only: stands in for the NPU writer

    // Call after the NPU completion of this slot and before the GPU submit that reads dst. dst[t] holds at least size(t) bytes.
    // dst_begin/dst_end open and close the CPU write section of the destination (empty for coherent memory).
    // Throws on a null destination or a failed sync ioctl.
    void copy(int slot, std::uint8_t* const dst[kTensors], const std::function<void()>& dst_begin = {},
              const std::function<void()>& dst_end = {}) {
        for (int t = 0; t < kTensors; ++t)
            if (!dst[t])
                throw std::runtime_error("OutputCopy: slot " + std::to_string(slot) + " tensor " + std::to_string(t) +
                                         " has no destination mapping");
        const auto t0 = Clock::now();
        int opened = 0;
        Clock::time_point t1, t2;
        try {
            for (; opened < kTensors; ++opened) dma_sync_begin(buf(slot, opened).fd, DmaSyncDir::Read);
            if (dst_begin) dst_begin();
            t1 = Clock::now();
            for (int t = 0; t < kTensors; ++t) std::memcpy(dst[t], buf(slot, t).ptr, sizes_[static_cast<std::size_t>(t)]);
            t2 = Clock::now();
            if (dst_end) dst_end();
            for (; opened > 0; --opened) dma_sync_end(buf(slot, opened - 1).fd, DmaSyncDir::Read);
        } catch (...) {
            for (; opened > 0; --opened) dma_sync_end(buf(slot, opened - 1).fd, DmaSyncDir::Read, /*strict=*/false);
            throw;
        }
        const auto t3 = Clock::now();
        memcpy_ms_.push_back(ms_of(t2 - t1));
        sync_ms_.push_back(ms_of((t1 - t0) + (t3 - t2)));
        total_ms_.push_back(ms_of(t3 - t0));
    }

    // Per-frame samples in ms. total = memcpy + sync (6 READ START/END of the source + the destination section).
    const std::vector<double>& total_ms() const { return total_ms_; }
    const std::vector<double>& memcpy_ms() const { return memcpy_ms_; }
    const std::vector<double>& sync_ms() const { return sync_ms_; }

private:
    using Clock = std::chrono::steady_clock;

    static double ms_of(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
    static DmaHeapConfig pool_config() {
        DmaHeapConfig c;
        c.name = "yvo_output_copy";
        return c;
    }
    const DmaHeapBuf& buf(int slot, int tensor) const {
        if (slot < 0 || slot >= slots_ || tensor < 0 || tensor >= kTensors)
            throw std::out_of_range("OutputCopy: slot " + std::to_string(slot) + " tensor " + std::to_string(tensor) + " out of range");
        return pool_.at(slot * kTensors + tensor);
    }

    const int slots_;
    const Sizes sizes_;
    DmaHeapPool pool_;
    std::vector<double> total_ms_, memcpy_ms_, sync_ms_;
};

} // namespace yvo::pipeline
