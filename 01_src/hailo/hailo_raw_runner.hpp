// HailoRT **6 raw output streams** async runner (yolov11m.hef, raw tensors without NMS)
//
// If hailo_runner.{hpp,cpp} is the **reference path** that parses the single HEF-embedded NMS output on the host,
// this is the **production path that receives the 6 pre-NMS raw tensors as dma-bufs and hands them to GPU post-processing**.
// Promoted from RawSession in 01_src/tools/hailo/hailo_raw_probe.cpp (passed the F0 gate, 51.90 FPS).
//
// Flow: Hef::create → VDevice(ROUND_ROBIN) → create_configure_params(batch, all streams ASYNC)
//       → VDevice::configure → per-slot bind_slot (maps 7 dma-buf fds)
//       → on each submit(): wait_for_ongoing_callbacks_count_under backpressure + infer_async (1 input + 6 outputs)
//       → once all 7 buffer callbacks arrive, the user callback fires once.
//
// ── Input/output is dma-buf only ─────────────────────────────────────────────────────────────
// There is no host pointer path, because the integrated pipeline uses only dma-bufs exported by V3D.
// The runner knows nothing about Vulkan — **it takes only fds and sizes** (yvo_hailo does not link to yvo_vulkan).
//
// ── Size contract (F0 checked exhaustively: 07_results/gpu_postprocess/evidence/f0_in_dmabuf.txt) ────
//   dma_map_dmabuf() gets the **actual dma-buf size** (the lseek(fd, 0, SEEK_END) value),
//   infer_async() gets **only hw_frame_size**.
//   If the transfer size is given as the actual dma-buf size, HailoRT returns HAILO_SUCCESS and then its internal worker
//   silently drops the request — **not a single callback arrives.** That is why drain() must have a deadline.
//
// ── Caller contract ─────────────────────────────────────────────────────────────────────────
//   · Call bind_slot() **exactly once** per slot right after construction. All slots must be bound before submit().
//   · The caller owns the fds. The runner does not close() them. However, while the runner is alive,
//     the fds and their dma-bufs must stay valid (the destructor calls dma_unmap_dmabuf).
//   · The on_done callback is invoked on a **HailoRT internal thread**. Keep it short, and inside it
//     **do not touch Vulkan** (no command submission, fence waits or queue work). Do not call submit()
//     directly from the callback either — doing the backpressure wait (wait_for_ongoing_callbacks_count_under) on the callback thread
//     can deadlock. Only push the slot number to a free queue and return.
//   · **When the callback is entered, the slot has already been released.** It is called only after all 7 buffer callbacks arrive,
//     so the NPU no longer touches that slot's input/output dma-bufs. Handing the slot to another thread from the callback
//     and resubmitting it immediately causes no race.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hailo/hailort.hpp>

#include "hailo/hailo_runner.hpp"   // yvo::HailoError (shared exception contract)

namespace yvo {

// Metadata for one raw output tensor. Derived entirely from HEF metadata (no fixture files are read).
struct RawTensorMeta {
    std::string stream_name;
    std::size_t hw_frame_size = 0;   // Transfer size. This is the value passed to infer_async
    int grid_h = 0;                  // 80 / 40 / 20
    int grid_w = 0;                  // Valid column count (shape.width).  80 / 40 / 20
    int w_padded = 0;                // Column count of the device layout (hw_shape.width). 80 / 40 / 24
    int stride = 0;                  // 8 / 16 / 32
    int features = 0;                // cls 80 / box 64
    float qp_scale = 0.0f;
    float qp_zp = 0.0f;
};

// Model info the runner read from the HEF. The caller (GPU post-processing glue) gets buffer sizes and decode parameters here.
struct RawModelInfo {
    std::string input_name;
    std::size_t input_frame_size = 0;   // 1,228,800 (NHCW 640x640x3)
    int input_w = 0, input_h = 0;
    // Numbering convention: cls(s) = 2s, box(s) = 2s+1  (s = 0,1,2 → stride 8,16,32)
    RawTensorMeta t[6];
    int num_classes = 80;
    int reg_max = 16;
    // Minimum get_async_max_queue_size() over the 7 input/output streams. The caller uses it to clamp the slot count.
    std::size_t min_queue = 0;
};

// Completion result for one frame. The callback is invoked on a HailoRT thread.
struct RawInferResult {
    int slot = -1;
    std::int64_t pts = 0;
    hailo_status status = HAILO_SUCCESS;
    double latency_ms = 0.0;   // submit (just before infer_async) ~ 7th buffer callback
};

using RawInferCallback = std::function<void(RawInferResult&&)>;

struct HailoRawRunnerConfig {
    std::string hef_path;  // Set by the caller (default path is YVO_DEFAULT_HEF in yvo_assets — the library does not know asset paths)
    std::uint16_t batch_size = 8;             // Scheduler burst size
    int num_slots = 8;                        // Number of frames that can be in flight at once. Clamped to min_queue
    std::chrono::milliseconds timeout{5000};  // drain() deadline (no unbounded waits)
    bool verbose = false;
};

class HailoRawRunner {
public:
    explicit HailoRawRunner(const HailoRawRunnerConfig& cfg);
    ~HailoRawRunner();   // Order: drain (deadline) → dma_unmap_dmabuf (per direction) → ng_ → vdevice_

    HailoRawRunner(const HailoRawRunner&) = delete;
    HailoRawRunner& operator=(const HailoRawRunner&) = delete;

    const RawModelInfo& info() const { return info_; }

    // Actual slot count: cfg.num_slots clamped to [1, info().min_queue].
    // The caller must call bind_slot() this many times.
    int num_slots() const { return num_slots_; }

    // Per-slot dma-buf registration. Call **exactly once** per slot right after construction.
    //   in_fd           : preprocessing output (V3D export). Mapped H2D
    //   in_map_size     : actual dma-buf size (lseek value). **Not the transfer size**
    //   out_fd[6]       : raw tensors (V3D export). Mapped D2H. Numbered in the same order as info().t[]
    //   out_map_size[6] : actual size of each dma-buf
    // The caller owns the fds. The runner does not close them.
    // Throws on failure (out-of-range slot, duplicate bind, insufficient size, dma_map_dmabuf failure).
    void bind_slot(int slot, int in_fd, std::size_t in_map_size,
                   const int out_fd[6], const std::size_t out_map_size[6]);

    // Submit one frame. The caller picks the slot (1:1 with the preprocessing slot).
    // Built-in backpressure: blocks on wait_for_ongoing_callbacks_count_under(num_slots).
    // The upper bound on in-flight frames is **exactly num_slots** (measured on HailoRT 4.24.0 — see the comment at the same spot in the .cpp).
    // on_done is invoked on the HailoRT callback thread — **keep it short and do not touch Vulkan inside it.**
    // When the callback is entered, the slot has already been released (another thread may resubmit it right away).
    // Resubmitting a slot that is still in flight throws HailoError(HAILO_INVALID_OPERATION).
    void submit(int slot, std::int64_t pts, RawInferCallback on_done);

    // Waits until the user callbacks of all in-flight frames have returned.
    // If not finished within cfg.timeout, HailoError(HAILO_TIMEOUT) — **never waits indefinitely.**
    void drain();

    // Chip temperature of the first physical device (max of ts0/ts1, °C). NaN on failure.
    float chip_temperature() const;

    // Number of in-flight frames (submitted and whose user callback has not yet returned)
    int in_flight() const { return in_flight_.load(); }

private:
    struct Slot {
        bool bound = false;
        int in_fd = -1;
        std::size_t in_map_size = 0;
        int out_fd[6] = {-1, -1, -1, -1, -1, -1};
        std::size_t out_map_size[6] = {0, 0, 0, 0, 0, 0};
        std::atomic<bool> busy{false};
        std::atomic<int> remaining{0};
        std::atomic<hailo_status> status{HAILO_SUCCESS};
        std::int64_t pts = 0;
        std::chrono::steady_clock::time_point t_submit;
        RawInferCallback on_done;
    };

    struct Mapping {
        int fd;
        std::size_t size;
        hailo_dma_buffer_direction_t dir;
    };

    void note_callback(int slot, hailo_status st);
    void finish(int slot);
    bool drain_for(std::chrono::milliseconds timeout);

    HailoRawRunnerConfig cfg_;
    RawModelInfo info_;
    std::unique_ptr<hailort::Hef> hef_;
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> ng_;
    hailort::InputStream* input_ = nullptr;
    // Output streams sorted by name. The mapping to tensor index i is held by info_.t[i].stream_name.
    std::vector<hailort::OutputStream*> outputs_;

    int num_slots_ = 1;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<Mapping> mapped_;

    std::atomic<int> in_flight_{0};
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace yvo
