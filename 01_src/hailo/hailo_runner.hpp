// HailoRT InferModel async inference runner (yolov11m.hef, NMS BY CLASS output)
//
// Flow: VDevice(ROUND_ROBIN scheduler) → create_infer_model(hef) → set_batch_size → configure()
//       → per-slot Bindings (output buffers owned by the runner) → wait_for_async_ready + run_async on each submit()
//       → in the completion callback (HailoRT internal thread): NMS parsing + inverse letterbox → user callback.
//
// Input layout (HailoRunnerConfig::input_layout):
//   - Nhwc (default): pass packed RGB888 and HailoRT transposes NHWC→NHCW on the CPU (about 14 ms/frame).
//   - Nhcw: matches the device input format, so the CPU conversion disappears. In exchange HailoRT DMA-maps the input buffer directly,
//     so it can only be used with dma-buf input. A V3D GEM mmap pointer fails to map
//     (Ioctl HAILO_VDMA_BUFFER_MAP failed due to invalid address).
//     The check can be relaxed with allow_host_ptr_nhcw only when testing with plain malloc buffers.
//
// Slot contract: PreprocOutput.slot (0..num_slots-1) is used directly as the runner slot.
//   - A submitted slot is "in use" until the user callback is invoked.
//   - Once the user callback is invoked, HailoRT no longer reads that slot's input buffer (host_ptr or dma-buf).
//     In other words, the callback is the slot release point. After the callback returns, the same slot can be submit()ted again.
//   - Calling submit() again on a slot in use throws HAILO_INVALID_OPERATION.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <hailo/hailort.hpp>

#include "core/frame_types.hpp"

namespace yvo {

// Exception for a failed HailoRT call. what() contains the call site and the status code name.
class HailoError : public std::runtime_error {
public:
    HailoError(const std::string& where, hailo_status status);
    hailo_status status() const { return status_; }

private:
    hailo_status status_;
};

struct HailoRunnerConfig {
    std::string hef_path;  // Set by the caller (default path is YVO_DEFAULT_HEF in yvo_assets — the library does not know asset paths)
    // Input tensor layout the runner receives. With Nhcw, set_format_order(HAILO_FORMAT_ORDER_NHCW)
    // removes HailoRT's CPU input conversion (dma-buf input only).
    TensorLayout input_layout = TensorLayout::Nhwc;
    // Allow host_ptr submission even with Nhcw (for plain malloc buffer experiments). Must not be used with V3D GEM pointers.
    bool allow_host_ptr_nhcw = false;
    std::uint16_t batch_size = 1;      // Scheduler burst size (1/4/8 recommended)
    int num_slots = kNumSlots;         // Max frames that can be in flight at once. Efficient when >= batch_size
    float score_threshold = -1.0f;     // <0 keeps the HEF value (0.2). >=0 sets it via set_nms_score_threshold
    int src_width = 0;                 // Source size for clamping inverse-mapped coordinates. 0 = derive from LetterboxParams
    int src_height = 0;
    std::chrono::milliseconds timeout{2000}; // wait_for_async_ready timeout
};

// Completion result. The callback runs on the HailoRT callback thread, so do no heavy work in it.
struct InferResult {
    int slot = -1;
    std::int64_t pts = 0;
    hailo_status status = HAILO_SUCCESS;
    std::vector<Detection> detections;  // Source-frame pixel coordinates
    double latency_ms = 0.0;            // submit (just before run_async) ~ completion callback (includes queue wait)
    // Per-frame time points for yvo_pipeline --frame-log. Default-constructed (epoch) = not recorded.
    std::chrono::steady_clock::time_point t_npu_done{};    // NPU completion callback arrival
    std::chrono::steady_clock::time_point t_post_done{};   // post-processing done (GpuPostprocStage only)
};

using InferCallback = std::function<void(InferResult&&)>;

// NMS BY CLASS float buffer parsing + inverse letterbox (public so GTest can use it directly)
// Layout (hailort.h HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS):
//   For class c = 0..num_classes-1 in order: [float bbox_count, hailo_bbox_float32_t × bbox_count]
//   hailo_bbox_float32_t = {y_min, x_min, y_max, x_max, score}, coordinates normalized to 0~1 relative to the input (dst_w×dst_h).
// Source coordinate = (normalized × dst - pad) / scale, clamped to [0, src_w]×[0, src_h].
std::vector<Detection> parse_nms_by_class(const float* data, std::size_t bytes, int num_classes,
                                          int max_bboxes_per_class, const LetterboxParams& lb,
                                          int src_w, int src_h, float score_threshold);

class HailoRunner {
public:
    explicit HailoRunner(const HailoRunnerConfig& cfg);
    ~HailoRunner();  // Waits for all in-flight jobs, then shuts down and unmaps dma-bufs

    HailoRunner(const HailoRunner&) = delete;
    HailoRunner& operator=(const HailoRunner&) = delete;

    // Submit one input frame. The slot is given by in.slot.
    //  - in.dmabuf_fd >= 0: set_dma_buffer({fd, size}) (stage 2, map_dmabuf beforehand recommended)
    //  - otherwise: set_buffer(MemoryView(host_ptr, size)) (stage 1). host_ptr should be 4KB-aligned.
    // When the pipeline is full, blocks on wait_for_async_ready (backpressure). On timeout, HailoError(HAILO_TIMEOUT).
    void submit(const PreprocOutput& in, InferCallback cb);

    // Wait until all in-flight jobs complete (user callbacks returned)
    void drain();

    // For stage 2: pre-map the dma-buf for H2D (removes per-frame mapping cost). Released automatically in the destructor.
    void map_dmabuf(int fd, std::size_t size);

    // Model info
    std::size_t input_frame_size() const { return input_frame_size_; }
    std::size_t output_frame_size() const { return output_frame_size_; }
    int input_width() const { return in_w_; }
    int input_height() const { return in_h_; }
    int num_classes() const { return num_classes_; }
    int max_bboxes_per_class() const { return max_bboxes_per_class_; }
    int num_slots() const { return cfg_.num_slots; }
    std::size_t async_queue_size() const { return async_queue_size_; }
    int in_flight() const { return in_flight_.load(); }

    // Chip temperature of the first physical device (max of ts0/ts1, °C). NaN on failure.
    float chip_temperature() const;

private:
    struct Slot {
        hailort::ConfiguredInferModel::Bindings bindings;
        std::shared_ptr<std::uint8_t> out_buf;  // Page-aligned, owned by the runner
        std::atomic<bool> busy{false};
        std::chrono::steady_clock::time_point t_submit;
    };

    HailoRunnerConfig cfg_;
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::InferModel> infer_model_;
    std::unique_ptr<hailort::ConfiguredInferModel> configured_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::string input_name_;
    std::string output_name_;
    std::size_t input_frame_size_ = 0;
    std::size_t output_frame_size_ = 0;
    std::size_t async_queue_size_ = 0;
    int in_w_ = 0, in_h_ = 0;
    int num_classes_ = 0;
    int max_bboxes_per_class_ = 0;
    float score_threshold_ = 0.0f;

    std::atomic<int> in_flight_{0};
    std::mutex mtx_;
    std::condition_variable cv_;

    std::mutex map_mtx_;
    std::vector<std::pair<int, std::size_t>> mapped_dmabufs_;
};

} // namespace yvo
