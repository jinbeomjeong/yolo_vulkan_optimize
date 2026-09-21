// GpuPostprocStage: 6 raw output streams (NPU) → GPU pass 1 → suppression (GPU EfficientNMS or CPU greedy).
//
// Where suppression runs is chosen with `NmsBackend` (--nms):
//   Gpu    : `efficient_nms.comp` in the same command buffer as pass 1. The host reads only the NMS result.
//            If the shader reports a fallback reason (candidates > kMaxSelected, IoU guard), only that frame is handled on the CPU.
//   Cpu    : the old path. Reads candidates and suppresses with `cpu_nms_greedy_by_class()`.
//   Verify : runs both, compares bit-exactly and counts mismatches. The emitted result is the CPU one.
//
// Replaces host NMS (the 2.2~3.2 ms/frame fixed cost HailoRT spends on the CPU) with
//   GPU pass 1 (`02_shaders/yolo_decode.comp`, measured 0.56 ms) + CPU suppression (tens of candidates, 3~11 µs).
// Both parts (HailoRawRunner, YoloPostproc) are individually verified; this file is **glue only**.
//
// ── Stage flow ─────────────────────────────────────────────────────────────────────────────────
//   submit(out, cb)  [main]  record pts_/lb_ → raw_->submit(slot, pts, internal callback)
//   internal callback  [HailoRT thread]  push {slot,status,latency} to ready_. **That is all.**
//                              (runner contract: no Vulkan calls, re-submit, or bulk GPU memory reads in the callback)
//   pump()           [main]  drain ready_ → pp_->submit(slot, lb, src_w, src_h) → push pp_q_
//                            if the front of pp_q_ is ready → wait → read_candidates → cpu_nms → Done
//   drain_one()      [main]  finishes the front of pp_q_ with a **blocking** wait
//
// ── Destruction order ─────────────────────────────────────────────────────────────────────────────────
// The runner keeps the dma-buf fds mapped, so **the runner must die first.**
// Members are declared pp_ → raw_ and destroyed in reverse, so raw_ goes first.
// Input fds are owned by the caller (FdCloser in main.cpp), output fds by YoloPostproc.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hailo/hailo_raw_runner.hpp"
#include "pipeline/infer_stage.hpp"
#include "vulkan/yolo_postproc.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::pipeline {

enum class NmsBackend { Gpu, Cpu, Verify };

struct GpuPostprocStageConfig {
    std::string hef_path;  // set by the caller (default path is YVO_DEFAULT_HEF from yvo_assets — the library does not know asset paths)
    std::uint16_t batch_size = 8;
    int num_slots = 8;
    float score_thr = 0.2f;    // same value as the HEF YOLOV8 post-process
    float iou_thr = 0.7f;      // HEF setting
    int max_per_class = 100;   // HEF setting
    int max_dets = 4096;
    std::uint32_t local_size_x = 10;   // measured choice 10x6 from yolo_postproc.hpp
    std::uint32_t local_size_y = 6;
    yvo::vk::YoloInputAlloc input_alloc = yvo::vk::YoloInputAlloc::ExportDmaBuf;
    std::chrono::milliseconds timeout{5000};
    bool verbose = false;
    NmsBackend nms = NmsBackend::Gpu;
};

class GpuPostprocStage final : public InferStage {
public:
    GpuPostprocStage(yvo::vk::VulkanContext& ctx, const GpuPostprocStageConfig& cfg);
    ~GpuPostprocStage() override;

    GpuPostprocStage(const GpuPostprocStage&) = delete;
    GpuPostprocStage& operator=(const GpuPostprocStage&) = delete;

    std::size_t input_frame_size() const override { return raw_->info().input_frame_size; }
    void bind_input_dmabuf(int slot, int fd) override;
    void set_source_size(int w, int h) override;
    void submit(const PreprocOutput& in, const Done& on_done) override;
    void pump() override;
    bool drain_one() override;
    bool wait_completion(std::chrono::milliseconds timeout) override;
    void drain() override;
    float chip_temperature() const override { return raw_->chip_temperature(); }

    double last_post_ms() const override { return last_post_ms_; }
    // CPU suppression time. With the Gpu backend it is updated only on fallback frames (otherwise -1).
    double last_nms_us() const override { return last_nms_us_; }
    double last_gpu_nms_ms() const { return last_gpu_nms_ms_; }   // efficient_nms.comp span (-1 for Cpu)

    // ── Accumulated stats for RESULT_CSV2 (updated and read only on the main thread) ──
    const std::vector<double>& post_ms() const { return post_ms_; }
    const std::vector<double>& nms_us() const { return nms_us_; }
    const std::vector<double>& cand() const { return cand_; }
    // From NPU completion callback arrival → slot return. If large, slots are stuck in the post-processing queue and NPU occupancy drops.
    const std::vector<double>& hold_ms() const { return hold_ms_; }
    long overflow_frames() const { return overflow_frames_; }
    const std::vector<double>& gpu_nms_ms() const { return gpu_nms_ms_; }
    long nms_fallback_frames() const { return nms_fallback_frames_; }       // total regardless of reason
    long nms_fallback_capacity() const { return nms_fallback_capacity_; }   // candidates > kMaxSelected etc.
    long nms_fallback_guard() const { return nms_fallback_guard_; }         // IoU guard
    long nms_verify_mismatch() const { return nms_verify_mismatch_; }       // Verify: number of frames where GPU != CPU
    NmsBackend nms_backend() const { return cfg_.nms; }
    const char* alloc_name() const;

private:
    struct RawDone {
        int slot = -1;
        hailo_status status = HAILO_SUCCESS;
        double latency_ms = 0.0;
        std::chrono::steady_clock::time_point t_done;   // callback arrival time
    };

    void finish(int slot);   // turns a slot whose pp_->wait() finished into Done via the NMS result (or CPU suppression)
    void push_ready_to_gpu();   // drains ready_ and hands it to pp_

    // Declaration order = construction order. Destruction is reversed, so raw_ dies before pp_ (the runner holds the fd mappings).
    GpuPostprocStageConfig cfg_;
    std::unique_ptr<yvo::vk::YoloPostproc> pp_;
    std::unique_ptr<yvo::HailoRawRunner> raw_;

    int src_w_ = 0, src_h_ = 0;
    std::vector<std::int64_t> pts_;
    std::vector<yvo::LetterboxParams> lb_;
    std::vector<bool> bound_;
    Done done_;

    std::mutex ready_mtx_;
    std::condition_variable ready_cv_;  // HailoRT callback → wake the main thread
    std::vector<RawDone> ready_;        // HailoRT callback thread → main thread
    std::vector<double> latency_;       // per-slot NPU latency (read in finish)
    std::vector<hailo_status> status_;  // per-slot NPU status
    std::deque<int> pp_q_;              // slots queued on GPU pass 1 (submit order)

    std::vector<double> post_ms_, nms_us_, cand_, hold_ms_, gpu_nms_ms_;
    std::vector<std::chrono::steady_clock::time_point> t_done_;   // per-slot callback arrival time
    long overflow_frames_ = 0;
    long nms_fallback_frames_ = 0, nms_fallback_capacity_ = 0, nms_fallback_guard_ = 0, nms_verify_mismatch_ = 0;
    double last_post_ms_ = -1.0, last_nms_us_ = -1.0, last_gpu_nms_ms_ = -1.0;
};

} // namespace yvo::pipeline
