// InferStage: the "one frame → detection list" abstraction that main.cpp sees.
//
// There are two implementations.
//   HostNmsStage     : today's baseline path. Wraps `yvo::HailoRunner` (single HEF-embedded NMS output) as is.
//                      All post-processing (NMS parsing + letterbox inverse transform) is done by HailoRT on the **host CPU**.
//   GpuPostprocStage : `yvo::HailoRawRunner` (6 raw output streams) + `yvo::vk::YoloPostproc` (GPU pass 1)
//                      + `cpu_nms_greedy_by_class` (CPU suppression). Lives in gpu_postproc_stage.{hpp,cpp}.
//
// ── Thread discipline (this decides the whole design) ──────────────────────────────────────────────────
// `VkQueue` is an externally synchronized object and this project has only one queue (`VulkanContext::queue()`).
// `SandLetterbox::submit` and `YoloPostproc::submit` both call `vkQueueSubmit` →
// **All Vulkan work happens on the main thread only. No new threads are created.**
//   submit()   : main thread
//   pump()     : main thread. Pushes NPU completions into GPU post-processing and returns finished ones as `Done`
//   drain_one(): main thread. Finishes exactly one, **blocking** (the escape hatch when no slot is free)
//   Done callback: host calls it from the HailoRT callback thread, gpu from the main thread (inside pump/drain).
//                Either way it goes into `SlotExchange::complete()`, so main.cpp does not distinguish them.
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "core/frame_types.hpp"
#include "hailo/hailo_runner.hpp"

namespace yvo::pipeline {

class InferStage {
public:
    using Done = std::function<void(yvo::InferResult&&)>;

    virtual ~InferStage() = default;

    virtual std::size_t input_frame_size() const = 0;

    // In dmabuf mode, registers the slot's preprocessing output dma-buf with the stage.
    // Call once per slot before the first submit(). Not called in ptr mode.
    virtual void bind_input_dmabuf(int slot, int fd) = 0;

    // Source frame size. Call after opening the decoder and before the first submit() (for coordinate clamping).
    virtual void set_source_size(int /*w*/, int /*h*/) {}

    virtual void submit(const PreprocOutput& in, const Done& on_done) = 0;

    // Main thread. Does not block. No-op for host.
    virtual void pump() {}

    // Main thread. If GPU post-processing is in flight, finishes one **blocking** and returns true.
    // Otherwise does nothing and returns false. Always false for host.
    virtual bool drain_one() { return false; }

    // Main thread. The last-resort wait when all slots are on the NPU and there is nothing else to do.
    // On the gpu path **nobody wakes the `SlotExchange` condvar** (completion notices go into the stage-internal
    // ready_, and slots are only returned when the main thread calls pump()). So we must wait
    // here. Returns true once at least one completion has arrived — the caller then calls pump()/drain_one() again.
    // For host the HailoRT callback wakes `SlotExchange` directly, so false (= existing path unchanged).
    virtual bool wait_completion(std::chrono::milliseconds) { return false; }

    virtual void drain() = 0;

    virtual float chip_temperature() const = 0;

    virtual double last_post_ms() const { return -1.0; }   // GPU pass 1 ms
    virtual double last_nms_us() const { return -1.0; }    // CPU suppression µs
};

// ── Baseline path: HEF-embedded NMS + HailoRT host post-processing ──────────────────────────────────────────
// Keeps **exactly the same call order** as today's main.cpp (it is the regression reference generator).
class HostNmsStage final : public InferStage {
public:
    explicit HostNmsStage(const yvo::HailoRunnerConfig& cfg)
        : runner_(std::make_unique<yvo::HailoRunner>(cfg)) {}

    std::size_t input_frame_size() const override { return runner_->input_frame_size(); }

    void bind_input_dmabuf(int /*slot*/, int fd) override {
        runner_->map_dmabuf(fd, runner_->input_frame_size());
    }

    void submit(const PreprocOutput& in, const Done& on_done) override { runner_->submit(in, on_done); }

    void drain() override { runner_->drain(); }

    float chip_temperature() const override { return runner_->chip_temperature(); }

private:
    std::unique_ptr<yvo::HailoRunner> runner_;
};

} // namespace yvo::pipeline
