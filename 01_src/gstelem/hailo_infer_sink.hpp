// yvohailosink: GStreamer sink running HailoRT inference on yvosandletterbox slot dma-bufs (research path, stage 2b)
//
//   sink: application/x-yvo-tensor, layout=nhcw, format=RGB8, width=640, height=640   (yvosandletterbox output only)
//
// Reproduces what `yvo_pipeline --mode dmabuf --layout nhcw [--postproc host]` does for inference + postprocessing:
// the same yvo::HailoRunner (InferModel async, ROUND_ROBIN scheduler, set_batch_size(batch), NHCW input format order,
// HEF-embedded NMS parsed by HailoRT on the host CPU + parse_nms_by_class inverse letterbox), the same
// map_dmabuf(fd, input_frame_size) per slot and the same per-frame submit. "batch" is the InferModel batch size (the
// scheduler groups frames), exactly like yvo_pipeline — the sink does NOT accumulate frames itself.
//
// GstBaseSink subclass, sync=false, async=true (prerolls), QoS off, enable-last-sample=false (a cached last sample
// would hold a slot buffer).
//   start()   : HailoRunner created (VDevice + configure), chip temperature t0. dump-dir → detections.jsonl opened.
//   render()  : validate the tensor meta, map the slot fd once (first sight; slot → fd), ref the GstBuffer, submit it
//               (blocks only on HailoRT backpressure, wait_for_async_ready), then write finished results in seq order.
//   callback  : HailoRT thread. Records NPU latency (runner: before run_async → completion) and e2e (completion −
//               TensorFrameInfo::arrival_mono_ns, CLOCK_MONOTONIC), queues the detections, then unrefs the buffer
//               (the slot returns to yvosandletterbox's ring). No file I/O here.
//   EOS       : waits for every in-flight completion (deadline), writes the remaining results, then lets the base
//               class post EOS → the application sees EOS only after the last NPU completion.
//   stop()    : drain (deadline) → last results → chip temperature t1 → HailoRunner destroyed (shutdown + dma-buf
//               unmap) → jsonl closed. After a successful drain no slot buffer is held any more; the unmap still
//               happens while yvosandletterbox owns its slot fds (a bin moves sinks to READY before upstream elements).
//               If the drain deadline passes, the runner is leaked on purpose (its destructor would wait forever) and
//               the error is logged.
//
// Properties
//   hef              (string)                  HEF path (required; the tools pass YVO_DEFAULT_HEF)
//   batch            (uint, 1..16, default 1)  InferModel batch size (scheduler burst), = yvo_pipeline --batch
//   score-threshold  (float, default -1)       <0 keeps the HEF value (0.2), = yvo_pipeline --score default
//   dump-dir         (string, default none)    writes <dir>/detections.jsonl in yvo_pipeline --dump-dir format
//   drain-timeout-ms (uint, default 10000)     EOS/stop wait for in-flight completions
//   frames, submitted, errors, detections, in-flight-max, mapped-slots (read-only counters)
//   npu-ms-mean, e2e-ms-mean (read-only)
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "core/frame_types.hpp"

typedef struct _GstPlugin GstPlugin;
typedef struct _GstElement GstElement;

namespace yvo::gst {

inline constexpr const char* kHailoSinkElementName = "yvohailosink";
// HailoRunner slots created by the sink (= max yvosandletterbox "slots"): runner slot index = ring slot index.
inline constexpr int kHailoSinkRunnerSlots = 32;

// One finished frame, delivered in seq order.
struct HailoSinkFrameResult {
    std::uint64_t seq = 0;           // TensorFrameInfo::seq (= frame index, same as yvo_pipeline's pts)
    int status = 0;                  // hailo_status of the completion (0 = HAILO_SUCCESS)
    std::vector<Detection> detections;  // source-frame pixel coordinates
    double npu_ms = 0.0;             // HailoRunner latency (run_async → completion callback)
    double e2e_ms = 0.0;             // arrival at yvosandletterbox → completion callback
};

// Called for every finished frame in seq order, from the streaming thread (render / EOS) or from stop().
// Keep it cheap. Set it before READY→PAUSED; pass an empty function to clear.
using HailoSinkResultCallback = std::function<void(const HailoSinkFrameResult&)>;

// Snapshot of the run statistics (reset in start(), kept after stop()).
struct HailoSinkStats {
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;     // results delivered in seq order
    std::uint64_t errors = 0;        // completions with status != HAILO_SUCCESS
    std::uint64_t detections = 0;
    std::vector<double> npu_ms, e2e_ms, gpu_ms;  // per completed frame / per submitted frame (gpu_ms from the meta)
    float hailo_t0 = 0.0f, hailo_t1 = 0.0f;      // chip temperature after configure / before runner release (NaN if unknown)
    int batch = 0;
    int in_flight_max = 0;
    int mapped_slots = 0;
    int ring_slots = 0;              // TensorFrameInfo::num_slots of the upstream ring
    std::size_t async_queue_size = 0;
    bool drain_timeout = false;      // EOS/stop deadline passed with frames still in flight
};

// Registers the element factory (plugin may be null: static registration). Returns false on failure.
bool register_hailo_sink(GstPlugin* plugin);

// Both return false if element is not a yvohailosink.
bool hailo_sink_set_result_callback(GstElement* element, HailoSinkResultCallback cb);
bool hailo_sink_get_stats(GstElement* element, HailoSinkStats& out);

} // namespace yvo::gst
