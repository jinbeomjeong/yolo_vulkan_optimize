#include "pipeline/gpu_postproc_stage.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

#include "ref/cpu_yolov8_decode.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::pipeline {

namespace {

using Clock = std::chrono::steady_clock;

// Bitwise comparison (count, order, class, anchor_key, score, coordinates). Floats are compared by bits, not ==.
bool same_bits(const std::vector<yvo::DetectionKeyed>& a, const std::vector<yvo::DetectionKeyed>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i];
        const auto& y = b[i];
        if (x.class_id != y.class_id || x.anchor_key != y.anchor_key) return false;
        const float fx[5] = {x.score, x.x1, x.y1, x.x2, x.y2};
        const float fy[5] = {y.score, y.x1, y.y1, y.x2, y.y2};
        if (std::memcmp(fx, fy, sizeof(fx)) != 0) return false;
    }
    return true;
}

// Actual dma-buf size. The mapping-size contract of `bind_slot()` is the **lseek value** (not the transfer size).
std::size_t dmabuf_size(int fd) {
    const off_t end = ::lseek(fd, 0, SEEK_END);
    if (end <= 0) throw std::runtime_error("lseek(dma-buf, SEEK_END) failed: fd " + std::to_string(fd));
    return static_cast<std::size_t>(end);
}

// Metadata the runner read from the HEF → geometry and quantization parameters used by the shader.
// Both sides use the same tensor numbering: cls(s) = 2s, box(s) = 2s+1.  **Does not read fixtures.json.**
yvo::vk::YoloTensorLayout layout_from_runner(const yvo::RawModelInfo& info) {
    yvo::vk::YoloTensorLayout L;
    L.num_classes = info.num_classes;
    L.reg_max = info.reg_max;
    for (int s = 0; s < yvo::vk::kYoloNumScales; ++s) {
        const yvo::RawTensorMeta& c = info.t[yvo::vk::yolo_cls_tensor(s)];
        const yvo::RawTensorMeta& b = info.t[yvo::vk::yolo_box_tensor(s)];
        if (c.features != info.num_classes || b.features != 4 * info.reg_max)
            throw std::runtime_error("tensor " + std::to_string(s) + " features do not match the cls/box convention: cls " +
                                     std::to_string(c.features) + ", box " + std::to_string(b.features));
        if (c.grid_h != b.grid_h || c.grid_w != b.grid_w || c.w_padded != b.w_padded || c.stride != b.stride)
            throw std::runtime_error("scale " + std::to_string(s) + " cls/box geometry mismatch");
        yvo::vk::YoloScaleSpec& d = L.s[s];
        d.grid_h = c.grid_h;
        d.grid_w = c.grid_w;
        d.w_padded = c.w_padded;
        d.stride = c.stride;
        d.cls_bytes = c.hw_frame_size;
        d.cls_scale = c.qp_scale;
        d.cls_zp = c.qp_zp;
        d.box_bytes = b.hw_frame_size;
        d.box_scale = b.qp_scale;
        d.box_zp = b.qp_zp;
    }
    return L;
}

} // namespace

GpuPostprocStage::GpuPostprocStage(yvo::vk::VulkanContext& ctx, const GpuPostprocStageConfig& cfg) : cfg_(cfg) {
    yvo::HailoRawRunnerConfig rc;
    rc.hef_path = cfg_.hef_path;
    rc.batch_size = cfg_.batch_size;
    rc.num_slots = cfg_.num_slots;
    rc.timeout = cfg_.timeout;
    rc.verbose = cfg_.verbose;
    raw_ = std::make_unique<yvo::HailoRawRunner>(rc);

    // The runner silently clamps num_slots to min_queue. The slot number space is shared with SandLetterbox and SlotExchange,
    // so on a mismatch we **stop here** (if slots are short, GPU slots are not bound to the NPU).
    if (raw_->num_slots() != cfg_.num_slots)
        throw std::runtime_error(
            "HailoRawRunner clamped the slot count: requested " + std::to_string(cfg_.num_slots) + ", actual " +
            std::to_string(raw_->num_slots()) + " (minimum get_async_max_queue_size over HEF streams = " +
            std::to_string(raw_->info().min_queue) + "). Lower --slots to " + std::to_string(raw_->info().min_queue) +
            " or less — the slot number space must not diverge from SandLetterbox");

    yvo::vk::YoloPostprocConfig pc;
    pc.num_slots = cfg_.num_slots;
    pc.max_dets = cfg_.max_dets;
    pc.local_size_x = cfg_.local_size_x;
    pc.local_size_y = cfg_.local_size_y;
    pc.score_thr = cfg_.score_thr;
    pc.input_alloc = cfg_.input_alloc;
    pc.input_sync = yvo::vk::YoloInputSync::Foreign;   // external memory written by Hailo PCIe DMA
    pc.checksum_pipeline = false;
    pc.verbose = cfg_.verbose;
    pc.enable_nms = cfg_.nms != NmsBackend::Cpu;
    pc.nms.iou_thr = cfg_.iou_thr;
    pc.nms.max_per_class = cfg_.max_per_class;
    pp_ = std::make_unique<yvo::vk::YoloPostproc>(ctx, layout_from_runner(raw_->info()), pc);

    const auto n = static_cast<std::size_t>(cfg_.num_slots);
    pts_.assign(n, 0);
    lb_.assign(n, yvo::LetterboxParams{});
    bound_.assign(n, false);
    latency_.assign(n, 0.0);
    status_.assign(n, HAILO_SUCCESS);
    t_done_.assign(n, Clock::time_point{});
}

GpuPostprocStage::~GpuPostprocStage() { raw_.reset(); }   // drain HailoRT callbacks while ready_* are alive (.hpp)

const char* GpuPostprocStage::alloc_name() const {
    switch (cfg_.input_alloc) {
        case yvo::vk::YoloInputAlloc::ExportDmaBuf: return "export";
        case yvo::vk::YoloInputAlloc::ImportHeap: return "heap";
        case yvo::vk::YoloInputAlloc::Plain: return "plain";
    }
    return "?";
}

void GpuPostprocStage::set_source_size(int w, int h) {
    src_w_ = w;
    src_h_ = h;
}

void GpuPostprocStage::bind_input_dmabuf(int slot, int fd) {
    if (slot < 0 || slot >= cfg_.num_slots) throw std::runtime_error("bind_input_dmabuf: slot out of range");
    int out_fd[yvo::vk::kYoloNumTensors];
    std::size_t out_sz[yvo::vk::kYoloNumTensors];
    for (int t = 0; t < yvo::vk::kYoloNumTensors; ++t) {
        out_fd[t] = pp_->raw_dmabuf_fd(slot, t);
        if (out_fd[t] < 0)
            throw std::runtime_error("YoloPostproc did not export a dma-buf (check input_alloc)");
        out_sz[t] = pp_->raw_dmabuf_size(slot, t);
    }
    // Both input and output are mapped at the **actual dma-buf size**. The transfer size is set by the runner from hw_frame_size.
    raw_->bind_slot(slot, fd, dmabuf_size(fd), out_fd, out_sz);
    bound_[static_cast<std::size_t>(slot)] = true;
}

void GpuPostprocStage::submit(const PreprocOutput& in, const Done& on_done) {
    const int slot = in.slot;
    if (slot < 0 || slot >= cfg_.num_slots) throw std::runtime_error("GpuPostprocStage::submit: slot out of range");
    if (!bound_[static_cast<std::size_t>(slot)])
        throw std::runtime_error("slot " + std::to_string(slot) + " was not bind_input_dmabuf'd");
    if (src_w_ <= 0 || src_h_ <= 0)
        throw std::runtime_error("call set_source_size() first (coordinate clamping needs the source size)");
    done_ = on_done;
    pts_[static_cast<std::size_t>(slot)] = in.pts;
    lb_[static_cast<std::size_t>(slot)] = in.lb;
    raw_->submit(slot, in.pts, [this](yvo::RawInferResult&& r) {
        // ⚠ This is the HailoRT callback thread. No Vulkan calls, re-submits, or GPU memory reads.
        {
            std::lock_guard<std::mutex> lk(ready_mtx_);
            ready_.push_back(RawDone{r.slot, r.status, r.latency_ms, Clock::now()});
        }
        ready_cv_.notify_all();
    });
}

void GpuPostprocStage::push_ready_to_gpu() {
    std::vector<RawDone> batch;
    {
        std::lock_guard<std::mutex> lk(ready_mtx_);
        batch.swap(ready_);
    }
    for (const RawDone& d : batch) {
        const auto s = static_cast<std::size_t>(d.slot);
        latency_[s] = d.latency_ms;
        status_[s] = d.status;
        t_done_[s] = d.t_done;
        if (d.status != HAILO_SUCCESS) {
            // Raw tensors of a frame the NPU failed on are not trustworthy. Return it immediately without sending it to the GPU.
            yvo::InferResult r;
            r.slot = d.slot;
            r.pts = pts_[s];
            r.status = d.status;
            r.latency_ms = d.latency_ms;
            if (done_) done_(std::move(r));
            continue;
        }
        pp_->submit(d.slot, lb_[s], src_w_, src_h_);
        pp_q_.push_back(d.slot);
    }
}

void GpuPostprocStage::finish(int slot) {
    const auto s = static_cast<std::size_t>(slot);
    last_post_ms_ = pp_->last_pass1_ms();
    post_ms_.push_back(last_post_ms_);
    const std::uint32_t raw_n = pp_->raw_count(slot);
    cand_.push_back(static_cast<double>(raw_n));
    if (pp_->overflow(slot)) ++overflow_frames_;

    // CPU suppression (Cpu backend, GPU fallback, Verify reference). last_nms_us_ is updated only on this path.
    auto cpu_nms = [&]() {
        std::vector<yvo::DetectionKeyed> c = pp_->read_candidates(slot);
        const auto n0 = Clock::now();
        yvo::cpu_nms_greedy_by_class(c, cfg_.iou_thr, cfg_.max_per_class);
        last_nms_us_ = std::chrono::duration<double, std::micro>(Clock::now() - n0).count();
        nms_us_.push_back(last_nms_us_);
        return c;
    };

    std::vector<yvo::DetectionKeyed> cand;
    if (cfg_.nms == NmsBackend::Cpu) {
        cand = cpu_nms();
    } else {
        last_gpu_nms_ms_ = pp_->last_nms_ms();
        gpu_nms_ms_.push_back(last_gpu_nms_ms_);
        std::vector<yvo::DetectionKeyed> gpu;
        const bool gpu_ok = pp_->read_nms(slot, gpu);
        if (!gpu_ok) {
            const std::uint32_t why = pp_->nms_fallback_reason(slot);
            ++nms_fallback_frames_;
            if (why & yvo::vk::kNmsFallbackIouGuard) ++nms_fallback_guard_;
            if (why & (yvo::vk::kNmsFallbackCapacity | yvo::vk::kNmsFallbackBadClass)) ++nms_fallback_capacity_;
        }
        if (cfg_.nms == NmsBackend::Verify) {
            cand = cpu_nms();
            if (gpu_ok && !same_bits(gpu, cand)) ++nms_verify_mismatch_;
        } else {
            last_nms_us_ = -1.0;
            cand = gpu_ok ? std::move(gpu) : cpu_nms();
        }
    }

    hold_ms_.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t_done_[s]).count());

    yvo::InferResult r;
    r.slot = slot;
    r.pts = pts_[s];
    r.status = status_[s];
    r.latency_ms = latency_[s];
    r.detections.reserve(cand.size());
    for (const auto& d : cand) r.detections.push_back(static_cast<const yvo::Detection&>(d));
    if (done_) done_(std::move(r));
}

void GpuPostprocStage::pump() {
    push_ready_to_gpu();
    while (!pp_q_.empty() && pp_->ready(pp_q_.front())) {
        const int slot = pp_q_.front();
        pp_q_.pop_front();
        pp_->wait(slot);   // fence is already signaled
        finish(slot);
    }
}

bool GpuPostprocStage::drain_one() {
    push_ready_to_gpu();
    if (pp_q_.empty()) return false;
    const int slot = pp_q_.front();
    pp_q_.pop_front();
    pp_->wait(slot);   // blocking
    finish(slot);
    return true;
}

// Wait for when all slots are on the NPU. Only the main thread can wake the `SlotExchange` condvar
// (Done is called inside pump()), so without waiting here the main thread would sleep until the 5 s timeout.
bool GpuPostprocStage::wait_completion(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(ready_mtx_);
    if (!ready_.empty()) return true;
    if (raw_->in_flight() == 0) return false;
    return ready_cv_.wait_for(lk, timeout, [this] { return !ready_.empty(); });
}

void GpuPostprocStage::drain() {
    raw_->drain();         // returns all internal callbacks of in-flight NPU work
    push_ready_to_gpu();   // remaining completions to the GPU
    while (!pp_q_.empty()) {
        const int slot = pp_q_.front();
        pp_q_.pop_front();
        pp_->wait(slot);
        finish(slot);
    }
}

} // namespace yvo::pipeline
