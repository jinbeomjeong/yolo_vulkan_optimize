#include "pipeline/gpu_postproc_stage.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

#include "ref/cpu_yolov8_decode.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::pipeline {

namespace {

using Clock = std::chrono::steady_clock;

// dma-buf 실크기. `bind_slot()` 의 매핑 크기 계약은 **lseek 값**이다(전송 크기가 아니다).
std::size_t dmabuf_size(int fd) {
    const off_t end = ::lseek(fd, 0, SEEK_END);
    if (end <= 0) throw std::runtime_error("lseek(dma-buf, SEEK_END) 실패: fd " + std::to_string(fd));
    return static_cast<std::size_t>(end);
}

// 러너가 HEF 에서 읽어낸 메타 → 셰이더가 쓰는 기하·양자화 파라미터.
// 텐서 번호 규약은 양쪽이 같다: cls(s) = 2s, box(s) = 2s+1.  **fixtures.json 을 읽지 않는다.**
yvo::vk::YoloTensorLayout layout_from_runner(const yvo::RawModelInfo& info) {
    yvo::vk::YoloTensorLayout L;
    L.num_classes = info.num_classes;
    L.reg_max = info.reg_max;
    for (int s = 0; s < yvo::vk::kYoloNumScales; ++s) {
        const yvo::RawTensorMeta& c = info.t[yvo::vk::yolo_cls_tensor(s)];
        const yvo::RawTensorMeta& b = info.t[yvo::vk::yolo_box_tensor(s)];
        if (c.features != info.num_classes || b.features != 4 * info.reg_max)
            throw std::runtime_error("텐서 " + std::to_string(s) + " 의 features 가 cls/box 규약과 다르다: cls " +
                                     std::to_string(c.features) + ", box " + std::to_string(b.features));
        if (c.grid_h != b.grid_h || c.grid_w != b.grid_w || c.w_padded != b.w_padded || c.stride != b.stride)
            throw std::runtime_error("스케일 " + std::to_string(s) + " 의 cls/box 기하가 어긋난다");
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

    // 러너는 num_slots 를 min_queue 로 조용히 클램프한다. 슬롯 번호 공간은 SandLetterbox·SlotExchange 와
    // 공유하므로 어긋나면 **여기서 멈춘다**(슬롯이 모자라면 GPU 슬롯이 NPU 에 바인딩되지 않는다).
    if (raw_->num_slots() != cfg_.num_slots)
        throw std::runtime_error(
            "HailoRawRunner 가 슬롯 수를 클램프했다: 요청 " + std::to_string(cfg_.num_slots) + ", 실제 " +
            std::to_string(raw_->num_slots()) + " (HEF 스트림의 get_async_max_queue_size 최솟값 = " +
            std::to_string(raw_->info().min_queue) + "). --slots 를 " + std::to_string(raw_->info().min_queue) +
            " 이하로 낮춰라 — 슬롯 번호 공간이 SandLetterbox 와 어긋나면 안 된다");

    yvo::vk::YoloPostprocConfig pc;
    pc.num_slots = cfg_.num_slots;
    pc.max_dets = cfg_.max_dets;
    pc.local_size_x = cfg_.local_size_x;
    pc.local_size_y = cfg_.local_size_y;
    pc.score_thr = cfg_.score_thr;
    pc.input_alloc = cfg_.input_alloc;
    pc.input_sync = yvo::vk::YoloInputSync::Foreign;   // Hailo PCIe DMA 가 쓴 외부 메모리다
    pc.checksum_pipeline = false;
    pc.verbose = cfg_.verbose;
    pp_ = std::make_unique<yvo::vk::YoloPostproc>(ctx, layout_from_runner(raw_->info()), pc);

    const auto n = static_cast<std::size_t>(cfg_.num_slots);
    pts_.assign(n, 0);
    lb_.assign(n, yvo::LetterboxParams{});
    bound_.assign(n, false);
    latency_.assign(n, 0.0);
    status_.assign(n, HAILO_SUCCESS);
    t_done_.assign(n, Clock::time_point{});
}

GpuPostprocStage::~GpuPostprocStage() = default;

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
    if (slot < 0 || slot >= cfg_.num_slots) throw std::runtime_error("bind_input_dmabuf: 슬롯 범위 밖");
    int out_fd[yvo::vk::kYoloNumTensors];
    std::size_t out_sz[yvo::vk::kYoloNumTensors];
    for (int t = 0; t < yvo::vk::kYoloNumTensors; ++t) {
        out_fd[t] = pp_->raw_dmabuf_fd(slot, t);
        if (out_fd[t] < 0)
            throw std::runtime_error("YoloPostproc 이 dma-buf 를 export 하지 않았다 (input_alloc 확인)");
        out_sz[t] = pp_->raw_dmabuf_size(slot, t);
    }
    // 입력·출력 모두 **dma-buf 실크기**로 매핑한다. 전송 크기는 러너가 hw_frame_size 로 정한다.
    raw_->bind_slot(slot, fd, dmabuf_size(fd), out_fd, out_sz);
    bound_[static_cast<std::size_t>(slot)] = true;
}

void GpuPostprocStage::submit(const PreprocOutput& in, const Done& on_done) {
    const int slot = in.slot;
    if (slot < 0 || slot >= cfg_.num_slots) throw std::runtime_error("GpuPostprocStage::submit: 슬롯 범위 밖");
    if (!bound_[static_cast<std::size_t>(slot)])
        throw std::runtime_error("슬롯 " + std::to_string(slot) + " 이 bind_input_dmabuf 되지 않았다");
    if (src_w_ <= 0 || src_h_ <= 0)
        throw std::runtime_error("set_source_size()를 먼저 불러야 한다 (좌표 클램프에 원본 크기가 필요하다)");
    done_ = on_done;
    pts_[static_cast<std::size_t>(slot)] = in.pts;
    lb_[static_cast<std::size_t>(slot)] = in.lb;
    raw_->submit(slot, in.pts, [this](yvo::RawInferResult&& r) {
        // ⚠ HailoRT 콜백 스레드다. Vulkan 호출·재submit·GPU 메모리 읽기 전부 금지.
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
            // NPU 가 실패한 프레임은 원시 텐서가 신뢰할 수 없다. GPU 로 보내지 않고 바로 돌려준다.
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

    std::vector<yvo::DetectionKeyed> cand = pp_->read_candidates(slot);
    const auto n0 = Clock::now();
    yvo::cpu_nms_greedy_by_class(cand, cfg_.iou_thr, cfg_.max_per_class);
    last_nms_us_ = std::chrono::duration<double, std::micro>(Clock::now() - n0).count();
    nms_us_.push_back(last_nms_us_);

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
        pp_->wait(slot);   // 이미 신호된 펜스다
        finish(slot);
    }
}

bool GpuPostprocStage::drain_one() {
    push_ready_to_gpu();
    if (pp_q_.empty()) return false;
    const int slot = pp_q_.front();
    pp_q_.pop_front();
    pp_->wait(slot);   // 블로킹
    finish(slot);
    return true;
}

// 슬롯이 전부 NPU 에 있을 때의 대기. `SlotExchange` 의 condvar 는 메인 스레드만 깨울 수 있으므로
// (pump() 안에서 Done 을 부른다) 여기서 기다리지 않으면 메인 스레드가 5 초 타임아웃까지 잠든다.
bool GpuPostprocStage::wait_completion(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(ready_mtx_);
    if (!ready_.empty()) return true;
    if (raw_->in_flight() == 0) return false;
    return ready_cv_.wait_for(lk, timeout, [this] { return !ready_.empty(); });
}

void GpuPostprocStage::drain() {
    raw_->drain();         // 비행 중인 NPU 작업의 내부 콜백까지 전부 반환
    push_ready_to_gpu();   // 남은 완료분을 GPU 로
    while (!pp_q_.empty()) {
        const int slot = pp_q_.front();
        pp_q_.pop_front();
        pp_->wait(slot);
        finish(slot);
    }
}

} // namespace yvo::pipeline
