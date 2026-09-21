#include "hailo/hailo_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace yvo {

namespace {

constexpr std::size_t kPageSize = 4096;

std::string status_name(hailo_status s) {
    const char* msg = hailo_get_status_message(s);
    return std::string(msg ? msg : "UNKNOWN") + "(" + std::to_string(static_cast<int>(s)) + ")";
}

const char* layout_name(TensorLayout l) {
    return l == TensorLayout::Nhcw ? "NHCW" : "NHWC";
}

void check(hailo_status s, const char* where) {
    if (s != HAILO_SUCCESS) throw HailoError(where, s);
}

template <typename T>
T unwrap(hailort::Expected<T>&& e, const char* where) {
    if (!e) throw HailoError(where, e.status());
    return e.release();
}

std::shared_ptr<std::uint8_t> alloc_page_aligned(std::size_t size) {
    const std::size_t rounded = (size + kPageSize - 1) / kPageSize * kPageSize;
    void* p = std::aligned_alloc(kPageSize, rounded);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, rounded);
    return std::shared_ptr<std::uint8_t>(static_cast<std::uint8_t*>(p), [](std::uint8_t* q) { std::free(q); });
}

} // namespace

HailoError::HailoError(const std::string& where, hailo_status status)
    : std::runtime_error("HailoRT 실패: " + where + " -> " + status_name(status)), status_(status) {}

std::vector<Detection> parse_nms_by_class(const float* data, std::size_t bytes, int num_classes,
                                          int max_bboxes_per_class, const LetterboxParams& lb,
                                          int src_w, int src_h, float score_threshold) {
    std::vector<Detection> out;
    const std::size_t n_floats = bytes / sizeof(float);
    constexpr std::size_t kBoxFloats = sizeof(hailo_bbox_float32_t) / sizeof(float); // 5
    static_assert(sizeof(hailo_bbox_float32_t) == 5 * sizeof(float));

    const float dw = static_cast<float>(lb.dst_w);
    const float dh = static_cast<float>(lb.dst_h);
    const float inv_scale = 1.0f / lb.scale;
    const float max_x = static_cast<float>(src_w);
    const float max_y = static_cast<float>(src_h);

    std::size_t off = 0;
    for (int c = 0; c < num_classes; ++c) {
        if (off >= n_floats) break;
        const int count = static_cast<int>(data[off]);
        off += 1;
        if (count < 0 || count > max_bboxes_per_class || off + count * kBoxFloats > n_floats) {
            break;  // 손상된 버퍼: 여기서 중단
        }
        for (int i = 0; i < count; ++i) {
            hailo_bbox_float32_t b;
            std::memcpy(&b, data + off, sizeof(b));
            off += kBoxFloats;
            if (b.score < score_threshold) continue;
            Detection d;
            d.class_id = c;
            d.score = b.score;
            d.x1 = std::clamp((b.x_min * dw - lb.pad_x) * inv_scale, 0.0f, max_x);
            d.y1 = std::clamp((b.y_min * dh - lb.pad_y) * inv_scale, 0.0f, max_y);
            d.x2 = std::clamp((b.x_max * dw - lb.pad_x) * inv_scale, 0.0f, max_x);
            d.y2 = std::clamp((b.y_max * dh - lb.pad_y) * inv_scale, 0.0f, max_y);
            out.push_back(d);
        }
    }
    return out;
}

HailoRunner::HailoRunner(const HailoRunnerConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_slots <= 0) throw std::invalid_argument("num_slots must be > 0");

    // C++ 비동기 InferModel API는 스케줄러를 켠 상태를 전제로 한다(activate() 불필요).
    hailo_vdevice_params_t params;
    check(hailo_init_vdevice_params(&params), "hailo_init_vdevice_params");
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    vdevice_ = unwrap(hailort::VDevice::create(params), "VDevice::create");

    infer_model_ = unwrap(vdevice_->create_infer_model(cfg_.hef_path), "VDevice::create_infer_model");
    infer_model_->set_batch_size(cfg_.batch_size);

    if (infer_model_->inputs().size() != 1 || infer_model_->outputs().size() != 1) {
        throw std::runtime_error("HailoRunner: 입력 1개/출력 1개 모델만 지원");
    }
    auto input = unwrap(infer_model_->input(), "InferModel::input");
    auto output = unwrap(infer_model_->output(), "InferModel::output");
    input_name_ = input.name();
    output_name_ = output.name();

    input.set_format_type(HAILO_FORMAT_TYPE_UINT8);
    // NHCW는 yolov11m.hef의 장치 입력 형식과 같아 HailoRT의 CPU 전치(약 14 ms/frame)가 사라진다.
    // 반드시 configure() 전에 설정해야 한다.
    if (cfg_.input_layout == TensorLayout::Nhcw) {
        input.set_format_order(HAILO_FORMAT_ORDER_NHCW);
    }
    output.set_format_type(HAILO_FORMAT_TYPE_FLOAT32);

    if (!output.is_nms() || output.format().order != HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS) {
        throw std::runtime_error("HailoRunner: 출력이 HAILO_NMS_BY_CLASS가 아님");
    }
    if (cfg_.score_threshold >= 0.0f) {
        output.set_nms_score_threshold(cfg_.score_threshold);
        score_threshold_ = cfg_.score_threshold;
    } else {
        score_threshold_ = 0.0f;  // HEF 설정(0.2)이 이미 적용된 출력이므로 추가 필터 없음
    }
    const auto nms_shape = unwrap(output.get_nms_shape(), "InferStream::get_nms_shape");
    num_classes_ = static_cast<int>(nms_shape.number_of_classes);
    max_bboxes_per_class_ = static_cast<int>(nms_shape.max_bboxes_per_class);

    const auto ishape = input.shape();
    in_w_ = static_cast<int>(ishape.width);
    in_h_ = static_cast<int>(ishape.height);
    if (ishape.features != 3) throw std::runtime_error("HailoRunner: 입력 채널이 3이 아님");

    configured_ = std::make_unique<hailort::ConfiguredInferModel>(
        unwrap(infer_model_->configure(), "InferModel::configure"));

    // 포맷 설정 후 프레임 크기 확정
    input_frame_size_ = unwrap(infer_model_->input(), "InferModel::input").get_frame_size();
    output_frame_size_ = unwrap(infer_model_->output(), "InferModel::output").get_frame_size();
    async_queue_size_ = unwrap(configured_->get_async_queue_size(), "ConfiguredInferModel::get_async_queue_size");

    // 두 레이아웃 모두 같은 크기(640*640*3)여야 한다. 다르면 HEF나 포맷 설정이 예상과 다른 것이다.
    const std::size_t expected_input_size = static_cast<std::size_t>(in_w_) * static_cast<std::size_t>(in_h_) * 3;
    if (input_frame_size_ != expected_input_size) {
        throw std::runtime_error("HailoRunner: 입력 프레임 크기 " + std::to_string(input_frame_size_) +
                                 " B가 예상값 " + std::to_string(expected_input_size) + " B와 다름 (layout=" +
                                 layout_name(cfg_.input_layout) + ")");
    }

    slots_.reserve(cfg_.num_slots);
    for (int i = 0; i < cfg_.num_slots; ++i) {
        auto s = std::make_unique<Slot>();
        s->bindings = unwrap(configured_->create_bindings(), "ConfiguredInferModel::create_bindings");
        s->out_buf = alloc_page_aligned(output_frame_size_);
        auto out_stream = unwrap(s->bindings.output(output_name_), "Bindings::output");
        check(out_stream.set_buffer(hailort::MemoryView(s->out_buf.get(), output_frame_size_)),
              "Bindings::output.set_buffer");
        slots_.push_back(std::move(s));
    }
}

HailoRunner::~HailoRunner() {
    try {
        drain();
    } catch (...) {
    }
    if (configured_) {
        configured_->shutdown();
    }
    configured_.reset();
    std::lock_guard<std::mutex> lk(map_mtx_);
    for (const auto& [fd, size] : mapped_dmabufs_) {
        vdevice_->dma_unmap_dmabuf(fd, size, HAILO_DMA_BUFFER_DIRECTION_H2D);
    }
    mapped_dmabufs_.clear();
}

void HailoRunner::submit(const PreprocOutput& in, InferCallback cb) {
    if (in.slot < 0 || in.slot >= cfg_.num_slots) {
        throw std::out_of_range("HailoRunner::submit: slot 범위 밖 " + std::to_string(in.slot));
    }
    if (in.size != input_frame_size_) {
        throw std::invalid_argument("HailoRunner::submit: 입력 크기 불일치 " + std::to_string(in.size) +
                                    " != " + std::to_string(input_frame_size_));
    }
    if (in.layout != cfg_.input_layout) {
        throw std::invalid_argument(std::string("HailoRunner::submit: 입력 레이아웃 불일치 (입력=") +
                                    layout_name(in.layout) + ", 러너 설정=" + layout_name(cfg_.input_layout) + ")");
    }
    if (cfg_.input_layout == TensorLayout::Nhcw && in.dmabuf_fd < 0 && !cfg_.allow_host_ptr_nhcw) {
        throw std::invalid_argument(
            "HailoRunner::submit: NHCW 입력은 dma-buf로만 넘길 수 있다. NHCW에서는 HailoRT의 CPU 입력 변환이"
            " 없어 입력 버퍼를 직접 DMA 매핑하는데, V3D GEM mmap 포인터는 이 매핑에 실패한다"
            " (Ioctl HAILO_VDMA_BUFFER_MAP failed due to invalid address). dmabuf 모드를 쓰거나 입력을 NHWC로 둘 것.");
    }
    Slot& s = *slots_[in.slot];
    bool expected = false;
    if (!s.busy.compare_exchange_strong(expected, true)) {
        throw HailoError("HailoRunner::submit(slot 사용 중)", HAILO_INVALID_OPERATION);
    }

    try {
        auto in_stream = unwrap(s.bindings.input(input_name_), "Bindings::input");
        if (in.dmabuf_fd >= 0) {
            hailo_dma_buffer_t dbuf{in.dmabuf_fd, in.size};
            check(in_stream.set_dma_buffer(dbuf), "Bindings::input.set_dma_buffer");
        } else {
            if (!in.host_ptr) throw std::invalid_argument("HailoRunner::submit: host_ptr == nullptr");
            check(in_stream.set_buffer(hailort::MemoryView(in.host_ptr, in.size)), "Bindings::input.set_buffer");
        }

        // 백프레셔: 내부 큐에 자리가 날 때까지 대기
        check(configured_->wait_for_async_ready(cfg_.timeout, 1), "ConfiguredInferModel::wait_for_async_ready");

        in_flight_.fetch_add(1);
        s.t_submit = std::chrono::steady_clock::now();
        const int slot = in.slot;
        const std::int64_t pts = in.pts;
        const LetterboxParams lb = in.lb;
        int src_w = cfg_.src_width, src_h = cfg_.src_height;
        if (src_w <= 0) src_w = static_cast<int>(std::lround((lb.dst_w - 2 * lb.pad_x) / lb.scale));
        if (src_h <= 0) src_h = static_cast<int>(std::lround((lb.dst_h - 2 * lb.pad_y) / lb.scale));

        auto job_exp = configured_->run_async(
            s.bindings, [this, slot, pts, lb, src_w, src_h, cb = std::move(cb)](const hailort::AsyncInferCompletionInfo& info) {
                Slot& sl = *slots_[slot];
                InferResult r;
                r.slot = slot;
                r.pts = pts;
                r.status = info.status;
                r.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sl.t_submit).count();
                if (info.status == HAILO_SUCCESS) {
                    r.detections = parse_nms_by_class(reinterpret_cast<const float*>(sl.out_buf.get()), output_frame_size_,
                                                      num_classes_, max_bboxes_per_class_, lb, src_w, src_h, score_threshold_);
                }
                sl.busy.store(false);  // 슬롯 반납(사용자 콜백 전에 풀어 콜백 안에서 재제출 가능)
                if (cb) cb(std::move(r));
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    in_flight_.fetch_sub(1);
                }
                cv_.notify_all();
            });
        if (!job_exp) {
            in_flight_.fetch_sub(1);
            throw HailoError("ConfiguredInferModel::run_async", job_exp.status());
        }
        job_exp.release().detach();
    } catch (...) {
        s.busy.store(false);
        throw;
    }
}

void HailoRunner::drain() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [this] { return in_flight_.load() == 0; });
}

float HailoRunner::chip_temperature() const {
    auto devs = vdevice_->get_physical_devices();
    if (!devs || devs->empty()) return std::nanf("");
    auto t = devs->front().get().get_chip_temperature();
    if (!t) return std::nanf("");
    return std::max(t->ts0_temperature, t->ts1_temperature);
}

void HailoRunner::map_dmabuf(int fd, std::size_t size) {
    check(vdevice_->dma_map_dmabuf(fd, size, HAILO_DMA_BUFFER_DIRECTION_H2D), "VDevice::dma_map_dmabuf");
    std::lock_guard<std::mutex> lk(map_mtx_);
    mapped_dmabufs_.emplace_back(fd, size);
}

} // namespace yvo
