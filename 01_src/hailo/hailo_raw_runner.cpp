#include "hailo/hailo_raw_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "ref/cpu_yolov8_decode.hpp"   // kDflRegMax / kDflSides (레이아웃 파생 상수)

namespace yvo {

namespace {

constexpr int kNumTensors = 6;
constexpr int kInputSide = 640;

void check(hailo_status s, const char* where) {
    if (s != HAILO_SUCCESS) throw HailoError(where, s);
}

template <typename T>
T unwrap(hailort::Expected<T>&& e, const char* where) {
    if (!e) throw HailoError(where, e.status());
    return e.release();
}

} // namespace

// ---------------------------------------------------------------- 생성

HailoRawRunner::HailoRawRunner(const HailoRawRunnerConfig& cfg) : cfg_(cfg) {
    hef_ = std::make_unique<hailort::Hef>(unwrap(hailort::Hef::create(cfg_.hef_path), "Hef::create"));

    hailo_vdevice_params_t vparams;
    check(hailo_init_vdevice_params(&vparams), "hailo_init_vdevice_params");
    vparams.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    vdevice_ = unwrap(hailort::VDevice::create(vparams), "VDevice::create");

    auto params_map = unwrap(vdevice_->create_configure_params(*hef_), "VDevice::create_configure_params");
    for (auto& kv : params_map) {
        kv.second.batch_size = cfg_.batch_size;
        // 7개 스트림 전부 ASYNC. infer_async 의 전제다.
        for (auto& sp : kv.second.stream_params_by_name) {
            sp.second.flags = HAILO_STREAM_FLAGS_ASYNC;
        }
    }
    auto ngs = unwrap(vdevice_->configure(*hef_, params_map), "VDevice::configure");
    if (ngs.size() != 1) throw std::runtime_error("HailoRawRunner: network group이 1개가 아니다");
    ng_ = ngs[0];

    auto in_refs = ng_->get_input_streams();
    auto out_refs = ng_->get_output_streams();
    if (in_refs.size() != 1) throw std::runtime_error("HailoRawRunner: 입력 스트림이 1개가 아니다");
    if (out_refs.size() != static_cast<std::size_t>(kNumTensors))
        throw std::runtime_error("HailoRawRunner: 원시 출력 스트림이 6개가 아니다: " +
                                 std::to_string(out_refs.size()));
    input_ = &in_refs[0].get();
    for (auto& r : out_refs) outputs_.push_back(&r.get());
    // 이름 순으로 고정해 스트림 순서를 실행 간에 결정적으로 만든다(프로브 RawSession 과 같은 규칙).
    std::sort(outputs_.begin(), outputs_.end(),
              [](const hailort::OutputStream* a, const hailort::OutputStream* b) { return a->name() < b->name(); });

    // ── 입력 메타
    {
        const hailo_stream_info_t si = input_->get_info();
        info_.input_name = input_->name();
        info_.input_frame_size = input_->get_frame_size();
        info_.input_h = static_cast<int>(si.hw_shape.height);
        info_.input_w = static_cast<int>(si.hw_shape.width);
        if (info_.input_frame_size != si.hw_frame_size)
            throw std::runtime_error("HailoRawRunner: 입력 get_frame_size != hw_frame_size — 호스트 변환이 걸려 있다");
    }

    // ── 출력 메타: HEF shape 만으로 텐서 번호(cls(s)=2s, box(s)=2s+1)를 판정한다.
    //    features 80 = cls, 64 = box.  grid_h 80/40/20 → s 0/1/2 (stride = 640 / grid_h).
    int num_classes = 0, reg_max = 0;
    for (int i = 0; i < kNumTensors; ++i) {
        const hailort::OutputStream& s = *outputs_[static_cast<std::size_t>(i)];
        const hailo_stream_info_t si = s.get_info();
        const auto& hw = si.hw_shape;
        const auto& sh = si.shape;
        const std::string name = s.name();

        const std::size_t expect = static_cast<std::size_t>(hw.height) * hw.width * hw.features;
        if (expect != si.hw_frame_size)
            throw std::runtime_error("HailoRawRunner: hw_shape와 hw_frame_size가 안 맞는다: " + name);
        if (s.get_frame_size() != si.hw_frame_size)
            throw std::runtime_error("HailoRawRunner: get_frame_size != hw_frame_size — 호스트 변환이 걸려 있다: " +
                                     name);

        const int h = static_cast<int>(hw.height);
        if (h == 0 || kInputSide % h != 0)
            throw std::runtime_error("HailoRawRunner: grid height에서 stride를 파생할 수 없다: " + name);
        const int stride = kInputSide / h;
        int si_idx = -1;
        if (stride == 8) si_idx = 0;
        else if (stride == 16) si_idx = 1;
        else if (stride == 32) si_idx = 2;
        else throw std::runtime_error("HailoRawRunner: 예상 밖 stride " + std::to_string(stride) + ": " + name);

        const int feats = static_cast<int>(hw.features);
        const bool is_box = (feats % kDflSides == 0) && (feats / kDflSides == kDflRegMax);
        if (is_box) {
            if (reg_max == 0) reg_max = feats / kDflSides;
            else if (reg_max != feats / kDflSides) throw std::runtime_error("HailoRawRunner: reg_max 불일치");
        } else {
            if (num_classes == 0) num_classes = feats;
            else if (num_classes != feats) throw std::runtime_error("HailoRawRunner: 클래스 수 불일치");
        }

        const int tensor = 2 * si_idx + (is_box ? 1 : 0);
        if (!info_.t[tensor].stream_name.empty())
            throw std::runtime_error("HailoRawRunner: 같은 (scale, role) 텐서가 둘이다: " + name);

        RawTensorMeta& m = info_.t[tensor];
        m.stream_name = name;
        m.hw_frame_size = si.hw_frame_size;
        m.grid_h = h;
        m.grid_w = static_cast<int>(sh.width);
        m.w_padded = static_cast<int>(hw.width);
        m.stride = stride;
        m.features = feats;

        const auto quant = s.get_quant_infos();
        // per-feature 양자화면 텐서당 스칼라 1쌍이라는 전제가 깨진다(GPU 디코드 설계가 바뀐다).
        if (quant.size() != 1)
            throw std::runtime_error("HailoRawRunner: get_quant_infos().size() != 1 (" +
                                     std::to_string(quant.size()) + "): " + name);
        m.qp_scale = quant[0].qp_scale;
        m.qp_zp = quant[0].qp_zp;
    }
    for (int i = 0; i < kNumTensors; ++i)
        if (info_.t[i].stream_name.empty())
            throw std::runtime_error("HailoRawRunner: 텐서 번호 " + std::to_string(i) + "에 맞는 스트림이 없다");
    if (num_classes != 80 || reg_max != kDflRegMax)
        throw std::runtime_error("HailoRawRunner: classes/reg_max가 80/16이 아니다");
    info_.num_classes = num_classes;
    info_.reg_max = reg_max;

    // ── 큐 크기와 슬롯 수
    {
        auto q_in = input_->get_async_max_queue_size();
        std::size_t min_queue = q_in ? q_in.value() : 1;
        for (auto* o : outputs_) {
            auto q = o->get_async_max_queue_size();
            min_queue = std::min(min_queue, q ? q.value() : static_cast<std::size_t>(1));
        }
        info_.min_queue = min_queue;
        // 클램프하지 않으면 HAILO_QUEUE_IS_FULL 이 난다.
        num_slots_ = std::max(1, std::min(cfg_.num_slots, static_cast<int>(min_queue)));
    }
    for (int s = 0; s < num_slots_; ++s) slots_.push_back(std::make_unique<Slot>());

    if (cfg_.verbose) {
        std::printf("[raw-runner] hef=%s batch=%u slots=%d (요청 %d, min_queue=%zu) input=%s %zu B\n",
                    cfg_.hef_path.c_str(), static_cast<unsigned>(cfg_.batch_size), num_slots_, cfg_.num_slots,
                    info_.min_queue, info_.input_name.c_str(), info_.input_frame_size);
        for (int i = 0; i < kNumTensors; ++i) {
            const RawTensorMeta& m = info_.t[i];
            std::printf("[raw-runner]   t[%d] %-28s %dx%d(pad %d) f=%-3d stride=%-2d %zu B  scale=%.9g zp=%.6g\n", i,
                        m.stream_name.c_str(), m.grid_h, m.grid_w, m.w_padded, m.features, m.stride, m.hw_frame_size,
                        static_cast<double>(m.qp_scale), static_cast<double>(m.qp_zp));
        }
    }
}

// ---------------------------------------------------------------- 소멸

HailoRawRunner::~HailoRawRunner() {
    try {
        if (!drain_for(cfg_.timeout))
            std::fprintf(stderr, "[warn] HailoRawRunner: drain 시한(%lld ms) 초과 — 콜백이 오지 않았다\n",
                         static_cast<long long>(cfg_.timeout.count()));
    } catch (...) {
    }
    for (const auto& m : mapped_) {
        if (vdevice_) vdevice_->dma_unmap_dmabuf(m.fd, m.size, m.dir);
    }
    mapped_.clear();
    ng_.reset();
    vdevice_.reset();
}

// ---------------------------------------------------------------- 슬롯 bind

void HailoRawRunner::bind_slot(int slot, int in_fd, std::size_t in_map_size, const int out_fd[6],
                               const std::size_t out_map_size[6]) {
    if (slot < 0 || slot >= num_slots_)
        throw std::runtime_error("HailoRawRunner::bind_slot: 슬롯 범위 밖 " + std::to_string(slot) + " (0.." +
                                 std::to_string(num_slots_ - 1) + ")");
    Slot& sl = *slots_[static_cast<std::size_t>(slot)];
    if (sl.bound)
        throw std::runtime_error("HailoRawRunner::bind_slot: 슬롯 " + std::to_string(slot) + "은 이미 묶여 있다");
    if (in_fd < 0) throw std::runtime_error("HailoRawRunner::bind_slot: 입력 fd가 음수다");
    // 크기 계약: 매핑 크기는 dma-buf 실크기이고, 전송 크기(hw_frame_size) 이상이어야 한다.
    if (in_map_size < info_.input_frame_size)
        throw std::runtime_error("HailoRawRunner::bind_slot: 입력 dma-buf 크기 " + std::to_string(in_map_size) +
                                 " < hw_frame_size " + std::to_string(info_.input_frame_size));
    for (int i = 0; i < kNumTensors; ++i) {
        if (out_fd[i] < 0)
            throw std::runtime_error("HailoRawRunner::bind_slot: 출력 fd[" + std::to_string(i) + "]가 음수다");
        if (out_map_size[i] < info_.t[i].hw_frame_size)
            throw std::runtime_error("HailoRawRunner::bind_slot: 출력 dma-buf[" + std::to_string(i) + "] 크기 " +
                                     std::to_string(out_map_size[i]) + " < hw_frame_size " +
                                     std::to_string(info_.t[i].hw_frame_size));
    }

    // ⚠ dma_map_dmabuf 에는 **dma-buf 실크기**를 넘긴다(방향도 다르다 — 입력 H2D, 출력 D2H).
    const auto st_in = vdevice_->dma_map_dmabuf(in_fd, in_map_size, HAILO_DMA_BUFFER_DIRECTION_H2D);
    if (st_in != HAILO_SUCCESS) throw HailoError("VDevice::dma_map_dmabuf(H2D)", st_in);
    mapped_.push_back({in_fd, in_map_size, HAILO_DMA_BUFFER_DIRECTION_H2D});

    for (int i = 0; i < kNumTensors; ++i) {
        const auto st = vdevice_->dma_map_dmabuf(out_fd[i], out_map_size[i], HAILO_DMA_BUFFER_DIRECTION_D2H);
        if (st != HAILO_SUCCESS) throw HailoError("VDevice::dma_map_dmabuf(D2H)", st);
        mapped_.push_back({out_fd[i], out_map_size[i], HAILO_DMA_BUFFER_DIRECTION_D2H});
    }

    sl.in_fd = in_fd;
    sl.in_map_size = in_map_size;
    for (int i = 0; i < kNumTensors; ++i) {
        sl.out_fd[i] = out_fd[i];
        sl.out_map_size[i] = out_map_size[i];
    }
    sl.bound = true;
}

// ---------------------------------------------------------------- 제출

void HailoRawRunner::submit(int slot, std::int64_t pts, RawInferCallback on_done) {
    if (slot < 0 || slot >= num_slots_)
        throw std::runtime_error("HailoRawRunner::submit: 슬롯 범위 밖 " + std::to_string(slot));
    Slot& sl = *slots_[static_cast<std::size_t>(slot)];
    if (!sl.bound)
        throw std::runtime_error("HailoRawRunner::submit: 슬롯 " + std::to_string(slot) +
                                 "이 bind_slot() 되지 않았다");
    bool expect_busy = false;
    // 아직 7개 버퍼 콜백이 다 오지 않은 슬롯을 다시 쓰려는 시도를 여기서 막는다.
    // (슬롯은 사용자 콜백 **진입 직전**에 반납되므로, 콜백 안에서 넘겨받은 슬롯 재제출은 통과한다.)
    if (!sl.busy.compare_exchange_strong(expect_busy, true))
        throw HailoError("HailoRawRunner::submit(슬롯이 이미 비행 중)", HAILO_INVALID_OPERATION);

    sl.pts = pts;
    sl.on_done = std::move(on_done);
    sl.status.store(HAILO_SUCCESS);
    sl.remaining.store(kNumTensors + 1);
    in_flight_.fetch_add(1);

    // 입력 1 + 출력 6을 한 요청으로 묶는다(FIFO 짝맞추기를 HailoRT 가 한다).
    // NamedBuffersCallbacks 는 입출력을 구분하지 않는 단일 맵이라 입력도 DMA_BUFFER 로 넣는다
    // (/usr/include/hailo/network_group.hpp:48). BufferRepresentation 은 union 이므로
    // buffer_type 을 먼저 세팅하고 해당 멤버만 대입한다.
    // ⚠ 전송 크기는 **hw_frame_size** 다. dma-buf 실크기를 주면 콜백이 오지 않는다.
    hailort::NamedBuffersCallbacks named;
    {
        hailort::BufferRepresentation br{};
        br.buffer_type = hailort::BufferType::DMA_BUFFER;
        br.dma_buffer = hailo_dma_buffer_t{sl.in_fd, info_.input_frame_size};
        named.emplace(info_.input_name,
                      std::make_pair(br, [this, slot](hailo_status st) { note_callback(slot, st); }));
    }
    for (int i = 0; i < kNumTensors; ++i) {
        hailort::BufferRepresentation br{};
        br.buffer_type = hailort::BufferType::DMA_BUFFER;
        br.dma_buffer = hailo_dma_buffer_t{sl.out_fd[i], info_.t[i].hw_frame_size};
        named.emplace(info_.t[i].stream_name,
                      std::make_pair(br, [this, slot](hailo_status st) { note_callback(slot, st); }));
    }

    // 백프레셔: ongoing callback 수가 num_slots_ 미만이 될 때까지 막는다.
    //
    // ⚠ 여기서 increase_ongoing_callbacks() 를 **직접 부르지 않는다.** infer_async() 가 내부에서
    //    요청당 1 을 이미 센다. 수동 쌍까지 넣으면 프레임당 2 가 올라가 in-flight 상한이 반으로 깎인다.
    //
    //    실측 (HailoRT 4.24.0, Hailo-8, yolov11m.hef, PCIe Gen3 x1, 호스트 버퍼 200프레임):
    //      수동 쌍 있음: threshold 4 → peak in-flight 2, threshold 8 → 4   (프레임당 +2)
    //      수동 쌍 없음: threshold 2 → peak in-flight 2, threshold 4 → 4, threshold 8 → 8 (프레임당 +1)
    //    → 수동 쌍 없이도 백프레셔는 그대로 걸린다(slots=8 인데 threshold 2 면 2 에서 묶였다).
    //    libhailort 4.24.0 디스어셈블로도 확인: wait_for_ongoing_callbacks_count_under 는
    //    m_ongoing_transfers(오프셋 0x38)에 대해 **strict <** 술어로 기다린다.
    //    이 결함 때문에 b2/s4 --postproc gpu 가 24.48 FPS 에 묶여 있었다(host 34.98).
    //    HailoRT 버전이 올라가 infer_async 의 내부 계수가 바뀌면 이 실험을 다시 해야 한다.
    const auto st_wait = ng_->wait_for_ongoing_callbacks_count_under(static_cast<std::size_t>(num_slots_));
    if (st_wait != HAILO_SUCCESS) {
        sl.remaining.store(0);
        sl.on_done = nullptr;
        sl.busy.store(false);
        {
            // drain_for() 의 술어와 경합하지 않게 잠금 안에서 줄인다(finish() 와 같은 규칙).
            std::lock_guard<std::mutex> lk(mtx_);
            in_flight_.fetch_sub(1);
        }
        cv_.notify_all();
        throw HailoError("wait_for_ongoing_callbacks_count_under", st_wait);
    }

    sl.t_submit = std::chrono::steady_clock::now();
    // 요청 완료 콜백은 비워 둔다. 프레임의 끝은 버퍼 콜백 7개(note_callback)로 판정하고,
    // ongoing 카운터는 HailoRT 가 알아서 올리고 내린다(위 주석의 실측).
    const auto st = ng_->infer_async(named, [](hailo_status) {});
    if (st != HAILO_SUCCESS) {
        // 버퍼 콜백이 오지 않으므로 여기서 프레임을 직접 끝낸다(사용자 콜백은 실패 상태로 불린다).
        sl.status.store(st);
        sl.remaining.store(0);
        finish(slot);
        throw HailoError("ConfiguredNetworkGroup::infer_async", st);
    }
}

void HailoRawRunner::note_callback(int slot, hailo_status st) {
    Slot& sl = *slots_[static_cast<std::size_t>(slot)];
    if (st != HAILO_SUCCESS) sl.status.store(st);
    if (sl.remaining.fetch_sub(1) == 1) finish(slot);
}

void HailoRawRunner::finish(int slot) {
    Slot& sl = *slots_[static_cast<std::size_t>(slot)];
    RawInferResult r;
    r.slot = slot;
    r.pts = sl.pts;
    r.status = sl.status.load();
    r.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sl.t_submit).count();

    RawInferCallback cb;
    cb.swap(sl.on_done);

    // **콜백에 들어가기 전에** 슬롯을 반납한다. 이 시점에 7개 버퍼 콜백이 모두 왔으므로
    // NPU 는 이 슬롯의 입출력 dma-buf 를 더 이상 만지지 않는다. 덕분에 콜백 안에서
    // 슬롯 번호를 다른 스레드(파이프라인의 free 큐)로 넘겨 바로 재제출해도 경합이 없다.
    sl.busy.store(false);
    if (cb) cb(std::move(r));

    {
        std::lock_guard<std::mutex> lk(mtx_);
        in_flight_.fetch_sub(1);
    }
    cv_.notify_all();
}

// ---------------------------------------------------------------- drain / 상태

bool HailoRawRunner::drain_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mtx_);
    return cv_.wait_for(lk, timeout, [&] { return in_flight_.load() == 0; });
}

void HailoRawRunner::drain() {
    if (!drain_for(cfg_.timeout))
        throw HailoError("HailoRawRunner::drain(시한 초과, 비행 중 " + std::to_string(in_flight_.load()) + ")",
                         HAILO_TIMEOUT);
}

float HailoRawRunner::chip_temperature() const {
    if (!vdevice_) return std::nanf("");
    auto devs = vdevice_->get_physical_devices();
    if (!devs || devs->empty()) return std::nanf("");
    auto t = devs->at(0).get().get_chip_temperature();
    if (!t) return std::nanf("");
    return std::max(t->ts0_temperature, t->ts1_temperature);
}

} // namespace yvo
