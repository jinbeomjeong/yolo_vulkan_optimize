#include "hailo/hailo_raw_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "ref/cpu_yolov8_decode.hpp"   // kDflRegMax / kDflSides (layout-derived constants)

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

// ---------------------------------------------------------------- construction

HailoRawRunner::HailoRawRunner(const HailoRawRunnerConfig& cfg) : cfg_(cfg) {
    hef_ = std::make_unique<hailort::Hef>(unwrap(hailort::Hef::create(cfg_.hef_path), "Hef::create"));

    hailo_vdevice_params_t vparams;
    check(hailo_init_vdevice_params(&vparams), "hailo_init_vdevice_params");
    vparams.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    vdevice_ = unwrap(hailort::VDevice::create(vparams), "VDevice::create");

    auto params_map = unwrap(vdevice_->create_configure_params(*hef_), "VDevice::create_configure_params");
    for (auto& kv : params_map) {
        kv.second.batch_size = cfg_.batch_size;
        // All 7 streams ASYNC. This is a precondition of infer_async.
        for (auto& sp : kv.second.stream_params_by_name) {
            sp.second.flags = HAILO_STREAM_FLAGS_ASYNC;
        }
    }
    auto ngs = unwrap(vdevice_->configure(*hef_, params_map), "VDevice::configure");
    if (ngs.size() != 1) throw std::runtime_error("HailoRawRunner: network group count is not 1");
    ng_ = ngs[0];

    auto in_refs = ng_->get_input_streams();
    auto out_refs = ng_->get_output_streams();
    if (in_refs.size() != 1) throw std::runtime_error("HailoRawRunner: input stream count is not 1");
    if (out_refs.size() != static_cast<std::size_t>(kNumTensors))
        throw std::runtime_error("HailoRawRunner: raw output stream count is not 6: " +
                                 std::to_string(out_refs.size()));
    input_ = &in_refs[0].get();
    for (auto& r : out_refs) outputs_.push_back(&r.get());
    // Fix the order by name so stream order is deterministic across runs (same rule as the probe's RawSession).
    std::sort(outputs_.begin(), outputs_.end(),
              [](const hailort::OutputStream* a, const hailort::OutputStream* b) { return a->name() < b->name(); });

    // ── Input metadata
    {
        const hailo_stream_info_t si = input_->get_info();
        info_.input_name = input_->name();
        info_.input_frame_size = input_->get_frame_size();
        info_.input_h = static_cast<int>(si.hw_shape.height);
        info_.input_w = static_cast<int>(si.hw_shape.width);
        if (info_.input_frame_size != si.hw_frame_size)
            throw std::runtime_error("HailoRawRunner: input get_frame_size != hw_frame_size — host conversion is active");
    }

    // ── Output metadata: determine the tensor index (cls(s)=2s, box(s)=2s+1) from the HEF shape alone.
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
            throw std::runtime_error("HailoRawRunner: hw_shape and hw_frame_size do not match: " + name);
        if (s.get_frame_size() != si.hw_frame_size)
            throw std::runtime_error("HailoRawRunner: get_frame_size != hw_frame_size — host conversion is active: " +
                                     name);

        const int h = static_cast<int>(hw.height);
        if (h == 0 || kInputSide % h != 0)
            throw std::runtime_error("HailoRawRunner: cannot derive stride from grid height: " + name);
        const int stride = kInputSide / h;
        int si_idx = -1;
        if (stride == 8) si_idx = 0;
        else if (stride == 16) si_idx = 1;
        else if (stride == 32) si_idx = 2;
        else throw std::runtime_error("HailoRawRunner: unexpected stride " + std::to_string(stride) + ": " + name);

        const int feats = static_cast<int>(hw.features);
        const bool is_box = (feats % kDflSides == 0) && (feats / kDflSides == kDflRegMax);
        if (is_box) {
            if (reg_max == 0) reg_max = feats / kDflSides;
            else if (reg_max != feats / kDflSides) throw std::runtime_error("HailoRawRunner: reg_max mismatch");
        } else {
            if (num_classes == 0) num_classes = feats;
            else if (num_classes != feats) throw std::runtime_error("HailoRawRunner: class count mismatch");
        }

        const int tensor = 2 * si_idx + (is_box ? 1 : 0);
        if (!info_.t[tensor].stream_name.empty())
            throw std::runtime_error("HailoRawRunner: two tensors with the same (scale, role): " + name);

        RawTensorMeta& m = info_.t[tensor];
        m.stream_name = name;
        m.hw_frame_size = si.hw_frame_size;
        m.grid_h = h;
        m.grid_w = static_cast<int>(sh.width);
        m.w_padded = static_cast<int>(hw.width);
        m.stride = stride;
        m.features = feats;

        const auto quant = s.get_quant_infos();
        // Per-feature quantization would break the assumption of one scalar pair per tensor (the GPU decode design would change).
        if (quant.size() != 1)
            throw std::runtime_error("HailoRawRunner: get_quant_infos().size() != 1 (" +
                                     std::to_string(quant.size()) + "): " + name);
        m.qp_scale = quant[0].qp_scale;
        m.qp_zp = quant[0].qp_zp;
    }
    for (int i = 0; i < kNumTensors; ++i)
        if (info_.t[i].stream_name.empty())
            throw std::runtime_error("HailoRawRunner: tensor index " + std::to_string(i) + " has no matching stream");
    if (num_classes != 80 || reg_max != kDflRegMax)
        throw std::runtime_error("HailoRawRunner: classes/reg_max is not 80/16");
    info_.num_classes = num_classes;
    info_.reg_max = reg_max;

    // ── Queue size and slot count
    {
        auto q_in = input_->get_async_max_queue_size();
        std::size_t min_queue = q_in ? q_in.value() : 1;
        for (auto* o : outputs_) {
            auto q = o->get_async_max_queue_size();
            min_queue = std::min(min_queue, q ? q.value() : static_cast<std::size_t>(1));
        }
        info_.min_queue = min_queue;
        // Without clamping, HAILO_QUEUE_IS_FULL occurs.
        num_slots_ = std::max(1, std::min(cfg_.num_slots, static_cast<int>(min_queue)));
    }
    for (int s = 0; s < num_slots_; ++s) slots_.push_back(std::make_unique<Slot>());

    if (cfg_.verbose) {
        std::printf("[raw-runner] hef=%s batch=%u slots=%d (requested %d, min_queue=%zu) input=%s %zu B\n",
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

// ---------------------------------------------------------------- destruction

HailoRawRunner::~HailoRawRunner() {
    try {
        if (!drain_for(cfg_.timeout))
            std::fprintf(stderr, "[warn] HailoRawRunner: drain deadline (%lld ms) exceeded — callbacks did not arrive\n",
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

// ---------------------------------------------------------------- slot bind

void HailoRawRunner::bind_slot(int slot, int in_fd, std::size_t in_map_size, const int out_fd[6],
                               const std::size_t out_map_size[6]) {
    if (slot < 0 || slot >= num_slots_)
        throw std::runtime_error("HailoRawRunner::bind_slot: slot out of range " + std::to_string(slot) + " (0.." +
                                 std::to_string(num_slots_ - 1) + ")");
    Slot& sl = *slots_[static_cast<std::size_t>(slot)];
    if (sl.bound)
        throw std::runtime_error("HailoRawRunner::bind_slot: slot " + std::to_string(slot) + " is already bound");
    if (in_fd < 0) throw std::runtime_error("HailoRawRunner::bind_slot: input fd is negative");
    // Size contract: the mapping size is the actual dma-buf size and must be >= the transfer size (hw_frame_size).
    if (in_map_size < info_.input_frame_size)
        throw std::runtime_error("HailoRawRunner::bind_slot: input dma-buf size " + std::to_string(in_map_size) +
                                 " < hw_frame_size " + std::to_string(info_.input_frame_size));
    for (int i = 0; i < kNumTensors; ++i) {
        if (out_fd[i] < 0)
            throw std::runtime_error("HailoRawRunner::bind_slot: output fd[" + std::to_string(i) + "] is negative");
        if (out_map_size[i] < info_.t[i].hw_frame_size)
            throw std::runtime_error("HailoRawRunner::bind_slot: output dma-buf[" + std::to_string(i) + "] size " +
                                     std::to_string(out_map_size[i]) + " < hw_frame_size " +
                                     std::to_string(info_.t[i].hw_frame_size));
    }

    // ⚠ dma_map_dmabuf gets the **actual dma-buf size** (the direction differs too — input H2D, output D2H).
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

// ---------------------------------------------------------------- submission

void HailoRawRunner::submit(int slot, std::int64_t pts, RawInferCallback on_done) {
    if (slot < 0 || slot >= num_slots_)
        throw std::runtime_error("HailoRawRunner::submit: slot out of range " + std::to_string(slot));
    Slot& sl = *slots_[static_cast<std::size_t>(slot)];
    if (!sl.bound)
        throw std::runtime_error("HailoRawRunner::submit: slot " + std::to_string(slot) +
                                 " has not been bind_slot()ed");
    bool expect_busy = false;
    // Block here any attempt to reuse a slot whose 7 buffer callbacks have not all arrived yet.
    // (The slot is released **right before entering** the user callback, so resubmitting a slot handed over inside the callback passes.)
    if (!sl.busy.compare_exchange_strong(expect_busy, true))
        throw HailoError("HailoRawRunner::submit(slot already in flight)", HAILO_INVALID_OPERATION);

    sl.pts = pts;
    sl.on_done = std::move(on_done);
    sl.status.store(HAILO_SUCCESS);
    sl.remaining.store(kNumTensors + 1);
    in_flight_.fetch_add(1);

    // Bundle 1 input + 6 outputs into one request (HailoRT does the FIFO pairing).
    // NamedBuffersCallbacks is a single map that does not distinguish inputs from outputs, so the input is also given as DMA_BUFFER
    // (/usr/include/hailo/network_group.hpp:48). BufferRepresentation is a union, so
    // set buffer_type first and assign only the corresponding member.
    // ⚠ The transfer size is **hw_frame_size**. Giving the actual dma-buf size means no callback arrives.
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

    // Backpressure: block until the ongoing callback count is below num_slots_.
    //
    // ⚠ Do **not call** increase_ongoing_callbacks() **directly** here. infer_async() already counts
    //    1 per request internally. Adding a manual pair raises it by 2 per frame and halves the in-flight limit.
    //
    //    Measured (HailoRT 4.24.0, Hailo-8, yolov11m.hef, PCIe Gen3 x1, host buffers, 200 frames):
    //      with manual pair:    threshold 4 → peak in-flight 2, threshold 8 → 4   (+2 per frame)
    //      without manual pair: threshold 2 → peak in-flight 2, threshold 4 → 4, threshold 8 → 8 (+1 per frame)
    //    → Backpressure still applies without the manual pair (with slots=8 and threshold 2 it was held at 2).
    //    Also confirmed by disassembling libhailort 4.24.0: wait_for_ongoing_callbacks_count_under
    //    waits on m_ongoing_transfers (offset 0x38) with a **strict <** predicate.
    //    Because of this defect, b2/s4 --postproc gpu was stuck at 24.48 FPS (host 34.98).
    //    If a newer HailoRT changes infer_async's internal counting, this experiment must be redone.
    const auto st_wait = ng_->wait_for_ongoing_callbacks_count_under(static_cast<std::size_t>(num_slots_));
    if (st_wait != HAILO_SUCCESS) {
        sl.remaining.store(0);
        sl.on_done = nullptr;
        sl.busy.store(false);
        {
            // Decrement under the lock so it does not race with drain_for()'s predicate (same rule as finish()).
            std::lock_guard<std::mutex> lk(mtx_);
            in_flight_.fetch_sub(1);
        }
        cv_.notify_all();
        throw HailoError("wait_for_ongoing_callbacks_count_under", st_wait);
    }

    sl.t_submit = std::chrono::steady_clock::now();
    // Leave the request completion callback empty. The end of a frame is determined by the 7 buffer callbacks (note_callback),
    // and HailoRT raises and lowers the ongoing counter by itself (measured, see comment above).
    const auto st = ng_->infer_async(named, [](hailo_status) {});
    if (st != HAILO_SUCCESS) {
        // No buffer callbacks will arrive, so finish the frame directly here (the user callback is invoked with a failure status).
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

    // Release the slot **before entering the callback**. At this point all 7 buffer callbacks have arrived,
    // so the NPU no longer touches this slot's input/output dma-bufs. Thanks to that, handing the slot number
    // to another thread (the pipeline's free queue) from inside the callback and resubmitting it right away causes no race.
    sl.busy.store(false);
    if (cb) cb(std::move(r));

    {
        std::lock_guard<std::mutex> lk(mtx_);
        in_flight_.fetch_sub(1);
    }
    cv_.notify_all();
}

// ---------------------------------------------------------------- drain / state

bool HailoRawRunner::drain_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mtx_);
    return cv_.wait_for(lk, timeout, [&] { return in_flight_.load() == 0; });
}

void HailoRawRunner::drain() {
    if (!drain_for(cfg_.timeout))
        throw HailoError("HailoRawRunner::drain(deadline exceeded, in flight " + std::to_string(in_flight_.load()) + ")",
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
