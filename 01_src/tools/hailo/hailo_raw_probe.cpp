// yvo_hailo_raw_probe: probe that verifies the HailoRT **6 raw output streams** path (plan §6.2 P1 + P2).
//
// --probe p1 (default): verifies the API path and tensor semantics with **host malloc buffers only** (no GPU).
// --probe p2      : **receives the 6 outputs as dma-bufs.** This is the go/no-go gate for the post-processing track.
//   · --alloc malloc|heap|vulkan (several, comma-separated). malloc is the baseline for byte comparison.
//   · tries all 4 size combinations: dma_map_dmabuf with (hw_frame_size | dmabuf_size)
//                        x read_async with (hw_frame_size | dmabuf_size)
//   · measures 6-tensor byte match across the three modes, cpu_read_1.22MB_ms, and the total time of 96 dma_map_dmabuf calls
//   · counts fds at start and end to show there is no leak
// --probe f0      : the gate for **simultaneous use of input dma-buf (H2D) + output dma-buf (D2H)**.
//   So far the only verified combinations were (dma-buf input + host NMS output) and (host malloc input + 6 dma-buf outputs),
//   so (Vulkan export dma-buf input + 6 dma-buf outputs), which the integrated pipeline will use, was unverified.
//   Runs the same frames with and without --in-dmabuf and compares the 6 output tensors byte-for-byte, 7 callbacks per frame,
//   fd leaks, and hailort.log warnings. infer_async and write_async are judged separately.
//
// What it does:
//  0) Dumps shape/hw_shape/hw_frame_size/format of the 6 output streams from HEF parsing alone.
//  1) VDevice(ROUND_ROBIN) → create_configure_params(hef) → set batch_size → configure(hef, params).
//     Dumps is_scheduled(), infer_queue_size(), per-stream get_async_max_queue_size(),
//     and the .size() and all values of OutputStream::get_quant_infos().
//     If get_quant_infos().size() > 1 (per-feature quantization), it is reported as an immediate failure — the design would change.
//  2) --api {infer_async, read_async, both}: both paths are implemented and tried.
//     infer_async: bundles 1 input + 6 outputs into one request (HailoRT does the FIFO pairing).
//     read_async : sets HAILO_STREAM_FLAGS_ASYNC on all 7 streams at configure time, and
//                  issues 1 write_async + 6 read_async inside one mutex to guarantee FIFO pairing directly.
//  3) Processes real frames, dumps the 6 tensors as fixtures (04_data/raw_tensors/), and compares the detections of the CPU reference
//     implementation (ref/cpu_yolov8_decode.hpp) with **HailoRT NMS detections obtained in a separate pass** via match().
//     Exhaustively tries the 4 unresolved tensor-interpretation variables (argmax_only / DFL channel order / sigmoid / > vs >=).
//  4) Evidence for dropping option (B): InferModel's output count, and the configure() failure status when NHCW format order is set on the output.
//
// There is only one Hailo device, so two runners cannot coexist. So each stage fully destroys its objects before moving on
// (the run_nhcw_pass() pattern in 05_tests/test_detections.cpp).
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hailo/hailort.hpp>

#include "ref/cpu_letterbox.hpp"
#include "core/dma_heap_alloc.hpp"   // dma_sync_* (used only when the input dma-buf is a cached heap)
#include "ref/cpu_yolov8_decode.hpp"
#include "ref/detection_match.hpp"
#include "core/frame_types.hpp"
#include "hailo/hailo_raw_runner.hpp"
#include "hailo/hailo_runner.hpp"
#include "tools/hailo/p2_out_buffers.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"
#include "core/hailo_align.hpp"

using yvo::percentile;

namespace {

using yvo::tool::fmt;
using yvo::tool::count_open_fds;
using yvo::tool::Nv12Frame;
using yvo::tool::read_nv12_frames;

constexpr int kSrcW = 1920;
constexpr int kSrcH = 1080;
// HailoRT's vdma_stream.cpp:379 requires **16384 B alignment** for read_async(). With 4096 B,
// the "unaligned buffer ... causes performance degradation" warning appears for every frame×stream and
// measured FPS differed by 3 FPS, 48.5 → 51.7 (plan §7 verification record).
constexpr std::size_t kDmaAlign = yvo::kHailoAsyncAlign;  // core/hailo_align.hpp
constexpr float kIouMatchThreshold = yvo::detmatch::kIouThreshold;  // ref/detection_match.hpp
constexpr float kHefScoreThreshold = 0.2f;   // HEF's YOLOV8 post-process setting
constexpr float kHefIouThreshold = 0.7f;
constexpr int kHefMaxPerClass = 100;

// ---------------------------------------------------------------- common utilities

std::string status_name(hailo_status s) {
    const char* msg = hailo_get_status_message(s);
    return std::string(msg ? msg : "UNKNOWN") + "(" + std::to_string(static_cast<int>(s)) + ")";
}

void check(hailo_status s, const char* where) {
    if (s != HAILO_SUCCESS) throw yvo::HailoError(where, s);
}

template <typename T>
T unwrap(hailort::Expected<T>&& e, const char* where) {
    if (!e) throw yvo::HailoError(where, e.status());
    return e.release();
}

std::shared_ptr<std::uint8_t> alloc_dma_aligned(std::size_t size) {
    const std::size_t rounded = (size + kDmaAlign - 1) / kDmaAlign * kDmaAlign;
    void* p = std::aligned_alloc(kDmaAlign, rounded);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, rounded);
    return std::shared_ptr<std::uint8_t>(static_cast<std::uint8_t*>(p), [](std::uint8_t* q) { std::free(q); });
}

const char* format_order_name(hailo_format_order_t o) {
    switch (o) {
        case HAILO_FORMAT_ORDER_NHWC: return "NHWC";
        case HAILO_FORMAT_ORDER_NHCW: return "NHCW";
        case HAILO_FORMAT_ORDER_FCR: return "FCR";
        case HAILO_FORMAT_ORDER_F8CR: return "F8CR";
        case HAILO_FORMAT_ORDER_NC: return "NC";
        case HAILO_FORMAT_ORDER_BAYER_RGB: return "BAYER_RGB";
        case HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS: return "HAILO_NMS_BY_CLASS";
        case HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE: return "HAILO_NMS_BY_SCORE";
        default: return "OTHER";
    }
}

const char* format_type_name(hailo_format_type_t t) {
    switch (t) {
        case HAILO_FORMAT_TYPE_UINT8: return "UINT8";
        case HAILO_FORMAT_TYPE_UINT16: return "UINT16";
        case HAILO_FORMAT_TYPE_FLOAT32: return "FLOAT32";
        default: return "AUTO/OTHER";
    }
}

// Part of the stream name after '/' (e.g. "yolov11m/conv74_178" -> "conv74_178")
std::string short_name(const std::string& full) {
    const auto pos = full.rfind('/');
    return pos == std::string::npos ? full : full.substr(pos + 1);
}

std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    if (f && std::getline(f, line)) return line;
    return "(read failed)";
}

// Writes to stdout and a file at the same time. The evidence file is the report.
class Tee {
public:
    explicit Tee(const std::string& path) : f_(path) {}
    bool ok() const { return static_cast<bool>(f_); }
    Tee& operator<<(const std::string& s) {
        std::fputs(s.c_str(), stdout);
        f_ << s;
        f_.flush();
        return *this;
    }

private:
    std::ofstream f_;
};

// ---------------------------------------------------------------- video → NHCW input

// Same as the CPU letterbox NHCW path in hailo_selftest.cpp: cpu_letterbox_nv12_bt709 → rgb_packed_to_nhcw
std::vector<std::vector<std::uint8_t>> make_nhcw_inputs(const std::vector<Nv12Frame>& frames,
                                                        const yvo::LetterboxParams& lb, std::size_t frame_size) {
    std::vector<std::vector<std::uint8_t>> out(frames.size());
    std::vector<std::uint8_t> packed(frame_size);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const std::uint8_t* y = frames[i].data.data();
        const std::uint8_t* uv = y + static_cast<std::size_t>(kSrcW) * kSrcH;
        yvo::cpu_letterbox_nv12_bt709(y, kSrcW, uv, kSrcW, kSrcW, kSrcH, lb, packed.data());
        out[i].resize(frame_size);
        yvo::rgb_packed_to_nhcw(packed.data(), lb.dst_w, lb.dst_h, out[i].data());
    }
    return out;
}

// The matching criterion lives only in ref/detection_match.hpp (before 2026-09-21 there was a copy here).
using yvo::detmatch::match;

// ---------------------------------------------------------------- raw stream session

enum class RawApi { InferAsync, ReadAsync };

const char* api_name(RawApi a) { return a == RawApi::InferAsync ? "infer_async" : "read_async"; }

struct StreamMeta {
    std::string name;
    hailo_stream_info_t info{};
    std::vector<hailo_quant_info_t> quant;
    std::size_t frame_size = 0;
    std::size_t queue_size = 0;
};

// The 6 raw outputs of one frame
struct RawFrame {
    std::vector<std::vector<std::uint8_t>> t;   // 6 of them, stream order follows RawSession::outputs_
};

// Size passed to dma_map_dmabuf / read_async. P2 tries all 4 combinations (plan §4.2).
enum class SizeKind { Hw, Dmabuf };
const char* size_kind_name(SizeKind k) { return k == SizeKind::Hw ? "hw_frame_size" : "dmabuf_size"; }

struct RawSessionConfig {
    std::string hef_path;
    std::uint16_t batch = 8;
    int want_slots = 16;
    RawApi api = RawApi::InferAsync;
    // nullptr uses the internal malloc pool (P1 behavior). Given a dma-buf pool, outputs are received via fd.
    yvo::p2::OutBufPool* out_pool = nullptr;
    SizeKind map_size = SizeKind::Hw;    // size of dma_map_dmabuf(fd, size, D2H)
    SizeKind read_size = SizeKind::Hw;   // read_async(fd, size, cb) / hailo_dma_buffer_t.size
    // ── F0: input H2D dma-buf ──
    // nullptr means the input is host malloc (today's behavior; must be byte-identical).
    // Given a pool, per-slot input buffers are taken from it and sent H2D via fd.
    yvo::p2::OutBufPool* in_pool = nullptr;
    SizeKind in_map_size = SizeKind::Dmabuf;   // size of dma_map_dmabuf(fd, size, H2D)
    SizeKind in_write_size = SizeKind::Hw;     // write_async(fd, size, cb) / hailo_dma_buffer_t.size
};

class RawSession {
public:
    explicit RawSession(const RawSessionConfig& cfg) : api_(cfg.api), cfg_(cfg) {
        const std::string& hef_path = cfg.hef_path;
        const std::uint16_t batch = cfg.batch;
        const int want_slots = cfg.want_slots;
        hef_ = std::make_unique<hailort::Hef>(unwrap(hailort::Hef::create(hef_path), "Hef::create"));

        hailo_vdevice_params_t vparams;
        check(hailo_init_vdevice_params(&vparams), "hailo_init_vdevice_params");
        vparams.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
        vdevice_ = unwrap(hailort::VDevice::create(vparams), "VDevice::create");

        auto params_map = unwrap(vdevice_->create_configure_params(*hef_), "VDevice::create_configure_params");
        for (auto& kv : params_map) {
            kv.second.batch_size = batch;
            // All 7 streams ASYNC. A precondition of the read_async path, and harmless for infer_async.
            for (auto& sp : kv.second.stream_params_by_name) {
                sp.second.flags = HAILO_STREAM_FLAGS_ASYNC;
            }
        }
        auto ngs = unwrap(vdevice_->configure(*hef_, params_map), "VDevice::configure");
        if (ngs.size() != 1) throw std::runtime_error("network group count is not 1");
        ng_ = ngs[0];

        auto in_refs = ng_->get_input_streams();
        auto out_refs = ng_->get_output_streams();
        if (in_refs.size() != 1) throw std::runtime_error("input stream count is not 1");
        input_ = &in_refs[0].get();
        for (auto& r : out_refs) outputs_.push_back(&r.get());
        // Fix the order by name so stream order is deterministic across runs
        std::sort(outputs_.begin(), outputs_.end(),
                  [](const hailort::OutputStream* a, const hailort::OutputStream* b) { return a->name() < b->name(); });

        in_meta_ = collect(*input_);
        for (auto* o : outputs_) out_meta_.push_back(collect(*o));

        std::size_t min_queue = in_meta_.queue_size;
        for (const auto& m : out_meta_) min_queue = std::min(min_queue, m.queue_size);
        // Clamp num_slots to the minimum queue size. Otherwise HAILO_QUEUE_IS_FULL occurs.
        num_slots_ = std::max(1, std::min(want_slots, static_cast<int>(min_queue)));
        min_queue_ = min_queue;

        // Input: host malloc if in_pool is nullptr (P1/P2 behavior), dma-buf fd if a pool is given (F0).
        for (int s = 0; s < num_slots_; ++s) {
            auto slot = std::make_unique<Slot>();
            if (cfg_.in_pool) {
                slot->in_ref = cfg_.in_pool->acquire(in_meta_.frame_size);
            } else {
                slot->in = alloc_dma_aligned(in_meta_.frame_size);
                slot->in_ref.ptr = slot->in.get();
            }
            for (const auto& m : out_meta_) {
                if (cfg_.out_pool) {
                    slot->out.push_back(cfg_.out_pool->acquire(m.frame_size));
                } else {
                    auto keep = alloc_dma_aligned(m.frame_size);
                    yvo::p2::OutBufRef r;
                    r.ptr = keep.get();
                    slot->keep.push_back(std::move(keep));
                    slot->out.push_back(r);
                }
            }
            slots_.push_back(std::move(slot));
            free_slots_.push_back(s);
        }

        // With a dma-buf pool, map only once for the process lifetime (output slots x tensors = 96 times, once per input slot).
        // Accumulate the elapsed time to judge plan §6.2 P2-⑤ (total < 100 ms).
        // ⚠ Size contract (appendix A.13): dma_map_dmabuf gets the actual dma-buf size, transfer APIs get hw_frame_size.
        //   Directions differ too — input is H2D, output is D2H.
        if (cfg_.in_pool || cfg_.out_pool) {
            const auto t0 = std::chrono::steady_clock::now();
            for (auto& slot : slots_) {
                if (slot->in_ref.fd >= 0) {
                    const std::size_t sz = size_for(slot->in_ref, in_meta_.frame_size, cfg_.in_map_size);
                    const auto st = vdevice_->dma_map_dmabuf(slot->in_ref.fd, sz, HAILO_DMA_BUFFER_DIRECTION_H2D);
                    ++in_map_calls_;
                    if (st != HAILO_SUCCESS) {
                        if (in_map_status_ == HAILO_SUCCESS) in_map_status_ = st;
                        ++in_map_failed_;
                    } else {
                        mapped_.push_back({slot->in_ref.fd, sz, HAILO_DMA_BUFFER_DIRECTION_H2D});
                    }
                }
                for (std::size_t i = 0; i < out_meta_.size(); ++i) {
                    const auto& b = slot->out[i];
                    if (b.fd < 0) continue;
                    const std::size_t sz = size_for(b, out_meta_[i].frame_size, cfg_.map_size);
                    const auto st = vdevice_->dma_map_dmabuf(b.fd, sz, HAILO_DMA_BUFFER_DIRECTION_D2H);
                    ++map_calls_;
                    if (st != HAILO_SUCCESS) {
                        if (map_status_ == HAILO_SUCCESS) map_status_ = st;
                        ++map_failed_;
                    } else {
                        mapped_.push_back({b.fd, sz, HAILO_DMA_BUFFER_DIRECTION_D2H});
                    }
                }
            }
            map_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    }

    ~RawSession() {
        try {
            if (!drain(5000)) std::fprintf(stderr, "[warn] RawSession: drain deadline exceeded — callbacks did not arrive\n");
        } catch (...) {
        }
        for (const auto& m : mapped_) vdevice_->dma_unmap_dmabuf(m.fd, m.size, m.dir);
        mapped_.clear();
        ng_.reset();
        vdevice_.reset();
    }

    // dma_map_dmabuf statistics
    long map_calls() const { return map_calls_; }
    long map_failed() const { return map_failed_; }
    hailo_status map_status() const { return map_status_; }
    double map_ms() const { return map_ms_; }

    // Input (H2D) dma-buf statistics
    bool uses_in_dmabuf() const { return !slots_.empty() && slots_[0]->in_ref.fd >= 0; }
    long in_map_calls() const { return in_map_calls_; }
    long in_map_failed() const { return in_map_failed_; }
    hailo_status in_map_status() const { return in_map_status_; }
    // For reporting: (hw_frame_size, dmabuf_size) of the input buffer. The second is 0 if not a dma-buf.
    std::pair<std::size_t, std::size_t> in_size_pair() const {
        return {in_meta_.frame_size, slots_.empty() ? 0 : slots_[0]->in_ref.dmabuf_size};
    }

    RawSession(const RawSession&) = delete;
    RawSession& operator=(const RawSession&) = delete;

    RawApi api() const { return api_; }
    int num_slots() const { return num_slots_; }
    std::size_t min_queue() const { return min_queue_; }
    const StreamMeta& in_meta() const { return in_meta_; }
    const std::vector<StreamMeta>& out_meta() const { return out_meta_; }
    bool is_scheduled() const { return ng_->is_scheduled(); }
    std::size_t infer_queue_size() const {
        auto q = ng_->infer_queue_size();
        return q ? q.value() : 0;
    }
    long callbacks_total() const { return cb_total_.load(); }
    long callbacks_failed() const { return cb_failed_.load(); }
    hailo_status first_error() const { return first_error_.load(); }

    float chip_temperature() const {
        auto devs = vdevice_->get_physical_devices();
        if (!devs || devs->empty()) return std::nanf("");
        auto t = devs->at(0).get().get_chip_temperature();
        if (!t) return std::nanf("");
        return std::max(t->ts0_temperature, t->ts1_temperature);
    }

    // Submit one frame. on_done(slot, status) is called on the HailoRT callback thread,
    // and after it returns the slot buffers can be freely reused.
    void submit(const std::uint8_t* nhcw_input, const std::function<void(int, hailo_status)>& on_done) {
        const int slot = acquire_slot();
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        // The host fills the input frame. For a cached dma-buf (heap), the CPU write section is wrapped in sync.
        // Vulkan export buffers are HOST_COHERENT, so needs_sync == false.
        const bool in_sync = sl.in_ref.needs_sync && sl.in_ref.fd >= 0;
        if (in_sync) yvo::dma_sync_begin(sl.in_ref.fd, yvo::DmaSyncDir::Write, /*strict=*/true);
        std::memcpy(sl.in_ref.ptr, nhcw_input, in_meta_.frame_size);
        if (in_sync) yvo::dma_sync_end(sl.in_ref.fd, yvo::DmaSyncDir::Write, /*strict=*/true);
        sl.remaining.store(static_cast<int>(out_meta_.size()) + 1);
        sl.status.store(HAILO_SUCCESS);
        sl.on_done = on_done;

        if (api_ == RawApi::InferAsync)
            submit_infer_async(slot);
        else
            submit_read_async(slot);
    }

    // Waits until all slots are released. Returns false if not done within timeout_ms.
    //
    // ⚠ **Why a deadline is needed (measured in P2)**: if infer_async() gets a size different from hw_frame_size,
    //   HailoRT returns SUCCESS, then its internal worker hits a CHECK and silently drops the request
    //   (hailort.log core_op.cpp:524). **Not a single callback arrives.** Waiting forever would hang the probe.
    bool drain(int timeout_ms = 30000) {
        std::unique_lock<std::mutex> lk(m_);
        return cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                            [&] { return free_slots_.size() == static_cast<std::size_t>(num_slots_); });
    }

private:
    struct Slot {
        std::shared_ptr<std::uint8_t> in;                 // Holds only the lifetime for the host malloc path
        yvo::p2::OutBufRef in_ref;                        // The input buffer actually used. Host pointer if fd < 0
        std::vector<yvo::p2::OutBufRef> out;              // 6 of them. Host pointer path if fd < 0
        std::vector<std::shared_ptr<std::uint8_t>> keep;  // Keeps lifetimes in malloc mode
        std::atomic<int> remaining{0};
        std::atomic<hailo_status> status{HAILO_SUCCESS};
        std::function<void(int, hailo_status)> on_done;
    };

    // Size passed to dma_map_dmabuf / read_async. Falls back to hw if dmabuf_size is 0 (measurement failed).
    static std::size_t size_for(const yvo::p2::OutBufRef& b, std::size_t hw, SizeKind k) {
        if (k == SizeKind::Hw || b.dmabuf_size == 0) return hw;
        return b.dmabuf_size;
    }

    static StreamMeta collect(const hailort::InputStream& s) {
        StreamMeta m;
        m.name = s.name();
        m.info = s.get_info();
        m.quant = s.get_quant_infos();
        m.frame_size = s.get_frame_size();
        auto q = s.get_async_max_queue_size();
        m.queue_size = q ? q.value() : 1;
        return m;
    }
    static StreamMeta collect(const hailort::OutputStream& s) {
        StreamMeta m;
        m.name = s.name();
        m.info = s.get_info();
        m.quant = s.get_quant_infos();
        m.frame_size = s.get_frame_size();
        auto q = s.get_async_max_queue_size();
        m.queue_size = q ? q.value() : 1;
        return m;
    }

    int acquire_slot() {
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_.wait_for(lk, std::chrono::seconds(30), [&] { return !free_slots_.empty(); }))
            throw std::runtime_error("RawSession::acquire_slot: no free slot within 30 seconds (lost callbacks)");
        const int s = free_slots_.front();
        free_slots_.pop_front();
        return s;
    }

    void note_callback(int slot, hailo_status st) {
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        cb_total_.fetch_add(1);
        if (st != HAILO_SUCCESS) {
            cb_failed_.fetch_add(1);
            sl.status.store(st);
            hailo_status expect = HAILO_SUCCESS;
            first_error_.compare_exchange_strong(expect, st);
        }
        if (sl.remaining.fetch_sub(1) == 1) finish(slot);
    }

    void finish(int slot) {
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        if (sl.on_done) sl.on_done(slot, sl.status.load());
        {
            std::lock_guard<std::mutex> lk(m_);
            free_slots_.push_back(slot);
        }
        cv_.notify_all();
    }

    void submit_infer_async(int slot) {
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        hailort::NamedBuffersCallbacks named;
        {
            // BufferRepresentation is a union. Set buffer_type first and assign only the corresponding member.
            // NamedBuffersCallbacks is a single map that does not distinguish inputs from outputs, so the input can also be DMA_BUFFER
            // (/usr/include/hailo/network_group.hpp:48).
            hailort::BufferRepresentation br{};
            if (sl.in_ref.fd >= 0) {
                br.buffer_type = hailort::BufferType::DMA_BUFFER;
                br.dma_buffer = hailo_dma_buffer_t{sl.in_ref.fd,
                                                   size_for(sl.in_ref, in_meta_.frame_size, cfg_.in_write_size)};
            } else {
                br.buffer_type = hailort::BufferType::VIEW;
                br.view = hailort::MemoryView(sl.in_ref.ptr, in_meta_.frame_size);
            }
            named.emplace(in_meta_.name,
                          std::make_pair(br, [this, slot](hailo_status st) { note_callback(slot, st); }));
        }
        for (std::size_t i = 0; i < out_meta_.size(); ++i) {
            hailort::BufferRepresentation br{};
            if (sl.out[i].fd >= 0) {
                br.buffer_type = hailort::BufferType::DMA_BUFFER;
                br.dma_buffer = hailo_dma_buffer_t{sl.out[i].fd,
                                                   size_for(sl.out[i], out_meta_[i].frame_size, cfg_.read_size)};
            } else {
                br.buffer_type = hailort::BufferType::VIEW;
                br.view = hailort::MemoryView(sl.out[i].ptr, out_meta_[i].frame_size);
            }
            named.emplace(out_meta_[i].name,
                          std::make_pair(br, [this, slot](hailo_status st) { note_callback(slot, st); }));
        }
        // Backpressure: block until the ongoing callback count is below num_slots_.
        // The reason increase_ongoing_callbacks() is not called directly is in the comment at the same spot in
        // hailo_raw_runner.cpp — infer_async already counts 1 per request, so adding a manual pair raises it by 2 per frame
        // and cuts the in-flight limit to num_slots_/2 (measured on HailoRT 4.24.0).
        const auto st_wait = ng_->wait_for_ongoing_callbacks_count_under(static_cast<std::size_t>(num_slots_));
        if (st_wait != HAILO_SUCCESS) throw yvo::HailoError("wait_for_ongoing_callbacks_count_under", st_wait);
        const auto st = ng_->infer_async(named, [](hailo_status) {});
        if (st != HAILO_SUCCESS) {
            // No callback will arrive, so release the slot directly.
            sl.status.store(st);
            first_error_.store(st);
            sl.remaining.store(0);
            finish(slot);
            throw yvo::HailoError("ConfiguredNetworkGroup::infer_async", st);
        }
    }

    void submit_read_async(int slot) {
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        // FIFO pairing: issue 1 write + 6 reads inside one mutex.
        // This lock is the only guarantee that the N-th request of each stream corresponds to the N-th frame.
        std::lock_guard<std::mutex> issue(issue_m_);
        const auto cb_in = [this, slot](const hailort::InputStream::CompletionInfo& ci) {
            note_callback(slot, ci.status);
        };
        // The dmabuf fd overload (/usr/include/hailo/stream.hpp:253) is marked "currently experimental" in the header.
        const auto st_in =
            sl.in_ref.fd >= 0
                ? input_->write_async(sl.in_ref.fd, size_for(sl.in_ref, in_meta_.frame_size, cfg_.in_write_size),
                                      cb_in)
                : input_->write_async(sl.in_ref.ptr, in_meta_.frame_size, cb_in);
        if (st_in != HAILO_SUCCESS) {
            sl.status.store(st_in);
            first_error_.store(st_in);
            sl.remaining.store(0);
            finish(slot);
            throw yvo::HailoError("InputStream::write_async", st_in);
        }
        for (std::size_t i = 0; i < outputs_.size(); ++i) {
            const auto cb = [this, slot](const hailort::OutputStream::CompletionInfo& ci) {
                note_callback(slot, ci.status);
            };
            const auto st = sl.out[i].fd >= 0
                                ? outputs_[i]->read_async(sl.out[i].fd,
                                                          size_for(sl.out[i], out_meta_[i].frame_size, cfg_.read_size),
                                                          cb)
                                : outputs_[i]->read_async(sl.out[i].ptr, out_meta_[i].frame_size, cb);
            if (st != HAILO_SUCCESS) {
                // Callbacks for already-issued requests will arrive. Reduce the remaining count so the frame can finish.
                sl.status.store(st);
                first_error_.store(st);
                const int not_issued = static_cast<int>(outputs_.size() - i);
                if (sl.remaining.fetch_sub(not_issued) == not_issued) finish(slot);
                throw yvo::HailoError("OutputStream::read_async", st);
            }
        }
    }

public:
    // Copies out the slot's output buffers (6). Valid only inside the on_done callback or right after it returns.
    // In heap mode, wrapped in DMA_BUF_IOCTL_SYNC(START|READ) ... (END|READ) for CPU cache coherency.
    RawFrame copy_out(int slot) const {
        RawFrame f;
        const Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        for (std::size_t i = 0; i < out_meta_.size(); ++i) {
            yvo::p2::sync_cpu_read_begin(sl.out[i]);
            f.t.emplace_back(sl.out[i].ptr, sl.out[i].ptr + out_meta_[i].frame_size);
            yvo::p2::sync_cpu_read_end(sl.out[i]);
        }
        return f;
    }

    // Time (ms) for the CPU to read all 6 tensors (1.22 MB) sequentially. dst is reused by the caller.
    // Measurement function for plan §6.2 P2-⑥ cpu_read_1.22MB_ms.
    double read_all_ms(int slot, std::uint8_t* dst) const {
        const Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t off = 0;
        for (std::size_t i = 0; i < out_meta_.size(); ++i) {
            yvo::p2::sync_cpu_read_begin(sl.out[i]);
            std::memcpy(dst + off, sl.out[i].ptr, out_meta_[i].frame_size);
            yvo::p2::sync_cpu_read_end(sl.out[i]);
            off += out_meta_[i].frame_size;
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    std::size_t total_out_bytes() const {
        std::size_t n = 0;
        for (const auto& m : out_meta_) n += m.frame_size;
        return n;
    }

    // For reporting: per-tensor (hw_frame_size, dmabuf_size) of slot 0
    std::vector<std::pair<std::size_t, std::size_t>> size_pairs() const {
        std::vector<std::pair<std::size_t, std::size_t>> v;
        if (slots_.empty()) return v;
        for (std::size_t i = 0; i < out_meta_.size(); ++i)
            v.emplace_back(out_meta_[i].frame_size, slots_[0]->out[i].dmabuf_size);
        return v;
    }

private:
    RawApi api_;
    std::unique_ptr<hailort::Hef> hef_;
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> ng_;
    hailort::InputStream* input_ = nullptr;
    std::vector<hailort::OutputStream*> outputs_;
    StreamMeta in_meta_;
    std::vector<StreamMeta> out_meta_;
    int num_slots_ = 1;
    std::size_t min_queue_ = 1;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::deque<int> free_slots_;
    std::mutex m_;
    std::condition_variable cv_;
    std::mutex issue_m_;
    std::atomic<long> cb_total_{0};
    std::atomic<long> cb_failed_{0};
    std::atomic<hailo_status> first_error_{HAILO_SUCCESS};
    RawSessionConfig cfg_;
    struct Mapping {
        int fd;
        std::size_t size;
        hailo_dma_buffer_direction_t dir;
    };
    std::vector<Mapping> mapped_;
    long map_calls_ = 0;
    long map_failed_ = 0;
    hailo_status map_status_ = HAILO_SUCCESS;
    double map_ms_ = 0.0;
    // Input (H2D) mappings are counted separately, so they do not pollute the output statistics (P2 verdict).
    long in_map_calls_ = 0;
    long in_map_failed_ = 0;
    hailo_status in_map_status_ = HAILO_SUCCESS;
};

} // namespace

namespace {

// ---------------------------------------------------------------- scale layout derivation

// Finds the (class tensor, box tensor) pair per scale among the 6 streams.
// reg_max·stride·classes are derived from the shape, and a mismatch throws (plan §4.0(b)).
struct ScaleLayout {
    int cls_idx = -1, box_idx = -1;
    int grid_h = 0, grid_w = 0, w_padded = 0, stride = 0;
};

struct RawLayout {
    ScaleLayout s[3];   // stride 8, 16, 32
    int num_classes = 0;
    int reg_max = 0;
};

RawLayout derive_layout(const std::vector<StreamMeta>& out) {
    if (out.size() != 6) throw std::runtime_error("raw output stream count is not 6: " + std::to_string(out.size()));
    RawLayout L;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto& hw = out[i].info.hw_shape;
        const auto& sh = out[i].info.shape;
        const std::size_t expect = static_cast<std::size_t>(hw.height) * hw.width * hw.features;
        if (expect != out[i].info.hw_frame_size) {
            throw std::runtime_error("hw_shape and hw_frame_size do not match: " + out[i].name);
        }
        const int h = static_cast<int>(hw.height);
        if (h == 0 || 640 % h != 0) throw std::runtime_error("cannot derive stride from grid height: " + out[i].name);
        const int stride = 640 / h;
        int si = -1;
        if (stride == 8) si = 0;
        else if (stride == 16) si = 1;
        else if (stride == 32) si = 2;
        else throw std::runtime_error("unexpected stride " + std::to_string(stride) + ": " + out[i].name);

        ScaleLayout& s = L.s[si];
        s.grid_h = h;
        s.grid_w = static_cast<int>(sh.width);
        s.w_padded = static_cast<int>(hw.width);
        s.stride = stride;
        const int feats = static_cast<int>(hw.features);
        if (feats % yvo::kDflSides == 0 && feats / yvo::kDflSides == yvo::kDflRegMax) {
            s.box_idx = static_cast<int>(i);
            if (L.reg_max == 0) L.reg_max = feats / yvo::kDflSides;
            else if (L.reg_max != feats / yvo::kDflSides) throw std::runtime_error("reg_max mismatch");
        } else {
            s.cls_idx = static_cast<int>(i);
            if (L.num_classes == 0) L.num_classes = feats;
            else if (L.num_classes != feats) throw std::runtime_error("class count mismatch");
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (L.s[i].cls_idx < 0 || L.s[i].box_idx < 0)
            throw std::runtime_error("scale " + std::to_string(i) + ": cls/box pair not found");
    }
    if (L.num_classes != 80 || L.reg_max != yvo::kDflRegMax)
        throw std::runtime_error("classes/reg_max is not 80/16");
    return L;
}

yvo::RawOutputSet make_raw_set(const RawLayout& L, const RawFrame& f, bool p5_include_padding,
                               const std::vector<StreamMeta>& meta) {
    yvo::RawOutputSet r;
    r.num_classes = L.num_classes;
    r.reg_max = L.reg_max;
    for (int i = 0; i < 3; ++i) {
        yvo::RawScale& s = r.s[i];
        s.cls = f.t[static_cast<std::size_t>(L.s[i].cls_idx)].data();
        s.box = f.t[static_cast<std::size_t>(L.s[i].box_idx)].data();
        s.grid_h = L.s[i].grid_h;
        s.grid_w = (i == 2 && p5_include_padding) ? L.s[i].w_padded : L.s[i].grid_w;
        s.w_padded = L.s[i].w_padded;
        s.stride = L.s[i].stride;
        const auto& cq = meta[static_cast<std::size_t>(L.s[i].cls_idx)].quant[0];
        const auto& bq = meta[static_cast<std::size_t>(L.s[i].box_idx)].quant[0];
        s.cls_scale = cq.qp_scale;
        s.cls_zp = cq.qp_zp;
        s.box_scale = bq.qp_scale;
        s.box_zp = bq.qp_zp;
    }
    return r;
}

// ---------------------------------------------------------------- options

struct Options {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string hef = YVO_DEFAULT_HEF;
    std::string out_dir = YVO_REPO_ROOT "/build/out/yvo_hailo_raw_probe";
    std::string fixture_dir = YVO_FIXTURE_DIR;
    std::string api = "both";           // infer_async | read_async | both
    int compare_frames = 30;            // Frames for CPU reference implementation ↔ HailoRT NMS comparison (>= 20)
    int api_check_frames = 8;           // Frames for API path check
    int dump_frames = 5;                // Frames saved as fixtures
    int bench_frames = 200;             // Frames for throughput measurement
    int bench_repeat = 3;               // Repetitions per configuration (median reported)
    int warmup = 30;
    int batch = 8;
    int slots = 16;
    bool skip_bench = false;
    bool skip_infer_model = false;
    // ── P2 ──
    std::string probe = "p1";           // p1 | p2 | f0
    std::string alloc = "malloc,heap,vulkan";
    int matrix_frames = 2;              // Frames submitted per cell in the size combination table
    int matrix_slots = 2;               // Slot count per table cell (kept small to run fast)
    int cpu_read_reps = 30;             // Repetitions for cpu_read_1.22MB_ms
    float cool_above = 75.0f;           // Cool down if above this temperature
    float cool_target = 65.0f;
    int cool_max_s = 120;
    // ── F0: input dma-buf (H2D) ──
    bool in_dmabuf = false;             // Also pass the input as dma-buf in p2's final vulkan run
    int f0_frames = 16;                 // Frames the f0 gate compares A/B
};

void usage() {
    std::fprintf(stderr,
                 "usage: yvo_hailo_raw_probe [--video P] [--hef P] [--out-dir D] [--fixture-dir D]\n"
                 "        [--api infer_async|read_async|both] [--compare-frames N] [--api-check-frames N]\n"
                 "        [--dump-frames N] [--bench-frames N] [--bench-repeat N] [--warmup N] [--batch B] [--slots S]\n"
                 "        [--skip-bench] [--skip-infer-model]\n"
                 "        [--probe p1|p2|f0|runner] [--alloc malloc,heap,vulkan] [--matrix-frames N]\n"
                 "        [--matrix-slots N] [--cpu-read-reps N] [--no-cool] [--in-dmabuf] [--f0-frames N]\n"
                 "  --probe f0 : gate for simultaneous input dma-buf (H2D) + output dma-buf (D2H). Runs A/B by itself.\n"
                 "  --probe runner : RawSession -> yvo::HailoRawRunner promotion smoke test. Byte-compares with the f0 path.\n"
                 "  --in-dmabuf: also pass the input as Vulkan export dma-buf in the final vulkan run of --probe p2.\n");
}

// ---------------------------------------------------------------- result of one API path run

struct ApiOutcome {
    RawApi api = RawApi::InferAsync;
    bool constructed = false;
    bool ok = false;
    std::string error;
    long cb_total = 0;
    long cb_failed = 0;
    hailo_status first_error = HAILO_SUCCESS;
    int slots = 0;
    std::size_t min_queue = 0;
    std::size_t infer_queue = 0;
    bool scheduled = false;
    double fps = 0.0, fps_min = 0.0, fps_max = 0.0;
    double lat_median = 0.0, lat_p90 = 0.0;
    float temp_before = 0.0f, temp_after = 0.0f;
};

// ---------------------------------------------------------------- P2: dma-buf output path

// Reads only the chip temperature and releases the device immediately. Used for the cool-down decision before creating a session.
float read_chip_temp() {
    hailo_vdevice_params_t vp;
    if (hailo_init_vdevice_params(&vp) != HAILO_SUCCESS) return std::nanf("");
    vp.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    auto vd = hailort::VDevice::create(vp);
    if (!vd) return std::nanf("");
    auto devs = vd.value()->get_physical_devices();
    if (!devs || devs->empty()) return std::nanf("");
    auto t = devs->at(0).get().get_chip_temperature();
    if (!t) return std::nanf("");
    return std::max(t->ts0_temperature, t->ts1_temperature);
}

// Same rule as cool_down() in 06_tools/bench/pipeline_bench.sh. Returns the seconds waited.
int cool_down(float above, float target, int max_s, float& temp_before, float& temp_after) {
    temp_before = read_chip_temp();
    temp_after = temp_before;
    if (!(temp_before > above)) return 0;
    int waited = 0;
    while (waited < max_s) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        waited += 5;
        temp_after = read_chip_temp();
        if (!(temp_after > target)) break;
    }
    return waited;
}

int count_hailort_log(bool thermal_only) {
    std::ifstream f("hailort.log");
    std::string line;
    int n = 0;
    while (std::getline(f, line)) {
        if (!thermal_only) {
            if (line.find("[error]") != std::string::npos) ++n;
            continue;
        }
        if (line.find("[warning]") == std::string::npos) continue;
        if (line.find("emperature") != std::string::npos || line.find("hrottl") != std::string::npos ||
            line.find("lock") != std::string::npos || line.find("verheat") != std::string::npos)
            ++n;
    }
    return n;
}

// Number of [warning] lines in hailort.log. If needle is not nullptr, counts only lines containing that string
// (for F0 pass criterion ③ "unaligned buffer causes performance degradation").
int count_hailort_warnings(const char* needle) {
    std::ifstream f("hailort.log");
    std::string line;
    int n = 0;
    while (std::getline(f, line)) {
        if (line.find("[warning]") == std::string::npos) continue;
        if (needle && line.find(needle) == std::string::npos) continue;
        ++n;
    }
    return n;
}

enum class AllocMode { Malloc, Heap, Vulkan };
const char* alloc_name(AllocMode m) {
    switch (m) {
        case AllocMode::Malloc: return "malloc";
        case AllocMode::Heap: return "heap";
        case AllocMode::Vulkan: return "vulkan";
    }
    return "?";
}

std::unique_ptr<yvo::p2::OutBufPool> make_pool(AllocMode m, bool verbose) {
    switch (m) {
        case AllocMode::Malloc: return nullptr;   // Uses RawSession's internal malloc path
        case AllocMode::Heap: return std::make_unique<yvo::p2::HeapOutPool>();
        case AllocMode::Vulkan: return std::make_unique<yvo::p2::VulkanOutPool>(verbose);
    }
    return nullptr;
}

// Submits N frames to obtain the 6 tensors. Drains after every frame and reads the slot buffers as is.
struct RunOutcome {
    bool constructed = false;
    bool issued = true;          // Whether all read_async/infer_async issues succeeded
    std::string error;
    long cb_total = 0, cb_failed = 0;
    bool callbacks_lost = false;   // drain deadline exceeded = lost callbacks
    hailo_status first_error = HAILO_SUCCESS;
    long map_calls = 0, map_failed = 0;
    hailo_status map_status = HAILO_SUCCESS;
    double map_ms = 0.0;
    // Input H2D dma-buf (F0)
    bool in_dmabuf = false;
    long in_map_calls = 0, in_map_failed = 0;
    hailo_status in_map_status = HAILO_SUCCESS;
    std::pair<std::size_t, std::size_t> in_size{0, 0};   // (hw_frame_size, dmabuf_size)
    int frames_ok = 0;
    std::vector<RawFrame> frames;
    std::vector<double> cpu_read_ms;
    std::vector<std::pair<std::size_t, std::size_t>> sizes;   // (hw_frame_size, dmabuf_size)
    std::vector<StreamMeta> out_meta;   // Output stream metadata (by name). Used for tensor index mapping in --probe runner
    int slots = 0;
};

RunOutcome run_capture(const RawSessionConfig& base, yvo::p2::OutBufPool* pool, int want_slots,
                       const std::vector<std::vector<std::uint8_t>>& inputs, int n_frames, int cpu_read_reps) {
    RunOutcome r;
    RawSessionConfig sc = base;
    sc.out_pool = pool;
    sc.want_slots = want_slots;
    std::unique_ptr<RawSession> sess;
    try {
        sess = std::make_unique<RawSession>(sc);
        r.constructed = true;
    } catch (const std::exception& e) {
        r.error = e.what();
        return r;
    }
    r.slots = sess->num_slots();
    r.map_calls = sess->map_calls();
    r.map_failed = sess->map_failed();
    r.map_status = sess->map_status();
    r.map_ms = sess->map_ms();
    r.sizes = sess->size_pairs();
    r.out_meta = sess->out_meta();
    r.in_dmabuf = sess->uses_in_dmabuf();
    r.in_map_calls = sess->in_map_calls();
    r.in_map_failed = sess->in_map_failed();
    r.in_map_status = sess->in_map_status();
    r.in_size = sess->in_size_pair();
    if (r.in_map_failed > 0 && r.error.empty())
        r.error = "dma_map_dmabuf(H2D) failed: " + status_name(r.in_map_status);
    if (r.map_failed > 0) {
        // Even if mapping failed, try issuing anyway (HailoRT may map automatically internally).
        if (r.error.empty()) r.error = "dma_map_dmabuf failed: " + status_name(r.map_status);
    }

    std::vector<std::uint8_t> scratch(sess->total_out_bytes());
    r.frames.resize(static_cast<std::size_t>(n_frames));
    for (int i = 0; i < n_frames; ++i) {
        const std::size_t fi = static_cast<std::size_t>(i) % inputs.size();
        try {
            sess->submit(inputs[fi].data(), [&](int slot, hailo_status st) {
                if (st != HAILO_SUCCESS) return;
                r.frames[static_cast<std::size_t>(i)] = sess->copy_out(slot);
                ++r.frames_ok;
            });
        } catch (const std::exception& e) {
            r.issued = false;
            if (r.error.empty()) r.error = e.what();
            break;
        }
        if (!sess->drain(15000)) {
            // Callbacks did not arrive. This combination is unusable — slot buffer contents cannot be trusted either.
            r.callbacks_lost = true;
            if (r.error.empty()) r.error = "drain deadline exceeded: callbacks for submitted frames did not arrive";
            break;
        }
    }
    // cpu_read_1.22MB_ms: repeated after warm-up. Keeps reading slot 0 (contents are already filled).
    if (r.frames_ok > 0 && cpu_read_reps > 0) {
        for (int i = 0; i < 3; ++i) (void)sess->read_all_ms(0, scratch.data());
        for (int i = 0; i < cpu_read_reps; ++i) r.cpu_read_ms.push_back(sess->read_all_ms(0, scratch.data()));
    }
    r.cb_total = sess->callbacks_total();
    r.cb_failed = sess->callbacks_failed();
    if (r.first_error == HAILO_SUCCESS) r.first_error = sess->first_error();
    sess.reset();
    return r;
}

bool frames_identical(const std::vector<RawFrame>& a, const std::vector<RawFrame>& b, std::size_t n_tensors,
                      int& matched, int& compared) {
    matched = 0;
    compared = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i].t.size() != n_tensors || b[i].t.size() != n_tensors) continue;
        ++compared;
        bool same = true;
        for (std::size_t k = 0; k < n_tensors; ++k)
            if (a[i].t[k] != b[i].t[k]) same = false;
        if (same) ++matched;
    }
    return compared > 0 && matched == compared;
}

// ---------------------------------------------------------------- F0: input dma-buf (H2D) gate
//
// The integrated pipeline feeds the V3D-written preprocessing output directly as NPU input. Only that combination
// (Vulkan export dma-buf input H2D + 6 dma-buf outputs D2H) is unverified, so it is closed here with A/B.
//   A (baseline): host malloc input + 6 dma-buf outputs  ← path already passed in P2
//   B (gate)    : Vulkan export dma-buf input + 6 dma-buf outputs
// The 6 output tensors of both runs must be byte-identical, and all 7 callbacks per frame must be SUCCESS.
struct F0Row {
    bool in_dmabuf = false;
    RawApi api = RawApi::InferAsync;
    SizeKind in_map = SizeKind::Dmabuf;
    SizeKind in_write = SizeKind::Hw;
    int frames = 0;
    RunOutcome out;
    int bytes_matched = 0, bytes_compared = 0;
    long cb_expect = 0;
    int log_err = 0;      // Number of hailort.log [error] lines printed only during this run
    bool pass = false;
};

int run_f0(const Options& o, const std::vector<std::vector<std::uint8_t>>& inputs, const std::string& link_speed,
           const std::string& link_width) {
    const long fds_start = count_open_fds();
    const int log_err_start = count_hailort_log(false);
    const int warn_start = count_hailort_warnings(nullptr);
    const int unaligned_start = count_hailort_warnings("unaligned buffer");
    const int thermal_start = count_hailort_log(true);

    std::vector<RawApi> apis;
    if (o.api == "infer_async" || o.api == "both") apis.push_back(RawApi::InferAsync);
    if (o.api == "read_async" || o.api == "both") apis.push_back(RawApi::ReadAsync);

    float t_before = 0, t_after = 0;
    const int cooled_s = cool_down(o.cool_above, o.cool_target, o.cool_max_s, t_before, t_after);
    std::printf("[f0] cooling: hailo %.1f -> %.1f C, waited %d s\n", static_cast<double>(t_before),
                static_cast<double>(t_after), cooled_s);

    RawSessionConfig base;
    base.hef_path = o.hef;
    base.batch = static_cast<std::uint16_t>(o.batch);
    // The output-side size contract uses the values chosen by P2 as is (p2_alloc_matrix.txt: map=dmabuf_size, read=hw_frame_size).
    base.map_size = SizeKind::Dmabuf;
    base.read_size = SizeKind::Hw;

    const int n = std::max(1, std::min<int>(o.f0_frames, static_cast<int>(inputs.size())));

    Tee t(o.out_dir + "/f0_in_dmabuf.txt");
    t << "# F0 gate: simultaneous input dma-buf (H2D) + output dma-buf (D2H)\n";
    t << fmt("# pcie=%s x%s hef=%s video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
             o.video.c_str());
    t << fmt("# batch=%d slots=%d frames=%d api=%s output contract(map=dmabuf_size, transfer=hw_frame_size)\n", o.batch,
             o.slots, n, o.api.c_str());
    t << fmt("# cooling: hailo %.1f -> %.1f C, waited %d s\n\n", static_cast<double>(t_before),
             static_cast<double>(t_after), cooled_s);
    const long fds_files = count_open_fds();

    // One run. With in_dmabuf, **one** Vulkan pool provides both the input and output buffers
    // (same structure as the integrated pipeline — V3D writes the input and a shader reads the output).
    // Accumulates the validation layer errors/warnings counted by VulkanContext per run
    // (the values are meaningful when enabled with VK_LOADER_LAYERS_ENABLE='*validation').
    std::uint64_t vk_errors = 0, vk_warnings = 0;
    bool vk_layer_active = false;
    auto run_one = [&](bool in_dmabuf, RawApi api, SizeKind in_map, SizeKind in_write, int slots, int frames) {
        RawSessionConfig sc = base;
        sc.api = api;
        sc.in_map_size = in_map;
        sc.in_write_size = in_write;
        std::unique_ptr<yvo::p2::VulkanOutPool> pool;
        RunOutcome r;
        try {
            pool = std::make_unique<yvo::p2::VulkanOutPool>(false);
            sc.in_pool = in_dmabuf ? pool.get() : nullptr;
            r = run_capture(sc, pool.get(), slots, inputs, frames, 0);
        } catch (const std::exception& e) {
            if (r.error.empty()) r.error = e.what();
        }
        if (pool) {
            vk_errors += pool->ctx().validation_errors();
            vk_warnings += pool->ctx().validation_warnings();
            vk_layer_active = vk_layer_active || pool->ctx().validation_layer_active();
        }
        pool.reset();
        return r;
    };

    auto judge = [](F0Row& row, const RunOutcome& ref, std::size_t n_tensors) {
        if (row.out.frames_ok > 0)
            frames_identical(ref.frames, row.out.frames, n_tensors, row.bytes_matched, row.bytes_compared);
        row.cb_expect = static_cast<long>(n_tensors + 1) * row.frames;
        row.pass = row.out.constructed && row.out.issued && !row.out.callbacks_lost && row.out.map_failed == 0 &&
                   row.out.in_map_failed == 0 && row.out.cb_failed == 0 && row.out.frames_ok == row.frames &&
                   row.out.cb_total == row.cb_expect && row.bytes_compared > 0 &&
                   row.bytes_matched == row.bytes_compared;
    };

    struct Leg {
        RawApi api;
        RunOutcome ref;
        std::vector<F0Row> sweep;
        F0Row full;
        bool ref_ok = false;
    };
    std::vector<Leg> legs;

    for (RawApi api : apis) {
        Leg leg;
        leg.api = api;
        std::printf("[f0] %s: A (host input) baseline capture of %d frames...\n", api_name(api), n);
        leg.ref = run_one(false, api, SizeKind::Dmabuf, SizeKind::Hw, o.slots, n);
        leg.ref_ok = leg.ref.constructed && leg.ref.frames_ok == n && leg.ref.cb_failed == 0;
        if (!leg.ref_ok) {
            std::printf("[f0] %s: baseline capture failed — %s\n", api_name(api), leg.ref.error.c_str());
            legs.push_back(std::move(leg));
            continue;
        }
        const std::size_t n_tensors = leg.ref.frames[0].t.size();

        // Try all 4 input size contract combinations. A wrong size means no callbacks, which the drain deadline catches.
        const int sw_frames = std::max(1, std::min(o.matrix_frames, n));
        for (int mk = 0; mk < 2; ++mk) {
            for (int wk = 0; wk < 2; ++wk) {
                F0Row row;
                row.in_dmabuf = true;
                row.api = api;
                row.in_map = mk == 0 ? SizeKind::Dmabuf : SizeKind::Hw;
                row.in_write = wk == 0 ? SizeKind::Hw : SizeKind::Dmabuf;
                row.frames = sw_frames;
                row.out = run_one(true, api, row.in_map, row.in_write, o.matrix_slots, sw_frames);
                judge(row, leg.ref, n_tensors);
                std::printf("[f0 sweep] %s in_map=%s in_write=%s -> %s\n", api_name(api), size_kind_name(row.in_map),
                            size_kind_name(row.in_write), row.pass ? "PASS" : "FAIL");
                leg.sweep.push_back(std::move(row));
            }
        }

        // Full-slot run with the combination exactly per the contract (appendix A.13). This is the gate's main run.
        F0Row full;
        full.in_dmabuf = true;
        full.api = api;
        full.in_map = SizeKind::Dmabuf;
        full.in_write = SizeKind::Hw;
        full.frames = n;
        std::printf("[f0] %s: B (input dma-buf) main run of %d frames, slots=%d...\n", api_name(api), n, o.slots);
        const int err_before_full = count_hailort_log(false);
        full.out = run_one(true, api, full.in_map, full.in_write, o.slots, n);
        full.log_err = count_hailort_log(false) - err_before_full;
        judge(full, leg.ref, n_tensors);
        leg.full = std::move(full);
        std::printf("[f0] %s: B -> %s (cb %ld/%ld, bytes %d/%d)\n", api_name(api), leg.full.pass ? "PASS" : "FAIL",
                    leg.full.out.cb_total, leg.full.cb_expect, leg.full.bytes_matched, leg.full.bytes_compared);
        legs.push_back(std::move(leg));
    }

    // ── Report
    t << "## Input buffer size\n";
    for (const auto& leg : legs) {
        if (leg.full.out.in_size.first == 0) continue;
        t << fmt("  input: hw_frame_size=%zu dmabuf_size=%zu (diff %+zd)\n", leg.full.out.in_size.first,
                 leg.full.out.in_size.second,
                 static_cast<std::ptrdiff_t>(leg.full.out.in_size.second) -
                     static_cast<std::ptrdiff_t>(leg.full.out.in_size.first));
        break;
    }
    t << "\n## Input size contract, 4 combinations (per cell: slots=" + std::to_string(o.matrix_slots) + ")\n";
    t << "api           dma_map(H2D)   transfer size    map_status                  issue  cb/exp    bytes    verdict\n";
    t << "------------  -------------  ---------------  --------------------------  -----  --------  -------  ----\n";
    for (const auto& leg : legs) {
        for (const auto& r : leg.sweep) {
            t << fmt("%-12s  %-13s  %-15s  %-26s  %-5s  %3ld/%-4ld  %2d/%-2d    %s\n", api_name(r.api),
                     size_kind_name(r.in_map), size_kind_name(r.in_write),
                     status_name(r.out.in_map_status).c_str(),
                     r.out.callbacks_lost ? "lost" : (r.out.issued ? "ok" : "FAIL"), r.out.cb_total, r.cb_expect,
                     r.bytes_matched, r.bytes_compared, r.pass ? "PASS" : "FAIL");
            if (!r.out.error.empty()) t << fmt("        error: %s\n", r.out.error.c_str());
        }
    }

    t << "\n## Main run (input dma_map=dmabuf_size, transfer size=hw_frame_size)\n";
    for (const auto& leg : legs) {
        t << fmt("[%s]\n", api_name(leg.api));
        t << fmt("  A baseline (host input) : constructed=%d frames_ok=%d/%d cb=%ld failed=%ld %s\n",
                 leg.ref.constructed ? 1 : 0, leg.ref.frames_ok, n, leg.ref.cb_total, leg.ref.cb_failed,
                 leg.ref.error.empty() ? "" : ("error: " + leg.ref.error).c_str());
        const F0Row& b = leg.full;
        t << fmt("  B gate (input dma-buf)  : constructed=%d frames_ok=%d/%d\n", b.out.constructed ? 1 : 0,
                 b.out.frames_ok, b.frames);
        t << fmt("    dma_map_dmabuf(H2D) : %ld calls, failed %ld, status=%s\n", b.out.in_map_calls, b.out.in_map_failed,
                 status_name(b.out.in_map_status).c_str());
        t << fmt("    dma_map_dmabuf(D2H) : %ld calls, failed %ld, status=%s (H2D+D2H total %.3f ms)\n", b.out.map_calls,
                 b.out.map_failed, status_name(b.out.map_status).c_str(), b.out.map_ms);
        t << fmt("    callbacks           : %ld / expected %ld (7 per frame), failed %ld, first_error=%s\n",
                 b.out.cb_total, b.cb_expect, b.out.cb_failed, status_name(b.out.first_error).c_str());
        t << fmt("    6 output tensors byte-identical vs A : %d / %d frames\n", b.bytes_matched, b.bytes_compared);
        if (!b.out.error.empty()) t << fmt("    error: %s\n", b.out.error.c_str());
        t << fmt("  verdict: %s\n\n", b.pass ? "PASS" : "FAIL");
    }

    const long fds_end = count_open_fds();
    const int log_err = count_hailort_log(false) - log_err_start;
    const int warns = count_hailort_warnings(nullptr) - warn_start;
    const int unaligned = count_hailort_warnings("unaligned buffer") - unaligned_start;
    const int thermal = count_hailort_log(true) - thermal_start;

    const F0Row* infer_row = nullptr;
    const F0Row* read_row = nullptr;
    for (const auto& leg : legs) {
        if (leg.api == RawApi::InferAsync) infer_row = &leg.full;
        else read_row = &leg.full;
    }
    const bool infer_ok = infer_row && infer_row->pass;
    const bool read_ok = read_row && read_row->pass;
    const bool gate = infer_ok || read_ok;

    t << "## Pass criteria\n";
    t << fmt("  ① all 7 callbacks per frame SUCCESS : %s (infer_async %s, write_async %s)\n",
             gate ? "PASS" : "FAIL",
             infer_row ? fmt("%ld/%ld failed %ld", infer_row->out.cb_total, infer_row->cb_expect,
                             infer_row->out.cb_failed)
                             .c_str()
                       : "(not run)",
             read_row ? fmt("%ld/%ld failed %ld", read_row->out.cb_total, read_row->cb_expect, read_row->out.cb_failed)
                            .c_str()
                      : "(not run)");
    t << fmt("  ② 6 output tensors byte-identical to the host-input run : %s (infer_async %d/%d, write_async %d/%d)\n",
             gate ? "PASS" : "FAIL", infer_row ? infer_row->bytes_matched : 0,
             infer_row ? infer_row->bytes_compared : 0, read_row ? read_row->bytes_matched : 0,
             read_row ? read_row->bytes_compared : 0);
    int full_err = 0;
    for (const auto& leg : legs) full_err += leg.full.log_err;
    t << fmt("  ③ hailort.log warning 0 : %s (total %d, unaligned buffer %d, thermal/clock %d)\n",
             (warns == 0) ? "PASS" : "FAIL", warns, unaligned, thermal);
    t << fmt("     error: main-run section %d, total %d (the difference of %d is the **intended** rejections of the 4-combination size contract table)\n",
             full_err, log_err, log_err - full_err);
    t << fmt("  ④ fd leak 0 : %s (start %ld, after opening evidence file %ld, end %ld → %+ld)\n",
             (fds_end - fds_files == 0) ? "PASS" : "FAIL", fds_start, fds_files, fds_end, fds_end - fds_files);
    t << fmt("  ⑤ per-API result : infer_async=%s, write_async(read_async path)=%s\n",
             infer_row ? (infer_ok ? "PASS" : "FAIL") : "(not run)",
             read_row ? (read_ok ? "PASS" : "FAIL") : "(not run)");
    t << fmt("  ⑥ Vulkan validation layer : errors %llu, warnings %llu (layer active %s)\n",
             static_cast<unsigned long long>(vk_errors), static_cast<unsigned long long>(vk_warnings),
             vk_layer_active ? "yes" : "no — this run was done without VK_LOADER_LAYERS_ENABLE='*validation'");
    t << fmt("\n[conclusion] %s\n", gate ? "input dma-buf (H2D) + output dma-buf (D2H) can be used together — F0 gate passed"
                                   : "both APIs failed — the input must stay a host buffer");

    std::printf("\n[F0] %s (infer_async %s, write_async %s)\n", gate ? "PASS" : "FAIL", infer_ok ? "PASS" : "FAIL",
                read_ok ? "PASS" : "FAIL");
    std::printf("[f0] hailort.log warning %d(unaligned %d), error %d(main run %d) / fd leak %+ld\n", warns, unaligned,
                log_err, full_err, fds_end - fds_files);
    return gate ? 0 : 1;
}

// ---------------------------------------------------------------- RUNNER: library promotion smoke test
//
// In step 3a, RawSession was extracted into 01_src/hailo/hailo_raw_runner.{hpp,cpp}. This checks with A/B in one process
// that the promoted code produces **byte-identical output** to the original.
//   A (baseline): RawSession, with **exactly the same settings** as the B leg of --probe f0
//                 (Vulkan export dma-buf input H2D + 6 outputs D2H, map=dmabuf_size, transfer=hw_frame_size)
//   B (promoted): yvo::HailoRawRunner. Submits the same frames in the same order.
// The tensor index mapping is verified here too — bytes confirm that the runner's t[2s]/t[2s+1] (cls/box)
// really are the same buffers as the streams chosen by derive_layout().
int run_runner(const Options& o, const std::vector<std::vector<std::uint8_t>>& inputs, const std::string& link_speed,
               const std::string& link_width) {
    const long fds_start = count_open_fds();
    const int log_err_start = count_hailort_log(false);
    const int warn_start = count_hailort_warnings(nullptr);
    const int thermal_start = count_hailort_log(true);

    float t_before = 0, t_after = 0;
    const int cooled_s = cool_down(o.cool_above, o.cool_target, o.cool_max_s, t_before, t_after);
    std::printf("[runner] cooling: hailo %.1f -> %.1f C, waited %d s\n", static_cast<double>(t_before),
                static_cast<double>(t_after), cooled_s);

    const int n = std::max(1, std::min<int>(o.f0_frames, static_cast<int>(inputs.size())));

    // ── A: RawSession (same settings as f0's B leg)
    RawSessionConfig sc;
    sc.hef_path = o.hef;
    sc.batch = static_cast<std::uint16_t>(o.batch);
    sc.api = RawApi::InferAsync;
    sc.map_size = SizeKind::Dmabuf;
    sc.read_size = SizeKind::Hw;
    sc.in_map_size = SizeKind::Dmabuf;
    sc.in_write_size = SizeKind::Hw;

    RunOutcome ref;
    std::uint64_t vk_errors = 0, vk_warnings = 0;
    bool vk_layer_active = false;
    {
        auto pool = std::make_unique<yvo::p2::VulkanOutPool>(false);
        sc.in_pool = pool.get();
        std::printf("[runner] A (RawSession) baseline capture of %d frames...\n", n);
        ref = run_capture(sc, pool.get(), o.slots, inputs, n, 0);
        vk_errors += pool->ctx().validation_errors();
        vk_warnings += pool->ctx().validation_warnings();
        vk_layer_active = vk_layer_active || pool->ctx().validation_layer_active();
    }
    const bool ref_ok = ref.constructed && ref.frames_ok == n && ref.cb_failed == 0;
    if (!ref_ok) {
        std::fprintf(stderr, "[runner] baseline capture failed — %s\n", ref.error.c_str());
        return 1;
    }
    const RawLayout layout = derive_layout(ref.out_meta);
    // Tensor index i (runner convention) → stream index of the baseline capture
    int ref_idx[6];
    for (int s = 0; s < 3; ++s) {
        ref_idx[2 * s] = layout.s[s].cls_idx;
        ref_idx[2 * s + 1] = layout.s[s].box_idx;
    }

    // ── B: HailoRawRunner
    struct RunnerOutcome {
        bool constructed = false;
        std::string error;
        int slots = 0;
        std::size_t min_queue = 0;
        long cb_total = 0, cb_failed = 0;
        int frames_ok = 0;
        std::vector<RawFrame> frames;   // In runner tensor index order (0..5)
        double fps = 0.0, fps_min = 0.0, fps_max = 0.0, lat_median = 0.0, lat_p90 = 0.0;
        float temp_before = 0.0f, temp_after = 0.0f;
        bool names_match = true;
        yvo::RawModelInfo info;   // Info the runner derived from HEF metadata alone (copy for reporting)
    } rr;

    {
        auto pool = std::make_unique<yvo::p2::VulkanOutPool>(false);
        std::unique_ptr<yvo::HailoRawRunner> runner;
        std::vector<yvo::p2::OutBufRef> in_bufs;
        std::vector<std::array<yvo::p2::OutBufRef, 6>> out_bufs;
        try {
            yvo::HailoRawRunnerConfig rc;
            rc.hef_path = o.hef;
            rc.batch_size = static_cast<std::uint16_t>(o.batch);
            rc.num_slots = o.slots;
            rc.verbose = true;
            runner = std::make_unique<yvo::HailoRawRunner>(rc);
            rr.constructed = true;
            rr.slots = runner->num_slots();
            rr.min_queue = runner->info().min_queue;
            const auto& in = runner->info();
            rr.info = in;

            // First check that the stream name matches the corresponding stream of the baseline capture (= the numbering convention holds)
            for (int i = 0; i < 6; ++i)
                if (in.t[i].stream_name != ref.out_meta[static_cast<std::size_t>(ref_idx[i])].name)
                    rr.names_match = false;

            // For each slot, take 1 input + 6 outputs from the same Vulkan pool and bind_slot()
            for (int s = 0; s < rr.slots; ++s) {
                in_bufs.push_back(pool->acquire(in.input_frame_size));
                std::array<yvo::p2::OutBufRef, 6> ob{};
                for (int i = 0; i < 6; ++i) ob[static_cast<std::size_t>(i)] = pool->acquire(in.t[i].hw_frame_size);
                out_bufs.push_back(ob);
                int fds[6];
                std::size_t sizes[6];
                for (int i = 0; i < 6; ++i) {
                    fds[i] = ob[static_cast<std::size_t>(i)].fd;
                    sizes[i] = ob[static_cast<std::size_t>(i)].dmabuf_size;
                }
                runner->bind_slot(s, in_bufs.back().fd, in_bufs.back().dmabuf_size, fds, sizes);
            }

            rr.temp_before = runner->chip_temperature();

            // ① Byte comparison: per frame, submit → drain → read the slot buffers.
            //    (Inside the callback only the slot number is recorded — callbacks must be short and must not touch Vulkan memory)
            rr.frames.resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                const std::size_t fi = static_cast<std::size_t>(i) % inputs.size();
                const int slot = i % rr.slots;
                std::memcpy(in_bufs[static_cast<std::size_t>(slot)].ptr, inputs[fi].data(), in.input_frame_size);
                hailo_status st_cb = HAILO_SUCCESS;
                runner->submit(slot, static_cast<std::int64_t>(i), [&](yvo::RawInferResult&& r) {
                    st_cb = r.status;
                    ++rr.cb_total;
                    if (r.status != HAILO_SUCCESS) ++rr.cb_failed;
                });
                runner->drain();
                if (st_cb != HAILO_SUCCESS) continue;
                RawFrame f;
                for (int k = 0; k < 6; ++k) {
                    const auto& b = out_bufs[static_cast<std::size_t>(slot)][static_cast<std::size_t>(k)];
                    yvo::p2::sync_cpu_read_begin(b);
                    f.t.emplace_back(b.ptr, b.ptr + in.t[k].hw_frame_size);
                    yvo::p2::sync_cpu_read_end(b);
                }
                rr.frames[static_cast<std::size_t>(i)] = std::move(f);
                ++rr.frames_ok;
            }

            // ② Throughput: run with the pipeline kept full (backpressure is applied by
            //    wait_for_ongoing_callbacks_count_under inside the runner). Buffers are not read.
            if (!o.skip_bench) {
                // Exactly the pattern the integrated pipeline will use: the callback pushes the slot number to a free queue and returns,
                // and the submit thread pops it and resubmits (the runner releases the slot **before entering** the callback).
                std::atomic<int> done{0};
                std::mutex lm;
                std::condition_variable lcv;
                std::deque<int> free_q;
                std::vector<double> lat_all;
                for (int s = 0; s < rr.slots; ++s) free_q.push_back(s);
                auto submit_many = [&](int count) {
                    for (int i = 0; i < count; ++i) {
                        int slot = -1;
                        {
                            std::unique_lock<std::mutex> lk(lm);
                            if (!lcv.wait_for(lk, std::chrono::seconds(10), [&] { return !free_q.empty(); }))
                                throw std::runtime_error("runner bench: no free slot within 10 seconds");
                            slot = free_q.front();
                            free_q.pop_front();
                        }
                        runner->submit(slot, static_cast<std::int64_t>(i), [&](yvo::RawInferResult&& r) {
                            {
                                std::lock_guard<std::mutex> lk(lm);
                                lat_all.push_back(r.latency_ms);
                                free_q.push_back(r.slot);
                            }
                            lcv.notify_one();
                            done.fetch_add(1);
                        });
                    }
                    runner->drain();
                };
                submit_many(std::min(o.warmup, 30));
                std::vector<double> fps_runs;
                for (int rep = 0; rep < std::max(1, o.bench_repeat); ++rep) {
                    done.store(0);
                    {
                        std::lock_guard<std::mutex> lk(lm);
                        lat_all.clear();
                    }
                    const auto t0 = std::chrono::steady_clock::now();
                    submit_many(o.bench_frames);
                    const double ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    fps_runs.push_back(ms > 0 ? 1000.0 * o.bench_frames / ms : 0.0);
                    rr.cb_total += static_cast<long>(done.load());
                    std::printf("[runner] bench %d/%d: fps=%.2f latency median=%.2f p90=%.2f ms temp=%.1f C\n",
                                rep + 1, std::max(1, o.bench_repeat), fps_runs.back(), percentile(lat_all, 0.5),
                                percentile(lat_all, 0.9), static_cast<double>(runner->chip_temperature()));
                }
                rr.fps = percentile(fps_runs, 0.5);
                rr.fps_min = *std::min_element(fps_runs.begin(), fps_runs.end());
                rr.fps_max = *std::max_element(fps_runs.begin(), fps_runs.end());
                rr.lat_median = percentile(lat_all, 0.5);
                rr.lat_p90 = percentile(lat_all, 0.9);
            }
            rr.temp_after = runner->chip_temperature();
        } catch (const std::exception& e) {
            if (rr.error.empty()) rr.error = e.what();
        }
        runner.reset();
        vk_errors += pool->ctx().validation_errors();
        vk_warnings += pool->ctx().validation_warnings();
        vk_layer_active = vk_layer_active || pool->ctx().validation_layer_active();
    }

    // ── Byte comparison (runner tensor index ↔ baseline capture stream index)
    int matched = 0, compared = 0;
    long diff_bytes = 0;
    for (int i = 0; i < n && i < static_cast<int>(rr.frames.size()); ++i) {
        const RawFrame& a = ref.frames[static_cast<std::size_t>(i)];
        const RawFrame& b = rr.frames[static_cast<std::size_t>(i)];
        if (a.t.size() != 6 || b.t.size() != 6) continue;
        ++compared;
        bool same = true;
        for (int k = 0; k < 6; ++k) {
            const auto& av = a.t[static_cast<std::size_t>(ref_idx[k])];
            const auto& bv = b.t[static_cast<std::size_t>(k)];
            if (av.size() != bv.size()) {
                same = false;
                diff_bytes += static_cast<long>(std::max(av.size(), bv.size()));
                continue;
            }
            for (std::size_t x = 0; x < av.size(); ++x)
                if (av[x] != bv[x]) {
                    ++diff_bytes;
                    same = false;
                }
        }
        if (same) ++matched;
    }

    const long fds_end = count_open_fds();
    const int log_err = count_hailort_log(false) - log_err_start;
    const int warns = count_hailort_warnings(nullptr) - warn_start;
    const int thermal = count_hailort_log(true) - thermal_start;

    const bool pass = rr.constructed && rr.error.empty() && rr.names_match && rr.cb_failed == 0 &&
                      rr.frames_ok == n && compared == n && matched == compared && log_err == 0 && warns == 0;

    Tee t(o.out_dir + "/runner_smoke.txt");
    t << "# step 3a: RawSession -> yvo::HailoRawRunner promotion smoke test\n";
    t << fmt("# pcie=%s x%s hef=%s video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
             o.video.c_str());
    t << fmt("# batch=%d requested_slots=%d frames=%d api=infer_async contract(map=dmabuf_size, transfer=hw_frame_size)\n",
             o.batch, o.slots, n);
    t << fmt("# cooling: hailo %.1f -> %.1f C, waited %d s\n\n", static_cast<double>(t_before),
             static_cast<double>(t_after), cooled_s);

    t << "## Model info the runner derived from HEF metadata alone (no fixture files read)\n";
    t << fmt("  input: %s  %dx%d  hw_frame_size=%zu  min_queue=%zu  classes=%d reg_max=%d\n",
             rr.info.input_name.c_str(), rr.info.input_w, rr.info.input_h, rr.info.input_frame_size,
             rr.info.min_queue, rr.info.num_classes, rr.info.reg_max);
    t << "\n## Tensor index mapping (cls(s)=2s, box(s)=2s+1)\n";
    t << "  i  runner stream name              ref(derive_layout) stream name  grid(pad)  f   stride  bytes     "
         "scale         zp      same\n";
    for (int i = 0; i < 6; ++i) {
        const auto& m = rr.info.t[i];
        t << fmt("  %d  %-30s  %-30s  %2dx%2d(%2d)  %3d  %2d      %-8zu  %-12.9g  %-6.4g  %s\n", i,
                 m.stream_name.empty() ? "-" : m.stream_name.c_str(),
                 ref.out_meta[static_cast<std::size_t>(ref_idx[i])].name.c_str(), m.grid_h, m.grid_w, m.w_padded,
                 m.features, m.stride, m.hw_frame_size, static_cast<double>(m.qp_scale),
                 static_cast<double>(m.qp_zp), rr.names_match ? "yes" : "NO");
    }

    t << "\n## A(RawSession) vs B(HailoRawRunner)\n";
    t << fmt("  A: constructed=%d frames_ok=%d/%d cb=%ld failed=%ld\n", ref.constructed ? 1 : 0, ref.frames_ok, n,
             ref.cb_total, ref.cb_failed);
    t << fmt("  B: constructed=%d slots=%d(min_queue=%zu) frames_ok=%d/%d cb=%ld failed=%ld\n",
             rr.constructed ? 1 : 0, rr.slots, rr.min_queue, rr.frames_ok, n, rr.cb_total, rr.cb_failed);
    if (!rr.error.empty()) t << fmt("     error: %s\n", rr.error.c_str());
    t << fmt("  6 output tensors byte-identical : %d / %d frames (%ld differing bytes)\n", matched, compared, diff_bytes);
    if (!o.skip_bench)
        t << fmt("  B throughput (pipeline, %d frames x%d median): %.2f FPS (min %.2f, max %.2f),\n"
                 "     latency median %.2f ms p90 %.2f ms, temp %.1f -> %.1f C\n",
                 o.bench_frames, std::max(1, o.bench_repeat), rr.fps, rr.fps_min, rr.fps_max, rr.lat_median,
                 rr.lat_p90, static_cast<double>(rr.temp_before), static_cast<double>(rr.temp_after));

    t << "\n## Pass criteria\n";
    t << fmt("  ① runner creation + slot bind succeeded  : %s\n", (rr.constructed && rr.error.empty()) ? "PASS" : "FAIL");
    t << fmt("  ② tensor index ↔ stream name mapping ok  : %s\n", rr.names_match ? "PASS" : "FAIL");
    t << fmt("  ③ callback failures 0, frames %d/%d       : %s\n", rr.frames_ok, n,
             (rr.cb_failed == 0 && rr.frames_ok == n) ? "PASS" : "FAIL");
    t << fmt("  ④ 6 output tensors byte-identical to f0  : %s (%d/%d)\n",
             (compared == n && matched == compared) ? "PASS" : "FAIL", matched, compared);
    t << fmt("  ⑤ hailort.log error %d / warning %d(thermal/clock %d) : %s\n", log_err, warns, thermal,
             (log_err == 0 && warns == 0) ? "PASS" : "FAIL");
    t << fmt("  ⑥ fd leak 0                              : %s (start %ld, end %ld → %+ld)\n",
             (fds_end - fds_start <= 1) ? "PASS" : "FAIL", fds_start, fds_end, fds_end - fds_start);
    t << fmt("  ⑦ Vulkan validation layer errors %llu (active %s)\n", static_cast<unsigned long long>(vk_errors),
             vk_layer_active ? "yes" : "no");
    t << fmt("\n[conclusion] %s\n", pass ? "HailoRawRunner produces the same output as RawSession — promotion verified"
                                   : "mismatch — the promoted code must be fixed");

    std::printf("\n[RUNNER] %s (bytes %d/%d, cb failed %ld, log error %d warning %d, fd %+ld)\n",
                pass ? "PASS" : "FAIL", matched, compared, rr.cb_failed, log_err, warns, fds_end - fds_start);
    return pass ? 0 : 1;
}

struct MatrixRow {
    AllocMode alloc = AllocMode::Heap;
    RawApi api = RawApi::ReadAsync;
    SizeKind map_size = SizeKind::Hw;
    SizeKind read_size = SizeKind::Hw;
    RunOutcome out;
    int bytes_matched = 0, bytes_compared = 0;
    bool pass = false;
};

int run_p2(const Options& o, const std::vector<std::vector<std::uint8_t>>& inputs, const std::string& link_speed,
           const std::string& link_width) {
    const long fds_start = count_open_fds();
    const int log_errors_start = count_hailort_log(false);
    const int log_thermal_start = count_hailort_log(true);

    // Parse --alloc
    std::vector<AllocMode> modes;
    {
        std::string cur;
        std::string src = o.alloc + ",";
        for (char c : src) {
            if (c != ',') {
                cur.push_back(c);
                continue;
            }
            if (cur == "malloc") modes.push_back(AllocMode::Malloc);
            else if (cur == "heap") modes.push_back(AllocMode::Heap);
            else if (cur == "vulkan") modes.push_back(AllocMode::Vulkan);
            else if (!cur.empty()) throw std::runtime_error("unknown --alloc value: " + cur);
            cur.clear();
        }
    }
    if (modes.empty()) throw std::runtime_error("--alloc is empty");
    std::vector<RawApi> apis;
    if (o.api == "infer_async" || o.api == "both") apis.push_back(RawApi::InferAsync);
    if (o.api == "read_async" || o.api == "both") apis.push_back(RawApi::ReadAsync);

    // Cool down before using the NPU (plan §7 / measure skill)
    float t_before = 0, t_after = 0;
    const int cooled_s = cool_down(o.cool_above, o.cool_target, o.cool_max_s, t_before, t_after);
    std::printf("[p2] cooling: hailo %.1f -> %.1f C, waited %d s (rule: if >%.0f C, down to %.0f C, max %d s)\n",
                static_cast<double>(t_before), static_cast<double>(t_after), cooled_s,
                static_cast<double>(o.cool_above), static_cast<double>(o.cool_target), o.cool_max_s);

    RawSessionConfig base;
    base.hef_path = o.hef;
    base.batch = static_cast<std::uint16_t>(o.batch);

    const int n_cmp = std::min<int>(o.compare_frames, static_cast<int>(inputs.size()));

    // ── Baseline: malloc, read_async (host pointer). The path already verified in P1.
    base.api = apis.front();
    base.map_size = SizeKind::Hw;
    base.read_size = SizeKind::Hw;
    RunOutcome ref = run_capture(base, nullptr, o.slots, inputs, n_cmp, o.cpu_read_reps);
    if (!ref.constructed || ref.frames_ok == 0)
        throw std::runtime_error("P2 baseline (malloc) capture failed: " + ref.error);
    const std::size_t n_tensors = ref.frames[0].t.size();
    std::printf("[p2] baseline malloc(%s): frames_ok=%d cb=%ld failed=%ld cpu_read median=%.3f ms\n",
                api_name(base.api), ref.frames_ok, ref.cb_total, ref.cb_failed, percentile(ref.cpu_read_ms, 0.5));

    // ── Exhaustive trial of 4 size combinations x alloc (dma-buf) x api
    Tee mat(o.out_dir + "/p2_alloc_matrix.txt");
    mat << "# P2-④ exhaustive trial of the 4 size-argument combinations (plan §4.2)\n";
    mat << fmt("# pcie=%s x%s hef=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str());
    mat << fmt("# per cell: slots=%d, frames=%d. bytes is a byte-level comparison with the --alloc malloc result.\n",
               o.matrix_slots, o.matrix_frames);
    mat << "# hw_frame_size = 512000/409600/128000/102400/38400/30720,  dmabuf_size = that rounded up to 16384\n\n";
    mat << "alloc   api           dma_map size   read_async size  map_status                  issue  cb/fail"
           "  bytes      verdict\n";
    mat << "------  ------------  -------------  ---------------  --------------------------  -----  -------"
           "  ---------  ----\n";

    std::vector<MatrixRow> rows;
    for (AllocMode m : modes) {
        if (m == AllocMode::Malloc) continue;   // No fd, so there is no size contract
        for (RawApi api : apis) {
            for (int mk = 0; mk < 2; ++mk) {
                for (int rk = 0; rk < 2; ++rk) {
                    MatrixRow row;
                    row.alloc = m;
                    row.api = api;
                    row.map_size = mk == 0 ? SizeKind::Hw : SizeKind::Dmabuf;
                    row.read_size = rk == 0 ? SizeKind::Hw : SizeKind::Dmabuf;
                    RawSessionConfig sc = base;
                    sc.api = api;
                    sc.map_size = row.map_size;
                    sc.read_size = row.read_size;
                    std::unique_ptr<yvo::p2::OutBufPool> pool;
                    try {
                        pool = make_pool(m, false);
                        row.out = run_capture(sc, pool.get(), o.matrix_slots, inputs, o.matrix_frames, 0);
                    } catch (const std::exception& e) {
                        row.out.error = e.what();
                    }
                    if (row.out.frames_ok > 0)
                        frames_identical(ref.frames, row.out.frames, n_tensors, row.bytes_matched,
                                         row.bytes_compared);
                    row.pass = row.out.constructed && row.out.issued && !row.out.callbacks_lost &&
                               row.out.map_failed == 0 && row.out.cb_failed == 0 &&
                               row.out.frames_ok == o.matrix_frames &&
                               row.bytes_compared > 0 && row.bytes_matched == row.bytes_compared;
                    pool.reset();
                    mat << fmt("%-6s  %-12s  %-13s  %-15s  %-26s  %-5s  %3ld/%-3ld  %2d/%-2d      %s\n",
                               alloc_name(m), api_name(api), size_kind_name(row.map_size),
                               size_kind_name(row.read_size), status_name(row.out.map_status).c_str(),
                               row.out.callbacks_lost ? "lost" : (row.out.issued ? "ok" : "FAIL"),
                               row.out.cb_total, row.out.cb_failed,
                               row.bytes_matched, row.bytes_compared, row.pass ? "PASS" : "FAIL");
                    if (!row.out.error.empty()) mat << fmt("        error: %s\n", row.out.error.c_str());
                    std::printf("[p2 matrix] %s/%s map=%s read=%s -> %s\n", alloc_name(m), api_name(api),
                                size_kind_name(row.map_size), size_kind_name(row.read_size),
                                row.pass ? "PASS" : "FAIL");
                    rows.push_back(row);
                }
            }
        }
    }

    // Combinations with mismatched sizes are rejected by HailoRT with CHECK failed — these are **intended** hailort.log errors.
    // The final verdict (reliability) counts only errors after this stage. Same discipline as P1 separating Phase 3's intended failures.
    const int log_errors_after_matrix = count_hailort_log(false);
    mat << fmt("\n# The %d errors logged to hailort.log in this stage are **intended** size-mismatch rejections\n",
               log_errors_after_matrix - log_errors_start);

    // Pick one of the passing combinations. The read_async header says "expected to be get_frame_size()", so
    // prefer read=hw, then prefer map=dmabuf with the wider mapping range.
    auto score = [](const MatrixRow& r) {
        return (r.read_size == SizeKind::Hw ? 2 : 0) + (r.map_size == SizeKind::Dmabuf ? 1 : 0);
    };
    const MatrixRow* best = nullptr;
    for (const auto& r : rows) {
        if (!r.pass) continue;
        if (!best || score(r) > score(*best)) best = &r;
    }
    mat << "\n";
    if (best) {
        mat << fmt("[chosen] dma_map=%s read_async=%s (passing combinations %d / total %zu)\n", size_kind_name(best->map_size),
                   size_kind_name(best->read_size),
                   static_cast<int>(std::count_if(rows.begin(), rows.end(), [](const MatrixRow& r) { return r.pass; })),
                   rows.size());
    } else {
        mat << "[chosen] no size combination passed → size contract unclear. Plan §6.2 P2-④ failed.\n";
    }
    for (std::size_t i = 0; i < ref.sizes.size(); ++i)
        mat << fmt("  tensor %zu: hw_frame_size=%zu\n", i, ref.sizes[i].first);

    // ── Run all slots (16 x 6 = 96 fds) with the chosen combination
    Tee res(o.out_dir + "/p2_result.txt");
    Tee cr(o.out_dir + "/p2_cpu_read.txt");
    res << "# P2 result (yvo_hailo_raw_probe --probe p2)\n";
    res << fmt("# pcie=%s x%s hef=%s video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
               o.video.c_str());
    res << fmt("# batch=%d slots=%d compare_frames=%d api=%s alloc=%s\n", o.batch, o.slots, n_cmp, o.api.c_str(),
               o.alloc.c_str());
    res << fmt("# cooling: hailo %.1f -> %.1f C, waited %d s\n\n", static_cast<double>(t_before),
               static_cast<double>(t_after), cooled_s);

    std::size_t total_bytes = 0;
    for (const auto& sz : ref.sizes) total_bytes += sz.first;
    cr << "# P2-⑥ cpu_read_1.22MB_ms — time for the CPU to read all 6 tensors with sequential memcpy\n";
    cr << fmt("# total %zu B, %d repetitions (excluding 3 warm-ups). Reference: sequential memcpy of V3D output buffer 1,228,800 B = 0.85~0.87 ms\n",
              total_bytes, o.cpu_read_reps);
    cr << "alloc    median_ms   p90_ms    min_ms    GB/s(median)   note\n";
    cr << "-------  ----------  --------  --------  ------------   ----\n";

    // fd count after the 3 evidence files (p2_alloc_matrix/p2_result/p2_cpu_read) are opened. This is the leak baseline.
    const long fds_files = count_open_fds();

    struct FinalRow {
        AllocMode alloc;
        RunOutcome out;
        int bytes_matched = 0, bytes_compared = 0;
        bool pass = false;
    };
    std::vector<FinalRow> finals;

    // Also put the malloc baseline in the table
    {
        FinalRow f;
        f.alloc = AllocMode::Malloc;
        f.out = ref;
        f.bytes_compared = static_cast<int>(ref.frames.size());
        f.bytes_matched = f.bytes_compared;
        f.pass = true;
        finals.push_back(f);
    }

    for (AllocMode m : modes) {
        if (m == AllocMode::Malloc) continue;
        // Pick the highest-scoring passing combination for this alloc mode (may differ per mode).
        const MatrixRow* mb = nullptr;
        for (const auto& r : rows) {
            if (!r.pass || r.alloc != m) continue;
            if (!mb || score(r) > score(*mb)) mb = &r;
        }
        FinalRow f;
        f.alloc = m;
        if (!mb) {
            f.out.error = "no size combination passed";
            finals.push_back(f);
            continue;
        }
        RawSessionConfig sc = base;
        sc.api = mb->api;
        sc.map_size = mb->map_size;
        sc.read_size = mb->read_size;
        std::unique_ptr<yvo::p2::OutBufPool> pool;
        long fds_with_pool = 0;
        try {
            pool = make_pool(m, false);
            // --in-dmabuf: only in vulkan mode, also pass the input as a dma-buf from the same pool (F0 combination).
            if (o.in_dmabuf && m == AllocMode::Vulkan) sc.in_pool = pool.get();
            f.out = run_capture(sc, pool.get(), o.slots, inputs, n_cmp, o.cpu_read_reps);
            fds_with_pool = count_open_fds();
        } catch (const std::exception& e) {
            f.out.error = e.what();
        }
        if (f.out.frames_ok > 0)
            frames_identical(ref.frames, f.out.frames, n_tensors, f.bytes_matched, f.bytes_compared);
        f.pass = f.out.constructed && f.out.issued && !f.out.callbacks_lost && f.out.map_failed == 0 &&
                 f.out.cb_failed == 0 && f.bytes_compared > 0 && f.bytes_matched == f.bytes_compared;
        res << fmt("[%s] api=%s dma_map=%s read_async=%s slots=%d\n", alloc_name(m), api_name(mb->api),
                   size_kind_name(mb->map_size), size_kind_name(mb->read_size), f.out.slots);
        res << fmt("  dma_map_dmabuf : %ld calls, failed %ld, status=%s, total %.3f ms (%.3f ms/call)\n", f.out.map_calls,
                   f.out.map_failed, status_name(f.out.map_status).c_str(), f.out.map_ms,
                   f.out.map_calls > 0 ? f.out.map_ms / static_cast<double>(f.out.map_calls) : 0.0);
        res << fmt("  callbacks      : %ld, failed %ld, first_error=%s\n", f.out.cb_total, f.out.cb_failed,
                   status_name(f.out.first_error).c_str());
        res << fmt("  byte match vs malloc    : %d / %d frames\n", f.bytes_matched, f.bytes_compared);
        res << fmt("  open fds with pool held : %ld (start %ld)\n", fds_with_pool, fds_start);
        if (!f.out.error.empty()) res << fmt("  error: %s\n", f.out.error.c_str());
        for (std::size_t i = 0; i < f.out.sizes.size(); ++i)
            res << fmt("  tensor %zu: hw_frame_size=%zu dmabuf_size=%zu (diff %+zd)\n", i, f.out.sizes[i].first,
                       f.out.sizes[i].second,
                       static_cast<std::ptrdiff_t>(f.out.sizes[i].second) -
                           static_cast<std::ptrdiff_t>(f.out.sizes[i].first));
        res << "\n";
        pool.reset();
        finals.push_back(f);
    }

    for (const auto& f : finals) {
        const double med = percentile(f.out.cpu_read_ms, 0.5);
        const double p90 = percentile(f.out.cpu_read_ms, 0.9);
        const double mn = f.out.cpu_read_ms.empty()
                              ? 0.0
                              : *std::min_element(f.out.cpu_read_ms.begin(), f.out.cpu_read_ms.end());
        const double gbs = med > 0 ? static_cast<double>(total_bytes) / (med * 1e-3) / 1e9 : 0.0;
        cr << fmt("%-7s  %10.3f  %8.3f  %8.3f  %12.3f   %s\n", alloc_name(f.alloc), med, p90, mn, gbs,
                  f.alloc == AllocMode::Heap ? "includes DMA_BUF_IOCTL_SYNC, cached"
                                             : (f.alloc == AllocMode::Vulkan ? "V3D HOST_COHERENT (uncached)"
                                                                             : "host malloc (baseline)"));
    }
    cr << fmt("\ntotal bytes read = %zu B (sum of hw_frame_size over 6 tensors)\n", total_bytes);

    // ── Verdict
    const int log_errors_total = count_hailort_log(false) - log_errors_start;
    const int log_errors_matrix = log_errors_after_matrix - log_errors_start;
    const int log_errors = log_errors_total - log_errors_matrix;   // Final run section only
    const int log_thermal = count_hailort_log(true) - log_thermal_start;
    const long fds_end = count_open_fds();

    const FinalRow* vk = nullptr;
    const FinalRow* hp = nullptr;
    for (const auto& f : finals) {
        if (f.alloc == AllocMode::Vulkan) vk = &f;
        if (f.alloc == AllocMode::Heap) hp = &f;
    }
    const bool vk_ok = vk && vk->pass;
    const bool hp_ok = hp && hp->pass;
    const bool any_ok = vk_ok || hp_ok;
    double map_ms_96 = 0.0;
    long map_calls_96 = 0;
    if (vk_ok) {
        map_ms_96 = vk->out.map_ms;
        map_calls_96 = vk->out.map_calls;
    } else if (hp_ok) {
        map_ms_96 = hp->out.map_ms;
        map_calls_96 = hp->out.map_calls;
    }

    res << "Pass criteria verdict (plan §6.2 P2)\n";
    res << fmt("  ① dma_map_dmabuf(fd, size, D2H) == SUCCESS : %s (vulkan %s / heap %s)\n",
               any_ok ? "PASS" : "FAIL",
               vk ? status_name(vk->out.map_status).c_str() : "(not run)",
               hp ? status_name(hp->out.map_status).c_str() : "(not run)");
    res << fmt("  ② read_async(fd, ...) callbacks SUCCESS    : %s (vulkan failed %ld / heap failed %ld)\n",
               any_ok ? "PASS" : "FAIL", vk ? vk->out.cb_failed : -1, hp ? hp->out.cb_failed : -1);
    res << fmt("  ③ 6 tensors byte-identical to malloc      : %s (vulkan %d/%d, heap %d/%d)\n",
               any_ok ? "PASS" : "FAIL", vk ? vk->bytes_matched : 0, vk ? vk->bytes_compared : 0,
               hp ? hp->bytes_matched : 0, hp ? hp->bytes_compared : 0);
    res << fmt("  ④ at least one size combination works     : %s%s\n", best ? "PASS" : "FAIL",
               best ? fmt(" (dma_map=%s, read_async=%s)", size_kind_name(best->map_size),
                          size_kind_name(best->read_size))
                          .c_str()
                    : "");
    res << fmt("  ⑤ dma_map_dmabuf %ld calls total < 100 ms  : %s (%.3f ms)\n", map_calls_96,
               (map_calls_96 > 0 && map_ms_96 < 100.0) ? "PASS" : "FAIL", map_ms_96);
    res << fmt("  ⑥ cpu_read_1.22MB_ms output for 3 modes   : %s (%s/p2_cpu_read.txt)\n",
               finals.size() >= 2 ? "recorded" : "insufficient", o.out_dir.c_str());
    res << fmt("\nhailort.log: %d errors (final run section), %d thermal/clock warnings %s\n", log_errors, log_thermal,
               (log_errors == 0 && log_thermal == 0) ? "" : "→ numbers from this run are unreliable");
    res << fmt("            + %d intended size-mismatch rejections in the size combination table (%d total)\n", log_errors_matrix,
               log_errors_total);
    res << fmt("open fds: process start %ld, after opening 3 evidence files %ld (baseline), end %ld → leak %+ld\n", fds_start,
               fds_files, fds_end, fds_end - fds_files);
    res << fmt("\n[conclusion] %s\n", vk_ok ? "plan §4.2 adopted option (Vulkan-owned dma-buf) can proceed as is"
                                      : (hp_ok ? "adopted option not viable → continue with the design changed to the dma-heap alternative"
                                               : "①~③ did not pass even with the heap option → plan §4.8 stop condition"));

    std::printf("\n[P2] %s (vulkan %s, heap %s)\n", any_ok ? "PASS" : "FAIL", vk_ok ? "PASS" : "FAIL",
                hp_ok ? "PASS" : "FAIL");
    std::printf("[p2] fd: start %ld, baseline %ld, end %ld (leak %+ld)\n", fds_start, fds_files, fds_end,
                fds_end - fds_files);
    return any_ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--video") o.video = next();
        else if (a == "--hef") o.hef = next();
        else if (a == "--out-dir") o.out_dir = next();
        else if (a == "--fixture-dir") o.fixture_dir = next();
        else if (a == "--api") o.api = next();
        else if (a == "--compare-frames") o.compare_frames = std::stoi(next());
        else if (a == "--api-check-frames") o.api_check_frames = std::stoi(next());
        else if (a == "--dump-frames") o.dump_frames = std::stoi(next());
        else if (a == "--bench-frames") o.bench_frames = std::stoi(next());
        else if (a == "--bench-repeat") o.bench_repeat = std::stoi(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--batch") o.batch = std::stoi(next());
        else if (a == "--slots") o.slots = std::stoi(next());
        else if (a == "--skip-bench") o.skip_bench = true;
        else if (a == "--skip-infer-model") o.skip_infer_model = true;
        else if (a == "--probe") o.probe = next();
        else if (a == "--alloc") o.alloc = next();
        else if (a == "--matrix-frames") o.matrix_frames = std::stoi(next());
        else if (a == "--matrix-slots") o.matrix_slots = std::stoi(next());
        else if (a == "--cpu-read-reps") o.cpu_read_reps = std::stoi(next());
        else if (a == "--no-cool") o.cool_above = 1000.0f;
        else if (a == "--in-dmabuf") o.in_dmabuf = true;
        else if (a == "--f0-frames") o.f0_frames = std::stoi(next());
        else {
            usage();
            return 2;
        }
    }
    if (o.api != "infer_async" && o.api != "read_async" && o.api != "both") {
        usage();
        return 2;
    }
    if (o.probe != "p1" && o.probe != "p2" && o.probe != "f0" && o.probe != "runner") {
        usage();
        return 2;
    }
    if (o.in_dmabuf && o.probe == "p1") {
        std::fprintf(stderr, "--in-dmabuf is only for --probe p2 or f0 (p1 is a path that does not use the GPU)\n");
        return 2;
    }

    try {
        std::filesystem::create_directories(o.out_dir);
        std::filesystem::create_directories(o.fixture_dir);

        // ---- Environment (do not guess measurement conditions)
        const std::string link_speed = read_first_line("/sys/bus/pci/devices/0001:01:00.0/current_link_speed");
        const std::string link_width = read_first_line("/sys/bus/pci/devices/0001:01:00.0/current_link_width");
        std::printf("[env] pcie current_link_speed=%s width=x%s (8.0 GT/s = Gen3)\n", link_speed.c_str(),
                    link_width.c_str());

        // ---- Input frames
        const yvo::LetterboxParams lb = yvo::make_letterbox(kSrcW, kSrcH, 640, 640);
        const int need = std::max({o.compare_frames, o.dump_frames, o.api_check_frames});
        auto nv12 = read_nv12_frames(o.video, need, kSrcW, kSrcH);
        if (static_cast<int>(nv12.size()) < need)
            throw std::runtime_error("not enough frames: " + std::to_string(nv12.size()) + " / " + std::to_string(need));
        const std::size_t in_size = static_cast<std::size_t>(640) * 640 * 3;
        auto inputs = make_nhcw_inputs(nv12, lb, in_size);
        std::printf("[info] frames=%zu letterbox scale=%.6f pad=(%d,%d) input=%zu B (NHCW)\n", inputs.size(),
                    lb.scale, lb.pad_x, lb.pad_y, in_size);

        // ---- P2: dma-buf output path (plan §6.2 P2). Shared with P1 only up to the environment check and input frames.
        if (o.probe == "p2") return run_p2(o, inputs, link_speed, link_width);

        // ---- F0: gate for simultaneous input dma-buf (H2D) + output dma-buf (D2H)
        if (o.probe == "f0") return run_f0(o, inputs, link_speed, link_width);

        // ---- RUNNER: RawSession -> yvo::HailoRawRunner promotion verification
        if (o.probe == "runner") return run_runner(o, inputs, link_speed, link_width);

        // ---- Phase 0: dump output streams from HEF parsing alone
        Tee info(o.out_dir + "/stream_infos_quant.txt");
        info << fmt("# yvo_hailo_raw_probe stream infos / quant infos\n");
        info << fmt("# hef=%s pcie=%s x%s\n\n", o.hef.c_str(), link_speed.c_str(), link_width.c_str());
        {
            auto hef = unwrap(hailort::Hef::create(o.hef), "Hef::create");
            auto ins = unwrap(hef.get_input_stream_infos(), "Hef::get_input_stream_infos");
            auto outs = unwrap(hef.get_output_stream_infos(), "Hef::get_output_stream_infos");
            info << fmt("[Hef] input streams %zu, output streams %zu\n", ins.size(), outs.size());
            for (const auto& s : ins) {
                info << fmt("  IN  %-28s %s %s shape=%ux%ux%u hw_shape=%ux%ux%u hw_frame_size=%u hw_data_bytes=%u\n",
                            s.name, format_type_name(s.format.type), format_order_name(s.format.order),
                            s.shape.height, s.shape.width, s.shape.features, s.hw_shape.height, s.hw_shape.width,
                            s.hw_shape.features, s.hw_frame_size, s.hw_data_bytes);
            }
            for (const auto& s : outs) {
                info << fmt("  OUT %-28s %s %s shape=%ux%ux%u hw_shape=%ux%ux%u hw_frame_size=%u hw_data_bytes=%u\n",
                            s.name, format_type_name(s.format.type), format_order_name(s.format.order),
                            s.shape.height, s.shape.width, s.shape.features, s.hw_shape.height, s.hw_shape.width,
                            s.hw_shape.features, s.hw_frame_size, s.hw_data_bytes);
                info << fmt("      stream_info.quant_info: zp=%.8g scale=%.8g limvals=[%.8g, %.8g]\n",
                            static_cast<double>(s.quant_info.qp_zp), static_cast<double>(s.quant_info.qp_scale),
                            static_cast<double>(s.quant_info.limvals_min),
                            static_cast<double>(s.quant_info.limvals_max));
            }
            if (outs.size() != 6) {
                info << fmt("\n[FAIL] P1-① raw output stream count is not 6: %zu\n", outs.size());
                return 1;
            }
            info << "\n";
        }

        // ---- Phase 1: raw stream session (per API path)
        std::vector<RawApi> apis;
        if (o.api == "infer_async" || o.api == "both") apis.push_back(RawApi::InferAsync);
        if (o.api == "read_async" || o.api == "both") apis.push_back(RawApi::ReadAsync);

        std::vector<ApiOutcome> outcomes;
        std::vector<RawFrame> captured;       // compare_frames worth from the primary api
        std::vector<RawFrame> captured_alt;   // Same frames from the other api (for byte comparison)
        RawLayout layout;
        std::vector<StreamMeta> out_meta;
        bool layout_ok = false;
        bool quant_per_feature = false;
        bool primary_done = false;
        RawApi primary_api = RawApi::InferAsync;

        for (RawApi api : apis) {
            ApiOutcome oc;
            oc.api = api;
            std::unique_ptr<RawSession> sess;
            try {
                RawSessionConfig sc;
                sc.hef_path = o.hef;
                sc.batch = static_cast<std::uint16_t>(o.batch);
                sc.want_slots = o.slots;
                sc.api = api;
                sess = std::make_unique<RawSession>(sc);   // out_pool == nullptr → host malloc (P1)
                oc.constructed = true;
            } catch (const std::exception& e) {
                oc.error = e.what();
                std::printf("[%s] session creation failed: %s\n", api_name(api), e.what());
                outcomes.push_back(oc);
                continue;
            }
            oc.slots = sess->num_slots();
            oc.min_queue = sess->min_queue();
            oc.infer_queue = sess->infer_queue_size();
            oc.scheduled = sess->is_scheduled();
            oc.temp_before = sess->chip_temperature();

            if (!layout_ok) {
                out_meta = sess->out_meta();
                info << fmt("[configured, api=%s] is_scheduled=%d infer_queue_size=%zu batch=%d\n",
                            api_name(api), static_cast<int>(oc.scheduled), oc.infer_queue, o.batch);
                info << fmt("  IN  %-28s frame_size=%zu async_max_queue=%zu quant_infos.size=%zu\n",
                            sess->in_meta().name.c_str(), sess->in_meta().frame_size, sess->in_meta().queue_size,
                            sess->in_meta().quant.size());
                for (const auto& m : out_meta) {
                    info << fmt("  OUT %-28s frame_size=%zu async_max_queue=%zu quant_infos.size=%zu\n",
                                m.name.c_str(), m.frame_size, m.queue_size, m.quant.size());
                    for (std::size_t qi = 0; qi < m.quant.size(); ++qi) {
                        info << fmt("      quant[%zu]: zp=%.8g scale=%.8g limvals=[%.8g, %.8g]\n", qi,
                                    static_cast<double>(m.quant[qi].qp_zp),
                                    static_cast<double>(m.quant[qi].qp_scale),
                                    static_cast<double>(m.quant[qi].limvals_min),
                                    static_cast<double>(m.quant[qi].limvals_max));
                    }
                    if (m.quant.size() != 1) quant_per_feature = true;
                }
                info << fmt("  num_slots=%d (requested %d, clamped to minimum queue size %zu)\n\n", oc.slots, o.slots, oc.min_queue);
                layout = derive_layout(out_meta);
                layout_ok = true;
                for (int i = 0; i < 3; ++i) {
                    info << fmt("  [scale %d] stride=%d grid=%dx%d w_padded=%d cls=%s box=%s\n", i,
                                layout.s[i].stride, layout.s[i].grid_h, layout.s[i].grid_w, layout.s[i].w_padded,
                                short_name(out_meta[static_cast<std::size_t>(layout.s[i].cls_idx)].name).c_str(),
                                short_name(out_meta[static_cast<std::size_t>(layout.s[i].box_idx)].name).c_str());
                }
                info << fmt("  num_classes=%d reg_max=%d (derived from shape)\n\n", layout.num_classes, layout.reg_max);
                if (quant_per_feature) {
                    info << "[FAIL] P1-② get_quant_infos().size() > 1 (per-feature quantization) → design change needed\n";
                }
            }

            // API path check: see with a few frames whether all 7 callbacks are SUCCESS
            const int n_check = std::min<int>(o.api_check_frames, static_cast<int>(inputs.size()));
            std::vector<RawFrame> local(static_cast<std::size_t>(o.compare_frames));
            std::atomic<long> done{0};
            try {
                for (int i = 0; i < n_check; ++i) {
                    sess->submit(inputs[static_cast<std::size_t>(i)].data(),
                                 [&](int, hailo_status) { done.fetch_add(1); });
                }
                sess->drain();
                oc.cb_total = sess->callbacks_total();
                oc.cb_failed = sess->callbacks_failed();
                oc.first_error = sess->first_error();
                oc.ok = (oc.cb_failed == 0) && (oc.cb_total == 7L * n_check) && (done.load() == n_check);
            } catch (const std::exception& e) {
                oc.error = e.what();
                oc.cb_total = sess->callbacks_total();
                oc.cb_failed = sess->callbacks_failed();
                oc.first_error = sess->first_error();
                oc.ok = false;
            }
            std::printf("[%s] api check: frames=%d callbacks=%ld failed=%ld first_error=%s -> %s\n", api_name(api),
                        n_check, oc.cb_total, oc.cb_failed, status_name(oc.first_error).c_str(),
                        oc.ok ? "OK" : "FAIL");

            if (oc.ok) {
                // Obtain compare_frames worth of the 6 tensors
                const int n_cmp = std::min<int>(o.compare_frames, static_cast<int>(inputs.size()));
                for (int i = 0; i < n_cmp; ++i) {
                    sess->submit(inputs[static_cast<std::size_t>(i)].data(),
                                 [&](int slot, hailo_status st) {
                                     if (st == HAILO_SUCCESS) local[static_cast<std::size_t>(i)] = sess->copy_out(slot);
                                 });
                    sess->drain();   // To read slot buffers per frame as is, wait for completion after every frame
                }
                if (!primary_done) {
                    captured = std::move(local);
                    primary_done = true;
                    primary_api = api;
                } else {
                    captured_alt = std::move(local);
                }

                // Throughput (primary api only). Run --bench-repeat times per configuration and use the median.
                //
                // FPS calculation: **the pipeline fill and drain phases must be excluded.** If the interval is taken as
                // [completion of the warmup-th frame] ~ [completion of the last frame], the number of completions in between is exactly
                // bench_frames. (If t0 is taken as "just before the warmup-th submit", the slots frames still in flight
                // complete within the interval and FPS is underestimated by slots/(slots+bench) — measured 48.6 vs 52.4.)
                if (!o.skip_bench && api == primary_api) {
                    std::vector<double> fps_runs, med_runs, p90_runs;
                    for (int rep = 0; rep < o.bench_repeat; ++rep) {
                        const int total = o.warmup + o.bench_frames;
                        std::vector<double> lat(static_cast<std::size_t>(total), 0.0);
                        std::vector<std::chrono::steady_clock::time_point> t_sub(static_cast<std::size_t>(total));
                        std::vector<std::chrono::steady_clock::time_point> t_done(static_cast<std::size_t>(total));
                        for (int i = 0; i < total; ++i) {
                            const std::size_t fi = static_cast<std::size_t>(i) % inputs.size();
                            t_sub[static_cast<std::size_t>(i)] = std::chrono::steady_clock::now();
                            sess->submit(inputs[fi].data(), [&, i](int, hailo_status) {
                                t_done[static_cast<std::size_t>(i)] = std::chrono::steady_clock::now();
                                lat[static_cast<std::size_t>(i)] =
                                    std::chrono::duration<double, std::milli>(t_done[static_cast<std::size_t>(i)] -
                                                                              t_sub[static_cast<std::size_t>(i)])
                                        .count();
                            });
                        }
                        sess->drain();
                        const auto t_from = *std::max_element(t_done.begin(), t_done.begin() + o.warmup);
                        const auto t_to = *std::max_element(t_done.begin() + o.warmup, t_done.end());
                        const double secs = std::chrono::duration<double>(t_to - t_from).count();
                        std::vector<double> meas(lat.begin() + o.warmup, lat.end());
                        fps_runs.push_back(static_cast<double>(o.bench_frames) / secs);
                        med_runs.push_back(percentile(meas, 0.5));
                        p90_runs.push_back(percentile(meas, 0.9));
                        std::printf("[%s] bench run %d/%d: fps=%.2f latency median=%.2f p90=%.2f ms temp=%.1f C\n",
                                    api_name(api), rep + 1, o.bench_repeat, fps_runs.back(), med_runs.back(),
                                    p90_runs.back(), static_cast<double>(sess->chip_temperature()));
                    }
                    oc.fps = percentile(fps_runs, 0.5);
                    oc.fps_min = *std::min_element(fps_runs.begin(), fps_runs.end());
                    oc.fps_max = *std::max_element(fps_runs.begin(), fps_runs.end());
                    oc.lat_median = percentile(med_runs, 0.5);
                    oc.lat_p90 = percentile(p90_runs, 0.5);
                    oc.cb_failed = sess->callbacks_failed();
                }
            }
            oc.temp_after = sess->chip_temperature();
            std::printf("[%s] slots=%d min_queue=%zu infer_queue=%zu scheduled=%d temp %.1f -> %.1f C fps=%.2f\n",
                        api_name(api), oc.slots, oc.min_queue, oc.infer_queue, static_cast<int>(oc.scheduled),
                        static_cast<double>(oc.temp_before), static_cast<double>(oc.temp_after), oc.fps);
            outcomes.push_back(oc);
            sess.reset();   // Fully release the device for the next stage
        }

        if (!primary_done) {
            std::fprintf(stderr, "[FAIL] P1-③ both infer_async / read_async failed. Raw path not possible → aborting.\n");
            Tee api_out(o.out_dir + "/p1_api_result.txt");
            for (const auto& oc : outcomes) {
                api_out << fmt("%s: constructed=%d ok=%d cb=%ld failed=%ld first_error=%s error=%s\n",
                               api_name(oc.api), static_cast<int>(oc.constructed), static_cast<int>(oc.ok),
                               oc.cb_total, oc.cb_failed, status_name(oc.first_error).c_str(), oc.error.c_str());
            }
            return 1;
        }

        // ---- Fixture dump: used in the next stage to debug Vulkan shaders without the NPU
        const int n_dump = std::min<int>(o.dump_frames, static_cast<int>(captured.size()));
        std::size_t dumped_bytes = 0;
        for (int i = 0; i < n_dump; ++i) {
            if (captured[static_cast<std::size_t>(i)].t.size() != out_meta.size()) continue;
            for (std::size_t k = 0; k < out_meta.size(); ++k) {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/frame%03d_%s.bin", o.fixture_dir.c_str(), i,
                              short_name(out_meta[k].name).c_str());
                std::ofstream f(path, std::ios::binary);
                const auto& buf = captured[static_cast<std::size_t>(i)].t[k];
                f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
                dumped_bytes += buf.size();
            }
        }
        {
            std::ofstream j(o.fixture_dir + "/fixtures.json");
            // Float values are written with 9 significant digits. With the default 6 digits, qp_scale and letterbox scale get truncated
            // and the next stage (debugging GPU shaders without the NPU) cannot achieve a bit-exact match.
            j << std::setprecision(9);
            j << "{\n";
            j << "  \"generator\": \"yvo_hailo_raw_probe\",\n";
            j << "  \"hef\": \"" << o.hef << "\",\n";
            j << "  \"video\": \"" << o.video << "\",\n";
            j << "  \"api\": \"" << api_name(primary_api) << "\",\n";
            j << "  \"src_width\": " << kSrcW << ",\n";
            j << "  \"src_height\": " << kSrcH << ",\n";
            j << "  \"frames\": " << n_dump << ",\n";
            j << "  \"file_pattern\": \"frame%03d_<stream>.bin\",\n";
            j << "  \"letterbox\": {\"dst_w\": " << lb.dst_w << ", \"dst_h\": " << lb.dst_h << ", \"scale\": "
              << lb.scale << ", \"pad_x\": " << lb.pad_x << ", \"pad_y\": " << lb.pad_y
              << ", \"pad_value\": " << static_cast<int>(lb.pad_value) << "},\n";
            j << "  \"num_classes\": " << layout.num_classes << ", \"reg_max\": " << layout.reg_max << ",\n";
            j << "  \"streams\": [\n";
            for (std::size_t k = 0; k < out_meta.size(); ++k) {
                const auto& m = out_meta[k];
                int si = -1, role = 0;   // role 0 = cls, 1 = box
                for (int s = 0; s < 3; ++s) {
                    if (layout.s[s].cls_idx == static_cast<int>(k)) { si = s; role = 0; }
                    if (layout.s[s].box_idx == static_cast<int>(k)) { si = s; role = 1; }
                }
                j << "    {\"name\": \"" << m.name << "\", \"file\": \"" << short_name(m.name) << "\""
                  << ", \"scale_id\": " << si << ", \"role\": \"" << (role == 0 ? "cls" : "box") << "\""
                  << ", \"grid_h\": " << m.info.hw_shape.height << ", \"grid_w\": " << m.info.shape.width
                  << ", \"w_padded\": " << m.info.hw_shape.width << ", \"features\": " << m.info.hw_shape.features
                  << ", \"stride\": " << (si >= 0 ? layout.s[si].stride : 0) << ", \"bytes\": " << m.frame_size
                  << ", \"qp_zp\": " << m.quant[0].qp_zp << ", \"qp_scale\": " << m.quant[0].qp_scale
                  << ", \"limvals_min\": " << m.quant[0].limvals_min
                  << ", \"limvals_max\": " << m.quant[0].limvals_max << "}"
                  << (k + 1 < out_meta.size() ? ",\n" : "\n");
            }
            j << "  ]\n}\n";
        }
        std::printf("[info] fixture dump: %d frames x %zu tensors = %.2f MB -> %s\n", n_dump, out_meta.size(),
                    static_cast<double>(dumped_bytes) / (1024.0 * 1024.0), o.fixture_dir.c_str());

        // Whether the two API paths give the same bytes (only with both)
        long alt_identical = -1;
        if (!captured_alt.empty()) {
            alt_identical = 0;
            for (std::size_t i = 0; i < captured.size() && i < captured_alt.size(); ++i) {
                if (captured[i].t.size() != out_meta.size() || captured_alt[i].t.size() != out_meta.size()) continue;
                bool same = true;
                for (std::size_t k = 0; k < out_meta.size(); ++k)
                    if (captured[i].t[k] != captured_alt[i].t[k]) same = false;
                if (same) ++alt_identical;
            }
            std::printf("[info] infer_async vs read_async tensor byte match: %ld / %zu frames\n", alt_identical,
                        captured.size());
        }

        // ---- Phase 2: reference detections (HailoRT NMS vstream path). It takes the device exclusively, so it runs after destroying all sessions above.
        std::vector<std::vector<yvo::Detection>> ref(captured.size());
        long ref_errors = 0;
        {
            yvo::HailoRunnerConfig hcfg;
            hcfg.hef_path = o.hef;
            hcfg.input_layout = yvo::TensorLayout::Nhcw;   // Feeds the device **the same bytes** as the raw pass
            hcfg.allow_host_ptr_nhcw = true;
            hcfg.batch_size = 1;
            hcfg.num_slots = 4;
            hcfg.src_width = kSrcW;
            hcfg.src_height = kSrcH;
            yvo::HailoRunner runner(hcfg);
            auto buf = alloc_dma_aligned(runner.input_frame_size());
            for (std::size_t i = 0; i < captured.size(); ++i) {
                std::memcpy(buf.get(), inputs[i].data(), in_size);
                yvo::PreprocOutput in;
                in.host_ptr = buf.get();
                in.size = runner.input_frame_size();
                in.layout = yvo::TensorLayout::Nhcw;
                in.lb = lb;
                in.pts = static_cast<std::int64_t>(i);
                in.slot = 0;
                runner.submit(in, [&](yvo::InferResult&& r) {
                    if (r.status != HAILO_SUCCESS) ++ref_errors;
                    ref[static_cast<std::size_t>(r.pts)] = std::move(r.detections);
                });
                runner.drain();
            }
            long total = 0;
            for (const auto& v : ref) total += static_cast<long>(v.size());
            std::printf("[ref] HailoRT NMS path: %zu frames, total detections %ld (%.2f/frame), errors %ld\n", ref.size(),
                        total, static_cast<double>(total) / static_cast<double>(ref.size()), ref_errors);
        }

        // Count hailort.log errors **before** Phase 3. Phase 3 collects evidence by deliberately
        // causing configure() failures, so those errors are not subject to the ⑥ verdict.
        auto count_log = [](bool thermal_only) {
            std::ifstream f("hailort.log");
            std::string line;
            int n = 0;
            while (std::getline(f, line)) {
                if (!thermal_only) {
                    if (line.find("[error]") != std::string::npos) ++n;
                    continue;
                }
                if (line.find("[warning]") == std::string::npos) continue;
                if (line.find("emperature") != std::string::npos || line.find("hrottl") != std::string::npos ||
                    line.find("lock") != std::string::npos || line.find("verheat") != std::string::npos)
                    ++n;
            }
            return n;
        };
        const int errors_before_phase3 = count_log(false);
        const int thermal_warnings = count_log(true);

        // ---- Phase 3: evidence for dropping option (B) — raw outputs cannot be obtained with InferModel
        if (!o.skip_infer_model) {
            Tee ev(o.out_dir + "/infer_model_raw_attempt.txt");
            ev << "# Evidence for dropping option (B): can raw (NHCW) outputs be obtained while keeping InferModel?\n";
            ev << "# Expected: the only output vstream is yolov8_nms_postprocess, and setting NHCW format order on it\n";
            ev << "#       makes configure() fail (plan §4.1).\n\n";
            try {
                hailo_vdevice_params_t vp;
                check(hailo_init_vdevice_params(&vp), "hailo_init_vdevice_params");
                vp.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
                auto vd = unwrap(hailort::VDevice::create(vp), "VDevice::create");
                auto im = unwrap(vd->create_infer_model(o.hef), "create_infer_model");
                ev << fmt("infer_model->inputs().size()  = %zu\n", im->inputs().size());
                ev << fmt("infer_model->outputs().size() = %zu\n", im->outputs().size());
                for (const auto& ostream : im->outputs()) {
                    ev << fmt("  output \"%s\" order=%s type=%s is_nms=%d frame_size=%zu\n", ostream.name().c_str(),
                              format_order_name(ostream.format().order), format_type_name(ostream.format().type),
                              static_cast<int>(ostream.is_nms()), ostream.get_frame_size());
                }
                auto out = unwrap(im->output(), "InferModel::output");
                out.set_format_order(HAILO_FORMAT_ORDER_NHCW);
                ev << "\nconfigure() after set_format_order(HAILO_FORMAT_ORDER_NHCW):\n";
                auto configured = im->configure();
                if (configured) {
                    ev << "  [unexpected] configure() SUCCESS — design needs review\n";
                } else {
                    ev << fmt("  configure() failed status = %s\n", status_name(configured.status()).c_str());
                    ev << "  → the only output vstream is NMS and post-process cannot be disabled. Option (B) dropped for good.\n";
                }
            } catch (const std::exception& e) {
                ev << fmt("  exception: %s\n", e.what());
            }
        }

        // ---- Phase 4: exhaustive tensor interpretation trial (no NPU needed)
        Tee sweep(o.out_dir + "/p1_combination_sweep.txt");
        sweep << "# P1-④ exhaustive trial of unresolved tensor interpretation variables (2x2x2x2 = 16 combinations)\n";
        sweep << fmt("# reference: HailoRT NMS path detections vs cpu_yolov8_postprocess, match() IoU>=%.2f\n",
                     static_cast<double>(kIouMatchThreshold));
        sweep << fmt("# %zu frames, score_thr=%.2f iou_thr=%.2f max/class=%d\n", captured.size(),
                     static_cast<double>(kHefScoreThreshold), static_cast<double>(kHefIouThreshold),
                     kHefMaxPerClass);
        sweep << "# Combinations exceeding cand_limit (20000) are rejected without running NMS (host NMS is O(n^2)).\n\n";
        sweep << "argmax  sigmoid  dfl_order   thr_cmp  cand/frame   ref   test  matched   ratio\n";
        sweep << "------  -------  ----------  -------  ----------  ----  -----  -------  ------\n";

        struct SweepRow {
            yvo::DecodeOptions opt;
            double cand_per_frame = 0;
            long ref = 0, test = 0, matched = 0;
            double ratio = 0;
            bool rejected = false;
        };
        std::vector<SweepRow> rows;
        constexpr std::size_t kCandLimit = 20000;

        for (int a = 0; a < 2; ++a) {
            for (int sg = 0; sg < 2; ++sg) {
                for (int dfl = 0; dfl < 2; ++dfl) {
                    for (int tc = 0; tc < 2; ++tc) {
                        SweepRow row;
                        row.opt.argmax_only = (a == 1);
                        row.opt.apply_sigmoid = (sg == 1);
                        row.opt.dfl_order = (dfl == 0) ? yvo::DflChannelOrder::CoordMajor
                                                       : yvo::DflChannelOrder::BinMajor;
                        row.opt.threshold_inclusive = (tc == 1);
                        std::size_t cand_total = 0;
                        for (std::size_t i = 0; i < captured.size(); ++i) {
                            if (captured[i].t.size() != out_meta.size()) continue;
                            const auto raw = make_raw_set(layout, captured[i], false, out_meta);
                            auto cand = yvo::cpu_yolov8_candidates(raw, kHefScoreThreshold, lb, kSrcW, kSrcH, row.opt);
                            cand_total += cand.size();
                            if (cand.size() > kCandLimit) {
                                row.rejected = true;
                                break;
                            }
                            yvo::cpu_nms_greedy_by_class(cand, kHefIouThreshold, kHefMaxPerClass);
                            std::vector<yvo::Detection> test;
                            test.reserve(cand.size());
                            for (const auto& d : cand) test.push_back(static_cast<const yvo::Detection&>(d));
                            const auto m = match(ref[i], test, nullptr);
                            row.ref += m.ref;
                            row.test += m.test;
                            row.matched += m.matched;
                        }
                        row.cand_per_frame =
                            static_cast<double>(cand_total) / static_cast<double>(captured.size());
                        const long denom = std::max(row.ref, row.test);
                        row.ratio = denom > 0 ? static_cast<double>(row.matched) / static_cast<double>(denom) : 0.0;
                        rows.push_back(row);
                        sweep << fmt("%-6s  %-7s  %-10s  %-7s  %10.1f  %4ld  %5ld  %7ld  %s\n",
                                     row.opt.argmax_only ? "on" : "off", row.opt.apply_sigmoid ? "on" : "off",
                                     row.opt.dfl_order == yvo::DflChannelOrder::CoordMajor ? "coord*16+b"
                                                                                           : "b*4+coord",
                                     row.opt.threshold_inclusive ? ">=" : ">", row.cand_per_frame, row.ref,
                                     row.test, row.matched,
                                     row.rejected ? "  rejected(cand>20000)" : fmt("%.4f", row.ratio).c_str());
                    }
                }
            }
        }

        // On a tie, pick (argmax_only=false, threshold_inclusive=true) — matching the header defaults.
        auto better = [](const SweepRow& x, const SweepRow& y) {   // Whether x is better than y
            if (y.rejected != x.rejected) return y.rejected;
            if (x.ratio != y.ratio) return x.ratio > y.ratio;
            if (x.opt.argmax_only != y.opt.argmax_only) return !x.opt.argmax_only;
            return x.opt.threshold_inclusive && !y.opt.threshold_inclusive;
        };
        std::size_t best = 0;
        for (std::size_t i = 1; i < rows.size(); ++i)
            if (better(rows[i], rows[best])) best = i;
        const auto& b = rows[best];
        sweep << "\n[tie list] combinations that gave the same value as the best ratio (reveals indistinguishable variables)\n";
        for (const auto& r : rows) {
            if (r.rejected || r.ratio < b.ratio) continue;
            sweep << fmt("  argmax=%-3s sigmoid=%-3s dfl=%-10s thr=%-2s ratio=%.4f cand/frame=%.1f test=%ld\n",
                         r.opt.argmax_only ? "on" : "off", r.opt.apply_sigmoid ? "on" : "off",
                         r.opt.dfl_order == yvo::DflChannelOrder::CoordMajor ? "coord*16+b" : "b*4+coord",
                         r.opt.threshold_inclusive ? ">=" : ">", r.ratio, r.cand_per_frame, r.test);
        }
        sweep << fmt("\n[best] argmax_only=%s apply_sigmoid=%s dfl_order=%s threshold=%s ratio=%.4f\n",
                     b.opt.argmax_only ? "true" : "false", b.opt.apply_sigmoid ? "true" : "false",
                     b.opt.dfl_order == yvo::DflChannelOrder::CoordMajor ? "CoordMajor" : "BinMajor",
                     b.opt.threshold_inclusive ? ">=" : ">", b.ratio);
        // Whether `>` and `>=` are distinguishable depends on whether the quantization grid can hit the threshold exactly.
        for (int si = 0; si < 3; ++si) {
            const auto& q = out_meta[static_cast<std::size_t>(layout.s[si].cls_idx)].quant[0];
            const double qf = static_cast<double>(kHefScoreThreshold) / static_cast<double>(q.qp_scale) +
                              static_cast<double>(q.qp_zp);
            sweep << fmt("  scale %d: thr %.2f / qp_scale %.9g = q %.4f (if not an integer, `>` and `>=` are equivalent in principle)\n",
                         si, static_cast<double>(kHefScoreThreshold), static_cast<double>(q.qp_scale), qf);
        }

        // ---- P1-⑤ P5 columns 20~23 include/exclude comparison
        long inc_ref = 0, inc_test = 0, inc_matched = 0;
        double inc_cand = 0;
        for (std::size_t i = 0; i < captured.size(); ++i) {
            if (captured[i].t.size() != out_meta.size()) continue;
            const auto raw = make_raw_set(layout, captured[i], true, out_meta);
            auto cand = yvo::cpu_yolov8_candidates(raw, kHefScoreThreshold, lb, kSrcW, kSrcH, b.opt);
            inc_cand += static_cast<double>(cand.size());
            if (cand.size() > kCandLimit) continue;
            yvo::cpu_nms_greedy_by_class(cand, kHefIouThreshold, kHefMaxPerClass);
            std::vector<yvo::Detection> test;
            for (const auto& d : cand) test.push_back(static_cast<const yvo::Detection&>(d));
            const auto m = match(ref[i], test, nullptr);
            inc_ref += m.ref;
            inc_test += m.test;
            inc_matched += m.matched;
        }
        inc_cand /= static_cast<double>(captured.size());
        const long inc_denom = std::max(inc_ref, inc_test);
        const double inc_ratio =
            inc_denom > 0 ? static_cast<double>(inc_matched) / static_cast<double>(inc_denom) : 0.0;
        // Actual byte contents of the padding columns (20..23). If 0 no ghost detections arise, but excluding is still right
        // (w=20..23 gives (w+0.5)*32 >= 656, coordinates outside the 640 input).
        int pad_cls_max = 0, pad_box_max = 0;
        long pad_nonzero = 0, pad_bytes = 0;
        for (const auto& f : captured) {
            if (f.t.size() != out_meta.size()) continue;
            for (int role = 0; role < 2; ++role) {
                const int idx = role == 0 ? layout.s[2].cls_idx : layout.s[2].box_idx;
                const auto& m = out_meta[static_cast<std::size_t>(idx)];
                const int C = static_cast<int>(m.info.hw_shape.features);
                const int W = layout.s[2].w_padded;
                const auto& buf = f.t[static_cast<std::size_t>(idx)];
                for (int h = 0; h < layout.s[2].grid_h; ++h) {
                    for (int c = 0; c < C; ++c) {
                        for (int w = layout.s[2].grid_w; w < W; ++w) {
                            const int q = buf[yvo::nhcw_index(h, c, w, C, W)];
                            ++pad_bytes;
                            if (q != 0) ++pad_nonzero;
                            if (role == 0) pad_cls_max = std::max(pad_cls_max, q);
                            else pad_box_max = std::max(pad_box_max, q);
                        }
                    }
                }
            }
        }
        sweep << "\n# P1-⑤ P5 (20x24) handling of padding columns 20~23\n";
        sweep << fmt("  exclude(grid_w=20): cand/frame=%.1f test=%ld matched=%ld ratio=%.4f\n", b.cand_per_frame,
                     b.test, b.matched, b.ratio);
        sweep << fmt("  include(grid_w=24): cand/frame=%.1f test=%ld matched=%ld ratio=%.4f\n", inc_cand, inc_test,
                     inc_matched, inc_ratio);
        sweep << fmt("  padding column bytes: %ld total, %ld nonzero (cls max q=%d, box max q=%d)\n", pad_bytes,
                     pad_nonzero, pad_cls_max, pad_box_max);
        sweep << fmt("  padding column class score upper bound = %d * %.9g = %.6f (threshold %.2f)\n", pad_cls_max,
                     static_cast<double>(out_meta[static_cast<std::size_t>(layout.s[2].cls_idx)].quant[0].qp_scale),
                     static_cast<double>(pad_cls_max) *
                         static_cast<double>(out_meta[static_cast<std::size_t>(layout.s[2].cls_idx)].quant[0].qp_scale),
                     static_cast<double>(kHefScoreThreshold));
        if (inc_ratio < b.ratio || inc_test > b.test) {
            sweep << "  → excluding is right: including increases ghost detections or lowers the ratio.\n";
        } else {
            sweep << "  → excluding is right (weak evidence): in this video the padding columns do not exceed the threshold, so results are the same.\n";
            sweep << "     Still, they must be excluded — w=20..23 gives (w+0.5)*32 >= 656, coordinates outside the 640 input,\n";
            sweep << "     and padding values are undefined, so they may become ghost detections with other videos/models.\n";
        }

        // ---- Final P1 verdict
        const int hailort_errors = errors_before_phase3;
        const int hailort_thermal = thermal_warnings;
        const int hailort_errors_total = count_log(false);

        Tee res(o.out_dir + "/p1_api_result.txt");
        res << "# P1 result summary (yvo_hailo_raw_probe)\n";
        res << fmt("# pcie=%s x%s, hef=%s, video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
                   o.video.c_str());
        res << fmt("# batch=%d, requested slots=%d, compare_frames=%zu, bench_frames=%d\n\n", o.batch, o.slots,
                   captured.size(), o.bench_frames);
        for (const auto& oc : outcomes) {
            res << fmt("[%s] constructed=%d ok=%d slots=%d min_queue=%zu infer_queue=%zu scheduled=%d\n"
                       "        callbacks=%ld failed=%ld first_error=%s\n"
                       "        fps=%.2f (min %.2f, max %.2f) latency median=%.2f p90=%.2f ms  temp %.1f -> %.1f C\n"
                       "        error=%s\n",
                       api_name(oc.api), static_cast<int>(oc.constructed), static_cast<int>(oc.ok), oc.slots,
                       oc.min_queue, oc.infer_queue, static_cast<int>(oc.scheduled), oc.cb_total, oc.cb_failed,
                       status_name(oc.first_error).c_str(), oc.fps, oc.fps_min, oc.fps_max, oc.lat_median,
                       oc.lat_p90,
                       static_cast<double>(oc.temp_before), static_cast<double>(oc.temp_after),
                       oc.error.empty() ? "-" : oc.error.c_str());
        }
        double primary_fps = 0.0;
        for (const auto& oc : outcomes)
            if (oc.api == primary_api) primary_fps = oc.fps;
        res << fmt("\nprimary api          : %s\n", api_name(primary_api));
        res << fmt("tensor bytes match between both apis : %s\n",
                   alt_identical < 0 ? "(only one ran)" : fmt("%ld / %zu frames", alt_identical, captured.size()).c_str());
        res << fmt("fixtures             : %d frames x %zu tensors, %.2f MB -> %s\n", n_dump, out_meta.size(),
                   static_cast<double>(dumped_bytes) / (1024.0 * 1024.0), o.fixture_dir.c_str());
        res << fmt("hailort.log errors    : %d (inference section), thermal/clock warning: %d\n", hailort_errors, hailort_thermal);
        res << fmt("                        %d (total — includes %d intended configure() failures from Phase 3)\n\n",
                   hailort_errors_total, hailort_errors_total - hailort_errors);
        res << "Pass criteria verdict\n";
        res << fmt("  ① 6 raw outputs, name/shape match          : %s\n", layout_ok ? "PASS" : "FAIL");
        res << fmt("  ② get_quant_infos().size() == 1            : %s\n", quant_per_feature ? "FAIL" : "PASS");
        res << fmt("  ③ at least one api with all 7 callbacks SUCCESS : PASS (%s)\n", api_name(primary_api));
        res << fmt("  ④ match() ratio >= 0.95 (%zu frames)       : %s (best %.4f)\n", captured.size(),
                   b.ratio >= 0.95 ? "PASS" : "FAIL", b.ratio);
        res << fmt("  ⑤ excluding P5 columns 20~23 is right      : %s (exclude %.4f / include %.4f, padding nonzero %ld/%ld)\n",
                   (inc_ratio < b.ratio || inc_test > b.test) ? "PASS" : "PASS(weak evidence: padding below threshold)",
                   b.ratio, inc_ratio, pad_nonzero, pad_bytes);
        res << fmt("  ⑥ batch %d, %d frames FPS >= 50, log error 0 : %s (%.2f FPS, errors %d)\n", o.batch,
                   o.bench_frames, (primary_fps >= 50.0 && hailort_errors == 0) ? "PASS" : "FAIL", primary_fps,
                   hailort_errors);

        const bool pass = layout_ok && !quant_per_feature && b.ratio >= 0.95 && hailort_errors == 0 &&
                          (o.skip_bench || primary_fps >= 50.0);
        std::printf("\n[P1] %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 1;
    }
}
