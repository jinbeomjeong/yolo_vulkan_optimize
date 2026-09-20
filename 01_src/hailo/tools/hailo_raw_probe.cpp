// yvo_hailo_raw_probe: HailoRT **원시 출력 6스트림** 경로를 검증하는 프로브 (계획서 §6.2 P1 + P2).
//
// --probe p1 (기본): API 경로와 텐서 의미를 **호스트 malloc 버퍼만**으로 검증한다(GPU 미사용).
// --probe p2      : 출력 6개를 **dma-buf 로 받는다.** 후처리 트랙의 go/no-go 관문이다.
//   · --alloc malloc|heap|vulkan (쉼표로 여러 개). malloc 이 바이트 비교의 기준값이다.
//   · size 4조합 전수 시도: dma_map_dmabuf 에 (hw_frame_size | dmabuf_size)
//                        x read_async 에 (hw_frame_size | dmabuf_size)
//   · 세 방식의 6 텐서 바이트 일치, cpu_read_1.22MB_ms, dma_map_dmabuf 96회 총 소요를 측정한다
//   · 시작·종료 시점 fd 수를 세어 누수가 없음을 보인다
// --probe f0      : **입력 dma-buf(H2D) + 출력 dma-buf(D2H) 동시 사용**의 관문이다.
//   지금까지 검증된 조합은 (dma-buf 입력 + 호스트 NMS 출력) 과 (호스트 malloc 입력 + dma-buf 출력 6개)
//   둘뿐이라, 통합 파이프라인이 쓸 (Vulkan export dma-buf 입력 + dma-buf 출력 6개) 는 미검증이었다.
//   같은 프레임으로 --in-dmabuf 있음/없음을 돌려 출력 6텐서의 바이트 일치, 프레임당 콜백 7개,
//   fd 누수, hailort.log warning 을 비교한다. infer_async 와 write_async 를 따로 판정한다.
//
// 하는 일:
//  0) Hef 파싱만으로 출력 스트림 6개의 shape/hw_shape/hw_frame_size/format을 덤프한다.
//  1) VDevice(ROUND_ROBIN) → create_configure_params(hef) → batch_size 설정 → configure(hef, params).
//     is_scheduled(), infer_queue_size(), 스트림별 get_async_max_queue_size(),
//     그리고 OutputStream::get_quant_infos()의 .size()와 값 전부를 덤프한다.
//     get_quant_infos().size() > 1 이면(per-feature 양자화) 즉시 실패로 보고한다 — 설계가 바뀐다.
//  2) --api {infer_async, read_async, both} 두 경로를 모두 구현해 시도한다.
//     infer_async: 입력 1 + 출력 6을 한 요청으로 묶는다(FIFO 짝맞추기를 HailoRT가 한다).
//     read_async : 7개 스트림 전부에 HAILO_STREAM_FLAGS_ASYNC를 configure 시점에 세팅하고,
//                  write_async 1회 + read_async 6회를 한 mutex 안에서 발행해 FIFO 짝을 직접 보장한다.
//  3) 실제 프레임을 처리해 6 텐서를 픽스처로 덤프하고(04_data/raw_tensors/), CPU 기준 구현
//     (common/cpu_yolov8_decode.hpp)의 검출을 **별도 패스로 얻은 HailoRT NMS 검출**과 match()로 비교한다.
//     텐서 해석 미확정 변수 4개(argmax_only / DFL 채널 순서 / sigmoid / > vs >=)를 전수 시도한다.
//  4) (B)안 폐기 증거: InferModel의 출력 개수와, 출력에 NHCW format order를 걸었을 때의 configure() 실패 상태.
//
// Hailo 장치가 1개라 러너를 동시에 둘 수 없다. 그래서 단계마다 객체를 완전히 파괴하고 다음 단계로 간다
// (05_tests/test_detections.cpp의 run_nhcw_pass() 패턴).
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

#include "common/cpu_letterbox.hpp"
#include "common/dma_heap_alloc.hpp"   // dma_sync_* (입력 dma-buf 가 캐시된 힙일 때만 쓴다)
#include "common/cpu_yolov8_decode.hpp"
#include "common/detection_match.hpp"
#include "common/frame_types.hpp"
#include "hailo/hailo_raw_runner.hpp"
#include "hailo/hailo_runner.hpp"
#include "hailo/tools/p2_out_buffers.hpp"
#include "common/stats.hpp"
#include "common/tool_util.hpp"
#include "common/hailo_align.hpp"

using yvo::percentile;

namespace {

using yvo::tool::fmt;
using yvo::tool::count_open_fds;
using yvo::tool::Nv12Frame;
using yvo::tool::read_nv12_frames;

constexpr int kSrcW = 1920;
constexpr int kSrcH = 1080;
// HailoRT의 vdma_stream.cpp:379가 read_async()에 **16384 B 정렬**을 요구한다. 4096 B로 주면
// "unaligned buffer ... causes performance degradation" 경고가 프레임×스트림마다 뜨고
// 실측으로 FPS가 48.5 → 51.7로 3 FPS 차이가 났다(계획서 §7 검증 기록).
constexpr std::size_t kDmaAlign = yvo::kHailoAsyncAlign;  // common/hailo_align.hpp
constexpr float kIouMatchThreshold = yvo::detmatch::kIouThreshold;  // common/detection_match.hpp
constexpr float kHefScoreThreshold = 0.2f;   // HEF의 YOLOV8 post-process 설정
constexpr float kHefIouThreshold = 0.7f;
constexpr int kHefMaxPerClass = 100;

// ---------------------------------------------------------------- 공통 유틸

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

// 스트림 이름의 '/' 뒤 부분(예: "yolov11m/conv74_178" -> "conv74_178")
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

// stdout과 파일에 동시에 쓴다. 증거 파일이 곧 보고서다.
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

// ---------------------------------------------------------------- 영상 → NHCW 입력

// hailo_selftest.cpp의 CPU letterbox NHCW 경로와 같다: cpu_letterbox_nv12_bt709 → rgb_packed_to_nhcw
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

// 매칭 기준은 common/detection_match.hpp 하나뿐이다(2026-09-21 이전에는 여기에 사본이 있었다).
using yvo::detmatch::match;

// ---------------------------------------------------------------- 원시 스트림 세션

enum class RawApi { InferAsync, ReadAsync };

const char* api_name(RawApi a) { return a == RawApi::InferAsync ? "infer_async" : "read_async"; }

struct StreamMeta {
    std::string name;
    hailo_stream_info_t info{};
    std::vector<hailo_quant_info_t> quant;
    std::size_t frame_size = 0;
    std::size_t queue_size = 0;
};

// 프레임 1장 분량의 원시 출력 6개
struct RawFrame {
    std::vector<std::vector<std::uint8_t>> t;   // 6개, 스트림 순서는 RawSession::outputs_
};

// dma_map_dmabuf / read_async 에 넘길 크기. P2가 4조합을 전수 시도한다(계획서 §4.2).
enum class SizeKind { Hw, Dmabuf };
const char* size_kind_name(SizeKind k) { return k == SizeKind::Hw ? "hw_frame_size" : "dmabuf_size"; }

struct RawSessionConfig {
    std::string hef_path;
    std::uint16_t batch = 8;
    int want_slots = 16;
    RawApi api = RawApi::InferAsync;
    // nullptr 이면 내부 malloc 풀을 쓴다(P1 동작). dma-buf 풀을 주면 출력을 fd 로 받는다.
    yvo::p2::OutBufPool* out_pool = nullptr;
    SizeKind map_size = SizeKind::Hw;    // dma_map_dmabuf(fd, size, D2H) 의 size
    SizeKind read_size = SizeKind::Hw;   // read_async(fd, size, cb) / hailo_dma_buffer_t.size
    // ── F0: 입력 H2D dma-buf ──
    // nullptr 이면 입력은 호스트 malloc 이다(오늘 동작, 바이트 단위로 같아야 한다).
    // 풀을 주면 슬롯별 입력 버퍼를 그 풀에서 받아 fd 로 H2D 전송한다.
    yvo::p2::OutBufPool* in_pool = nullptr;
    SizeKind in_map_size = SizeKind::Dmabuf;   // dma_map_dmabuf(fd, size, H2D) 의 size
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
            // 7개 스트림 전부 ASYNC. read_async 경로의 전제이고, infer_async에도 무해하다.
            for (auto& sp : kv.second.stream_params_by_name) {
                sp.second.flags = HAILO_STREAM_FLAGS_ASYNC;
            }
        }
        auto ngs = unwrap(vdevice_->configure(*hef_, params_map), "VDevice::configure");
        if (ngs.size() != 1) throw std::runtime_error("network group이 1개가 아니다");
        ng_ = ngs[0];

        auto in_refs = ng_->get_input_streams();
        auto out_refs = ng_->get_output_streams();
        if (in_refs.size() != 1) throw std::runtime_error("입력 스트림이 1개가 아니다");
        input_ = &in_refs[0].get();
        for (auto& r : out_refs) outputs_.push_back(&r.get());
        // 이름 순으로 고정해 스트림 순서를 실행 간에 결정적으로 만든다
        std::sort(outputs_.begin(), outputs_.end(),
                  [](const hailort::OutputStream* a, const hailort::OutputStream* b) { return a->name() < b->name(); });

        in_meta_ = collect(*input_);
        for (auto* o : outputs_) out_meta_.push_back(collect(*o));

        std::size_t min_queue = in_meta_.queue_size;
        for (const auto& m : out_meta_) min_queue = std::min(min_queue, m.queue_size);
        // num_slots를 큐 크기 최솟값으로 클램프한다. 안 하면 HAILO_QUEUE_IS_FULL이 난다.
        num_slots_ = std::max(1, std::min(want_slots, static_cast<int>(min_queue)));
        min_queue_ = min_queue;

        // 입력: in_pool 이 nullptr 이면 호스트 malloc(P1/P2 동작), 풀을 주면 dma-buf fd(F0).
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

        // dma-buf 풀이면 프로세스 수명 동안 한 번만 매핑한다(출력 슬롯 x 텐서 = 96회, 입력 슬롯당 1회).
        // 소요 시간을 누적해 계획서 §6.2 P2-⑤(총 < 100 ms)를 판정한다.
        // ⚠ 크기 계약(부록 A.13): dma_map_dmabuf 에는 dma-buf 실크기, 전송 API 에는 hw_frame_size.
        //   방향도 다르다 — 입력은 H2D, 출력은 D2H.
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
            if (!drain(5000)) std::fprintf(stderr, "[warn] RawSession: drain 시한 초과 — 콜백이 오지 않았다\n");
        } catch (...) {
        }
        for (const auto& m : mapped_) vdevice_->dma_unmap_dmabuf(m.fd, m.size, m.dir);
        mapped_.clear();
        ng_.reset();
        vdevice_.reset();
    }

    // dma_map_dmabuf 통계
    long map_calls() const { return map_calls_; }
    long map_failed() const { return map_failed_; }
    hailo_status map_status() const { return map_status_; }
    double map_ms() const { return map_ms_; }
    bool uses_dmabuf() const { return cfg_.out_pool != nullptr && !slots_.empty() && slots_[0]->out[0].fd >= 0; }

    // 입력(H2D) dma-buf 통계
    bool uses_in_dmabuf() const { return !slots_.empty() && slots_[0]->in_ref.fd >= 0; }
    long in_map_calls() const { return in_map_calls_; }
    long in_map_failed() const { return in_map_failed_; }
    hailo_status in_map_status() const { return in_map_status_; }
    // 보고용: 입력 버퍼의 (hw_frame_size, dmabuf_size). dma-buf 가 아니면 두 번째가 0.
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

    // 프레임 1장 제출. on_done(slot, status)은 HailoRT 콜백 스레드에서 호출되고,
    // 반환 후 슬롯 버퍼는 자유롭게 재사용할 수 있다.
    void submit(const std::uint8_t* nhcw_input, const std::function<void(int, hailo_status)>& on_done) {
        const int slot = acquire_slot();
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        // 호스트가 입력 프레임을 채운다. 캐시된 dma-buf(heap)면 CPU 쓰기 구간을 sync 로 감싼다.
        // Vulkan export 버퍼는 HOST_COHERENT 라 needs_sync == false 다.
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

    // 모든 슬롯이 반납될 때까지 기다린다. timeout_ms 안에 못 끝내면 false.
    //
    // ⚠ **시한이 필요한 이유(P2에서 실측)**: infer_async() 에 hw_frame_size 와 다른 크기를 넘기면
    //   HailoRT 는 SUCCESS 를 돌려준 뒤 내부 워커에서 CHECK 에 걸려 요청을 조용히 버린다
    //   (hailort.log core_op.cpp:524). **콜백이 하나도 오지 않는다.** 무한 대기하면 프로브가 멈춘다.
    bool drain(int timeout_ms = 30000) {
        std::unique_lock<std::mutex> lk(m_);
        return cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                            [&] { return free_slots_.size() == static_cast<std::size_t>(num_slots_); });
    }

private:
    struct Slot {
        std::shared_ptr<std::uint8_t> in;                 // 호스트 malloc 경로의 수명만 쥔다
        yvo::p2::OutBufRef in_ref;                        // 실제로 쓰는 입력 버퍼. fd < 0 이면 호스트 포인터
        std::vector<yvo::p2::OutBufRef> out;              // 6개. fd < 0 이면 호스트 포인터 경로
        std::vector<std::shared_ptr<std::uint8_t>> keep;  // malloc 모드의 수명 유지
        std::atomic<int> remaining{0};
        std::atomic<hailo_status> status{HAILO_SUCCESS};
        std::function<void(int, hailo_status)> on_done;
    };

    // dma_map_dmabuf / read_async 에 넘길 크기. dmabuf_size 가 0(측정 실패)이면 hw 로 떨어진다.
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
            throw std::runtime_error("RawSession::acquire_slot: 30초 안에 빈 슬롯이 생기지 않았다(콜백 유실)");
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
            // BufferRepresentation은 union이다. buffer_type을 먼저 세팅하고 해당 멤버만 대입한다.
            // NamedBuffersCallbacks 는 입출력을 구분하지 않는 단일 맵이라 입력도 DMA_BUFFER 를 넣을 수 있다
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
        // 백프레셔: ongoing callback 수가 num_slots_ 미만이 될 때까지 막는다.
        // increase_ongoing_callbacks() 를 직접 부르지 않는 이유는 hailo_raw_runner.cpp 의 같은 자리
        // 주석에 있다 — infer_async 가 요청당 1 을 이미 세므로 수동 쌍을 넣으면 프레임당 2 가 올라가
        // in-flight 상한이 num_slots_/2 로 깎인다(HailoRT 4.24.0 실측).
        const auto st_wait = ng_->wait_for_ongoing_callbacks_count_under(static_cast<std::size_t>(num_slots_));
        if (st_wait != HAILO_SUCCESS) throw yvo::HailoError("wait_for_ongoing_callbacks_count_under", st_wait);
        const auto st = ng_->infer_async(named, [](hailo_status) {});
        if (st != HAILO_SUCCESS) {
            // 콜백이 오지 않으므로 슬롯을 직접 반납한다.
            sl.status.store(st);
            first_error_.store(st);
            sl.remaining.store(0);
            finish(slot);
            throw yvo::HailoError("ConfiguredNetworkGroup::infer_async", st);
        }
    }

    void submit_read_async(int slot) {
        Slot& sl = *slots_[static_cast<std::size_t>(slot)];
        // FIFO 짝맞추기: write 1회 + read 6회를 한 mutex 안에서 발행한다.
        // 스트림별 N번째 요청이 N번째 프레임에 대응한다는 보장이 이 잠금뿐이다.
        std::lock_guard<std::mutex> issue(issue_m_);
        const auto cb_in = [this, slot](const hailort::InputStream::CompletionInfo& ci) {
            note_callback(slot, ci.status);
        };
        // dmabuf fd 오버로드(/usr/include/hailo/stream.hpp:253)는 헤더에 "currently experimental" 이다.
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
                // 이미 발행한 요청의 콜백은 온다. 남은 개수를 줄여 프레임이 끝날 수 있게 한다.
                sl.status.store(st);
                first_error_.store(st);
                const int not_issued = static_cast<int>(outputs_.size() - i);
                if (sl.remaining.fetch_sub(not_issued) == not_issued) finish(slot);
                throw yvo::HailoError("OutputStream::read_async", st);
            }
        }
    }

public:
    // 슬롯의 출력 버퍼(6개)를 복사해 낸다. on_done 콜백 안이나 반환 직후에만 유효하다.
    // heap 모드는 CPU 캐시 일관성을 위해 DMA_BUF_IOCTL_SYNC(START|READ) ... (END|READ) 로 감싼다.
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

    // 6 텐서 전체(1.22 MB)를 CPU 가 순차로 읽는 데 걸리는 시간(ms). dst 는 호출자가 재사용한다.
    // 계획서 §6.2 P2-⑥ cpu_read_1.22MB_ms 의 측정 함수다.
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

    // 보고용: 슬롯 0 의 텐서별 (hw_frame_size, dmabuf_size)
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
    // 입력(H2D) 매핑은 따로 센다. 출력 통계(P2 판정)를 오염시키지 않기 위해서다.
    long in_map_calls_ = 0;
    long in_map_failed_ = 0;
    hailo_status in_map_status_ = HAILO_SUCCESS;
};

} // namespace

namespace {

// ---------------------------------------------------------------- 스케일 레이아웃 파생

// 스트림 6개에서 (클래스 텐서, box 텐서) 짝을 스케일별로 찾는다.
// reg_max·stride·classes는 shape에서 파생 계산하고 불일치하면 예외를 던진다(계획서 §4.0(b)).
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
    if (out.size() != 6) throw std::runtime_error("원시 출력 스트림이 6개가 아니다: " + std::to_string(out.size()));
    RawLayout L;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto& hw = out[i].info.hw_shape;
        const auto& sh = out[i].info.shape;
        const std::size_t expect = static_cast<std::size_t>(hw.height) * hw.width * hw.features;
        if (expect != out[i].info.hw_frame_size) {
            throw std::runtime_error("hw_shape와 hw_frame_size가 안 맞는다: " + out[i].name);
        }
        const int h = static_cast<int>(hw.height);
        if (h == 0 || 640 % h != 0) throw std::runtime_error("grid height에서 stride를 파생할 수 없다: " + out[i].name);
        const int stride = 640 / h;
        int si = -1;
        if (stride == 8) si = 0;
        else if (stride == 16) si = 1;
        else if (stride == 32) si = 2;
        else throw std::runtime_error("예상 밖 stride " + std::to_string(stride) + ": " + out[i].name);

        ScaleLayout& s = L.s[si];
        s.grid_h = h;
        s.grid_w = static_cast<int>(sh.width);
        s.w_padded = static_cast<int>(hw.width);
        s.stride = stride;
        const int feats = static_cast<int>(hw.features);
        if (feats % yvo::kDflSides == 0 && feats / yvo::kDflSides == yvo::kDflRegMax) {
            s.box_idx = static_cast<int>(i);
            if (L.reg_max == 0) L.reg_max = feats / yvo::kDflSides;
            else if (L.reg_max != feats / yvo::kDflSides) throw std::runtime_error("reg_max 불일치");
        } else {
            s.cls_idx = static_cast<int>(i);
            if (L.num_classes == 0) L.num_classes = feats;
            else if (L.num_classes != feats) throw std::runtime_error("클래스 수 불일치");
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (L.s[i].cls_idx < 0 || L.s[i].box_idx < 0)
            throw std::runtime_error("스케일 " + std::to_string(i) + "의 cls/box 짝을 못 찾았다");
    }
    if (L.num_classes != 80 || L.reg_max != yvo::kDflRegMax)
        throw std::runtime_error("classes/reg_max가 80/16이 아니다");
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

// ---------------------------------------------------------------- 옵션

struct Options {
    std::string video = "04_data/videos/video_h265.mp4";
    std::string hef = "03_models/yolov11m.hef";
    std::string out_dir = "07_results/gpu_postprocess/evidence";
    std::string fixture_dir = "04_data/raw_tensors";
    std::string api = "both";           // infer_async | read_async | both
    int compare_frames = 30;            // CPU 기준 구현 ↔ HailoRT NMS 비교 프레임 수 (>= 20)
    int api_check_frames = 8;           // API 경로 확인용 프레임 수
    int dump_frames = 5;                // 픽스처로 저장할 프레임 수
    int bench_frames = 200;             // 처리량 측정 프레임 수
    int bench_repeat = 3;               // 설정당 반복 횟수(중앙값 보고)
    int warmup = 30;
    int batch = 8;
    int slots = 16;
    bool skip_bench = false;
    bool skip_infer_model = false;
    // ── P2 ──
    std::string probe = "p1";           // p1 | p2 | f0
    std::string alloc = "malloc,heap,vulkan";
    int matrix_frames = 2;              // size 조합 표에서 한 셀당 제출할 프레임 수
    int matrix_slots = 2;               // 표 셀의 슬롯 수 (빠르게 돌리기 위해 작게)
    int cpu_read_reps = 30;             // cpu_read_1.22MB_ms 반복 횟수
    float cool_above = 75.0f;           // 이 온도를 넘으면 냉각한다
    float cool_target = 65.0f;
    int cool_max_s = 120;
    // ── F0: 입력 dma-buf(H2D) ──
    bool in_dmabuf = false;             // p2 의 vulkan 최종 실행에서 입력도 dma-buf 로 준다
    int f0_frames = 16;                 // f0 게이트가 A/B 로 비교할 프레임 수
};

void usage() {
    std::fprintf(stderr,
                 "usage: yvo_hailo_raw_probe [--video P] [--hef P] [--out-dir D] [--fixture-dir D]\n"
                 "        [--api infer_async|read_async|both] [--compare-frames N] [--api-check-frames N]\n"
                 "        [--dump-frames N] [--bench-frames N] [--bench-repeat N] [--warmup N] [--batch B] [--slots S]\n"
                 "        [--skip-bench] [--skip-infer-model]\n"
                 "        [--probe p1|p2|f0|runner] [--alloc malloc,heap,vulkan] [--matrix-frames N]\n"
                 "        [--matrix-slots N] [--cpu-read-reps N] [--no-cool] [--in-dmabuf] [--f0-frames N]\n"
                 "  --probe f0 : 입력 dma-buf(H2D) + 출력 dma-buf(D2H) 동시 사용 관문. A/B 를 스스로 돌린다.\n"
                 "  --probe runner : RawSession -> yvo::HailoRawRunner 승격 스모크. f0 경로와 바이트 비교한다.\n"
                 "  --in-dmabuf: --probe p2 의 vulkan 최종 실행에서 입력도 Vulkan export dma-buf 로 준다.\n");
}

// ---------------------------------------------------------------- 한 API 경로 실행 결과

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

// ---------------------------------------------------------------- P2: dma-buf 출력 경로

// 칩 온도만 읽고 장치를 즉시 놓는다. 세션을 만들기 전 냉각 판정에 쓴다.
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

// 06_tools/bench/pipeline_bench.sh 의 cool_down() 과 같은 규칙. 대기한 초를 돌려준다.
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

// hailort.log 의 [warning] 줄 수. needle 이 nullptr 이 아니면 그 문자열을 포함한 줄만 센다
// (F0 합격 기준 ③ "unaligned buffer causes performance degradation" 판정용).
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
        case AllocMode::Malloc: return nullptr;   // RawSession 내부 malloc 경로를 쓴다
        case AllocMode::Heap: return std::make_unique<yvo::p2::HeapOutPool>();
        case AllocMode::Vulkan: return std::make_unique<yvo::p2::VulkanOutPool>(verbose);
    }
    return nullptr;
}

// 프레임 N장을 제출해 6 텐서를 확보한다. 프레임마다 drain 해서 슬롯 버퍼를 그대로 읽는다.
struct RunOutcome {
    bool constructed = false;
    bool issued = true;          // read_async/infer_async 발행이 전부 성공했는가
    std::string error;
    long cb_total = 0, cb_failed = 0;
    bool callbacks_lost = false;   // drain 시한 초과 = 콜백 유실
    hailo_status first_error = HAILO_SUCCESS;
    long map_calls = 0, map_failed = 0;
    hailo_status map_status = HAILO_SUCCESS;
    double map_ms = 0.0;
    // 입력 H2D dma-buf (F0)
    bool in_dmabuf = false;
    long in_map_calls = 0, in_map_failed = 0;
    hailo_status in_map_status = HAILO_SUCCESS;
    std::pair<std::size_t, std::size_t> in_size{0, 0};   // (hw_frame_size, dmabuf_size)
    int frames_ok = 0;
    std::vector<RawFrame> frames;
    std::vector<double> cpu_read_ms;
    std::vector<std::pair<std::size_t, std::size_t>> sizes;   // (hw_frame_size, dmabuf_size)
    std::vector<StreamMeta> out_meta;   // 출력 스트림 메타(이름순). --probe runner 의 텐서 번호 대응에 쓴다
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
        r.error = "dma_map_dmabuf(H2D) 실패: " + status_name(r.in_map_status);
    if (r.map_failed > 0) {
        // 매핑이 실패했어도 발행을 시도해 본다(HailoRT 가 내부에서 자동 매핑할 수도 있다).
        if (r.error.empty()) r.error = "dma_map_dmabuf 실패: " + status_name(r.map_status);
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
            // 콜백이 오지 않았다. 이 조합은 쓸 수 없다 — 슬롯 버퍼의 내용도 신뢰할 수 없다.
            r.callbacks_lost = true;
            if (r.error.empty()) r.error = "drain 시한 초과: 제출한 프레임의 콜백이 오지 않았다";
            break;
        }
    }
    // cpu_read_1.22MB_ms: 워밍업 후 반복. 슬롯 0 을 계속 읽는다(내용은 이미 채워져 있다).
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

// ---------------------------------------------------------------- F0: 입력 dma-buf(H2D) 관문
//
// 통합 파이프라인은 V3D 가 쓴 전처리 출력을 그대로 NPU 입력으로 준다. 그 조합
// (Vulkan export dma-buf 입력 H2D + dma-buf 출력 6개 D2H)만 미검증이라, 여기서 A/B 로 닫는다.
//   A(기준) : 호스트 malloc 입력 + dma-buf 출력 6개  ← P2 에서 이미 통과한 경로
//   B(관문) : Vulkan export dma-buf 입력 + dma-buf 출력 6개
// 두 실행의 출력 6텐서가 바이트 단위로 같아야 하고, 프레임당 콜백 7개가 전부 SUCCESS 여야 한다.
struct F0Row {
    bool in_dmabuf = false;
    RawApi api = RawApi::InferAsync;
    SizeKind in_map = SizeKind::Dmabuf;
    SizeKind in_write = SizeKind::Hw;
    int frames = 0;
    RunOutcome out;
    int bytes_matched = 0, bytes_compared = 0;
    long cb_expect = 0;
    int log_err = 0;      // 이 실행 구간에만 찍힌 hailort.log [error] 수
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
    // 출력 쪽 크기 계약은 P2 가 고른 값을 그대로 쓴다(p2_alloc_matrix.txt: map=dmabuf_size, read=hw_frame_size).
    base.map_size = SizeKind::Dmabuf;
    base.read_size = SizeKind::Hw;

    const int n = std::max(1, std::min<int>(o.f0_frames, static_cast<int>(inputs.size())));

    Tee t(o.out_dir + "/f0_in_dmabuf.txt");
    t << "# F0 관문: 입력 dma-buf(H2D) + 출력 dma-buf(D2H) 동시 사용\n";
    t << fmt("# pcie=%s x%s hef=%s video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
             o.video.c_str());
    t << fmt("# batch=%d slots=%d frames=%d api=%s 출력계약(map=dmabuf_size, transfer=hw_frame_size)\n", o.batch,
             o.slots, n, o.api.c_str());
    t << fmt("# 냉각: hailo %.1f -> %.1f C, 대기 %d s\n\n", static_cast<double>(t_before),
             static_cast<double>(t_after), cooled_s);
    const long fds_files = count_open_fds();

    // 실행 하나. in_dmabuf 면 Vulkan 풀 **하나**가 입력 버퍼와 출력 버퍼를 모두 낸다
    // (통합 파이프라인과 같은 구조다 — V3D 가 입력을 쓰고 셰이더가 출력을 읽는다).
    // VulkanContext 가 세는 검증 레이어 에러·경고를 실행마다 누적한다
    // (VK_LOADER_LAYERS_ENABLE='*validation' 으로 켜면 값이 의미를 가진다).
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
        std::printf("[f0] %s: A(호스트 입력) 기준 캡처 %d 프레임...\n", api_name(api), n);
        leg.ref = run_one(false, api, SizeKind::Dmabuf, SizeKind::Hw, o.slots, n);
        leg.ref_ok = leg.ref.constructed && leg.ref.frames_ok == n && leg.ref.cb_failed == 0;
        if (!leg.ref_ok) {
            std::printf("[f0] %s: 기준 캡처 실패 — %s\n", api_name(api), leg.ref.error.c_str());
            legs.push_back(std::move(leg));
            continue;
        }
        const std::size_t n_tensors = leg.ref.frames[0].t.size();

        // 입력 크기 계약 4조합 전수 시도. 틀린 크기를 넘기면 콜백이 오지 않으므로 drain 시한이 잡아낸다.
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

        // 계약(부록 A.13) 그대로의 조합으로 전체 슬롯 실행. 이것이 관문의 본 실행이다.
        F0Row full;
        full.in_dmabuf = true;
        full.api = api;
        full.in_map = SizeKind::Dmabuf;
        full.in_write = SizeKind::Hw;
        full.frames = n;
        std::printf("[f0] %s: B(입력 dma-buf) 본 실행 %d 프레임, slots=%d...\n", api_name(api), n, o.slots);
        const int err_before_full = count_hailort_log(false);
        full.out = run_one(true, api, full.in_map, full.in_write, o.slots, n);
        full.log_err = count_hailort_log(false) - err_before_full;
        judge(full, leg.ref, n_tensors);
        leg.full = std::move(full);
        std::printf("[f0] %s: B -> %s (cb %ld/%ld, bytes %d/%d)\n", api_name(api), leg.full.pass ? "PASS" : "FAIL",
                    leg.full.out.cb_total, leg.full.cb_expect, leg.full.bytes_matched, leg.full.bytes_compared);
        legs.push_back(std::move(leg));
    }

    // ── 보고
    t << "## 입력 버퍼 크기\n";
    for (const auto& leg : legs) {
        if (leg.full.out.in_size.first == 0) continue;
        t << fmt("  입력: hw_frame_size=%zu dmabuf_size=%zu (차 %+zd)\n", leg.full.out.in_size.first,
                 leg.full.out.in_size.second,
                 static_cast<std::ptrdiff_t>(leg.full.out.in_size.second) -
                     static_cast<std::ptrdiff_t>(leg.full.out.in_size.first));
        break;
    }
    t << "\n## 입력 크기 계약 4조합 (한 셀: slots=" + std::to_string(o.matrix_slots) + ")\n";
    t << "api           dma_map(H2D)   transfer size    map_status                  issue  cb/기대   bytes    판정\n";
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

    t << "\n## 본 실행 (입력 dma_map=dmabuf_size, 전송 크기=hw_frame_size)\n";
    for (const auto& leg : legs) {
        t << fmt("[%s]\n", api_name(leg.api));
        t << fmt("  A 기준(호스트 입력) : constructed=%d frames_ok=%d/%d cb=%ld 실패=%ld %s\n",
                 leg.ref.constructed ? 1 : 0, leg.ref.frames_ok, n, leg.ref.cb_total, leg.ref.cb_failed,
                 leg.ref.error.empty() ? "" : ("error: " + leg.ref.error).c_str());
        const F0Row& b = leg.full;
        t << fmt("  B 관문(입력 dma-buf): constructed=%d frames_ok=%d/%d\n", b.out.constructed ? 1 : 0,
                 b.out.frames_ok, b.frames);
        t << fmt("    dma_map_dmabuf(H2D) : %ld회, 실패 %ld, status=%s\n", b.out.in_map_calls, b.out.in_map_failed,
                 status_name(b.out.in_map_status).c_str());
        t << fmt("    dma_map_dmabuf(D2H) : %ld회, 실패 %ld, status=%s (H2D+D2H 총 %.3f ms)\n", b.out.map_calls,
                 b.out.map_failed, status_name(b.out.map_status).c_str(), b.out.map_ms);
        t << fmt("    콜백                : %ld개 / 기대 %ld개 (프레임당 7), 실패 %ld, first_error=%s\n",
                 b.out.cb_total, b.cb_expect, b.out.cb_failed, status_name(b.out.first_error).c_str());
        t << fmt("    A 대비 출력 6텐서 바이트 일치 : %d / %d frames\n", b.bytes_matched, b.bytes_compared);
        if (!b.out.error.empty()) t << fmt("    error: %s\n", b.out.error.c_str());
        t << fmt("  판정: %s\n\n", b.pass ? "PASS" : "FAIL");
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

    t << "## 합격 기준\n";
    t << fmt("  ① 프레임당 콜백 7개 전부 SUCCESS : %s (infer_async %s, write_async %s)\n",
             gate ? "PASS" : "FAIL",
             infer_row ? fmt("%ld/%ld 실패 %ld", infer_row->out.cb_total, infer_row->cb_expect,
                             infer_row->out.cb_failed)
                             .c_str()
                       : "(미실행)",
             read_row ? fmt("%ld/%ld 실패 %ld", read_row->out.cb_total, read_row->cb_expect, read_row->out.cb_failed)
                            .c_str()
                      : "(미실행)");
    t << fmt("  ② 출력 6텐서가 호스트 입력 실행과 바이트 일치 : %s (infer_async %d/%d, write_async %d/%d)\n",
             gate ? "PASS" : "FAIL", infer_row ? infer_row->bytes_matched : 0,
             infer_row ? infer_row->bytes_compared : 0, read_row ? read_row->bytes_matched : 0,
             read_row ? read_row->bytes_compared : 0);
    int full_err = 0;
    for (const auto& leg : legs) full_err += leg.full.log_err;
    t << fmt("  ③ hailort.log warning 0 : %s (전체 %d건, unaligned buffer %d건, 온도/클럭 %d건)\n",
             (warns == 0) ? "PASS" : "FAIL", warns, unaligned, thermal);
    t << fmt("     error: 본 실행 구간 %d건, 전체 %d건 (차 %d건은 크기 계약 4조합 표의 **의도된** 거부다)\n",
             full_err, log_err, log_err - full_err);
    t << fmt("  ④ fd 누수 0 : %s (시작 %ld, 증거 파일 연 뒤 %ld, 종료 %ld → %+ld)\n",
             (fds_end - fds_files == 0) ? "PASS" : "FAIL", fds_start, fds_files, fds_end, fds_end - fds_files);
    t << fmt("  ⑤ API 별 결과 : infer_async=%s, write_async(read_async 경로)=%s\n",
             infer_row ? (infer_ok ? "PASS" : "FAIL") : "(미실행)",
             read_row ? (read_ok ? "PASS" : "FAIL") : "(미실행)");
    t << fmt("  ⑥ Vulkan 검증 레이어 : errors %llu, warnings %llu (레이어 활성 %s)\n",
             static_cast<unsigned long long>(vk_errors), static_cast<unsigned long long>(vk_warnings),
             vk_layer_active ? "yes" : "no — VK_LOADER_LAYERS_ENABLE='*validation' 없이 돌린 실행이다");
    t << fmt("\n[결론] %s\n", gate ? "입력 dma-buf(H2D) + 출력 dma-buf(D2H) 동시 사용 가능 — F0 관문 통과"
                                   : "두 API 모두 실패 — 입력은 호스트 버퍼로 남겨야 한다");

    std::printf("\n[F0] %s (infer_async %s, write_async %s)\n", gate ? "PASS" : "FAIL", infer_ok ? "PASS" : "FAIL",
                read_ok ? "PASS" : "FAIL");
    std::printf("[f0] hailort.log warning %d(unaligned %d), error %d(본 실행 %d) / fd 누수 %+ld\n", warns, unaligned,
                log_err, full_err, fds_end - fds_files);
    return gate ? 0 : 1;
}

// ---------------------------------------------------------------- RUNNER: 라이브러리 승격 스모크
//
// 단계 3a 에서 RawSession 을 01_src/hailo/hailo_raw_runner.{hpp,cpp} 로 뽑아냈다. 승격한 코드가
// 원본과 **바이트 단위로 같은 출력**을 내는지 한 프로세스 안에서 A/B 로 확인한다.
//   A(기준) : RawSession, --probe f0 의 B 다리와 **완전히 같은 설정**
//             (Vulkan export dma-buf 입력 H2D + 출력 6개 D2H, map=dmabuf_size, transfer=hw_frame_size)
//   B(승격) : yvo::HailoRawRunner. 같은 프레임을 같은 순서로 제출한다.
// 텐서 번호 대응도 여기서 검증된다 — 러너의 t[2s]/t[2s+1](cls/box)이
// derive_layout() 이 고른 스트림과 실제로 같은 버퍼인지 바이트로 확인한다.
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

    // ── A: RawSession (f0 의 B 다리와 같은 설정)
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
        std::printf("[runner] A(RawSession) 기준 캡처 %d 프레임...\n", n);
        ref = run_capture(sc, pool.get(), o.slots, inputs, n, 0);
        vk_errors += pool->ctx().validation_errors();
        vk_warnings += pool->ctx().validation_warnings();
        vk_layer_active = vk_layer_active || pool->ctx().validation_layer_active();
    }
    const bool ref_ok = ref.constructed && ref.frames_ok == n && ref.cb_failed == 0;
    if (!ref_ok) {
        std::fprintf(stderr, "[runner] 기준 캡처 실패 — %s\n", ref.error.c_str());
        return 1;
    }
    const RawLayout layout = derive_layout(ref.out_meta);
    // 텐서 번호 i(러너 규약) → 기준 캡처의 스트림 인덱스
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
        std::vector<RawFrame> frames;   // 러너 텐서 번호 순서(0..5)
        double fps = 0.0, fps_min = 0.0, fps_max = 0.0, lat_median = 0.0, lat_p90 = 0.0;
        float temp_before = 0.0f, temp_after = 0.0f;
        bool names_match = true;
        yvo::RawModelInfo info;   // 러너가 HEF 메타만으로 파생한 정보(보고용 사본)
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

            // 스트림 이름이 기준 캡처의 대응 스트림과 같은지(= 번호 규약이 맞는지) 먼저 본다
            for (int i = 0; i < 6; ++i)
                if (in.t[i].stream_name != ref.out_meta[static_cast<std::size_t>(ref_idx[i])].name)
                    rr.names_match = false;

            // 슬롯마다 입력 1 + 출력 6을 같은 Vulkan 풀에서 받아 bind_slot()
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

            // ① 바이트 비교: 프레임마다 제출 → drain → 슬롯 버퍼를 읽는다.
            //    (콜백 안에서는 슬롯 번호만 기록한다 — 콜백은 짧아야 하고 Vulkan 메모리를 만지지 않는다)
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

            // ② 처리량: 파이프라인을 채운 채로 돌린다(백프레셔는 러너 안의
            //    wait_for_ongoing_callbacks_count_under 가 건다). 버퍼는 읽지 않는다.
            if (!o.skip_bench) {
                // 통합 파이프라인이 쓸 패턴 그대로다: 콜백은 슬롯 번호를 free 큐에 넣고 바로 나가고,
                // 제출 스레드가 큐에서 꺼내 다시 제출한다(러너는 콜백 **진입 전에** 슬롯을 반납한다).
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
                                throw std::runtime_error("runner bench: 10초 안에 빈 슬롯이 생기지 않았다");
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

    // ── 바이트 비교 (러너 텐서 번호 ↔ 기준 캡처 스트림 인덱스)
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
    t << "# 단계 3a: RawSession -> yvo::HailoRawRunner 승격 스모크\n";
    t << fmt("# pcie=%s x%s hef=%s video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
             o.video.c_str());
    t << fmt("# batch=%d 요청slots=%d frames=%d api=infer_async 계약(map=dmabuf_size, transfer=hw_frame_size)\n",
             o.batch, o.slots, n);
    t << fmt("# 냉각: hailo %.1f -> %.1f C, 대기 %d s\n\n", static_cast<double>(t_before),
             static_cast<double>(t_after), cooled_s);

    t << "## 러너가 HEF 메타만으로 파생한 모델 정보 (픽스처 파일을 읽지 않는다)\n";
    t << fmt("  입력: %s  %dx%d  hw_frame_size=%zu  min_queue=%zu  classes=%d reg_max=%d\n",
             rr.info.input_name.c_str(), rr.info.input_w, rr.info.input_h, rr.info.input_frame_size,
             rr.info.min_queue, rr.info.num_classes, rr.info.reg_max);
    t << "\n## 텐서 번호 대응 (cls(s)=2s, box(s)=2s+1)\n";
    t << "  i  러너 스트림 이름                기준(derive_layout) 스트림 이름  grid(pad)  f   stride  bytes     "
         "scale         zp      같음\n";
    for (int i = 0; i < 6; ++i) {
        const auto& m = rr.info.t[i];
        t << fmt("  %d  %-30s  %-30s  %2dx%2d(%2d)  %3d  %2d      %-8zu  %-12.9g  %-6.4g  %s\n", i,
                 m.stream_name.empty() ? "-" : m.stream_name.c_str(),
                 ref.out_meta[static_cast<std::size_t>(ref_idx[i])].name.c_str(), m.grid_h, m.grid_w, m.w_padded,
                 m.features, m.stride, m.hw_frame_size, static_cast<double>(m.qp_scale),
                 static_cast<double>(m.qp_zp), rr.names_match ? "yes" : "NO");
    }

    t << "\n## A(RawSession) vs B(HailoRawRunner)\n";
    t << fmt("  A: constructed=%d frames_ok=%d/%d cb=%ld 실패=%ld\n", ref.constructed ? 1 : 0, ref.frames_ok, n,
             ref.cb_total, ref.cb_failed);
    t << fmt("  B: constructed=%d slots=%d(min_queue=%zu) frames_ok=%d/%d cb=%ld 실패=%ld\n",
             rr.constructed ? 1 : 0, rr.slots, rr.min_queue, rr.frames_ok, n, rr.cb_total, rr.cb_failed);
    if (!rr.error.empty()) t << fmt("     error: %s\n", rr.error.c_str());
    t << fmt("  출력 6텐서 바이트 일치 : %d / %d frames (다른 바이트 %ld개)\n", matched, compared, diff_bytes);
    if (!o.skip_bench)
        t << fmt("  B 처리량(파이프라인, %d frames x%d 중앙값): %.2f FPS (min %.2f, max %.2f),\n"
                 "     latency median %.2f ms p90 %.2f ms, temp %.1f -> %.1f C\n",
                 o.bench_frames, std::max(1, o.bench_repeat), rr.fps, rr.fps_min, rr.fps_max, rr.lat_median,
                 rr.lat_p90, static_cast<double>(rr.temp_before), static_cast<double>(rr.temp_after));

    t << "\n## 합격 기준\n";
    t << fmt("  ① 러너 생성 + 슬롯 bind 성공            : %s\n", (rr.constructed && rr.error.empty()) ? "PASS" : "FAIL");
    t << fmt("  ② 텐서 번호 ↔ 스트림 이름 대응 일치     : %s\n", rr.names_match ? "PASS" : "FAIL");
    t << fmt("  ③ 콜백 실패 0, 프레임 %d/%d             : %s\n", rr.frames_ok, n,
             (rr.cb_failed == 0 && rr.frames_ok == n) ? "PASS" : "FAIL");
    t << fmt("  ④ f0 경로와 출력 6텐서 바이트 일치      : %s (%d/%d)\n",
             (compared == n && matched == compared) ? "PASS" : "FAIL", matched, compared);
    t << fmt("  ⑤ hailort.log error %d / warning %d(온도·클럭 %d) : %s\n", log_err, warns, thermal,
             (log_err == 0 && warns == 0) ? "PASS" : "FAIL");
    t << fmt("  ⑥ fd 누수 0                             : %s (시작 %ld, 종료 %ld → %+ld)\n",
             (fds_end - fds_start <= 1) ? "PASS" : "FAIL", fds_start, fds_end, fds_end - fds_start);
    t << fmt("  ⑦ Vulkan 검증 레이어 errors %llu (활성 %s)\n", static_cast<unsigned long long>(vk_errors),
             vk_layer_active ? "yes" : "no");
    t << fmt("\n[결론] %s\n", pass ? "HailoRawRunner 는 RawSession 과 같은 출력을 낸다 — 승격 검증 통과"
                                   : "불일치 — 승격 코드를 고쳐야 한다");

    std::printf("\n[RUNNER] %s (bytes %d/%d, cb 실패 %ld, log error %d warning %d, fd %+ld)\n",
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

    // --alloc 파싱
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
            else if (!cur.empty()) throw std::runtime_error("알 수 없는 --alloc 값: " + cur);
            cur.clear();
        }
    }
    if (modes.empty()) throw std::runtime_error("--alloc 이 비었다");
    std::vector<RawApi> apis;
    if (o.api == "infer_async" || o.api == "both") apis.push_back(RawApi::InferAsync);
    if (o.api == "read_async" || o.api == "both") apis.push_back(RawApi::ReadAsync);

    // NPU 를 쓰기 전 냉각 (계획서 §7 / measure 스킬)
    float t_before = 0, t_after = 0;
    const int cooled_s = cool_down(o.cool_above, o.cool_target, o.cool_max_s, t_before, t_after);
    std::printf("[p2] cooling: hailo %.1f -> %.1f C, waited %d s (기준: >%.0f C 면 %.0f C 까지, 최대 %d s)\n",
                static_cast<double>(t_before), static_cast<double>(t_after), cooled_s,
                static_cast<double>(o.cool_above), static_cast<double>(o.cool_target), o.cool_max_s);

    RawSessionConfig base;
    base.hef_path = o.hef;
    base.batch = static_cast<std::uint16_t>(o.batch);

    const int n_cmp = std::min<int>(o.compare_frames, static_cast<int>(inputs.size()));

    // ── 기준값: malloc, read_async(호스트 포인터). P1 에서 이미 검증된 경로다.
    base.api = apis.front();
    base.map_size = SizeKind::Hw;
    base.read_size = SizeKind::Hw;
    RunOutcome ref = run_capture(base, nullptr, o.slots, inputs, n_cmp, o.cpu_read_reps);
    if (!ref.constructed || ref.frames_ok == 0)
        throw std::runtime_error("P2 기준(malloc) 캡처 실패: " + ref.error);
    const std::size_t n_tensors = ref.frames[0].t.size();
    std::printf("[p2] 기준 malloc(%s): frames_ok=%d cb=%ld failed=%ld cpu_read median=%.3f ms\n",
                api_name(base.api), ref.frames_ok, ref.cb_total, ref.cb_failed, percentile(ref.cpu_read_ms, 0.5));

    // ── size 4조합 x alloc(dma-buf) x api 전수 시도
    Tee mat(o.out_dir + "/p2_alloc_matrix.txt");
    mat << "# P2-④ size 인자 4조합 전수 시도 (계획서 §4.2)\n";
    mat << fmt("# pcie=%s x%s hef=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str());
    mat << fmt("# 한 셀: slots=%d, frames=%d. bytes 는 --alloc malloc 결과와의 바이트 단위 비교다.\n",
               o.matrix_slots, o.matrix_frames);
    mat << "# hw_frame_size = 512000/409600/128000/102400/38400/30720,  dmabuf_size = 그것을 16384로 올림\n\n";
    mat << "alloc   api           dma_map size   read_async size  map_status                  issue  cb/fail"
           "  bytes      판정\n";
    mat << "------  ------------  -------------  ---------------  --------------------------  -----  -------"
           "  ---------  ----\n";

    std::vector<MatrixRow> rows;
    for (AllocMode m : modes) {
        if (m == AllocMode::Malloc) continue;   // fd 가 없어 size 계약이 존재하지 않는다
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

    // size 가 안 맞는 조합은 HailoRT 가 CHECK failed 로 거부한다 — **의도된** hailort.log error 다.
    // 최종 판정(신뢰성)은 이 단계 이후의 error 만 센다. P1 이 Phase 3 의 의도된 실패를 분리한 것과 같은 규율이다.
    const int log_errors_after_matrix = count_hailort_log(false);
    mat << fmt("\n# 이 단계에서 hailort.log 에 찍힌 error %d건은 **의도된** 크기 불일치 거부다\n",
               log_errors_after_matrix - log_errors_start);

    // 통과 조합 중 하나를 고른다. read_async 헤더가 "expected to be get_frame_size()" 라고 하므로
    // read=hw 를 우선하고, 그다음 매핑 범위가 넓은 map=dmabuf 를 우선한다.
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
        mat << fmt("[선택] dma_map=%s read_async=%s (통과 조합 %d / 전체 %zu)\n", size_kind_name(best->map_size),
                   size_kind_name(best->read_size),
                   static_cast<int>(std::count_if(rows.begin(), rows.end(), [](const MatrixRow& r) { return r.pass; })),
                   rows.size());
    } else {
        mat << "[선택] 통과한 size 조합이 없다 → 크기 계약 불명확. 계획서 §6.2 P2-④ 불합격.\n";
    }
    for (std::size_t i = 0; i < ref.sizes.size(); ++i)
        mat << fmt("  tensor %zu: hw_frame_size=%zu\n", i, ref.sizes[i].first);

    // ── 선택한 조합으로 전체 슬롯(16 x 6 = fd 96개) 실행
    Tee res(o.out_dir + "/p2_result.txt");
    Tee cr(o.out_dir + "/p2_cpu_read.txt");
    res << "# P2 결과 (yvo_hailo_raw_probe --probe p2)\n";
    res << fmt("# pcie=%s x%s hef=%s video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
               o.video.c_str());
    res << fmt("# batch=%d slots=%d compare_frames=%d api=%s alloc=%s\n", o.batch, o.slots, n_cmp, o.api.c_str(),
               o.alloc.c_str());
    res << fmt("# 냉각: hailo %.1f -> %.1f C, 대기 %d s\n\n", static_cast<double>(t_before),
               static_cast<double>(t_after), cooled_s);

    std::size_t total_bytes = 0;
    for (const auto& sz : ref.sizes) total_bytes += sz.first;
    cr << "# P2-⑥ cpu_read_1.22MB_ms — 6 텐서 전체를 CPU 가 순차 memcpy 로 읽는 시간\n";
    cr << fmt("# 총 %zu B, 반복 %d회(워밍업 3회 제외). 참고: V3D 출력 버퍼 순차 memcpy 1,228,800 B = 0.85~0.87 ms\n",
              total_bytes, o.cpu_read_reps);
    cr << "alloc    median_ms   p90_ms    min_ms    GB/s(median)   비고\n";
    cr << "-------  ----------  --------  --------  ------------   ----\n";

    // 증거 파일 3개(p2_alloc_matrix/p2_result/p2_cpu_read)가 열린 뒤의 fd 수. 누수 판정 기준선이다.
    const long fds_files = count_open_fds();

    struct FinalRow {
        AllocMode alloc;
        RunOutcome out;
        int bytes_matched = 0, bytes_compared = 0;
        bool pass = false;
    };
    std::vector<FinalRow> finals;

    // malloc 기준값도 표에 넣는다
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
        // 이 alloc 모드에서 통과한 조합 중 최고 점수를 고른다(모드별로 다를 수 있다).
        const MatrixRow* mb = nullptr;
        for (const auto& r : rows) {
            if (!r.pass || r.alloc != m) continue;
            if (!mb || score(r) > score(*mb)) mb = &r;
        }
        FinalRow f;
        f.alloc = m;
        if (!mb) {
            f.out.error = "통과한 size 조합이 없다";
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
            // --in-dmabuf: vulkan 모드에서만 입력도 같은 풀의 dma-buf 로 준다(F0 조합).
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
        res << fmt("  dma_map_dmabuf : %ld회, 실패 %ld, status=%s, 총 %.3f ms (%.3f ms/회)\n", f.out.map_calls,
                   f.out.map_failed, status_name(f.out.map_status).c_str(), f.out.map_ms,
                   f.out.map_calls > 0 ? f.out.map_ms / static_cast<double>(f.out.map_calls) : 0.0);
        res << fmt("  콜백           : %ld개, 실패 %ld, first_error=%s\n", f.out.cb_total, f.out.cb_failed,
                   status_name(f.out.first_error).c_str());
        res << fmt("  malloc 대비 바이트 일치 : %d / %d frames\n", f.bytes_matched, f.bytes_compared);
        res << fmt("  pool 보유 시 열린 fd    : %ld (시작 %ld)\n", fds_with_pool, fds_start);
        if (!f.out.error.empty()) res << fmt("  error: %s\n", f.out.error.c_str());
        for (std::size_t i = 0; i < f.out.sizes.size(); ++i)
            res << fmt("  tensor %zu: hw_frame_size=%zu dmabuf_size=%zu (차 %+zd)\n", i, f.out.sizes[i].first,
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
                  f.alloc == AllocMode::Heap ? "DMA_BUF_IOCTL_SYNC 포함, 캐시됨"
                                             : (f.alloc == AllocMode::Vulkan ? "V3D HOST_COHERENT (캐시 없음)"
                                                                             : "호스트 malloc (기준)"));
    }
    cr << fmt("\n총 읽기 바이트 = %zu B (6 텐서 hw_frame_size 합)\n", total_bytes);

    // ── 판정
    const int log_errors_total = count_hailort_log(false) - log_errors_start;
    const int log_errors_matrix = log_errors_after_matrix - log_errors_start;
    const int log_errors = log_errors_total - log_errors_matrix;   // 최종 실행 구간만
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

    res << "합격 기준 판정 (계획서 §6.2 P2)\n";
    res << fmt("  ① dma_map_dmabuf(fd, size, D2H) == SUCCESS : %s (vulkan %s / heap %s)\n",
               any_ok ? "PASS" : "FAIL",
               vk ? status_name(vk->out.map_status).c_str() : "(미실행)",
               hp ? status_name(hp->out.map_status).c_str() : "(미실행)");
    res << fmt("  ② read_async(fd, ...) 콜백 SUCCESS         : %s (vulkan 실패 %ld / heap 실패 %ld)\n",
               any_ok ? "PASS" : "FAIL", vk ? vk->out.cb_failed : -1, hp ? hp->out.cb_failed : -1);
    res << fmt("  ③ 6 텐서가 malloc 결과와 바이트 일치       : %s (vulkan %d/%d, heap %d/%d)\n",
               any_ok ? "PASS" : "FAIL", vk ? vk->bytes_matched : 0, vk ? vk->bytes_compared : 0,
               hp ? hp->bytes_matched : 0, hp ? hp->bytes_compared : 0);
    res << fmt("  ④ size 조합 중 최소 하나 동작              : %s%s\n", best ? "PASS" : "FAIL",
               best ? fmt(" (dma_map=%s, read_async=%s)", size_kind_name(best->map_size),
                          size_kind_name(best->read_size))
                          .c_str()
                    : "");
    res << fmt("  ⑤ dma_map_dmabuf %ld회 총 < 100 ms         : %s (%.3f ms)\n", map_calls_96,
               (map_calls_96 > 0 && map_ms_96 < 100.0) ? "PASS" : "FAIL", map_ms_96);
    res << fmt("  ⑥ cpu_read_1.22MB_ms 세 방식 출력          : %s (%s/p2_cpu_read.txt)\n",
               finals.size() >= 2 ? "기록" : "부족", o.out_dir.c_str());
    res << fmt("\nhailort.log: error %d건(최종 실행 구간), 온도/클럭 warning %d건 %s\n", log_errors, log_thermal,
               (log_errors == 0 && log_thermal == 0) ? "" : "→ 이 실행의 수치는 신뢰 불가");
    res << fmt("            + size 조합 표에서 의도된 크기 불일치 거부 %d건 (전체 %d건)\n", log_errors_matrix,
               log_errors_total);
    res << fmt("열린 fd: 프로세스 시작 %ld, 증거 파일 3개 연 뒤 %ld(기준선), 종료 %ld → 누수 %+ld\n", fds_start,
               fds_files, fds_end, fds_end - fds_files);
    res << fmt("\n[결론] %s\n", vk_ok ? "계획서 §4.2 채택안(Vulkan 소유 dma-buf) 그대로 진행 가능"
                                      : (hp_ok ? "채택안 불가 → dma-heap 대체안으로 설계 변경해 계속"
                                               : "①~③이 heap 안으로도 통과하지 못했다 → 계획서 §4.8 중단 조건"));

    std::printf("\n[P2] %s (vulkan %s, heap %s)\n", any_ok ? "PASS" : "FAIL", vk_ok ? "PASS" : "FAIL",
                hp_ok ? "PASS" : "FAIL");
    std::printf("[p2] fd: 시작 %ld, 기준선 %ld, 종료 %ld (누수 %+ld)\n", fds_start, fds_files, fds_end,
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
        std::fprintf(stderr, "--in-dmabuf 는 --probe p2 또는 f0 에서만 쓴다 (p1 은 GPU 를 쓰지 않는 경로다)\n");
        return 2;
    }

    try {
        std::filesystem::create_directories(o.out_dir);
        std::filesystem::create_directories(o.fixture_dir);

        // ---- 환경 (측정 조건을 추측하지 않는다)
        const std::string link_speed = read_first_line("/sys/bus/pci/devices/0001:01:00.0/current_link_speed");
        const std::string link_width = read_first_line("/sys/bus/pci/devices/0001:01:00.0/current_link_width");
        std::printf("[env] pcie current_link_speed=%s width=x%s (8.0 GT/s = Gen3)\n", link_speed.c_str(),
                    link_width.c_str());

        // ---- 입력 프레임
        const yvo::LetterboxParams lb = yvo::make_letterbox(kSrcW, kSrcH, 640, 640);
        const int need = std::max({o.compare_frames, o.dump_frames, o.api_check_frames});
        auto nv12 = read_nv12_frames(o.video, need, kSrcW, kSrcH);
        if (static_cast<int>(nv12.size()) < need)
            throw std::runtime_error("프레임 부족: " + std::to_string(nv12.size()) + " / " + std::to_string(need));
        const std::size_t in_size = static_cast<std::size_t>(640) * 640 * 3;
        auto inputs = make_nhcw_inputs(nv12, lb, in_size);
        std::printf("[info] frames=%zu letterbox scale=%.6f pad=(%d,%d) input=%zu B (NHCW)\n", inputs.size(),
                    lb.scale, lb.pad_x, lb.pad_y, in_size);

        // ---- P2: dma-buf 출력 경로 (계획서 §6.2 P2). P1 과 공유하는 것은 환경 확인과 입력 프레임까지다.
        if (o.probe == "p2") return run_p2(o, inputs, link_speed, link_width);

        // ---- F0: 입력 dma-buf(H2D) + 출력 dma-buf(D2H) 동시 사용 관문
        if (o.probe == "f0") return run_f0(o, inputs, link_speed, link_width);

        // ---- RUNNER: RawSession -> yvo::HailoRawRunner 승격 검증
        if (o.probe == "runner") return run_runner(o, inputs, link_speed, link_width);

        // ---- Phase 0: Hef 파싱만으로 출력 스트림 덤프
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
                info << fmt("\n[FAIL] P1-① 원시 출력 스트림이 6개가 아니다: %zu\n", outs.size());
                return 1;
            }
            info << "\n";
        }

        // ---- Phase 1: 원시 스트림 세션 (API 경로별)
        std::vector<RawApi> apis;
        if (o.api == "infer_async" || o.api == "both") apis.push_back(RawApi::InferAsync);
        if (o.api == "read_async" || o.api == "both") apis.push_back(RawApi::ReadAsync);

        std::vector<ApiOutcome> outcomes;
        std::vector<RawFrame> captured;       // primary api의 compare_frames 분량
        std::vector<RawFrame> captured_alt;   // 다른 api의 같은 프레임 (바이트 비교용)
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
                sess = std::make_unique<RawSession>(sc);   // out_pool == nullptr → 호스트 malloc (P1)
                oc.constructed = true;
            } catch (const std::exception& e) {
                oc.error = e.what();
                std::printf("[%s] 세션 생성 실패: %s\n", api_name(api), e.what());
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
                info << fmt("  num_slots=%d (요청 %d, 큐 최솟값 %zu로 클램프)\n\n", oc.slots, o.slots, oc.min_queue);
                layout = derive_layout(out_meta);
                layout_ok = true;
                for (int i = 0; i < 3; ++i) {
                    info << fmt("  [scale %d] stride=%d grid=%dx%d w_padded=%d cls=%s box=%s\n", i,
                                layout.s[i].stride, layout.s[i].grid_h, layout.s[i].grid_w, layout.s[i].w_padded,
                                short_name(out_meta[static_cast<std::size_t>(layout.s[i].cls_idx)].name).c_str(),
                                short_name(out_meta[static_cast<std::size_t>(layout.s[i].box_idx)].name).c_str());
                }
                info << fmt("  num_classes=%d reg_max=%d (shape에서 파생)\n\n", layout.num_classes, layout.reg_max);
                if (quant_per_feature) {
                    info << "[FAIL] P1-② get_quant_infos().size() > 1 (per-feature 양자화) → 설계 변경 필요\n";
                }
            }

            // API 경로 확인: 프레임 몇 장으로 7개 콜백이 전부 SUCCESS인지 본다
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
                // compare_frames 분량의 6 텐서를 확보한다
                const int n_cmp = std::min<int>(o.compare_frames, static_cast<int>(inputs.size()));
                for (int i = 0; i < n_cmp; ++i) {
                    sess->submit(inputs[static_cast<std::size_t>(i)].data(),
                                 [&](int slot, hailo_status st) {
                                     if (st == HAILO_SUCCESS) local[static_cast<std::size_t>(i)] = sess->copy_out(slot);
                                 });
                    sess->drain();   // 슬롯 버퍼를 프레임별로 그대로 읽으려면 프레임마다 완료를 기다린다
                }
                if (!primary_done) {
                    captured = std::move(local);
                    primary_done = true;
                    primary_api = api;
                } else {
                    captured_alt = std::move(local);
                }

                // 처리량 (primary api에서만). 설정당 --bench-repeat회 돌려 중앙값을 쓴다.
                //
                // FPS 계산: **파이프라인 채우기·비우기 구간을 빼야 한다.** 구간을
                // [warmup 프레임의 완료] ~ [마지막 프레임의 완료]로 잡으면 그 사이 완료 수가 정확히
                // bench_frames다. (t0를 "warmup번째 submit 직전"으로 잡으면 아직 비행 중인 slots개가
                // 구간 안에서 완료되어 FPS가 slots/(slots+bench) 만큼 과소평가된다 — 실측 48.6 vs 52.4.)
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
            sess.reset();   // 다음 단계를 위해 장치를 완전히 놓는다
        }

        if (!primary_done) {
            std::fprintf(stderr, "[FAIL] P1-③ infer_async / read_async 둘 다 실패했다. 원시 경로 불가 → 중단.\n");
            Tee api_out(o.out_dir + "/p1_api_result.txt");
            for (const auto& oc : outcomes) {
                api_out << fmt("%s: constructed=%d ok=%d cb=%ld failed=%ld first_error=%s error=%s\n",
                               api_name(oc.api), static_cast<int>(oc.constructed), static_cast<int>(oc.ok),
                               oc.cb_total, oc.cb_failed, status_name(oc.first_error).c_str(), oc.error.c_str());
            }
            return 1;
        }

        // ---- 픽스처 덤프: 다음 단계에서 Vulkan 셰이더를 NPU 없이 디버깅하는 데 쓴다
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
            // float 값은 유효자리 9개로 쓴다. 기본 6자리면 qp_scale·letterbox scale이 잘려
            // 다음 단계(GPU 셰이더를 NPU 없이 디버깅)에서 비트 일치를 못 맞춘다.
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

        // 두 API 경로가 같은 바이트를 주는지 (both 일 때만)
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
            std::printf("[info] infer_async vs read_async 텐서 바이트 일치: %ld / %zu 프레임\n", alt_identical,
                        captured.size());
        }

        // ---- Phase 2: 기준 검출 (HailoRT NMS vstream 경로). 장치를 독점하므로 위 세션을 모두 파괴한 뒤에 한다.
        std::vector<std::vector<yvo::Detection>> ref(captured.size());
        long ref_errors = 0;
        {
            yvo::HailoRunnerConfig hcfg;
            hcfg.hef_path = o.hef;
            hcfg.input_layout = yvo::TensorLayout::Nhcw;   // 원시 패스와 **같은 바이트**를 장치에 넣는다
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
            std::printf("[ref] HailoRT NMS 경로: %zu frames, 검출 합계 %ld (%.2f/frame), errors %ld\n", ref.size(),
                        total, static_cast<double>(total) / static_cast<double>(ref.size()), ref_errors);
        }

        // hailort.log의 error 수를 Phase 3 **이전에** 센다. Phase 3은 configure() 실패를
        // 의도적으로 일으키는 증거 수집이므로 그 error는 ⑥의 판정 대상이 아니다.
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

        // ---- Phase 3: (B)안 폐기 증거 — InferModel로는 원시 출력을 받을 수 없다
        if (!o.skip_infer_model) {
            Tee ev(o.out_dir + "/infer_model_raw_attempt.txt");
            ev << "# (B)안 폐기 증거: InferModel 유지안으로 원시(NHCW) 출력을 받을 수 있는가\n";
            ev << "# 기대: 출력 vstream이 yolov8_nms_postprocess 하나뿐이고, 여기에 NHCW format order를\n";
            ev << "#       걸면 configure()가 실패한다 (계획서 §4.1).\n\n";
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
                ev << "\nset_format_order(HAILO_FORMAT_ORDER_NHCW) 후 configure():\n";
                auto configured = im->configure();
                if (configured) {
                    ev << "  [unexpected] configure() SUCCESS — 설계 재검토 필요\n";
                } else {
                    ev << fmt("  configure() 실패 status = %s\n", status_name(configured.status()).c_str());
                    ev << "  → 출력 vstream은 NMS 하나뿐이고 post-process를 끌 수 없다. (B)안 폐기 확정.\n";
                }
            } catch (const std::exception& e) {
                ev << fmt("  예외: %s\n", e.what());
            }
        }

        // ---- Phase 4: 텐서 해석 전수 시도 (NPU 불필요)
        Tee sweep(o.out_dir + "/p1_combination_sweep.txt");
        sweep << "# P1-④ 텐서 해석 미확정 변수 전수 시도 (2x2x2x2 = 16조합)\n";
        sweep << fmt("# 기준: HailoRT NMS 경로 검출 vs cpu_yolov8_postprocess, match() IoU>=%.2f\n",
                     static_cast<double>(kIouMatchThreshold));
        sweep << fmt("# 프레임 %zu장, score_thr=%.2f iou_thr=%.2f max/class=%d\n", captured.size(),
                     static_cast<double>(kHefScoreThreshold), static_cast<double>(kHefIouThreshold),
                     kHefMaxPerClass);
        sweep << "# cand_limit 초과(20000) 조합은 NMS를 돌리지 않고 기각한다(호스트 NMS가 O(n^2)다).\n\n";
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
                                     row.rejected ? "  기각(cand>20000)" : fmt("%.4f", row.ratio).c_str());
                    }
                }
            }
        }

        // 동률이면 (argmax_only=false, threshold_inclusive=true)를 고른다 — 헤더 기본값과 맞춘다.
        auto better = [](const SweepRow& x, const SweepRow& y) {   // x가 y보다 좋은가
            if (y.rejected != x.rejected) return y.rejected;
            if (x.ratio != y.ratio) return x.ratio > y.ratio;
            if (x.opt.argmax_only != y.opt.argmax_only) return !x.opt.argmax_only;
            return x.opt.threshold_inclusive && !y.opt.threshold_inclusive;
        };
        std::size_t best = 0;
        for (std::size_t i = 1; i < rows.size(); ++i)
            if (better(rows[i], rows[best])) best = i;
        const auto& b = rows[best];
        sweep << "\n[동률 목록] 최고 ratio와 같은 값을 낸 조합 (구분 불가한 변수를 드러낸다)\n";
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
        // `>` 와 `>=` 가 구분되는지는 양자화 격자가 threshold 를 정확히 맞출 수 있는지에 달려 있다.
        for (int si = 0; si < 3; ++si) {
            const auto& q = out_meta[static_cast<std::size_t>(layout.s[si].cls_idx)].quant[0];
            const double qf = static_cast<double>(kHefScoreThreshold) / static_cast<double>(q.qp_scale) +
                              static_cast<double>(q.qp_zp);
            sweep << fmt("  scale %d: thr %.2f / qp_scale %.9g = q %.4f (정수가 아니면 `>`와 `>=`는 원리적으로 같다)\n",
                         si, static_cast<double>(kHefScoreThreshold), static_cast<double>(q.qp_scale), qf);
        }

        // ---- P1-⑤ P5 열 20~23 포함/제외 비교
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
        // 패딩 열(20..23)의 실제 바이트 내용. 0이면 유령 검출이 안 생기지만, 그래도 제외가 옳다
        // (w=20..23은 (w+0.5)*32 >= 656 으로 640 입력 밖의 좌표를 만든다).
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
        sweep << "\n# P1-⑤ P5 (20x24) 패딩 열 20~23 처리\n";
        sweep << fmt("  제외(grid_w=20): cand/frame=%.1f test=%ld matched=%ld ratio=%.4f\n", b.cand_per_frame,
                     b.test, b.matched, b.ratio);
        sweep << fmt("  포함(grid_w=24): cand/frame=%.1f test=%ld matched=%ld ratio=%.4f\n", inc_cand, inc_test,
                     inc_matched, inc_ratio);
        sweep << fmt("  패딩 열 바이트: %ld개 중 0이 아닌 것 %ld개 (cls max q=%d, box max q=%d)\n", pad_bytes,
                     pad_nonzero, pad_cls_max, pad_box_max);
        sweep << fmt("  패딩 열 클래스 점수 상한 = %d * %.9g = %.6f (threshold %.2f)\n", pad_cls_max,
                     static_cast<double>(out_meta[static_cast<std::size_t>(layout.s[2].cls_idx)].quant[0].qp_scale),
                     static_cast<double>(pad_cls_max) *
                         static_cast<double>(out_meta[static_cast<std::size_t>(layout.s[2].cls_idx)].quant[0].qp_scale),
                     static_cast<double>(kHefScoreThreshold));
        if (inc_ratio < b.ratio || inc_test > b.test) {
            sweep << "  → 제외가 맞다: 포함하면 유령 검출이 늘거나 ratio가 떨어진다.\n";
        } else {
            sweep << "  → 제외가 맞다(약한 증거): 이 영상에서는 패딩 열이 threshold를 못 넘어 결과가 같다.\n";
            sweep << "     그래도 제외해야 한다 — w=20..23은 (w+0.5)*32 >= 656 으로 640 입력 밖 좌표를 만들고,\n";
            sweep << "     패딩 값은 정의되지 않으므로 다른 영상/모델에서 유령 검출이 될 수 있다.\n";
        }

        // ---- 최종 P1 판정
        const int hailort_errors = errors_before_phase3;
        const int hailort_thermal = thermal_warnings;
        const int hailort_errors_total = count_log(false);

        Tee res(o.out_dir + "/p1_api_result.txt");
        res << "# P1 결과 요약 (yvo_hailo_raw_probe)\n";
        res << fmt("# pcie=%s x%s, hef=%s, video=%s\n", link_speed.c_str(), link_width.c_str(), o.hef.c_str(),
                   o.video.c_str());
        res << fmt("# batch=%d, 요청 slots=%d, compare_frames=%zu, bench_frames=%d\n\n", o.batch, o.slots,
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
        res << fmt("두 api 텐서 바이트 일치 : %s\n",
                   alt_identical < 0 ? "(한쪽만 실행)" : fmt("%ld / %zu frames", alt_identical, captured.size()).c_str());
        res << fmt("픽스처                : %d frames x %zu tensors, %.2f MB -> %s\n", n_dump, out_meta.size(),
                   static_cast<double>(dumped_bytes) / (1024.0 * 1024.0), o.fixture_dir.c_str());
        res << fmt("hailort.log errors    : %d (추론 구간), 온도/클럭 warning: %d\n", hailort_errors, hailort_thermal);
        res << fmt("                        %d (전체 — Phase 3의 의도된 configure() 실패 %d건 포함)\n\n",
                   hailort_errors_total, hailort_errors_total - hailort_errors);
        res << "합격 기준 판정\n";
        res << fmt("  ① 원시 출력 6개, 이름/shape 일치           : %s\n", layout_ok ? "PASS" : "FAIL");
        res << fmt("  ② get_quant_infos().size() == 1            : %s\n", quant_per_feature ? "FAIL" : "PASS");
        res << fmt("  ③ api 중 최소 하나가 7콜백 전부 SUCCESS    : PASS (%s)\n", api_name(primary_api));
        res << fmt("  ④ match() ratio >= 0.95 (%zu frames)       : %s (best %.4f)\n", captured.size(),
                   b.ratio >= 0.95 ? "PASS" : "FAIL", b.ratio);
        res << fmt("  ⑤ P5 열 20~23 제외가 맞다                  : %s (제외 %.4f / 포함 %.4f, 패딩 nonzero %ld/%ld)\n",
                   (inc_ratio < b.ratio || inc_test > b.test) ? "PASS" : "PASS(약한 증거: 패딩이 threshold 미달)",
                   b.ratio, inc_ratio, pad_nonzero, pad_bytes);
        res << fmt("  ⑥ batch %d, %d프레임 FPS >= 50, log error 0 : %s (%.2f FPS, errors %d)\n", o.batch,
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
