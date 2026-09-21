// HailoRT InferModel 비동기 추론 러너 (yolov11m.hef, NMS BY CLASS 출력)
//
// 흐름: VDevice(ROUND_ROBIN 스케줄러) → create_infer_model(hef) → set_batch_size → configure()
//       → 슬롯별 Bindings(출력 버퍼는 러너 소유) → submit()마다 wait_for_async_ready + run_async
//       → 완료 콜백(HailoRT 내부 스레드)에서 NMS 파싱 + letterbox 역변환 → 사용자 콜백.
//
// 입력 레이아웃(HailoRunnerConfig::input_layout):
//   - Nhwc(기본): packed RGB888을 넘기면 HailoRT가 CPU에서 NHWC→NHCW로 전치한다(약 14 ms/frame).
//   - Nhcw: 장치 입력 형식과 같아 CPU 변환이 사라진다. 대신 HailoRT가 입력 버퍼를 직접 DMA 매핑하므로
//     dma-buf 입력에서만 쓸 수 있다. V3D GEM mmap 포인터는 매핑에 실패한다
//     (Ioctl HAILO_VDMA_BUFFER_MAP failed due to invalid address).
//     일반 malloc 버퍼로 시험할 때만 allow_host_ptr_nhcw로 검사를 풀 수 있다.
//
// 슬롯 규약: PreprocOutput.slot(0..num_slots-1)을 그대로 러너 슬롯으로 쓴다.
//   - submit()한 슬롯은 사용자 콜백이 호출될 때까지 "사용 중"이다.
//   - 사용자 콜백이 호출되면 그 슬롯의 입력 버퍼(host_ptr 또는 dma-buf)는 HailoRT가 더 이상 읽지 않는다.
//     즉 콜백 = 슬롯 반납 시점이다. 콜백 반환 후 같은 슬롯을 다시 submit()할 수 있다.
//   - 사용 중인 슬롯을 다시 submit()하면 HAILO_INVALID_OPERATION 예외.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <hailo/hailort.hpp>

#include "core/frame_types.hpp"

namespace yvo {

// HailoRT 호출 실패 예외. what()에 호출 위치와 상태 코드 이름이 들어간다.
class HailoError : public std::runtime_error {
public:
    HailoError(const std::string& where, hailo_status status);
    hailo_status status() const { return status_; }

private:
    hailo_status status_;
};

struct HailoRunnerConfig {
    std::string hef_path;  // 호출자가 지정한다(기본 경로는 yvo_assets 의 YVO_DEFAULT_HEF — 라이브러리는 자산 경로를 모른다)
    // 러너가 받을 입력 텐서 레이아웃. Nhcw면 set_format_order(HAILO_FORMAT_ORDER_NHCW)로
    // HailoRT의 CPU 입력 변환을 없앤다(dma-buf 입력 전용).
    TensorLayout input_layout = TensorLayout::Nhwc;
    // Nhcw인데도 host_ptr 제출을 허용한다(일반 malloc 버퍼 실험용). V3D GEM 포인터에는 쓰면 안 된다.
    bool allow_host_ptr_nhcw = false;
    std::uint16_t batch_size = 1;      // 스케줄러 burst 크기(1/4/8 권장)
    int num_slots = kNumSlots;         // 동시에 비행 중일 수 있는 최대 프레임 수. batch_size 이상이어야 효율적
    float score_threshold = -1.0f;     // <0이면 HEF 값(0.2) 유지. >=0이면 set_nms_score_threshold로 설정
    int src_width = 0;                 // 역변환 좌표 클램프용 원본 크기. 0이면 LetterboxParams에서 추정
    int src_height = 0;
    std::chrono::milliseconds timeout{2000}; // wait_for_async_ready 타임아웃
};

// 완료 결과. 콜백은 HailoRT 콜백 스레드에서 호출되므로 무거운 작업을 하지 말 것.
struct InferResult {
    int slot = -1;
    std::int64_t pts = 0;
    hailo_status status = HAILO_SUCCESS;
    std::vector<Detection> detections;  // 원본 프레임 픽셀 좌표
    double latency_ms = 0.0;            // submit(run_async 직전) ~ 완료 콜백 (큐 대기 포함)
};

using InferCallback = std::function<void(InferResult&&)>;

// NMS BY CLASS float 버퍼 파싱 + letterbox 역변환 (GTest에서 직접 쓸 수 있게 공개)
// 레이아웃(hailort.h HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS):
//   클래스 c = 0..num_classes-1 순서로 [float bbox_count, hailo_bbox_float32_t × bbox_count]
//   hailo_bbox_float32_t = {y_min, x_min, y_max, x_max, score}, 좌표는 입력(dst_w×dst_h) 기준 0~1 정규화.
// 원본 좌표 = (정규화 × dst - pad) / scale, [0, src_w]×[0, src_h]로 클램프.
std::vector<Detection> parse_nms_by_class(const float* data, std::size_t bytes, int num_classes,
                                          int max_bboxes_per_class, const LetterboxParams& lb,
                                          int src_w, int src_h, float score_threshold);

class HailoRunner {
public:
    explicit HailoRunner(const HailoRunnerConfig& cfg);
    ~HailoRunner();  // 비행 중 작업을 모두 기다린 뒤 shutdown, dma-buf 매핑 해제

    HailoRunner(const HailoRunner&) = delete;
    HailoRunner& operator=(const HailoRunner&) = delete;

    // 입력 1프레임 제출. in.slot으로 슬롯 지정.
    //  - in.dmabuf_fd >= 0: set_dma_buffer({fd, size}) (2단계, 미리 map_dmabuf 권장)
    //  - 그 외: set_buffer(MemoryView(host_ptr, size)) (1단계). host_ptr는 4KB 정렬 권장.
    // 파이프라인이 가득 차면 wait_for_async_ready로 블록(백프레셔). 타임아웃이면 HailoError(HAILO_TIMEOUT).
    void submit(const PreprocOutput& in, InferCallback cb);

    // 비행 중인 모든 작업의 완료(사용자 콜백 반환)까지 대기
    void drain();

    // 2단계용: dma-buf를 H2D로 미리 매핑(프레임마다 매핑 비용 제거). 소멸자에서 자동 해제.
    void map_dmabuf(int fd, std::size_t size);

    // 모델 정보
    std::size_t input_frame_size() const { return input_frame_size_; }
    std::size_t output_frame_size() const { return output_frame_size_; }
    int input_width() const { return in_w_; }
    int input_height() const { return in_h_; }
    int num_classes() const { return num_classes_; }
    int max_bboxes_per_class() const { return max_bboxes_per_class_; }
    int num_slots() const { return cfg_.num_slots; }
    std::size_t async_queue_size() const { return async_queue_size_; }
    int in_flight() const { return in_flight_.load(); }

    // 첫 물리 장치의 칩 온도(ts0/ts1 중 큰 값, °C). 실패하면 NaN.
    float chip_temperature() const;

private:
    struct Slot {
        hailort::ConfiguredInferModel::Bindings bindings;
        std::shared_ptr<std::uint8_t> out_buf;  // 페이지 정렬, 러너 소유
        std::atomic<bool> busy{false};
        std::chrono::steady_clock::time_point t_submit;
    };

    HailoRunnerConfig cfg_;
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::InferModel> infer_model_;
    std::unique_ptr<hailort::ConfiguredInferModel> configured_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::string input_name_;
    std::string output_name_;
    std::size_t input_frame_size_ = 0;
    std::size_t output_frame_size_ = 0;
    std::size_t async_queue_size_ = 0;
    int in_w_ = 0, in_h_ = 0;
    int num_classes_ = 0;
    int max_bboxes_per_class_ = 0;
    float score_threshold_ = 0.0f;

    std::atomic<int> in_flight_{0};
    std::mutex mtx_;
    std::condition_variable cv_;

    std::mutex map_mtx_;
    std::vector<std::pair<int, std::size_t>> mapped_dmabufs_;
};

} // namespace yvo
