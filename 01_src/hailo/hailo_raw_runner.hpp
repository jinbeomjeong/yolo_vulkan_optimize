// HailoRT **원시 출력 6스트림** 비동기 러너 (yolov11m.hef, NMS 없는 raw 텐서)
//
// hailo_runner.{hpp,cpp} 가 HEF 내장 NMS 출력 1개를 호스트에서 파싱하는 **기준 경로**라면,
// 이쪽은 NMS 앞단의 원시 텐서 6개를 **dma-buf 로 받아 GPU 후처리에 넘기는 프로덕션 경로**다.
// 01_src/tools/hailo/hailo_raw_probe.cpp 의 RawSession(F0 관문 통과, 51.90 FPS)에서 승격했다.
//
// 흐름: Hef::create → VDevice(ROUND_ROBIN) → create_configure_params(batch, 스트림 전부 ASYNC)
//       → VDevice::configure → 슬롯별 bind_slot(dma-buf fd 7개 매핑)
//       → submit()마다 wait_for_ongoing_callbacks_count_under 백프레셔 + infer_async(입력 1 + 출력 6)
//       → 버퍼 콜백 7개가 모두 오면 사용자 콜백 1회.
//
// ── 입출력은 dma-buf 전용이다 ────────────────────────────────────────────────────────────────
// 호스트 포인터 경로는 없다. 통합 파이프라인이 V3D 가 export 한 dma-buf 만 쓰기 때문이다.
// 러너는 Vulkan 을 알지 못한다 — **fd 와 크기만 받는다**(yvo_hailo 는 yvo_vulkan 에 링크하지 않는다).
//
// ── 크기 계약 (F0 가 전수 확인했다: 07_results/gpu_postprocess/evidence/f0_in_dmabuf.txt) ────
//   dma_map_dmabuf() 에는 **dma-buf 실크기**(lseek(fd, 0, SEEK_END) 값)를,
//   infer_async() 에는 **hw_frame_size 만** 넘긴다.
//   전송 크기를 dma-buf 실크기로 주면 HailoRT 는 HAILO_SUCCESS 를 돌려준 뒤 내부 워커에서
//   요청을 조용히 버린다 — **콜백이 하나도 오지 않는다.** 그래서 drain() 에 시한이 필수다.
//
// ── 호출자 규약 ─────────────────────────────────────────────────────────────────────────────
//   · bind_slot() 은 생성 직후 슬롯마다 **한 번만** 부른다. submit() 전에 모든 슬롯이 묶여 있어야 한다.
//   · fd 소유권은 호출자에게 있다. 러너는 close() 하지 않는다. 다만 러너가 살아 있는 동안
//     fd 와 그 dma-buf 는 유효해야 한다(소멸자가 dma_unmap_dmabuf 를 부른다).
//   · on_done 콜백은 **HailoRT 내부 스레드**에서 불린다. 짧게 유지하고, 그 안에서
//     **Vulkan 을 건드리지 마라**(명령 제출·펜스 대기·큐 작업 금지). 콜백 안에서 직접 submit()
//     하지도 마라 — 백프레셔 대기(wait_for_ongoing_callbacks_count_under)를 콜백 스레드에서 하면
//     교착할 수 있다. 슬롯 번호를 free 큐에 넣고 나가는 정도만 한다.
//   · **콜백 진입 시점에 그 슬롯은 이미 반납된 상태다.** 7개 버퍼 콜백이 모두 온 뒤에만 불리므로
//     NPU 는 그 슬롯의 입출력 dma-buf 를 더 이상 만지지 않는다. 콜백 안에서 슬롯을 다른 스레드로
//     넘겨 즉시 재제출해도 경합이 없다.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hailo/hailort.hpp>

#include "hailo/hailo_runner.hpp"   // yvo::HailoError (예외 규약 공유)

namespace yvo {

// 원시 출력 텐서 1개의 메타데이터. 전부 HEF 메타에서만 파생한다(픽스처 파일을 읽지 않는다).
struct RawTensorMeta {
    std::string stream_name;
    std::size_t hw_frame_size = 0;   // 전송 크기. infer_async 에 넘기는 값이다
    int grid_h = 0;                  // 80 / 40 / 20
    int grid_w = 0;                  // 유효 열 수 (shape.width).  80 / 40 / 20
    int w_padded = 0;                // 장치 레이아웃의 열 수 (hw_shape.width). 80 / 40 / 24
    int stride = 0;                  // 8 / 16 / 32
    int features = 0;                // cls 80 / box 64
    float qp_scale = 0.0f;
    float qp_zp = 0.0f;
};

// 러너가 HEF 에서 읽어낸 모델 정보. 호출자(GPU 후처리 글루)가 버퍼 크기·디코드 파라미터를 여기서 얻는다.
struct RawModelInfo {
    std::string input_name;
    std::size_t input_frame_size = 0;   // 1,228,800 (NHCW 640x640x3)
    int input_w = 0, input_h = 0;
    // 번호 규약: cls(s) = 2s, box(s) = 2s+1  (s = 0,1,2 → stride 8,16,32)
    RawTensorMeta t[6];
    int num_classes = 80;
    int reg_max = 16;
    // 입력·출력 7개 스트림의 get_async_max_queue_size() 최솟값. 호출자가 슬롯 수 클램프에 쓴다.
    std::size_t min_queue = 0;
};

// 프레임 1장의 완료 결과. 콜백은 HailoRT 스레드에서 불린다.
struct RawInferResult {
    int slot = -1;
    std::int64_t pts = 0;
    hailo_status status = HAILO_SUCCESS;
    double latency_ms = 0.0;   // submit(infer_async 직전) ~ 7번째 버퍼 콜백
};

using RawInferCallback = std::function<void(RawInferResult&&)>;

struct HailoRawRunnerConfig {
    std::string hef_path = "03_models/yolov11m.hef";
    std::uint16_t batch_size = 8;             // 스케줄러 burst 크기
    int num_slots = 8;                        // 동시에 비행할 수 있는 프레임 수. min_queue 로 클램프된다
    std::chrono::milliseconds timeout{5000};  // drain() 시한 (무한 대기 금지)
    bool verbose = false;
};

class HailoRawRunner {
public:
    explicit HailoRawRunner(const HailoRawRunnerConfig& cfg);
    ~HailoRawRunner();   // drain(시한) → dma_unmap_dmabuf(방향별) → ng_ → vdevice_ 순서

    HailoRawRunner(const HailoRawRunner&) = delete;
    HailoRawRunner& operator=(const HailoRawRunner&) = delete;

    const RawModelInfo& info() const { return info_; }

    // 실제 슬롯 수. cfg.num_slots 를 [1, info().min_queue] 로 클램프한 값이다.
    // 호출자는 이 개수만큼 bind_slot() 해야 한다.
    int num_slots() const { return num_slots_; }

    // 슬롯별 dma-buf 등록. 생성 직후 슬롯마다 **한 번만** 부른다.
    //   in_fd           : 전처리 출력(V3D export). H2D 로 매핑한다
    //   in_map_size     : dma-buf 실크기 (lseek 값). **전송 크기가 아니다**
    //   out_fd[6]       : 원시 텐서 (V3D export). D2H 로 매핑한다. 번호는 info().t[] 와 같은 순서
    //   out_map_size[6] : 각 dma-buf 실크기
    // fd 소유권은 호출자에게 있다. 러너는 close 하지 않는다.
    // 실패(범위 밖 슬롯, 중복 bind, 크기 부족, dma_map_dmabuf 실패)하면 예외를 던진다.
    void bind_slot(int slot, int in_fd, std::size_t in_map_size,
                   const int out_fd[6], const std::size_t out_map_size[6]);

    // 프레임 1장 제출. 슬롯은 호출자가 고른다(전처리 슬롯과 1:1).
    // 백프레셔 내장: wait_for_ongoing_callbacks_count_under(num_slots) 로 블록한다.
    // 비행 중 프레임 수의 상한은 **정확히 num_slots** 다(HailoRT 4.24.0 실측 — .cpp 의 같은 자리 주석).
    // on_done 은 HailoRT 콜백 스레드에서 불린다 — **짧게 유지하고 그 안에서 Vulkan 을 건드리지 마라.**
    // 콜백 진입 시점에 슬롯은 이미 반납돼 있다(다른 스레드가 바로 재제출해도 된다).
    // 아직 비행 중인 슬롯을 다시 제출하면 HailoError(HAILO_INVALID_OPERATION).
    void submit(int slot, std::int64_t pts, RawInferCallback on_done);

    // 비행 중인 모든 프레임의 사용자 콜백이 반환할 때까지 대기한다.
    // cfg.timeout 안에 끝나지 않으면 HailoError(HAILO_TIMEOUT) — **무한 대기하지 않는다.**
    void drain();

    // 첫 물리 장치의 칩 온도(ts0/ts1 중 큰 값, °C). 실패하면 NaN.
    float chip_temperature() const;

    // 비행 중(제출됐고 사용자 콜백이 아직 반환하지 않은) 프레임 수
    int in_flight() const { return in_flight_.load(); }

private:
    struct Slot {
        bool bound = false;
        int in_fd = -1;
        std::size_t in_map_size = 0;
        int out_fd[6] = {-1, -1, -1, -1, -1, -1};
        std::size_t out_map_size[6] = {0, 0, 0, 0, 0, 0};
        std::atomic<bool> busy{false};
        std::atomic<int> remaining{0};
        std::atomic<hailo_status> status{HAILO_SUCCESS};
        std::int64_t pts = 0;
        std::chrono::steady_clock::time_point t_submit;
        RawInferCallback on_done;
    };

    struct Mapping {
        int fd;
        std::size_t size;
        hailo_dma_buffer_direction_t dir;
    };

    void note_callback(int slot, hailo_status st);
    void finish(int slot);
    bool drain_for(std::chrono::milliseconds timeout);

    HailoRawRunnerConfig cfg_;
    RawModelInfo info_;
    std::unique_ptr<hailort::Hef> hef_;
    std::unique_ptr<hailort::VDevice> vdevice_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> ng_;
    hailort::InputStream* input_ = nullptr;
    // 이름순으로 정렬된 출력 스트림. 텐서 번호 i 와의 대응은 info_.t[i].stream_name 이 쥔다.
    std::vector<hailort::OutputStream*> outputs_;

    int num_slots_ = 1;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<Mapping> mapped_;

    std::atomic<int> in_flight_{0};
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace yvo
