// GpuPostprocStage: 원시 출력 6스트림(NPU) → GPU pass 1 → CPU greedy 억제.
//
// 호스트 NMS(HailoRT 가 CPU 에서 하는 2.2~3.2 ms/frame 고정비)를
//   GPU pass 1 (`02_shaders/yolo_decode.comp`, 실측 0.56 ms) + CPU 억제(후보 수십 개, 3~11 µs)
// 로 바꾼다. 두 부품(HailoRawRunner, YoloPostproc)은 각각 검증이 끝났고 이 파일은 **글루뿐**이다.
//
// ── 단계 흐름 ─────────────────────────────────────────────────────────────────────────────────
//   submit(out, cb)  [메인]  pts_/lb_ 기록 → raw_->submit(slot, pts, 내부콜백)
//   내부콜백  [HailoRT 스레드]  {slot,status,latency} 를 ready_ 에 push. **그게 전부다.**
//                              (러너 규약: 콜백에서 Vulkan 호출·재submit·GPU 메모리 대량 읽기 금지)
//   pump()           [메인]  ready_ 비움 → pp_->submit(slot, lb, src_w, src_h) → pp_q_ push
//                            pp_q_ 앞이 ready 면 → wait → read_candidates → cpu_nms → Done
//   drain_one()      [메인]  pp_q_ 앞을 **블로킹** wait 로 하나 끝낸다
//
// ── 소멸 순서 ─────────────────────────────────────────────────────────────────────────────────
// 러너가 dma-buf fd 를 매핑해 두므로 **러너가 먼저 죽어야 한다.**
// 멤버 선언 순서는 pp_ → raw_ 이고 파괴는 역순이라 raw_ 가 먼저다.
// 입력 fd 는 호출자(main.cpp 의 FdCloser)가, 출력 fd 는 YoloPostproc 이 소유한다.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hailo/hailo_raw_runner.hpp"
#include "pipeline/infer_stage.hpp"
#include "vulkan/yolo_postproc.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::pipeline {

struct GpuPostprocStageConfig {
    std::string hef_path;  // 호출자가 지정한다(기본 경로는 yvo_assets 의 YVO_DEFAULT_HEF — 라이브러리는 자산 경로를 모른다)
    std::uint16_t batch_size = 8;
    int num_slots = 8;
    float score_thr = 0.2f;    // HEF YOLOV8 post-process 와 같은 값
    float iou_thr = 0.7f;      // HEF 설정
    int max_per_class = 100;   // HEF 설정
    int max_dets = 4096;
    std::uint32_t local_size_x = 10;   // yolo_postproc.hpp 의 실측 채택값 10x6
    std::uint32_t local_size_y = 6;
    yvo::vk::YoloInputAlloc input_alloc = yvo::vk::YoloInputAlloc::ExportDmaBuf;
    std::chrono::milliseconds timeout{5000};
    bool verbose = false;
};

class GpuPostprocStage final : public InferStage {
public:
    GpuPostprocStage(yvo::vk::VulkanContext& ctx, const GpuPostprocStageConfig& cfg);
    ~GpuPostprocStage() override;

    GpuPostprocStage(const GpuPostprocStage&) = delete;
    GpuPostprocStage& operator=(const GpuPostprocStage&) = delete;

    std::size_t input_frame_size() const override { return raw_->info().input_frame_size; }
    void bind_input_dmabuf(int slot, int fd) override;
    void set_source_size(int w, int h) override;
    void submit(const PreprocOutput& in, const Done& on_done) override;
    void pump() override;
    bool drain_one() override;
    bool wait_completion(std::chrono::milliseconds timeout) override;
    void drain() override;
    float chip_temperature() const override { return raw_->chip_temperature(); }

    double last_post_ms() const override { return last_post_ms_; }
    double last_nms_us() const override { return last_nms_us_; }

    // ── RESULT_CSV2 용 누적 통계 (메인 스레드에서만 갱신·조회한다) ──
    const std::vector<double>& post_ms() const { return post_ms_; }
    const std::vector<double>& nms_us() const { return nms_us_; }
    const std::vector<double>& cand() const { return cand_; }
    // NPU 완료 콜백 도착 → 슬롯 반납까지. 이 값이 크면 슬롯이 후처리 대기열에 묶여 NPU 점유가 떨어진다.
    const std::vector<double>& hold_ms() const { return hold_ms_; }
    long overflow_frames() const { return overflow_frames_; }
    const char* alloc_name() const;

private:
    struct RawDone {
        int slot = -1;
        hailo_status status = HAILO_SUCCESS;
        double latency_ms = 0.0;
        std::chrono::steady_clock::time_point t_done;   // 콜백 도착 시각
    };

    void finish(int slot);   // pp_->wait() 가 끝난 슬롯을 CPU 억제 → Done
    void push_ready_to_gpu();   // ready_ 를 비워 pp_ 로 넘긴다

    // 선언 순서 = 생성 순서. 파괴는 역순이므로 raw_ 가 pp_ 보다 먼저 죽는다(러너가 fd 매핑을 쥔다).
    GpuPostprocStageConfig cfg_;
    std::unique_ptr<yvo::vk::YoloPostproc> pp_;
    std::unique_ptr<yvo::HailoRawRunner> raw_;

    int src_w_ = 0, src_h_ = 0;
    std::vector<std::int64_t> pts_;
    std::vector<yvo::LetterboxParams> lb_;
    std::vector<bool> bound_;
    Done done_;

    std::mutex ready_mtx_;
    std::condition_variable ready_cv_;  // HailoRT 콜백 → 메인 스레드 깨우기
    std::vector<RawDone> ready_;        // HailoRT 콜백 스레드 → 메인 스레드
    std::vector<double> latency_;       // 슬롯별 NPU latency (finish 에서 읽는다)
    std::vector<hailo_status> status_;  // 슬롯별 NPU 상태
    std::deque<int> pp_q_;              // GPU pass 1 에 걸린 슬롯 (submit 순서)

    std::vector<double> post_ms_, nms_us_, cand_, hold_ms_;
    std::vector<std::chrono::steady_clock::time_point> t_done_;   // 슬롯별 콜백 도착 시각
    long overflow_frames_ = 0;
    double last_post_ms_ = -1.0, last_nms_us_ = -1.0;
};

} // namespace yvo::pipeline
