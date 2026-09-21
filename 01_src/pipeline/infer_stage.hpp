// InferStage: main.cpp 가 보는 "프레임 1장 → 검출 목록" 추상화.
//
// 두 구현이 있다.
//   HostNmsStage     : 오늘의 기준 경로. `yvo::HailoRunner`(HEF 내장 NMS 출력 1개)를 그대로 감싼다.
//                      후처리 전부(NMS 파싱 + letterbox 역변환)를 HailoRT 가 **호스트 CPU** 에서 한다.
//   GpuPostprocStage : `yvo::HailoRawRunner`(원시 출력 6스트림) + `yvo::vk::YoloPostproc`(GPU pass 1)
//                      + `cpu_nms_greedy_by_class`(CPU 억제). gpu_postproc_stage.{hpp,cpp} 에 있다.
//
// ── 스레드 규율 (이것이 전체 설계를 정한다) ──────────────────────────────────────────────────
// `VkQueue` 는 외부 동기화 객체이고 이 프로젝트에는 큐가 하나뿐이다(`VulkanContext::queue()`).
// `SandLetterbox::submit` 과 `YoloPostproc::submit` 이 둘 다 `vkQueueSubmit` 을 한다 →
// **Vulkan 을 만지는 일은 전부 메인 스레드에서만 한다. 새 스레드를 만들지 않는다.**
//   submit()   : 메인 스레드
//   pump()     : 메인 스레드. NPU 완료분을 GPU 후처리로 밀고, 끝난 것을 `Done` 으로 돌려준다
//   drain_one(): 메인 스레드. **블로킹**으로 하나만 끝낸다(빈 슬롯이 하나도 없을 때의 탈출구)
//   Done 콜백  : host 는 HailoRT 콜백 스레드, gpu 는 메인 스레드(pump/drain 안)에서 불린다.
//                어느 쪽이든 `SlotExchange::complete()` 로 들어가므로 main.cpp 는 구분하지 않는다.
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "core/frame_types.hpp"
#include "hailo/hailo_runner.hpp"

namespace yvo::pipeline {

class InferStage {
public:
    using Done = std::function<void(yvo::InferResult&&)>;

    virtual ~InferStage() = default;

    virtual std::size_t input_frame_size() const = 0;

    // dmabuf 모드에서 슬롯의 전처리 출력 dma-buf 를 단계에 등록한다.
    // 첫 submit() 전에 슬롯마다 한 번씩 부른다. ptr 모드에서는 부르지 않는다.
    virtual void bind_input_dmabuf(int slot, int fd) = 0;

    // 원본 프레임 크기. 디코더를 연 뒤 첫 submit() 전에 부른다(좌표 클램프용).
    virtual void set_source_size(int /*w*/, int /*h*/) {}

    virtual void submit(const PreprocOutput& in, const Done& on_done) = 0;

    // 메인 스레드. 블로킹하지 않는다. host 는 no-op.
    virtual void pump() {}

    // 메인 스레드. 진행 중인 GPU 후처리가 있으면 **블로킹**으로 하나 끝내고 true.
    // 없으면 아무것도 하지 않고 false. host 는 항상 false.
    virtual bool drain_one() { return false; }

    // 메인 스레드. 슬롯이 전부 NPU 에 있어 할 일이 아무것도 없을 때의 마지막 대기다.
    // gpu 경로에서는 **`SlotExchange` 의 condvar 를 아무도 깨우지 않는다**(완료 알림은 단계 내부의
    // ready_ 로 들어가고, 슬롯 반납은 메인 스레드가 pump() 할 때 비로소 일어난다). 그래서 여기서
    // 기다려야 한다. 완료가 하나라도 도착했으면 true — 호출자는 다시 pump()/drain_one() 한다.
    // host 는 HailoRT 콜백이 직접 `SlotExchange` 를 깨우므로 false(= 기존 경로 그대로).
    virtual bool wait_completion(std::chrono::milliseconds) { return false; }

    virtual void drain() = 0;

    virtual float chip_temperature() const = 0;

    virtual double last_post_ms() const { return -1.0; }   // GPU pass 1 ms
    virtual double last_nms_us() const { return -1.0; }    // CPU 억제 µs
};

// ── 기준 경로: HEF 내장 NMS + HailoRT 호스트 후처리 ──────────────────────────────────────────
// 오늘의 main.cpp 와 **완전히 같은 호출 순서**를 유지한다(회귀 기준값 생성기다).
class HostNmsStage final : public InferStage {
public:
    explicit HostNmsStage(const yvo::HailoRunnerConfig& cfg)
        : runner_(std::make_unique<yvo::HailoRunner>(cfg)) {}

    std::size_t input_frame_size() const override { return runner_->input_frame_size(); }

    void bind_input_dmabuf(int /*slot*/, int fd) override {
        runner_->map_dmabuf(fd, runner_->input_frame_size());
    }

    void submit(const PreprocOutput& in, const Done& on_done) override { runner_->submit(in, on_done); }

    void drain() override { runner_->drain(); }

    float chip_temperature() const override { return runner_->chip_temperature(); }

private:
    std::unique_ptr<yvo::HailoRunner> runner_;
};

} // namespace yvo::pipeline
