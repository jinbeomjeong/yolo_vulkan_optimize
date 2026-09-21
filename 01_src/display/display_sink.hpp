// DisplaySink: 파이프라인이 보는 표시 계층의 공개 API (순수 가상 인터페이스).
//
// 파이프라인(main.cpp)은 이 인터페이스만 쥔다. 구현은 둘을 염두에 둔다.
//   · `DisplayPipeline` — V3D 합성(display_convert.comp + overlay_boxes.comp) → Vulkan 스왑체인/오프스크린.
//   · `PlaneDisplay` (plane_display.hpp) — 평면 스캔아웃. 디코더 NV12 SAND dma-buf 를 HVS 영상 평면에, GPU 로
//     그린 박스(OverlayPlane, AB24 dma-buf)를 HVS 오버레이 평면에 올린다(Wayland linux-dmabuf, 다음 단계 KMS atomic).
//
// `DisplayBackend`(offscreen_backend.hpp)는 "그릴 VkImage 를 내주는" 한 단계 아래 인터페이스라 평면
// 스캔아웃이 맞지 않는다. 그래서 다형성의 이음새를 `show()` 수준에 둔다.
//
// ── 계약 (구현이 지켜야 하는 것) ───────────────────────────────────────────────────────────────
//   · 슬롯 인자는 전부 **파이프라인 슬롯**(0..num_slots-1)이다. 구현 내부의 표시 슬롯과 섞지 않는다.
//   · `show()` 가 false 를 돌려주면(표시 드롭) 호출자가 즉시 display_done(slot) 을 세운다.
//     true 면 구현이 나중에 `pump()`/`drain()` 안에서 콜백으로 display_done(slot) 을 부른다.
//   · 전부 메인 스레드에서 부른다.
//   · Vulkan 합성 전용 통계(convert_ms, uniform_scale, blit_mode, image_count, recreates …)는 기본 구현이
//     0/빈 값을 돌려준다. 해당 개념이 없는 구현은 덮어쓰지 않는다.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"

namespace yvo::display {

class DisplaySink {
public:
    using Clock = std::chrono::steady_clock;

    virtual ~DisplaySink() = default;
    DisplaySink() = default;
    DisplaySink(const DisplaySink&) = delete;
    DisplaySink& operator=(const DisplaySink&) = delete;

    // 표시가 끝나 파이프라인 슬롯을 놓아도 된다고 알리는 콜백.
    virtual void set_display_done_callback(std::function<void(int)> cb) = 0;

    // 원본 프레임 크기. 디코더를 연 뒤 첫 show() 전에 부른다.
    virtual void set_source_size(int w, int h) = 0;

    // 디코드 직후: 이 슬롯이 쓸 프레임의 keepalive 를 추가로 붙든다. glass-to-glass 시작점.
    virtual void retain(int slot, const DrmFrame& f) = 0;

    // 한 프레임을 낸다. false = 표시 드롭(호출자가 즉시 display_done).
    virtual bool show(int slot, const std::vector<Detection>& dets, std::string_view hud = {}) = 0;

    // 비차단 완료 회수 → display_done 콜백. 매 바퀴 부른다.
    virtual void pump() = 0;

    // 남은 표시 작업을 전부 끝낸다(블로킹). 종료 때 한 번.
    virtual void drain() = 0;

    // 창/장치 이벤트 처리. false = 창이 닫혔거나 종료 키. 창이 없으면 항상 true.
    virtual bool poll_window() = 0;

    // 종료 키(Esc, q, Ctrl+C)가 눌렸다.
    virtual bool quit_requested() const = 0;

    // 이 경로가 지금 붙들고 있는 디코더 프레임 수.
    virtual int frames_held() const = 0;

    // pace 에 따른 이 프레임의 처리 마감 시각. 페이싱이 없으면 빈 optional.
    virtual std::optional<Clock::time_point> pace_deadline(std::int64_t frame_index) = 0;

    // ── 통계 (공통) ──
    virtual const char* backend_name() const = 0;         // 시작 줄·CSV 의 display_backend 값
    virtual std::uint64_t presented() const = 0;
    virtual std::uint64_t display_dropped() const = 0;
    virtual std::uint64_t overlay_overflows() const = 0;
    virtual VkExtent2D extent() const = 0;                // 화면(또는 출력) 크기
    virtual const std::vector<double>& overlay_ms() const = 0;
    virtual const std::vector<double>& present_interval_ms() const = 0;
    virtual const std::vector<double>& glass_ms() const = 0;

    // ── 통계 (Vulkan 합성 경로 전용 — 기본값 0/빈 값) ──
    virtual std::uint64_t recreates() const { return 0; }
    virtual std::uint32_t image_count() const { return 0; }
    virtual bool offscreen() const { return false; }
    virtual bool uniform_scale() const { return false; }
    virtual bool blit_mode() const { return false; }
    virtual const std::vector<double>& convert_ms() const { return empty_series(); }
    virtual const std::vector<double>& convert_total_ms() const { return empty_series(); }

    // ── 평면 스캔아웃 전용 (기본값: 해당 없음) ──
    // 시작 줄 `[display] backend=...` 뒤에 붙일 설명. 비어 있으면 호출자가 Vulkan 경로 형식으로 찍는다.
    virtual std::string describe() const { return {}; }
    struct ScanoutReport {
        bool applicable = false;           // false = 이 경로에는 해당 없음 (CSV 에 "-")
        std::string zero_copy = "-";       // debugfs 판정: "1" / "0" / "unverified"
        std::string color_ok = "-";        // 영상 평면 COLOR_ENCODING BT.709 && COLOR_RANGE limited
        long long underrun_delta = -1;     // hvs_underrun 끝 - 시작 (-1 = 읽지 못함)
        std::uint64_t overlay_skipped = 0; // 박스 링이 비어(전부 GPU/컴포지터 보유) 박스 갱신을 건너뛴 수
        std::uint64_t frame_cb_timeouts = 0; // wl_surface.frame 콜백을 기다리다 타임아웃해 그냥 커밋한 수
        std::uint64_t zero_copy_feedback = 0; // wp_presentation ZERO_COPY 플래그가 선 표시 수
        std::uint64_t feedback_presented = 0; // presented 피드백 수
        std::uint64_t busy_reuse = 0;      // 컴포지터가 아직 쥔 버퍼를 다시 붙이려 한 수 (0 이어야 한다)
        std::vector<std::string> lines;    // 요약에 그대로 찍을 판정 줄 (debugfs 발췌)
    };
    virtual ScanoutReport scanout_report() const { return {}; }

protected:
    static const std::vector<double>& empty_series() {
        static const std::vector<double> kEmpty;
        return kEmpty;
    }
};

} // namespace yvo::display
