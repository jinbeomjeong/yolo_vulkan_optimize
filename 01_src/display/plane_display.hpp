// PlaneDisplay: HVS 두 평면 스캔아웃 표시 (DisplaySink 의 두 번째 구현).
//
//   영상 : 디코더 NV12 SAND128 dma-buf 를 **그대로** HVS 영상 평면에 올린다(V3D 변환 0).
//   박스 : OverlayPlane 이 GPU(overlay_boxes_buf.spv)로 그린 AB24 LINEAR dma-buf 를 영상 위 평면에 올린다.
//   HVS 가 스캔아웃 중에 YUV→RGB 변환과 블렌딩을 한다. 계획서 atomic-munching-kite §2.
//
// 전송(WaylandPlaneScanout = Wayland linux-dmabuf / KmsScanout = KMS atomic)은 `PlaneScanout` 인터페이스 뒤에 있다. 이 클래스는 전송과
// 무관한 공통부 — 프레임 수명, 박스 링, 커밋 게이트(페이싱), 통계, debugfs 검증 — 를 맡는다.
//
// ── 프레임 수명 (display_done 을 언제 세우는가) ─────────────────────────────────────────────────
//   retain(slot)  : 디코드 직후 keepalive 를 붙든다 (glass-to-glass 시작점)
//   show(slot)    : retained → pending (아직 커밋하지 않음). 박스를 그린다(acquire → draw, 비차단).
//   커밋          : pending → committed (전송층이 스캔아웃에 쥔다). 전송층 token 으로 구분한다.
//   release       : 전송층이 그 버퍼를 놓았다(wl_buffer.release / 다음 flip) → keepalive 를 놓고 display_done(slot).
//   DisplayPipeline 은 submit 펜스에서 display_done 을 세웠지만, 여기서는 **HVS 가 그 버퍼를 스캔아웃하는 동안**
//   디코더가 같은 버퍼에 다음 프레임을 쓰면 안 되므로 release 까지 쥔다(1~2 vsync 더 길다).
//   보유 상한: pending kMaxPending(2) + committed(미해제) kMaxCommitted(3: 화면 1 + 다음 flip 대기 1 + release 지연 1).
//   pending 은 vblank 마다 하나씩 커밋되는 FIFO 다. 넘치면 **가장 오래된** pending 을 버린다(드롭).
//
// ── 페이싱 (절대 영원히 막히지 않는다) ──────────────────────────────────────────────────────────
//   pace(source/asap) 의미는 DisplayPipeline 과 같다(pace_deadline). pace=source 면 커밋 간격도 소스 간격 - 8 ms
//   이상으로 벌린다(matched + batch 의 몰림을 고르게 편다). 그 위에 vblank 게이트가 하나 더 있다:
//   커밋마다 frame 콜백(wl_surface.frame / page flip)을 요청하고, 그것이 오기 전에는 다음 커밋을 미룬다.
//   **콜백이 kFrameCallbackTimeout 안에 안 오면 그냥 커밋한다**(frame_cb_timeouts 에 센다). 커밋은 비차단
//   요청이라 컴포지터가 멈춰 있어도 이 프로세스는 멈추지 않는다. pump()/show() 는 어떤 경우에도 블로킹하지 않고,
//   drain() 만 유한 시간(kDrainTimeoutMs) 기다린다.
//   (기존 Vulkan 경로의 결함 — 컴포지터가 frame 콜백을 주지 않으면 FIFO vkQueuePresentKHR 가 Mesa WSI 안에서
//    무기한 대기해 SIGTERM 도 안 먹는다 — 의 원인은 드라이버 안 무한 대기이고, 이 경로에는 그런 호출이 없다.)
//
// ── 박스 ────────────────────────────────────────────────────────────────────────────────────────
//   show() 에서 OverlayPlane::acquire → draw(비차단 submit). 커밋은 ready() 가 true 가 된 뒤에만 한다
//   (dma-buf 암묵적 펜스에 기대지 않는다). acquire 가 -1 이면(링 3개가 전부 GPU/컴포지터 보유) 박스 갱신을
//   건너뛰고 직전 박스를 유지한다(overlay_skipped). 박스 wl_buffer.release → OverlayPlane::release.
//
// ── 통계 ────────────────────────────────────────────────────────────────────────────────────────
//   presented       : 전송층이 표시했다고 알린 프레임 수(Wayland: wp_presentation.presented, 없으면 커밋 수)
//   display_dropped : every 로 건너뜀 + pending 넘침 + 전송층 거부 + 종료 때 못 낸 것
//   overlay_ms      : OverlayPlane 의 last_overlay_ms + last_clear_ms (V3D timestamp)
//   present_interval_ms / glass_ms : presented 시각(CLOCK_MONOTONIC) 기준. glass = retain() → presented.
//   convert_ms      : 없다(변환이 HVS 에서 일어난다) — 빈 값.
//
// 스레드: 전부 메인 스레드.
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "display/display_pipeline.hpp"   // DisplayPipelineConfig 와 enum 들
#include "display/display_sink.hpp"
#include "display/drm_state.hpp"
#include "display/overlay_plane.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

// 전송층 → PlaneDisplay 이벤트. 전부 PlaneScanout::dispatch() 안에서(메인 스레드) 온다.
class PlaneScanoutListener {
public:
    virtual ~PlaneScanoutListener() = default;
    virtual void on_video_released(std::uint64_t token) = 0;          // 그 커밋의 영상 버퍼를 놓았다
    virtual void on_overlay_released(int overlay_idx) = 0;            // 박스 링 항목을 놓았다
    virtual void on_frame_done() = 0;                                 // 다음 커밋을 해도 된다 (vblank)
    // 표시됐다. t_ns = CLOCK_MONOTONIC 기준 ns (clock_monotonic=false 면 다른 시계라 glass 에 쓰지 않는다).
    virtual void on_presented(std::uint64_t token, std::int64_t t_ns, bool clock_monotonic, bool zero_copy) = 0;
    virtual void on_discarded(std::uint64_t token) = 0;               // 화면에 나가지 않고 대체됐다
};

// 평면 스캔아웃 전송층 (구현: WaylandPlaneScanout, KmsScanout)
class PlaneScanout {
public:
    virtual ~PlaneScanout() = default;
    virtual const char* name() const = 0;                 // backend_name() 값
    virtual void set_listener(PlaneScanoutListener* l) = 0;
    virtual VkExtent2D output_extent() const = 0;         // 박스 평면 크기(= 화면 크기)
    // OverlayPlane 을 만든 뒤 한 번: 링 항목을 전송층 버퍼(wl_buffer / FB)로 등록한다.
    virtual void register_overlay(const OverlayPlane& ov) = 0;
    // 한 프레임 커밋. overlay_idx = -1 이면 박스 평면은 그대로 둔다.
    // false = 전송층이 이 프레임을 받지 않았다(버퍼 생성 실패·busy 재사용) — 호출자가 드롭으로 처리한다.
    virtual bool commit(const DrmFrame& video, std::uint64_t token, int overlay_idx) = 0;
    // 이벤트 처리. timeout_ms 까지만 기다린다(0 = 비차단). false = 연결이 끊겼거나 프로토콜 오류.
    virtual bool dispatch(int timeout_ms) = 0;
    virtual bool quit_requested() const = 0;              // 종료 키
    virtual bool should_close() const = 0;                // 창 닫힘·연결 끊김
    // 종료: 화면에서 내리고 남은 release 를 timeout_ms 까지 기다린다. 반환 후 오는 release 는 없다.
    virtual void shutdown(int timeout_ms) = 0;
    // 영상 버퍼 캐시를 비운다(디코더를 닫기 전에). shutdown 뒤에 부른다.
    virtual void clear_video_cache() = 0;
    virtual std::string describe() const = 0;             // 시작 줄 (평면·색 요청)
    virtual std::uint64_t busy_reuse() const { return 0; }
    // 표시 시각 피드백(on_presented)을 주는가. false 면 PlaneDisplay 가 커밋 수를 presented 로 센다.
    virtual bool reports_presentation() const = 0;
    // 박스를 커밋했을 때 debugfs 에서 영상 위 AB24 평면을 기대하는가.
    virtual bool expects_debugfs_overlay() const { return true; }
    // 요약에 덧붙일 전송층 고유 줄 (예: kms 의 flip_timeouts·master_lost·종료 복원 결과). 기본 없음.
    virtual std::vector<std::string> report_lines() const { return {}; }
    // 복구 불가 오류 (should_close 가 true 가 된 이유). "" = 없음. 호출자는 실행을 실패로 처리한다.
    virtual std::string fatal_error() const { return {}; }
};

class PlaneDisplay final : public DisplaySink, private PlaneScanoutListener {
public:
    using Clock = DisplaySink::Clock;

    static constexpr int kMaxCommitted = 3;          // 커밋했지만 아직 release 되지 않은 프레임 상한
    // matched + batch 2 에서는 검출이 두 장씩 한꺼번에 나온다(flush_results 가 연달아 show). pending 이 1 이면
    // 두 번째가 첫 번째를 밀어내 source 페이싱에서도 ~9% 가 드롭됐다(실측 300 프레임 중 26). 2 면 다음 두 vblank 에
    // 차례로 나간다.
    static constexpr int kMaxPending = 2;
    // latest 모드에서 슬롯 밖으로 더 쥘 수 있는 프레임 수(main.cpp 의 디코더 보유 계산이 쓴다)
    static constexpr int kExtraHeldBeyondSlots = kMaxCommitted + kMaxPending;
    static constexpr int kOverlayRing = 3;
    static constexpr auto kFrameCallbackTimeout = std::chrono::milliseconds(50);   // ≈ 3 vblank @60 Hz
    static constexpr int kDrainTimeoutMs = 500;
    static constexpr std::uint64_t kSnapshotAfterPresented = 60;   // 시작 후 debugfs 를 읽을 시점

    PlaneDisplay(yvo::vk::VulkanContext& ctx, std::unique_ptr<PlaneScanout> scanout, const DisplayPipelineConfig& cfg);
    ~PlaneDisplay() override;

    void set_display_done_callback(std::function<void(int)> cb) override { done_cb_ = std::move(cb); }
    void set_source_size(int w, int h) override;
    void retain(int slot, const DrmFrame& f) override;
    bool show(int slot, const std::vector<Detection>& dets, std::string_view hud = {}) override;
    void pump() override;
    void drain() override;
    bool poll_window() override;
    bool quit_requested() const override;
    int frames_held() const override;
    std::optional<Clock::time_point> pace_deadline(std::int64_t frame_index) override;

    const char* backend_name() const override { return scanout_->name(); }
    std::uint64_t presented() const override { return presented_; }
    std::uint64_t display_dropped() const override { return skipped_ + replaced_ + rejected_ + shutdown_dropped_; }
    std::uint64_t overlay_overflows() const override { return ov_ ? ov_->overlay_overflows() : 0; }
    VkExtent2D extent() const override { return scanout_->output_extent(); }
    const std::vector<double>& overlay_ms() const override { return overlay_ms_; }
    const std::vector<double>& present_interval_ms() const override { return interval_ms_; }
    const std::vector<double>& glass_ms() const override { return glass_ms_; }

    std::string describe() const override;
    ScanoutReport scanout_report() const override;

private:
    struct Retained {
        DrmFrame frame;
        Clock::time_point t{};
        bool valid = false;
    };
    struct Item {                 // pending 또는 committed 프레임
        DrmFrame frame;
        int pipe = -1;            // -1 = 호출자가 스스로 display_done 을 세운다(latest)
        int overlay = -1;         // 이 프레임과 함께 올릴 박스 링 항목 (-1 = 직전 박스 유지)
        bool overlay_stat = false;
        Clock::time_point t_retain{};
        std::uint64_t token = 0;
    };

    // PlaneScanoutListener
    void on_video_released(std::uint64_t token) override;
    void on_overlay_released(int overlay_idx) override;
    void on_frame_done() override;
    void on_presented(std::uint64_t token, std::int64_t t_ns, bool clock_monotonic, bool zero_copy) override;
    void on_discarded(std::uint64_t token) override;

    void try_commit(bool force);
    void drop_item(Item& it);          // pending 을 버린다: 박스 반납 + keepalive 해제 + display_done
    void finish_item(Item& it);        // release 됐다: keepalive 해제 + display_done
    bool overlay_ready(Item& it);      // 박스 GPU 완료(비차단) + 통계 1회
    void snapshot(const char* tag, DrmStateSnapshot* out);

    yvo::vk::VulkanContext& ctx_;
    DisplayPipelineConfig cfg_;
    std::unique_ptr<PlaneScanout> scanout_;
    std::unique_ptr<OverlayPlane> ov_;
    std::function<void(int)> done_cb_;

    std::vector<Retained> retained_;
    // 아직 커밋하지 않은 프레임 (표시 순서, 최대 kMaxPending). 넘치면 가장 오래된 것을 버린다.
    std::deque<Item> pending_;
    std::unordered_map<std::uint64_t, Item> committed_;   // token → 커밋된 프레임 (release 대기)
    std::unordered_map<std::uint64_t, Clock::time_point> await_present_;   // token → t_retain (피드백 대기)
    std::uint64_t next_token_ = 1;

    bool frame_cb_pending_ = false;
    Clock::duration min_commit_interval_{0};   // pace=source: 소스 간격 - 8 ms (반 vblank). 0 = 게이트 없음
    Clock::time_point last_commit_{};
    bool overlay_ever_committed_ = false;

    Clock::time_point pace_t0_{};
    bool pace_started_ = false;
    std::int64_t shown_counter_ = 0;

    std::uint64_t presented_ = 0, commits_ = 0;
    std::uint64_t skipped_ = 0, replaced_ = 0, rejected_ = 0, shutdown_dropped_ = 0;
    std::uint64_t overlay_skipped_ = 0, frame_cb_timeouts_ = 0, zero_copy_fb_ = 0, feedback_presented_ = 0;
    std::uint64_t discarded_ = 0;
    bool drained_ = false;

    std::int64_t last_present_ns_ = 0;
    std::vector<double> overlay_ms_, interval_ms_, glass_ms_;

    bool snap_start_taken_ = false;
    DrmStateSnapshot snap_start_, snap_end_;
    std::string dri_debug_dir_ = kDefaultDriDebugDir;
};

} // namespace yvo::display
