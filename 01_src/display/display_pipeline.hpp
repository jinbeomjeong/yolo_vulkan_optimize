// DisplayPipeline: 디코더 SAND dma-buf + 검출 목록 → HDMI 한 프레임.
//
// `yvo_display_probe --probe convert` 의 `run_convert_display()` 루프를 그대로 클래스로 옮긴 것이다.
// 한 프레임의 순서는 프로브와 **완전히 같다**:
//     acquire(timeout=0) → begin → record_acquire_transition → DisplayConvert::record
//       → OverlayRenderer::set_frame/record → record_present_transition
//       → end_and_submit → present
// 다른 점은 셋뿐이다.
//   ① 프레임이 디코드 루프가 아니라 **파이프라인 슬롯**에서 온다(`retain` → `show`).
//   ② 완료 회수가 `while(ready)` 블로킹이 아니라 **`pump()` 펜스 폴링 + 콜백**이다.
//   ③ Wayland 가 없으면 `OffscreenBackend` 로 내려간다(`--display-backend offscreen`).
//
// ── 설계 결정: display_done 은 present 펜스가 아니라 **submit 펜스**에서 세운다 ────────────────
// 디코더 프레임(SAND dma-buf)을 놓아도 되는 시점은 **GPU 가 그것을 다 읽은 때**이고, 그것이
// `DisplayConvert::finish(d)` 가 기다리는 submit 펜스다. present 펜스(VkSwapchainPresentFenceInfoEXT)는
// "스왑체인 이미지를 다시 써도 된다"는 더 나중의 사건이라 1~2 vsync(16~33 ms) 더 잡게 된다.
// 스왑체인 이미지의 재사용 안전성은 `VulkanSwapchain::acquire()` 가 sync 슬롯 펜스로 따로 본다.
//
// ── 슬롯 두 개의 번호 체계를 섞지 않는다 ──────────────────────────────────────────────────────
//   · **파이프라인 슬롯** (0..cfg.num_slots-1) : main.cpp 의 GPU·NPU 공용 슬롯. retain/show/콜백의 인자.
//   · **표시 슬롯**       (0..image_count-1)   : `SwapchainFrame::sync_index`. DisplayConvert/Overlay 의 슬롯.
//   둘의 대응은 `show()` 가 `inflight_` 에 기록하고, `pump()` 가 펜스 완료 때 되돌려 콜백에 넘긴다.
//
// ── 스레드 ────────────────────────────────────────────────────────────────────────────────────
// 전부 메인 스레드다. `vkQueueSubmit`/`vkQueuePresentKHR` 를 하므로 다른 스레드에서 부르면 안 된다
// (`infer_stage.hpp` 의 스레드 규율과 같은 이유 — VkQueue 가 하나뿐이다).
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
#include "display/display_convert.hpp"
#include "display/display_sink.hpp"
#include "display/offscreen_backend.hpp"
#include "display/overlay_renderer.hpp"
#include "display/vk_swapchain.hpp"
#include "display/wayland_window.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

enum class DisplayBackendKind {
    Auto,        // Wayland 를 먼저 시도하고 실패하면 offscreen
    Wayland,     // strict — 실패하면 예외(조용한 폴백 금지)
    Offscreen,
    WaylandPlanes,  // HVS 두 평면 스캔아웃 (PlaneDisplay + WaylandPlaneScanout). DisplayPipeline 은 만들지 않는다
};

// wayland-planes 가 컴포지터에 요청할 COLOR_RANGE (wp_color_representation_v1)
enum class ColorRangeRequest {
    Auto,      // = Full: labwc 0.20.1/wlroots 0.20.2 는 요청과 **뒤집혀** 나간다("fullhack") → 실제로 limited
    Full,
    Limited,
};

// 어느 프레임을 어떤 검출과 함께 보여 줄 것인가.
enum class DisplaySync {
    Matched,   // 그 프레임의 검출이 나온 뒤 함께 표시 (기본). 지연은 늘지만 박스가 정확히 맞는다
    Latest,    // 전처리가 끝나는 대로 표시하고 검출은 **가장 최근 것**을 얹는다. 표시 지연 최소
};

enum class DisplayPace {
    None,      // 페이싱 없음 (표시 off)
    Source,    // 소스 프레임 레이트(기본 30 fps)로 맞춘다 — 지연·드롭 판정용
    Asap,      // 최대 속도 — 처리량 판정용
};

struct DisplayPipelineConfig {
    DisplayBackendKind backend = DisplayBackendKind::Auto;
    DisplaySync sync = DisplaySync::Matched;
    DisplayPace pace = DisplayPace::Source;

    bool fullscreen = false;
    int width = 1280;                  // --display-size (fullscreen 이면 컴포지터가 덮어쓴다)
    int height = 720;
    int every = 1;                     // K 프레임마다 1장만 표시 (1 = 전부)
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    bool force_blit = false;           // --display-blit

    int overlay_line_width = 2;
    int overlay_text_scale = 2;
    int max_dets = 80;

    int src_w = 1920;                  // 디코더 원본 크기 (오버레이 affine + 1:1 판정)
    int src_h = 1080;
    double source_fps = 30.0;          // pace = Source 의 목표
    int num_slots = 4;                 // 파이프라인 슬롯 수 (retain 배열 크기)

    ColorRangeRequest color_range = ColorRangeRequest::Auto;   // wayland-planes 전용

    std::string title = "yolo_vulkan_optimize";
    std::string app_id = "yvo_pipeline";
    bool verbose = false;
};

class DisplayPipeline final : public DisplaySink {
public:
    using Clock = DisplaySink::Clock;

    // ⚠ **VulkanContext 보다 먼저** 불러야 한다. vkGetPhysicalDeviceWaylandPresentationSupportKHR 가
    //   wl_display* 를 요구하기 때문이다(vulkan_context.hpp 의 닭-달걀 설명).
    //   backend == Offscreen 이면 아무것도 하지 않고 false 를 돌려준다.
    //   backend == Wayland 에서 실패해도 **예외를 던지지 않는다** — 호출자가 error() 를 보고 판단한다.
    static bool open_window(WaylandWindow& win, const DisplayPipelineConfig& cfg);

    // win == nullptr 또는 ctx.present_enabled()==false 면 오프스크린 백엔드로 만든다.
    DisplayPipeline(yvo::vk::VulkanContext& ctx, WaylandWindow* win, const DisplayPipelineConfig& cfg);
    ~DisplayPipeline() override;
    DisplayPipeline(const DisplayPipeline&) = delete;
    DisplayPipeline& operator=(const DisplayPipeline&) = delete;

    // 표시가 끝나 슬롯을 놓아도 된다고 알린다(submit 펜스 기준). 인자는 **파이프라인 슬롯**.
    void set_display_done_callback(std::function<void(int)> cb) override { done_cb_ = std::move(cb); }

    // 원본 프레임 크기. **디코더를 연 뒤 첫 show() 전에** 부른다(`InferStage::set_source_size` 와 같은 규약).
    // 초기화 순서가 창 → 컨텍스트 → 표시 → 디코더라서 생성자 시점에는 아직 알 수 없다
    // (`extra_hw_frames` 가 `image_count()` 에 의존하므로 디코더가 마지막이어야 한다).
    // 1:1 인지 균일 스케일인지가 여기서 정해진다 — 잘못 두면 조용히 2.6배 느린 경로로 간다.
    void set_source_size(int w, int h) override;

    // 디코드 직후: 이 슬롯이 쓸 프레임의 keepalive 를 **추가로** 붙든다(main 의 f.reset() 앞에서).
    // 동시에 이 시각을 glass-to-glass 의 시작점으로 기록한다.
    void retain(int slot, const DrmFrame& f) override;

    // 한 프레임을 화면에 낸다. false = 표시 드롭 — 호출자는 **즉시** display_done(slot) 을 세워야 한다.
    bool show(int slot, const std::vector<Detection>& dets, std::string_view hud = {}) override;

    // 펜스 폴링 → finish → display_done 콜백. 블로킹하지 않는다. 매 바퀴 부른다.
    void pump() override;

    // 남은 표시 작업을 전부 끝낸다(블로킹). 종료 때 한 번.
    void drain() override;

    // Wayland 이벤트 처리 + 창 크기 변경 예약. false = 창이 닫혔거나 종료 키가 눌렸다(오프스크린이면 항상 true).
    // 둘은 quit_requested() 로 구분한다.
    bool poll_window() override;

    // 표시 창에 포커스가 있을 때 종료 키(Esc, q, Ctrl+C)가 눌렸다 — 호출자는 SIGINT 와 같은 중단으로 취급한다.
    bool quit_requested() const override;

    // 이 경로가 지금 붙들고 있는 디코더 프레임 수 (retained + DisplayConvert 인플라이트).
    int frames_held() const override;

    // pace 에 따른 이 프레임의 처리 마감 시각. 페이싱이 없으면 빈 optional.
    // 호출자는 그때까지 pump 루프를 돈다(블로킹 sleep 을 쓰면 NPU·GPU 후처리가 굶는다).
    std::optional<Clock::time_point> pace_deadline(std::int64_t frame_index) override;

    // ── 통계 ──
    const char* backend_name() const override { return offscreen_ ? "offscreen" : "wayland"; }
    std::uint64_t presented() const override { return backend_->presented(); }
    std::uint64_t display_dropped() const override { return backend_->display_dropped() + skipped_ + busy_dropped_; }
    std::uint64_t recreates() const override { return backend_->recreates(); }
    std::uint32_t image_count() const override { return backend_->image_count(); }
    bool offscreen() const override { return offscreen_; }
    bool uniform_scale() const override { return uniform_scale_; }
    bool blit_mode() const override { return backend_->blit_mode(); }
    VkExtent2D extent() const override { return backend_->extent(); }
    std::uint64_t overlay_overflows() const override { return overlay_overflow_; }

    const std::vector<double>& convert_ms() const override { return convert_ms_; }
    const std::vector<double>& convert_total_ms() const override { return convert_total_ms_; }
    const std::vector<double>& overlay_ms() const override { return overlay_ms_; }
    const std::vector<double>& present_interval_ms() const override { return interval_ms_; }
    const std::vector<double>& glass_ms() const override { return glass_ms_; }

private:
    struct Pending {              // 표시 슬롯 → 파이프라인 슬롯 + 지연 측정 정보
        int disp = -1;
        int pipe = -1;            // -1 = 호출자가 스스로 display_done 을 세운다(latest 모드)
        Clock::time_point t_retain{};
        bool have_retain = false;
    };

    void build_passes();            // DisplayConvert/Overlay 를 백엔드 이미지 수·크기에 맞춰 만든다
    void apply_extent();            // 크기가 바뀐 뒤 uniform_scale·LUT 상한·슬롯 수를 맞춘다
    void recreate_if_needed();
    void collect(const Pending& p); // finish + timestamp 회수 + display_done
    void drain_present_waits(bool blocking);

    yvo::vk::VulkanContext& ctx_;
    DisplayPipelineConfig cfg_;
    WaylandWindow* win_ = nullptr;
    bool offscreen_ = false;
    bool uniform_scale_ = false;

    std::unique_ptr<DisplayBackend> backend_;
    std::unique_ptr<DisplayConvert> conv_;
    std::unique_ptr<OverlayRenderer> ov_;

    std::function<void(int)> done_cb_;

    // 파이프라인 슬롯이 붙들고 있는 프레임 (show() 가 소비한다)
    struct Retained {
        DrmFrame frame;
        Clock::time_point t{};
        bool valid = false;
    };
    std::vector<Retained> retained_;

    std::deque<Pending> inflight_;     // 표시 제출 순서
    struct PresentWait {
        std::uint64_t id = 0;
        Clock::time_point t_retain{};
        bool have_retain = false;
    };
    std::deque<PresentWait> present_ids_;
    Clock::time_point last_present_{};
    bool have_last_present_ = false;
    bool pace_blocking_ok_ = true;   // pace=asap 에서는 present_wait 로 스로틀하면 안 된다

    std::int64_t shown_counter_ = 0;   // --display-every K 카운터
    std::uint64_t skipped_ = 0;        // every 로 건너뛴 수
    std::uint64_t busy_dropped_ = 0;   // 표시 슬롯이 전부 GPU 에 걸려 있어 버린 수
    std::uint64_t overlay_overflow_ = 0;

    Clock::time_point pace_t0_{};
    bool pace_started_ = false;

    std::vector<double> convert_ms_, convert_total_ms_, overlay_ms_, interval_ms_, glass_ms_;
};

const char* sync_name(DisplaySync s);
const char* pace_name(DisplayPace p);
bool parse_backend_kind(const std::string& s, DisplayBackendKind* out);
bool parse_color_range(const std::string& s, ColorRangeRequest* out);
const char* color_range_name(ColorRangeRequest r);
bool parse_display_sync(const std::string& s, DisplaySync* out);
bool parse_display_pace(const std::string& s, DisplayPace* out);

} // namespace yvo::display
