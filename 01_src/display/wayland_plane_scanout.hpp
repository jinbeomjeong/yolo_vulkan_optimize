// WaylandPlaneScanout: PlaneScanout 의 Wayland 구현 (--display-backend wayland-planes).
//
//   부모 wl_surface (기존 WaylandWindow 의 xdg_toplevel, **전체화면 강제**)
//     ← 디코더 DrmFrame(NV12, SAND128 modifier, Y/UV 별도 dma-buf, pitch 필드 = SAND 열 높이)을
//       zwp_linux_buffer_params_v1 로 감싼 wl_buffer. labwc(wlroots)가 이것을 HVS 영상 평면에 직접 올린다.
//   sync subsurface (위치 0,0, 입력 영역 비움)
//     ← OverlayPlane 링 항목(AB24 LINEAR dma-buf)으로 만든 wl_buffer 3개. HVS 가 영상 위 평면으로 블렌딩한다.
//   커밋 순서: subsurface commit(캐시됨, sync) → 부모 commit → 둘이 **원자적으로** 반영된다.
//
// 왜 전체화면을 강제하는가: 창 모드면 labwc 가 GPU(GL)로 합성한다(hvs_wl_probe 실측: V3D 6 ms/frame, 늦은
// 프레임 67 — 07_results/display_vulkan_hdmi/notes/kms_vs_wayland.md). 이 백엔드의 존재 이유가 사라진다.
//
// ── 버퍼 ─────────────────────────────────────────────────────────────────────────────────────
//   · 영상 wl_buffer 는 (Y, UV) dma-buf 의 (dev, inode) 쌍을 키로 캐시한다. 디코더 풀이 고정이라 풀 크기만큼만
//     만들어진다(첫 몇 프레임에서만 생성 비용). 캐시는 **이 객체의 멤버**다(프로브의 전역 g_created 결함 수정).
//   · 생성은 비동기 create + 요청별 컨텍스트(created/failed 이벤트) + **유한 대기**(kCreateTimeoutMs)다.
//     컴포지터가 응답하지 않아도 무한히 기다리지 않는다(시간 초과 = 이 프레임 거부).
//   · fd 소유권: zwp_linux_buffer_params_v1.add 에 넘긴 fd 는 libwayland 가 마샬링할 때 dup 해서 보낸다
//     (connection.c wl_closure_marshal — WL_ARG_FD 는 wl_os_dupfd_cloexec). 우리 fd 는 그대로 우리 소유다.
//   · busy 재사용 검사: 컴포지터가 아직 release 하지 않은 영상 wl_buffer 를 다시 붙이려 하면 붙이지 않고 거부한다
//     (busy_reuse 에 센다 — 0 이어야 한다. 디코더는 우리가 keepalive 를 놓기 전에는 그 버퍼를 재사용하지 않는다).
//
// ── 색 (wp_color_representation_v1, staging) ─────────────────────────────────────────────────
//   BT.709 + range 를 요청한다. labwc 0.20.1 / wlroots 0.20.2 는 range 를 **뒤집어** HVS 에 싣는다(실측 —
//   full 을 요청하면 COLOR_RANGE=limited 로 나간다: "fullhack"). 그래서 --display-color-range auto 는 full 을
//   요청한다(= 실제 limited, 디코더 출력과 맞다). 실제 평면 값은 debugfs 로 확인한다(PlaneDisplay 스냅샷).
//   관리자가 그 (coef, range) 조합을 광고하지 않았으면 **보내지 않는다** — 광고 안 한 조합은 프로토콜 오류
//   (coefficients)로 연결이 끊긴다.
//
// 이벤트는 전부 WaylandWindow::poll() 로 dispatch 된다(같은 wl_display, 기본 큐). 스레드: 메인 스레드.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "display/display_pipeline.hpp"   // ColorRangeRequest
#include "display/plane_display.hpp"
#include "display/wayland_window.hpp"

struct wl_buffer;
struct wl_callback;
struct wl_subsurface;
struct wl_output;
struct wp_presentation_feedback;
struct wp_viewport;
struct wp_color_representation_surface_v1;
struct zwp_linux_buffer_params_v1;

namespace yvo::display {

struct WaylandPlaneScanoutOptions {
    ColorRangeRequest color_range = ColorRangeRequest::Auto;
    bool verbose = false;
};

class WaylandPlaneScanout final : public PlaneScanout {
public:
    static constexpr int kCreateTimeoutMs = 1000;

    // win 은 plane_globals=true, fullscreen=true 로 이미 열려 있어야 한다. 필수 전역(zwp_linux_dmabuf_v1 v2+,
    // wl_subcompositor)이 없으면 std::runtime_error.
    WaylandPlaneScanout(WaylandWindow& win, const WaylandPlaneScanoutOptions& opt);
    ~WaylandPlaneScanout() override;
    WaylandPlaneScanout(const WaylandPlaneScanout&) = delete;
    WaylandPlaneScanout& operator=(const WaylandPlaneScanout&) = delete;

    const char* name() const override { return "wayland-planes"; }
    void set_listener(PlaneScanoutListener* l) override { listener_ = l; }
    VkExtent2D output_extent() const override;
    void register_overlay(const OverlayPlane& ov) override;
    bool commit(const DrmFrame& video, std::uint64_t token, int overlay_idx) override;
    bool dispatch(int timeout_ms) override;
    bool quit_requested() const override { return win_.quit_requested(); }
    bool should_close() const override { return win_.should_close(); }
    void shutdown(int timeout_ms) override;
    void clear_video_cache() override;
    std::string describe() const override;
    std::uint64_t busy_reuse() const override { return busy_reuse_; }
    bool reports_presentation() const override { return win_.presentation() != nullptr; }

    std::uint64_t video_buffers_created() const { return video_created_; }
    std::uint64_t create_failed() const { return create_failed_; }

private:
    using VideoKey = std::array<std::uint64_t, 4>;   // (Y dev, Y ino, UV dev, UV ino)
    struct VideoBuf {
        WaylandPlaneScanout* self = nullptr;
        wl_buffer* buf = nullptr;
        bool busy = false;          // 컴포지터가 쥐고 있다 (attach 후 release 전)
        std::uint64_t token = 0;    // 마지막으로 붙인 커밋
        int w = 0, h = 0;
        std::uint64_t modifier = 0;
    };
    struct OverlayBuf {
        WaylandPlaneScanout* self = nullptr;
        int idx = -1;
        wl_buffer* buf = nullptr;
        bool busy = false;
    };
    struct Feedback {
        WaylandPlaneScanout* self = nullptr;
        wp_presentation_feedback* fb = nullptr;
        std::uint64_t token = 0;
    };
    struct CreateReq {              // zwp_linux_buffer_params_v1 요청 하나의 결과 (스택 수명, 전역 없음)
        bool done = false;
        bool failed = false;
        wl_buffer* buf = nullptr;
    };
    struct PlaneDesc {
        int fd;
        std::uint32_t offset;
        std::uint32_t stride;
        std::uint64_t modifier;
    };

    wl_buffer* create_dmabuf_buffer(int w, int h, std::uint32_t fourcc, const std::vector<PlaneDesc>& planes);
    VideoBuf* video_buffer_for(const DrmFrame& f);
    void update_viewport(int buf_w, int buf_h);
    void setup_color();
    void destroy_all();

    static void on_params_created(void* data, zwp_linux_buffer_params_v1* p, wl_buffer* b);
    static void on_params_failed(void* data, zwp_linux_buffer_params_v1* p);
    static void on_video_release(void* data, wl_buffer* b);
    static void on_overlay_release(void* data, wl_buffer* b);
    static void on_frame_done(void* data, wl_callback* cb, std::uint32_t t);
    static void on_fb_sync_output(void* data, wp_presentation_feedback* f, wl_output* o);
    static void on_fb_presented(void* data, wp_presentation_feedback* f, std::uint32_t sec_hi, std::uint32_t sec_lo,
                                std::uint32_t nsec, std::uint32_t refresh, std::uint32_t seq_hi, std::uint32_t seq_lo,
                                std::uint32_t flags);
    static void on_fb_discarded(void* data, wp_presentation_feedback* f);

    WaylandWindow& win_;
    WaylandPlaneScanoutOptions opt_;
    PlaneScanoutListener* listener_ = nullptr;
    int out_w_ = 0, out_h_ = 0;

    wl_surface* ov_surface_ = nullptr;
    wl_subsurface* ov_sub_ = nullptr;
    wp_viewport* viewport_ = nullptr;
    int vp_buf_w_ = 0, vp_buf_h_ = 0;   // 뷰포트를 맞춘 버퍼 크기 (0 = 아직)
    wp_color_representation_surface_v1* color_surf_ = nullptr;
    std::string color_desc_;

    std::map<VideoKey, std::unique_ptr<VideoBuf>> video_;
    std::vector<std::unique_ptr<OverlayBuf>> overlay_;
    std::unordered_map<wp_presentation_feedback*, std::unique_ptr<Feedback>> feedback_;
    std::unordered_set<wl_callback*> frame_cbs_;
    bool shut_ = false;

    std::uint64_t busy_reuse_ = 0, video_created_ = 0, create_failed_ = 0;
};

} // namespace yvo::display
