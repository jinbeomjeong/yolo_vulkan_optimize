// WaylandWindow: xdg-shell 클라이언트 창. **Vulkan 심볼을 하나도 쓰지 않는 순수 플랫폼 글루다.**
//
// 왜 Vulkan 을 모르는가 (계획서 §5.2 닭-달걀 문제):
//   vkGetPhysicalDeviceSurfaceSupportKHR 는 VkSurfaceKHR 를 요구하고 surface 는 wl_display 를 요구한다.
//   그러나 vkGetPhysicalDeviceWaylandPresentationSupportKHR(pd, qf, wl_display) 는 surface 가 필요 없다.
//   따라서 순서는 **WaylandWindow::open() → VulkanContext(wayland_display=native_display()) → VulkanSwapchain
//   (여기서 처음 VkSurfaceKHR 를 만든다)** 이다. 이 클래스는 wl_display*/wl_surface* 를 void* 로만 내보낸다.
//
// 붙이는 것: wl_compositor, xdg_wm_base(+ping/pong), xdg_surface(+ack_configure), xdg_toplevel
//            (title/app_id/fullscreen/configure/close), wl_seat + wl_keyboard(종료 키만, 아래 "키보드"),
//            그리고 선택적으로 wl_subcompositor (아래 "선택 전역" — wayland-planes 가 박스 subsurface 에 쓴다),
//            plane_globals=true 일 때 linux-dmabuf·presentation·viewporter·color-representation.
// 붙이지 않는 것: wl_shm, wl_buffer·wl_callback(스왑체인 또는 WaylandPlaneScanout 이 만든다), 포인터·터치, EGL.
//
// ── 키보드 (종료 키 전용) ──────────────────────────────────────────────────────────────────────────
//  왜 붙이는가: --display-fullscreen 창이 키보드 포커스를 가져가면 Pi 키보드의 Ctrl+C 가 터미널(SIGINT)이
//  아니라 이 창으로 온다. 키보드를 붙이지 않으면 그 키가 버려져 실행을 멈출 수 없다.
//  xkbcommon 은 이 시스템에 없고 의존을 늘리지 않는다 → keymap 은 해석하지 않고(fd 는 즉시 close) evdev
//  원시 keycode(<linux/input-event-codes.h>)로 판정한다. 레이아웃을 무시한 **물리 키** 기준이다.
//  종료 키: Esc, q, Ctrl(좌/우)+C — 눌림 시 quit_requested() 가 true 가 된다. 처리는 poll() 안에서 일어난다.
//
// ── 선택 전역 (실측: labwc 가 광고한다. bind + 접근자만) ─────────────────────────────────────────────
//  Vulkan 스왑체인 경로(기본, plane_globals=false)는 wp_viewporter·wp_presentation 을 bind 하지 않는다(쓰는 곳이
//  없다). 표시 시각은 VK_KHR_present_wait 로 잰다(vk_swapchain.hpp "지연 계측").
//  · plane_globals=true (--display-backend wayland-planes, wayland_plane_scanout.hpp) 일 때만 추가로 bind 한다:
//      zwp_linux_dmabuf_v1 (v4 까지)  — 디코더 NV12 SAND dma-buf / 박스 AB24 dma-buf → wl_buffer
//      wp_presentation (v1)           — 표시 시각(+ clock_id 이벤트를 여기서 받는다: bind 직후에 오므로)
//      wp_viewporter (v1)             — 출력 크기 ≠ 원본 크기일 때 HVS 스케일
//      wp_color_representation_manager_v1 (v1, staging, 빌드 시 XML 이 있을 때만) — BT.709 / range 요청.
//        labwc 0.20.1 은 이 전역을 **두 개** 광고한다(실측, hvs_wl_probe.cpp:176-184). 전부 bind 해서 지원 목록을
//        받은 뒤 BT.709 를 지원하는 첫 관리자 하나만 남기고 나머지는 open() 끝에서 파괴한다.
//    이것들은 Vulkan 경로의 동작을 바꾸지 않도록 옵션으로 막아 둔다(기존 경로 동작 변화 0).
//  · wl_subcompositor (version 1, core 프로토콜이라 wayland-scanner 스텁이 필요 없다) — 광고되면 항상 bind 한다.
//      --display-backend wayland-planes(WaylandPlaneScanout, wayland_plane_scanout.cpp)가 디코더 SAND dma-buf 를
//      주 surface 에 붙이고 박스(AB24) 를 subsurface 로 올린다(계획서 §5.3 안 (III)). Vulkan 스왑체인 경로
//      (안 (I): present 시점 변환 + 스왑체인 STORAGE 직접 쓰기)는 bind 만 하고 쓰지 않는다.
//
// 실패 표현: 연결 실패(XDG_RUNTIME_DIR 없음, 소켓 없음, xdg_wm_base 미광고 …)는 **예외가 아니라 open() 의
// false 반환 + error()** 다. 상위(DisplayPipeline)가 오프스크린으로 폴백해야 하기 때문이다.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// 전방 선언만 한다 — 이 헤더를 include 하는 쪽에 wayland-client.h 를 강요하지 않는다.
struct wl_array;
struct wl_display;
struct wl_registry;
struct wl_compositor;
struct wl_subcompositor;
struct wl_surface;
struct wl_seat;
struct wl_keyboard;
struct xdg_wm_base;
struct xdg_surface;
struct xdg_toplevel;
struct zwp_linux_dmabuf_v1;
struct wp_presentation;
struct wp_viewporter;
struct wp_color_representation_manager_v1;

namespace yvo::display {

struct WaylandWindowOptions {
    int width = 1280;                       // 초기 창 크기 (컴포지터가 configure 로 바꿀 수 있다)
    int height = 720;
    std::string title = "yolo_vulkan_optimize";
    std::string app_id = "yvo_pipeline";
    bool fullscreen = false;                // true면 xdg_toplevel.set_fullscreen (HDMI 1920x1080 direct scanout)
    bool verbose = false;                   // 광고된 전역·configure 를 출력
    bool plane_globals = false;             // wayland-planes: dmabuf/presentation/viewporter/color-representation bind
};

class WaylandWindow {
public:
    WaylandWindow() = default;
    ~WaylandWindow();
    WaylandWindow(const WaylandWindow&) = delete;
    WaylandWindow& operator=(const WaylandWindow&) = delete;

    // 연결 → registry → surface → xdg_surface → xdg_toplevel → commit → roundtrip(첫 configure 수신).
    // 성공하면 true. 실패하면 false 를 돌려주고 error() 에 이유가 남는다(예외를 던지지 않는다).
    bool open(const WaylandWindowOptions& opt);
    void close();
    bool is_open() const { return display_ != nullptr; }
    const std::string& error() const { return error_; }

    // 논블로킹 이벤트 처리. 소켓에 읽을 것이 있을 때만 읽고, 큐에 쌓인 것을 dispatch 한 뒤 flush 한다.
    // **블로킹 wl_display_dispatch() 는 쓰지 않는다** — 디코드 루프가 멈추면 안 된다.
    // timeout_ms > 0 이면 소켓을 최대 그만큼만 기다린다(유한 대기 — wayland-planes 의 동기 버퍼 생성·종료용).
    // 프로토콜 오류나 연결 종료면 false (이때 should_close() 도 true 가 된다).
    bool poll(int timeout_ms = 0);

    // xdg_toplevel.close 를 받았거나 연결이 끊어졌다.
    bool should_close() const { return should_close_; }

    // 창에 포커스가 있을 때 종료 키(Esc, q, Ctrl+C)가 눌렸다. should_close() 와 구분한다 —
    // 상위는 이것을 "사용자 중단"(SIGINT 와 같은 종료)으로, 창 닫기는 정상 종료로 취급한다.
    bool quit_requested() const { return quit_requested_; }

    // xdg_toplevel.configure 로 새 크기가 왔으면 true 를 돌려주고 **그 사실을 소비한다**(스왑체인 재생성 트리거).
    bool take_pending_size(int* w, int* h);

    int width() const { return width_; }
    int height() const { return height_; }
    bool configured() const { return configured_; }

    // 불투명 핸들. VulkanContextOptions::wayland_display 와 VulkanSwapchain 이 쓴다.
    void* native_display() const { return static_cast<void*>(display_); }
    void* native_surface() const { return static_cast<void*>(surface_); }

    // ── 선택 전역 ──
    bool has_subcompositor() const { return subcompositor_ != nullptr; }

    // wayland-planes 용 접근자 (plane_globals=false 면 wl_display/compositor/subcompositor/surface 외에는 nullptr)
    wl_display* display() const { return display_; }
    wl_compositor* compositor() const { return compositor_; }
    wl_subcompositor* subcompositor() const { return subcompositor_; }
    wl_surface* surface() const { return surface_; }
    zwp_linux_dmabuf_v1* linux_dmabuf() const { return dmabuf_; }
    std::uint32_t linux_dmabuf_version() const { return dmabuf_version_; }
    wp_presentation* presentation() const { return presentation_; }
    std::uint32_t presentation_clock() const { return presentation_clock_; }   // clockid_t (기본 CLOCK_MONOTONIC=1)
    wp_viewporter* viewporter() const { return viewporter_; }
    // BT.709 를 지원하는 관리자 (없거나 빌드에서 빠졌으면 nullptr). 광고된 관리자 수와 (coef,range) 목록.
    wp_color_representation_manager_v1* color_representation_manager() const { return color_mgr_; }
    int color_managers_advertised() const { return color_mgrs_advertised_; }
    // 선택된 관리자가 이 (coefficients, range) 조합을 광고했는가 (프로토콜 enum 값)
    bool color_pair_supported(std::uint32_t coefficients, std::uint32_t range) const;
    std::string color_pairs_text() const;

    void set_fullscreen(bool on);
    // xdg_toplevel.configure 를 유발하기 위한 빈 commit. 스왑체인이 붙기 전 초기화용이다.
    void commit();

private:
    // 리스너 콜백 (C 콜백 → this)
    static void on_registry_global(void* data, wl_registry* r, std::uint32_t name, const char* iface,
                                   std::uint32_t version);
    static void on_registry_global_remove(void* data, wl_registry* r, std::uint32_t name);
    static void on_wm_base_ping(void* data, xdg_wm_base* b, std::uint32_t serial);
    static void on_xdg_surface_configure(void* data, xdg_surface* s, std::uint32_t serial);
    static void on_toplevel_configure(void* data, xdg_toplevel* t, std::int32_t w, std::int32_t h, wl_array* states);
    static void on_toplevel_close(void* data, xdg_toplevel* t);
    static void on_seat_capabilities(void* data, wl_seat* s, std::uint32_t caps);
    static void on_seat_name(void* data, wl_seat* s, const char* name);
    static void on_keyboard_keymap(void* data, wl_keyboard* k, std::uint32_t format, std::int32_t fd,
                                   std::uint32_t size);
    static void on_keyboard_enter(void* data, wl_keyboard* k, std::uint32_t serial, wl_surface* s, wl_array* keys);
    static void on_keyboard_leave(void* data, wl_keyboard* k, std::uint32_t serial, wl_surface* s);
    static void on_keyboard_key(void* data, wl_keyboard* k, std::uint32_t serial, std::uint32_t time,
                                std::uint32_t key, std::uint32_t state);
    static void on_keyboard_modifiers(void* data, wl_keyboard* k, std::uint32_t serial, std::uint32_t depressed,
                                      std::uint32_t latched, std::uint32_t locked, std::uint32_t group);
    static void on_keyboard_repeat_info(void* data, wl_keyboard* k, std::int32_t rate, std::int32_t delay);
    void release_keyboard();
    void select_color_manager();
    void bind_plane_global(wl_registry* r, std::uint32_t name, const char* iface, std::uint32_t version);

    static void on_presentation_clock_id(void* data, wp_presentation* p, std::uint32_t clk);
    struct ColorMgrCand {                   // color-representation 관리자 후보 (labwc 는 두 개 광고)
        WaylandWindow* self = nullptr;
        wp_color_representation_manager_v1* mgr = nullptr;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;   // (coefficients, range)
    };
    static void on_color_alpha_mode(void* data, wp_color_representation_manager_v1* m, std::uint32_t mode);
    static void on_color_coef_range(void* data, wp_color_representation_manager_v1* m, std::uint32_t coef,
                                    std::uint32_t range);
    static void on_color_done(void* data, wp_color_representation_manager_v1* m);

    wl_display* display_ = nullptr;
    wl_registry* registry_ = nullptr;
    wl_compositor* compositor_ = nullptr;
    wl_subcompositor* subcompositor_ = nullptr;
    xdg_wm_base* wm_base_ = nullptr;
    wl_surface* surface_ = nullptr;
    xdg_surface* xdg_surface_ = nullptr;
    xdg_toplevel* toplevel_ = nullptr;
    wl_seat* seat_ = nullptr;               // 첫 번째로 광고된 seat 하나만 쓴다
    wl_keyboard* keyboard_ = nullptr;
    std::uint32_t seat_version_ = 0;        // bind 한 버전 (release 요청 가능 여부: keyboard v3+, seat v5+)
    bool plane_globals_ = false;
    zwp_linux_dmabuf_v1* dmabuf_ = nullptr;
    std::uint32_t dmabuf_version_ = 0;
    wp_presentation* presentation_ = nullptr;
    std::uint32_t presentation_clock_ = 1;  // CLOCK_MONOTONIC
    wp_viewporter* viewporter_ = nullptr;
    std::vector<std::unique_ptr<ColorMgrCand>> color_cands_;
    wp_color_representation_manager_v1* color_mgr_ = nullptr;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> color_pairs_;   // 선택된 관리자의 지원 목록
    int color_mgrs_advertised_ = 0;

    int width_ = 0, height_ = 0;
    int pending_w_ = 0, pending_h_ = 0;
    bool size_pending_ = false;
    bool configured_ = false;
    bool should_close_ = false;
    bool quit_requested_ = false;
    bool ctrl_left_ = false, ctrl_right_ = false;  // 물리 Ctrl 눌림 상태 (leave 때 초기화)
    bool verbose_ = false;
    std::string error_;
};

} // namespace yvo::display
