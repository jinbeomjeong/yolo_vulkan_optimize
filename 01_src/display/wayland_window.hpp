// WaylandWindow: xdg-shell 클라이언트 창. **Vulkan 심볼을 하나도 쓰지 않는 순수 플랫폼 글루다.**
//
// 왜 Vulkan 을 모르는가 (계획서 §5.2 닭-달걀 문제):
//   vkGetPhysicalDeviceSurfaceSupportKHR 는 VkSurfaceKHR 를 요구하고 surface 는 wl_display 를 요구한다.
//   그러나 vkGetPhysicalDeviceWaylandPresentationSupportKHR(pd, qf, wl_display) 는 surface 가 필요 없다.
//   따라서 순서는 **WaylandWindow::open() → VulkanContext(wayland_display=native_display()) → VulkanSwapchain
//   (여기서 처음 VkSurfaceKHR 를 만든다)** 이다. 이 클래스는 wl_display*/wl_surface* 를 void* 로만 내보낸다.
//
// 붙이는 것: wl_compositor, xdg_wm_base(+ping/pong), xdg_surface(+ack_configure), xdg_toplevel
//            (title/app_id/fullscreen/configure/close), 그리고 선택적으로 wl_subcompositor
//            (아래 "선택 전역" 참고 — bind 만 한다).
// 붙이지 않는 것: wl_shm, wl_buffer(스왑체인이 만든다), wl_callback(FIFO 가 스로틀한다),
//                 wl_seat/키보드(xkbcommon 의존성 회피 — 종료는 xdg_toplevel.close 와 SIGINT 로 충분), EGL.
//
// ── 선택 전역 (실측: labwc 가 광고한다. bind + 접근자만) ─────────────────────────────────────────────
//  wp_viewporter·wp_presentation 은 bind 하지 않는다(쓰는 곳이 없다). 표시 시각은 VK_KHR_present_wait 로
//  잰다(vk_swapchain.hpp "지연 계측").
//  · wl_subcompositor (version 1, core 프로토콜이라 wayland-scanner 스텁이 필요 없다)
//      계획서 §5.3 안 (III)(디코더 SAND dma-buf 를 surface 에 직접 붙이고 박스를 subsurface 로 올리는 안)의
//      전제다. 지금은 쓰지 않는다 — 안 (I)(present 시점 변환 + 스왑체인 STORAGE 직접 쓰기)이 1순위다.
//
// 실패 표현: 연결 실패(XDG_RUNTIME_DIR 없음, 소켓 없음, xdg_wm_base 미광고 …)는 **예외가 아니라 open() 의
// false 반환 + error()** 다. 상위(DisplayPipeline)가 오프스크린으로 폴백해야 하기 때문이다.
#pragma once

#include <cstdint>
#include <string>

// 전방 선언만 한다 — 이 헤더를 include 하는 쪽에 wayland-client.h 를 강요하지 않는다.
struct wl_array;
struct wl_display;
struct wl_registry;
struct wl_compositor;
struct wl_subcompositor;
struct wl_surface;
struct xdg_wm_base;
struct xdg_surface;
struct xdg_toplevel;

namespace yvo::display {

struct WaylandWindowOptions {
    int width = 1280;                       // 초기 창 크기 (컴포지터가 configure 로 바꿀 수 있다)
    int height = 720;
    std::string title = "yolo_vulkan_optimize";
    std::string app_id = "yvo_pipeline";
    bool fullscreen = false;                // true면 xdg_toplevel.set_fullscreen (HDMI 1920x1080 direct scanout)
    bool verbose = false;                   // 광고된 전역·configure 를 출력
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
    // 프로토콜 오류나 연결 종료면 false (이때 should_close() 도 true 가 된다).
    bool poll();

    // xdg_toplevel.close 를 받았거나 연결이 끊어졌다.
    bool should_close() const { return should_close_; }

    // xdg_toplevel.configure 로 새 크기가 왔으면 true 를 돌려주고 **그 사실을 소비한다**(스왑체인 재생성 트리거).
    bool take_pending_size(int* w, int* h);

    int width() const { return width_; }
    int height() const { return height_; }
    bool configured() const { return configured_; }

    // 불투명 핸들. VulkanContextOptions::wayland_display 와 VulkanSwapchain 이 쓴다.
    void* native_display() const { return static_cast<void*>(display_); }
    void* native_surface() const { return static_cast<void*>(surface_); }

    // ── 선택 전역 (자리만 만들어 둔다) ──
    bool has_subcompositor() const { return subcompositor_ != nullptr; }

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

    wl_display* display_ = nullptr;
    wl_registry* registry_ = nullptr;
    wl_compositor* compositor_ = nullptr;
    wl_subcompositor* subcompositor_ = nullptr;
    xdg_wm_base* wm_base_ = nullptr;
    wl_surface* surface_ = nullptr;
    xdg_surface* xdg_surface_ = nullptr;
    xdg_toplevel* toplevel_ = nullptr;

    int width_ = 0, height_ = 0;
    int pending_w_ = 0, pending_h_ = 0;
    bool size_pending_ = false;
    bool configured_ = false;
    bool should_close_ = false;
    bool verbose_ = false;
    std::string error_;
};

} // namespace yvo::display
