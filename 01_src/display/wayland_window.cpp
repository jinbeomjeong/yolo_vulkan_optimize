#include "display/wayland_window.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <poll.h>

#include <wayland-client.h>

// wayland-scanner 로 생성한 스텁 (01_src/display/CMakeLists.txt 의 yvo_wayland_protocols 타깃)
#include "presentation-time-client-protocol.h"
#include "viewporter-client-protocol.h"
#include "xdg-shell-client-protocol.h"

namespace yvo::display {

namespace {

// bind 할 버전. 필요한 요청만 쓰므로 낮게 잡고 컴포지터가 더 높은 버전을 광고해도 올리지 않는다.
//   wl_compositor 4 : wl_surface_damage_buffer (Mesa WSI 의 incremental present 가 쓸 수 있다)
//   xdg_wm_base   1 : 우리가 쓰는 요청(get_xdg_surface/get_toplevel/set_title/set_app_id/set_fullscreen/pong)은
//                     전부 v1 에 있다. 버전을 올리면 처리해야 할 이벤트만 늘어난다.
constexpr std::uint32_t kCompositorVersion = 4;
constexpr std::uint32_t kWmBaseVersion = 1;
constexpr std::uint32_t kSubcompositorVersion = 1;
constexpr std::uint32_t kViewporterVersion = 1;
constexpr std::uint32_t kPresentationVersion = 1;

template <typename T>
T* bind_global(wl_registry* r, std::uint32_t name, const wl_interface* iface, std::uint32_t advertised,
               std::uint32_t want) {
    return static_cast<T*>(wl_registry_bind(r, name, iface, std::min(advertised, want)));
}

} // namespace

WaylandWindow::~WaylandWindow() { close(); }

void WaylandWindow::on_registry_global(void* data, wl_registry* r, std::uint32_t name, const char* iface,
                                       std::uint32_t version) {
    auto* self = static_cast<WaylandWindow*>(data);
    if (self->verbose_) std::printf("[wayland] global %s v%u\n", iface, version);
    if (std::strcmp(iface, wl_compositor_interface.name) == 0) {
        self->compositor_ =
            bind_global<wl_compositor>(r, name, &wl_compositor_interface, version, kCompositorVersion);
    } else if (std::strcmp(iface, xdg_wm_base_interface.name) == 0) {
        self->wm_base_ = bind_global<xdg_wm_base>(r, name, &xdg_wm_base_interface, version, kWmBaseVersion);
    } else if (std::strcmp(iface, wl_subcompositor_interface.name) == 0) {
        // 안 (III) 의 전제. 지금은 bind 만 한다.
        self->subcompositor_ =
            bind_global<wl_subcompositor>(r, name, &wl_subcompositor_interface, version, kSubcompositorVersion);
    } else if (std::strcmp(iface, wp_viewporter_interface.name) == 0) {
        // supportedPresentScaling == None 이라 HVS 확대를 쓰려면 이것뿐이다. 동작은 다음 단계.
        self->viewporter_ =
            bind_global<wp_viewporter>(r, name, &wp_viewporter_interface, version, kViewporterVersion);
    } else if (std::strcmp(iface, wp_presentation_interface.name) == 0) {
        // present_wait 가 없을 때의 지연 계측 폴백. 동작은 다음 단계.
        self->presentation_ =
            bind_global<wp_presentation>(r, name, &wp_presentation_interface, version, kPresentationVersion);
    }
}

void WaylandWindow::on_registry_global_remove(void* /*data*/, wl_registry* /*r*/, std::uint32_t /*name*/) {
    // 우리가 쓰는 전역은 컴포지터 수명 동안 사라지지 않는다. 사라지면 다음 poll() 이 프로토콜 오류로 잡는다.
}

void WaylandWindow::on_wm_base_ping(void* /*data*/, xdg_wm_base* b, std::uint32_t serial) {
    // pong 을 빠뜨리면 컴포지터가 우리를 "응답 없음"으로 보고 창을 죽인다. 필수다.
    xdg_wm_base_pong(b, serial);
}

void WaylandWindow::on_xdg_surface_configure(void* data, xdg_surface* s, std::uint32_t serial) {
    auto* self = static_cast<WaylandWindow*>(data);
    xdg_surface_ack_configure(s, serial);
    self->configured_ = true;
}

void WaylandWindow::on_toplevel_configure(void* data, xdg_toplevel* /*t*/, std::int32_t w, std::int32_t h,
                                          wl_array* /*states*/) {
    auto* self = static_cast<WaylandWindow*>(data);
    // 0 은 "클라이언트가 정하라"는 뜻이다 → 현재 크기를 유지한다.
    if (w <= 0 || h <= 0) return;
    if (w == self->width_ && h == self->height_) return;
    self->pending_w_ = w;
    self->pending_h_ = h;
    self->size_pending_ = true;
    self->width_ = w;
    self->height_ = h;
    if (self->verbose_) std::printf("[wayland] toplevel configure %dx%d\n", w, h);
}

void WaylandWindow::on_toplevel_close(void* data, xdg_toplevel* /*t*/) {
    static_cast<WaylandWindow*>(data)->should_close_ = true;
}

bool WaylandWindow::open(const WaylandWindowOptions& opt) {
    // 프로토콜 버전이 올라가 이벤트가 추가돼도 컴파일이 깨지지 않게, 지정 초기화 대신 값 초기화 + 멤버 대입을
    // 쓴다(채우지 않은 멤버는 nullptr = "이 이벤트는 무시"). 정적 지역이므로 객체 수명보다 오래 산다.
    static const wl_registry_listener registry_listener = [] {
        wl_registry_listener l{};
        l.global = &WaylandWindow::on_registry_global;
        l.global_remove = &WaylandWindow::on_registry_global_remove;
        return l;
    }();
    static const xdg_wm_base_listener wm_base_listener = [] {
        xdg_wm_base_listener l{};
        l.ping = &WaylandWindow::on_wm_base_ping;
        return l;
    }();
    static const xdg_surface_listener xdg_surface_listener_ = [] {
        xdg_surface_listener l{};
        l.configure = &WaylandWindow::on_xdg_surface_configure;
        return l;
    }();
    static const xdg_toplevel_listener toplevel_listener = [] {
        xdg_toplevel_listener l{};
        l.configure = &WaylandWindow::on_toplevel_configure;
        l.close = &WaylandWindow::on_toplevel_close;
        return l;
    }();

    if (display_ != nullptr) {
        error_ = "이미 열려 있다";
        return false;
    }
    verbose_ = opt.verbose;
    error_.clear();
    width_ = std::max(1, opt.width);
    height_ = std::max(1, opt.height);

    if (std::getenv("XDG_RUNTIME_DIR") == nullptr) {
        error_ = "XDG_RUNTIME_DIR 가 없다 (SSH 라면 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0)";
        return false;
    }
    display_ = wl_display_connect(nullptr);
    if (display_ == nullptr) {
        error_ = "wl_display_connect() 실패 (WAYLAND_DISPLAY 소켓 확인)";
        return false;
    }

    registry_ = wl_display_get_registry(display_);
    wl_registry_add_listener(registry_, &registry_listener, this);
    if (wl_display_roundtrip(display_) < 0) {
        error_ = "registry roundtrip 실패";
        close();
        return false;
    }
    if (compositor_ == nullptr) {
        error_ = "wl_compositor 를 광고하지 않는다";
        close();
        return false;
    }
    if (wm_base_ == nullptr) {
        error_ = "xdg_wm_base 를 광고하지 않는다 (xdg-shell 미지원 컴포지터)";
        close();
        return false;
    }
    xdg_wm_base_add_listener(wm_base_, &wm_base_listener, this);

    surface_ = wl_compositor_create_surface(compositor_);
    if (surface_ == nullptr) {
        error_ = "wl_compositor_create_surface() 실패";
        close();
        return false;
    }
    xdg_surface_ = xdg_wm_base_get_xdg_surface(wm_base_, surface_);
    if (xdg_surface_ == nullptr) {
        error_ = "xdg_wm_base_get_xdg_surface() 실패";
        close();
        return false;
    }
    xdg_surface_add_listener(xdg_surface_, &xdg_surface_listener_, this);
    toplevel_ = xdg_surface_get_toplevel(xdg_surface_);
    if (toplevel_ == nullptr) {
        error_ = "xdg_surface_get_toplevel() 실패";
        close();
        return false;
    }
    xdg_toplevel_add_listener(toplevel_, &toplevel_listener, this);
    xdg_toplevel_set_title(toplevel_, opt.title.c_str());
    xdg_toplevel_set_app_id(toplevel_, opt.app_id.c_str());
    if (opt.fullscreen) xdg_toplevel_set_fullscreen(toplevel_, nullptr);

    // 버퍼 없이 commit 하면 컴포지터가 첫 configure 를 보낸다(xdg-shell 규약).
    wl_surface_commit(surface_);
    if (wl_display_roundtrip(display_) < 0) {
        error_ = "첫 configure roundtrip 실패";
        close();
        return false;
    }
    // 첫 configure 로 받은 크기는 "변경"이 아니다 — 스왑체인이 width()/height() 로 만들면 된다.
    size_pending_ = false;
    if (verbose_)
        std::printf("[wayland] open %dx%d%s (viewporter=%d presentation=%d subcompositor=%d)\n", width_, height_,
                    opt.fullscreen ? " fullscreen" : "", has_viewporter(), has_presentation(), has_subcompositor());
    return true;
}

void WaylandWindow::close() {
    // 생성의 역순으로 파괴한다.
    if (toplevel_ != nullptr) {
        xdg_toplevel_destroy(toplevel_);
        toplevel_ = nullptr;
    }
    if (xdg_surface_ != nullptr) {
        xdg_surface_destroy(xdg_surface_);
        xdg_surface_ = nullptr;
    }
    if (surface_ != nullptr) {
        wl_surface_destroy(surface_);
        surface_ = nullptr;
    }
    if (presentation_ != nullptr) {
        wp_presentation_destroy(presentation_);
        presentation_ = nullptr;
    }
    if (viewporter_ != nullptr) {
        wp_viewporter_destroy(viewporter_);
        viewporter_ = nullptr;
    }
    if (wm_base_ != nullptr) {
        xdg_wm_base_destroy(wm_base_);
        wm_base_ = nullptr;
    }
    if (subcompositor_ != nullptr) {
        wl_subcompositor_destroy(subcompositor_);
        subcompositor_ = nullptr;
    }
    if (compositor_ != nullptr) {
        wl_compositor_destroy(compositor_);
        compositor_ = nullptr;
    }
    if (registry_ != nullptr) {
        wl_registry_destroy(registry_);
        registry_ = nullptr;
    }
    if (display_ != nullptr) {
        wl_display_disconnect(display_);
        display_ = nullptr;
    }
    configured_ = false;
    size_pending_ = false;
}

bool WaylandWindow::poll() {
    if (display_ == nullptr) return false;
    auto fail = [&](const char* what) {
        const int err = wl_display_get_error(display_);
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s (wl_display_get_error=%d, errno=%d)", what, err, errno);
        error_ = buf;
        should_close_ = true;
        return false;
    };

    if (wl_display_flush(display_) < 0 && errno != EAGAIN) return fail("wl_display_flush");
    // wl_display_prepare_read 는 기본 큐가 비어 있을 때만 성공한다. 남아 있으면 먼저 소비한다.
    while (wl_display_prepare_read(display_) != 0) {
        if (wl_display_dispatch_pending(display_) < 0) return fail("wl_display_dispatch_pending");
    }
    pollfd pfd{};
    pfd.fd = wl_display_get_fd(display_);
    pfd.events = POLLIN;
    const int pr = ::poll(&pfd, 1, 0);  // timeout 0 = 논블로킹. 블로킹 dispatch 는 쓰지 않는다.
    if (pr > 0 && (pfd.revents & POLLIN) != 0) {
        if (wl_display_read_events(display_) < 0) return fail("wl_display_read_events");
    } else {
        wl_display_cancel_read(display_);
        if (pr < 0 && errno != EINTR) return fail("poll");
    }
    if (wl_display_dispatch_pending(display_) < 0) return fail("wl_display_dispatch_pending");
    if (wl_display_flush(display_) < 0 && errno != EAGAIN) return fail("wl_display_flush");
    return true;
}

bool WaylandWindow::take_pending_size(int* w, int* h) {
    if (!size_pending_) return false;
    if (w != nullptr) *w = pending_w_;
    if (h != nullptr) *h = pending_h_;
    size_pending_ = false;
    return true;
}

wp_viewport* WaylandWindow::create_viewport() {
    if (viewporter_ == nullptr || surface_ == nullptr) return nullptr;
    return wp_viewporter_get_viewport(viewporter_, surface_);
}

void WaylandWindow::set_fullscreen(bool on) {
    if (toplevel_ == nullptr) return;
    if (on)
        xdg_toplevel_set_fullscreen(toplevel_, nullptr);
    else
        xdg_toplevel_unset_fullscreen(toplevel_);
}

void WaylandWindow::commit() {
    if (surface_ != nullptr) wl_surface_commit(surface_);
}

} // namespace yvo::display
