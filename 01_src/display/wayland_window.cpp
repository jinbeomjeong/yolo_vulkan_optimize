#include "display/wayland_window.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <linux/input-event-codes.h>
#include <poll.h>
#include <unistd.h>

#include <wayland-client.h>

// wayland-scanner 로 생성한 스텁 (01_src/display/CMakeLists.txt 의 yvo_wayland_protocols 타깃)
#include "xdg-shell-client-protocol.h"
// wayland-planes 용 (plane_globals=true 일 때만 bind 한다)
#include "display/linux-dmabuf-client-protocol.h"
#include "display/presentation-time-client-protocol.h"
#include "display/viewporter-client-protocol.h"
#ifdef YVO_HAVE_COLOR_REPRESENTATION
#include "display/color-representation-client-protocol.h"
#endif

namespace yvo::display {

namespace {

// bind 할 버전. 필요한 요청만 쓰므로 낮게 잡고 컴포지터가 더 높은 버전을 광고해도 올리지 않는다.
//   wl_compositor 4 : wl_surface_damage_buffer (Mesa WSI 의 incremental present 가 쓸 수 있다)
//   xdg_wm_base   1 : 우리가 쓰는 요청(get_xdg_surface/get_toplevel/set_title/set_app_id/set_fullscreen/pong)은
//                     전부 v1 에 있다. 버전을 올리면 처리해야 할 이벤트만 늘어난다.
constexpr std::uint32_t kCompositorVersion = 4;
constexpr std::uint32_t kWmBaseVersion = 1;
constexpr std::uint32_t kSubcompositorVersion = 1;
//   wl_seat       5 : wl_seat.release(v5)·wl_keyboard.release(v3)까지. v7+ 는 keymap mmap 규약만 바뀌고(우리는 매핑하지
//                     않는다) v8+ 는 axis 계열(포인터)이라 필요 없다. 광고 버전이 낮으면 그에 맞춘다.
constexpr std::uint32_t kSeatVersion = 5;
//   zwp_linux_dmabuf_v1 4 : v4 부터 format/modifier 이벤트를 보내지 않는다(리스너 불필요). create/create_immed 는 v2~.
//   wp_presentation 1 / wp_viewporter 1 / wp_color_representation_manager_v1 1 : 쓰는 요청이 전부 v1 에 있다.
constexpr std::uint32_t kLinuxDmabufVersion = 4;
constexpr std::uint32_t kPresentationVersion = 1;
constexpr std::uint32_t kViewporterVersion = 1;

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
    } else if (std::strcmp(iface, wl_seat_interface.name) == 0 && self->seat_ == nullptr) {
        // 종료 키 전용(헤더 "키보드"). seat 이 여럿이면 첫 번째만 쓴다.
        static const wl_seat_listener seat_listener = [] {
            wl_seat_listener l{};
            l.capabilities = &WaylandWindow::on_seat_capabilities;
            l.name = &WaylandWindow::on_seat_name;
            return l;
        }();
        self->seat_version_ = std::min(version, kSeatVersion);
        self->seat_ = bind_global<wl_seat>(r, name, &wl_seat_interface, version, kSeatVersion);
        wl_seat_add_listener(self->seat_, &seat_listener, self);
    } else if (std::strcmp(iface, wl_subcompositor_interface.name) == 0) {
        // wayland-planes 의 박스 subsurface 가 쓴다. Vulkan 경로는 bind 만 한다.
        self->subcompositor_ =
            bind_global<wl_subcompositor>(r, name, &wl_subcompositor_interface, version, kSubcompositorVersion);
    } else if (self->plane_globals_) {
        self->bind_plane_global(r, name, iface, version);
    }
}

void WaylandWindow::bind_plane_global(wl_registry* r, std::uint32_t name, const char* iface, std::uint32_t version) {
    if (std::strcmp(iface, zwp_linux_dmabuf_v1_interface.name) == 0 && dmabuf_ == nullptr) {
        dmabuf_version_ = std::min(version, kLinuxDmabufVersion);
        dmabuf_ = bind_global<zwp_linux_dmabuf_v1>(r, name, &zwp_linux_dmabuf_v1_interface, version,
                                                   kLinuxDmabufVersion);
    } else if (std::strcmp(iface, wp_presentation_interface.name) == 0 && presentation_ == nullptr) {
        // clock_id 이벤트는 bind 직후에 온다 → 리스너를 여기서 붙여야 놓치지 않는다.
        static const wp_presentation_listener pres_listener = [] {
            wp_presentation_listener l{};
            l.clock_id = &WaylandWindow::on_presentation_clock_id;
            return l;
        }();
        presentation_ = bind_global<wp_presentation>(r, name, &wp_presentation_interface, version, kPresentationVersion);
        wp_presentation_add_listener(presentation_, &pres_listener, this);
    } else if (std::strcmp(iface, wp_viewporter_interface.name) == 0 && viewporter_ == nullptr) {
        viewporter_ = bind_global<wp_viewporter>(r, name, &wp_viewporter_interface, version, kViewporterVersion);
    }
#ifdef YVO_HAVE_COLOR_REPRESENTATION
    else if (std::strcmp(iface, wp_color_representation_manager_v1_interface.name) == 0) {
        // 지원 목록 이벤트(supported_*, done)도 bind 직후에 온다. 후보마다 자기 컨텍스트를 준다(전역 상태 없음).
        static const wp_color_representation_manager_v1_listener color_listener = [] {
            wp_color_representation_manager_v1_listener l{};
            l.supported_alpha_mode = &WaylandWindow::on_color_alpha_mode;
            l.supported_coefficients_and_ranges = &WaylandWindow::on_color_coef_range;
            l.done = &WaylandWindow::on_color_done;
            return l;
        }();
        auto c = std::make_unique<ColorMgrCand>();
        c->self = this;
        c->mgr = bind_global<wp_color_representation_manager_v1>(r, name, &wp_color_representation_manager_v1_interface,
                                                                 version, 1);
        wp_color_representation_manager_v1_add_listener(c->mgr, &color_listener, c.get());
        color_cands_.push_back(std::move(c));
        ++color_mgrs_advertised_;
    }
#endif
}

void WaylandWindow::on_presentation_clock_id(void* data, wp_presentation* /*p*/, std::uint32_t clk) {
    static_cast<WaylandWindow*>(data)->presentation_clock_ = clk;
}

void WaylandWindow::on_color_alpha_mode(void* /*data*/, wp_color_representation_manager_v1* /*m*/,
                                        std::uint32_t /*mode*/) {}

void WaylandWindow::on_color_coef_range(void* data, wp_color_representation_manager_v1* /*m*/, std::uint32_t coef,
                                        std::uint32_t range) {
    static_cast<ColorMgrCand*>(data)->pairs.emplace_back(coef, range);
}

void WaylandWindow::on_color_done(void* /*data*/, wp_color_representation_manager_v1* /*m*/) {}

void WaylandWindow::select_color_manager() {
#ifdef YVO_HAVE_COLOR_REPRESENTATION
    // BT.709 를 지원하는 첫 관리자 하나만 남긴다(hvs_wl_probe.cpp:793-799 와 같은 규칙).
    for (auto& c : color_cands_) {
        const bool has709 = std::any_of(c->pairs.begin(), c->pairs.end(), [](const auto& pr) {
            return pr.first == WP_COLOR_REPRESENTATION_SURFACE_V1_COEFFICIENTS_BT709;
        });
        if (color_mgr_ == nullptr && has709) {
            color_mgr_ = c->mgr;
            color_pairs_ = c->pairs;
        } else {
            wp_color_representation_manager_v1_destroy(c->mgr);
        }
        c->mgr = nullptr;
    }
    color_cands_.clear();
#endif
}

bool WaylandWindow::color_pair_supported(std::uint32_t coefficients, std::uint32_t range) const {
    return std::any_of(color_pairs_.begin(), color_pairs_.end(),
                       [&](const auto& pr) { return pr.first == coefficients && pr.second == range; });
}

std::string WaylandWindow::color_pairs_text() const {
    std::string s;
    for (const auto& [c, r] : color_pairs_) {
        if (!s.empty()) s += ' ';
        s += "(coef=" + std::to_string(c) + ",range=" + std::to_string(r) + ")";
    }
    return s.empty() ? std::string("-") : s;
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

// ── wl_seat / wl_keyboard (종료 키 전용) ──────────────────────────────────────────────────────────
void WaylandWindow::on_seat_capabilities(void* data, wl_seat* s, std::uint32_t caps) {
    static const wl_keyboard_listener keyboard_listener = [] {
        wl_keyboard_listener l{};
        l.keymap = &WaylandWindow::on_keyboard_keymap;
        l.enter = &WaylandWindow::on_keyboard_enter;
        l.leave = &WaylandWindow::on_keyboard_leave;
        l.key = &WaylandWindow::on_keyboard_key;
        l.modifiers = &WaylandWindow::on_keyboard_modifiers;
        l.repeat_info = &WaylandWindow::on_keyboard_repeat_info;
        return l;
    }();
    auto* self = static_cast<WaylandWindow*>(data);
    const bool has_kb = (caps & WL_SEAT_CAPABILITY_KEYBOARD) != 0;
    if (has_kb && self->keyboard_ == nullptr) {
        self->keyboard_ = wl_seat_get_keyboard(s);
        wl_keyboard_add_listener(self->keyboard_, &keyboard_listener, self);
    } else if (!has_kb && self->keyboard_ != nullptr) {
        self->release_keyboard();  // 키보드가 뽑혔다 — 다시 꽂히면 다음 capabilities 로 새로 얻는다
    }
    if (self->verbose_) std::printf("[wayland] seat caps=0x%x keyboard=%d\n", caps, self->keyboard_ != nullptr);
}

void WaylandWindow::on_seat_name(void* /*data*/, wl_seat* /*s*/, const char* /*name*/) {}

void WaylandWindow::on_keyboard_keymap(void* /*data*/, wl_keyboard* /*k*/, std::uint32_t /*format*/, std::int32_t fd,
                                       std::uint32_t /*size*/) {
    // keymap 은 해석하지 않는다(xkbcommon 없음). **fd 는 반드시 닫는다** — 컴포지터가 seat/keymap 이 바뀔 때마다
    // 새 fd 를 보내므로 닫지 않으면 샌다.
    if (fd >= 0) ::close(fd);
}

void WaylandWindow::on_keyboard_enter(void* data, wl_keyboard* /*k*/, std::uint32_t /*serial*/, wl_surface* /*s*/,
                                      wl_array* keys) {
    // 포커스를 얻는 순간 이미 눌려 있는 키 목록이 온다. Ctrl 상태만 맞춰 둔다(종료 판정은 새 눌림으로만 한다).
    auto* self = static_cast<WaylandWindow*>(data);
    self->ctrl_left_ = self->ctrl_right_ = false;
    if (keys == nullptr) return;
    const auto* p = static_cast<const std::uint32_t*>(keys->data);
    for (std::size_t i = 0; i < keys->size / sizeof(std::uint32_t); ++i) {
        if (p[i] == KEY_LEFTCTRL) self->ctrl_left_ = true;
        if (p[i] == KEY_RIGHTCTRL) self->ctrl_right_ = true;
    }
}

void WaylandWindow::on_keyboard_leave(void* data, wl_keyboard* /*k*/, std::uint32_t /*serial*/, wl_surface* /*s*/) {
    // 포커스를 잃으면 뗌 이벤트가 오지 않는다 → Ctrl 이 눌린 채로 남지 않게 초기화한다.
    auto* self = static_cast<WaylandWindow*>(data);
    self->ctrl_left_ = self->ctrl_right_ = false;
}

void WaylandWindow::on_keyboard_key(void* data, wl_keyboard* /*k*/, std::uint32_t /*serial*/, std::uint32_t /*time*/,
                                    std::uint32_t key, std::uint32_t state) {
    // key 는 evdev 원시 코드다(xkb keycode 는 여기에 +8). 레이아웃과 무관한 물리 키로 판정한다.
    auto* self = static_cast<WaylandWindow*>(data);
    const bool pressed = state == WL_KEYBOARD_KEY_STATE_PRESSED;
    if (key == KEY_LEFTCTRL) self->ctrl_left_ = pressed;
    else if (key == KEY_RIGHTCTRL) self->ctrl_right_ = pressed;
    if (!pressed) return;
    const bool ctrl = self->ctrl_left_ || self->ctrl_right_;
    if (key == KEY_ESC || key == KEY_Q || (ctrl && key == KEY_C)) {
        if (self->verbose_ && !self->quit_requested_) std::printf("[wayland] 종료 키 (evdev %u)\n", key);
        self->quit_requested_ = true;
    }
}

void WaylandWindow::on_keyboard_modifiers(void* /*data*/, wl_keyboard* /*k*/, std::uint32_t /*serial*/,
                                          std::uint32_t /*depressed*/, std::uint32_t /*latched*/,
                                          std::uint32_t /*locked*/, std::uint32_t /*group*/) {
    // 수정자 비트의 의미는 keymap 에 달려 있어 xkbcommon 없이는 해석할 수 없다 → Ctrl 은 key 이벤트로 추적한다.
}

void WaylandWindow::on_keyboard_repeat_info(void* /*data*/, wl_keyboard* /*k*/, std::int32_t /*rate*/,
                                            std::int32_t /*delay*/) {}

void WaylandWindow::release_keyboard() {
    if (keyboard_ == nullptr) return;
    // keyboard 는 seat 의 버전을 물려받는다. release(v3+)는 서버 쪽 자원까지 정리한다.
    if (seat_version_ >= WL_KEYBOARD_RELEASE_SINCE_VERSION)
        wl_keyboard_release(keyboard_);
    else
        wl_keyboard_destroy(keyboard_);
    keyboard_ = nullptr;
    ctrl_left_ = ctrl_right_ = false;
}

bool WaylandWindow::open(const WaylandWindowOptions& opt) {
    // 프로토콜 버전이 올라가 멤버가 추가돼도 컴파일이 깨지지 않게, 지정 초기화 대신 값 초기화 + 멤버 대입을 쓴다.
    // ⚠ 채우지 않은 멤버(nullptr)는 "무시"가 **아니다** — libwayland(1.24 확인)는 그 opcode 의 이벤트가 오면
    //   "listener function for opcode %u of %s is NULL" 로 abort 한다. 아래 리스너는 bind 한 버전(xdg_wm_base v1:
    //   ping / xdg_surface: configure / xdg_toplevel v1: configure·close)의 이벤트를 전부 채운 것이다.
    //   bind 버전을 올리면 새 이벤트 멤버도 반드시 채운다. 정적 지역이므로 객체 수명보다 오래 산다.
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
    plane_globals_ = opt.plane_globals;
    error_.clear();
    quit_requested_ = false;
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
    // color-representation 관리자의 지원 목록은 위 두 번째 roundtrip 까지 다 왔다 → 하나만 고른다.
    if (plane_globals_) select_color_manager();
    if (verbose_)
        std::printf("[wayland] open %dx%d%s (subcompositor=%d)\n", width_, height_,
                    opt.fullscreen ? " fullscreen" : "", has_subcompositor());
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
    if (wm_base_ != nullptr) {
        xdg_wm_base_destroy(wm_base_);
        wm_base_ = nullptr;
    }
    // wayland-planes 전역 (plane_globals=false 면 전부 nullptr)
#ifdef YVO_HAVE_COLOR_REPRESENTATION
    for (auto& c : color_cands_)
        if (c->mgr != nullptr) wp_color_representation_manager_v1_destroy(c->mgr);
    if (color_mgr_ != nullptr) wp_color_representation_manager_v1_destroy(color_mgr_);
#endif
    color_cands_.clear();
    color_mgr_ = nullptr;
    color_pairs_.clear();
    color_mgrs_advertised_ = 0;
    if (viewporter_ != nullptr) {
        wp_viewporter_destroy(viewporter_);
        viewporter_ = nullptr;
    }
    if (presentation_ != nullptr) {
        wp_presentation_destroy(presentation_);
        presentation_ = nullptr;
    }
    if (dmabuf_ != nullptr) {
        zwp_linux_dmabuf_v1_destroy(dmabuf_);
        dmabuf_ = nullptr;
        dmabuf_version_ = 0;
    }
    // keyboard 는 seat 에서 얻었으므로 seat 보다 먼저 해제한다.
    release_keyboard();
    if (seat_ != nullptr) {
        if (seat_version_ >= WL_SEAT_RELEASE_SINCE_VERSION)
            wl_seat_release(seat_);
        else
            wl_seat_destroy(seat_);
        seat_ = nullptr;
        seat_version_ = 0;
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

bool WaylandWindow::poll(int timeout_ms) {
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
    // timeout 0 = 논블로킹(기본). 양수면 그 시간까지만 기다린다. 블로킹 dispatch(무한 대기)는 쓰지 않는다.
    const int pr = ::poll(&pfd, 1, std::max(timeout_ms, 0));
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
