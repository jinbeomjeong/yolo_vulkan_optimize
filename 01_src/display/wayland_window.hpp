// WaylandWindow: xdg-shell client window. **Pure platform glue that uses not a single Vulkan symbol.**
//
// Why it does not know Vulkan (plan §5.2 chicken-and-egg problem):
//   vkGetPhysicalDeviceSurfaceSupportKHR requires a VkSurfaceKHR, and a surface requires a wl_display.
//   But vkGetPhysicalDeviceWaylandPresentationSupportKHR(pd, qf, wl_display) needs no surface.
//   So the order is **WaylandWindow::open() → VulkanContext(wayland_display=native_display()) → VulkanSwapchain
//   (which creates the VkSurfaceKHR for the first time)**. This class exposes wl_display*/wl_surface* only as void*.
//
// Binds: wl_compositor, xdg_wm_base(+ping/pong), xdg_surface(+ack_configure), xdg_toplevel
//            (title/app_id/fullscreen/configure/close), wl_seat + wl_keyboard (quit keys only, see "Keyboard" below),
//            and optionally wl_subcompositor (see "Optional globals" below — wayland-planes uses it for the box subsurface),
//            and when plane_globals=true, linux-dmabuf, presentation, viewporter, color-representation.
// Does not bind: wl_shm, wl_buffer/wl_callback (created by the swapchain or WaylandPlaneScanout), pointer/touch, EGL.
//
// ── Keyboard (quit keys only) ────────────────────────────────────────────────────────────────────────
//  Why bind it: when a --display-fullscreen window takes keyboard focus, Ctrl+C on the Pi keyboard goes to this window
//  instead of the terminal (SIGINT). Without a keyboard, that key is discarded and the run cannot be stopped.
//  xkbcommon is not on this system and we do not add dependencies → the keymap is not interpreted (the fd is closed immediately) and
//  we judge by raw evdev keycodes (<linux/input-event-codes.h>). It is based on **physical keys**, ignoring the layout.
//  Quit keys: Esc, q, Ctrl(left/right)+C — on press, quit_requested() becomes true. Handling happens inside poll().
//
// ── Optional globals (measured: labwc advertises them. bind + accessors only) ──────────────────────────
//  The Vulkan swapchain path (default, plane_globals=false) does not bind wp_viewporter/wp_presentation (nothing
//  uses them). Display time is measured with VK_KHR_present_wait (vk_swapchain.hpp "Latency measurement").
//  · Only when plane_globals=true (--display-backend wayland-planes, wayland_plane_scanout.hpp) are these additionally bound:
//      zwp_linux_dmabuf_v1 (up to v4)  — decoder NV12 SAND dma-buf / box AB24 dma-buf → wl_buffer
//      wp_presentation (v1)           — display time (+ the clock_id event is received here: it arrives right after bind)
//      wp_viewporter (v1)             — HVS scaling when output size ≠ source size
//      wp_color_representation_manager_v1 (v1, staging, only when the XML is present at build time) — BT.709 / range request.
//        labwc 0.20.1 advertises **two** of this global (measured, hvs_wl_probe.cpp:176-184). We bind all of them to receive
//        the supported lists, keep only the first manager that supports BT.709, and destroy the rest at the end of open().
//    These are gated behind an option so they do not change the Vulkan path's behavior (zero behavior change on the existing path).
//  · wl_subcompositor (version 1, core protocol so no wayland-scanner stub is needed) — always bound when advertised.
//      --display-backend wayland-planes (WaylandPlaneScanout, wayland_plane_scanout.cpp) attaches the decoder SAND dma-buf
//      to the main surface and raises the box (AB24) as a subsurface (plan §5.3 option (III)). The Vulkan swapchain path
//      (option (I): conversion at present time + direct STORAGE write to the swapchain) only binds it and does not use it.
//
// Failure representation: connection failures (no XDG_RUNTIME_DIR, no socket, xdg_wm_base not advertised …) are **not exceptions but
// open() returning false + error()**. Because the upper layer (DisplayPipeline) must fall back to offscreen.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Forward declarations only — do not force wayland-client.h on code that includes this header.
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
    int width = 1280;                       // initial window size (the compositor may change it via configure)
    int height = 720;
    std::string title = "yolo_vulkan_optimize";
    std::string app_id = "yvo_pipeline";
    bool fullscreen = false;                // if true, xdg_toplevel.set_fullscreen (HDMI 1920x1080 direct scanout)
    bool verbose = false;                   // print advertised globals and configure
    bool plane_globals = false;             // wayland-planes: dmabuf/presentation/viewporter/color-representation bind
};

class WaylandWindow {
public:
    WaylandWindow() = default;
    ~WaylandWindow();
    WaylandWindow(const WaylandWindow&) = delete;
    WaylandWindow& operator=(const WaylandWindow&) = delete;

    // connect → registry → surface → xdg_surface → xdg_toplevel → commit → roundtrip (receive the first configure).
    // Returns true on success. On failure returns false and the reason is left in error() (does not throw).
    bool open(const WaylandWindowOptions& opt);
    void close();
    bool is_open() const { return display_ != nullptr; }
    const std::string& error() const { return error_; }

    // Non-blocking event processing. Reads only when the socket has something to read, dispatches what is queued, then flushes.
    // **Blocking wl_display_dispatch() is not used** — the decode loop must not stall.
    // If timeout_ms > 0, waits on the socket for at most that long (bounded wait — for wayland-planes' synchronous buffer creation/shutdown).
    // false on a protocol error or connection close (should_close() also becomes true then).
    bool poll(int timeout_ms = 0);

    // Received xdg_toplevel.close or the connection was lost.
    bool should_close() const { return should_close_; }

    // A quit key (Esc, q, Ctrl+C) was pressed while the window had focus. Distinguished from should_close() —
    // the upper layer treats this as a "user interrupt" (same as SIGINT termination), and window close as normal termination.
    bool quit_requested() const { return quit_requested_; }

    // Returns true if xdg_toplevel.configure delivered a new size, and **consumes that fact** (swapchain recreation trigger).
    bool take_pending_size(int* w, int* h);

    int width() const { return width_; }
    int height() const { return height_; }
    bool configured() const { return configured_; }

    // Opaque handles. Used by VulkanContextOptions::wayland_display and VulkanSwapchain.
    void* native_display() const { return static_cast<void*>(display_); }
    void* native_surface() const { return static_cast<void*>(surface_); }

    // ── Optional globals ──
    bool has_subcompositor() const { return subcompositor_ != nullptr; }

    // Accessors for wayland-planes (if plane_globals=false, nullptr except wl_display/compositor/subcompositor/surface)
    wl_display* display() const { return display_; }
    wl_compositor* compositor() const { return compositor_; }
    wl_subcompositor* subcompositor() const { return subcompositor_; }
    wl_surface* surface() const { return surface_; }
    zwp_linux_dmabuf_v1* linux_dmabuf() const { return dmabuf_; }
    std::uint32_t linux_dmabuf_version() const { return dmabuf_version_; }
    wp_presentation* presentation() const { return presentation_; }
    std::uint32_t presentation_clock() const { return presentation_clock_; }   // clockid_t (default CLOCK_MONOTONIC=1)
    wp_viewporter* viewporter() const { return viewporter_; }
    // Manager that supports BT.709 (nullptr if absent or left out of the build). Number of advertised managers and (coef,range) list.
    wp_color_representation_manager_v1* color_representation_manager() const { return color_mgr_; }
    int color_managers_advertised() const { return color_mgrs_advertised_; }
    // Did the selected manager advertise this (coefficients, range) combination (protocol enum values)
    bool color_pair_supported(std::uint32_t coefficients, std::uint32_t range) const;
    std::string color_pairs_text() const;

    void set_fullscreen(bool on);
    // Empty commit to trigger xdg_toplevel.configure. For initialization before the swapchain is attached.
    void commit();

private:
    // Listener callbacks (C callback → this)
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
    struct ColorMgrCand {                   // color-representation manager candidate (labwc advertises two)
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
    wl_seat* seat_ = nullptr;               // use only the first advertised seat
    wl_keyboard* keyboard_ = nullptr;
    std::uint32_t seat_version_ = 0;        // bound version (whether release requests are available: keyboard v3+, seat v5+)
    bool plane_globals_ = false;
    zwp_linux_dmabuf_v1* dmabuf_ = nullptr;
    std::uint32_t dmabuf_version_ = 0;
    wp_presentation* presentation_ = nullptr;
    std::uint32_t presentation_clock_ = 1;  // CLOCK_MONOTONIC
    wp_viewporter* viewporter_ = nullptr;
    std::vector<std::unique_ptr<ColorMgrCand>> color_cands_;
    wp_color_representation_manager_v1* color_mgr_ = nullptr;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> color_pairs_;   // supported list of the selected manager
    int color_mgrs_advertised_ = 0;

    int width_ = 0, height_ = 0;
    int pending_w_ = 0, pending_h_ = 0;
    bool size_pending_ = false;
    bool configured_ = false;
    bool should_close_ = false;
    bool quit_requested_ = false;
    bool ctrl_left_ = false, ctrl_right_ = false;  // physical Ctrl pressed state (reset on leave)
    bool verbose_ = false;
    std::string error_;
};

} // namespace yvo::display
