// WaylandWindow: xdg-shell client window. **Pure platform glue that uses not a single Vulkan symbol.**
//
// Why it does not know Vulkan (plan §5.2 chicken-and-egg problem):
//   vkGetPhysicalDeviceSurfaceSupportKHR requires a VkSurfaceKHR, and a surface requires a wl_display.
//   But vkGetPhysicalDeviceWaylandPresentationSupportKHR(pd, qf, wl_display) needs no surface.
//   So the order is **WaylandWindow::open() → VulkanContext(wayland_display=native_display()) → VulkanSwapchain
//   (which creates the VkSurfaceKHR for the first time)**. This class exposes wl_display*/wl_surface* only as void*.
//
// Binds: wl_compositor, xdg_wm_base(+ping/pong), xdg_surface(+ack_configure), xdg_toplevel
//            (title/app_id/fullscreen/configure/close), wl_seat + wl_keyboard (quit keys only, see "Keyboard" below).
// Does not bind: wl_shm, wl_buffer/wl_callback (created by the swapchain), wp_viewporter/wp_presentation (display time
//            is measured with VK_KHR_present_wait, vk_swapchain.hpp "Latency measurement"), pointer/touch, EGL.
//
// ── Keyboard (quit keys only) ────────────────────────────────────────────────────────────────────────
//  Why bind it: when a --display-fullscreen window takes keyboard focus, Ctrl+C on the Pi keyboard goes to this window
//  instead of the terminal (SIGINT). Without a keyboard, that key is discarded and the run cannot be stopped.
//  xkbcommon is not on this system and we do not add dependencies → the keymap is not interpreted (the fd is closed immediately) and
//  we judge by raw evdev keycodes (<linux/input-event-codes.h>). It is based on **physical keys**, ignoring the layout.
//  Quit keys: Esc, q, Ctrl(left/right)+C — on press, quit_requested() becomes true. Handling happens inside poll().
//
// Failure representation: connection failures (no XDG_RUNTIME_DIR, no socket, xdg_wm_base not advertised …) are **not exceptions but
// open() returning false + error()**. The caller (yvo_pipeline) reports error() and exits.
#pragma once

#include <cstdint>
#include <string>

// Forward declarations only — do not force wayland-client.h on code that includes this header.
struct wl_array;
struct wl_display;
struct wl_registry;
struct wl_compositor;
struct wl_surface;
struct wl_seat;
struct wl_keyboard;
struct xdg_wm_base;
struct xdg_surface;
struct xdg_toplevel;

namespace yvo::display {

struct WaylandWindowOptions {
    int width = 1280;                       // initial window size (the compositor may change it via configure)
    int height = 720;
    std::string title = "yolo_vulkan_optimize";
    std::string app_id = "yvo_pipeline";
    bool fullscreen = false;                // if true, xdg_toplevel.set_fullscreen (HDMI 1920x1080 direct scanout)
    bool verbose = false;                   // print advertised globals and configure
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
    // If timeout_ms > 0, waits on the socket for at most that long (bounded wait; DisplayPipeline always passes 0).
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

    wl_display* display_ = nullptr;
    wl_registry* registry_ = nullptr;
    wl_compositor* compositor_ = nullptr;
    xdg_wm_base* wm_base_ = nullptr;
    wl_surface* surface_ = nullptr;
    xdg_surface* xdg_surface_ = nullptr;
    xdg_toplevel* toplevel_ = nullptr;
    wl_seat* seat_ = nullptr;               // use only the first advertised seat
    wl_keyboard* keyboard_ = nullptr;
    std::uint32_t seat_version_ = 0;        // bound version (whether release requests are available: keyboard v3+, seat v5+)

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
