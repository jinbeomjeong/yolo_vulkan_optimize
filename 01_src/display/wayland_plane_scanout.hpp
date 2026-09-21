// WaylandPlaneScanout: Wayland implementation of PlaneScanout (--display-backend wayland-planes).
//
//   parent wl_surface (the existing WaylandWindow's xdg_toplevel, **fullscreen forced**)
//     ← wl_buffer wrapping a decoder DrmFrame (NV12, SAND128 modifier, separate Y/UV dma-bufs, pitch field = SAND column height)
//       with zwp_linux_buffer_params_v1. labwc (wlroots) puts it directly on the HVS video plane.
//   sync subsurface (position 0,0, empty input region)
//     ← 3 wl_buffers made from OverlayPlane ring entries (AB24 LINEAR dma-buf). The HVS blends them as a plane above the video.
//   Commit order: subsurface commit (cached, sync) → parent commit → both are applied **atomically**.
//
// Why fullscreen is forced: in windowed mode labwc composites with the GPU (GL) (hvs_wl_probe measured: V3D 6 ms/frame, late
// frames 67 — 07_results/display_vulkan_hdmi/notes/kms_vs_wayland.md). The reason for this backend to exist would disappear.
//
// ── Buffers ──────────────────────────────────────────────────────────────────────────────────
//   · Video wl_buffers are cached keyed by the (dev, inode) pair of the (Y, UV) dma-bufs. The decoder pool is fixed, so only as many
//     as the pool size are created (creation cost only in the first few frames). The cache is **a member of this object** (fixes the probe's global g_created defect).
//   · Creation is async create + per-request context (created/failed events) + **bounded wait** (kCreateTimeoutMs).
//     We do not wait forever even if the compositor does not respond (timeout = reject this frame).
//   · fd ownership: fds passed to zwp_linux_buffer_params_v1.add are dup'ed by libwayland when marshalling
//     (connection.c wl_closure_marshal — WL_ARG_FD uses wl_os_dupfd_cloexec). Our fds remain ours.
//   · Busy-reuse check: if we try to re-attach a video wl_buffer the compositor has not released yet, we reject instead of attaching
//     (counted in busy_reuse — must be 0. The decoder does not reuse that buffer until we drop the keepalive).
//
// ── Color (wp_color_representation_v1, staging) ──────────────────────────────────────────────
//   Requests BT.709 + range. labwc 0.20.1 / wlroots 0.20.2 **inverts** the range when loading it into the HVS (measured —
//   requesting full goes out as COLOR_RANGE=limited: "fullhack"). So --display-color-range auto requests full
//   (= actually limited, matching the decoder output). The actual plane values are checked via debugfs (PlaneDisplay snapshot).
//   If the manager did not advertise that (coef, range) combination, **it is not sent** — an unadvertised combination causes a protocol error
//   (coefficients) and the connection is dropped.
//
// All events are dispatched via WaylandWindow::poll() (same wl_display, default queue). Thread: main thread.
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

    // win must already be open with plane_globals=true, fullscreen=true. If the required globals (zwp_linux_dmabuf_v1 v2+,
    // wl_subcompositor) are missing, std::runtime_error.
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
        bool busy = false;          // held by the compositor (after attach, before release)
        std::uint64_t token = 0;    // last commit it was attached to
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
    struct CreateReq {              // result of one zwp_linux_buffer_params_v1 request (stack lifetime, no globals)
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
    int vp_buf_w_ = 0, vp_buf_h_ = 0;   // buffer size the viewport was set for (0 = not yet)
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
