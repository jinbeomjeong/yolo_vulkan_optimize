// hvs_wl_probe: wraps the decoder's SAND128 NV12 dma-buf as-is in a wl_buffer and hands it to labwc,
// and measures whether labwc (wlroots) puts it on an HVS hardware plane (direct scanout) or composites it with GL.
//
//   video surface  : fullscreen xdg_toplevel. zwp_linux_buffer_params_v1(NV12, SAND128, per-plane offset/pitch)
//   overlay        : ARGB family (AB24 = memory R,G,B,A) wl_subsurface. Drawn by the CPU into a dma-heap (CMA) dmabuf ring.
//   pacing         : on each wl_surface.frame callback (≈vblank), commit a new frame per the source 30 fps schedule
//   measure        : wp_presentation feedback (display time, seq, ZERO_COPY flag), buffers held until wl_buffer.release,
//                    V3D gpu_stats per-queue utilization, CPU time of this process and labwc, debugfs state snapshots (plane placement)
//
// Standalone probe unrelated to the pipeline itself. Does not use the NPU; replays detections.jsonl.
#include <drm_fourcc.h>
#include <fcntl.h>
#include <xf86drm.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <wayland-client.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "color-representation-client-protocol.h"
#include "linux-dmabuf-client-protocol.h"
#include "presentation-time-client-protocol.h"
#include "viewporter-client-protocol.h"
#include "xdg-shell-client-protocol.h"

#include "common/dma_heap_alloc.hpp"
#include "overlay_painter.hpp"
#include "pipeline/ffmpeg_drm_decoder.hpp"
#include "probe_common.hpp"

using namespace hvsprobe;

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Args {
    std::string video = "04_data/videos/video_h265.mp4";
    std::string dets;
    std::string out_dir = ".";
    std::string label = "wl";
    int frames = 300;
    bool overlay = false;
    bool fullscreen = true;
    int win_w = 1280, win_h = 720;
    int vp_w = 0, vp_h = 0; // viewport destination uses this size even in fullscreen (0 = unused)
    bool color_repr = false;
    bool color_range_full = false; // range sent to wp_color_representation (default limited)
    int overlay_bufs = 3;
    int sand_param = 0;      // modifier parameter (column height). If 0, the pitch field is the column height (HVS6 kernel convention)
    long pitch_y = -1;       // -1 = original decoder descriptor value (column height)
    long pitch_uv = -1;
    int freeze_at = -1;      // after presenting this frame, pause for freeze_ms and capture with grim
    int freeze_ms = 2500;
    std::string heap = "/dev/dma_heap/linux,cma";
    std::string overlay_alloc = "heap"; // heap = dma-heap (CMA, cached mapping) / dumb = card1 dumb buffer (WC mapping), PRIME-exported
    bool verbose = false;
};

void usage(const char* p) {
    std::fprintf(stderr,
                 "usage: %s [--video F] [--dets detections.jsonl] [--frames N] [--overlay on|off]\n"
                 "          [--window fullscreen|WxH] [--viewport WxH] [--color-repr on|off] [--color-range limited|full] [--overlay-bufs N]\n"
                 "          [--sand-param N] [--pitch-y N] [--pitch-uv N] [--heap PATH] [--overlay-alloc heap|dumb]\n"
                 "          [--freeze-at K] [--freeze-ms MS] [--out-dir DIR] [--label L] [--verbose]\n",
                 p);
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--dets" && (v = next())) a.dets = v;
        else if (s == "--frames" && (v = next())) a.frames = std::atoi(v);
        else if (s == "--overlay" && (v = next())) a.overlay = std::string(v) == "on";
        else if (s == "--window" && (v = next())) {
            if (std::string(v) == "fullscreen") a.fullscreen = true;
            else if (std::sscanf(v, "%dx%d", &a.win_w, &a.win_h) == 2) a.fullscreen = false;
            else return false;
        } else if (s == "--color-repr" && (v = next())) a.color_repr = std::string(v) == "on";
        else if (s == "--viewport" && (v = next())) {
            if (std::sscanf(v, "%dx%d", &a.vp_w, &a.vp_h) != 2) return false;
        } else if (s == "--color-range" && (v = next())) a.color_range_full = std::string(v) == "full";
        else if (s == "--overlay-bufs" && (v = next())) a.overlay_bufs = std::max(1, std::atoi(v));
        else if (s == "--sand-param" && (v = next())) a.sand_param = std::atoi(v);
        else if (s == "--pitch-y" && (v = next())) a.pitch_y = std::atol(v);
        else if (s == "--pitch-uv" && (v = next())) a.pitch_uv = std::atol(v);
        else if (s == "--freeze-at" && (v = next())) a.freeze_at = std::atoi(v);
        else if (s == "--freeze-ms" && (v = next())) a.freeze_ms = std::atoi(v);
        else if (s == "--out-dir" && (v = next())) a.out_dir = v;
        else if (s == "--label" && (v = next())) a.label = v;
        else if (s == "--heap" && (v = next())) a.heap = v;
        else if (s == "--overlay-alloc" && (v = next())) a.overlay_alloc = v;
        else if (s == "--verbose") a.verbose = true;
        else return false;
    }
    return a.frames > 0;
}

// ── Frame records ────────────────────────────────────────────────────────────────────────────
struct FrameRec {
    std::int64_t pts = -1;
    std::int64_t decoded_ns = 0;   // decoder.next() return time
    std::int64_t commit_ns = 0;    // parent surface commit time
    std::int64_t presented_ns = 0; // wp_presentation.presented (clock = presentation clock_id)
    std::uint64_t seq = 0;
    std::uint32_t flags = 0;
    std::uint32_t refresh_ns = 0;
    bool discarded = false;
    bool got_feedback = false;
    int held_at_commit = 0;
    double overlay_draw_ms = 0;    // CPU drawing (clear + draw) time
    double overlay_sync_ms = 0;    // total time including DMA_BUF_SYNC START/END
    int overlay_buf = -1;
    int ndets = 0;
};

struct App;

struct VideoBuf {
    App* app = nullptr;
    wl_buffer* buf = nullptr;
    bool busy = false;
    std::shared_ptr<void> keepalive;
    std::int64_t pts = -1;
    std::int64_t committed_ns = 0;
    std::int64_t release_count = 0;
};

struct OverlayBuf {
    App* app = nullptr;
    int idx = -1;
    wl_buffer* buf = nullptr;
    bool busy = false;
    std::uint8_t* ptr = nullptr; // CPU mapping (w*h*4 packed)
    int fd = -1;                 // dma-buf fd (heap: owned by pool / dumb: owned by this struct)
    bool needs_sync = false;     // heap (cached mapping) needs DMA_BUF_SYNC, dumb (WC mapping) does not
    std::uint32_t dumb_handle = 0;
    std::size_t map_len = 0;
};

struct FeedbackCtx {
    App* app = nullptr;
    std::size_t rec = 0;
    struct wp_presentation_feedback* proxy = nullptr;
};

struct App {
    Args args;
    // globals
    wl_display* display = nullptr;
    wl_registry* registry = nullptr;
    wl_compositor* compositor = nullptr;
    wl_subcompositor* subcompositor = nullptr;
    xdg_wm_base* wm_base = nullptr;
    zwp_linux_dmabuf_v1* dmabuf = nullptr;
    std::uint32_t dmabuf_ver = 0;
    wp_presentation* presentation = nullptr;
    clockid_t pres_clock = CLOCK_MONOTONIC;
    wp_viewporter* viewporter = nullptr;
    wp_color_representation_manager_v1* color_repr_mgr = nullptr; // selected manager (the one supporting BT.709 limited)
    std::vector<std::string> globals;
    std::vector<std::string> crm_supported;
    // The number of wp_color_representation_manager_v1 globals depends on the version (measured: labwc 0.20.1 = two, labwc 0.20.2 / wlroots 0.20.2+rpt4 = one). Bind all of them,
    // receive their supported lists, then pick the one that supports BT.709.
    struct CrmCand {
        App* app = nullptr;
        std::size_t idx = 0;
        wp_color_representation_manager_v1* mgr = nullptr;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
    };
    std::vector<std::unique_ptr<CrmCand>> crm_cands;

    // surface
    wl_surface* surface = nullptr;
    xdg_surface* xsurface = nullptr;
    xdg_toplevel* toplevel = nullptr;
    wp_viewport* viewport = nullptr;
    wp_color_representation_surface_v1* color_repr = nullptr;
    wl_surface* ov_surface = nullptr;
    wl_subsurface* ov_sub = nullptr;
    bool configured = false;
    int conf_w = 0, conf_h = 0;
    bool closed = false;

    // dmabuf feedback (per-surface scanout tranche)
    std::vector<std::pair<std::uint32_t, std::uint64_t>> fmt_table;
    std::uint32_t tranche_flags = 0;
    std::vector<std::uint16_t> tranche_idx;
    std::vector<std::string> feedback_log;
    bool feedback_done = false;

    // video
    yvo::FfmpegDrmDecoder dec;
    std::map<std::pair<ino_t, ino_t>, std::unique_ptr<VideoBuf>> vbufs;
    std::deque<yvo::DrmFrame> ahead; // pre-decoded frame (1)
    int held = 0;                    // number of AVFrames we hold (busy wl_buffer + ahead)
    int held_max = 0;
    std::int64_t create_failed = 0;
    bool last_create_failed = false;
    bool create_done = false;

    // overlay
    std::unique_ptr<yvo::DmaHeapPool> ov_pool;
    int card_fd = -1; // /dev/dri/card1 when --overlay-alloc dumb
    std::vector<std::unique_ptr<OverlayBuf>> ov_bufs;
    std::unique_ptr<OverlayPainter> painter;
    std::string ov_backend = "none";
    std::int64_t overlay_skips = 0;
    std::map<std::int64_t, std::vector<yvo::Detection>> dets;

    // pacing / records
    std::vector<FrameRec> recs;
    std::int64_t t_start = 0;         // first frame commit time
    double frame_period_ns = 1e9 / 30.0;
    int presented_video = 0;          // number of video frames committed
    std::int64_t callbacks = 0;
    std::int64_t idle_commits = 0;
    bool frozen = false;
    std::int64_t freeze_until = 0;
    bool finished = false;

    // measurement window (steady state): frame 30 to end
    V3dStats v3d_a, v3d_b;
    std::int64_t meas_t0 = 0, meas_t1 = 0, cpu0 = 0, cpu1 = 0, lab0 = 0, lab1 = 0;
    int meas_f0 = 0, meas_f1 = 0;
    int labwc_pid = -1;
    std::vector<std::string> snapshots;
    std::vector<FeedbackCtx*> live_feedback; // feedback not yet presented/discarded (cleaned up at exit)
};

void forget_feedback(FeedbackCtx* ctx) {
    auto& v = ctx->app->live_feedback;
    v.erase(std::remove(v.begin(), v.end(), ctx), v.end());
}

// ── Listeners ─────────────────────────────────────────────────────────────────────────────────
void wm_ping(void*, xdg_wm_base* b, std::uint32_t serial) { xdg_wm_base_pong(b, serial); }
const xdg_wm_base_listener kWmListener = {wm_ping};

void xs_configure(void* data, xdg_surface* s, std::uint32_t serial) {
    auto* app = static_cast<App*>(data);
    xdg_surface_ack_configure(s, serial);
    app->configured = true;
}
const xdg_surface_listener kXsListener = {xs_configure};

void tl_configure(void* data, xdg_toplevel*, std::int32_t w, std::int32_t h, wl_array*) {
    auto* app = static_cast<App*>(data);
    app->conf_w = w;
    app->conf_h = h;
    if (app->args.verbose) std::printf("[xdg] configure %dx%d\n", w, h);
}
void tl_close(void* data, xdg_toplevel*) { static_cast<App*>(data)->closed = true; }
void tl_configure_bounds(void*, xdg_toplevel*, std::int32_t, std::int32_t) {}
void tl_wm_caps(void*, xdg_toplevel*, wl_array*) {}
const xdg_toplevel_listener kTlListener = {tl_configure, tl_close, tl_configure_bounds, tl_wm_caps};

void pres_clock_id(void* data, wp_presentation*, std::uint32_t clk) {
    static_cast<App*>(data)->pres_clock = static_cast<clockid_t>(clk);
}
const wp_presentation_listener kPresListener = {pres_clock_id};

void crm_alpha(void*, wp_color_representation_manager_v1*, std::uint32_t) {}
void crm_coef(void* data, wp_color_representation_manager_v1*, std::uint32_t coef, std::uint32_t range) {
    auto* c = static_cast<App::CrmCand*>(data);
    c->pairs.emplace_back(coef, range);
    c->app->crm_supported.push_back("[mgr" + std::to_string(c->idx) + "]coef=" + std::to_string(coef) +
                                    ",range=" + std::to_string(range));
}
void crm_done(void*, wp_color_representation_manager_v1*) {}
const wp_color_representation_manager_v1_listener kCrmListener = {crm_alpha, crm_coef, crm_done};

void reg_global(void* data, wl_registry* r, std::uint32_t name, const char* iface, std::uint32_t ver) {
    auto* app = static_cast<App*>(data);
    app->globals.push_back(std::string(iface) + " v" + std::to_string(ver));
    const std::string i = iface;
    if (i == wl_compositor_interface.name)
        app->compositor = static_cast<wl_compositor*>(wl_registry_bind(r, name, &wl_compositor_interface, std::min(ver, 4u)));
    else if (i == wl_subcompositor_interface.name)
        app->subcompositor = static_cast<wl_subcompositor*>(wl_registry_bind(r, name, &wl_subcompositor_interface, 1));
    else if (i == xdg_wm_base_interface.name) {
        app->wm_base = static_cast<xdg_wm_base*>(wl_registry_bind(r, name, &xdg_wm_base_interface, 1));
        xdg_wm_base_add_listener(app->wm_base, &kWmListener, app);
    } else if (i == zwp_linux_dmabuf_v1_interface.name) {
        app->dmabuf_ver = std::min(ver, 4u);
        app->dmabuf = static_cast<zwp_linux_dmabuf_v1*>(wl_registry_bind(r, name, &zwp_linux_dmabuf_v1_interface, app->dmabuf_ver));
    } else if (i == wp_presentation_interface.name) {
        app->presentation = static_cast<wp_presentation*>(wl_registry_bind(r, name, &wp_presentation_interface, 1));
        wp_presentation_add_listener(app->presentation, &kPresListener, app);
    } else if (i == wp_viewporter_interface.name)
        app->viewporter = static_cast<wp_viewporter*>(wl_registry_bind(r, name, &wp_viewporter_interface, 1));
    else if (i == wp_color_representation_manager_v1_interface.name) {
        auto c = std::make_unique<App::CrmCand>();
        c->app = app;
        c->idx = app->crm_cands.size();
        c->mgr = static_cast<wp_color_representation_manager_v1*>(
            wl_registry_bind(r, name, &wp_color_representation_manager_v1_interface, 1));
        wp_color_representation_manager_v1_add_listener(c->mgr, &kCrmListener, c.get());
        app->crm_cands.push_back(std::move(c));
    }
}
void reg_remove(void*, wl_registry*, std::uint32_t) {}
const wl_registry_listener kRegListener = {reg_global, reg_remove};

// dmabuf surface feedback (v4)
void fb_done(void* data, zwp_linux_dmabuf_feedback_v1*) { static_cast<App*>(data)->feedback_done = true; }
void fb_format_table(void* data, zwp_linux_dmabuf_feedback_v1*, std::int32_t fd, std::uint32_t size) {
    auto* app = static_cast<App*>(data);
    void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p != MAP_FAILED) {
        app->fmt_table.clear();
        const auto* e = static_cast<const std::uint8_t*>(p);
        for (std::uint32_t off = 0; off + 16 <= size; off += 16) {
            std::uint32_t f;
            std::uint64_t m;
            std::memcpy(&f, e + off, 4);
            std::memcpy(&m, e + off + 8, 8);
            app->fmt_table.emplace_back(f, m);
        }
        munmap(p, size);
    }
    close(fd);
}
void fb_main_device(void* data, zwp_linux_dmabuf_feedback_v1*, wl_array* dev) {
    auto* app = static_cast<App*>(data);
    if (dev->size >= sizeof(dev_t)) {
        dev_t d;
        std::memcpy(&d, dev->data, sizeof(d));
        app->feedback_log.push_back("main_device " + std::to_string(major(d)) + ":" + std::to_string(minor(d)));
    }
}
void fb_tranche_done(void* data, zwp_linux_dmabuf_feedback_v1*) {
    auto* app = static_cast<App*>(data);
    bool nv12_sand = false, nv12_lin = false, ab24_lin = false, ar24_lin = false;
    for (auto i : app->tranche_idx) {
        if (i >= app->fmt_table.size()) continue;
        const auto [f, m] = app->fmt_table[i];
        if (f == DRM_FORMAT_NV12 && fourcc_mod_broadcom_mod(m) == DRM_FORMAT_MOD_BROADCOM_SAND128) nv12_sand = true;
        if (f == DRM_FORMAT_NV12 && m == DRM_FORMAT_MOD_LINEAR) nv12_lin = true;
        if (f == DRM_FORMAT_ABGR8888 && m == DRM_FORMAT_MOD_LINEAR) ab24_lin = true;
        if (f == DRM_FORMAT_ARGB8888 && m == DRM_FORMAT_MOD_LINEAR) ar24_lin = true;
    }
    char b[256];
    std::snprintf(b, sizeof(b), "tranche flags=0x%x(%s) n=%zu NV12xSAND128=%d NV12xLINEAR=%d AB24xLINEAR=%d AR24xLINEAR=%d",
                  app->tranche_flags, (app->tranche_flags & ZWP_LINUX_DMABUF_FEEDBACK_V1_TRANCHE_FLAGS_SCANOUT) ? "SCANOUT" : "render",
                  app->tranche_idx.size(), nv12_sand, nv12_lin, ab24_lin, ar24_lin);
    app->feedback_log.push_back(b);
    app->tranche_idx.clear();
    app->tranche_flags = 0;
}
void fb_tranche_target(void* data, zwp_linux_dmabuf_feedback_v1*, wl_array* dev) {
    auto* app = static_cast<App*>(data);
    if (dev->size >= sizeof(dev_t)) {
        dev_t d;
        std::memcpy(&d, dev->data, sizeof(d));
        app->feedback_log.push_back("tranche_target_device " + std::to_string(major(d)) + ":" + std::to_string(minor(d)));
    }
}
void fb_tranche_formats(void* data, zwp_linux_dmabuf_feedback_v1*, wl_array* idx) {
    auto* app = static_cast<App*>(data);
    const auto* p = static_cast<const std::uint16_t*>(idx->data);
    for (std::size_t i = 0; i < idx->size / 2; ++i) app->tranche_idx.push_back(p[i]);
}
void fb_tranche_flags(void* data, zwp_linux_dmabuf_feedback_v1*, std::uint32_t flags) {
    static_cast<App*>(data)->tranche_flags = flags;
}
const zwp_linux_dmabuf_feedback_v1_listener kFbListener = {fb_done,           fb_format_table,  fb_main_device,
                                                           fb_tranche_done,   fb_tranche_target, fb_tranche_formats,
                                                           fb_tranche_flags};

// wl_buffer.release
void vbuf_release(void* data, wl_buffer*) {
    auto* vb = static_cast<VideoBuf*>(data);
    if (!vb->busy) return;
    vb->busy = false;
    vb->keepalive.reset(); // return AVFrame → decoder pool
    vb->release_count++;
    vb->app->held--;
}
const wl_buffer_listener kVbufListener = {vbuf_release};

void obuf_release(void* data, wl_buffer*) { static_cast<OverlayBuf*>(data)->busy = false; }
const wl_buffer_listener kObufListener = {obuf_release};

// zwp_linux_buffer_params_v1 (async create)
void params_created(void* data, zwp_linux_buffer_params_v1*, wl_buffer*) {
    auto* app = static_cast<App*>(data);
    app->create_done = true;
    app->last_create_failed = false;
}
void params_failed(void* data, zwp_linux_buffer_params_v1*) {
    auto* app = static_cast<App*>(data);
    app->create_done = true;
    app->last_create_failed = true;
}

wl_buffer* g_created = nullptr; // wl_buffer received via params.created (taken by create_dmabuf_buffer)
void params_created_capture(void* data, zwp_linux_buffer_params_v1* p, wl_buffer* b) {
    g_created = b;
    params_created(data, p, b);
}
const zwp_linux_buffer_params_v1_listener kParamsListener = {params_created_capture, params_failed};

// Create a dmabuf wl_buffer synchronously (roundtrip). Returns nullptr on failure.
wl_buffer* create_dmabuf_buffer(App& app, int w, int h, std::uint32_t fmt, std::uint32_t flags,
                                const std::vector<std::tuple<int, std::uint32_t, std::uint32_t, std::uint64_t>>& planes) {
    zwp_linux_buffer_params_v1* p = zwp_linux_dmabuf_v1_create_params(app.dmabuf);
    zwp_linux_buffer_params_v1_add_listener(p, &kParamsListener, &app);
    std::uint32_t idx = 0;
    for (const auto& [fd, off, stride, mod] : planes) {
        zwp_linux_buffer_params_v1_add(p, fd, idx++, off, stride, static_cast<std::uint32_t>(mod >> 32),
                                       static_cast<std::uint32_t>(mod & 0xffffffffu));
    }
    g_created = nullptr;
    app.create_done = false;
    zwp_linux_buffer_params_v1_create(p, w, h, fmt, flags);
    while (!app.create_done) {
        if (wl_display_roundtrip(app.display) < 0) break;
    }
    zwp_linux_buffer_params_v1_destroy(p);
    return app.last_create_failed ? nullptr : g_created;
}

// wp_presentation_feedback
void pf_sync_output(void*, struct wp_presentation_feedback*, wl_output*) {}
void pf_presented(void* data, struct wp_presentation_feedback* f, std::uint32_t sec_hi, std::uint32_t sec_lo,
                  std::uint32_t nsec, std::uint32_t refresh, std::uint32_t seq_hi, std::uint32_t seq_lo,
                  std::uint32_t flags) {
    auto* ctx = static_cast<FeedbackCtx*>(data);
    FrameRec& r = ctx->app->recs[ctx->rec];
    const std::uint64_t sec = (static_cast<std::uint64_t>(sec_hi) << 32) | sec_lo;
    r.presented_ns = static_cast<std::int64_t>(sec) * 1000000000LL + nsec;
    r.refresh_ns = refresh;
    r.seq = (static_cast<std::uint64_t>(seq_hi) << 32) | seq_lo;
    r.flags = flags;
    r.got_feedback = true;
    wp_presentation_feedback_destroy(f);
    forget_feedback(ctx);
    delete ctx;
}
void pf_discarded(void* data, struct wp_presentation_feedback* f) {
    auto* ctx = static_cast<FeedbackCtx*>(data);
    FrameRec& r = ctx->app->recs[ctx->rec];
    r.discarded = true;
    r.got_feedback = true;
    wp_presentation_feedback_destroy(f);
    forget_feedback(ctx);
    delete ctx;
}
const wp_presentation_feedback_listener kPfListener = {pf_sync_output, pf_presented, pf_discarded};

// ── Video ─────────────────────────────────────────────────────────────────────────────────
std::pair<ino_t, ino_t> buf_key(const yvo::DrmFrame& f) {
    struct stat sy{}, suv{};
    fstat(f.fd, &sy);
    fstat(f.uv_fd, &suv);
    return {sy.st_ino, suv.st_ino};
}

VideoBuf* video_buffer_for(App& app, const yvo::DrmFrame& f) {
    const auto key = buf_key(f);
    auto it = app.vbufs.find(key);
    if (it != app.vbufs.end()) return it->second.get();
    const std::uint64_t mod = app.args.sand_param
                                  ? DRM_FORMAT_MOD_BROADCOM_SAND128_COL_HEIGHT(static_cast<std::uint64_t>(app.args.sand_param))
                                  : f.modifier;
    const auto py = static_cast<std::uint32_t>(app.args.pitch_y >= 0 ? app.args.pitch_y : static_cast<long>(f.y_pitch));
    const auto puv = static_cast<std::uint32_t>(app.args.pitch_uv >= 0 ? app.args.pitch_uv : static_cast<long>(f.uv_pitch));
    wl_buffer* b = create_dmabuf_buffer(app, f.width, f.height, DRM_FORMAT_NV12, 0,
                                        {{f.fd, static_cast<std::uint32_t>(f.y_offset), py, mod},
                                         {f.uv_fd, static_cast<std::uint32_t>(f.uv_offset), puv, mod}});
    if (!b) {
        app.create_failed++;
        return nullptr;
    }
    auto vb = std::make_unique<VideoBuf>();
    vb->app = &app;
    vb->buf = b;
    wl_buffer_add_listener(b, &kVbufListener, vb.get());
    VideoBuf* raw = vb.get();
    app.vbufs[key] = std::move(vb);
    if (app.args.verbose)
        std::printf("[video] new wl_buffer #%zu fd=%d/%d off=%zu/%zu pitch=%u/%u mod=0x%016llx\n", app.vbufs.size(), f.fd,
                    f.uv_fd, f.y_offset, f.uv_offset, py, puv, static_cast<unsigned long long>(mod));
    return raw;
}

bool decode_ahead(App& app) {
    if (!app.ahead.empty()) return true;
    auto f = app.dec.next();
    if (!f) return false;
    app.ahead.push_back(std::move(*f));
    app.held++;
    app.held_max = std::max(app.held_max, app.held);
    return true;
}

// ── Overlay ───────────────────────────────────────────────────────────────────────────────
bool setup_overlay(App& app, int w, int h) {
    const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4;
    try {
        if (app.args.overlay_alloc == "dumb") {
            // card1 (vc4) dumb buffer: contiguous memory (CMA) + write-combined mapping → no CPU cache sync needed.
            // CREATE_DUMB/MAP_DUMB/PRIME export work without DRM master.
            app.card_fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
            if (app.card_fd < 0) throw std::runtime_error(std::string("open card1: ") + std::strerror(errno));
            for (int i = 0; i < app.args.overlay_bufs; ++i) {
                drm_mode_create_dumb c{};
                c.width = static_cast<std::uint32_t>(w);
                c.height = static_cast<std::uint32_t>(h);
                c.bpp = 32;
                if (drmIoctl(app.card_fd, DRM_IOCTL_MODE_CREATE_DUMB, &c) != 0)
                    throw std::runtime_error(std::string("CREATE_DUMB: ") + std::strerror(errno));
                if (c.pitch != static_cast<std::uint32_t>(w * 4))
                    throw std::runtime_error("dumb pitch != w*4 (the painter assumes a packed buffer)");
                drm_mode_map_dumb m{};
                m.handle = c.handle;
                if (drmIoctl(app.card_fd, DRM_IOCTL_MODE_MAP_DUMB, &m) != 0)
                    throw std::runtime_error(std::string("MAP_DUMB: ") + std::strerror(errno));
                void* p = mmap(nullptr, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, app.card_fd, static_cast<off_t>(m.offset));
                if (p == MAP_FAILED) throw std::runtime_error(std::string("mmap dumb: ") + std::strerror(errno));
                std::memset(p, 0, c.size);
                int fd = -1;
                if (drmPrimeHandleToFD(app.card_fd, c.handle, DRM_CLOEXEC | DRM_RDWR, &fd) != 0)
                    throw std::runtime_error(std::string("drmPrimeHandleToFD: ") + std::strerror(errno));
                auto ob = std::make_unique<OverlayBuf>();
                ob->app = &app;
                ob->idx = i;
                ob->ptr = static_cast<std::uint8_t*>(p);
                ob->fd = fd;
                ob->dumb_handle = c.handle;
                ob->map_len = c.size;
                app.ov_bufs.push_back(std::move(ob));
            }
            app.ov_backend = "card1 dumb(WC) PRIME AB24 LINEAR x" + std::to_string(app.args.overlay_bufs);
        } else {
            yvo::DmaHeapConfig cfg;
            cfg.heaps = {app.args.heap};
            cfg.align = 16384;
            cfg.name = "hvs_overlay";
            app.ov_pool = std::make_unique<yvo::DmaHeapPool>(cfg);
            for (int i = 0; i < app.args.overlay_bufs; ++i) {
                const int idx = app.ov_pool->allocate(bytes);
                const auto& hb = app.ov_pool->at(idx);
                auto ob = std::make_unique<OverlayBuf>();
                ob->app = &app;
                ob->idx = idx;
                ob->ptr = hb.ptr;
                ob->fd = hb.fd;
                ob->needs_sync = true;
                app.ov_bufs.push_back(std::move(ob));
            }
            app.ov_backend = "dma-heap(" + app.ov_pool->heap() + ", cached) AB24 LINEAR x" + std::to_string(app.args.overlay_bufs);
        }
        for (auto& ob : app.ov_bufs) {
            ob->buf = create_dmabuf_buffer(app, w, h, DRM_FORMAT_ABGR8888, 0,
                                           {{ob->fd, 0u, static_cast<std::uint32_t>(w * 4), DRM_FORMAT_MOD_LINEAR}});
            if (!ob->buf) {
                std::fprintf(stderr, "[overlay] dmabuf wl_buffer creation failed (%s)\n", app.args.overlay_alloc.c_str());
                return false;
            }
            wl_buffer_add_listener(ob->buf, &kObufListener, ob.get());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[overlay] %s\n", e.what());
        return false;
    }
    app.painter = std::make_unique<OverlayPainter>(w, h, app.args.overlay_bufs);
    return true;
}

// Draw one overlay buffer and attach+commit it to ov_surface (applied with the parent commit — sync subsurface)
void update_overlay(App& app, FrameRec& rec, double fps_now) {
    OverlayBuf* ob = nullptr;
    for (auto& b : app.ov_bufs)
        if (!b->busy) {
            ob = b.get();
            break;
        }
    if (!ob) {
        app.overlay_skips++;
        return;
    }
    static const std::vector<yvo::Detection> kEmpty;
    auto it = app.dets.find(rec.pts);
    const auto& dets = it != app.dets.end() ? it->second : kEmpty;
    char hud[96];
    std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %zu HVS-WL", fps_now, dets.size());
    const std::int64_t t0 = now_ns();
    std::int64_t t1 = 0, t2 = 0;
    if (ob->needs_sync) {
        yvo::DmaCpuAccess acc(ob->fd, yvo::DmaSyncDir::Write, false);
        t1 = now_ns();
        app.painter->paint(ob->ptr, ob->idx, dets, hud);
        t2 = now_ns();
        acc.finish();
    } else {
        t1 = now_ns();
        app.painter->paint(ob->ptr, ob->idx, dets, hud);
        t2 = now_ns();
    }
    const std::int64_t t3 = now_ns();
    rec.overlay_draw_ms = ms(t2 - t1);
    rec.overlay_sync_ms = ms(t3 - t0);
    rec.overlay_buf = ob->idx;
    rec.ndets = static_cast<int>(dets.size());
    ob->busy = true;
    wl_surface_attach(app.ov_surface, ob->buf, 0, 0);
    wl_surface_damage_buffer(app.ov_surface, 0, 0, INT32_MAX, INT32_MAX);
    wl_surface_commit(app.ov_surface);
}

void snapshot_state(App& app, const std::string& tag) {
    const std::string st = read_text("/sys/kernel/debug/dri/1/state");
    const std::string path = app.args.out_dir + "/" + app.args.label + "_state_" + tag + ".txt";
    write_text(path, st);
    const auto planes = parse_active_planes(st);
    const auto v = judge_placement(planes);
    std::string line = "[state " + tag + "] " + v.text + "\n" + describe_planes(planes);
    const std::string underrun = read_text("/sys/kernel/debug/dri/1/hvs_underrun");
    line += "  hvs_underrun: " + underrun;
    app.snapshots.push_back(line);
    std::printf("%s", line.c_str());
}

void request_frame(App& app);

void commit_video(App& app, std::int64_t now) {
    yvo::DrmFrame f = std::move(app.ahead.front());
    app.ahead.pop_front();
    FrameRec rec;
    rec.pts = f.pts;
    rec.decoded_ns = 0;
    VideoBuf* vb = video_buffer_for(app, f);
    if (!vb) {
        std::fprintf(stderr, "[video] frame %lld: wl_buffer creation failed (compositor rejected)\n", static_cast<long long>(f.pts));
        app.held--;
        app.finished = true;
        return;
    }
    if (vb->busy) {
        // The same dma-buf is still in the compositor — must not happen (we hold the AVFrame).
        std::fprintf(stderr, "[video] BUG: busy buffer reuse at frame %lld\n", static_cast<long long>(f.pts));
    }
    if (app.args.freeze_at >= 0 && f.pts == app.args.freeze_at) {
        // CPU reference conversion for color comparison (BT.709 limited → full RGB, same formula as display_convert.comp)
        const auto rgb = sand_to_rgb(f);
        if (!rgb.empty())
            write_ppm(app.args.out_dir + "/" + app.args.label + "_ref_f" + std::to_string(f.pts) + ".ppm", rgb.data(),
                      f.width, f.height);
    }
    vb->busy = true;
    vb->pts = f.pts;
    vb->keepalive = std::move(f.keepalive); // held until release (held was already counted in ahead)
    app.recs.push_back(rec);
    FrameRec& r = app.recs.back();
    const double el = app.t_start ? static_cast<double>(now - app.t_start) / 1e9 : 0.0;
    const double fps_now = el > 0 ? static_cast<double>(app.presented_video) / el : 0.0;
    if (app.args.overlay) update_overlay(app, r, fps_now);
    wl_surface_attach(app.surface, vb->buf, 0, 0);
    wl_surface_damage_buffer(app.surface, 0, 0, INT32_MAX, INT32_MAX);
    if (app.presentation) {
        auto* ctx = new FeedbackCtx{&app, app.recs.size() - 1};
        struct wp_presentation_feedback* fb = wp_presentation_feedback(app.presentation, app.surface);
        wp_presentation_feedback_add_listener(fb, &kPfListener, ctx);
        ctx->proxy = fb;
        app.live_feedback.push_back(ctx);
    }
    request_frame(app);
    r.held_at_commit = app.held;
    r.commit_ns = now_ns();
    wl_surface_commit(app.surface);
    if (!app.t_start) app.t_start = r.commit_ns;
    app.presented_video++;
}

void frame_done(void* data, wl_callback* cb, std::uint32_t);
const wl_callback_listener kFrameListener = {frame_done};

void request_frame(App& app) {
    wl_callback* cb = wl_surface_frame(app.surface);
    wl_callback_add_listener(cb, &kFrameListener, &app);
}

void on_tick(App& app) {
    const std::int64_t now = now_ns();
    const int n = app.presented_video;
    if (n == app.args.frames || app.finished || g_stop) {
        app.finished = true;
        return;
    }
    // measurement window start/end, snapshots
    if (n == 30 && app.meas_t0 == 0) {
        app.meas_t0 = now;
        app.meas_f0 = n;
        app.v3d_a = read_v3d_stats();
        app.cpu0 = self_cpu_ns();
        app.lab0 = proc_cpu_ns(app.labwc_pid);
    }
    if (app.frozen) {
        if (now < app.freeze_until) {
            request_frame(app);
            wl_surface_commit(app.surface);
            return;
        }
        app.frozen = false;
        app.t_start += (now - app.freeze_until) + app.args.freeze_ms * 1000000LL; // shift the schedule
    }
    // Source 30 fps schedule: frame n should be displayed at t_start + n*33.3ms. Commit half a vblank early.
    const bool due = (app.t_start == 0) ||
                     (static_cast<double>(now - app.t_start) + 8.0e6 >= static_cast<double>(n) * app.frame_period_ns);
    if (due && decode_ahead(app)) {
        commit_video(app, now);
        if (n + 1 == app.args.frames / 4 || n + 1 == app.args.frames / 2 || n + 1 == (3 * app.args.frames) / 4)
            snapshot_state(app, "f" + std::to_string(n + 1));
        if (app.args.freeze_at >= 0 && n == app.args.freeze_at) {
            app.frozen = true;
            app.freeze_until = now + app.args.freeze_ms * 1000000LL;
        }
    } else if (due) {
        app.finished = true; // EOF
    } else {
        app.idle_commits++;
        request_frame(app);
        wl_surface_commit(app.surface);
    }
}

void frame_done(void* data, wl_callback* cb, std::uint32_t) {
    wl_callback_destroy(cb);
    auto* app = static_cast<App*>(data);
    app->callbacks++;
    on_tick(*app);
}

} // namespace

int main(int argc, char** argv) {
    App app;
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // line-buffered logging (so output survives interruption during tee/redirect)
    if (!parse_args(argc, argv, app.args)) {
        usage(argv[0]);
        return 2;
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    Args& a = app.args;

    if (!a.dets.empty()) {
        std::size_t total = 0;
        app.dets = load_detections(a.dets, &total);
        std::printf("[dets] %s: %zu frames, %zu detections\n", a.dets.c_str(), app.dets.size(), total);
    }
    if (!app.dec.open(a.video)) {
        std::fprintf(stderr, "decoder open failed: %s\n", app.dec.error().c_str());
        return 1;
    }
    const double fr = app.dec.frame_rate();
    if (fr > 1) app.frame_period_ns = 1e9 / fr;

    app.display = wl_display_connect(nullptr);
    if (!app.display) {
        std::fprintf(stderr, "wl_display_connect failed (check WAYLAND_DISPLAY/XDG_RUNTIME_DIR)\n");
        return 1;
    }
    app.registry = wl_display_get_registry(app.display);
    wl_registry_add_listener(app.registry, &kRegListener, &app);
    wl_display_roundtrip(app.display);
    wl_display_roundtrip(app.display);
    std::printf("[globals]");
    for (const auto& g : app.globals) std::printf(" %s;", g.c_str());
    std::printf("\n");
    std::printf("[globals] dmabuf v%u presentation %s viewporter %s color_representation x%zu subcompositor %s\n",
                app.dmabuf_ver, app.presentation ? "yes" : "no", app.viewporter ? "yes" : "no",
                app.crm_cands.size(), app.subcompositor ? "yes" : "no");
    if (!app.compositor || !app.wm_base || !app.dmabuf) {
        std::fprintf(stderr, "required globals (wl_compositor/xdg_wm_base/zwp_linux_dmabuf_v1) missing\n");
        return 1;
    }
    for (auto& c : app.crm_cands) {
        const bool has709 = std::any_of(c->pairs.begin(), c->pairs.end(), [](const auto& pr) {
            return pr.first == WP_COLOR_REPRESENTATION_SURFACE_V1_COEFFICIENTS_BT709;
        });
        if (!app.color_repr_mgr && has709) app.color_repr_mgr = c->mgr;
        else wp_color_representation_manager_v1_destroy(c->mgr);
    }
    std::printf("[color-repr] managers advertised %zu, selected %s\n", app.crm_cands.size(),
                app.color_repr_mgr ? "one supporting BT.709" : "none");
    if (!app.crm_cands.empty()) {
        std::printf("[color-repr] supported coefficients/ranges:");
        for (const auto& s : app.crm_supported) std::printf(" %s", s.c_str());
        std::printf("\n");
    }
    app.labwc_pid = find_pid_by_comm("labwc");

    // surface
    app.surface = wl_compositor_create_surface(app.compositor);
    app.xsurface = xdg_wm_base_get_xdg_surface(app.wm_base, app.surface);
    xdg_surface_add_listener(app.xsurface, &kXsListener, &app);
    app.toplevel = xdg_surface_get_toplevel(app.xsurface);
    xdg_toplevel_add_listener(app.toplevel, &kTlListener, &app);
    xdg_toplevel_set_title(app.toplevel, "hvs_wl_probe");
    xdg_toplevel_set_app_id(app.toplevel, "hvs_wl_probe");
    if (a.fullscreen) xdg_toplevel_set_fullscreen(app.toplevel, nullptr);

    // dmabuf surface feedback
    zwp_linux_dmabuf_feedback_v1* feedback = nullptr;
    if (app.dmabuf_ver >= 4) {
        feedback = zwp_linux_dmabuf_v1_get_surface_feedback(app.dmabuf, app.surface);
        zwp_linux_dmabuf_feedback_v1_add_listener(feedback, &kFbListener, &app);
    }
    if (a.color_repr) {
        if (!app.color_repr_mgr) {
            std::printf("[color-repr] requested but labwc does not advertise wp_color_representation_manager_v1 — skipping\n");
        } else {
            app.color_repr = wp_color_representation_manager_v1_get_surface(app.color_repr_mgr, app.surface);
            wp_color_representation_surface_v1_set_coefficients_and_range(
                app.color_repr, WP_COLOR_REPRESENTATION_SURFACE_V1_COEFFICIENTS_BT709,
                a.color_range_full ? WP_COLOR_REPRESENTATION_SURFACE_V1_RANGE_FULL
                                   : WP_COLOR_REPRESENTATION_SURFACE_V1_RANGE_LIMITED);
            wp_color_representation_surface_v1_set_chroma_location(app.color_repr,
                                                                   WP_COLOR_REPRESENTATION_SURFACE_V1_CHROMA_LOCATION_TYPE_0);
            std::printf("[color-repr] BT.709 %s, chroma type_0 set\n", a.color_range_full ? "full" : "limited");
        }
    }
    wl_surface_commit(app.surface);
    while (!app.configured && wl_display_dispatch(app.display) >= 0) {}
    wl_display_roundtrip(app.display);
    std::printf("[xdg] configured %dx%d (fullscreen=%d)\n", app.conf_w, app.conf_h, a.fullscreen);

    int out_w = 1920, out_h = 1080;
    if (!a.fullscreen || a.vp_w > 0) {
        out_w = a.fullscreen ? a.vp_w : a.win_w;
        out_h = a.fullscreen ? a.vp_h : a.win_h;
        if (!app.viewporter) {
            std::fprintf(stderr, "wp_viewporter missing — cannot scale in windowed mode\n");
            return 1;
        }
        app.viewport = wp_viewporter_get_viewport(app.viewporter, app.surface);
        wp_viewport_set_source(app.viewport, wl_fixed_from_int(0), wl_fixed_from_int(0), wl_fixed_from_int(1920),
                               wl_fixed_from_int(1080));
        wp_viewport_set_destination(app.viewport, out_w, out_h);
        std::printf("[viewport] source 1920x1080 -> destination %dx%d\n", out_w, out_h);
    }

    if (a.overlay) {
        if (!app.subcompositor) {
            std::fprintf(stderr, "wl_subcompositor missing\n");
            return 1;
        }
        app.ov_surface = wl_compositor_create_surface(app.compositor);
        app.ov_sub = wl_subcompositor_get_subsurface(app.subcompositor, app.ov_surface, app.surface);
        wl_subsurface_set_position(app.ov_sub, 0, 0);
        wl_subsurface_set_sync(app.ov_sub);
        wl_region* empty = wl_compositor_create_region(app.compositor);
        wl_surface_set_input_region(app.ov_surface, empty);
        wl_region_destroy(empty);
        if (!setup_overlay(app, out_w, out_h)) return 1;
        std::printf("[overlay] backend %s, %dx%d\n", app.ov_backend.c_str(), out_w, out_h);
    }

    app.recs.reserve(static_cast<std::size_t>(a.frames) + 8);
    // first frame
    if (!decode_ahead(app)) {
        std::fprintf(stderr, "first frame decode failed: %s\n", app.dec.error().c_str());
        return 1;
    }
    {
        const yvo::DrmFrame& f = app.ahead.front();
        std::printf("[video] %dx%d coded_h=%d mod=0x%016llx Y fd=%d off=%zu pitch(colh)=%zu size=%zu | UV fd=%d off=%zu "
                    "pitch(colh)=%zu size=%zu\n",
                    f.width, f.height, f.coded_height, static_cast<unsigned long long>(f.modifier), f.fd, f.y_offset,
                    f.y_pitch, f.size, f.uv_fd, f.uv_offset, f.uv_pitch, f.uv_size);
    }
    commit_video(app, now_ns());
    wl_display_flush(app.display);

    bool grabbed = false;
    while (!app.finished && !app.closed && !g_stop) {
        if (wl_display_dispatch(app.display) < 0) {
            std::fprintf(stderr, "wl_display_dispatch failed: %s\n", std::strerror(errno));
            break;
        }
        if (app.frozen && !grabbed && now_ns() > app.freeze_until - a.freeze_ms * 1000000LL / 2) {
            grabbed = true;
            const std::string shot = a.out_dir + "/" + a.label + "_grim_f" + std::to_string(a.freeze_at) + ".ppm";
            const std::string cmd = "grim -t ppm -o HDMI-A-1 '" + shot + "' 2>&1";
            const int rc = std::system(cmd.c_str());
            std::printf("[freeze] frame %d: %s -> rc=%d\n", a.freeze_at, cmd.c_str(), rc);
            snapshot_state(app, "freeze");
        }
    }
    const std::int64_t t_end = now_ns();
    app.meas_t1 = t_end;
    app.meas_f1 = app.presented_video;
    app.v3d_b = read_v3d_stats();
    app.cpu1 = self_cpu_ns();
    app.lab1 = proc_cpu_ns(app.labwc_pid);
    // receive remaining feedback
    for (int i = 0; i < 6; ++i) wl_display_roundtrip(app.display);

    // ── summary ──
    const int nf = app.meas_f1 - app.meas_f0;
    const std::int64_t wall = app.meas_t1 - app.meas_t0;
    std::vector<double> intervals, latency, draw_ms, sync_ms, seq_gap;
    int zero_copy = 0, discarded = 0, no_fb = 0, late = 0, feedback_n = 0;
    std::int64_t prev_pres = 0;
    std::uint64_t prev_seq = 0;
    for (std::size_t i = 0; i < app.recs.size(); ++i) {
        const FrameRec& r = app.recs[i];
        if (!r.got_feedback) {
            ++no_fb;
            continue;
        }
        if (r.discarded) {
            ++discarded;
            prev_pres = 0;
            continue;
        }
        ++feedback_n;
        if (r.flags & WP_PRESENTATION_FEEDBACK_KIND_ZERO_COPY) ++zero_copy;
        if (r.presented_ns && r.commit_ns && app.pres_clock == CLOCK_MONOTONIC)
            latency.push_back(ms(r.presented_ns - r.commit_ns));
        if (prev_pres && i >= 30) {
            intervals.push_back(ms(r.presented_ns - prev_pres));
            const double g = static_cast<double>(r.seq - prev_seq);
            seq_gap.push_back(g);
            if (g > 2.0) ++late;
        }
        prev_pres = r.presented_ns;
        prev_seq = r.seq;
        if (r.overlay_buf >= 0 && i >= 30) {
            draw_ms.push_back(r.overlay_draw_ms);
            sync_ms.push_back(r.overlay_sync_ms);
        }
    }
    double render_ms = 0, csd_ms = 0;
    const std::string v3d = v3d_delta_str(app.v3d_a, app.v3d_b, wall, nf, &render_ms, &csd_ms);
    const double cpu_ms_f = nf > 0 ? ms(app.cpu1 - app.cpu0) / nf : 0;
    const double lab_ms_f = (nf > 0 && app.lab0 >= 0) ? ms(app.lab1 - app.lab0) / nf : -1;

    std::printf("\n== hvs_wl_probe [%s] overlay=%s window=%s color_repr=%s ==\n", a.label.c_str(), a.overlay ? "on" : "off",
                a.fullscreen ? "fullscreen" : (std::to_string(a.win_w) + "x" + std::to_string(a.win_h)).c_str(),
                a.color_repr ? (app.color_repr ? "on" : "unsupported") : "off");
    std::printf("frames committed %d (measured window %d frames, %.3f s, %.2f FPS)\n", app.presented_video, nf,
                wall / 1e9, nf > 0 ? nf / (wall / 1e9) : 0.0);
    std::printf("wl_buffers video %zu (cache by dmabuf inode), create_failed %lld\n", app.vbufs.size(),
                static_cast<long long>(app.create_failed));
    std::printf("feedback presented %d, discarded %d, missing %d, ZERO_COPY %d/%d\n", feedback_n, discarded, no_fb,
                zero_copy, feedback_n);
    std::printf("present interval ms %s\n", summary_str(summarize(intervals)).c_str());
    std::printf("present seq gap     %s  (2 = 30fps on 60Hz; late(>2) %d)\n", summary_str(summarize(seq_gap)).c_str(), late);
    std::printf("commit->present ms  %s\n", summary_str(summarize(latency)).c_str());
    std::printf("held frames max %d (decoder limit ~13)\n", app.held_max);
    std::printf("frame callbacks %lld, idle commits %lld\n", static_cast<long long>(app.callbacks),
                static_cast<long long>(app.idle_commits));
    if (a.overlay) {
        std::printf("overlay backend %s, skips %lld\n", app.ov_backend.c_str(), static_cast<long long>(app.overlay_skips));
        std::printf("overlay draw ms     %s\n", summary_str(summarize(draw_ms)).c_str());
        std::printf("overlay draw+sync ms %s\n", summary_str(summarize(sync_ms)).c_str());
    }
    std::printf("V3D (system-wide, measured window) %s\n", v3d.c_str());
    std::printf("V3D render ms/frame %.3f, csd ms/frame %.3f\n", render_ms, csd_ms);
    std::printf("CPU probe ms/frame %.3f, labwc ms/frame %.3f (pid %d)\n", cpu_ms_f, lab_ms_f, app.labwc_pid);
    std::printf("dmabuf surface feedback:\n");
    for (const auto& l : app.feedback_log) std::printf("  %s\n", l.c_str());

    std::printf("RESULT_WL,%s,%s,%s,%s,%d,%.2f,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                a.label.c_str(), a.overlay ? "on" : "off", a.fullscreen ? "fs" : "win",
                a.color_repr ? (app.color_repr ? "on" : "na") : "off", nf, nf > 0 ? nf / (wall / 1e9) : 0.0, zero_copy,
                feedback_n, discarded, late, summarize(intervals).median, summarize(intervals).p99,
                summarize(latency).median, summarize(draw_ms).median, app.held_max, render_ms, csd_ms, cpu_ms_f,
                lab_ms_f, summarize(sync_ms).median);

    // CSV
    const std::string csv = a.out_dir + "/" + a.label + "_frames.csv";
    if (FILE* f = std::fopen(csv.c_str(), "w")) {
        std::fprintf(f, "idx,pts,commit_ns,presented_ns,seq,flags,refresh_ns,discarded,held_at_commit,overlay_buf,ndets,"
                        "overlay_draw_ms,overlay_sync_ms\n");
        for (std::size_t i = 0; i < app.recs.size(); ++i) {
            const auto& r = app.recs[i];
            std::fprintf(f, "%zu,%lld,%lld,%lld,%llu,%u,%u,%d,%d,%d,%d,%.4f,%.4f\n", i, static_cast<long long>(r.pts),
                         static_cast<long long>(r.commit_ns), static_cast<long long>(r.presented_ns),
                         static_cast<unsigned long long>(r.seq), r.flags, r.refresh_ns, r.discarded ? 1 : 0,
                         r.held_at_commit, r.overlay_buf, r.ndets, r.overlay_draw_ms, r.overlay_sync_ms);
        }
        std::fclose(f);
    }

    // cleanup
    if (feedback) zwp_linux_dmabuf_feedback_v1_destroy(feedback);
    if (app.color_repr) wp_color_representation_surface_v1_destroy(app.color_repr);
    if (app.viewport) wp_viewport_destroy(app.viewport);
    for (auto& ob : app.ov_bufs) {
        if (ob->buf) wl_buffer_destroy(ob->buf);
        if (ob->dumb_handle) {
            munmap(ob->ptr, ob->map_len);
            close(ob->fd);
            drm_mode_destroy_dumb d{};
            d.handle = ob->dumb_handle;
            drmIoctl(app.card_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
        }
    }
    if (app.card_fd >= 0) close(app.card_fd);
    if (app.ov_sub) wl_subsurface_destroy(app.ov_sub);
    if (app.ov_surface) wl_surface_destroy(app.ov_surface);
    xdg_toplevel_destroy(app.toplevel);
    xdg_surface_destroy(app.xsurface);
    wl_surface_destroy(app.surface);
    wl_display_roundtrip(app.display);
    for (auto& [k, vb] : app.vbufs) {
        wl_buffer_destroy(vb->buf);
        vb->keepalive.reset();
    }
    app.ahead.clear();
    for (FeedbackCtx* ctx : app.live_feedback) {
        wp_presentation_feedback_destroy(ctx->proxy);
        delete ctx;
    }
    app.live_feedback.clear();
    if (app.color_repr_mgr) wp_color_representation_manager_v1_destroy(app.color_repr_mgr);
    if (app.viewporter) wp_viewporter_destroy(app.viewporter);
    if (app.presentation) wp_presentation_destroy(app.presentation);
    zwp_linux_dmabuf_v1_destroy(app.dmabuf);
    xdg_wm_base_destroy(app.wm_base);
    if (app.subcompositor) wl_subcompositor_destroy(app.subcompositor);
    wl_compositor_destroy(app.compositor);
    wl_registry_destroy(app.registry);
    wl_display_roundtrip(app.display);
    wl_display_disconnect(app.display);
    return 0;
}
