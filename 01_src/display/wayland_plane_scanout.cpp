#include "display/wayland_plane_scanout.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <ctime>
#include <stdexcept>

#include <sys/stat.h>

#include <wayland-client.h>

#include "display/linux-dmabuf-client-protocol.h"
#include "display/presentation-time-client-protocol.h"
#include "display/viewporter-client-protocol.h"
#ifdef YVO_HAVE_COLOR_REPRESENTATION
#include "display/color-representation-client-protocol.h"
#endif

namespace yvo::display {

namespace {

constexpr std::uint32_t fourcc(char a, char b, char c, char d) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24);
}
constexpr std::uint32_t kDrmFormatNv12 = fourcc('N', 'V', '1', '2');

using SteadyClock = std::chrono::steady_clock;

} // namespace

// ── Creation / destruction ────────────────────────────────────────────────────────────────────
WaylandPlaneScanout::WaylandPlaneScanout(WaylandWindow& win, const WaylandPlaneScanoutOptions& opt)
    : win_(win), opt_(opt) {
    if (!win_.is_open()) throw std::runtime_error("wayland-planes: Wayland window is not open");
    if (win_.linux_dmabuf() == nullptr || win_.linux_dmabuf_version() < 2)
        throw std::runtime_error("wayland-planes: compositor does not advertise zwp_linux_dmabuf_v1 (v2+)");
    if (win_.subcompositor() == nullptr)
        throw std::runtime_error("wayland-planes: compositor does not advertise wl_subcompositor");
    out_w_ = win_.width();
    out_h_ = win_.height();

    // Box subsurface: above the parent, (0,0), sync (applied together with the parent commit), input passes through to the parent (empty input region).
    ov_surface_ = wl_compositor_create_surface(win_.compositor());
    ov_sub_ = wl_subcompositor_get_subsurface(win_.subcompositor(), ov_surface_, win_.surface());
    wl_subsurface_set_position(ov_sub_, 0, 0);
    wl_subsurface_set_sync(ov_sub_);
    wl_region* empty = wl_compositor_create_region(win_.compositor());
    wl_surface_set_input_region(ov_surface_, empty);
    wl_region_destroy(empty);

    setup_color();   // before the first buffer commit (double-buffered state, so it is applied with the first commit)
    wl_display_flush(win_.display());
}

WaylandPlaneScanout::~WaylandPlaneScanout() {
    try {
        if (!shut_) shutdown(0);
    } catch (...) {
    }
    destroy_all();
}

void WaylandPlaneScanout::setup_color() {
#ifdef YVO_HAVE_COLOR_REPRESENTATION
    wp_color_representation_manager_v1* mgr = win_.color_representation_manager();
    const bool want_limited = opt_.color_range == ColorRangeRequest::Limited;
    const std::uint32_t range = want_limited ? WP_COLOR_REPRESENTATION_SURFACE_V1_RANGE_LIMITED
                                             : WP_COLOR_REPRESENTATION_SURFACE_V1_RANGE_FULL;
    const char* range_s = want_limited ? "limited" : "full";
    const char* why = opt_.color_range == ColorRangeRequest::Auto
                          ? " (auto = fullhack: labwc 0.20.1~0.20.2 inverts it and loads limited)"
                          : "";
    if (mgr == nullptr) {
        color_desc_ = win_.color_managers_advertised() > 0 ? "none (no manager supporting BT.709)"
                                                           : "none (wp_color_representation not advertised)";
        return;
    }
    if (!win_.color_pair_supported(WP_COLOR_REPRESENTATION_SURFACE_V1_COEFFICIENTS_BT709, range)) {
        color_desc_ = std::string("none (BT.709/") + range_s + " combination not advertised — not sent. supported: " +
                      win_.color_pairs_text() + ")";
        return;
    }
    color_surf_ = wp_color_representation_manager_v1_get_surface(mgr, win_.surface());
    wp_color_representation_surface_v1_set_coefficients_and_range(
        color_surf_, WP_COLOR_REPRESENTATION_SURFACE_V1_COEFFICIENTS_BT709, range);
    // HEVC 4:2:0 default chroma location (left, type 0). Same value as hvs_wl_probe.
    wp_color_representation_surface_v1_set_chroma_location(color_surf_,
                                                           WP_COLOR_REPRESENTATION_SURFACE_V1_CHROMA_LOCATION_TYPE_0);
    color_desc_ = std::string("BT.709/") + range_s + " requested" + why + " [managers advertised " +
                  std::to_string(win_.color_managers_advertised()) + "]";
#else
    color_desc_ = "none (built without color-representation-v1.xml)";
#endif
}

void WaylandPlaneScanout::destroy_all() {
    for (auto& [p, f] : feedback_) wp_presentation_feedback_destroy(f->fb);
    feedback_.clear();
    for (wl_callback* cb : frame_cbs_) wl_callback_destroy(cb);
    frame_cbs_.clear();
    for (auto& o : overlay_)
        if (o->buf) wl_buffer_destroy(o->buf);
    overlay_.clear();
    for (auto& [k, v] : video_)
        if (v->buf) wl_buffer_destroy(v->buf);
    video_.clear();
#ifdef YVO_HAVE_COLOR_REPRESENTATION
    if (color_surf_) wp_color_representation_surface_v1_destroy(color_surf_);
#endif
    color_surf_ = nullptr;
    if (viewport_) wp_viewport_destroy(viewport_);
    viewport_ = nullptr;
    if (ov_sub_) wl_subsurface_destroy(ov_sub_);
    ov_sub_ = nullptr;
    if (ov_surface_) wl_surface_destroy(ov_surface_);
    ov_surface_ = nullptr;
    // No roundtrip (infinite wait if the compositor is stalled). Flush only — WaylandWindow::close drops the connection.
    if (win_.display()) wl_display_flush(win_.display());
}

VkExtent2D WaylandPlaneScanout::output_extent() const {
    return VkExtent2D{static_cast<std::uint32_t>(out_w_ > 0 ? out_w_ : 0), static_cast<std::uint32_t>(out_h_ > 0 ? out_h_ : 0)};
}

// ── dma-buf → wl_buffer ───────────────────────────────────────────────────────────────────────
void WaylandPlaneScanout::on_params_created(void* data, zwp_linux_buffer_params_v1* /*p*/, wl_buffer* b) {
    auto* req = static_cast<CreateReq*>(data);
    req->buf = b;
    req->done = true;
}

void WaylandPlaneScanout::on_params_failed(void* data, zwp_linux_buffer_params_v1* /*p*/) {
    auto* req = static_cast<CreateReq*>(data);
    req->failed = true;
    req->done = true;
}

wl_buffer* WaylandPlaneScanout::create_dmabuf_buffer(int w, int h, std::uint32_t fmt,
                                                     const std::vector<PlaneDesc>& planes) {
    static const zwp_linux_buffer_params_v1_listener listener = [] {
        zwp_linux_buffer_params_v1_listener l{};
        l.created = &WaylandPlaneScanout::on_params_created;
        l.failed = &WaylandPlaneScanout::on_params_failed;
        return l;
    }();
    CreateReq req;   // result for this request only. The listener data points to it (no global state).
    zwp_linux_buffer_params_v1* p = zwp_linux_dmabuf_v1_create_params(win_.linux_dmabuf());
    zwp_linux_buffer_params_v1_add_listener(p, &listener, &req);
    std::uint32_t idx = 0;
    for (const PlaneDesc& d : planes) {
        zwp_linux_buffer_params_v1_add(p, d.fd, idx++, d.offset, d.stride, static_cast<std::uint32_t>(d.modifier >> 32),
                                       static_cast<std::uint32_t>(d.modifier & 0xffffffffu));
    }
    zwp_linux_buffer_params_v1_create(p, w, h, fmt, 0);
    // Bounded wait: give up if created/failed does not arrive within kCreateTimeoutMs (no infinite roundtrip).
    const auto deadline = SteadyClock::now() + std::chrono::milliseconds(kCreateTimeoutMs);
    bool ok = true;
    while (!req.done && ok && SteadyClock::now() < deadline) ok = win_.poll(10);
    // params is destroyed here whether or not a result was received. The new_id of a late created is discarded by libwayland as a zombie proxy.
    zwp_linux_buffer_params_v1_destroy(p);
    if (!req.done || req.failed || req.buf == nullptr) {
        ++create_failed_;
        std::fprintf(stderr, "[wayland-planes] dma-buf wl_buffer creation failed (%dx%d fourcc=0x%08x planes=%zu: %s)\n", w, h,
                     fmt, planes.size(), !req.done ? (ok ? "timeout" : "connection error") : "rejected by compositor (failed)");
        return nullptr;
    }
    return req.buf;
}

void WaylandPlaneScanout::register_overlay(const OverlayPlane& ov) {
    static const wl_buffer_listener listener = [] {
        wl_buffer_listener l{};
        l.release = &WaylandPlaneScanout::on_overlay_release;
        return l;
    }();
    overlay_.clear();
    for (int i = 0; i < ov.ring_size(); ++i) {
        const OverlayPlaneBuffer& b = ov.buffer(i);
        // The fd is passed while still owned by OverlayPlane — libwayland dups it when marshalling (header "fd ownership").
        wl_buffer* wb = create_dmabuf_buffer(static_cast<int>(b.width), static_cast<int>(b.height), b.fourcc,
                                             {PlaneDesc{b.fd, b.offset, b.pitch, b.modifier}});
        if (wb == nullptr) throw std::runtime_error("wayland-planes: failed to create box plane (AB24 LINEAR) wl_buffer");
        auto o = std::make_unique<OverlayBuf>();
        o->self = this;
        o->idx = i;
        o->buf = wb;
        wl_buffer_add_listener(wb, &listener, o.get());
        overlay_.push_back(std::move(o));
    }
}

WaylandPlaneScanout::VideoBuf* WaylandPlaneScanout::video_buffer_for(const DrmFrame& f) {
    static const wl_buffer_listener listener = [] {
        wl_buffer_listener l{};
        l.release = &WaylandPlaneScanout::on_video_release;
        return l;
    }();
    struct stat sy{}, suv{};
    if (fstat(f.fd, &sy) != 0 || fstat(f.uv_fd, &suv) != 0) return nullptr;
    const VideoKey key{static_cast<std::uint64_t>(sy.st_dev), static_cast<std::uint64_t>(sy.st_ino),
                       static_cast<std::uint64_t>(suv.st_dev), static_cast<std::uint64_t>(suv.st_ino)};
    auto it = video_.find(key);
    if (it != video_.end()) {
        VideoBuf* vb = it->second.get();
        if (vb->w == f.width && vb->h == f.height && vb->modifier == f.modifier) return vb;
        // Same dma-buf but the descriptor changed (resolution change) — recreate only when not held.
        if (vb->busy) return vb;   // the caller rejects it as busy
        wl_buffer_destroy(vb->buf);
        video_.erase(it);
    }
    if (f.drm_format != kDrmFormatNv12 || f.fd < 0 || f.uv_fd < 0) {
        std::fprintf(stderr, "[wayland-planes] unsupported frame (fourcc=0x%08x fd=%d/%d)\n", f.drm_format, f.fd, f.uv_fd);
        return nullptr;
    }
    // SAND128: modifier parameter 0 + each plane's pitch field = SAND column height (Y 1088 / UV 544). This is the kernel vc4 HVS6 convention, and
    // hvs_wl_probe measurements show labwc puts it on the HVS plane as-is (frame_types.hpp DrmFrame comment).
    wl_buffer* b = create_dmabuf_buffer(
        f.width, f.height, kDrmFormatNv12,
        {PlaneDesc{f.fd, static_cast<std::uint32_t>(f.y_offset), static_cast<std::uint32_t>(f.y_pitch), f.modifier},
         PlaneDesc{f.uv_fd, static_cast<std::uint32_t>(f.uv_offset), static_cast<std::uint32_t>(f.uv_pitch),
                   f.modifier}});
    if (b == nullptr) return nullptr;
    auto vb = std::make_unique<VideoBuf>();
    vb->self = this;
    vb->buf = b;
    vb->w = f.width;
    vb->h = f.height;
    vb->modifier = f.modifier;
    wl_buffer_add_listener(b, &listener, vb.get());
    VideoBuf* raw = vb.get();
    video_.emplace(key, std::move(vb));
    ++video_created_;
    if (opt_.verbose)
        std::printf("[wayland-planes] new video wl_buffer #%llu %dx%d mod=0x%016llx pitch(colh)=%zu/%zu\n",
                    static_cast<unsigned long long>(video_created_), f.width, f.height,
                    static_cast<unsigned long long>(f.modifier), f.y_pitch, f.uv_pitch);
    return raw;
}

void WaylandPlaneScanout::update_viewport(int buf_w, int buf_h) {
    if (buf_w == vp_buf_w_ && buf_h == vp_buf_h_) return;
    vp_buf_w_ = buf_w;
    vp_buf_h_ = buf_h;
    const bool scale = buf_w != out_w_ || buf_h != out_h_;
    if (scale && win_.viewporter() != nullptr) {
        if (viewport_ == nullptr) viewport_ = wp_viewporter_get_viewport(win_.viewporter(), win_.surface());
        wp_viewport_set_destination(viewport_, out_w_, out_h_);   // the HVS scales
    } else if (viewport_ != nullptr) {
        wp_viewport_set_destination(viewport_, -1, -1);           // unset (1:1)
    } else if (scale) {
        std::fprintf(stderr, "[wayland-planes] warning: video %dx%d ≠ output %dx%d but there is no wp_viewporter — attaching without scaling\n",
                     buf_w, buf_h, out_w_, out_h_);
    }
}

// ── Commit ────────────────────────────────────────────────────────────────────────────────────
bool WaylandPlaneScanout::commit(const DrmFrame& video, std::uint64_t token, int overlay_idx) {
    static const wl_callback_listener frame_listener = [] {
        wl_callback_listener l{};
        l.done = &WaylandPlaneScanout::on_frame_done;
        return l;
    }();
    static const wp_presentation_feedback_listener fb_listener = [] {
        wp_presentation_feedback_listener l{};
        l.sync_output = &WaylandPlaneScanout::on_fb_sync_output;
        l.presented = &WaylandPlaneScanout::on_fb_presented;
        l.discarded = &WaylandPlaneScanout::on_fb_discarded;
        return l;
    }();
    if (shut_) return false;
    // Secure the video buffer first — on failure we must reject without attaching anything so the box ring state does not diverge.
    VideoBuf* vb = video_buffer_for(video);
    if (vb == nullptr) return false;
    if (vb->busy) {
        ++busy_reuse_;
        std::fprintf(stderr, "[wayland-planes] BUG: tried to re-attach a video buffer still held by the compositor (token %llu) — rejected\n",
                     static_cast<unsigned long long>(token));
        return false;
    }
    OverlayBuf* ob = nullptr;
    if (overlay_idx >= 0) {
        if (overlay_idx >= static_cast<int>(overlay_.size())) return false;
        ob = overlay_[static_cast<std::size_t>(overlay_idx)].get();
        if (ob->busy) {   // diverged from the OverlayPlane ring state — must not happen
            ++busy_reuse_;
            return false;
        }
    }

    // ① Box subsurface: since it is sync, this commit is cached and applied together with the parent commit.
    if (ob != nullptr) {
        wl_surface_attach(ov_surface_, ob->buf, 0, 0);
        wl_surface_damage_buffer(ov_surface_, 0, 0, INT32_MAX, INT32_MAX);
        wl_surface_commit(ov_surface_);
        ob->busy = true;
    }
    // ② Parent: video + frame callback + presentation feedback → commit (atomic application)
    update_viewport(video.width, video.height);
    wl_surface_attach(win_.surface(), vb->buf, 0, 0);
    wl_surface_damage_buffer(win_.surface(), 0, 0, INT32_MAX, INT32_MAX);
    wl_callback* cb = wl_surface_frame(win_.surface());
    wl_callback_add_listener(cb, &frame_listener, this);
    frame_cbs_.insert(cb);
    if (win_.presentation() != nullptr) {
        auto f = std::make_unique<Feedback>();
        f->self = this;
        f->token = token;
        f->fb = wp_presentation_feedback(win_.presentation(), win_.surface());
        wp_presentation_feedback_add_listener(f->fb, &fb_listener, f.get());
        feedback_.emplace(f->fb, std::move(f));
    }
    wl_surface_commit(win_.surface());
    vb->busy = true;
    vb->token = token;
    wl_display_flush(win_.display());   // non-blocking (on EAGAIN the next poll sends the rest)
    return true;
}

bool WaylandPlaneScanout::dispatch(int timeout_ms) { return win_.poll(timeout_ms); }

// ── Events ────────────────────────────────────────────────────────────────────────────────────
void WaylandPlaneScanout::on_video_release(void* data, wl_buffer* /*b*/) {
    auto* vb = static_cast<VideoBuf*>(data);
    if (!vb->busy) return;
    vb->busy = false;
    if (vb->self->listener_) vb->self->listener_->on_video_released(vb->token);
}

void WaylandPlaneScanout::on_overlay_release(void* data, wl_buffer* /*b*/) {
    auto* ob = static_cast<OverlayBuf*>(data);
    if (!ob->busy) return;
    ob->busy = false;
    if (ob->self->listener_) ob->self->listener_->on_overlay_released(ob->idx);
}

void WaylandPlaneScanout::on_frame_done(void* data, wl_callback* cb, std::uint32_t /*t*/) {
    auto* self = static_cast<WaylandPlaneScanout*>(data);
    self->frame_cbs_.erase(cb);
    wl_callback_destroy(cb);
    if (self->listener_) self->listener_->on_frame_done();
}

void WaylandPlaneScanout::on_fb_sync_output(void* /*data*/, struct wp_presentation_feedback* /*f*/, wl_output* /*o*/) {}

void WaylandPlaneScanout::on_fb_presented(void* data, struct wp_presentation_feedback* f, std::uint32_t sec_hi,
                                          std::uint32_t sec_lo, std::uint32_t nsec, std::uint32_t refresh,
                                          std::uint32_t seq_hi, std::uint32_t seq_lo, std::uint32_t flags) {
    auto* fbc = static_cast<Feedback*>(data);
    WaylandPlaneScanout* self = fbc->self;
    const std::uint64_t token = fbc->token;
    const std::uint64_t sec = (static_cast<std::uint64_t>(sec_hi) << 32) | sec_lo;
    const auto t_ns = static_cast<std::int64_t>(sec) * 1000000000LL + static_cast<std::int64_t>(nsec);
    // presented/discarded are destructor events — we destroy the proxy. fbc also goes away here.
    wp_presentation_feedback_destroy(f);
    self->feedback_.erase(f);
    if (self->listener_)
        self->listener_->on_presented(token, t_ns, self->win_.presentation_clock() == CLOCK_MONOTONIC,
                                      (flags & WP_PRESENTATION_FEEDBACK_KIND_ZERO_COPY) != 0,
                                      (flags & WP_PRESENTATION_FEEDBACK_KIND_VSYNC)   // no vsync: seq means nothing
                                          ? (static_cast<std::uint64_t>(seq_hi) << 32) | seq_lo : 0, refresh);   // 0 = none
}

void WaylandPlaneScanout::on_fb_discarded(void* data, struct wp_presentation_feedback* f) {
    auto* fbc = static_cast<Feedback*>(data);
    WaylandPlaneScanout* self = fbc->self;
    const std::uint64_t token = fbc->token;
    wp_presentation_feedback_destroy(f);
    self->feedback_.erase(f);
    if (self->listener_) self->listener_->on_discarded(token);
}

// ── Shutdown ──────────────────────────────────────────────────────────────────────────────────
void WaylandPlaneScanout::shutdown(int timeout_ms) {
    if (shut_) return;
    shut_ = true;
    if (win_.display() == nullptr || win_.surface() == nullptr) return;
    // Take it off the screen: commit without a buffer → the compositor releases the buffers it holds. (sub first, applied by the parent commit)
    if (ov_surface_ != nullptr) {
        wl_surface_attach(ov_surface_, nullptr, 0, 0);
        wl_surface_commit(ov_surface_);
    }
    wl_surface_attach(win_.surface(), nullptr, 0, 0);
    wl_surface_commit(win_.surface());
    wl_display_flush(win_.display());
    auto busy = [&] {
        for (const auto& [k, v] : video_)
            if (v->busy) return true;
        for (const auto& o : overlay_)
            if (o->busy) return true;
        return false;
    };
    const auto deadline = SteadyClock::now() + std::chrono::milliseconds(std::max(timeout_ms, 0));
    while (busy() && SteadyClock::now() < deadline)
        if (!win_.poll(10)) break;
    if (busy() && opt_.verbose)
        std::printf("[wayland-planes] shutdown: buffers not released within %d ms remain — dropping them forcibly\n", timeout_ms);
    // Events that arrive afterwards are not received (libwayland discards them once the proxy is destroyed).
    for (auto& [p, f] : feedback_) wp_presentation_feedback_destroy(f->fb);
    feedback_.clear();
    for (wl_callback* cb : frame_cbs_) wl_callback_destroy(cb);
    frame_cbs_.clear();
    listener_ = nullptr;
}

void WaylandPlaneScanout::clear_video_cache() {
    for (auto& [k, v] : video_)
        if (v->buf) wl_buffer_destroy(v->buf);
    video_.clear();
    if (win_.display()) wl_display_flush(win_.display());
}

std::string WaylandPlaneScanout::describe() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%dx%d fullscreen | video=NV12/SAND128 dma-buf -> parent wl_surface (zwp_linux_dmabuf v%u) | "
                  "presentation=%s viewporter=%s | color=",
                  out_w_, out_h_, win_.linux_dmabuf_version(),
                  win_.presentation() ? (win_.presentation_clock() == CLOCK_MONOTONIC ? "yes(MONOTONIC)" : "yes(other clock)")
                                      : "no",
                  win_.viewporter() ? "yes" : "no");
    return buf + color_desc_;
}

} // namespace yvo::display
