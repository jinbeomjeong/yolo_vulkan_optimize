// hvs_wl_probe: 디코더 SAND128 NV12 dma-buf 를 그대로 wl_buffer 로 감싸 labwc 에 넘기고,
// labwc(wlroots)가 그것을 HVS 하드웨어 평면에 올리는지(direct scanout) GL 로 합성하는지 측정한다.
//
//   video surface  : fullscreen xdg_toplevel. zwp_linux_buffer_params_v1(NV12, SAND128, 평면별 offset/pitch)
//   overlay        : ARGB 계열(AB24 = 메모리 R,G,B,A) wl_subsurface. dma-heap(CMA) dmabuf 링에 CPU 로 그린다.
//   pacing         : wl_surface.frame 콜백(≈vblank) 마다 소스 30 fps 시간표를 보고 새 프레임을 commit
//   측정           : wp_presentation 피드백(표시 시각, seq, ZERO_COPY 플래그), wl_buffer.release 까지의 보유 수,
//                    V3D gpu_stats 큐별 사용률, 이 프로세스·labwc CPU 시간, debugfs state 스냅샷(평면 배치)
//
// 파이프라인 본체와 무관한 독립 프로브다. NPU 를 쓰지 않고 detections.jsonl 을 재생한다.
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
    int vp_w = 0, vp_h = 0; // fullscreen 에서도 viewport destination 을 이 크기로 (0 = 안 씀)
    bool color_repr = false;
    bool color_range_full = false; // wp_color_representation 에 보낼 range (기본 limited)
    int overlay_bufs = 3;
    int sand_param = 0;      // modifier 파라미터(열 높이). 0 이면 pitch 필드가 열 높이(HVS6 커널 관례)
    long pitch_y = -1;       // -1 = 디코더 descriptor 원값(열 높이)
    long pitch_uv = -1;
    int freeze_at = -1;      // 이 프레임을 표시한 뒤 freeze_ms 동안 멈추고 grim 으로 캡처
    int freeze_ms = 2500;
    std::string heap = "/dev/dma_heap/linux,cma";
    std::string overlay_alloc = "heap"; // heap = dma-heap(CMA, 캐시 매핑) / dumb = card1 dumb buffer(WC 매핑) 를 PRIME export
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

// ── 프레임 기록 ────────────────────────────────────────────────────────────────────────────
struct FrameRec {
    std::int64_t pts = -1;
    std::int64_t decoded_ns = 0;   // decoder.next() 반환 시각
    std::int64_t commit_ns = 0;    // 부모 surface commit 시각
    std::int64_t presented_ns = 0; // wp_presentation.presented (clock = presentation clock_id)
    std::uint64_t seq = 0;
    std::uint32_t flags = 0;
    std::uint32_t refresh_ns = 0;
    bool discarded = false;
    bool got_feedback = false;
    int held_at_commit = 0;
    double overlay_draw_ms = 0;    // CPU 그리기(지우기 + 그리기) 시간
    double overlay_sync_ms = 0;    // DMA_BUF_SYNC START/END 포함 총 시간
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
    std::uint8_t* ptr = nullptr; // CPU 매핑 (w*h*4 패킹)
    int fd = -1;                 // dma-buf fd (heap: 풀 소유 / dumb: 이 구조체 소유)
    bool needs_sync = false;     // heap(캐시 매핑)이면 DMA_BUF_SYNC 필요, dumb(WC 매핑)이면 불필요
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
    // 전역
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
    wp_color_representation_manager_v1* color_repr_mgr = nullptr; // 선택된(BT.709 limited 를 지원하는) 관리자
    std::vector<std::string> globals;
    std::vector<std::string> crm_supported;
    // labwc 0.20.1 은 wp_color_representation_manager_v1 전역을 **두 개** 광고한다(실측). 둘 다 bind 해서
    // 지원 목록을 받은 뒤 BT.709 를 지원하는 쪽을 고른다.
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

    // dmabuf 피드백(표면별 scanout tranche)
    std::vector<std::pair<std::uint32_t, std::uint64_t>> fmt_table;
    std::uint32_t tranche_flags = 0;
    std::vector<std::uint16_t> tranche_idx;
    std::vector<std::string> feedback_log;
    bool feedback_done = false;

    // 비디오
    yvo::FfmpegDrmDecoder dec;
    std::map<std::pair<ino_t, ino_t>, std::unique_ptr<VideoBuf>> vbufs;
    std::deque<yvo::DrmFrame> ahead; // 미리 디코드한 프레임(1개)
    int held = 0;                    // 우리가 들고 있는 AVFrame 수(busy wl_buffer + ahead)
    int held_max = 0;
    std::int64_t create_failed = 0;
    bool last_create_failed = false;
    bool create_done = false;

    // 오버레이
    std::unique_ptr<yvo::DmaHeapPool> ov_pool;
    int card_fd = -1; // --overlay-alloc dumb 일 때 /dev/dri/card1
    std::vector<std::unique_ptr<OverlayBuf>> ov_bufs;
    std::unique_ptr<OverlayPainter> painter;
    std::string ov_backend = "none";
    std::int64_t overlay_skips = 0;
    std::map<std::int64_t, std::vector<yvo::Detection>> dets;

    // 페이싱·기록
    std::vector<FrameRec> recs;
    std::int64_t t_start = 0;         // 첫 프레임 commit 시각
    double frame_period_ns = 1e9 / 30.0;
    int presented_video = 0;          // commit 한 비디오 프레임 수
    std::int64_t callbacks = 0;
    std::int64_t idle_commits = 0;
    bool frozen = false;
    std::int64_t freeze_until = 0;
    bool finished = false;

    // 측정 구간(정상 상태): 프레임 30 ~ 끝
    V3dStats v3d_a, v3d_b;
    std::int64_t meas_t0 = 0, meas_t1 = 0, cpu0 = 0, cpu1 = 0, lab0 = 0, lab1 = 0;
    int meas_f0 = 0, meas_f1 = 0;
    int labwc_pid = -1;
    std::vector<std::string> snapshots;
    std::vector<FeedbackCtx*> live_feedback; // 아직 presented/discarded 를 못 받은 피드백 (종료 시 정리)
};

void forget_feedback(FeedbackCtx* ctx) {
    auto& v = ctx->app->live_feedback;
    v.erase(std::remove(v.begin(), v.end(), ctx), v.end());
}

// ── 리스너 ─────────────────────────────────────────────────────────────────────────────────
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

// dmabuf 표면 피드백 (v4)
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
    vb->keepalive.reset(); // AVFrame 반납 → 디코더 풀
    vb->release_count++;
    vb->app->held--;
}
const wl_buffer_listener kVbufListener = {vbuf_release};

void obuf_release(void* data, wl_buffer*) { static_cast<OverlayBuf*>(data)->busy = false; }
const wl_buffer_listener kObufListener = {obuf_release};

// zwp_linux_buffer_params_v1 (비동기 create)
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

wl_buffer* g_created = nullptr; // params.created 로 받은 wl_buffer (create_dmabuf_buffer 가 가져간다)
void params_created_capture(void* data, zwp_linux_buffer_params_v1* p, wl_buffer* b) {
    g_created = b;
    params_created(data, p, b);
}
const zwp_linux_buffer_params_v1_listener kParamsListener = {params_created_capture, params_failed};

// 동기적으로(roundtrip) dmabuf wl_buffer 를 만든다. 실패하면 nullptr.
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

// ── 비디오 ─────────────────────────────────────────────────────────────────────────────────
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

// ── 오버레이 ───────────────────────────────────────────────────────────────────────────────
bool setup_overlay(App& app, int w, int h) {
    const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4;
    try {
        if (app.args.overlay_alloc == "dumb") {
            // card1(vc4) dumb buffer: 연속 메모리(CMA) + write-combined 매핑 → CPU 캐시 동기화가 필요 없다.
            // DRM master 없이도 CREATE_DUMB/MAP_DUMB/PRIME export 가 된다.
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
                    throw std::runtime_error("dumb pitch != w*4 (페인터는 패킹 버퍼를 가정한다)");
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
                std::fprintf(stderr, "[overlay] dmabuf wl_buffer 생성 실패 (%s)\n", app.args.overlay_alloc.c_str());
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

// 오버레이 버퍼 하나를 그려서 ov_surface 에 attach+commit (부모 commit 때 함께 적용 — sync subsurface)
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
        std::fprintf(stderr, "[video] frame %lld: wl_buffer 생성 실패 (compositor 거부)\n", static_cast<long long>(f.pts));
        app.held--;
        app.finished = true;
        return;
    }
    if (vb->busy) {
        // 같은 dma-buf 가 아직 컴포지터에 있다 — 일어나면 안 된다(우리가 AVFrame 을 들고 있으므로).
        std::fprintf(stderr, "[video] BUG: busy buffer reuse at frame %lld\n", static_cast<long long>(f.pts));
    }
    if (app.args.freeze_at >= 0 && f.pts == app.args.freeze_at) {
        // 색 비교용 CPU 기준 변환 (BT.709 limited → full RGB, display_convert.comp 와 같은 수식)
        const auto rgb = sand_to_rgb(f);
        if (!rgb.empty())
            write_ppm(app.args.out_dir + "/" + app.args.label + "_ref_f" + std::to_string(f.pts) + ".ppm", rgb.data(),
                      f.width, f.height);
    }
    vb->busy = true;
    vb->pts = f.pts;
    vb->keepalive = std::move(f.keepalive); // release 때까지 보유 (held 는 ahead 에서 이미 셈)
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
    // 측정 구간 시작/끝, 스냅샷
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
        app.t_start += (now - app.freeze_until) + app.args.freeze_ms * 1000000LL; // 시간표를 밀어 준다
    }
    // 소스 30 fps 시간표: 프레임 n 은 t_start + n*33.3ms 에 표시되어야 한다. 반 vblank 앞당겨 commit.
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
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // 로그를 줄 단위로 (tee/리다이렉트 중 중단돼도 남도록)
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
        std::fprintf(stderr, "wl_display_connect 실패 (WAYLAND_DISPLAY/XDG_RUNTIME_DIR 확인)\n");
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
        std::fprintf(stderr, "필수 전역(wl_compositor/xdg_wm_base/zwp_linux_dmabuf_v1) 없음\n");
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

    // dmabuf 표면 피드백
    zwp_linux_dmabuf_feedback_v1* feedback = nullptr;
    if (app.dmabuf_ver >= 4) {
        feedback = zwp_linux_dmabuf_v1_get_surface_feedback(app.dmabuf, app.surface);
        zwp_linux_dmabuf_feedback_v1_add_listener(feedback, &kFbListener, &app);
    }
    if (a.color_repr) {
        if (!app.color_repr_mgr) {
            std::printf("[color-repr] 요청했지만 labwc 가 wp_color_representation_manager_v1 을 광고하지 않는다 — 건너뜀\n");
        } else {
            app.color_repr = wp_color_representation_manager_v1_get_surface(app.color_repr_mgr, app.surface);
            wp_color_representation_surface_v1_set_coefficients_and_range(
                app.color_repr, WP_COLOR_REPRESENTATION_SURFACE_V1_COEFFICIENTS_BT709,
                a.color_range_full ? WP_COLOR_REPRESENTATION_SURFACE_V1_RANGE_FULL
                                   : WP_COLOR_REPRESENTATION_SURFACE_V1_RANGE_LIMITED);
            wp_color_representation_surface_v1_set_chroma_location(app.color_repr,
                                                                   WP_COLOR_REPRESENTATION_SURFACE_V1_CHROMA_LOCATION_TYPE_0);
            std::printf("[color-repr] BT.709 %s, chroma type_0 설정\n", a.color_range_full ? "full" : "limited");
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
            std::fprintf(stderr, "wp_viewporter 없음 — 창 모드 스케일 불가\n");
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
            std::fprintf(stderr, "wl_subcompositor 없음\n");
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
    // 첫 프레임
    if (!decode_ahead(app)) {
        std::fprintf(stderr, "첫 프레임 디코드 실패: %s\n", app.dec.error().c_str());
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
            std::fprintf(stderr, "wl_display_dispatch 실패: %s\n", std::strerror(errno));
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
    // 남은 피드백 수신
    for (int i = 0; i < 6; ++i) wl_display_roundtrip(app.display);

    // ── 요약 ──
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

    // 정리
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
