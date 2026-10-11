// yvo_pipeline: HEVC HW decode (DRM_PRIME) → Vulkan SAND128 letterbox → HailoRT yolov11m inference
//
// Thread structure
//   - Main thread: decode → acquire free slot → GPU submit → (fence-completed slot) wait → HailoRT submit.
//   - HailoRT callback thread: pushes the result into the completion queue and returns the slot to the free list (mutex + condvar).
//   - Result recording (JSON Lines, PNG, stats) is done by the main thread in pts order.
// Slot contract (slot numbers shared by GPU, NPU and display, 0..slots-1) — **there are two return conditions**
//   free → [GPU submit] → gpu queue → [SandLetterbox::wait: fence complete, AVFrame keepalive released]
//        → [HailoRunner::submit] → [HailoRT completion callback = compute_done]
//        → [display complete = display_done: submit fence for DisplayPipeline, scanout release for PlaneDisplay]  → free once both are set
//   With --display off there is no display condition at all (SlotExchange(need_display=false)) → exactly the same as before.
//   ⚠ The one who sets display_done is **the main thread itself**. So blocking while waiting for a free slot
//     deadlocks — the slot acquisition loop below runs only on acquire_poll + pump (it does not use acquire_wait).
//   Frames simultaneously in flight on the GPU are at most --gpu-depth (default 3). If the decoder has about 13+ frames held, it
//   **stalls silently without an error** → at startup we check slots+gpu_depth <= 10, and at runtime the actual held count
//   (held_frames_max) is measured and emitted to the CSV. This is the only means of detection.
// Modes
//   ptr    : HailoRT set_buffer(MemoryView(V3D output mapped pointer))
//   dmabuf : (default) at init, export per-slot output memory as dma-buf fds → pre-map with HailoRunner::map_dmabuf
//            → set_dma_buffer with PreprocOutput.dmabuf_fd
// Display (--display, conditionally compiled with YVO_HAVE_DISPLAY)
//   off  : the path up to today. Does not even create a DisplayPipeline (default).
//   on   : decoder SAND dma-buf → display_convert.comp → overlay_boxes.comp → swapchain present.
//          With --display-backend wayland-planes, the decoder NV12 SAND dma-buf goes onto the HVS video plane without V3D conversion,
//          and the GPU-drawn boxes (AB24 dma-buf) go onto the plane above it (PlaneDisplay + WaylandPlaneScanout).
//          With --display-backend kms, the same two planes are committed directly via KMS atomic without a compositor
//          (PlaneDisplay + KmsScanout, needs card1 DRM master — exit code 2 if unavailable).
//          matched = shown together with that frame's detections (show in flush_results),
//          latest  = shown as soon as preprocessing finishes, with the most recent detections overlaid (show in finish_gpu).
// Post-processing (--postproc)
//   host : single HEF-embedded NMS output → parsed by HailoRT on the host CPU (baseline path; implicit default under ptr/nhwc)
//   gpu  : (default) 6 raw output streams (dma-buf) → yolo_decode.comp pass 1 → suppression (--nms).
//          --nms gpu (default): efficient_nms.comp in the same command buffer. Only guard/over-capacity frames fall back to CPU.
//          --nms cpu: CPU greedy suppression (cpu_nms_greedy_by_class). --nms verify: runs both and compares bit-exactly.
//          Pushing NPU completions to the GPU is also done by the **main thread** (InferStage::pump/drain_one) —
//          because there is only one VkQueue, vkQueueSubmit must not be called from another thread.
// Shutdown order: InferStage::drain → flush_results → display drain → SandLetterbox::clear_import_cache
//            (display drain = DisplayPipeline::drain or PlaneDisplay::drain: remaining pending commits → scanout shutdown
//             (taken off screen; for kms, KMS state restore/planes-off) → force remaining releases → clear video buffer cache)
//            → decoder close → stage release (runner does dma-buf unmap) → display release (swapchain/surface / transport/box ring)
//            → export fd close → VulkanContext → WaylandWindow
//            (the window must die later than the context, so it is declared before the context)
// Signals: SIGINT/SIGTERM are received via sigaction(SA_RESTART) and **only set an atomic flag**.
//         The main loop, display pacing loop and slot acquisition loop check this flag every iteration, break out and
//         follow the **shutdown order above as is** → the stats summary comes out in the same format as a normal exit.
//         [interrupted] is prepended to the summary header, ok in RESULT_CSV/RESULT_CSV2 is 0,
//         and the exit code is 128+signo (SIGINT 130 / SIGTERM 143). A second signal calls _exit(130).
//         With --display on, Esc/q/Ctrl+C in the display window also stops it (for when a fullscreen window takes focus and terminal
//         SIGINT does not arrive) — the main loop sets the same flag, so it is the same exit as SIGINT (only the summary text differs).
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/coco_labels.hpp"
#include "ref/cpu_letterbox.hpp"
#include "hailo/hailo_runner.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "pipeline/gpu_postproc_stage.hpp"
#include "pipeline/infer_stage.hpp"
#include "pipeline/slot_exchange.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

#ifdef YVO_HAVE_DISPLAY
#include "display/display_pipeline.hpp"
#include "display/display_sink.hpp"
#include "display/plane_display.hpp"
#include "display/kms_scanout.hpp"
#include "display/wayland_plane_scanout.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"

using yvo::Stats;
using yvo::stats;
#endif

namespace {

using yvo::tool::cpu_ms;
using yvo::tool::soc_temperature;

using Clock = std::chrono::steady_clock;

// ── SIGINT/SIGTERM graceful shutdown ────────────────────────────────────────────────────────────
// The **only** thing the handler does is set an atomic flag (async-signal-safe: no printf or malloc).
// The actual shutdown is done by the main thread, which sees the flag and follows the existing shutdown path as is.
std::atomic<bool> g_stop{false};
std::atomic<int> g_stop_signal{0};
static_assert(std::atomic<bool>::is_always_lock_free,
              "the stop flag must be lock-free (it is used inside a signal handler)");
static_assert(std::atomic<int>::is_always_lock_free,
              "the signal number storage must also be lock-free (it is used inside a signal handler)");

void on_stop_signal(int sig) {
    // If exchange returns true it is the second signal → cut off immediately without waiting for the drain.
    if (g_stop.exchange(true, std::memory_order_relaxed)) _exit(130);
    g_stop_signal.store(sig, std::memory_order_relaxed);
}

bool stop_requested() { return g_stop.load(std::memory_order_relaxed); }

// Was the stop requested by a display window exit key (Esc/q/Ctrl+C)? Written and read only by the main thread (the signal handler does not touch it).
bool g_stop_by_key = false;

// Enable SA_RESTART: blocking system calls must not be broken by EINTR (the decoder's av_read_frame).
// Installation failure is not fatal — just warn and continue (the signal merely falls back to its default action).
bool install_stop_handlers() {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_stop_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    bool ok = true;
    for (const int sig : {SIGINT, SIGTERM}) {
        if (sigaction(sig, &sa, nullptr) != 0) {
            std::fprintf(stderr, "[warn] sigaction(%d) failed: %s — this signal falls back to its default action\n", sig,
                         std::strerror(errno));
            ok = false;
        }
    }
    return ok;
}

// COCO labels and class colors are gathered in one place, core/coco_labels.hpp (shared with the GPU overlay).
using yvo::class_name;

enum class Mode { Ptr, Dmabuf };

// Who does the post-processing.
//   Host: single HEF-embedded NMS output → parsed by HailoRT on the host CPU (today's baseline path)
//   Gpu : 6 raw output streams → GPU pass 1 (yolo_decode.comp) + CPU greedy suppression
enum class Postproc { Host, Gpu };

struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string hef = YVO_DEFAULT_HEF;
    long frames = 0;  // 0 = all
    Mode mode = Mode::Dmabuf;
    // Preprocessing output / NPU input tensor layout. With Nhcw, HailoRT's CPU input conversion disappears (dmabuf mode only).
    yvo::TensorLayout layout = yvo::TensorLayout::Nhcw;
    int batch = 8;
    int slots = 0;      // 0 = batch*2 (minimum 4)
    int gpu_depth = 3;  // number of frames kept in flight on the GPU at once
    float score = -1.0f;
    Postproc postproc = Postproc::Gpu;
    // Allocation method for the GPU post-processing raw output buffers. Default export (= Vulkan allocates and exports).
    // heap (dma-heap import) has not yet been verified for three-way coherence (PCIe DMA ↔ CPU cache ↔ V3D).
    std::string postproc_alloc = "export";
    // Where --postproc gpu does suppression. Default gpu (efficient_nms.comp).
    yvo::pipeline::NmsBackend nms = yvo::pipeline::NmsBackend::Gpu;
    bool nms_explicit = false, layout_explicit = false, postproc_explicit = false;   // given on the command line
    std::string dump_dir;
    int dump_every = 100;
    bool quiet = false;
    std::string csv_label;  // if set, print one CSV line (RESULT_CSV,...) at the end

    // ── Display (all defaults are "off": running without arguments takes the same path as before) ──
    bool display = false;
#ifdef YVO_HAVE_DISPLAY
    yvo::display::DisplayPipelineConfig disp;
    bool pace_explicit = false;   // did the user pass --display-pace explicitly
#endif
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--hef PATH] [--frames N] [--mode ptr|dmabuf] [--layout nhwc|nhcw]\n"
                 "          [--batch N] [--slots N] [--postproc host|gpu] [--postproc-alloc export|heap]\n"
                 "          [--nms gpu|cpu|verify] [--gpu-depth N] [--score T] [--dump-dir DIR] [--dump-every K]\n"
                 "          [--quiet] [--csv-label L]\n"
                 "  --layout     preprocessing output / NPU input tensor layout (default nhcw with the default --mode dmabuf; nhwc under --mode ptr)\n"
                 "               nhcw removes HailoRT's CPU input conversion (about 14 ms/frame). --mode dmabuf only\n"
                 "  --slots      number of slots shared by GPU and NPU (default max(4, batch*2))\n"
                 "  --gpu-depth  max frames in flight on the GPU at once (default 3, 2~4 recommended: decoder buffer limit)\n"
                 "  --postproc   post-processing location (default gpu; host under --mode ptr or --layout nhwc). gpu = 6 raw output streams + yolo_decode.comp + efficient_nms.comp suppression\n"
                 "               (only edge-case frames fall back to CPU, see --nms)\n"
                 "               --mode dmabuf --layout nhcw only\n"
                 "  --postproc-alloc  raw output buffer allocation for gpu post-processing (default export)\n"
                 "  --nms        suppression location for gpu post-processing (default gpu = efficient_nms.comp, only guard frames fall back to CPU)\n"
                 "               cpu = cpu_nms_greedy_by_class, verify = run both and compare bit-exactly (output is the CPU side)\n"
                 "               --postproc gpu only\n"
                 "  --dump-dir   detections.jsonl (one line per frame) + box PNG every K frames (on the letterbox input)\n"
#ifdef YVO_HAVE_DISPLAY
                 "  --display {off,on}                       real-time HDMI display (default off)\n"
                 "  --display-backend {auto,wayland,offscreen,wayland-planes,kms}  default auto. wayland is strict (no silent fallback)\n"
                 "               wayland-planes = HVS two-plane scanout (video NV12 SAND plane + box AB24 plane, no V3D conversion).\n"
                 "               forces fullscreen and does not use Vulkan present. strict (exits on failure)\n"
                 "               kms = commits the same two planes directly via KMS atomic without a compositor. needs card1 DRM master\n"
                 "               (if the desktop is on the active VT, master cannot be acquired: exit code 2). no window, so stop with Ctrl+C/SIGTERM\n"
                 "  --display-connector NAME                 connector for kms (default: first connected HDMI-A. e.g. HDMI-A-2)\n"
                 "  --display-color-range {auto,full,limited}   COLOR_RANGE for plane scanout (default auto)\n"
                 "               wayland-planes auto = request full: labwc 0.20.1~0.20.2 / wlroots 0.20.2+rpt3~+rpt4 invert it and program limited\n"
                 "               kms auto = programs limited as is (the actual plane value is judged via debugfs in both cases)\n"
                 "  --display-sync {matched,latest}          default matched (shown together with that frame's detections)\n"
                 "  --display-pace {source,asap}             default source when display is on. asap = no pacing\n"
                 "  --display-fullscreen | --display-size WxH | --display-every K\n"
                 "  --display-warmup N                       leave the first N presentations out of glass / decode-to-screen /\n"
                 "               present-interval / late-frame stats, every backend (default 0 = keep all; kms 1st commit = modeset)\n"
                 "  --present-mode {fifo,mailbox,immediate}  default fifo\n"
                 "  --display-blit                           force the blit fallback even if STORAGE is available\n"
                 "  --overlay-line-width N | --overlay-text-scale N   default 2 / 2\n"
                 "  Esc/q/Ctrl+C in the display window also stops it (same [interrupted] exit as terminal Ctrl+C, code 130)\n"
                 "  ⚠ Always measure with --display-fullscreen. Windowed mode falls onto the uniform-scale path and\n"
                 "    display conversion becomes 2.6x slower (4.25 → 10.9 ms) with no error or warning\n"
#endif
                 ,
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--hef" && (v = next())) a.hef = v;
        else if (s == "--frames" && (v = next())) a.frames = std::atol(v);
        else if (s == "--mode" && (v = next())) {
            const std::string m = v;
            if (m == "ptr") a.mode = Mode::Ptr;
            else if (m == "dmabuf") a.mode = Mode::Dmabuf;
            else return false;
        } else if (s == "--layout" && (v = next())) {
            const std::string l = v;  a.layout_explicit = true;
            if (l == "nhwc") a.layout = yvo::TensorLayout::Nhwc;
            else if (l == "nhcw") a.layout = yvo::TensorLayout::Nhcw;
            else return false;
        } else if (s == "--batch" && (v = next())) a.batch = std::atoi(v);
        else if (s == "--slots" && (v = next())) a.slots = std::atoi(v);
        else if (s == "--gpu-depth" && (v = next())) a.gpu_depth = std::atoi(v);
        else if (s == "--postproc" && (v = next())) {
            const std::string p = v;  a.postproc_explicit = true;
            if (p == "host") a.postproc = Postproc::Host;
            else if (p == "gpu") a.postproc = Postproc::Gpu;
            else return false;
        } else if (s == "--postproc-alloc" && (v = next())) {
            const std::string p = v;
            if (p != "export" && p != "heap") return false;
            a.postproc_alloc = p;
        } else if (s == "--nms" && (v = next())) {
            const std::string p = v;
            if (p == "gpu") a.nms = yvo::pipeline::NmsBackend::Gpu;
            else if (p == "cpu") a.nms = yvo::pipeline::NmsBackend::Cpu;
            else if (p == "verify") a.nms = yvo::pipeline::NmsBackend::Verify;
            else return false;
            a.nms_explicit = true;
        }
        else if (s == "--score" && (v = next())) a.score = static_cast<float>(std::atof(v));
        else if (s == "--dump-dir" && (v = next())) a.dump_dir = v;
        else if (s == "--dump-every" && (v = next())) a.dump_every = std::atoi(v);
        else if (s == "--csv-label" && (v = next())) a.csv_label = v;
        else if (s == "--quiet") a.quiet = true;
#ifdef YVO_HAVE_DISPLAY
        else if (s == "--display" && (v = next())) {
            const std::string d = v;
            if (d == "on") a.display = true;
            else if (d == "off") a.display = false;
            else return false;
        } else if (s == "--display-backend" && (v = next())) {
            if (!yvo::display::parse_backend_kind(v, &a.disp.backend)) return false;
        } else if (s == "--display-connector" && (v = next())) {
            a.disp.connector = v;
        } else if (s == "--display-color-range" && (v = next())) {
            if (!yvo::display::parse_color_range(v, &a.disp.color_range)) return false;
        } else if (s == "--display-sync" && (v = next())) {
            if (!yvo::display::parse_display_sync(v, &a.disp.sync)) return false;
        } else if (s == "--display-pace" && (v = next())) {
            if (!yvo::display::parse_display_pace(v, &a.disp.pace)) return false;
            a.pace_explicit = true;
        } else if (s == "--display-fullscreen") a.disp.fullscreen = true;
        else if (s == "--display-size" && (v = next())) {
            int w = 0, h = 0;
            if (std::sscanf(v, "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) return false;
            a.disp.width = w;
            a.disp.height = h;
        } else if (s == "--display-every" && (v = next())) a.disp.every = std::atoi(v);
        else if (s == "--display-warmup" && (v = next())) a.disp.warmup = std::atoi(v);
        else if (s == "--present-mode" && (v = next())) {
            if (!yvo::display::parse_present_mode(v, &a.disp.present_mode)) return false;
        } else if (s == "--display-blit") a.disp.force_blit = true;
        else if (s == "--overlay-line-width" && (v = next())) a.disp.overlay_line_width = std::atoi(v);
        else if (s == "--overlay-text-scale" && (v = next())) a.disp.overlay_text_scale = std::atoi(v);
#endif
        else if (s == "-h" || s == "--help") return false;
        else return false;
    }
    const bool lay_dn = !a.layout_explicit && a.mode == Mode::Ptr;  // implicit dependents follow an explicit --mode ptr / --layout nhwc
    const bool pp_dn = !a.postproc_explicit && (a.mode == Mode::Ptr || a.layout == yvo::TensorLayout::Nhwc);
    if (lay_dn || pp_dn) std::fprintf(stderr, "note: default lowered to%s%s (nhcw needs --mode dmabuf, --postproc gpu needs dmabuf + nhcw)\n", lay_dn ? " --layout nhwc" : "", pp_dn ? " --postproc host" : "");
    a.layout = lay_dn ? yvo::TensorLayout::Nhwc : a.layout;  a.postproc = pp_dn ? Postproc::Host : a.postproc;
    if (a.slots <= 0) a.slots = std::max(4, a.batch * 2);
    if (a.batch < 1 || a.batch > 64 || a.gpu_depth < 1 || a.dump_every < 1) return false;
    a.gpu_depth = std::min(a.gpu_depth, a.slots);
#ifdef YVO_HAVE_DISPLAY
    if (a.disp.every < 1 || a.disp.warmup < 0 || a.disp.overlay_line_width < 1 || a.disp.overlay_text_scale < 1) return false;
    // Pacing default: source when display is on (follows the 30 fps source), none when off.
    if (!a.pace_explicit)
        a.disp.pace = a.display ? yvo::display::DisplayPace::Source : yvo::display::DisplayPace::None;
    a.disp.num_slots = a.slots;
    // Plane scanout (PlaneDisplay): wayland-planes and kms. Same frame-holding rules (same PlaneDisplay limit).
    const bool planes = a.disp.backend == yvo::display::DisplayBackendKind::WaylandPlanes ||
                        a.disp.backend == yvo::display::DisplayBackendKind::Kms;
    if (!a.disp.connector.empty() && a.disp.backend != yvo::display::DisplayBackendKind::Kms)
        std::fprintf(stderr, "warning: --display-connector is used only with --display-backend kms (ignored for now)\n");
    if (planes) a.disp.fullscreen = true;   // in windowed mode labwc composites with GL → this backend becomes pointless (kms is always fullscreen)
    // ⚠ Decoder 13-frame constraint: holding more than about 13 frames at once makes the decoder **stall without an error.**
    //   Vulkan composition path: matched holds slots + gpu_depth, latest holds gpu_depth + image_count.
    //   We block at 10 to leave a margin.
    //   wayland-planes / kms (PlaneDisplay): frames are held **until the compositor does wl_buffer.release** (for kms, until the next flip).
    //     matched — display_done is set on release, so every frame held by the display occupies its own slot
    //               (the slot does not come back during retained → pending → committed). gpu_depth is also within the slots.
    //               → at most slots + 1 (one just-decoded frame waiting for a slot).
    //     latest  — the slot is returned right after show, and the display holds pending 2 + committed (unreleased) 3 more outside the slots
    //               (PlaneDisplay::kExtraHeldBeyondSlots = on screen 1 + waiting for next flip 1 + release delay 1 + pending 2).
    //               → at most slots + 1 + 5.
    //     Blocked with the same margin (10). The actual maximum is measured as held_frames_max.
    if (a.display) {
        const std::string err = yvo::pipeline::validate_slot_budget(
            !planes ? yvo::pipeline::SlotBudgetPath::Composite
            : a.disp.backend == yvo::display::DisplayBackendKind::Kms ? yvo::pipeline::SlotBudgetPath::Kms
                                                                       : yvo::pipeline::SlotBudgetPath::WaylandPlanes,
            a.disp.sync == yvo::display::DisplaySync::Latest, a.slots, a.gpu_depth,
            yvo::display::PlaneDisplay::kExtraHeldBeyondSlots);
        if (!err.empty()) {
            std::fputs(err.c_str(), stderr);
            return false;
        }
    }
#endif
    return true;
}

// Thread-safe slot pool + completion queue (HailoRT callback ↔ main thread): pipeline/slot_exchange.hpp
using yvo::pipeline::Completion;
using yvo::pipeline::SlotExchange;

void draw_and_save(const std::string& path, const std::vector<std::uint8_t>& rgb, const yvo::LetterboxParams& lb,
                   const std::vector<yvo::Detection>& dets) {
    cv::Mat m(lb.dst_h, lb.dst_w, CV_8UC3, const_cast<std::uint8_t*>(rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);
    for (const auto& d : dets) {
        // source coordinates → letterbox coordinates
        const int x1 = static_cast<int>(std::lround(d.x1 * lb.scale + static_cast<float>(lb.pad_x)));
        const int y1 = static_cast<int>(std::lround(d.y1 * lb.scale + static_cast<float>(lb.pad_y)));
        const int x2 = static_cast<int>(std::lround(d.x2 * lb.scale + static_cast<float>(lb.pad_x)));
        const int y2 = static_cast<int>(std::lround(d.y2 * lb.scale + static_cast<float>(lb.pad_y)));
        // class_color() returns RGB, so flip it for OpenCV (BGR). Colors are defined only in coco_labels.hpp.
        const auto rgb_col = yvo::class_color(d.class_id);
        const cv::Scalar color(rgb_col[2], rgb_col[1], rgb_col[0]);
        cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f", class_name(d.class_id), static_cast<double>(d.score));
        cv::putText(bgr, label, cv::Point(x1, std::max(10, y1 - 3)), cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    }
    if (!cv::imwrite(path, bgr)) std::fprintf(stderr, "[warn] failed to save PNG: %s\n", path.c_str());
}

struct FdCloser {
    std::vector<int> fds;
    ~FdCloser() {
        for (int fd : fds)
            if (fd >= 0) ::close(fd);
    }
};

const char* layout_name(yvo::TensorLayout l) { return l == yvo::TensorLayout::Nhcw ? "nhcw" : "nhwc"; }
const char* nms_name(yvo::pipeline::NmsBackend b) {
    switch (b) {
        case yvo::pipeline::NmsBackend::Gpu: return "gpu";
        case yvo::pipeline::NmsBackend::Cpu: return "cpu";
        case yvo::pipeline::NmsBackend::Verify: return "verify";
    }
    return "?";
}

int run(const Args& a) {
    const bool dmabuf = a.mode == Mode::Dmabuf;
    // With NHCW, HailoRT DMA-maps the input buffer directly, but DMA mapping fails for V3D GEM mmap pointers.
    // Reject it at startup rather than letting the runner throw on the first submit.
    if (!dmabuf && a.layout == yvo::TensorLayout::Nhcw) {
        std::fprintf(stderr,
                     "--mode ptr --layout nhcw cannot be used: with NHCW, HailoRT DMA-maps the input buffer directly, but\n"
                     "DMA mapping fails for V3D GEM mmap pointers. Use --mode dmabuf --layout nhcw.\n");
        return 2;
    }
    // GPU post-processing needs a dma-buf (NHCW) input too: HailoRawRunner has no host-pointer path,
    // and the 6 raw output streams were verified assuming NHCW input.
    if (a.postproc == Postproc::Gpu && (!dmabuf || a.layout != yvo::TensorLayout::Nhcw)) {
        std::fprintf(stderr,
                     "--postproc gpu is only for --mode dmabuf --layout nhcw: HailoRawRunner accepts input and output\n"
                     "only as dma-buf (no host-pointer path), and the 6-raw-output-stream path was verified only with NHCW input.\n");
        return 2;
    }
    if (a.nms_explicit && a.postproc != Postproc::Gpu) {
        std::fprintf(stderr, "--nms is only for --postproc gpu (on the host path HailoRT does NMS per the HEF settings).\n");
        return 2;
    }
    const double soc_t0 = soc_temperature();

    // Export fds must be closed later than the runner (unmap), so they are declared first (reverse destruction).
    FdCloser export_fds;

    // ── Init order: WaylandWindow → VulkanContext → SandLetterbox → InferStage → DisplayPipeline → decoder ──
    // ⚠ The window comes before the context: vkGetPhysicalDeviceWaylandPresentationSupportKHR requires a
    //   wl_display* (see the chicken-and-egg note in vulkan_context.hpp). And the window must be destroyed **later**
    //   than the context, so it is declared here (reverse destruction).
    yvo::vk::VulkanContextOptions copt;
#ifdef YVO_HAVE_DISPLAY
    yvo::display::WaylandWindow win;
    bool have_win = false;
    const bool kms = a.display && a.disp.backend == yvo::display::DisplayBackendKind::Kms;
    const bool planes = a.display && (a.disp.backend == yvo::display::DisplayBackendKind::WaylandPlanes || kms);
    // kms: no window and no Vulkan present (copt default = present off). If master cannot be acquired, exit with
    // code 2 before opening the NPU/GPU — so device open and master acquisition are done **first**. The screen changes on the first commit.
    std::unique_ptr<yvo::display::KmsScanout> kms_scanout;
    if (kms) {
        yvo::display::KmsScanoutOptions ko;
        ko.connector = a.disp.connector;
        ko.color_range = a.disp.color_range;
        ko.verbose = a.disp.verbose;
        try {
            kms_scanout = std::make_unique<yvo::display::KmsScanout>(ko);
        } catch (const yvo::display::KmsMasterError& e) {
            std::fprintf(stderr, "--display-backend kms: %s\n", e.what());
            return 2;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "--display-backend kms: %s\n", e.what());
            return 1;
        }
    } else if (planes) {
        // wayland-planes: fullscreen window + linux-dmabuf/presentation/viewporter/color-representation globals.
        // Vulkan present is not used (copt default = present off) — video is the decoder dma-buf, boxes are the OverlayPlane dma-buf.
        yvo::display::WaylandWindowOptions wo;
        wo.width = a.disp.width;
        wo.height = a.disp.height;
        wo.fullscreen = true;
        wo.plane_globals = true;
        wo.title = a.disp.title;
        wo.app_id = a.disp.app_id;
        wo.verbose = a.disp.verbose;
        have_win = win.open(wo);
        if (!have_win) {
            std::fprintf(stderr, "--display-backend wayland-planes: could not open window — %s\n", win.error().c_str());
            return 1;
        }
    } else if (a.display) {
        have_win = yvo::display::DisplayPipeline::open_window(win, a.disp);
        if (!have_win && a.disp.backend == yvo::display::DisplayBackendKind::Wayland) {
            std::fprintf(stderr, "--display-backend wayland: could not open window — %s\n", win.error().c_str());
            return 1;
        }
        copt.enable_present = have_win;
        copt.wayland_display = have_win ? win.native_display() : nullptr;
        // strict: no silent fallback. With auto, continue offscreen even if present cannot be enabled.
        copt.require_present = have_win && a.disp.backend == yvo::display::DisplayBackendKind::Wayland;
    }
#endif
    yvo::vk::VulkanContext ctx(copt);
    yvo::vk::SandLetterboxConfig pcfg;
    pcfg.num_slots = a.slots;
    pcfg.export_dmabuf = dmabuf;
    pcfg.out_layout = a.layout;
    yvo::vk::SandLetterbox pre(ctx, pcfg);

    std::unique_ptr<yvo::pipeline::InferStage> stage;
    yvo::pipeline::GpuPostprocStage* gpu_stage = nullptr;
    if (a.postproc == Postproc::Gpu) {
        yvo::pipeline::GpuPostprocStageConfig gcfg;
        gcfg.hef_path = a.hef;
        gcfg.batch_size = static_cast<std::uint16_t>(a.batch);
        gcfg.num_slots = a.slots;
        gcfg.score_thr = a.score >= 0.0f ? a.score : 0.2f;
        gcfg.nms = a.nms;
        gcfg.input_alloc = a.postproc_alloc == "heap" ? yvo::vk::YoloInputAlloc::ImportHeap
                                                      : yvo::vk::YoloInputAlloc::ExportDmaBuf;
        auto g = std::make_unique<yvo::pipeline::GpuPostprocStage>(ctx, gcfg);
        gpu_stage = g.get();
        stage = std::move(g);
    } else {
        yvo::HailoRunnerConfig hcfg;
        hcfg.hef_path = a.hef;
        hcfg.input_layout = a.layout;  // allow_host_ptr_nhcw stays false: the runner blocks V3D pointer submission
        hcfg.batch_size = static_cast<std::uint16_t>(a.batch);
        hcfg.num_slots = a.slots;
        hcfg.score_threshold = a.score;
        stage = std::make_unique<yvo::pipeline::HostNmsStage>(hcfg);
    }
    if (stage->input_frame_size() != pre.output_size()) {
        std::fprintf(stderr, "input size mismatch: NPU %zu, GPU %zu\n", stage->input_frame_size(), pre.output_size());
        return 1;
    }

    std::vector<int> slot_fd(static_cast<std::size_t>(a.slots), -1);
    if (dmabuf) {
        for (int s = 0; s < a.slots; ++s) {
            const int fd = pre.export_dmabuf_fd(s);
            export_fds.fds.push_back(fd);
            stage->bind_input_dmabuf(s, fd);
            slot_fd[static_cast<std::size_t>(s)] = fd;
        }
    }

    // Display path. Created **before** the decoder — extra_hw_frames depends on image_count().
    // The source size and fps are passed with set_source_size()/set_source_fps() after the decoder is opened.
#ifdef YVO_HAVE_DISPLAY
    // The pipeline holds only the DisplaySink interface (implementation: DisplayPipeline = V3D composition + swapchain).
    std::unique_ptr<yvo::display::DisplaySink> display;
    if (kms) {
        display = std::make_unique<yvo::display::PlaneDisplay>(ctx, std::move(kms_scanout), a.disp);
    } else if (planes) {
        yvo::display::WaylandPlaneScanoutOptions so;
        so.color_range = a.disp.color_range;
        so.verbose = a.disp.verbose;
        display = std::make_unique<yvo::display::PlaneDisplay>(
            ctx, std::make_unique<yvo::display::WaylandPlaneScanout>(win, so), a.disp);
    } else if (a.display) {
        display = std::make_unique<yvo::display::DisplayPipeline>(ctx, have_win ? &win : nullptr, a.disp);
    }
#endif

    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt; dopt.threads = 1; if (const char* e = std::getenv("YVO_DEC_THREADS")) dopt.threads = std::atoi(e);  // 1 frame thread (07_results/dec_threads); YVO_DEC_THREADS=0 -> FFmpeg auto
    dopt.extra_hw_frames = std::max(2 * yvo::kNumSlots, a.gpu_depth + 4);
#ifdef YVO_HAVE_DISPLAY
    if (display && planes) {
        // wayland-planes / kms: margin of 4 on top of parse()'s holding count (matched slots+1, latest slots+1+5).
        const int hold = a.slots + 1 +
                         (a.disp.sync == yvo::display::DisplaySync::Latest
                              ? yvo::display::PlaneDisplay::kExtraHeldBeyondSlots
                              : 0);
        dopt.extra_hw_frames = std::max(8, hold + 4);
    } else if (display) {
        // The display holds frames longer. matched holds one per slot (= slots), latest holds one per swapchain
        // image (= image_count) until the GPU has read it all. Plus gpu_depth and a margin of 4.
        const int need = a.disp.sync == yvo::display::DisplaySync::Matched
                             ? a.slots + a.gpu_depth + 4
                             : a.gpu_depth + static_cast<int>(display->image_count()) + 4;
        dopt.extra_hw_frames = std::max(8, need);
    }
#endif
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }
    const int src_w = dec.width(), src_h = dec.height();
    stage->set_source_size(src_w, src_h);
#ifdef YVO_HAVE_DISPLAY
    const double dec_fps = dec.frame_rate(), src_fps = dec_fps >= 1.0 && dec_fps <= 240.0 ? dec_fps : a.disp.source_fps;  // NaN fails the range test
    if (display && src_fps != dec_fps) std::fprintf(stderr, "warning: stream frame rate %.3f not in [1, 240], display pace keeps %.2f fps\n", dec_fps, src_fps);
    if (display) { display->set_source_size(src_w, src_h); display->set_source_fps(src_fps); }
#endif

    std::unique_ptr<std::FILE, decltype([](std::FILE* fp) { std::fclose(fp); })> jsonl;  // closed on every exit path
    if (!a.dump_dir.empty()) {
        std::filesystem::create_directories(a.dump_dir);
        const std::string p = a.dump_dir + "/detections.jsonl";
        jsonl.reset(std::fopen(p.c_str(), "w"));
        if (!jsonl) {
            std::fprintf(stderr, "cannot open %s\n", p.c_str());
            return 1;
        }
    }

    const float hailo_t0 = stage->chip_temperature();
    if (!a.quiet) {
        const std::string pp_desc = gpu_stage ? std::string("gpu/") + gpu_stage->alloc_name() : std::string("host");
        std::printf("[cfg] video=%s %dx%d mode=%s layout=%s batch=%d slots=%d gpu_depth=%d postproc=%s device=%s\n",
                    a.video.c_str(), src_w, src_h, dmabuf ? "dmabuf" : "ptr", layout_name(a.layout), a.batch, a.slots,
                    a.gpu_depth, pp_desc.c_str(), ctx.device_name());
#ifdef YVO_HAVE_DISPLAY
        if (display && !display->describe().empty())
            std::printf("[display] backend=%s %s | sync=%s pace=%s@%.2f every=%d hw_frames=%d color_range=%s\n",
                        display->backend_name(), display->describe().c_str(), yvo::display::sync_name(a.disp.sync),
                        yvo::display::pace_name(a.disp.pace), src_fps, a.disp.every, dopt.extra_hw_frames,
                        yvo::display::color_range_name(a.disp.color_range));
        else if (display)
            std::printf("[display] backend=%s %ux%u %s / %s images=%u sync=%s pace=%s@%.2f every=%d hw_frames=%d\n",
                        display->backend_name(), display->extent().width,
                        display->extent().height, display->uniform_scale() ? "uniform-scale(bilinear)" : "1:1",
                        display->blit_mode() ? "blit fallback" : "compute -> storage image", display->image_count(),
                        yvo::display::sync_name(a.disp.sync), yvo::display::pace_name(a.disp.pace), src_fps, a.disp.every,
                        dopt.extra_hw_frames);
#endif
        std::fflush(stdout);
    }

    bool need_display_ack = false;
#ifdef YVO_HAVE_DISPLAY
    need_display_ack = display != nullptr;
#endif
    SlotExchange ex(a.slots, need_display_ack);
#ifdef YVO_HAVE_DISPLAY
    // Display completion (= DisplayConvert's submit fence) is the second slot return condition. Called on the main thread.
    if (display) display->set_display_done_callback([&ex](int s) { ex.display_done(s); });
#endif
    std::deque<int> gpu_q;  // slots in flight on the GPU (submit order)
    std::vector<double> decode_ms, gpu_ms, npu_ms, e2e_ms, wait_block_ms;
    std::unordered_map<std::int64_t, Clock::time_point> t_decoded;  // pts → decode return time
    std::map<std::int64_t, Completion> reorder;                      // for recording in pts order
    std::unordered_map<std::int64_t, std::vector<std::uint8_t>> png_src;  // letterbox copies of frames to dump
    std::unordered_map<std::int64_t, yvo::LetterboxParams> png_lb;
    std::int64_t next_pts = 0;
    long submitted_npu = 0, completed = 0, total_dets = 0, errors = 0;

    // Decoder 13-frame constraint measurement: count the AVFrames we actually hold using weak_ptr.
    // (SandLetterbox's keepalive and DisplayPipeline's retained share the same shared_ptr, so
    //  nothing is counted twice. This is the only means of detecting the "silent stall" failure mode.)
    std::deque<std::weak_ptr<void>> live_frames;
    long held_frames_max = 0, slot_timeout_count = 0;
    std::vector<yvo::Detection> latest_dets;   // most recent detections overlaid by --display-sync latest
    double fps_now = 0.0;

    auto on_result = [&ex](yvo::InferResult&& r) { ex.complete(std::move(r)); };
    // Every exit path, exceptions included: stage (HailoRT callbacks → &ex) and display (done callback → &ex, decoder
    // frames / import pins) must be destroyed before ex and dec. Their destructors drain and never throw.
    auto release_stages = [&]() noexcept {
        gpu_stage = nullptr;
        stage.reset();
#ifdef YVO_HAVE_DISPLAY
        display.reset();
#endif
    };
    struct ReleaseOnExit { decltype(release_stages)& f; ~ReleaseOnExit() { f(); } } release_on_exit{release_stages};

    // Record completed results in pts order
    auto flush_results = [&]() {
        for (auto& c : ex.take_done()) {
            const std::int64_t pts = c.result.pts;
            reorder.emplace(pts, std::move(c));
        }
        while (!reorder.empty() && reorder.begin()->first == next_pts) {
            Completion& c = reorder.begin()->second;
            const yvo::InferResult& r = c.result;
            ++completed;
            if (r.status != HAILO_SUCCESS) {
                ++errors;
                std::fprintf(stderr, "[error] frame %lld: HailoRT status %d\n", static_cast<long long>(r.pts),
                             static_cast<int>(r.status));
            }
            npu_ms.push_back(r.latency_ms);
            if (auto it = t_decoded.find(r.pts); it != t_decoded.end()) {
                e2e_ms.push_back(std::chrono::duration<double, std::milli>(c.t_done - it->second).count());
                t_decoded.erase(it);
            }
            total_dets += static_cast<long>(r.detections.size());
            if (jsonl) {
                std::fprintf(jsonl.get(), "{\"frame\":%lld,\"detections\":[", static_cast<long long>(r.pts));
                for (std::size_t i = 0; i < r.detections.size(); ++i) {
                    const auto& d = r.detections[i];
                    std::fprintf(jsonl.get(), "%s{\"class\":%d,\"name\":\"%s\",\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}",
                                 i ? "," : "", d.class_id, class_name(d.class_id), static_cast<double>(d.score),
                                 static_cast<double>(d.x1), static_cast<double>(d.y1), static_cast<double>(d.x2),
                                 static_cast<double>(d.y2));
                }
                std::fprintf(jsonl.get(), "]}\n");
            }
            if (auto it = png_src.find(r.pts); it != png_src.end()) {
                char name[64];
                std::snprintf(name, sizeof(name), "/frame%05lld.png", static_cast<long long>(r.pts));
                draw_and_save(a.dump_dir + name, it->second, png_lb[r.pts], r.detections);
                png_src.erase(it);
                png_lb.erase(r.pts);
            }
#ifdef YVO_HAVE_DISPLAY
            // Display happens after all recording is done (end of drain). In matched this is the only show point.
            if (display && a.disp.sync == yvo::display::DisplaySync::Latest) latest_dets = r.detections;
            if (display && a.disp.sync == yvo::display::DisplaySync::Matched) {
                char hud[64];
                std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d NPU %3.0fms", fps_now,
                              static_cast<int>(r.detections.size()), r.latency_ms);
                // If show() returns false the frame does not reach the screen → the slot must be returned **immediately**.
                if (!display->show(r.slot, r.detections, hud)) ex.display_done(r.slot);
            }
#endif
            reorder.erase(reorder.begin());
            ++next_pts;
        }
    };

    // Finish the slot at the front of the GPU queue and hand it to the NPU
    auto finish_gpu = [&]() {
        const int slot = gpu_q.front();
        gpu_q.pop_front();
        yvo::PreprocOutput out = pre.wait(slot);
        if (pre.last_gpu_ms() >= 0) gpu_ms.push_back(pre.last_gpu_ms());
        if (jsonl && out.pts % a.dump_every == 0) {
            // draw_and_save() expects packed RGB, so NHCW output is converted back before being stored.
            std::vector<std::uint8_t>& buf = png_src[out.pts];
            buf.resize(out.size);
            if (out.layout == yvo::TensorLayout::Nhcw)
                yvo::nhcw_to_rgb_packed(out.host_ptr, out.lb.dst_w, out.lb.dst_h, buf.data());
            else
                std::memcpy(buf.data(), out.host_ptr, out.size);
            png_lb[out.pts] = out.lb;
        }
        if (dmabuf) out.dmabuf_fd = slot_fd[static_cast<std::size_t>(slot)];
        stage->submit(out, on_result);
        ++submitted_npu;
#ifdef YVO_HAVE_DISPLAY
        // latest: without waiting for detections, put it on screen **as soon as preprocessing finishes** (overlaying the most recent detections).
        // Set display_done right here so the display does not extend the slot lifetime.
        if (display && a.disp.sync == yvo::display::DisplaySync::Latest) {
            char hud[64];
            std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d", fps_now, static_cast<int>(latest_dets.size()));
            display->show(slot, latest_dets, hud);
            ex.display_done(slot);
        }
#endif
        // submit() can block on NPU backpressure. Immediately collect completions that piled up meanwhile
        // and return their slots (on the gpu path slot hold time directly eats into NPU occupancy). No-op for host.
        stage->pump();
    };

    const double c0 = cpu_ms(); yvo::tool::perf_window(true);  // perf -D -1 --control window opens here
    const auto t0 = Clock::now();
    long n = 0;
    bool fail = false;
    try {
        while (!stop_requested() && (a.frames <= 0 || n < a.frames)) {
#ifdef YVO_HAVE_DISPLAY
            if (display) {
                if (!display->poll_window()) {
                    // An exit key takes the same stop path as SIGINT ([interrupted], ok=0, exit code 130).
                    // Closing the window (xdg_toplevel.close) remains a normal exit, not a stop, as before.
                    if (display->quit_requested() && !stop_requested()) {
                        g_stop_by_key = true;
                        g_stop_signal.store(SIGINT, std::memory_order_relaxed);
                        g_stop.store(true, std::memory_order_relaxed);
                    }
                    break;
                }
                // --display-pace source: follow the source frame rate. **Does not use a blocking sleep** —
                // NPU completions, GPU post-processing and display completions must keep being collected meanwhile, or latency gets distorted.
                if (auto dl = display->pace_deadline(n)) {
                    while (Clock::now() < *dl && !stop_requested()) {
                        stage->pump();
                        while (!gpu_q.empty() && pre.ready(gpu_q.front())) finish_gpu();
                        display->pump();
                        flush_results();
                        std::this_thread::sleep_for(std::chrono::microseconds(500));
                    }
                }
            }
#endif
            const auto d0 = Clock::now();
            auto f = dec.next();
            const auto d1 = Clock::now();
            stage->pump();   // completions that arrived while waiting for decode (no-op for host)
            if (!f) break;
            decode_ms.push_back(std::chrono::duration<double, std::milli>(d1 - d0).count());
            f->pts = n;  // unify to frame sequence number
            t_decoded[n] = d1;

            // Acquire a free slot. If none, first hand the GPU queue to the NPU; if still none, wait for callbacks.
            std::optional<int> slot = ex.try_acquire();
            const auto w0 = Clock::now();
            // ⚠ **No blocking wait.** The one who sets the second slot return condition (display_done) is
            //   the main thread itself, so holding the condvar here for long deadlocks outright.
            //   Instead, each iteration advances all three:
            //     stage->pump()  NPU completion → GPU post-processing → compute_done (no-op for host)
            //     finish_gpu()   pushes preprocessing results to the NPU so slots move to the next stage
            //     flush_results() **this is where display is issued** (matched). Omit it and only compute_done gets set,
            //                    display_done never arrives and it times out after 5 s (this actually happened)
            //     display->pump() reclaims display fences → display_done
            //   With --display off && --postproc host, both pumps are no-ops and flush_results was originally called every
            //   frame anyway, so it is equivalent to before.
            if (!slot) {
                const auto deadline = Clock::now() + std::chrono::seconds(5);
                while (!slot) {
                    stage->pump();
                    if (!gpu_q.empty()) finish_gpu();
                    flush_results();
#ifdef YVO_HAVE_DISPLAY
                    if (display) display->pump();
#endif
                    slot = ex.acquire_poll(std::chrono::milliseconds(2));
                    // Stop signal. One iteration is 2 ms so we break out immediately (no waiting for the 5 s timeout).
                    // **No exception** — the catch below is for real failures, and a user stop is not a failure.
                    if (!slot && stop_requested()) break;
                    if (!slot && Clock::now() > deadline) {
                        ++slot_timeout_count;
                        throw std::runtime_error("slot wait timeout (5 s): compute/display completion did not arrive");
                    }
                }
            }
            // Left without a slot = stop signal. This already-decoded frame is dropped without being submitted.
            // n is not incremented either, so the decoded/submitted/completed counts still match.
            if (!slot) break;
            wait_block_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - w0).count());

#ifdef YVO_HAVE_DISPLAY
            // ⚠ Captured **before** f.reset(). Glass-to-glass starts here; decode-to-screen at d1 (includes the slot wait).
            if (display) display->retain(*slot, *f, d1);
#endif
            pre.submit(*f, *slot);
            gpu_q.push_back(*slot);
            live_frames.push_back(f->keepalive);
            f.reset();  // keepalive is held by SandLetterbox (and the display path)

            // GPU depth limit + slots already finished go straight to the NPU
            while (static_cast<int>(gpu_q.size()) > a.gpu_depth) finish_gpu();
            while (!gpu_q.empty() && pre.ready(gpu_q.front())) finish_gpu();

            stage->pump();
            flush_results();
#ifdef YVO_HAVE_DISPLAY
            if (display) display->pump();
#endif
            ++n;
            // Number of decoder frames actually held (limit about 13). Sweep expired weak_ptrs and keep the maximum.
            for (auto it = live_frames.begin(); it != live_frames.end();)
                it = it->expired() ? live_frames.erase(it) : it + 1;
            held_frames_max = std::max(held_frames_max, static_cast<long>(live_frames.size()));
            {
                const double el = std::chrono::duration<double>(Clock::now() - t0).count();
                fps_now = el > 0 ? static_cast<double>(n) / el : 0.0;
            }
            if (!a.quiet && n % 300 == 0) {
                const double el = std::chrono::duration<double>(Clock::now() - t0).count();
                std::printf("[progress] %ld frames, %.1f FPS, hailo %.1f C\n", n, static_cast<double>(n) / el,
                            static_cast<double>(stage->chip_temperature()));
                std::fflush(stdout);
            }
        }
        while (!gpu_q.empty()) finish_gpu();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] frame %ld: %s\n", n, e.what());
        fail = true;
        while (!gpu_q.empty()) {  // clean up remaining GPU work without handing it to the NPU
            pre.wait(gpu_q.front());
            gpu_q.pop_front();
        }
    }
    stage->drain();
    flush_results();   // the last shows of matched go out here
#ifdef YVO_HAVE_DISPLAY
    if (display) display->drain();   // wait for all display fences and release decoder frames / import pins
#endif
    // Whether it was interrupted is decided only after the drain finishes (a signal received during the drain also counts as a stop).
    int stop_signal = 0;
    if (g_stop.load(std::memory_order_relaxed)) {
        stop_signal = g_stop_signal.load(std::memory_order_relaxed);
        if (stop_signal == 0) stop_signal = SIGINT;  // treat as SIGINT if the handler was just about to write the number
    }
    const bool interrupted = stop_signal != 0;
    if (interrupted) fail = true;  // ok in RESULT_CSV/RESULT_CSV2 becomes 0 (must not mix into bench aggregation)

    const auto t1 = Clock::now();
    yvo::tool::perf_window(false); const double c1 = cpu_ms();  // ...and closes here

    if (!dec.error().empty()) {
        std::fprintf(stderr, "decode error after %ld frames: %s\n", n, dec.error().c_str());
        fail = true;
    }
    const float hailo_t1 = stage->chip_temperature();
    // Snapshot GPU post-processing stats before releasing stage.
    Stats sp, sq, sc, sh, sgn;
    long cand_overflow = 0, nms_fb = 0, nms_fb_cap = 0, nms_fb_guard = 0, nms_mismatch = 0;
    if (gpu_stage) {
        sgn = stats(gpu_stage->gpu_nms_ms());
        nms_fb = gpu_stage->nms_fallback_frames();
        nms_fb_cap = gpu_stage->nms_fallback_capacity();
        nms_fb_guard = gpu_stage->nms_fallback_guard();
        nms_mismatch = gpu_stage->nms_verify_mismatch();
        sp = stats(gpu_stage->post_ms());
        sq = stats(gpu_stage->nms_us());
        sc = stats(gpu_stage->cand());
        sh = stats(gpu_stage->hold_ms());
        cand_overflow = gpu_stage->overflow_frames();
    }
    // Snapshot display stats too before releasing display.
    Stats sdc, sdo, sdi, sdg, sdd;
    unsigned long long presented = 0, disp_dropped = 0, recreates = 0, ov_overflow = 0;
    const char* disp_backend = "off";
    // Plane scanout verdict (wayland-planes / kms only. Otherwise "-")
    std::string sc_zero_copy = "-", sc_color_ok = "-", sc_underrun = "-", sc_ov_skipped = "-", sc_cb_timeouts = "-",
                sc_zc_feedback = "-";
    std::vector<std::string> sc_lines, sc_transport_lines;
    std::string sc_fatal;
    unsigned long long sc_busy_reuse = 0, sc_feedback = 0;
    bool sc_applicable = false;
    // RESULT_CSV3: warmup cut (--display-warmup) and plane-path late frames. "-" = not applicable
    std::size_t disp_warmup = 0, glass_raw_n = 0;
    std::string sc_late = "-", sc_late_gap = "-", sc_gap_max = "-", sc_gaps = "-", sc_discarded = "-", sc_refresh = "-",
                sc_late_rule = "-", sc_breaks = "-";
#ifdef YVO_HAVE_DISPLAY
    if (display) {
        sdc = stats(display->convert_ms());
        sdo = stats(display->overlay_ms());
        // The same first-N cut for every backend: interval j ends at presentation j+1, so dropping N of each series
        // removes every sample that touches the first N presentations.
        disp_warmup = static_cast<std::size_t>(a.disp.warmup);
        auto cut = [&](const std::vector<double>& v) {
            return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(std::min(v.size(), disp_warmup)), v.end());
        };
        glass_raw_n = display->glass_ms().size();
        sdi = stats(cut(display->present_interval_ms()));
        sdg = stats(cut(display->glass_ms()));
        sdd = stats(cut(display->decode_to_screen_ms()));   // pairs with glass_ms (same screen time), so the same cut
        presented = display->presented();
        disp_dropped = display->display_dropped();
        recreates = display->recreates();
        ov_overflow = display->overlay_overflows();
        disp_backend = display->backend_name();
        const yvo::display::DisplaySink::ScanoutReport rep = display->scanout_report();
        if (rep.applicable) {
            sc_applicable = true;
            sc_zero_copy = rep.zero_copy;
            sc_color_ok = rep.color_ok;
            sc_underrun = rep.underrun_delta >= 0 ? std::to_string(rep.underrun_delta) : std::string("unverified");
            sc_ov_skipped = std::to_string(rep.overlay_skipped);
            sc_cb_timeouts = std::to_string(rep.frame_cb_timeouts);
            sc_zc_feedback = std::to_string(rep.zero_copy_feedback);
            sc_lines = rep.lines;
            sc_transport_lines = rep.transport_lines;
            sc_fatal = rep.fatal;
            sc_busy_reuse = rep.busy_reuse;
            sc_feedback = rep.feedback_presented;
            sc_late = rep.late_frames >= 0 ? std::to_string(rep.late_frames) : std::string("unverified");
            sc_late_gap = rep.late_expected_gap > 0 ? std::to_string(rep.late_expected_gap) : std::string("unverified");
            sc_gap_max = std::to_string(rep.seq_gap_max);
            sc_gaps = std::to_string(rep.seq_gaps);
            sc_breaks = std::to_string(rep.seq_breaks);
            sc_discarded = std::to_string(rep.discarded);
            char hz[96];
            std::snprintf(hz, sizeof(hz), "%.3f", rep.refresh_hz);
            sc_refresh = rep.refresh_hz > 0.0 ? std::string(hz) : std::string("unverified");
            std::snprintf(hz, sizeof(hz), "gap > %d = max(1, ceil(every %d x %.3f Hz / %.3f fps - 0.01))",
                          rep.late_expected_gap, a.disp.every, rep.refresh_hz, src_fps);
            sc_late_rule = rep.late_expected_gap > 0 ? std::string(hz) : std::string("gap rule unverified: refresh unknown");
        }
    }
#endif
    // Shutdown order: drain (above) → clear import cache → decoder close → runner release (dma-buf unmap)
    //            → display release (swapchain/window / plane transport/box ring) → export fd close (FdCloser) → ctx → window
    pre.clear_import_cache();
    dec.close();
    release_stages();   // runner release → display release (release_on_exit is a no-op after this)
    jsonl.reset();   // fclose: same flush point as before
    const double soc_t1 = soc_temperature();

    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    const double fps = static_cast<double>(completed) / wall_s;
    const double cpu_per_frame = completed ? (c1 - c0) / static_cast<double>(completed) : 0.0;
    const Stats sd = stats(decode_ms), sg = stats(gpu_ms), sn = stats(npu_ms), se = stats(e2e_ms),
                sw = stats(wait_block_ms);
    // Empty after the --display-warmup cut (or display off) → "nan", not a 0 that looks like a measurement.
    char num_buf[32];
    std::snprintf(num_buf, sizeof(num_buf), "%.2f", sdg.median);
    const std::string glass_med_s = sdg.n ? std::string(num_buf) : std::string("nan");
    std::snprintf(num_buf, sizeof(num_buf), "%.3f", sdi.p90);
    const std::string pres_p90_s = sdi.n ? std::string(num_buf) : std::string("nan");
    const auto num2 = [&](double v) {   // decode-to-screen: "nan" when empty after the cut, like glass_median
        std::snprintf(num_buf, sizeof(num_buf), "%.2f", v);
        return sdd.n ? std::string(num_buf) : std::string("nan");
    };
    const std::string d2s_mean_s = num2(sdd.mean), d2s_med_s = num2(sdd.median), d2s_p90_s = num2(sdd.p90);
    if (static_cast<long>(completed) != n || static_cast<long>(completed) != submitted_npu) {
        std::fprintf(stderr, "[error] frame count mismatch: decoded=%ld npu_submitted=%ld completed=%ld\n", n, submitted_npu,
                     completed);
        fail = true;
    }
    if (errors) fail = true;

    const std::string postproc_desc =
        a.postproc == Postproc::Gpu ? std::string("gpu/") + a.postproc_alloc : std::string("host");
    if (interrupted)
        std::printf("[interrupted] %s — stopped at frame %ld (the stats below are partial-run values, ok=0)\n",
                    g_stop_by_key                ? "display window key press (Esc/q/Ctrl+C)"
                    : stop_signal == SIGTERM ? "SIGTERM received"
                                             : "SIGINT received",
                    n);
    std::printf("== yvo_pipeline (%s, %s, batch %d, slots %d, gpu_depth %d, postproc %s) ==\n",
                dmabuf ? "dmabuf" : "ptr", layout_name(a.layout), a.batch, a.slots, a.gpu_depth,
                postproc_desc.c_str());
    std::printf("frames          %ld (decoded %ld, errors %ld)\n", completed, n, errors);
    std::printf("wall            %.3f s, %.2f FPS\n", wall_s, fps);
    std::printf("cpu             %.3f ms/frame (process user+sys %.1f ms, %.0f%% of one core)\n", cpu_per_frame,
                c1 - c0, (c1 - c0) / (wall_s * 10.0));
    std::printf("decode call ms  mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sd.mean, sd.median, sd.p90, sd.max);
    std::printf("gpu shader ms   mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sg.mean, sg.median, sg.p90, sg.max);
    std::printf("npu latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f\n", sn.mean, sn.median, sn.p90, sn.max);
    std::printf("e2e latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f  (decode return → NPU completion callback)\n", se.mean,
                se.median, se.p90, se.max);
    std::printf("slot wait ms    mean %.3f  p90 %.3f  (time the main thread waited for a free slot)\n", sw.mean, sw.p90);
    if (a.postproc == Postproc::Gpu) {
        std::printf("gpu post ms     mean %.3f  median %.3f  p90 %.3f  max %.3f  (yolo_decode.comp pass 1)\n", sp.mean,
                    sp.median, sp.p90, sp.max);
        std::printf("cpu nms us      mean %.3f  median %.3f  p90 %.3f  max %.3f  (cpu_nms_greedy_by_class, n=%zu)\n",
                    sq.mean, sq.median, sq.p90, sq.max, sq.n);
        std::printf("nms backend     %s\n", nms_name(a.nms));
        if (a.nms != yvo::pipeline::NmsBackend::Cpu) {
            std::printf("gpu nms ms      mean %.4f  median %.4f  p90 %.4f  max %.4f  (efficient_nms.comp)\n", sgn.mean,
                        sgn.median, sgn.p90, sgn.max);
            std::printf("nms fallback    %ld frames (capacity %ld, iou guard %ld)\n", nms_fb, nms_fb_cap, nms_fb_guard);
        }
        if (a.nms == yvo::pipeline::NmsBackend::Verify)
            std::printf("nms verify mismatch %ld frames (frames where the GPU result differs bitwise from the CPU reference, excluding fallback)\n",
                        nms_mismatch);
        std::printf("candidates      mean %.2f  max %.0f  overflow frames %ld\n", sc.mean, sc.max, cand_overflow);
        std::printf("slot hold ms    mean %.3f  median %.3f  p90 %.3f  max %.3f  (NPU completion callback → slot return)\n",
                    sh.mean, sh.median, sh.p90, sh.max);
    }
    if (!sc_fatal.empty()) {
        std::fprintf(stderr, "[error] display transport: %s\n", sc_fatal.c_str());
        fail = true;
    }
    if (a.display && sc_applicable) {
        std::printf("disp convert ms -  (%s: YUV→RGB conversion is done by the HVS during scanout, V3D 0)\n", disp_backend);
        std::printf("disp overlay ms mean %.3f  median %.3f  p90 %.3f  (overlay_boxes_buf.spv + clear, box plane)\n",
                    sdo.mean, sdo.median, sdo.p90);
    } else if (a.display) {
        std::printf("disp convert ms mean %.3f  median %.3f  p90 %.3f  (display_convert.comp)\n", sdc.mean,
                    sdc.median, sdc.p90);
        std::printf("disp overlay ms mean %.3f  median %.3f  p90 %.3f  (overlay_boxes.comp)\n", sdo.mean, sdo.median,
                    sdo.p90);
    }
    if (a.display) {
        std::printf("present interval ms median %.3f  p90 %.3f  (n=%zu)\n", sdi.median, sdi.p90, sdi.n);
        std::printf("glass-to-glass  mean %.2f  median %.2f  p90 %.2f  (n=%zu, slot acquired → reaches screen)\n", sdg.mean,
                    sdg.median, sdg.p90, sdg.n);
        std::printf("decode-to-screen mean %.2f  median %.2f  p90 %.2f  (n=%zu, decode return → reaches screen, includes slot wait)\n",
                    sdd.mean, sdd.median, sdd.p90, sdd.n);
        std::printf("display         presented %llu, dropped %llu, recreates %llu, overlay overflow %llu\n",
                    presented, disp_dropped, recreates, ov_overflow);
        std::printf("display warmup  %zu presentations excluded from glass / decode-to-screen / present interval / late frames "
                    "(glass n %zu of %zu, glass median %s)\n", disp_warmup, sdg.n, glass_raw_n, glass_med_s.c_str());
        if (sc_applicable) {
            std::printf("scanout         zero_copy %s, color_ok %s, hvs_underrun delta %s (debugfs verdict)\n",
                        sc_zero_copy.c_str(), sc_color_ok.c_str(), sc_underrun.c_str());
            for (const auto& l : sc_lines) std::printf("  debugfs %s\n", l.c_str());
            for (const auto& l : sc_transport_lines) std::printf("  %s\n", l.c_str());
            std::printf("scanout         overlay skipped %s, frame-callback timeouts %s, busy reuse %llu, "
                        "presentation ZERO_COPY %s/%llu\n",
                        sc_ov_skipped.c_str(), sc_cb_timeouts.c_str(), sc_busy_reuse, sc_zc_feedback.c_str(),
                        sc_feedback);
            std::printf("scanout         late frames %s (vblank seq %s; gaps %s, max gap %s, chain breaks %s), discarded %s\n",
                        sc_late.c_str(), sc_late_rule.c_str(), sc_gaps.c_str(), sc_gap_max.c_str(), sc_breaks.c_str(),
                        sc_discarded.c_str());
        }
    }
    std::printf("held frames max %ld (decoder limit about 13), slot timeout %ld\n", held_frames_max, slot_timeout_count);
    std::printf("detections      %ld total, %.2f / frame\n", total_dets,
                completed ? static_cast<double>(total_dets) / static_cast<double>(completed) : 0.0);
    std::printf("hailo chip temp %.1f -> %.1f C\n", static_cast<double>(hailo_t0), static_cast<double>(hailo_t1));
    std::printf("soc temp        %.1f -> %.1f C\n", soc_t0, soc_t1);
    if (ctx.debug_messenger_active())
        std::printf("vk validation   errors %llu, warnings %llu\n",
                    static_cast<unsigned long long>(ctx.validation_errors()),
                    static_cast<unsigned long long>(ctx.validation_warnings()));
    if (!a.csv_label.empty()) {
        // label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_mean,gpu_mean,gpu_p90,npu_mean,
        // npu_p90,e2e_mean,e2e_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok
        std::printf(
            "RESULT_CSV,%s,%s,%s,%d,%d,%ld,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%ld,%.1f,%.1f,%.1f,%.1f,%d\n",
            a.csv_label.c_str(), dmabuf ? "dmabuf" : "ptr", layout_name(a.layout), a.batch, a.slots, completed, wall_s,
            fps, cpu_per_frame, sd.mean, sg.mean, sg.p90, sn.mean, sn.p90, se.mean, se.p90, total_dets,
            static_cast<double>(hailo_t0), static_cast<double>(hailo_t1), soc_t0, soc_t1, fail ? 0 : 1);
        // New metrics go on a **separate line**. The 22 fields of RESULT_CSV do not change order or count
        // (06_tools/bench/pipeline_bench.sh:120 compares the header with grep -qx).
        // label,postproc,alloc,gpu_post_mean,gpu_post_p90,cpu_nms_us_mean,cand_mean,cand_max,cand_overflow,ok
        // + display metrics are appended **only at the end** (the first 10 fields do not change order or count):
        //   display,sync,pace,disp_convert_mean,disp_convert_p90,disp_overlay_mean,present_interval_median,
        //   e2e_glass_mean,e2e_glass_p90,presented,display_dropped,recreates,held_frames_max,slot_timeout_count
        // + GPU NMS metrics are also appended **only at the end**: nms,gpu_nms_mean,gpu_nms_p90,nms_fallback,nms_verify_mismatch
        // + plane scanout metrics are also appended **only at the end** (all "-" except for wayland-planes / kms):
        //   scanout_zero_copy,plane_color_ok,hvs_underrun_delta,overlay_skipped,frame_cb_timeouts,zero_copy_feedback
        //   (scanout_zero_copy/plane_color_ok = 1/0/unverified — judged from debugfs state; hvs_underrun_delta is unverified if it cannot be read)
        const char* sync_s = "-";
        const char* pace_s = "-";
#ifdef YVO_HAVE_DISPLAY
        if (a.display) {
            sync_s = yvo::display::sync_name(a.disp.sync);
            pace_s = yvo::display::pace_name(a.disp.pace);
        }
#endif
        std::printf("RESULT_CSV2,%s,%s,%s,%.4f,%.4f,%.3f,%.2f,%.0f,%ld,%d"
                    ",%s,%s,%s,%.4f,%.4f,%.4f,%.3f,%.2f,%.2f,%llu,%llu,%llu,%ld,%ld"
                    ",%s,%.4f,%.4f,%ld,%ld"
                    ",%s,%s,%s,%s,%s,%s\n",
                    a.csv_label.c_str(), a.postproc == Postproc::Gpu ? "gpu" : "host",
                    a.postproc == Postproc::Gpu ? a.postproc_alloc.c_str() : "-", sp.mean, sp.p90, sq.mean, sc.mean,
                    sc.max, cand_overflow, fail ? 0 : 1,
                    disp_backend, sync_s, pace_s, sdc.mean, sdc.p90, sdo.mean, sdi.median, sdg.mean, sdg.p90,
                    presented, disp_dropped, recreates, held_frames_max, slot_timeout_count,
                    a.postproc == Postproc::Gpu ? nms_name(a.nms) : "-", sgn.mean, sgn.p90, nms_fb, nms_mismatch,
                    sc_zero_copy.c_str(), sc_color_ok.c_str(), sc_underrun.c_str(), sc_ov_skipped.c_str(),
                    sc_cb_timeouts.c_str(), sc_zc_feedback.c_str());
        // Third line (RESULT_CSV/RESULT_CSV2 keep their field counts: display_bench.sh 59, dec_threads_bench.sh 65 and
        // nms_bench.sh 77 columns check them). label first, ok at field 14 (d2s_* follow it):
        //   label,display_warmup,glass_median,glass_n,present_interval_p90,present_interval_n,late_frames,late_expected_gap,
        //   seq_gap_max,seq_gaps,seq_breaks,discarded,refresh_hz,ok,d2s_mean,d2s_median,d2s_p90,d2s_n
        // d2s_* (decode-to-screen) are appended **after ok** (fields 1..14 keep their positions; same cut as glass_*).
        // glass_* / present_interval_* use the same --display-warmup cut as RESULT_CSV2's e2e_glass_* / present_interval_median
        // (nan when nothing is left after the cut). late_frames .. refresh_hz are "-" except for wayland-planes / kms;
        // late_frames / late_expected_gap / refresh_hz are "unverified" when there is no vblank seq, refresh or gap to judge.
        std::printf("RESULT_CSV3,%s,%zu,%s,%zu,%s,%zu,%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%zu\n", a.csv_label.c_str(), disp_warmup,
                    glass_med_s.c_str(), sdg.n, pres_p90_s.c_str(), sdi.n, sc_late.c_str(), sc_late_gap.c_str(),
                    sc_gap_max.c_str(), sc_gaps.c_str(), sc_breaks.c_str(), sc_discarded.c_str(), sc_refresh.c_str(),
                    fail ? 0 : 1, d2s_mean_s.c_str(), d2s_med_s.c_str(), d2s_p90_s.c_str(), sdd.n);
    }
    std::fflush(stdout);
    if (ctx.debug_messenger_active() && ctx.validation_errors() > 0) fail = true;
    // An interrupted run returns 128+signo per shell convention (SIGINT 130, SIGTERM 143).
    if (interrupted) return 128 + stop_signal;
    return fail ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }
    install_stop_handlers();  // continue even on failure (the warning is emitted inside the function)
    try {
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
