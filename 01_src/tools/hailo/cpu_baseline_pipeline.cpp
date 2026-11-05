// yvo_cpu_baseline: the real-time "before" path of the before/after comparison. Every stage the proposed method offloads
// runs on the CPU here; only inference stays on the NPU. No GPU, no Vulkan, no dma-buf, no hardware decoder.
//
//   decode      libavcodec software HEVC (native "hevc" decoder, get_format refuses every hwaccel format)
//   preprocess  libswscale (NEON) YUV420P -> letterbox 640x640 packed RGB, same geometry as make_letterbox()
//   inference   HailoRunner: host pointer NHWC input + HailoRT NMS (HailoRT transposes NHWC -> NHCW and runs the NMS
//               post-process op on the host CPU, both before the completion callback)
//   results     parse_nms_by_class + inverse letterbox (inside HailoRunner's completion callback)
//   display     libswscale YUV420P -> RGBA in a wl_shm buffer + boxes/labels drawn by ref/overlay_reference.hpp
//
// Operating point (same as the yvo_pipeline default O2): batch 1, one frame in flight, frames arrive at the source
// frame rate (input_pacer.hpp ArrivalSchedule) into a leaky queue of 1 (full -> drop the oldest).
//
// Threads, so a slow stage does not stall another one for no reason (this path must not be slower than it has to be):
//   decode thread   waits for each frame's arrival time, decodes it, pushes it into the leaky queue
//   main thread     pops the oldest frame, swscale, submit to the NPU, waits for the completion, hands the frame on
//   display thread  owns all Wayland calls after startup; a mailbox of 1 keeps the newest result (older: display drop)
//
// --frame-log columns (toolkit/frame_log.hpp; times are ms since t0, see below):
//   arrival_ms     scheduled source arrival     decode_end_ms  decode call returned
//   submit_ms      swscale finished, right before HailoRunner::submit
//   npu_done_ms    HailoRT completion callback entered. **Includes the HailoRT NMS and the NHWC -> NHCW transpose**
//   complete_ms    main thread received the parsed detections   gpu_post_done_ms  always empty
//   present_ms     wl_surface.commit of the shm buffer flushed (not "on screen": wl_shm has no present feedback here)
//   commit_ms, present_seq  always empty (plane-path columns; kept so a display-on file always has 12 columns)
// --stage-log adds what the frame log cannot give: per-frame durations of decode, swscale, NPU and display.
//
// t0: CPU window and time base open exactly as in yvo_pipeline: cpu_ms(), perf_window(true), steady_clock::now().
// YVO_PERF_CTL therefore marks t0 for an external sampler; the summary prints t0 on CLOCK_MONOTONIC as well.
// Per-frame records stay in memory; every file is written after the run.
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <unistd.h>
#include <wayland-client.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
}

#include "core/coco_labels.hpp"
#include "core/frame_types.hpp"
#include "display/wayland_window.hpp"
#include "hailo/hailo_runner.hpp"
#include "pipeline/input_pacer.hpp"
#include "ref/overlay_reference.hpp"
#include "toolkit/cli_parse.hpp"
#include "toolkit/frame_log.hpp"
#include "toolkit/result_csv.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/sws_preproc.hpp"
#include "toolkit/tool_util.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using yvo::Detection;
using yvo::Stats;
using yvo::stats;
using yvo::tool::arg_number;
using yvo::tool::cpu_ms;
using yvo::tool::soc_temperature;

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

double ms_of(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

// ── Stop flag: same contract as pipeline/main.cpp (the handler only sets atomics; a second signal exits at once) ──
std::atomic<bool> g_stop{false};
std::atomic<int> g_stop_signal{0};
std::atomic<bool> g_stop_by_key{false};

void on_stop_signal(int sig) {
    if (g_stop.exchange(true, std::memory_order_relaxed)) _exit(130);
    g_stop_signal.store(sig, std::memory_order_relaxed);
}
bool stop_requested() { return g_stop.load(std::memory_order_relaxed); }

void install_stop_handlers() {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_stop_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    for (const int sig : {SIGINT, SIGTERM})
        if (sigaction(sig, &sa, nullptr) != 0)
            std::fprintf(stderr, "[warn] sigaction(%d) failed: %s\n", sig, std::strerror(errno));
}

struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string hef = YVO_DEFAULT_HEF;
    long frames = 0;              // arrived frames; 0 = the whole video
    int dec_threads = 0;          // libavcodec thread_count; 0 = library auto
    int sws_threads = 4;          // libswscale slice threads, preprocessing and display conversion
    // The 3:1 scaler. fast_bilinear (2 taps) is the closest swscale kernel to the reference letterbox (also 2 taps) and
    // the fastest; full_chroma_int interpolates chroma per output pixel. Detection comparison with the other kernels:
    // reported with this tool's introduction (bilinear loses 18.5% of the boxes at the 0.2 score threshold).
    std::string sws_flags = "fast_bilinear+full_chroma_int";
    int input_queue = 1;
    bool display = false;
    bool fullscreen = false;
    int disp_w = 1280, disp_h = 720;
    std::string output = "HDMI-A-1";
    int display_timeout_ms = 2000;
    int line_width = 2, text_scale = 2;   // yvo_pipeline's --overlay-line-width / --overlay-text-scale defaults
    std::string frame_log, stage_log, dump_dir, csv_label;
    bool quiet = false;
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--hef PATH] [--frames N] [--dec-threads N] [--sws-threads N] [--sws-flags F]\n"
                 "          [--input-queue N] [--display on|off] [--display-fullscreen] [--display-size WxH]\n"
                 "          [--display-output NAME] [--display-timeout-ms N] [--overlay-line-width N] [--overlay-text-scale N]\n"
                 "          [--frame-log PATH] [--stage-log PATH] [--dump-dir DIR] [--quiet] [--csv-label L]\n"
                 "  CPU-only baseline: software HEVC -> libswscale letterbox -> HailoRT NHWC + host NMS -> CPU drawing -> wl_shm.\n"
                 "  Operating point: batch 1, one frame in flight, frames arrive at the source rate into a leaky queue.\n"
                 "  --frames N          arrived frames (default: the whole video)\n"
                 "  --dec-threads N     libavcodec threads (default 0 = library auto)\n"
                 "  --sws-threads N     libswscale slice threads (default 4)\n"
                 "  --sws-flags F       preprocessing scaler flags (default fast_bilinear+full_chroma_int)\n"
                 "  --input-queue N     leaky queue size, 1..3 (default 1)\n"
                 "  --display on|off    wl_shm window (default off)\n"
                 "  --display-fullscreen | --display-size WxH   default window 1280x720\n"
                 "  --display-output N  output that must exist before the window opens (default HDMI-A-1)\n"
                 "  --display-timeout-ms N  longest wait for the compositor (frame callback / free buffer), default 2000\n"
                 "  --frame-log PATH    toolkit/frame_log.hpp CSV, written at the end\n"
                 "  --stage-log PATH    src,decode_ms,swscale_ms,npu_ms,post_ms,disp_convert_ms,disp_overlay_ms per frame, written at the end\n"
                 "  --dump-dir DIR      detections.jsonl (same line format as yvo_pipeline), written at the end. No PNG\n"
                 "  Esc/q/Ctrl+C in the window stops the run like SIGINT (exit code 130)\n",
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--hef" && (v = next())) a.hef = v;
        else if (s == "--frames" && (v = next())) a.frames = arg_number<long>(argv[0], s, v);
        else if (s == "--dec-threads" && (v = next())) a.dec_threads = arg_number<int>(argv[0], s, v);
        else if (s == "--sws-threads" && (v = next())) a.sws_threads = arg_number<int>(argv[0], s, v);
        else if (s == "--sws-flags" && (v = next())) a.sws_flags = v;
        else if (s == "--input-queue" && (v = next())) a.input_queue = arg_number<int>(argv[0], s, v);
        else if (s == "--display" && (v = next())) {
            const std::string d = v;
            if (d == "on") a.display = true;
            else if (d == "off") a.display = false;
            else return false;
        } else if (s == "--display-fullscreen") a.fullscreen = true;
        else if (s == "--display-size" && (v = next())) {
            if (!yvo::tool::parse_size(v, a.disp_w, a.disp_h)) return false;
        } else if (s == "--display-output" && (v = next())) a.output = v;
        else if (s == "--display-timeout-ms" && (v = next())) a.display_timeout_ms = arg_number<int>(argv[0], s, v);
        else if (s == "--overlay-line-width" && (v = next())) a.line_width = arg_number<int>(argv[0], s, v);
        else if (s == "--overlay-text-scale" && (v = next())) a.text_scale = arg_number<int>(argv[0], s, v);
        else if (s == "--frame-log" && (v = next())) a.frame_log = v;
        else if (s == "--stage-log" && (v = next())) a.stage_log = v;
        else if (s == "--dump-dir" && (v = next())) a.dump_dir = v;
        else if (s == "--csv-label" && (v = next())) a.csv_label = v;
        else if (s == "--quiet") a.quiet = true;
        else return false;
    }
    if (a.frames < 0 || a.dec_threads < 0 || a.sws_threads < 1 || a.display_timeout_ms < 1 || a.line_width < 1 ||
        a.text_scale < 1)
        return false;
    if (a.input_queue < yvo::pipeline::kMinInputQueue || a.input_queue > yvo::pipeline::kMaxInputQueue) return false;
    return true;
}

// ── Software HEVC decoder ──────────────────────────────────────────────────────────────────────
struct FrameDeleter {
    void operator()(AVFrame* f) const { av_frame_free(&f); }
};
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;

std::string av_err(int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(err, buf, sizeof(buf));
    return buf;
}

// Never lets a hwaccel format win: the first format without AV_PIX_FMT_FLAG_HWACCEL is the software output.
AVPixelFormat pick_software_format(AVCodecContext*, const AVPixelFormat* fmts) {
    for (; *fmts != AV_PIX_FMT_NONE; ++fmts) {
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(*fmts);
        if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) return *fmts;
    }
    return AV_PIX_FMT_NONE;
}

class SwHevcDecoder {
public:
    SwHevcDecoder(const std::string& path, int threads) {
        int r = avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr);
        if (r < 0) throw std::runtime_error("avformat_open_input(" + path + "): " + av_err(r));
        r = avformat_find_stream_info(fmt_, nullptr);
        if (r < 0) throw std::runtime_error("avformat_find_stream_info: " + av_err(r));
        vs_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (vs_ < 0) throw std::runtime_error("no video stream in " + path);
        const AVStream* st = fmt_->streams[vs_];
        if (st->codecpar->codec_id != AV_CODEC_ID_HEVC) throw std::runtime_error("the video stream is not HEVC");
        const AVCodec* codec = avcodec_find_decoder_by_name("hevc");   // never hevc_v4l2m2m or another wrapper
        if (!codec) throw std::runtime_error("libavcodec has no native \"hevc\" decoder");
        if (codec->capabilities & AV_CODEC_CAP_HARDWARE)
            throw std::runtime_error(std::string("decoder ") + codec->name + " is hardware-backed");
        ctx_ = avcodec_alloc_context3(codec);
        if (!ctx_) throw std::runtime_error("avcodec_alloc_context3 failed");
        r = avcodec_parameters_to_context(ctx_, st->codecpar);
        if (r < 0) throw std::runtime_error("avcodec_parameters_to_context: " + av_err(r));
        ctx_->get_format = pick_software_format;   // no hw_device_ctx is set either
        ctx_->thread_count = threads;              // 0 = library auto
        ctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
        r = avcodec_open2(ctx_, codec, nullptr);
        if (r < 0) throw std::runtime_error("avcodec_open2: " + av_err(r));
        pkt_ = av_packet_alloc();
        if (!pkt_) throw std::runtime_error("av_packet_alloc failed");
        const AVRational fr = st->avg_frame_rate;
        fps_ = fr.num > 0 && fr.den > 0 ? av_q2d(fr) : 30.0;
        std::printf("[decoder] codec=%s (%s) software=%d hw_device_ctx=%s threads=%d thread_type=%s fps=%.3f\n", codec->name,
                    codec->long_name, (codec->capabilities & AV_CODEC_CAP_HARDWARE) ? 0 : 1, ctx_->hw_device_ctx ? "set" : "none",
                    ctx_->thread_count,
                    ctx_->active_thread_type == FF_THREAD_FRAME   ? "frame"
                    : ctx_->active_thread_type == FF_THREAD_SLICE ? "slice"
                                                                  : "none",
                    fps_);
    }
    ~SwHevcDecoder() {
        av_packet_free(&pkt_);
        avcodec_free_context(&ctx_);
        avformat_close_input(&fmt_);
    }
    SwHevcDecoder(const SwHevcDecoder&) = delete;
    SwHevcDecoder& operator=(const SwHevcDecoder&) = delete;

    double fps() const { return fps_; }

    // The next frame in display order; nullptr at the end of the stream. Throws on a decode error.
    FramePtr next() {
        FramePtr f(av_frame_alloc());
        if (!f) throw std::runtime_error("av_frame_alloc failed");
        for (;;) {
            int r = avcodec_receive_frame(ctx_, f.get());
            if (r == 0) {
                if (!checked_) check_first_frame(*f);
                return f;
            }
            if (r == AVERROR_EOF) return nullptr;
            if (r != AVERROR(EAGAIN)) throw std::runtime_error("avcodec_receive_frame: " + av_err(r));
            if (flushing_) throw std::runtime_error("avcodec_receive_frame: EAGAIN after the flush packet");
            r = av_read_frame(fmt_, pkt_);
            if (r == AVERROR_EOF) {
                flushing_ = true;
                r = avcodec_send_packet(ctx_, nullptr);
            } else if (r < 0) {
                throw std::runtime_error("av_read_frame: " + av_err(r));
            } else if (pkt_->stream_index != vs_) {
                av_packet_unref(pkt_);
                continue;
            } else {
                r = avcodec_send_packet(ctx_, pkt_);
                av_packet_unref(pkt_);
            }
            if (r < 0 && r != AVERROR_EOF) throw std::runtime_error("avcodec_send_packet: " + av_err(r));
        }
    }

private:
    // The colour conversion assumes 8-bit limited-range YUV420P; a different stream must stop here, not look wrong.
    void check_first_frame(const AVFrame& f) {
        checked_ = true;
        const char* name = av_get_pix_fmt_name(static_cast<AVPixelFormat>(f.format));
        std::printf("[decoder] first frame: pix_fmt=%s %dx%d hw_frames_ctx=%s range=%s colorspace=%s chroma=%s\n",
                    name ? name : "?", f.width, f.height, f.hw_frames_ctx ? "set" : "none",
                    av_color_range_name(f.color_range), av_color_space_name(f.colorspace),
                    av_chroma_location_name(f.chroma_location));
        if (f.format != AV_PIX_FMT_YUV420P || f.hw_frames_ctx)
            throw std::runtime_error("expected software yuv420p from the decoder");
    }

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* ctx_ = nullptr;
    AVPacket* pkt_ = nullptr;
    int vs_ = -1;
    double fps_ = 30.0;
    bool flushing_ = false;
    bool checked_ = false;
};

// ── Per-frame record. One owner at a time: decode thread -> queue -> main thread -> display thread; read after join ──
struct FrameRec {
    std::int64_t src = 0;
    std::int64_t seq = -1;
    bool dropped = false;
    Clock::time_point arrival{}, decode_end{}, submit{}, npu_done{}, complete{}, present{};
    double decode_ms = kNan, swscale_ms = kNan, npu_ms = kNan, post_ms = kNan;
    double disp_convert_ms = kNan, disp_overlay_ms = kNan;
    std::vector<Detection> dets;
};

struct QueuedFrame {
    FramePtr f;
    FrameRec* rec = nullptr;
};

// ── wl_shm display ─────────────────────────────────────────────────────────────────────────────
// WaylandWindow is reused for the xdg-shell window and the quit keys. This class adds wl_shm, the wl_output check and
// the buffers. Nothing here touches Vulkan: the link line drops libvulkan (--as-needed, 01_src/tools/CMakeLists.txt).
class ShmDisplay {
public:
    struct Item {
        FramePtr f;
        FrameRec* rec = nullptr;
        double fps = 0.0;
    };

    explicit ShmDisplay(const Args& a) : a_(a) {
        wake_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (wake_fd_ < 0) throw std::runtime_error("eventfd failed");
        yvo::display::WaylandWindowOptions opt;
        opt.width = a.disp_w;
        opt.height = a.disp_h;
        opt.title = "yvo_cpu_baseline";
        opt.app_id = "yvo_cpu_baseline";
        opt.fullscreen = a.fullscreen;
        if (!win_.open(opt)) throw std::runtime_error("Wayland window: " + win_.error());

        // Own registry: wl_shm and wl_output are not bound by WaylandWindow. Two roundtrips: globals, then their events.
        registry_ = wl_display_get_registry(win_.display());
        static const wl_registry_listener reg_listener = {on_global, on_global_remove};
        wl_registry_add_listener(registry_, &reg_listener, this);
        if (wl_display_roundtrip(win_.display()) < 0 || wl_display_roundtrip(win_.display()) < 0)
            throw std::runtime_error("Wayland roundtrip failed");
        if (!shm_) throw std::runtime_error("the compositor does not offer wl_shm");
        if (!xbgr_ok_)   // memory order R,G,B,X: what ref/overlay_reference.hpp and AV_PIX_FMT_RGBA write
            throw std::runtime_error("the compositor's wl_shm does not list XBGR8888");
        if (outputs_.empty()) throw std::runtime_error("the compositor has no wl_output");
        bool found = false, named = false;
        std::string names;
        for (const auto& o : outputs_) {
            names += (names.empty() ? "" : ", ") + (o->name.empty() ? std::string("?") : o->name);
            named = named || !o->name.empty();
            found = found || o->name == a.output;
        }
        if (named && !found)
            throw std::runtime_error("output " + a.output + " is not connected or not enabled (compositor outputs: " + names + ")");
        if (!named) std::fprintf(stderr, "[warn] wl_output below v4: cannot verify output %s\n", a.output.c_str());
        std::printf("[display] wl_shm XBGR8888 window: outputs %s, requested %s, %s\n", names.c_str(), a.output.c_str(),
                    a.fullscreen ? "fullscreen" : "windowed");
    }
    ~ShmDisplay() {
        destroy_buffers();
        if (frame_cb_) wl_callback_destroy(frame_cb_);
        for (auto& o : outputs_)
            if (o->out) wl_output_destroy(o->out);
        if (shm_) wl_shm_destroy(shm_);
        if (registry_) wl_registry_destroy(registry_);
        win_.close();
        if (wake_fd_ >= 0) ::close(wake_fd_);
    }
    ShmDisplay(const ShmDisplay&) = delete;
    ShmDisplay& operator=(const ShmDisplay&) = delete;

    // Main thread: hand over a finished frame. An item the display has not taken yet is replaced (a display drop).
    void post(Item item) {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (pending_) ++dropped_;
            pending_ = std::make_unique<Item>(std::move(item));
        }
        wake();
    }
    // Main thread: no more items. run() returns after the last one is on its way to the compositor.
    void close_input() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        wake();
    }

    // Display thread body. Every wait is bounded by --display-timeout-ms, so a compositor that stopped presenting
    // (output disabled, DRM master lost) ends the run with an error instead of hanging it.
    void run() {
        try {
            loop();
        } catch (const std::exception& e) {
            error_ = e.what();
            g_stop.store(true, std::memory_order_relaxed);
        }
    }

    const std::string& error() const { return error_; }
    long presented() const { return presented_; }
    long dropped() const { return dropped_; }
    bool window_closed() const { return window_closed_; }

private:
    struct Output {
        ShmDisplay* self = nullptr;
        wl_output* out = nullptr;
        std::string name;
    };
    struct Buffer {
        wl_buffer* buf = nullptr;
        std::uint8_t* px = nullptr;
        bool busy = false;
    };

    static void on_global(void* data, wl_registry* r, std::uint32_t name, const char* iface, std::uint32_t version) {
        auto* self = static_cast<ShmDisplay*>(data);
        if (std::strcmp(iface, wl_shm_interface.name) == 0 && !self->shm_) {
            self->shm_ = static_cast<wl_shm*>(wl_registry_bind(r, name, &wl_shm_interface, 1));
            static const wl_shm_listener l = {on_shm_format};
            wl_shm_add_listener(self->shm_, &l, self);
        } else if (std::strcmp(iface, wl_output_interface.name) == 0) {
            auto o = std::make_unique<Output>();
            o->self = self;
            o->out = static_cast<wl_output*>(wl_registry_bind(r, name, &wl_output_interface, std::min(version, 4u)));
            static const wl_output_listener l = {
                [](void*, wl_output*, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, const char*,
                   const char*, std::int32_t) {},
                [](void*, wl_output*, std::uint32_t, std::int32_t, std::int32_t, std::int32_t) {},
                [](void*, wl_output*) {},
                [](void*, wl_output*, std::int32_t) {},
                [](void* d, wl_output*, const char* n) { static_cast<Output*>(d)->name = n; },
                [](void*, wl_output*, const char*) {},
            };
            wl_output_add_listener(o->out, &l, o.get());
            self->outputs_.push_back(std::move(o));
        }
    }
    static void on_global_remove(void*, wl_registry*, std::uint32_t) {}
    static void on_shm_format(void* data, wl_shm*, std::uint32_t format) {
        if (format == WL_SHM_FORMAT_XBGR8888) static_cast<ShmDisplay*>(data)->xbgr_ok_ = true;
    }
    static void on_buffer_release(void* data, wl_buffer*) { static_cast<Buffer*>(data)->busy = false; }
    static void on_frame_done(void* data, wl_callback* cb, std::uint32_t) {
        auto* self = static_cast<ShmDisplay*>(data);
        wl_callback_destroy(cb);
        self->frame_cb_ = nullptr;
    }

    void wake() {
        const std::uint64_t one = 1;
        [[maybe_unused]] const ssize_t w = ::write(wake_fd_, &one, sizeof(one));
    }

    void destroy_buffers() {
        for (Buffer& b : buffers_)
            if (b.buf) wl_buffer_destroy(b.buf);
        buffers_.clear();
        if (map_) ::munmap(map_, map_size_);
        map_ = nullptr;
        map_size_ = 0;
    }

    void create_buffers(int w, int h) {
        destroy_buffers();
        constexpr int kBuffers = 3;
        const std::size_t stride = static_cast<std::size_t>(w) * 4;
        const std::size_t size = stride * static_cast<std::size_t>(h);
        map_size_ = size * kBuffers;
        const int fd = memfd_create("yvo-cpu-baseline-shm", MFD_CLOEXEC);
        if (fd < 0) throw std::runtime_error("memfd_create failed");
        if (ftruncate(fd, static_cast<off_t>(map_size_)) != 0) {
            ::close(fd);
            throw std::runtime_error("ftruncate of the shm file failed");
        }
        void* p = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd);
            map_size_ = 0;
            throw std::runtime_error("mmap of the shm file failed");
        }
        map_ = static_cast<std::uint8_t*>(p);
        wl_shm_pool* pool = wl_shm_create_pool(shm_, fd, static_cast<std::int32_t>(map_size_));
        buffers_.resize(kBuffers);   // the listeners keep pointers into this vector: no resize until destroy_buffers()
        static const wl_buffer_listener l = {on_buffer_release};
        for (int i = 0; i < kBuffers; ++i) {
            Buffer& b = buffers_[static_cast<std::size_t>(i)];
            b.px = map_ + size * static_cast<std::size_t>(i);
            b.buf = wl_shm_pool_create_buffer(pool, static_cast<std::int32_t>(size * static_cast<std::size_t>(i)), w, h,
                                              static_cast<std::int32_t>(stride), WL_SHM_FORMAT_XBGR8888);
            wl_buffer_add_listener(b.buf, &l, &b);
        }
        wl_shm_pool_destroy(pool);
        ::close(fd);
        w_ = w;
        h_ = h;
        sws_.reset();   // rebuilt on the next frame, from that frame's size
    }

    Buffer* free_buffer() {
        for (Buffer& b : buffers_)
            if (!b.busy) return &b;
        return nullptr;
    }

    void draw_and_commit(Item& it, Buffer& b) {
        const AVFrame& f = *it.f;
        const auto t0 = Clock::now();
        const yvo::LetterboxParams lb = yvo::make_letterbox(f.width, f.height, w_, h_);
        const int cw = static_cast<int>(f.width * lb.scale + 0.5f), ch = static_cast<int>(f.height * lb.scale + 0.5f);
        if (!sws_)
            sws_ = std::make_unique<yvo::tool::SwsResize>(f.width, f.height, cw, ch, AV_PIX_FMT_RGBA, "bilinear", a_.sws_threads);
        const std::uint8_t* const src[3] = {f.data[0], f.data[1], f.data[2]};
        const int src_stride[3] = {f.linesize[0], f.linesize[1], f.linesize[2]};
        const int row = w_ * 4;
        std::uint8_t* const dst[3] = {b.px + static_cast<std::size_t>(lb.pad_y) * static_cast<std::size_t>(row) +
                                          static_cast<std::size_t>(lb.pad_x) * 4,
                                      nullptr, nullptr};
        const int dst_stride[3] = {row, 0, 0};
        sws_->scale(src, src_stride, dst, dst_stride);
        const auto t1 = Clock::now();

        char hud[64];   // the same HUD line as yvo_pipeline's display path
        std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d NPU %3.0fms", it.fps, static_cast<int>(it.rec->dets.size()), it.rec->npu_ms);
        yvo::display::OverlayStyle st;
        st.line_width = a_.line_width;
        st.text_scale = a_.text_scale;
        yvo::display::overlay_cpu_render(b.px, w_, h_, it.rec->dets, hud, yvo::display::make_overlay_affine(f.width, f.height, w_, h_), st);
        const auto t2 = Clock::now();

        wl_surface* s = win_.surface();
        b.busy = true;
        wl_surface_attach(s, b.buf, 0, 0);
        wl_surface_damage_buffer(s, 0, 0, w_, h_);
        frame_cb_ = wl_surface_frame(s);
        static const wl_callback_listener cb = {on_frame_done};
        wl_callback_add_listener(frame_cb_, &cb, this);
        wl_surface_commit(s);
        if (wl_display_flush(win_.display()) < 0 && errno != EAGAIN) throw std::runtime_error("wl_display_flush failed");
        it.rec->present = Clock::now();
        it.rec->disp_convert_ms = ms_of(t1 - t0);
        it.rec->disp_overlay_ms = ms_of(t2 - t1);
        ++presented_;
    }

    void loop() {
        const auto timeout = std::chrono::milliseconds(a_.display_timeout_ms);
        std::unique_ptr<Item> cur;
        std::optional<Clock::time_point> wait_since;
        for (;;) {
            pollfd fds[2] = {{wl_display_get_fd(win_.display()), POLLIN, 0}, {wake_fd_, POLLIN, 0}};
            if (::poll(fds, 2, 100) < 0 && errno != EINTR) throw std::runtime_error("poll failed");
            if (fds[1].revents & POLLIN) {
                std::uint64_t v = 0;
                [[maybe_unused]] const ssize_t r = ::read(wake_fd_, &v, sizeof(v));
            }
            if (!win_.poll(0)) {
                if (win_.should_close()) window_closed_ = true;
                throw std::runtime_error("Wayland connection: " + win_.error());
            }
            if (win_.quit_requested() && !stop_requested()) {
                g_stop_by_key.store(true, std::memory_order_relaxed);
                g_stop_signal.store(SIGINT, std::memory_order_relaxed);
                g_stop.store(true, std::memory_order_relaxed);
            }
            if (win_.should_close() && !stop_requested()) {   // xdg_toplevel.close: a normal end, not an interrupt
                window_closed_ = true;
                g_stop.store(true, std::memory_order_relaxed);
            }
            if (stop_requested()) return;

            int nw = 0, nh = 0;
            if (win_.take_pending_size(&nw, &nh) && nw > 0 && nh > 0 && (nw != w_ || nh != h_)) create_buffers(nw, nh);
            if (buffers_.empty()) {
                const int iw = win_.width() > 0 ? win_.width() : a_.disp_w;
                const int ih = win_.height() > 0 ? win_.height() : a_.disp_h;
                create_buffers(iw, ih);
            }

            bool closed = false;
            {
                std::lock_guard<std::mutex> lk(m_);
                closed = closed_;
                if (pending_) {
                    if (cur) ++dropped_;   // cur never reached a buffer and a newer result is here
                    cur = std::move(pending_);
                }
            }
            if (cur) {
                Buffer* b = frame_cb_ ? nullptr : free_buffer();
                if (b) {
                    draw_and_commit(*cur, *b);
                    cur.reset();
                    wait_since.reset();
                } else {
                    if (!wait_since) wait_since = Clock::now();
                    if (Clock::now() - *wait_since > timeout)
                        throw std::runtime_error("the compositor sent no frame callback / buffer release for " +
                                                 std::to_string(a_.display_timeout_ms) +
                                                 " ms (output off or compositor stalled?)");
                }
            } else {
                wait_since.reset();
                if (closed) return;
            }
        }
    }

    const Args& a_;
    yvo::display::WaylandWindow win_;
    wl_registry* registry_ = nullptr;
    wl_shm* shm_ = nullptr;
    bool xbgr_ok_ = false;
    std::vector<std::unique_ptr<Output>> outputs_;
    std::vector<Buffer> buffers_;
    std::uint8_t* map_ = nullptr;
    std::size_t map_size_ = 0;
    int w_ = 0, h_ = 0;
    wl_callback* frame_cb_ = nullptr;
    std::unique_ptr<yvo::tool::SwsResize> sws_;
    int wake_fd_ = -1;
    std::mutex m_;
    std::unique_ptr<Item> pending_;
    bool closed_ = false;
    long dropped_ = 0;     // written under m_; read after the thread is joined
    long presented_ = 0;
    bool window_closed_ = false;
    std::string error_;
};

// ── Output files ───────────────────────────────────────────────────────────────────────────────
std::string cell(double v) {
    char buf[32];
    if (!std::isfinite(v)) return "";
    std::snprintf(buf, sizeof(buf), "%.3f", v);
    return buf;
}

bool write_outputs(const Args& a, const std::vector<std::unique_ptr<FrameRec>>& recs, Clock::time_point t0) {
    bool ok = true;
    const auto rel = [t0](Clock::time_point tp) {
        return tp == Clock::time_point{} ? yvo::tool::FrameLogRow::kNone : ms_of(tp - t0);
    };
    const auto finish = [&ok](std::FILE* fp, const std::string& path) {
        const bool err = std::ferror(fp) != 0;
        if (std::fclose(fp) != 0 || err) {
            std::fprintf(stderr, "[error] writing %s failed\n", path.c_str());
            ok = false;
        }
    };
    const auto open_out = [&ok](const std::string& path) {
        std::FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp) {
            std::fprintf(stderr, "[error] cannot open %s: %s\n", path.c_str(), std::strerror(errno));
            ok = false;
        }
        return fp;
    };
    if (!a.frame_log.empty()) {
        if (std::FILE* fp = open_out(a.frame_log)) {
            std::fputs(yvo::tool::format_frame_log_header(a.display, a.display).c_str(), fp);
            for (const auto& r : recs)
                std::fputs(yvo::tool::format_frame_log_row(
                               {.src = r->src, .seq = r->seq, .dropped = r->seq >= 0 ? 0 : r->dropped ? 1 : 2,
                                .arrival_ms = rel(r->arrival), .decode_end_ms = rel(r->decode_end),
                                .submit_ms = rel(r->submit), .npu_done_ms = rel(r->npu_done),
                                .complete_ms = rel(r->complete), .present_ms = rel(r->present)},
                               a.display, a.display)
                               .c_str(),
                           fp);
            finish(fp, a.frame_log);
        }
    }
    if (!a.stage_log.empty()) {
        if (std::FILE* fp = open_out(a.stage_log)) {
            std::fputs("src,decode_ms,swscale_ms,npu_ms,post_ms,disp_convert_ms,disp_overlay_ms\n", fp);
            for (const auto& r : recs)
                std::fprintf(fp, "%" PRId64 ",%s,%s,%s,%s,%s,%s\n", r->src, cell(r->decode_ms).c_str(), cell(r->swscale_ms).c_str(),
                             cell(r->npu_ms).c_str(), cell(r->post_ms).c_str(), cell(r->disp_convert_ms).c_str(),
                             cell(r->disp_overlay_ms).c_str());
            finish(fp, a.stage_log);
        }
    }
    if (!a.dump_dir.empty()) {
        std::filesystem::create_directories(a.dump_dir);
        const std::string p = a.dump_dir + "/detections.jsonl";
        if (std::FILE* fp = open_out(p)) {
            for (const auto& r : recs) {   // the line format of yvo_pipeline's write_jsonl_line (pipeline/main.cpp)
                if (r->seq < 0 || r->complete == Clock::time_point{}) continue;
                std::fprintf(fp, "{\"frame\":%" PRId64 ",\"detections\":[", r->src);
                for (std::size_t i = 0; i < r->dets.size(); ++i) {
                    const Detection& d = r->dets[i];
                    std::fprintf(fp, "%s{\"class\":%d,\"name\":\"%s\",\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}", i ? "," : "",
                                 d.class_id, yvo::class_name(d.class_id), static_cast<double>(d.score), static_cast<double>(d.x1),
                                 static_cast<double>(d.y1), static_cast<double>(d.x2), static_cast<double>(d.y2));
                }
                std::fputs("]}\n", fp);
            }
            finish(fp, p);
        }
    }
    return ok;
}

// ── On-CPU time per thread ─────────────────────────────────────────────────────────────────────
// With frame threads the decode call only collects a finished frame (about 0.1 ms): the decoding itself runs on
// libavcodec's threads and shows up nowhere else. Likewise HailoRT's transpose and NMS. So the run is also split by
// thread name, from /proc/self/task/<tid>/schedstat (first field: ns on CPU). Read only right before t0 and after the run.
using ThreadCpu = std::map<std::string, double>;   // tid -> ms on CPU

ThreadCpu thread_cpu_snapshot() {
    ThreadCpu out;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        std::ifstream sched(e.path() / "schedstat");
        double ns = 0.0;
        if (sched >> ns) out[e.path().filename().string()] = ns / 1e6;
    }
    return out;
}

double this_thread_cpu_ms() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) / 1e6;
}

// Thread name without trailing digits -> (ms on CPU since `before`, threads). Call while the threads still exist.
using ThreadGroups = std::map<std::string, std::pair<double, int>>;
ThreadGroups thread_cpu_groups(const ThreadCpu& before) {
    ThreadGroups groups;
    for (const auto& [tid, ms] : thread_cpu_snapshot()) {
        std::ifstream comm("/proc/self/task/" + tid + "/comm");
        std::string name;
        std::getline(comm, name);
        while (!name.empty() && std::isdigit(static_cast<unsigned char>(name.back())) != 0) name.pop_back();
        const auto it = before.find(tid);
        auto& g = groups[name.empty() ? "?" : name];
        g.first += ms - (it == before.end() ? 0.0 : it->second);
        ++g.second;
    }
    return groups;
}

// ── Run ────────────────────────────────────────────────────────────────────────────────────────
std::vector<double> collect(const std::vector<std::unique_ptr<FrameRec>>& recs, double FrameRec::*field) {
    std::vector<double> v;
    v.reserve(recs.size());
    for (const auto& r : recs)
        if (std::isfinite((*r).*field)) v.push_back((*r).*field);
    return v;
}

void print_line(const char* label, const Stats& s, const char* note = "") {
    std::printf("%-15s mean %.3f  median %.3f  p90 %.3f  max %.3f%s\n", label, s.mean, s.median, s.p90, s.max, note);
}

int run(const Args& a) {
    av_log_set_level(AV_LOG_ERROR);   // swscale warns "No accelerated colorspace conversion" per slice context, then uses NEON
    SwHevcDecoder dec(a.video, a.dec_threads);
    const double src_fps = dec.fps();

    // The first frame fixes the source size, the letterbox and the runner config. It is decoded before t0 and kept: on
    // the schedule it is frame 0, whose decode time is therefore not counted (frame threads make that call the longest).
    FramePtr first = dec.next();
    if (!first) throw std::runtime_error("the video has no frames");
    const int src_w = first->width, src_h = first->height;
    const yvo::LetterboxParams lb = yvo::make_letterbox(src_w, src_h);
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f), new_h = static_cast<int>(src_h * lb.scale + 0.5f);

    yvo::HailoRunnerConfig cfg;
    cfg.hef_path = a.hef;
    cfg.input_layout = yvo::TensorLayout::Nhwc;
    cfg.batch_size = 1;
    cfg.num_slots = 1;
    cfg.src_width = src_w;
    cfg.src_height = src_h;
    auto runner = std::make_unique<yvo::HailoRunner>(cfg);
    if (runner->input_width() != lb.dst_w || runner->input_height() != lb.dst_h)
        throw std::runtime_error("the HEF input size differs from the 640x640 letterbox");
    std::printf("[npu] HailoRunner host pointer NHWC, batch 1, slots 1, HailoRT NMS, input %zu B, output %zu B\n",
                runner->input_frame_size(), runner->output_frame_size());

    // Input tensor: one page-aligned buffer, padded once; swscale overwrites the content rectangle only.
    const std::size_t in_size = runner->input_frame_size();
    std::unique_ptr<std::uint8_t, decltype([](std::uint8_t* p) { std::free(p); })> in_buf(
        static_cast<std::uint8_t*>(std::aligned_alloc(4096, (in_size + 4095) / 4096 * 4096)));
    if (!in_buf) throw std::bad_alloc();
    std::memset(in_buf.get(), lb.pad_value, in_size);
    yvo::tool::SwsResize sws(src_w, src_h, new_w, new_h, AV_PIX_FMT_RGB24, a.sws_flags, a.sws_threads);
    std::printf("[preproc] libswscale %dx%d yuv420p -> %dx%d rgb24 at (%d,%d) in 640x640 (pad %d), sws_flags=%s threads=%d, libswscale " AV_STRINGIFY(LIBSWSCALE_VERSION) "\n",
                src_w, src_h, new_w, new_h, lb.pad_x, lb.pad_y, static_cast<int>(lb.pad_value), sws.flags().c_str(), sws.threads());

    std::unique_ptr<ShmDisplay> display;
    if (a.display) display = std::make_unique<ShmDisplay>(a);
    std::printf("[gpu] not used: this process creates no Vulkan instance and opens no DRM device\n");

    const float hailo_t0 = runner->chip_temperature();
    const double soc_t0 = soc_temperature();
    std::fflush(stdout);

    // Shared with the decode thread.
    std::mutex m;
    std::condition_variable cv;
    yvo::pipeline::LeakyQueue<QueuedFrame> in_q(static_cast<std::size_t>(a.input_queue));
    bool ingest_done = false;
    long arrived = 0, dropped = 0, queue_depth_max = 0;   // written under m
    std::string decode_error;
    std::vector<std::unique_ptr<FrameRec>> recs;          // decode thread only until it is joined
    std::vector<double> decode_late_ms;                   // decode thread only until it is joined
    if (a.frames > 0) recs.reserve(static_cast<std::size_t>(a.frames));

    // Completion handed from the HailoRT callback thread to the main thread.
    std::mutex done_m;
    std::condition_variable done_cv;
    std::optional<yvo::InferResult> done;

    long n = 0, completed = 0, errors = 0, total_dets = 0;
    std::vector<double> e2e_ms, e2e_arr_ms, queue_wait_ms;
    std::vector<std::int64_t> seq_src;
    bool fail = false;

    const ThreadCpu thr0 = thread_cpu_snapshot();
    const double c0 = cpu_ms(); yvo::tool::perf_window(true);  // same order as yvo_pipeline: CPU window, marker, t0
    const auto t0 = Clock::now();
    yvo::pipeline::ArrivalSchedule sched(src_fps);
    sched.start(t0);

    double decode_thread_cpu = 0.0, display_thread_cpu = 0.0;   // read after join
    std::thread decode_thread([&] {
        const auto finish = [&] {
            decode_thread_cpu = this_thread_cpu_ms();
            {
                std::lock_guard<std::mutex> lk(m);
                ingest_done = true;
            }
            cv.notify_all();
        };
        try {
            for (long k = 0; !stop_requested() && (a.frames == 0 || k < a.frames); ++k) {
                const auto t_arr = sched.arrival(k);
                std::this_thread::sleep_until(t_arr);
                if (stop_requested()) break;
                const auto d0 = Clock::now();
                FramePtr f = first ? std::move(first) : dec.next();
                const auto d1 = Clock::now();
                if (!f) break;
                auto rec = std::make_unique<FrameRec>();
                rec->src = k;
                rec->arrival = t_arr;
                rec->decode_end = d1;
                rec->decode_ms = ms_of(d1 - d0);
                decode_late_ms.push_back(ms_of(d1 - t_arr));
                QueuedFrame qf{std::move(f), rec.get()};
                recs.push_back(std::move(rec));
                std::optional<QueuedFrame> old;
                {
                    std::lock_guard<std::mutex> lk(m);
                    old = in_q.push(std::move(qf));
                    if (old) {
                        old->rec->dropped = true;   // the record's owner is the queue until here
                        ++dropped;
                    }
                    ++arrived;
                    queue_depth_max = std::max(queue_depth_max, static_cast<long>(in_q.size()));
                }
                cv.notify_all();
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(m);
            decode_error = e.what();
        }
        finish();
    });

    std::thread display_thread;
    if (display)
        display_thread = std::thread([&] {
            display->run();
            display_thread_cpu = this_thread_cpu_ms();
        });

    try {
        for (;;) {
            QueuedFrame qf;
            {
                std::unique_lock<std::mutex> lk(m);
                // Timed: a signal handler cannot notify, and the display thread can set the stop flag too.
                while (in_q.empty() && !ingest_done && !stop_requested()) cv.wait_for(lk, std::chrono::milliseconds(50));
                if (stop_requested()) break;
                auto got = in_q.pop_oldest();
                if (!got) break;   // ingest done and the queue is empty
                qf = std::move(*got);
            }
            FrameRec& rec = *qf.rec;
            const auto w1 = Clock::now();
            queue_wait_ms.push_back(ms_of(w1 - rec.decode_end));
            rec.seq = n;
            seq_src.push_back(rec.src);

            const std::uint8_t* const src[3] = {qf.f->data[0], qf.f->data[1], qf.f->data[2]};
            const int src_stride[3] = {qf.f->linesize[0], qf.f->linesize[1], qf.f->linesize[2]};
            std::uint8_t* const dst[3] = {in_buf.get() + static_cast<std::size_t>(lb.pad_y) * static_cast<std::size_t>(lb.dst_w) * 3 +
                                              static_cast<std::size_t>(lb.pad_x) * 3,
                                          nullptr, nullptr};
            const int dst_stride[3] = {lb.dst_w * 3, 0, 0};
            sws.scale(src, src_stride, dst, dst_stride);
            rec.submit = Clock::now();
            rec.swscale_ms = ms_of(rec.submit - w1);

            yvo::PreprocOutput in;
            in.host_ptr = in_buf.get();
            in.size = in_size;
            in.layout = yvo::TensorLayout::Nhwc;
            in.lb = lb;
            in.pts = n;
            in.slot = 0;
            runner->submit(in, [&](yvo::InferResult&& r) {
                {
                    std::lock_guard<std::mutex> lk(done_m);
                    done.emplace(std::move(r));
                }
                done_cv.notify_one();
            });
            ++n;
            yvo::InferResult r;
            {
                std::unique_lock<std::mutex> lk(done_m);
                if (!done_cv.wait_for(lk, std::chrono::seconds(5), [&] { return done.has_value(); }))
                    throw std::runtime_error("NPU completion did not arrive within 5 s");
                r = std::move(*done);
                done.reset();
            }
            const auto t_got = Clock::now();
            ++completed;
            if (r.status != HAILO_SUCCESS) {
                ++errors;
                std::fprintf(stderr, "[error] frame %" PRId64 ": HailoRT status %d\n", rec.src, static_cast<int>(r.status));
            }
            rec.npu_done = r.t_npu_done;
            rec.npu_ms = r.latency_ms;
            rec.post_ms = ms_of(t_got - r.t_npu_done);
            rec.dets = std::move(r.detections);
            rec.complete = t_got;
            total_dets += static_cast<long>(rec.dets.size());
            e2e_ms.push_back(ms_of(r.t_npu_done - rec.decode_end));
            e2e_arr_ms.push_back(ms_of(r.t_npu_done - rec.arrival));
            const double fps_now = static_cast<double>(n) / std::max(std::chrono::duration<double>(t_got - t0).count(), 1e-9);
            if (display) display->post({std::move(qf.f), &rec, fps_now});
            if (!a.quiet && n % 300 == 0) {
                std::printf("[progress] %ld frames, %.1f FPS\n", n, fps_now);
                std::fflush(stdout);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        fail = true;
        g_stop.store(true, std::memory_order_relaxed);
    }

    if (display) {
        display->close_input();
        display_thread.join();
    }
    decode_thread.join();
    if (!decode_error.empty()) {
        std::fprintf(stderr, "[error] decode: %s\n", decode_error.c_str());
        fail = true;
    }
    in_q.clear();   // abandoned frames: released here, counted below

    int stop_signal = 0;
    if (stop_requested()) stop_signal = g_stop_signal.load(std::memory_order_relaxed);
    const bool interrupted = stop_signal != 0;
    if (interrupted) fail = true;

    const auto t1 = Clock::now();
    yvo::tool::perf_window(false); const double c1 = cpu_ms();
    ThreadGroups thr_groups = thread_cpu_groups(thr0);   // the worker threads of the libraries are still alive here
    thr_groups["(decode thread)"] = {decode_thread_cpu, 1};
    if (display) thr_groups["(display thread)"] = {display_thread_cpu, 1};

    if (display && !display->error().empty()) {
        std::fprintf(stderr, "[error] display: %s\n", display->error().c_str());
        fail = true;
    }
    const float hailo_t1 = runner->chip_temperature();
    runner.reset();   // bounded drain and NPU shutdown, also after a failure
    const double soc_t1 = soc_temperature();

    const long abandoned = arrived - n - dropped;
    if (completed != n || errors) fail = true;
    if (abandoned < 0) {
        std::fprintf(stderr, "[error] frame count mismatch: arrived=%ld processed=%ld dropped=%ld\n", arrived, n, dropped);
        fail = true;
    }
    if (!write_outputs(a, recs, t0)) fail = true;

    const long presented = display ? display->presented() : 0;
    const long disp_dropped = display ? display->dropped() : 0;
    const bool window_closed = display && display->window_closed();
    display.reset();   // closes the window

    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    const double fps = static_cast<double>(completed) / wall_s;
    const double cpu_total = c1 - c0;
    const double cpu_per_frame = completed ? cpu_total / static_cast<double>(completed) : 0.0;
    const Stats sd = stats(collect(recs, &FrameRec::decode_ms)), ss = stats(collect(recs, &FrameRec::swscale_ms));
    const Stats sn = stats(collect(recs, &FrameRec::npu_ms)), sp = stats(collect(recs, &FrameRec::post_ms));
    const Stats sc = stats(collect(recs, &FrameRec::disp_convert_ms)), so = stats(collect(recs, &FrameRec::disp_overlay_ms));
    const Stats se = stats(e2e_ms), sa = stats(e2e_arr_ms), sq = stats(queue_wait_ms), sl = stats(decode_late_ms);
    const long max_drop_run = yvo::pipeline::max_gap(seq_src);
    const auto t0_mono = std::chrono::duration<double, std::milli>(t0.time_since_epoch()).count();

    if (interrupted)
        std::printf("[interrupted] %s — stopped at frame %ld (the stats below are partial-run values, ok=0)\n",
                    g_stop_by_key.load() ? "display window key press (Esc/q/Ctrl+C)"
                    : stop_signal == SIGTERM ? "SIGTERM received"
                                             : "SIGINT received",
                    n);
    if (window_closed && !interrupted) std::printf("[stopped] the compositor closed the window at frame %ld\n", n);
    std::printf("== yvo_cpu_baseline (software HEVC, libswscale, ptr, NHWC, batch 1, slots 1, postproc host) ==\n");
    std::printf("frames          %ld (decoded %ld, errors %ld)\n", completed, n, errors);
    std::printf("wall            %.3f s, %.2f FPS\n", wall_s, fps);
    std::printf("cpu             %.3f ms/frame (process user+sys %.1f ms, %.0f%% of one core)\n", cpu_per_frame, cpu_total,
                cpu_total / (wall_s * 10.0));
    print_line("decode call ms", sd);
    print_line("swscale ms", ss, "  (640x640 letterbox, main thread)");
    print_line("npu latency ms", sn, "  (submit -> completion callback: HailoRT transpose + NPU + HailoRT NMS)");
    print_line("post ms", sp, "  (callback entered -> main thread has the parsed detections)");
    print_line("e2e latency ms", se, "  (decode return → NPU completion callback)");
    std::printf("input pace      source @ %.3f fps, input queue %d (leaky: full -> drop the oldest)\n", src_fps, a.input_queue);
    std::printf("arrived         %ld (processed %ld, dropped %ld = %.2f%%, abandoned %ld, longest drop run %ld, queue depth max %ld)\n",
                arrived, n, dropped, arrived ? 100.0 * static_cast<double>(dropped) / static_cast<double>(arrived) : 0.0,
                abandoned, max_drop_run, queue_depth_max);
    std::printf("e2e arrival ms  mean %.3f  median %.3f  p90 %.3f  max %.3f  (source arrival → NPU completion callback)\n", sa.mean,
                sa.median, sa.p90, sa.max);
    std::printf("queue wait ms   mean %.3f  p90 %.3f  max %.3f  (decode return → swscale start)\n", sq.mean, sq.p90, sq.max);
    std::printf("decode late ms  mean %.3f  p90 %.3f  max %.3f  (source arrival → decode return)\n", sl.mean, sl.p90, sl.max);
    if (a.display) {
        print_line("disp convert ms", sc, "  (libswscale -> RGBA in the shm buffer)");
        print_line("disp overlay ms", so, "  (ref/overlay_reference.hpp)");
        std::printf("display         presented %ld, display dropped %ld\n", presented, disp_dropped);
    }
    std::printf("detections      %ld total\n", total_dets);
    for (const auto& [name, g] : thr_groups)
        std::printf("thread cpu      %-20s threads %2d  %9.1f ms  %7.3f ms/arrived  %7.3f ms/processed\n", name.c_str(), g.second,
                    g.first, arrived ? g.first / static_cast<double>(arrived) : 0.0,
                    n ? g.first / static_cast<double>(n) : 0.0);
    std::printf("t0_monotonic_ms %.3f  (time base of --frame-log on CLOCK_MONOTONIC)\n", t0_mono);
    std::printf("hailo chip temp %.1f -> %.1f C\n", static_cast<double>(hailo_t0), static_cast<double>(hailo_t1));
    std::printf("soc temp        %.1f -> %.1f C\n", soc_t0, soc_t1);
    if (!a.csv_label.empty()) {
        yvo::tool::print_result_csv({.label = a.csv_label, .mode = "ptr", .layout = "nhwc", .batch = 1, .slots = 1,
                                     .frames = completed, .wall_s = wall_s, .fps = fps, .cpu_ms_per_frame = cpu_per_frame,
                                     .decode_mean = sd.mean, .gpu_mean = kNan, .gpu_p90 = kNan, .npu_mean = sn.mean,
                                     .npu_p90 = sn.p90, .e2e_mean = se.mean, .e2e_p90 = se.p90, .detections = total_dets,
                                     .hailo_t0 = static_cast<double>(hailo_t0), .hailo_t1 = static_cast<double>(hailo_t1),
                                     .soc_t0 = soc_t0, .soc_t1 = soc_t1, .ok = !fail});
        std::fputs(yvo::tool::format_result_csv4(
                       {.label = a.csv_label, .input_queue = a.input_queue, .src_fps = src_fps, .arrived = arrived,
                        .processed = n, .dropped = dropped, .abandoned = abandoned,
                        .drop_rate = arrived ? static_cast<double>(dropped) / static_cast<double>(arrived) : 0.0,
                        .max_drop_run = max_drop_run, .queue_depth_max = queue_depth_max, .e2e_arr_mean = sa.mean,
                        .e2e_arr_median = sa.median, .e2e_arr_p90 = sa.p90, .e2e_arr_max = sa.max,
                        .queue_wait_mean = sq.mean, .queue_wait_p90 = sq.p90, .queue_wait_max = sq.max,
                        .decode_late_mean = sl.mean, .decode_late_p90 = sl.p90, .decode_late_max = sl.max,
                        .held_frames_max = 0, .ok = !fail})
                       .c_str(),
                   stdout);
    }
    std::fflush(stdout);
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
    install_stop_handlers();
    try {
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
