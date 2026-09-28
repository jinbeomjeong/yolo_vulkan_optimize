// yvo_gst_pipeline: GStreamer research path (B plan) — the yvo_pipeline comparison tool
//   filesrc ! qtdemux ! h265parse ! v4l2slh265dec ! queue ! yvosandletterbox ! queue ! <sink>
//   The elements are registered statically (yvo::gst::register_elements, no plugin .so).
//
// Sinks
//   hailo : yvohailosink (stage 2b, default). Same inference + postprocessing as `yvo_pipeline --mode dmabuf --layout
//           nhcw` (HailoRunner, InferModel batch --batch, HEF NMS parsed on the host). --dump-dir writes detections.jsonl
//           in the yvo_pipeline format; --csv-label also prints RESULT_CSV with yvo_pipeline's 22 columns.
//   fake  : fakesink sync=false (throughput of decode + Vulkan letterbox)
//   hold  : appsink whose callback keeps the newest --hold N samples before releasing the oldest.
//           Exercises the ring-slot backpressure: with N >= --slots every slot is held, the element blocks in
//           prepare_output_buffer and the watchdog must fire.
// Timing: the pipeline is prerolled in PAUSED first (Vulkan + HailoRT init, first buffers), then wall/CPU are measured
//   from PLAYING to the EOS message (the hailo sink posts EOS only after its last NPU completion).
// Watchdog: no buffer at the sink pad for --stall-timeout seconds → STALL diagnostic (queue levels, slots in use),
//   exit 4. A starved decoder pool also stalls silently, so this is the only detection.
// Exit codes: 0 pass, 1 runtime error, 2 bad arguments, 3 checks failed, 4 stall.
// Nothing is written to disk except --dump-dir (and hailort.log, which HailoRT writes to the CWD; the tool never
// changes the CWD). The summary goes to stdout; RESULT_GST_CSV (+ RESULT_CSV for the hailo sink) with --csv-label.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <time.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#ifdef YVO_GSTELEM_HAVE_HAILO
#include "gstelem/hailo_infer_sink.hpp"
#endif
#include "gstelem/register.hpp"
#include "gstelem/tensor_meta.hpp"
#include "toolkit/stats.hpp"
#include "toolkit/tool_util.hpp"

namespace {

enum class SinkKind { Fake, Hold, Hailo };

struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string hef = YVO_DEFAULT_HEF;
    int frames = 0;           // 0 = whole file
    int expect_frames = 1816; // used when --frames is 0; 0 = no check
    int slots = 0;            // 0 = default: hailo max(4, 2*batch) (= yvo_pipeline default), fake/hold 16
    int batch = 1;            // hailo sink: InferModel batch size (= yvo_pipeline --batch); ignored by fake/hold
    double score = -1.0;      // hailo sink: < 0 keeps the HEF value (0.2), = yvo_pipeline --score default
    std::string dump_dir;     // hailo sink: detections.jsonl
    int queue = 2;            // decoder-side queue max-size-buffers
    int decoder_pool_min = 4;
    SinkKind sink = SinkKind::Hailo;
    int hold = 2;
    int stall_timeout_s = 5;
    bool validation = false;
    bool quiet = false;
    std::string csv_label;
};

void usage(const char* argv0) {
    std::printf(
        "usage: %s [--video PATH] [--hef PATH] [--frames N] [--expect-frames N] [--slots N] [--batch N] [--score T]\n"
        "          [--dump-dir DIR] [--queue N] [--decoder-pool-min N] [--sink hailo|fake|hold] [--hold N]\n"
        "          [--stall-timeout S] [--validation] [--quiet] [--csv-label L]\n"
        "  GStreamer HW decode (v4l2slh265dec, DMA_DRM NV12 SAND128) -> yvosandletterbox (Vulkan, NHCW 640x640) -> sink\n"
        "  --video          input (default %s)\n"
        "  --hef            HEF for the hailo sink (default %s)\n"
        "  --frames         stop after N frames at the sink (default 0 = whole file)\n"
        "  --expect-frames  expected frame count when --frames is 0 (default 1816, 0 = no check)\n"
        "  --slots          output ring slots of yvosandletterbox (2..32; default hailo max(4, 2*batch) = the\n"
        "                   yvo_pipeline --slots default, fake/hold 16). The hailo sink returns each slot in its\n"
        "                   completion callback, so up to <slots> frames are in flight on the NPU as in yvo_pipeline:\n"
        "                   the same --slots gives the same NPU queue depth (latency); the ring blocks, never stalls\n"
        "  --batch          hailo sink: InferModel batch size 1..16 (default 1, = yvo_pipeline --batch)\n"
        "  --score          hailo sink: NMS score threshold (default -1 = HEF value 0.2, = yvo_pipeline --score)\n"
        "  --dump-dir       hailo sink: write DIR/detections.jsonl (yvo_pipeline --dump-dir line format, no PNGs)\n"
        "  --queue          max-size-buffers of the queue between decoder and element (default 2)\n"
        "  --decoder-pool-min  min buffers proposed to the decoder pool (default 4)\n"
        "  --sink           hailo = yvohailosink (default), fake = fakesink sync=false,\n"
        "                   hold = appsink keeping the newest --hold N samples\n"
        "  --hold           samples kept by --sink hold (default 2). N >= --slots must stall (backpressure check)\n"
        "  --stall-timeout  seconds without a buffer at the sink before the watchdog fires (default 5)\n"
        "  --validation     enable VK_LAYER_KHRONOS_validation in the element (VK_LOADER_LAYERS_ENABLE also works)\n"
        "  --csv-label      print RESULT_GST_CSV,<label>,... at the end (hailo sink: also RESULT_CSV in the\n"
        "                   yvo_pipeline 22-column format; decode_ms_mean = nan, see the summary note)\n"
        "Exit codes: 0 pass, 1 runtime error, 2 bad arguments, 3 checks failed, 4 stall.\n",
        argv0, YVO_DEFAULT_VIDEO, YVO_DEFAULT_HEF);
}

// 0 = ok, 1 = --help, 2 = error
int parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        auto next_int = [&](int& out, int lo, int hi) {
            const char* v = next();
            if (!v) return false;
            char* end = nullptr;
            const long x = std::strtol(v, &end, 10);
            if (*end != '\0' || x < lo || x > hi) return false;
            out = static_cast<int>(x);
            return true;
        };
        bool ok = true;
        if (s == "--help" || s == "-h") return 1;
        if (s == "--video") {
            const char* v = next();
            ok = v != nullptr;
            if (ok) a.video = v;
        } else if (s == "--hef") {
            const char* v = next();
            ok = v != nullptr;
            if (ok) a.hef = v;
        } else if (s == "--dump-dir") {
            const char* v = next();
            ok = v != nullptr && *v;
            if (ok) a.dump_dir = v;
        } else if (s == "--score") {
            const char* v = next();
            char* end = nullptr;
            ok = v != nullptr;
            if (ok) {
                a.score = std::strtod(v, &end);
                ok = *end == '\0' && a.score <= 1.0;
            }
        } else if (s == "--frames") ok = next_int(a.frames, 0, 1 << 30);
        else if (s == "--expect-frames") ok = next_int(a.expect_frames, 0, 1 << 30);
        else if (s == "--slots") ok = next_int(a.slots, 2, 32);
        else if (s == "--batch") ok = next_int(a.batch, 1, 16);
        else if (s == "--queue") ok = next_int(a.queue, 1, 64);
        else if (s == "--decoder-pool-min") ok = next_int(a.decoder_pool_min, 1, 32);
        else if (s == "--hold") ok = next_int(a.hold, 0, 64);
        else if (s == "--stall-timeout") ok = next_int(a.stall_timeout_s, 1, 3600);
        else if (s == "--validation") a.validation = true;
        else if (s == "--quiet") a.quiet = true;
        else if (s == "--csv-label") {
            const char* v = next();
            ok = v != nullptr;
            if (ok) a.csv_label = v;
        } else if (s == "--sink") {
            const char* v = next();
            const std::string k = v ? v : "";
            if (k == "fake") a.sink = SinkKind::Fake;
            else if (k == "hold") a.sink = SinkKind::Hold;
            else if (k == "hailo") a.sink = SinkKind::Hailo;
            else ok = false;
        } else {
            ok = false;
        }
        if (!ok) {
            std::fprintf(stderr, "bad argument: %s\n", s.c_str());
            return 2;
        }
    }
    if (a.slots == 0) a.slots = a.sink == SinkKind::Hailo ? std::max(4, 2 * a.batch) : 16;
    return 0;
}

const char* sink_name(SinkKind k) {
    switch (k) {
    case SinkKind::Fake: return "fake";
    case SinkKind::Hold: return "hold";
    case SinkKind::Hailo: return "hailo";
    }
    return "?";
}

// Sink part of the launch description. Every sink is named "sink" (the watchdog probe sits on its sink pad).
std::string sink_description(const Args& a) {
    switch (a.sink) {
    case SinkKind::Fake: return "fakesink name=sink sync=false enable-last-sample=false";
    case SinkKind::Hold:
        return "appsink name=sink sync=false emit-signals=false max-buffers=2 drop=false enable-last-sample=false "
               "wait-on-eos=false";
    case SinkKind::Hailo:
        return yvo::tool::fmt("yvohailosink name=sink hef=\"%s\" batch=%d score-threshold=%g", a.hef.c_str(), a.batch,
                              a.score) +
               (a.dump_dir.empty() ? std::string() : " dump-dir=\"" + a.dump_dir + "\"");
    }
    return "";
}

std::uint64_t mono_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<std::uint64_t>(ts.tv_nsec);
}

std::string shell_line(const char* cmd) {
    FILE* p = popen(cmd, "r");
    if (!p) return "?";
    char buf[256] = {};
    const bool got = std::fgets(buf, sizeof(buf), p) != nullptr;
    pclose(p);
    std::string s = got ? buf : "?";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

// Collected on the sink pad (streaming thread of the output queue); read by main after teardown.
struct SinkStats {
    std::mutex m;
    std::atomic<std::uint64_t> last_ns{0};   // CLOCK_MONOTONIC of the last buffer (watchdog)
    std::atomic<int> frames{0};
    std::atomic<bool> limit_reached{false};
    int limit = 0;                           // --frames (0 = none)
    std::uint64_t first_ns = 0;
    std::vector<double> gpu_ms, lat_ms, wait_ms;
    std::uint64_t next_seq = 0;
    int seq_gaps = 0, missing_meta = 0, fd_mismatch = 0, lb_mismatch = 0, bad_slot = 0;
    std::map<int, std::set<int>> slot_fds;   // slot → fds seen (stable = exactly one)
    std::map<int, int> slot_count;
    yvo::LetterboxParams lb_first;
    bool have_lb = false;
    int src_w = 0, src_h = 0;
    std::uint64_t dmabuf_bytes = 0;
    std::uint32_t tensor_bytes = 0;
};

GstPadProbeReturn sink_probe(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user) {
    auto* st = static_cast<SinkStats*>(user);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    const std::uint64_t now = mono_ns();
    st->last_ns.store(now);
    if (st->limit > 0 && st->frames.load() >= st->limit) {
        st->limit_reached.store(true);
        return GST_PAD_PROBE_DROP;  // in-flight frames after the limit (EOS is on its way)
    }
    yvo::gst::TensorFrameInfo ti;
    const bool has = yvo::gst::tensor_meta_get(buf, ti);
    {
        std::lock_guard<std::mutex> lk(st->m);
        if (st->frames.load() == 0) st->first_ns = now;
        if (!has) {
            ++st->missing_meta;
        } else {
            if (ti.seq != st->next_seq) ++st->seq_gaps;
            st->next_seq = ti.seq + 1;
            if (ti.slot < 0 || ti.slot >= ti.num_slots || yvo::gst::tensor_buffer_slot(buf) != ti.slot) ++st->bad_slot;
            if (yvo::gst::tensor_buffer_fd(buf) != ti.dmabuf_fd) ++st->fd_mismatch;
            st->slot_fds[ti.slot].insert(ti.dmabuf_fd);
            ++st->slot_count[ti.slot];
            if (ti.gpu_ms >= 0) st->gpu_ms.push_back(ti.gpu_ms);
            st->lat_ms.push_back(static_cast<double>(now - ti.arrival_mono_ns) * 1e-6);
            st->wait_ms.push_back(static_cast<double>(ti.acquire_wait_ns) * 1e-6);
            if (!st->have_lb) {
                st->lb_first = ti.lb;
                st->have_lb = true;
                st->src_w = ti.src_width;
                st->src_h = ti.src_height;
                st->dmabuf_bytes = ti.dmabuf_bytes;
                st->tensor_bytes = ti.tensor_bytes;
            } else if (ti.lb.scale != st->lb_first.scale || ti.lb.pad_x != st->lb_first.pad_x ||
                       ti.lb.pad_y != st->lb_first.pad_y) {
                ++st->lb_mismatch;
            }
        }
    }
    st->frames.fetch_add(1);
    if (st->limit > 0 && st->frames.load() >= st->limit) st->limit_reached.store(true);
    return GST_PAD_PROBE_OK;
}

// --sink hold: keep the newest N samples, release older ones. While draining, release immediately.
struct HoldState {
    std::mutex m;
    std::deque<GstSample*> held;
    int n = 0;
    std::atomic<bool> draining{false};
    int max_held = 0;
    void release_all() {
        std::lock_guard<std::mutex> lk(m);
        for (GstSample* s : held) gst_sample_unref(s);
        held.clear();
    }
};

GstFlowReturn hold_new_sample(GstAppSink* sink, gpointer user) {
    auto* h = static_cast<HoldState*>(user);
    GstSample* s = gst_app_sink_pull_sample(sink);
    if (!s) return GST_FLOW_EOS;
    std::lock_guard<std::mutex> lk(h->m); // draining is checked under the lock: release_all() cannot race a push
    if (h->draining.load()) {
        gst_sample_unref(s);
        return GST_FLOW_OK;
    }
    h->held.push_back(s);
    while (static_cast<int>(h->held.size()) > h->n) {
        gst_sample_unref(h->held.front());
        h->held.pop_front();
    }
    h->max_held = std::max(h->max_held, static_cast<int>(h->held.size()));
    return GST_FLOW_OK;
}

guint queue_level(GstElement* q) {
    guint n = 0;
    if (q) g_object_get(q, "current-level-buffers", &n, nullptr);
    return n;
}

std::string pad_caps(GstElement* e, const char* pad) {
    if (!e) return "?";
    GstPad* p = gst_element_get_static_pad(e, pad);
    if (!p) return "?";
    GstCaps* c = gst_pad_get_current_caps(p);
    gst_object_unref(p);
    if (!c) return "(none)";
    gchar* s = gst_caps_to_string(c);
    std::string out = s;
    g_free(s);
    gst_caps_unref(c);
    return out;
}

void print_message(GstMessage* m) {
    GError* err = nullptr;
    gchar* dbg = nullptr;
    const bool is_err = GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR;
    if (is_err) gst_message_parse_error(m, &err, &dbg);
    else gst_message_parse_warning(m, &err, &dbg);
    std::printf("%s from %s: %s\n  debug: %s\n", is_err ? "ERROR" : "WARNING", GST_OBJECT_NAME(GST_MESSAGE_SRC(m)),
                err ? err->message : "?", dbg ? dbg : "");
    if (err) g_error_free(err);
    g_free(dbg);
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    const int pr = parse(argc, argv, a);
    if (pr == 1) {
        usage(argv[0]);
        return 0;
    }
    if (pr != 0) {
        usage(argv[0]);
        return 2;
    }
#ifndef YVO_GSTELEM_HAVE_HAILO
    if (a.sink == SinkKind::Hailo) {
        std::fprintf(stderr, "--sink hailo: yvohailosink was not built (yvo_gstelem without yvo_hailo); use fake or hold.\n");
        return 2;
    }
#endif
    gst_init(&argc, &argv);
    if (!yvo::gst::register_elements()) {
        std::fprintf(stderr, "register_elements failed\n");
        return 1;
    }
    const int expected = a.frames > 0 ? a.frames : a.expect_frames;

    const std::string desc = yvo::tool::fmt(
        "filesrc location=\"%s\" ! qtdemux ! h265parse ! v4l2slh265dec name=dec ! "
        "queue name=q_dec max-size-buffers=%d max-size-bytes=0 max-size-time=0 ! "
        "yvosandletterbox name=lb slots=%d decoder-pool-min=%d validation=%s ! "
        "queue name=q_out max-size-buffers=%d max-size-bytes=0 max-size-time=0 ! %s",
        a.video.c_str(), a.queue, a.slots, a.decoder_pool_min, a.validation ? "true" : "false", a.slots,
        sink_description(a).c_str());

    gchar* gv = gst_version_string();
    std::string versions = gv;
    g_free(gv);
    if (GstPlugin* pl = gst_registry_find_plugin(gst_registry_get(), "v4l2codecs")) {
        versions += yvo::tool::fmt(", v4l2codecs %s (%s)", gst_plugin_get_version(pl), gst_plugin_get_package(pl));
        gst_object_unref(pl);
    } else {
        versions += ", v4l2codecs NOT FOUND";
    }
    versions += ", gstreamer1.0-plugins-bad " +
                shell_line("dpkg-query -W -f='${Version}' gstreamer1.0-plugins-bad 2>/dev/null");
    if (!a.quiet) {
        std::printf("== yvo_gst_pipeline ==\n");
        std::printf("pipeline: %s\n", desc.c_str());
        std::printf("versions: %s\n", versions.c_str());
        std::printf("sink=%s hold=%d slots=%d batch=%d queue=%d decoder-pool-min=%d stall-timeout=%d s expected=%d\n",
                    sink_name(a.sink), a.hold, a.slots, a.batch, a.queue, a.decoder_pool_min, a.stall_timeout_s,
                    expected);
        if (a.sink == SinkKind::Hailo)
            std::printf("hailo: hef=%s batch=%d score=%g dump-dir=%s (ring slots %d = max frames in flight on the NPU)\n",
                        a.hef.c_str(), a.batch, a.score, a.dump_dir.empty() ? "-" : a.dump_dir.c_str(), a.slots);
        std::fflush(stdout);
    }

    GError* perr = nullptr;
    GstElement* pipeline = gst_parse_launch(desc.c_str(), &perr);
    if (!pipeline || perr) {
        std::fprintf(stderr, "gst_parse_launch: %s\n", perr ? perr->message : "?");
        if (perr) g_error_free(perr);
        if (pipeline) gst_object_unref(pipeline);
        return 1;
    }
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    GstElement* lb = gst_bin_get_by_name(GST_BIN(pipeline), "lb");
    GstElement* q_dec = gst_bin_get_by_name(GST_BIN(pipeline), "q_dec");
    GstElement* q_out = gst_bin_get_by_name(GST_BIN(pipeline), "q_out");

    SinkStats st;
    st.limit = a.frames;
    GstPad* sinkpad = gst_element_get_static_pad(sink, "sink");
    gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_BUFFER, sink_probe, &st, nullptr);
    gst_object_unref(sinkpad);

    HoldState hold;
    hold.n = a.hold;
    if (a.sink == SinkKind::Hold) {
        GstAppSinkCallbacks cb{};
        cb.new_sample = hold_new_sample;
        gst_app_sink_set_callbacks(GST_APP_SINK(sink), &cb, &hold, nullptr);
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    const long fds_before = yvo::tool::count_open_fds();
    [[maybe_unused]] const double soc_t0 = yvo::tool::soc_temperature();
    bool eos = false, error = false, stalled = false, eos_sent = false;
    std::string caps_in, caps_out;
    // Preroll first: element start() (VulkanContext, HailoRT VDevice + configure) and the first buffers happen
    // before t0, like yvo_pipeline's init before its timed loop. The sink holds one prerolled buffer; the ring fills.
    if (gst_element_set_state(pipeline, GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
        std::printf("set_state(PAUSED) failed\n");
        error = true;
    }
    const std::uint64_t preroll_deadline = mono_ns() + 60'000'000'000ull;
    bool prerolled = false;
    while (!error && !prerolled) {
        GstMessage* m = gst_bus_timed_pop_filtered(
            bus, 100 * GST_MSECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING | GST_MESSAGE_ASYNC_DONE |
                                        GST_MESSAGE_EOS));
        if (m) {
            const GstMessageType ty = GST_MESSAGE_TYPE(m);
            if (ty == GST_MESSAGE_ASYNC_DONE && GST_MESSAGE_SRC(m) == GST_OBJECT(pipeline)) prerolled = true;
            else if (ty == GST_MESSAGE_EOS) prerolled = eos = true;  // empty input: EOS prerolls too
            else if (ty == GST_MESSAGE_ERROR || ty == GST_MESSAGE_WARNING) {
                print_message(m);
                if (ty == GST_MESSAGE_ERROR) error = true;
            }
            gst_message_unref(m);
        }
        if (!prerolled && !error && mono_ns() > preroll_deadline) {
            std::printf("preroll: no ASYNC_DONE within 60 s\n");
            error = true;
        }
    }
    const double c0 = yvo::tool::cpu_ms();
    const std::uint64_t t0 = mono_ns();
    if (!error && gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::printf("set_state(PLAYING) failed\n");
        error = true;
    }
    std::uint64_t t_end = t0;
    while (!error && !eos) {
        GstMessage* m = gst_bus_timed_pop_filtered(
            bus, 100 * GST_MSECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING | GST_MESSAGE_EOS));
        if (m) {
            if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_EOS) {
                eos = true;
            } else {
                print_message(m);
                if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR) error = true;
            }
            gst_message_unref(m);
            if (eos || error) break;
        }
        if (st.frames.load() > 0 && caps_out.empty()) {
            caps_in = pad_caps(lb, "sink");
            caps_out = pad_caps(lb, "src");
        }
        if (st.limit_reached.load() && !eos_sent) {
            gst_element_send_event(pipeline, gst_event_new_eos());
            eos_sent = true;
        }
        const std::uint64_t now = mono_ns();
        const std::uint64_t last = std::max(st.last_ns.load(), t0);
        if (now - last > static_cast<std::uint64_t>(a.stall_timeout_s) * 1'000'000'000ull) {
            stalled = true;
            guint64 ef = 0;
            guint in_use = 0, detached = 0;
            g_object_get(lb, "frames", &ef, "slots-in-use", &in_use, "slots-detached", &detached, nullptr);
            int held = 0;
            {
                std::lock_guard<std::mutex> lk(hold.m);
                held = static_cast<int>(hold.held.size());
            }
            const guint qd = queue_level(q_dec), qo = queue_level(q_out);
            std::printf("STALL: no buffer at the sink for %d s (watchdog)\n", a.stall_timeout_s);
            std::printf("  frames at sink %d, element frames %llu, slots in use %u/%d (detached memories %u), "
                        "held by hold-sink %d (N=%d)\n",
                        st.frames.load(), static_cast<unsigned long long>(ef), in_use, a.slots, detached, held,
                        a.sink == SinkKind::Hold ? a.hold : 0);
            std::printf("  queue levels: q_dec %u/%d buffers, q_out %u buffers\n", qd, a.queue, qo);
            if (in_use >= static_cast<guint>(a.slots))
                std::printf("  diagnosis: every output slot is held downstream -> element blocked acquiring a slot "
                            "(ring backpressure)\n");
            else if (qd == 0)
                std::printf("  diagnosis: decoder is not producing (decoder pool starved or decoder error)\n");
            else
                std::printf("  diagnosis: element not consuming (Vulkan / element blocked)\n");
            break;
        }
    }
    t_end = mono_ns();
    const double cpu_total = yvo::tool::cpu_ms() - c0;
    if (caps_out.empty()) {
        caps_in = pad_caps(lb, "sink");
        caps_out = pad_caps(lb, "src");
    }
    guint64 elem_frames = 0;
    gdouble elem_gpu = -1.0;
    g_object_get(lb, "frames", &elem_frames, "gpu-ms-mean", &elem_gpu, nullptr);

    // Teardown: stop holding (so the element can get every slot back at stop()), then NULL.
    hold.draining.store(true);
    hold.release_all();
    gst_element_set_state(pipeline, GST_STATE_NULL);
    guint64 val_errors = 0;
    gboolean val_active = FALSE;
    g_object_get(lb, "validation-errors", &val_errors, "validation-active", &val_active, nullptr);
#ifdef YVO_GSTELEM_HAVE_HAILO
    // The sink keeps its statistics after stop() (NULL): runner released, chip temperature t1 read before that.
    yvo::gst::HailoSinkStats hs;
    const bool have_hs = a.sink == SinkKind::Hailo && yvo::gst::hailo_sink_get_stats(sink, hs);
#endif
    while (GstMessage* m = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING))) {
        print_message(m);
        if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR && !stalled) error = true;
        gst_message_unref(m);
    }
    gst_object_unref(bus);
    gst_object_unref(q_out);
    gst_object_unref(q_dec);
    gst_object_unref(lb);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    // After the pipeline is finalized: slot fds, import-cache dups, Vulkan and decoder fds must all be gone.
    // (Right after NULL a stalled v4l2 decoder may still hold its fds until finalize.)
    const long fds_after = yvo::tool::count_open_fds();
    [[maybe_unused]] const double soc_t1 = yvo::tool::soc_temperature();

    // ── summary ──
    const int frames = st.frames.load();
    const double wall_s = static_cast<double>(t_end - t0) * 1e-9;
    const double fps = wall_s > 0 ? frames / wall_s : 0.0;
    const double steady_s = frames > 1 ? static_cast<double>(st.last_ns.load() - st.first_ns) * 1e-9 : 0.0;
    const double steady_fps = steady_s > 0 ? (frames - 1) / steady_s : 0.0;
    const double cpu_per_frame = frames ? cpu_total / frames : 0.0;
    const yvo::Stats sg = yvo::stats(st.gpu_ms), sl = yvo::stats(st.lat_ms), sw = yvo::stats(st.wait_ms);
    int unstable_slots = 0;
    for (const auto& kv : st.slot_fds)
        if (kv.second.size() != 1) ++unstable_slots;
    const yvo::LetterboxParams lb_ref = yvo::make_letterbox(st.src_w > 0 ? st.src_w : 1920, st.src_h > 0 ? st.src_h : 1080);
    const bool lb_ok = st.have_lb && st.lb_first.scale == lb_ref.scale && st.lb_first.pad_x == lb_ref.pad_x &&
                       st.lb_first.pad_y == lb_ref.pad_y && st.lb_first.dst_w == 640 && st.lb_first.dst_h == 640;

    std::vector<std::string> fails;
    if (stalled) fails.push_back("stall (watchdog)");
    if (error) fails.push_back("pipeline error");
    if (!stalled && !error && !eos) fails.push_back("no EOS");
    if (expected > 0 && frames != expected) fails.push_back(yvo::tool::fmt("frames %d != expected %d", frames, expected));
    if (st.missing_meta || st.seq_gaps || st.bad_slot || st.fd_mismatch || st.lb_mismatch || unstable_slots)
        fails.push_back(yvo::tool::fmt("meta checks: missing %d, seq gaps %d, bad slot %d, fd mismatch %d, lb changes %d, "
                                       "unstable slot fds %d",
                                       st.missing_meta, st.seq_gaps, st.bad_slot, st.fd_mismatch, st.lb_mismatch,
                                       unstable_slots));
    if (frames > 0 && !lb_ok) fails.push_back("letterbox params differ from make_letterbox(src)");
    if (fds_after > fds_before) fails.push_back(yvo::tool::fmt("fd leak: %ld -> %ld", fds_before, fds_after));
    if (val_errors > 0) fails.push_back(yvo::tool::fmt("Vulkan validation errors %llu", static_cast<unsigned long long>(val_errors)));
#ifdef YVO_GSTELEM_HAVE_HAILO
    if (a.sink == SinkKind::Hailo) {
        if (!have_hs) fails.push_back("hailo sink statistics unavailable");
        else {
            if (hs.completed != static_cast<std::uint64_t>(frames) || hs.submitted != hs.completed)
                fails.push_back(yvo::tool::fmt("hailo frame count mismatch: at sink %d, submitted %llu, completed %llu",
                                               frames, static_cast<unsigned long long>(hs.submitted),
                                               static_cast<unsigned long long>(hs.completed)));
            if (hs.errors) fails.push_back(yvo::tool::fmt("HailoRT completion errors %llu",
                                                          static_cast<unsigned long long>(hs.errors)));
            if (hs.drain_timeout) fails.push_back("hailo drain timeout (inferences still in flight at EOS/stop)");
        }
    }
#endif
    const bool pass = fails.empty();

    std::printf("\n== yvo_gst_pipeline summary%s ==\n", stalled ? " [stalled]" : "");
    std::printf("versions        %s\n", versions.c_str());
    std::printf("caps in         %s\n", caps_in.c_str());
    std::printf("caps out        %s\n", caps_out.c_str());
    std::printf("sink            %s (hold %d, max held %d), slots %d, queue %d, decoder-pool-min %d, batch %d\n",
                sink_name(a.sink), a.sink == SinkKind::Hold ? a.hold : 0, hold.max_held, a.slots, a.queue,
                a.decoder_pool_min, a.batch);
    std::printf("frames          %d at sink (expected %d), element %llu\n", frames, expected,
                static_cast<unsigned long long>(elem_frames));
    std::printf("wall            %.3f s, %.2f FPS (PLAYING→EOS, after preroll); steady %.2f FPS (first→last sink buffer)\n", wall_s, fps,
                steady_fps);
    std::printf("cpu             %.3f ms/frame (getrusage self user+sys, whole process incl. GStreamer threads)\n",
                cpu_per_frame);
    std::printf("gpu ms          mean %.3f median %.3f p90 %.3f max %.3f (n %zu; element mean %.3f)\n", sg.mean,
                sg.median, sg.p90, sg.max, sg.n, elem_gpu);
    std::printf("slot wait ms    mean %.3f p90 %.3f max %.3f (element blocked acquiring an output slot)\n", sw.mean,
                sw.p90, sw.max);
    std::printf("elem->sink ms   mean %.3f median %.3f p90 %.3f (input arrival at element -> sink pad)\n", sl.mean,
                sl.median, sl.p90);
    std::printf("slots           %zu used, fd per slot stable: %s, tensor %u B in dma-buf %llu B\n",
                st.slot_fds.size(), unstable_slots == 0 ? "yes" : "NO", st.tensor_bytes,
                static_cast<unsigned long long>(st.dmabuf_bytes));
    std::printf("letterbox       src %dx%d scale %.6f pad %d,%d (make_letterbox: %s)\n", st.src_w, st.src_h,
                static_cast<double>(st.lb_first.scale), st.lb_first.pad_x, st.lb_first.pad_y, lb_ok ? "equal" : "DIFFERENT");
    std::printf("open fds        %ld before PAUSED, %ld after pipeline unref (all released: %s)\n", fds_before,
                fds_after, fds_after <= fds_before ? "yes" : "NO");
    std::printf("vk validation   layer %s, errors %llu\n", val_active ? "on" : "off",
                static_cast<unsigned long long>(val_errors));
#ifdef YVO_GSTELEM_HAVE_HAILO
    yvo::Stats sn, se, shg;
    if (have_hs) {
        sn = yvo::stats(hs.npu_ms);
        se = yvo::stats(hs.e2e_ms);
        shg = yvo::stats(hs.gpu_ms);
        const double per = hs.completed ? static_cast<double>(hs.detections) / static_cast<double>(hs.completed) : 0.0;
        std::printf("npu latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f  (run_async → completion callback)\n",
                    sn.mean, sn.median, sn.p90, sn.max);
        std::printf("e2e latency ms  mean %.3f  median %.3f  p90 %.3f  max %.3f  (letterbox arrival → NPU completion "
                    "callback)\n", se.mean, se.median, se.p90, se.max);
        std::printf("hailo           submitted %llu, completed %llu, errors %llu, batch %d, async queue %zu, in-flight "
                    "max %d, mapped slots %d/%d\n",
                    static_cast<unsigned long long>(hs.submitted), static_cast<unsigned long long>(hs.completed),
                    static_cast<unsigned long long>(hs.errors), hs.batch, hs.async_queue_size, hs.in_flight_max,
                    hs.mapped_slots, hs.ring_slots);
        std::printf("detections      %llu total, %.2f / frame\n", static_cast<unsigned long long>(hs.detections), per);
        std::printf("hailo chip temp %.1f -> %.1f C\n", static_cast<double>(hs.hailo_t0), static_cast<double>(hs.hailo_t1));
        std::printf("soc temp        %.1f -> %.1f C\n", soc_t0, soc_t1);
    }
#endif
    std::printf("result          %s\n", pass ? "PASS" : "FAIL");
    for (const auto& f : fails) std::printf("  - %s\n", f.c_str());
    if (!a.csv_label.empty()) {
        // label,sink,hold,slots,queue,batch,frames,wall_s,fps,steady_fps,cpu_ms_per_frame,gpu_mean,gpu_median,gpu_p90,
        // slot_wait_mean,elem_sink_mean,elem_sink_p90,vk_errors,stalled,ok
        std::printf("RESULT_GST_CSV,%s,%s,%d,%d,%d,%d,%d,%.3f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%d,%d\n",
                    a.csv_label.c_str(), sink_name(a.sink), a.sink == SinkKind::Hold ? a.hold : 0, a.slots, a.queue,
                    a.batch, frames, wall_s, fps, steady_fps, cpu_per_frame, sg.mean, sg.median, sg.p90, sw.mean,
                    sl.mean, sl.p90, static_cast<unsigned long long>(val_errors), stalled ? 1 : 0, pass ? 1 : 0);
#ifdef YVO_GSTELEM_HAVE_HAILO
        if (have_hs) {
            // Same 22 columns and printf formats as yvo_pipeline (pipeline/main.cpp RESULT_CSV), parsed by
            // 06_tools/bench/pipeline_bench.sh. Differences in meaning (documented in the tool header / report):
            //   mode,layout      always dmabuf,nhcw (the only yvosandletterbox output)
            //   slots            yvosandletterbox ring size (yvo_pipeline: GPU+NPU slots)
            //   decode_ms_mean   nan — there is no blocking decode call in the GStreamer path
            //   e2e_*            letterbox arrival → completion (yvo_pipeline: decode return → completion)
            //   wall_s,cpu       PLAYING → EOS after preroll (yvo_pipeline: first decode → drain)
            const long completed = static_cast<long>(hs.completed);
            const double fps_c = wall_s > 0 ? static_cast<double>(completed) / wall_s : 0.0;
            const double cpu_c = completed ? cpu_total / static_cast<double>(completed) : 0.0;
            std::printf(
                "RESULT_CSV,%s,%s,%s,%d,%d,%ld,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%ld,%.1f,%.1f,%.1f,%.1f,%d\n",
                a.csv_label.c_str(), "dmabuf", "nhcw", a.batch, a.slots, completed, wall_s, fps_c, cpu_c, std::nan(""),
                shg.mean, shg.p90, sn.mean, sn.p90, se.mean, se.p90, static_cast<long>(hs.detections),
                static_cast<double>(hs.hailo_t0), static_cast<double>(hs.hailo_t1), soc_t0, soc_t1, pass ? 1 : 0);
        }
#endif
    }
    std::fflush(stdout);
    gst_deinit();
    if (stalled) return 4;
    if (error) return 1;
    return pass ? 0 : 3;
}
