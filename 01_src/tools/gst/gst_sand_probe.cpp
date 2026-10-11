// yvo_gst_sand_probe: GStreamer research path, stage-1 gate
//   Decodes the video with GStreamer (filesrc ! qtdemux ! h265parse ! v4l2slh265dec ! appsink, DMA_DRM NV12 + SAND128),
//   converts target frames with yvo::gst::drm_frame_from_buffer → SandLetterbox (NHCW), and compares the 640x640x3
//   output byte-for-byte with the existing FfmpegDrmDecoder → SandLetterbox path (same VulkanContext, same thread).
//
// Notes on the GStreamer side:
//   - v4l2slh265dec refuses DMA_DRM caps unless downstream advertises GstVideoMeta in the ALLOCATION query;
//     appsink does not, so a sink-pad query probe answers ALLOCATION (VideoMeta + pool min).
//   - The decoder pool never grows and acquire blocks: a starved pool stalls silently. Samples are released right away,
//     appsink keeps max-buffers=2 and no last-sample, and a pull timeout turns a stall into an error.
//   - The two passes run one after the other (not concurrently) to keep CMA use low (CmaTotal is 64 MB).
// Exit 0 only if every target frame matches exactly, 2 planes were found, column heights are 1088/544 and the frame
// count equals --expect-frames. Nothing is written to disk.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>

#include <gst/allocators/allocators.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "core/frame_types.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "gstelem/gst_drm_frame.hpp"
#include "toolkit/cli_parse.hpp"
#include "toolkit/tool_util.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

namespace {

constexpr const char* kSandCaps =
    "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12:0x0700000000000004";

struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    std::vector<int> targets = {0, 100, 500, 900, 1815};
    int expect_frames = 1816;  // 0 = do not check
    int pool_min = 4;          // min buffers proposed in the ALLOCATION answer
    int pull_timeout_s = 5;    // no sample for this long (and no EOS/error) = stall
    bool gpu_all = false;      // run SandLetterbox on every GStreamer frame (import cache behaviour)
};

void usage(const char* argv0) {
    std::printf(
        "usage: %s [options] [VIDEO]\n"
        "  Stage-1 gate: GStreamer v4l2slh265dec DMABuf (SAND128) -> DrmFrame -> SandLetterbox NHCW,\n"
        "  byte-compared with FfmpegDrmDecoder -> SandLetterbox for the same frame indices.\n"
        "  VIDEO                 input (default: %s)\n"
        "  --frames LIST         comma-separated target frame indices (default 0,100,500,900,1815)\n"
        "  --expect-frames N     expected total decoded frames, 0 = no check (default 1816)\n"
        "  --pool-min N          min buffers proposed to the decoder in the ALLOCATION answer (default 4)\n"
        "  --pull-timeout S      seconds without a sample before declaring a stall (default 5)\n"
        "  --gpu-all             run SandLetterbox on every GStreamer frame (compares only targets)\n"
        "  --help                this text\n"
        "Exit codes: 0 pass, 1 runtime error, 2 bad arguments, 3 gate failed.\n",
        argv0, YVO_DEFAULT_VIDEO);
}

bool parse_list(const std::string& s, std::vector<int>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) return false;
        char* end = nullptr;
        const long v = std::strtol(tok.c_str(), &end, 10);
        if (*end != '\0' || v < 0) return false;
        out.push_back(static_cast<int>(v));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return !out.empty();
}

// 0 = ok, 1 = --help, 2 = error
int parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (s == "--help" || s == "-h") return 1;
        if (s == "--frames") {
            const char* v = next();
            if (!v || !parse_list(v, a.targets)) return 2;
        } else if (s == "--expect-frames") {
            const char* v = next();
            if (!v) return 2;
            a.expect_frames = yvo::tool::arg_number<int>(argv[0], s, v);
        } else if (s == "--pool-min") {
            const char* v = next();
            if (!v) return 2;
            a.pool_min = yvo::tool::arg_number<int>(argv[0], s, v);
        } else if (s == "--pull-timeout") {
            const char* v = next();
            if (!v) return 2;
            a.pull_timeout_s = std::max(1, yvo::tool::arg_number<int>(argv[0], s, v));
        } else if (s == "--gpu-all") {
            a.gpu_all = true;
        } else if (!s.empty() && s[0] == '-') {
            return 2;
        } else {
            a.video = s;
        }
    }
    return 0;
}

long cma_free_kb() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    long val = -1;
    std::string unit;
    while (f >> key >> val) {
        std::getline(f, unit);
        if (key == "CmaFree:") return val;
    }
    return -1;
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

std::string frame_fields(const yvo::DrmFrame& f) {
    return yvo::tool::fmt(
        "%dx%d coded_h=%d fourcc=0x%08x modifier=0x%016llx\n"
        "    Y : fd=%d size=%zu offset=%zu pitch=%zu col_height=%u\n"
        "    UV: fd=%d size=%zu offset=%zu pitch=%zu col_height=%u",
        f.width, f.height, f.coded_height, f.drm_format, static_cast<unsigned long long>(f.modifier), f.fd, f.size,
        f.y_offset, f.y_pitch, f.sand_col_height, f.uv_fd, f.uv_size, f.uv_offset, f.uv_pitch, f.uv_sand_col_height);
}

// Fields that determine SAND addressing (a mismatch here would change what the shader reads)
struct Layout {
    int width = 0, height = 0, coded_height = 0;
    std::uint32_t drm_format = 0, col_y = 0, col_uv = 0;
    std::uint64_t modifier = 0;
    std::size_t y_offset = 0, uv_offset = 0, y_pitch = 0, uv_pitch = 0, size = 0, uv_size = 0;
    bool two_objects = false;
};

Layout layout_of(const yvo::DrmFrame& f) {
    Layout l;
    l.width = f.width;
    l.height = f.height;
    l.coded_height = f.coded_height;
    l.drm_format = f.drm_format;
    l.col_y = f.sand_col_height;
    l.col_uv = f.uv_sand_col_height;
    l.modifier = f.modifier;
    l.y_offset = f.y_offset;
    l.uv_offset = f.uv_offset;
    l.y_pitch = f.y_pitch;
    l.uv_pitch = f.uv_pitch;
    l.size = f.size;
    l.uv_size = f.uv_size;
    l.two_objects = f.uv_fd >= 0 && f.uv_fd != f.fd;
    return l;
}

bool same_addressing(const Layout& a, const Layout& b) {
    constexpr std::uint64_t kParamMask = 0x00ffffffffffff00ull; // Broadcom modifier parameter bits
    return a.width == b.width && a.height == b.height && a.drm_format == b.drm_format && a.col_y == b.col_y &&
           a.col_uv == b.col_uv && (a.modifier & ~kParamMask) == (b.modifier & ~kParamMask) &&
           a.y_offset == b.y_offset && a.uv_offset == b.uv_offset && a.two_objects == b.two_objects;
}

struct GstStats {
    int allocation_queries = 0;
    std::string allocation_caps;
    int pool_min = 4;
};

// Answers the decoder's ALLOCATION query on the appsink sink pad: VideoMeta support + a small pool hint.
GstPadProbeReturn allocation_probe(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user) {
    GstQuery* q = GST_PAD_PROBE_INFO_QUERY(info);
    if (GST_QUERY_TYPE(q) != GST_QUERY_ALLOCATION) return GST_PAD_PROBE_OK;
    auto* st = static_cast<GstStats*>(user);
    GstCaps* caps = nullptr;
    gboolean need_pool = FALSE;
    gst_query_parse_allocation(q, &caps, &need_pool);
    guint size = 0;
    if (caps) {
        gchar* s = gst_caps_to_string(caps);
        st->allocation_caps = s;
        g_free(s);
        GstVideoInfoDmaDrm di;
        if (gst_video_info_dma_drm_from_caps(&di, caps)) size = static_cast<guint>(GST_VIDEO_INFO_SIZE(&di.vinfo));
    }
    gst_query_add_allocation_meta(q, GST_VIDEO_META_API_TYPE, nullptr);
    gst_query_add_allocation_pool(q, nullptr, size, static_cast<guint>(st->pool_min), 0);
    ++st->allocation_queries;
    return GST_PAD_PROBE_HANDLED; // query answered: returns TRUE to the decoder
}

std::string bus_error(GstElement* pipeline) {
    GstBus* bus = gst_element_get_bus(pipeline);
    std::string out;
    while (GstMessage* m = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING))) {
        GError* err = nullptr;
        gchar* dbg = nullptr;
        const bool is_err = GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR;
        if (is_err)
            gst_message_parse_error(m, &err, &dbg);
        else
            gst_message_parse_warning(m, &err, &dbg);
        out += yvo::tool::fmt("%s from %s: %s (%s)\n", is_err ? "ERROR" : "WARNING", GST_OBJECT_NAME(GST_MESSAGE_SRC(m)),
                              err ? err->message : "?", dbg ? dbg : "");
        if (err) g_error_free(err);
        g_free(dbg);
        gst_message_unref(m);
    }
    gst_object_unref(bus);
    return out;
}

struct PassResult {
    std::map<int, std::vector<std::uint8_t>> out;  // target index → NHCW bytes
    std::map<int, Layout> layout;                  // target index → DrmFrame layout
    std::map<int, double> gpu_ms;
    int frames = 0;
    bool ok = true;
    std::string error;
};

void run_target(yvo::vk::SandLetterbox& pre, const yvo::DrmFrame& f, int idx, bool keep, PassResult& r) {
    const yvo::PreprocOutput o = pre.process(f, 0);
    if (!keep) return;
    r.out[idx].assign(o.host_ptr, o.host_ptr + o.size);
    r.layout[idx] = layout_of(f);
    r.gpu_ms[idx] = pre.last_gpu_ms();
}

struct GstPassInfo {
    bool two_planes = false;
    std::uint32_t col_y = 0, col_uv = 0;
    long cma_min_kb = -1;
    std::set<std::pair<int, int>> fd_pairs;
    std::set<std::pair<ino_t, ino_t>> ino_pairs;
    int layout_changes = 0;
    int non_dmabuf = 0;
    double wall_s = 0.0, cpu_ms = 0.0;
};

PassResult gst_pass(const Args& a, yvo::vk::SandLetterbox& pre, GstPassInfo& gi) {
    PassResult r;
    const std::string desc = "filesrc location=\"" + a.video +
                             "\" ! qtdemux ! h265parse ! v4l2slh265dec name=dec ! appsink name=sink sync=false "
                             "max-buffers=2 drop=false enable-last-sample=false caps=\"" +
                             kSandCaps + "\"";
    std::printf("[gst] pipeline: %s\n", desc.c_str());
    GError* perr = nullptr;
    GstElement* pipeline = gst_parse_launch(desc.c_str(), &perr);
    if (!pipeline || perr) {
        r.ok = false;
        r.error = std::string("gst_parse_launch: ") + (perr ? perr->message : "?");
        if (perr) g_error_free(perr);
        if (pipeline) gst_object_unref(pipeline);
        return r;
    }
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    GstStats st;
    st.pool_min = a.pool_min;
    GstPad* sinkpad = gst_element_get_static_pad(sink, "sink");
    gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM, allocation_probe, &st, nullptr);
    gst_object_unref(sinkpad);

    const int last_target = a.targets.back();
    const auto t0 = std::chrono::steady_clock::now();
    const double c0 = yvo::tool::cpu_ms();
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        r.ok = false;
        r.error = "set_state(PLAYING) failed\n" + bus_error(pipeline);
    }

    yvo::gst::SandCaps sc;
    GstCaps* cur_caps = nullptr;
    std::optional<Layout> first_layout;
    int idx = 0;
    while (r.ok) {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink),
                                                         static_cast<GstClockTime>(a.pull_timeout_s) * GST_SECOND);
        if (!sample) {
            if (gst_app_sink_is_eos(GST_APP_SINK(sink))) break;
            const std::string errs = bus_error(pipeline);
            r.ok = false;
            r.error = errs.empty() ? yvo::tool::fmt("stall: no sample for %d s after %d frames (no EOS, no bus error)"
                                                    " - decoder pool starved?",
                                                    a.pull_timeout_s, idx)
                                   : "bus: " + errs;
            break;
        }
        GstBuffer* buf = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        if (caps != cur_caps) {
            std::string why;
            gchar* cs = gst_caps_to_string(caps);
            std::printf("[gst] sample caps (frame %d): %s\n", idx, cs);
            g_free(cs);
            if (!yvo::gst::sand_caps_from_caps(caps, sc, &why)) {
                r.ok = false;
                r.error = "caps rejected: " + why;
                gst_sample_unref(sample);
                break;
            }
            GstVideoInfoDmaDrm di;
            if (gst_video_info_dma_drm_from_caps(&di, caps))
                std::printf("[gst] GstVideoInfoDmaDrm: vinfo.format=%s %dx%d size=%zu drm_fourcc=0x%08x modifier=0x%016llx\n",
                            gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&di.vinfo)), GST_VIDEO_INFO_WIDTH(&di.vinfo),
                            GST_VIDEO_INFO_HEIGHT(&di.vinfo), static_cast<std::size_t>(GST_VIDEO_INFO_SIZE(&di.vinfo)),
                            di.drm_fourcc, static_cast<unsigned long long>(di.drm_modifier));
            else
                std::printf("[gst] gst_video_info_dma_drm_from_caps failed\n");
            cur_caps = caps;
        }

        yvo::DrmFrame f;
        std::string why;
        if (!yvo::gst::drm_frame_from_buffer(buf, sc, f, &why)) {
            r.ok = false;
            r.error = yvo::tool::fmt("frame %d: adapter rejected buffer: %s\n", idx, why.c_str()) +
                      yvo::gst::describe_buffer(buf);
            gst_sample_unref(sample);
            break;
        }
        const Layout l = layout_of(f);
        if (idx == 0) {
            std::printf("[gst] frame 0 buffer:\n%s", yvo::gst::describe_buffer(buf).c_str());
            std::printf("[gst] frame 0 DrmFrame: %s\n", frame_fields(f).c_str());
            gi.two_planes = gst_buffer_n_memory(buf) == 2 && l.two_objects;
            gi.col_y = l.col_y;
            gi.col_uv = l.col_uv;
            first_layout = l;
        } else if (!same_addressing(*first_layout, l) || l.size != first_layout->size || l.uv_size != first_layout->uv_size) {
            if (++gi.layout_changes <= 3)
                std::printf("[gst] frame %d: layout differs from frame 0:\n%s  DrmFrame: %s\n", idx,
                            yvo::gst::describe_buffer(buf).c_str(), frame_fields(f).c_str());
        }
        for (guint m = 0; m < gst_buffer_n_memory(buf); ++m)
            if (!gst_is_dmabuf_memory(gst_buffer_peek_memory(buf, m))) ++gi.non_dmabuf;
        gi.fd_pairs.insert({f.fd, f.uv_fd});
        struct stat sy {}, su {};
        if (fstat(f.fd, &sy) == 0 && fstat(f.uv_fd, &su) == 0) gi.ino_pairs.insert({sy.st_ino, su.st_ino});

        const bool is_target = std::binary_search(a.targets.begin(), a.targets.end(), idx);
        if (is_target) {
            const long cma = cma_free_kb();
            if (gi.cma_min_kb < 0 || (cma >= 0 && cma < gi.cma_min_kb)) gi.cma_min_kb = cma;
        }
        if (is_target || a.gpu_all) {
            try {
                run_target(pre, f, idx, is_target, r);
            } catch (const std::exception& e) {
                r.ok = false;
                r.error = yvo::tool::fmt("frame %d: SandLetterbox: %s\n", idx, e.what()) + yvo::gst::describe_buffer(buf) +
                          "  DrmFrame: " + frame_fields(f) + "\n";
            }
        }
        f = {};  // drop the keepalive ref before the sample so the buffer returns to the decoder pool
        gst_sample_unref(sample);
        ++idx;
        if (!a.gpu_all && a.expect_frames == 0 && idx > last_target) break;
    }
    gi.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    gi.cpu_ms = yvo::tool::cpu_ms() - c0;
    r.frames = idx;
    const std::string late = bus_error(pipeline);
    if (!late.empty()) std::printf("[gst] bus messages:\n%s", late.c_str());
    std::printf("[gst] ALLOCATION queries answered: %d (caps: %s)\n", st.allocation_queries, st.allocation_caps.c_str());
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return r;
}

PassResult ffmpeg_pass(const Args& a, yvo::vk::SandLetterbox& pre, yvo::DrmFrame* first) {
    PassResult r;
    yvo::FfmpegDrmDecoder dec;
    if (!dec.open(a.video)) {
        r.ok = false;
        r.error = "FfmpegDrmDecoder::open: " + dec.error();
        return r;
    }
    const int last_target = a.targets.back();
    int idx = 0;
    while (idx <= last_target) {
        std::optional<yvo::DrmFrame> f = dec.next();
        if (!f) {
            if (!dec.eof()) {
                r.ok = false;
                r.error = "FfmpegDrmDecoder::next: " + dec.error();
            }
            break;
        }
        if (idx == 0 && first) *first = *f;
        if (std::binary_search(a.targets.begin(), a.targets.end(), idx)) {
            try {
                run_target(pre, *f, idx, true, r);
            } catch (const std::exception& e) {
                r.ok = false;
                r.error = yvo::tool::fmt("frame %d: SandLetterbox: %s", idx, e.what());
                break;
            }
        }
        ++idx;
    }
    r.frames = idx;
    return r;
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
    gst_init(&argc, &argv);

    std::printf("== yvo_gst_sand_probe ==\n");
    std::printf("video: %s\ntargets:", a.video.c_str());
    for (int t : a.targets) std::printf(" %d", t);
    std::printf("\nexpect_frames=%d pool_min=%d gpu_all=%d\n", a.expect_frames, a.pool_min, a.gpu_all ? 1 : 0);
    gchar* gv = gst_version_string();
    std::printf("GStreamer runtime: %s\n", gv);
    g_free(gv);
    if (GstPlugin* pl = gst_registry_find_plugin(gst_registry_get(), "v4l2codecs")) {
        std::printf("v4l2codecs plugin: version=%s package=\"%s\" file=%s\n", gst_plugin_get_version(pl),
                    gst_plugin_get_package(pl), gst_plugin_get_filename(pl));
        gst_object_unref(pl);
    } else {
        std::printf("v4l2codecs plugin: NOT FOUND\n");
    }
    std::printf("gstreamer1.0-plugins-bad (dpkg): %s\n",
                shell_line("dpkg-query -W -f='${Version}' gstreamer1.0-plugins-bad 2>/dev/null").c_str());

    const long cma_before = cma_free_kb();
    std::printf("CmaFree before: %ld kB\n", cma_before);

    int rc = 0;
    try {
        yvo::vk::VulkanContext ctx;
        yvo::vk::SandLetterboxConfig cfg;
        cfg.num_slots = 1;
        cfg.out_layout = yvo::TensorLayout::Nhcw;
        yvo::vk::SandLetterbox pre(ctx, cfg);
        std::printf("Vulkan device: %s\n", ctx.device_name());

        // ---- pass 1: GStreamer ----
        GstPassInfo gi;
        PassResult g = gst_pass(a, pre, gi);
        const auto& ic = pre.import_cache();
        std::printf("[gst] frames pulled: %d  wall=%.2f s (%.1f FPS)  cpu=%.3f ms/frame\n", g.frames, gi.wall_s,
                    gi.wall_s > 0 ? g.frames / gi.wall_s : 0.0, g.frames ? gi.cpu_ms / g.frames : 0.0);
        std::printf("[gst] distinct (Y fd, UV fd) pairs: %zu, distinct (Y ino, UV ino) pairs: %zu, "
                    "non-dmabuf memories: %d, layout changes vs frame 0: %d\n",
                    gi.fd_pairs.size(), gi.ino_pairs.size(), gi.non_dmabuf, gi.layout_changes);
        std::printf("[gst] fd pairs:");
        for (const auto& p : gi.fd_pairs) std::printf(" (%d,%d)", p.first, p.second);
        std::printf("\n[gst] import cache: entries=%zu imports=%llu hits=%llu\n", ic.entries(),
                    static_cast<unsigned long long>(ic.imports()), static_cast<unsigned long long>(ic.hits()));
        std::printf("CmaFree min during GStreamer pass (at target frames): %ld kB\n", gi.cma_min_kb);
        if (!g.ok) std::printf("[gst] FAILED: %s\n", g.error.c_str());
        pre.clear_import_cache();  // cache entries hold dup'ed fds → release the decoder's dma-bufs
        const long cma_after_gst = cma_free_kb();
        std::printf("CmaFree after GStreamer teardown + cache clear: %ld kB\n", cma_after_gst);

        // ---- pass 2: FFmpeg reference ----
        yvo::DrmFrame ff0;
        PassResult r = ffmpeg_pass(a, pre, &ff0);
        std::printf("[ffmpeg] frames decoded: %d\n", r.frames);
        if (ff0.fd >= 0) std::printf("[ffmpeg] frame 0 DrmFrame: %s\n", frame_fields(ff0).c_str());
        ff0 = {};
        if (!r.ok) std::printf("[ffmpeg] FAILED: %s\n", r.error.c_str());
        pre.clear_import_cache();

        // ---- compare ----
        bool pass = g.ok && r.ok;
        std::printf("\n== comparison (NHCW 640x640x3, byte-exact) ==\n");
        std::printf("%6s %10s %10s %12s  %s\n", "frame", "gst_gpu_ms", "ff_gpu_ms", "mismatches", "first mismatch");
        for (int t : a.targets) {
            const auto gi_it = g.out.find(t);
            const auto ri_it = r.out.find(t);
            if (gi_it == g.out.end() || ri_it == r.out.end()) {
                std::printf("%6d %10s %10s %12s  missing (gst=%d ffmpeg=%d)\n", t, "-", "-", "-",
                            gi_it != g.out.end(), ri_it != r.out.end());
                pass = false;
                continue;
            }
            const auto& x = gi_it->second;
            const auto& y = ri_it->second;
            std::size_t mism = 0, first = SIZE_MAX;
            if (x.size() != y.size()) {
                mism = std::max(x.size(), y.size());
                first = 0;
            } else {
                for (std::size_t i = 0; i < x.size(); ++i)
                    if (x[i] != y[i]) {
                        if (first == SIZE_MAX) first = i;
                        ++mism;
                    }
            }
            std::string fm = "-";
            if (first != SIZE_MAX && first < x.size() && first < y.size()) {
                const std::size_t dw = 640, dx = first % dw, row = first / dw;
                fm = yvo::tool::fmt("byte %zu (dy=%zu c=%zu dx=%zu) gst=%u ffmpeg=%u", first, row / 3, row % 3, dx,
                                    x[first], y[first]);
            }
            std::printf("%6d %10.3f %10.3f %12zu  %s\n", t, g.gpu_ms[t], r.gpu_ms[t], mism, fm.c_str());
            if (mism != 0) pass = false;
        }

        std::printf("\n== DrmFrame layout, gst vs ffmpeg (per target) ==\n");
        for (int t : a.targets) {
            if (!g.layout.count(t) || !r.layout.count(t)) continue;
            const Layout& p = g.layout[t];
            const Layout& q = r.layout[t];
            const bool addr = same_addressing(p, q);
            std::printf("frame %d: addressing %s | size gst=%zu/%zu ffmpeg=%zu/%zu | coded_h gst=%d ffmpeg=%d | "
                        "pitch gst=%zu/%zu ffmpeg=%zu/%zu | offsets gst=%zu/%zu ffmpeg=%zu/%zu | col gst=%u/%u ffmpeg=%u/%u"
                        " | modifier gst=0x%016llx ffmpeg=0x%016llx | objects gst=%d ffmpeg=%d\n",
                        t, addr ? "SAME" : "DIFFERENT", p.size, p.uv_size, q.size, q.uv_size, p.coded_height,
                        q.coded_height, p.y_pitch, p.uv_pitch, q.y_pitch, q.uv_pitch, p.y_offset, p.uv_offset, q.y_offset,
                        q.uv_offset, p.col_y, p.col_uv, q.col_y, q.col_uv, static_cast<unsigned long long>(p.modifier),
                        static_cast<unsigned long long>(q.modifier), p.two_objects ? 2 : 1, q.two_objects ? 2 : 1);
            if (!addr) pass = false;
        }

        const long cma_after = cma_free_kb();
        std::printf("\nCmaFree: before=%ld kB, min during gst=%ld kB, after gst=%ld kB, after all=%ld kB\n", cma_before,
                    gi.cma_min_kb, cma_after_gst, cma_after);
        std::printf("validation: layer=%s messenger=%s errors=%llu warnings=%llu\n",
                    ctx.validation_layer_active() ? "on" : "off", ctx.debug_messenger_active() ? "on" : "off",
                    static_cast<unsigned long long>(ctx.validation_errors()),
                    static_cast<unsigned long long>(ctx.validation_warnings()));

        // ---- gate ----
        std::vector<std::string> fails;
        if (!g.ok) fails.push_back("GStreamer pass failed");
        if (!r.ok) fails.push_back("FFmpeg pass failed");
        if (!gi.two_planes) fails.push_back("frame 0 did not have 2 plane memories / 2 dma-buf objects");
        if (gi.col_y != 1088 || gi.col_uv != 544)
            fails.push_back(yvo::tool::fmt("column heights %u/%u (expected 1088/544)", gi.col_y, gi.col_uv));
        if (a.expect_frames > 0 && g.frames != a.expect_frames)
            fails.push_back(yvo::tool::fmt("GStreamer frames %d (expected %d)", g.frames, a.expect_frames));
        if (!pass) fails.push_back("output mismatch or layout difference (see tables)");
        if (ctx.validation_errors() != 0) fails.push_back("Vulkan validation errors");
        if (fails.empty()) {
            std::printf("\nGATE: PASS\n");
        } else {
            std::printf("\nGATE: FAIL\n");
            for (const auto& s : fails) std::printf("  - %s\n", s.c_str());
            rc = 3;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rc = 1;
    }
    gst_deinit();
    return rc;
}
