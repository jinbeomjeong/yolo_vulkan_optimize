// GStreamer research path (stage 2a): adapter and yvosandletterbox element vs the FFmpeg DrmFrame path
// (GPU + HEVC HW decoder, no NPU). Skipped when the video or v4l2slh265dec is missing.
//
//   GstAdapter.*  (a) v4l2slh265dec ! appsink → drm_frame_from_buffer → SandLetterbox NHCW, compared with
//                     FfmpegDrmDecoder → SandLetterbox for the same frame index: DrmFrame addressing equal, 0 mismatching bytes.
//   GstElementPath*  (b) v4l2slh265dec ! queue ! yvosandletterbox ! appsink → the output slot dma-buf (DmaBufReadMap) of the
//                     target frames vs the same FFmpeg reference: 0 mismatching bytes. TensorMeta checks on every frame
//                     (seq, slot range, fd per slot stable, letterbox == make_letterbox(1920,1080)).
//   GstElementCaps.* sink pad accepts only DMA_DRM NV12:0x0700000000000004 (no CPU fallback). No GPU needed.
//   GstElementSlots.* consumer misuse: a slot memory moved into another buffer is never overwritten (explicit element
//                     error instead), and replacing a slot buffer's memory does not leak the slot (all 1816 frames).
// Frame indices are counted buffers (PTS of frame 0 is 66.67 ms, so PTS is not an index).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "core/dmabuf_map.hpp"
#include "core/frame_types.hpp"
#include "toolkit/tool_util.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "gstelem/gst_drm_frame.hpp"
#include "gstelem/register.hpp"
#include "gstelem/tensor_meta.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

namespace {

const std::vector<int> kTargets = {0, 100, 1815};
constexpr int kExpectFrames = 1816;
constexpr int kElementSlots = 4;
constexpr GstClockTime kPullTimeout = 10 * GST_SECOND; // a starved decoder pool stalls silently
constexpr const char* kSandCaps = "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12:0x0700000000000004";

std::string video_path() {
    const char* env = std::getenv("YVO_TEST_VIDEO");
    return env && *env ? env : YVO_TEST_VIDEO;
}

void ensure_gst() {
    static bool done = false;
    if (done) return;
    gst_init(nullptr, nullptr);
    yvo::gst::register_elements();
    done = true;
}

bool have_hw_decoder() {
    ensure_gst();
    GstElementFactory* f = gst_element_factory_find("v4l2slh265dec");
    if (!f) return false;
    gst_object_unref(f);
    return true;
}

// Fields that determine SAND addressing (same criterion as yvo_gst_sand_probe)
struct Layout {
    int width = 0, height = 0;
    std::uint32_t drm_format = 0, col_y = 0, col_uv = 0;
    std::uint64_t modifier = 0;
    std::size_t y_offset = 0, uv_offset = 0;
    bool two_objects = false;
};

Layout layout_of(const yvo::DrmFrame& f) {
    Layout l;
    l.width = f.width;
    l.height = f.height;
    l.drm_format = f.drm_format;
    l.col_y = f.sand_col_height;
    l.col_uv = f.uv_sand_col_height;
    l.modifier = f.modifier & ~0x00ffffffffffff00ull; // Broadcom modifier parameter bits ignored
    l.y_offset = f.y_offset;
    l.uv_offset = f.uv_offset;
    l.two_objects = f.uv_fd >= 0 && f.uv_fd != f.fd;
    return l;
}

bool same_addressing(const Layout& a, const Layout& b) {
    return a.width == b.width && a.height == b.height && a.drm_format == b.drm_format && a.col_y == b.col_y &&
           a.col_uv == b.col_uv && a.modifier == b.modifier && a.y_offset == b.y_offset && a.uv_offset == b.uv_offset &&
           a.two_objects == b.two_objects;
}

bool is_target(int i) { return std::find(kTargets.begin(), kTargets.end(), i) != kTargets.end(); }

std::size_t count_mismatch(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b, std::size_t* first) {
    *first = SIZE_MAX;
    if (a.size() != b.size()) return std::max(a.size(), b.size());
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            if (*first == SIZE_MAX) *first = i;
            ++n;
        }
    return n;
}

struct Ref {
    Layout layout;
    std::vector<std::uint8_t> nhcw;
};

// FFmpeg DrmFrame → SandLetterbox NHCW for the target frames (once per process, own VulkanContext, released after)
const std::map<int, Ref>& ffmpeg_reference(std::string& err) {
    static std::map<int, Ref> ref;
    static std::string ref_err;
    static bool done = false;
    if (!done) {
        done = true;
        try {
            yvo::vk::VulkanContext ctx;
            yvo::vk::SandLetterboxConfig cfg;
            cfg.num_slots = 1;
            cfg.out_layout = yvo::TensorLayout::Nhcw;
            yvo::vk::SandLetterbox pre(ctx, cfg);
            yvo::FfmpegDrmDecoder dec;
            if (!dec.open(video_path())) {
                ref_err = "FfmpegDrmDecoder::open: " + dec.error();
            } else {
                const int last = kTargets.back();
                for (int i = 0; i <= last; ++i) {
                    std::optional<yvo::DrmFrame> f = dec.next();
                    if (!f) {
                        ref_err = "FFmpeg decode stopped at frame " + std::to_string(i) + ": " + dec.error();
                        break;
                    }
                    if (!is_target(i)) continue;
                    const yvo::PreprocOutput o = pre.process(*f, 0);
                    ref[i] = Ref{layout_of(*f), std::vector<std::uint8_t>(o.host_ptr, o.host_ptr + o.size)};
                }
            }
            dec.close();
            pre.clear_import_cache();
        } catch (const std::exception& e) {
            ref_err = std::string("FFmpeg reference: ") + e.what();
        }
    }
    err = ref_err;
    return ref;
}

// appsink directly after the decoder: answer ALLOCATION with VideoMeta (the decoder refuses DMA_DRM otherwise)
GstPadProbeReturn allocation_probe(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer /*user*/) {
    GstQuery* q = GST_PAD_PROBE_INFO_QUERY(info);
    if (GST_QUERY_TYPE(q) != GST_QUERY_ALLOCATION) return GST_PAD_PROBE_OK;
    GstCaps* caps = nullptr;
    gboolean need_pool = FALSE;
    gst_query_parse_allocation(q, &caps, &need_pool);
    guint size = 0;
    GstVideoInfoDmaDrm di;
    if (caps && gst_video_info_dma_drm_from_caps(&di, caps)) size = static_cast<guint>(GST_VIDEO_INFO_SIZE(&di.vinfo));
    gst_query_add_allocation_meta(q, GST_VIDEO_META_API_TYPE, nullptr);
    gst_query_add_allocation_pool(q, nullptr, size, 4, 0);
    return GST_PAD_PROBE_HANDLED;
}

std::string drain_bus_errors(GstElement* pipeline) {
    GstBus* bus = gst_element_get_bus(pipeline);
    std::string out;
    while (GstMessage* m = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR)) {
        GError* e = nullptr;
        gchar* d = nullptr;
        gst_message_parse_error(m, &e, &d);
        out += std::string(GST_OBJECT_NAME(GST_MESSAGE_SRC(m))) + ": " + (e ? e->message : "?") + " (" + (d ? d : "") + ")\n";
        if (e) g_error_free(e);
        g_free(d);
        gst_message_unref(m);
    }
    gst_object_unref(bus);
    return out;
}

// Runs desc (must contain an appsink named "sink") to EOS, calling on_sample(index, sample) for every sample.
// on_sample returns false to fail. Returns the frame count, -1 on error (err filled).
template <typename F>
int run_appsink(const std::string& desc, bool allocation_answer, F&& on_sample, std::string& err,
                GstElement** keep_named = nullptr, const char* keep_name = nullptr) {
    GError* perr = nullptr;
    GstElement* pipeline = gst_parse_launch(desc.c_str(), &perr);
    if (!pipeline || perr) {
        err = std::string("gst_parse_launch: ") + (perr ? perr->message : "?");
        if (perr) g_error_free(perr);
        if (pipeline) gst_object_unref(pipeline);
        return -1;
    }
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (allocation_answer) {
        GstPad* p = gst_element_get_static_pad(sink, "sink");
        gst_pad_add_probe(p, GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM, allocation_probe, nullptr, nullptr);
        gst_object_unref(p);
    }
    if (keep_named) *keep_named = gst_bin_get_by_name(GST_BIN(pipeline), keep_name);
    int idx = 0;
    bool ok = gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE;
    if (!ok) err = "set_state(PLAYING) failed: " + drain_bus_errors(pipeline);
    while (ok) {
        GstSample* s = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), kPullTimeout);
        if (!s) {
            if (gst_app_sink_is_eos(GST_APP_SINK(sink))) break;
            const std::string bus = drain_bus_errors(pipeline);
            err = bus.empty() ? "stall: no sample for 10 s after " + std::to_string(idx) + " frames" : "bus: " + bus;
            ok = false;
            break;
        }
        if (!on_sample(idx, s)) ok = false;
        gst_sample_unref(s);
        ++idx;
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    if (ok) {
        const std::string late = drain_bus_errors(pipeline);
        if (!late.empty()) {
            err = "bus: " + late;
            ok = false;
        }
    }
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return ok ? idx : -1;
}

std::string decode_desc(const std::string& video) {
    return "filesrc location=\"" + video + "\" ! qtdemux ! h265parse ! v4l2slh265dec";
}

// ── (a) adapter ─────────────────────────────────────────────────────────────────────────────
class GstAdapter : public ::testing::TestWithParam<int> {
protected:
    struct Out {
        Layout layout;
        std::vector<std::uint8_t> nhcw;
    };
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(video_path()) || !have_hw_decoder()) return;
        std::string rerr;
        ffmpeg_reference(rerr);
        if (!rerr.empty()) {
            setup_error_ = rerr;
            return;
        }
        try {
            yvo::vk::VulkanContext ctx;
            yvo::vk::SandLetterboxConfig cfg;
            cfg.num_slots = 1;
            cfg.out_layout = yvo::TensorLayout::Nhcw;
            yvo::vk::SandLetterbox pre(ctx, cfg);
            yvo::gst::SandCaps sc;
            GstCaps* cur = nullptr;
            std::string err;
            const std::string desc = decode_desc(video_path()) +
                                     " ! appsink name=sink sync=false max-buffers=2 drop=false enable-last-sample=false "
                                     "caps=\"" + kSandCaps + "\"";
            frames_ = run_appsink(
                desc, true,
                [&](int i, GstSample* s) {
                    if (!is_target(i)) return true;
                    GstCaps* caps = gst_sample_get_caps(s);
                    std::string why;
                    if (caps != cur && !yvo::gst::sand_caps_from_caps(caps, sc, &why)) {
                        err = "caps rejected: " + why;
                        return false;
                    }
                    cur = caps;
                    yvo::DrmFrame f;
                    if (!yvo::gst::drm_frame_from_buffer(gst_sample_get_buffer(s), sc, f, &why)) {
                        err = "adapter rejected frame " + std::to_string(i) + ": " + why;
                        return false;
                    }
                    const yvo::PreprocOutput o = pre.process(f, 0);
                    out_[i] = Out{layout_of(f), std::vector<std::uint8_t>(o.host_ptr, o.host_ptr + o.size)};
                    return true;
                },
                err);
            pre.clear_import_cache();
            if (frames_ < 0) setup_error_ = err;
        } catch (const std::exception& e) {
            setup_error_ = std::string("exception: ") + e.what();
        }
    }
    void SetUp() override {
        if (!std::filesystem::exists(video_path())) GTEST_SKIP() << "video not found: " << video_path();
        if (!have_hw_decoder()) GTEST_SKIP() << "v4l2slh265dec not available";
        ASSERT_TRUE(setup_error_.empty()) << setup_error_;
    }
    static std::string setup_error_;
    static int frames_;
    static std::map<int, Out> out_;
};
std::string GstAdapter::setup_error_;
int GstAdapter::frames_ = 0;
std::map<int, GstAdapter::Out> GstAdapter::out_;

TEST_P(GstAdapter, LayoutAndNhcwMatchFfmpeg) {
    const int idx = GetParam();
    std::string rerr;
    const auto& ref = ffmpeg_reference(rerr);
    ASSERT_TRUE(rerr.empty()) << rerr;
    EXPECT_EQ(frames_, kExpectFrames);
    ASSERT_TRUE(out_.count(idx)) << "GStreamer frame " << idx << " not collected";
    ASSERT_TRUE(ref.count(idx)) << "FFmpeg frame " << idx << " not collected";
    const Layout& g = out_[idx].layout;
    const Layout& r = ref.at(idx).layout;
    EXPECT_TRUE(same_addressing(g, r)) << "DrmFrame addressing differs: gst col " << g.col_y << "/" << g.col_uv
                                       << " offs " << g.y_offset << "/" << g.uv_offset << " vs ffmpeg col " << r.col_y
                                       << "/" << r.col_uv << " offs " << r.y_offset << "/" << r.uv_offset;
    EXPECT_EQ(g.col_y, 1088u);
    EXPECT_EQ(g.col_uv, 544u);
    EXPECT_TRUE(g.two_objects);
    std::size_t first = 0;
    const std::size_t mism = count_mismatch(out_[idx].nhcw, ref.at(idx).nhcw, &first);
    EXPECT_EQ(mism, 0u) << "first mismatching byte " << first;
}

INSTANTIATE_TEST_SUITE_P(Frames, GstAdapter, ::testing::ValuesIn(kTargets));

// ── (b) element ─────────────────────────────────────────────────────────────────────────────
struct ElemFrame {
    yvo::gst::TensorFrameInfo info;
    bool has_meta = false;
    int mem_slot = -1;
    int mem_fd = -1;
    std::size_t mem_size = 0;
};

class GstElementPath : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (done_) return;
        done_ = true;
        if (!std::filesystem::exists(video_path()) || !have_hw_decoder()) return;
        std::string rerr;
        ffmpeg_reference(rerr);
        if (!rerr.empty()) {
            setup_error_ = rerr;
            return;
        }
        std::string err;
        const std::string desc = decode_desc(video_path()) +
                                 " ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! yvosandletterbox name=lb "
                                 "slots=" + std::to_string(kElementSlots) +
                                 " ! appsink name=sink sync=false max-buffers=2 drop=false enable-last-sample=false "
                                 "wait-on-eos=false";
        GstElement* lb = nullptr;
        const int n = run_appsink(
            desc, false,
            [&](int i, GstSample* s) {
                GstBuffer* b = gst_sample_get_buffer(s);
                if (i == 0) {
                    gchar* c = gst_caps_to_string(gst_sample_get_caps(s));
                    caps_ = c;
                    g_free(c);
                }
                ElemFrame ef;
                ef.has_meta = yvo::gst::tensor_meta_get(b, ef.info);
                ef.mem_slot = yvo::gst::tensor_buffer_slot(b);
                ef.mem_fd = yvo::gst::tensor_buffer_fd(b);
                ef.mem_size = gst_buffer_get_size(b);
                if (is_target(i) && ef.has_meta && ef.mem_fd >= 0) {
                    yvo::DmaBufReadMap m(ef.mem_fd, static_cast<std::size_t>(ef.info.dmabuf_bytes));
                    if (!m.data()) {
                        err = "mmap of slot dma-buf failed at frame " + std::to_string(i);
                        return false;
                    }
                    bytes_[i].assign(m.data(), m.data() + ef.info.tensor_bytes);
                }
                frames_.push_back(ef);
                return true;
            },
            err, &lb, "lb");
        if (lb) {
            guint64 ve = 0;
            g_object_get(lb, "validation-errors", &ve, nullptr);
            validation_errors_ = ve;
            gst_object_unref(lb);
        }
        if (n < 0) setup_error_ = err;
    }
    void SetUp() override {
        if (!std::filesystem::exists(video_path())) GTEST_SKIP() << "video not found: " << video_path();
        if (!have_hw_decoder()) GTEST_SKIP() << "v4l2slh265dec not available";
        ASSERT_TRUE(setup_error_.empty()) << setup_error_;
    }
    static bool done_;
    static std::string setup_error_;
    static std::string caps_;
    static std::vector<ElemFrame> frames_;
    static std::map<int, std::vector<std::uint8_t>> bytes_;
    static std::uint64_t validation_errors_;
};
bool GstElementPath::done_ = false;
std::string GstElementPath::setup_error_;
std::string GstElementPath::caps_;
std::vector<ElemFrame> GstElementPath::frames_;
std::map<int, std::vector<std::uint8_t>> GstElementPath::bytes_;
std::uint64_t GstElementPath::validation_errors_ = 0;

class GstElementPathFrame : public GstElementPath, public ::testing::WithParamInterface<int> {};

TEST_P(GstElementPathFrame, OutputMatchesDirectSandLetterbox) {
    const int idx = GetParam();
    std::string rerr;
    const auto& ref = ffmpeg_reference(rerr);
    ASSERT_TRUE(rerr.empty()) << rerr;
    ASSERT_TRUE(bytes_.count(idx)) << "element frame " << idx << " not collected";
    ASSERT_TRUE(ref.count(idx));
    std::size_t first = 0;
    const std::size_t mism = count_mismatch(bytes_[idx], ref.at(idx).nhcw, &first);
    EXPECT_EQ(mism, 0u) << "first mismatching byte " << first;
}

INSTANTIATE_TEST_SUITE_P(Frames, GstElementPathFrame, ::testing::ValuesIn(kTargets));

TEST_F(GstElementPath, TensorMetaAndSlots) {
    ASSERT_EQ(frames_.size(), static_cast<std::size_t>(kExpectFrames));
    EXPECT_NE(caps_.find("application/x-yvo-tensor"), std::string::npos) << caps_;
    EXPECT_NE(caps_.find("layout=(string)nhcw"), std::string::npos) << caps_;
    const yvo::LetterboxParams lb = yvo::make_letterbox(1920, 1080, 640, 640);
    std::map<int, std::set<int>> slot_fds;
    std::set<int> all_fds;
    for (std::size_t i = 0; i < frames_.size(); ++i) {
        const ElemFrame& f = frames_[i];
        ASSERT_TRUE(f.has_meta) << "frame " << i;
        const yvo::gst::TensorFrameInfo& t = f.info;
        ASSERT_EQ(t.seq, i) << "seq must increment by 1";
        ASSERT_GE(t.slot, 0);
        ASSERT_LT(t.slot, kElementSlots);
        ASSERT_EQ(t.num_slots, kElementSlots);
        ASSERT_EQ(f.mem_slot, t.slot) << "memory slot tag vs meta, frame " << i;
        ASSERT_EQ(f.mem_fd, t.dmabuf_fd) << "memory fd vs meta, frame " << i;
        ASSERT_EQ(f.mem_size, t.tensor_bytes);
        ASSERT_EQ(t.tensor_bytes, 640u * 640u * 3u);
        ASSERT_GE(t.dmabuf_bytes, t.tensor_bytes);
        ASSERT_EQ(t.src_width, 1920);
        ASSERT_EQ(t.src_height, 1080);
        ASSERT_EQ(t.lb.scale, lb.scale);
        ASSERT_EQ(t.lb.pad_x, lb.pad_x);
        ASSERT_EQ(t.lb.pad_y, lb.pad_y);
        ASSERT_EQ(t.lb.dst_w, 640);
        ASSERT_EQ(t.lb.dst_h, 640);
        ASSERT_EQ(t.layout, yvo::TensorLayout::Nhcw);
        ASSERT_GE(t.src_pts_ns, 0);
        ASSERT_GT(t.gpu_ms, 0.0);
        ASSERT_GT(t.arrival_mono_ns, 0u);
        ASSERT_GE(t.gpu_done_mono_ns, t.arrival_mono_ns);
        slot_fds[t.slot].insert(t.dmabuf_fd);
        all_fds.insert(t.dmabuf_fd);
    }
    EXPECT_EQ(slot_fds.size(), static_cast<std::size_t>(kElementSlots)) << "every slot should be used";
    for (const auto& kv : slot_fds) EXPECT_EQ(kv.second.size(), 1u) << "slot " << kv.first << " fd changed";
    EXPECT_EQ(all_fds.size(), static_cast<std::size_t>(kElementSlots)) << "slots must not share fds";
    EXPECT_EQ(validation_errors_, 0u);
}

// ── caps: no silent CPU fallback ────────────────────────────────────────────────────────────
TEST(GstElementCaps, AcceptsOnlySandDmaDrm) {
    ensure_gst();
    GstElement* e = gst_element_factory_make("yvosandletterbox", nullptr);
    ASSERT_NE(e, nullptr) << "yvosandletterbox not registered";
    GstPad* sink = gst_element_get_static_pad(e, "sink");
    auto accepts = [&](const char* s) {
        GstCaps* c = gst_caps_from_string(s);
        const bool ok = gst_pad_query_accept_caps(sink, c) != FALSE;
        gst_caps_unref(c);
        return ok;
    };
    EXPECT_TRUE(accepts("video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12:0x0700000000000004,"
                        "width=1920,height=1080,framerate=30000/1001"));
    EXPECT_FALSE(accepts("video/x-raw,format=NV12,width=1920,height=1080"));
    EXPECT_FALSE(accepts("video/x-raw,format=NV12_128C8,width=1920,height=1080"));
    EXPECT_FALSE(accepts("video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=NV12,width=1920,height=1080"));
    EXPECT_FALSE(accepts("video/x-raw(memory:DMABuf),format=NV12,width=1920,height=1080"));
    gst_object_unref(sink);

    // A system-memory source cannot link at all (parse_launch reports it; no pipeline is run)
    GError* perr = nullptr;
    GstElement* p = gst_parse_launch(
        "videotestsrc num-buffers=1 ! video/x-raw,format=NV12,width=64,height=64 ! yvosandletterbox ! fakesink", &perr);
    EXPECT_NE(perr, nullptr) << "linking a system-memory NV12 source must fail";
    if (perr) g_error_free(perr);
    if (p) gst_object_unref(p);
    gst_object_unref(e);
}

// ── slot lifetime (review S1/S2 regression) ─────────────────────────────────────────────────
class GstElementSlots : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(video_path())) GTEST_SKIP() << "video not found: " << video_path();
        if (!have_hw_decoder()) GTEST_SKIP() << "v4l2slh265dec not available";
    }
    static std::string element_desc(const char* sink_extra) {
        return decode_desc(video_path()) +
               " ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! yvosandletterbox name=lb slots=" +
               std::to_string(kElementSlots) + " ! appsink name=sink sync=false max-buffers=1 drop=false "
               "enable-last-sample=false wait-on-eos=false " + sink_extra;
    }
};

// A consumer moves each slot memory into its own buffer and keeps it. Every slot memory stays alive, so the element
// must not wrap/overwrite any of them again: it delivers at most kElementSlots frames and then posts an explicit error.
TEST_F(GstElementSlots, TransplantedMemoryIsNeverOverwritten) {
    const long fds0 = yvo::tool::count_open_fds();
    GError* perr = nullptr;
    GstElement* pipeline = gst_parse_launch(element_desc("").c_str(), &perr);
    ASSERT_TRUE(pipeline && !perr) << (perr ? perr->message : "?");
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    GstElement* lb = gst_bin_get_by_name(GST_BIN(pipeline), "lb");
    ASSERT_NE(gst_element_set_state(pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);

    std::vector<GstBuffer*> kept;                 // foreign buffers holding the slot memories
    std::vector<std::vector<std::uint8_t>> snap;  // tensor bytes at receive time
    std::set<int> slots;
    while (true) {
        GstSample* smp = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 4 * GST_SECOND);
        if (!smp) break;
        GstBuffer* b = gst_sample_get_buffer(smp);
        GstMemory* mem = gst_buffer_peek_memory(b, 0);
        slots.insert(yvo::gst::tensor_buffer_slot(b));
        GstBuffer* other = gst_buffer_new();
        gst_buffer_append_memory(other, gst_memory_ref(mem));
        kept.push_back(other);
        GstMapInfo mi;
        EXPECT_TRUE(gst_memory_map(mem, &mi, GST_MAP_READ));
        snap.emplace_back(mi.data, mi.data + mi.size);
        gst_memory_unmap(mem, &mi);
        gst_sample_unref(smp);
        if (kept.size() > static_cast<std::size_t>(kElementSlots) + 2) break;
    }
    guint detached = 0;
    g_object_get(lb, "slots-detached", &detached, nullptr);
    const std::string errs = drain_bus_errors(pipeline);

    EXPECT_EQ(kept.size(), static_cast<std::size_t>(kElementSlots)) << "a slot was re-used while its memory was held";
    EXPECT_EQ(slots.size(), static_cast<std::size_t>(kElementSlots));
    EXPECT_GE(detached, static_cast<guint>(kElementSlots - 1));
    EXPECT_NE(errs.find("slot memory still referenced downstream"), std::string::npos) << "bus: " << errs;
    // The kept memories were not overwritten (the element stopped instead of re-using their slots)
    for (std::size_t i = 0; i < kept.size(); ++i) {
        GstMapInfo mi;
        GstMemory* mem = gst_buffer_peek_memory(kept[i], 0);
        ASSERT_TRUE(gst_memory_map(mem, &mi, GST_MAP_READ));
        EXPECT_TRUE(mi.size == snap[i].size() && std::equal(snap[i].begin(), snap[i].end(), mi.data))
            << "kept slot memory " << i << " changed";
        gst_memory_unmap(mem, &mi);
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(lb);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    const long fds_held = yvo::tool::count_open_fds();
    EXPECT_GT(fds_held, fds0) << "slot fds must stay open while their memories are alive";
    for (GstBuffer* b : kept) gst_buffer_unref(b); // last slot memories → deferred Resources teardown here
    EXPECT_LE(yvo::tool::count_open_fds(), fds0) << "slot / Vulkan fds leaked after the last memory died";
}

GstPadProbeReturn replace_memory_probe(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user) {
    GstBuffer* b = gst_buffer_make_writable(GST_PAD_PROBE_INFO_BUFFER(info));
    if (yvo::gst::tensor_buffer_own_slot(b) >= 0) ++*static_cast<int*>(user); // still the pool's slot buffer
    gst_buffer_replace_all_memory(b, gst_allocator_alloc(nullptr, 16, nullptr));
    GST_PAD_PROBE_INFO_DATA(info) = b;
    return GST_PAD_PROBE_OK;
}

// Downstream replaces the slot buffer's memory: the buffer returns to the pool with foreign memory and is discarded,
// the slot memory dies → the slot must be re-usable (no leak: the whole file runs through 4 slots).
TEST_F(GstElementSlots, ReplacedMemoryDoesNotLeakSlots) {
    GError* perr = nullptr;
    GstElement* pipeline = gst_parse_launch(element_desc("").c_str(), &perr);
    ASSERT_TRUE(pipeline && !perr) << (perr ? perr->message : "?");
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    GstElement* lb = gst_bin_get_by_name(GST_BIN(pipeline), "lb");
    int replaced_slot_buffers = 0;
    GstPad* pad = gst_element_get_static_pad(sink, "sink");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, replace_memory_probe, &replaced_slot_buffers, nullptr);
    gst_object_unref(pad);
    ASSERT_NE(gst_element_set_state(pipeline, GST_STATE_PLAYING), GST_STATE_CHANGE_FAILURE);
    int frames = 0;
    while (GstSample* smp = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), kPullTimeout)) {
        ++frames;
        gst_sample_unref(smp);
    }
    const bool eos = gst_app_sink_is_eos(GST_APP_SINK(sink));
    guint detached = 0, in_use = 0; // at EOS, before NULL (appsink still caches the last buffer → in_use <= 1)
    g_object_get(lb, "slots-detached", &detached, "slots-in-use", &in_use, nullptr);
    const std::string errs = drain_bus_errors(pipeline);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(lb);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    EXPECT_TRUE(eos) << "bus: " << errs;
    EXPECT_TRUE(errs.empty()) << errs;
    EXPECT_EQ(frames, kExpectFrames);
    EXPECT_EQ(replaced_slot_buffers, frames) << "probe did not operate on the pool's slot buffers";
    EXPECT_EQ(detached, 0u);
    EXPECT_LE(in_use, 1u);
}

} // namespace
