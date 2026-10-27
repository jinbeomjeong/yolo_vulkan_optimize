// GStreamer research path (stage 2b): yvohailosink detections vs the yvo_pipeline inference path (needs Hailo-8)
//
//   reference : FfmpegDrmDecoder → SandLetterbox (NHCW, exported dma-buf per slot) → HailoRunner (NHCW, batch B,
//               map_dmabuf(fd, input_frame_size) per slot, HEF NMS on the host) — the calls yvo_pipeline makes with
//               --mode dmabuf --layout nhcw --input-pace asap (HostNmsStage), slots = max(4, 2*B) like its asap default.
//   gstreamer : filesrc ! qtdemux ! h265parse ! v4l2slh265dec ! queue ! yvosandletterbox ! queue ! yvohailosink batch=B
//               (the yvo_gst_pipeline --sink hailo chain), results collected in seq order by the sink's callback.
// Criterion: for every frame the detection lists are EQUAL (same order, class, score and box floats) — the NHCW
// tensor bytes of both paths are identical (stage 1/2a gates) and the NPU is deterministic. The IoU match ratio
// (ref/detection_match.hpp) is printed as well and must be 1.0.
// Env: YVO_GST_DET_FRAMES (default 120), YVO_GST_DET_BATCH (default 4), YVO_TEST_VIDEO.
// Skips: no /dev/hailo0, no video, no v4l2slh265dec, or the Hailo device held by another process.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include <gst/gst.h>

#include "core/frame_types.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "gstelem/hailo_infer_sink.hpp"
#include "gstelem/register.hpp"
#include "hailo/hailo_runner.hpp"
#include "ref/detection_match.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

namespace {

using Dets = std::vector<yvo::Detection>;

std::string env_str(const char* k, const char* def) {
    const char* v = std::getenv(k);
    return v && *v ? v : def;
}
long env_long(const char* k, long def) {
    const char* v = std::getenv(k);
    return v && *v ? std::atol(v) : def;
}

bool device_busy(hailo_status s) { return s == HAILO_DEVICE_IN_USE || s == HAILO_OUT_OF_PHYSICAL_DEVICES; }

bool same(const yvo::Detection& a, const yvo::Detection& b) {
    return a.class_id == b.class_id && a.score == b.score && a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 &&
           a.y2 == b.y2;
}

class GstDetections : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        video_ = env_str("YVO_TEST_VIDEO", YVO_TEST_VIDEO);
        frames_want_ = std::max(1L, env_long("YVO_GST_DET_FRAMES", 120));
        batch_ = static_cast<int>(std::clamp(env_long("YVO_GST_DET_BATCH", 4), 1L, 16L));
        if (!std::filesystem::exists("/dev/hailo0")) {
            skip_ = "/dev/hailo0 not found — environment without Hailo-8";
            return;
        }
        if (!std::filesystem::exists(video_)) {
            skip_ = "video not found: " + video_;
            return;
        }
        gst_init(nullptr, nullptr);
        yvo::gst::register_elements();
        GstElementFactory* f = gst_element_factory_find("v4l2slh265dec");
        if (!f) {
            skip_ = "v4l2slh265dec not available";
            return;
        }
        gst_object_unref(f);
        // One Hailo device: the reference runner is fully destroyed before the GStreamer pass opens its own.
        run_reference();
        if (skip_.empty() && error_.empty()) run_gstreamer();
    }

    // yvo_pipeline --mode dmabuf --layout nhcw --batch B --input-pace asap (slots max(4, 2B)), pipelined over the ring like the app.
    static void run_reference() {
        const int slots = std::max(4, 2 * batch_);
        std::vector<int> fds;
        try {
            yvo::vk::VulkanContext ctx;
            if (!ctx.dmabuf_export_supported()) {
                error_ = "Vulkan dma-buf export not supported";
                return;
            }
            yvo::vk::SandLetterboxConfig pcfg;
            pcfg.num_slots = slots;
            pcfg.export_dmabuf = true;
            pcfg.out_layout = yvo::TensorLayout::Nhcw;
            yvo::vk::SandLetterbox pre(ctx, pcfg);

            yvo::HailoRunnerConfig hcfg;
            hcfg.hef_path = YVO_TEST_HEF;
            hcfg.input_layout = yvo::TensorLayout::Nhcw;
            hcfg.batch_size = static_cast<std::uint16_t>(batch_);
            hcfg.num_slots = slots;
            std::unique_ptr<yvo::HailoRunner> runner;
            try {
                runner = std::make_unique<yvo::HailoRunner>(hcfg);
            } catch (const yvo::HailoError& e) {
                if (device_busy(e.status())) skip_ = std::string("Hailo device busy: ") + e.what();
                else error_ = std::string("HailoRunner: ") + e.what();
                return;
            }
            if (runner->input_frame_size() != pre.output_size()) {
                error_ = "input size mismatch";
                return;
            }
            for (int s = 0; s < slots; ++s) {
                fds.push_back(pre.export_dmabuf_fd(s));
                runner->map_dmabuf(fds.back(), runner->input_frame_size());  // = HostNmsStage::bind_input_dmabuf
            }
            yvo::FfmpegDrmDecoder dec;
            if (!dec.open(video_)) {
                error_ = "decoder open failed: " + dec.error();
                return;
            }
            std::mutex m;
            std::condition_variable cv;
            std::vector<bool> busy(static_cast<std::size_t>(slots), false);
            ref_.assign(static_cast<std::size_t>(frames_want_), {});
            long n = 0;
            for (; n < frames_want_; ++n) {
                auto f = dec.next();
                if (!f) break;
                f->pts = n;
                const int slot = static_cast<int>(n % slots);
                {
                    std::unique_lock<std::mutex> lk(m);
                    if (!cv.wait_for(lk, std::chrono::seconds(10), [&] { return !busy[static_cast<std::size_t>(slot)]; })) {
                        error_ = "reference: slot wait timeout at frame " + std::to_string(n);
                        break;
                    }
                    busy[static_cast<std::size_t>(slot)] = true;
                }
                yvo::PreprocOutput out = pre.process(*f, slot);
                f.reset();
                out.pts = n;
                out.slot = slot;
                out.dmabuf_fd = fds[static_cast<std::size_t>(slot)];
                runner->submit(out, [&, n, slot](yvo::InferResult&& r) {
                    std::lock_guard<std::mutex> lk(m);
                    if (r.status != HAILO_SUCCESS) ++ref_errors_;
                    ref_[static_cast<std::size_t>(n)] = std::move(r.detections);
                    busy[static_cast<std::size_t>(slot)] = false;
                    cv.notify_all();
                });
            }
            runner->drain();
            ref_frames_ = n;
            ref_.resize(static_cast<std::size_t>(n));
            pre.clear_import_cache();
            dec.close();
            runner.reset();  // dma-buf unmap before the fds are closed
        } catch (const std::exception& e) {
            error_ = std::string("reference exception: ") + e.what();
        }
        for (int fd : fds)
            if (fd >= 0) ::close(fd);
    }

    static void run_gstreamer() {
        const int slots = std::max(4, 2 * batch_);  // yvo_gst_pipeline default for the hailo sink (= reference)
        const std::string desc =
            "filesrc location=\"" + video_ + "\" ! qtdemux ! h265parse ! v4l2slh265dec ! "
            "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! yvosandletterbox slots=" + std::to_string(slots) +
            " ! queue max-size-buffers=" + std::to_string(slots) + " max-size-bytes=0 max-size-time=0 ! "
            "yvohailosink name=sink hef=\"" YVO_TEST_HEF "\" batch=" + std::to_string(batch_);
        GError* perr = nullptr;
        GstElement* pipeline = gst_parse_launch(desc.c_str(), &perr);
        if (!pipeline || perr) {
            error_ = std::string("gst_parse_launch: ") + (perr ? perr->message : "?");
            if (perr) g_error_free(perr);
            if (pipeline) gst_object_unref(pipeline);
            return;
        }
        GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
        std::mutex m;
        yvo::gst::hailo_sink_set_result_callback(sink, [&](const yvo::gst::HailoSinkFrameResult& r) {
            std::lock_guard<std::mutex> lk(m);
            if (r.status != 0) ++gst_errors_;
            gst_order_.push_back(r.seq);
            gst_[r.seq] = r.detections;
        });
        // Frame limit: buffers beyond N are dropped before the sink and EOS is sent from the main thread.
        struct Limit {
            long n;
            std::atomic<long> seen{0};
        } limit{frames_want_};
        GstPad* pad = gst_element_get_static_pad(sink, "sink");
        gst_pad_add_probe(
            pad, GST_PAD_PROBE_TYPE_BUFFER,
            [](GstPad*, GstPadProbeInfo*, gpointer u) -> GstPadProbeReturn {
                auto* l = static_cast<Limit*>(u);
                return l->seen.fetch_add(1) < l->n ? GST_PAD_PROBE_OK : GST_PAD_PROBE_DROP;
            },
            &limit, nullptr);
        gst_object_unref(pad);

        GstBus* bus = gst_element_get_bus(pipeline);
        bool eos_sent = false, done = false;
        if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            error_ = "set_state(PLAYING) failed";
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (error_.empty() && !done) {
            GstMessage* msg = gst_bus_timed_pop_filtered(
                bus, 100 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            if (msg) {
                if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
                    done = true;
                } else {
                    GError* err = nullptr;
                    gchar* dbg = nullptr;
                    gst_message_parse_error(msg, &err, &dbg);
                    error_ = std::string("bus error from ") + GST_OBJECT_NAME(GST_MESSAGE_SRC(msg)) + ": " +
                             (err ? err->message : "?") + " / " + (dbg ? dbg : "");
                    if (err) g_error_free(err);
                    g_free(dbg);
                }
                gst_message_unref(msg);
            }
            if (!eos_sent && limit.seen.load() >= limit.n) {
                gst_element_send_event(pipeline, gst_event_new_eos());
                eos_sent = true;
            }
            if (!done && std::chrono::steady_clock::now() > deadline) error_ = "GStreamer pass: no EOS within 120 s";
        }
        gst_element_set_state(pipeline, GST_STATE_NULL);
        yvo::gst::HailoSinkStats st;
        if (yvo::gst::hailo_sink_get_stats(sink, st)) {
            gst_stats_ok_ = true;
            gst_submitted_ = st.submitted;
            gst_drain_timeout_ = st.drain_timeout;
            gst_mapped_ = st.mapped_slots;
        }
        yvo::gst::hailo_sink_set_result_callback(sink, {});
        gst_object_unref(bus);
        gst_object_unref(sink);
        gst_object_unref(pipeline);
    }

    void SetUp() override {
        if (!skip_.empty()) GTEST_SKIP() << skip_;
        ASSERT_TRUE(error_.empty()) << error_;
    }

    static inline std::string video_;
    static inline long frames_want_ = 0;
    static inline int batch_ = 1;
    static inline std::string skip_;
    static inline std::string error_;
    static inline std::vector<Dets> ref_;
    static inline long ref_frames_ = 0;
    static inline long ref_errors_ = 0;
    static inline std::map<std::uint64_t, Dets> gst_;
    static inline std::vector<std::uint64_t> gst_order_;
    static inline long gst_errors_ = 0;
    static inline bool gst_stats_ok_ = false;
    static inline std::uint64_t gst_submitted_ = 0;
    static inline bool gst_drain_timeout_ = false;
    static inline int gst_mapped_ = 0;
};

TEST_F(GstDetections, SinkDeliversEveryFrameInOrder) {
    EXPECT_EQ(ref_frames_, frames_want_);
    ASSERT_TRUE(gst_stats_ok_);
    EXPECT_EQ(gst_submitted_, static_cast<std::uint64_t>(frames_want_));
    EXPECT_FALSE(gst_drain_timeout_);
    EXPECT_GT(gst_mapped_, 0);
    ASSERT_EQ(gst_order_.size(), static_cast<std::size_t>(frames_want_));
    for (std::size_t i = 0; i < gst_order_.size(); ++i) ASSERT_EQ(gst_order_[i], i) << "results not in seq order";
    EXPECT_EQ(ref_errors_, 0);
    EXPECT_EQ(gst_errors_, 0);
}

TEST_F(GstDetections, DetectionsEqualAppPath) {
    ASSERT_GT(ref_frames_, 0);
    long ref_total = 0, gst_total = 0, differing_frames = 0, printed = 0;
    yvo::detmatch::MatchResult total;
    for (long n = 0; n < ref_frames_; ++n) {
        const Dets& r = ref_[static_cast<std::size_t>(n)];
        auto it = gst_.find(static_cast<std::uint64_t>(n));
        ASSERT_NE(it, gst_.end()) << "GStreamer result missing for frame " << n;
        const Dets& g = it->second;
        ref_total += static_cast<long>(r.size());
        gst_total += static_cast<long>(g.size());
        bool eq = r.size() == g.size();
        for (std::size_t i = 0; eq && i < r.size(); ++i) eq = same(r[i], g[i]);
        if (!eq) {
            ++differing_frames;
            if (printed++ < 10)
                std::printf("[gst_det] frame %ld differs: app %zu dets, gst %zu dets\n", n, r.size(), g.size());
        }
        const auto m = yvo::detmatch::match(r, g, nullptr);
        total.ref += m.ref;
        total.test += m.test;
        total.matched += m.matched;
    }
    const long denom = std::max(total.ref, total.test);
    const double ratio = denom > 0 ? static_cast<double>(total.matched) / static_cast<double>(denom) : 1.0;
    std::printf("[gst_det] batch %d, frames %ld, detections app %ld / gst %ld, identical frames %ld/%ld, IoU match %.4f\n",
                batch_, ref_frames_, ref_total, gst_total, ref_frames_ - differing_frames, ref_frames_, ratio);
    EXPECT_GT(ref_total, 0) << "app path produced no detections";
    EXPECT_EQ(ref_total, gst_total);
    EXPECT_EQ(differing_frames, 0);
    EXPECT_EQ(ratio, 1.0);
}

} // namespace
