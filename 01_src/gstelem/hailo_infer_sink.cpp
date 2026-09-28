#include "gstelem/hailo_infer_sink.hpp"

#include <gst/base/gstbasesink.h>
#include <gst/gst.h>

#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "core/coco_labels.hpp"
#include "core/frame_types.hpp"
#include "gstelem/tensor_meta.hpp"
#include "hailo/hailo_runner.hpp"

GST_DEBUG_CATEGORY_STATIC(yvo_hailo_sink_debug);
#define GST_CAT_DEFAULT yvo_hailo_sink_debug

#define YVO_TENSOR_SINK_CAPS \
    "application/x-yvo-tensor, layout=(string)nhcw, format=(string)RGB8, width=(int)640, height=(int)640"

namespace {

constexpr guint kDefaultBatch = 1;
constexpr guint kMaxBatch = 16;
constexpr guint kDefaultDrainTimeoutMs = 10000;

std::uint64_t mono_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<std::uint64_t>(ts.tv_nsec);
}

// One HailoRT completion, queued by the callback thread and ordered by the streaming thread.
struct Done {
    std::uint64_t seq = 0;
    int status = 0;
    std::vector<yvo::Detection> detections;
    double npu_ms = 0.0;
    double e2e_ms = -1.0;
};

// Shared with every in-flight completion callback (captured by value), so it outlives a leaked runner.
struct Shared {
    std::mutex m;
    std::condition_variable cv;
    int in_flight = 0;
    int in_flight_max = 0;
    std::deque<Done> done;
};

struct Impl {
    // properties (GST_OBJECT_LOCK)
    std::string hef;
    guint batch = kDefaultBatch;
    float score = -1.0f;
    std::string dump_dir;
    guint drain_timeout_ms = kDefaultDrainTimeoutMs;

    // runtime: start() → stop()
    std::unique_ptr<yvo::HailoRunner> runner;
    std::shared_ptr<Shared> sh;
    std::vector<int> mapped_fd = std::vector<int>(yvo::gst::kHailoSinkRunnerSlots, -1);  // streaming thread only

    // ordered output (out_m): submit order, results waiting for their turn, jsonl
    std::mutex out_m;
    std::deque<std::uint64_t> order;
    std::map<std::uint64_t, Done> reorder;
    std::FILE* jsonl = nullptr;

    // stats + result callback (stats_m)
    std::mutex stats_m;
    yvo::gst::HailoSinkStats stats;
    yvo::gst::HailoSinkResultCallback cb;
};

} // namespace

struct YvoHailoSink {
    GstBaseSink parent;
    Impl* impl;
};
struct YvoHailoSinkClass {
    GstBaseSinkClass parent_class;
};

G_DEFINE_TYPE(YvoHailoSink, yvo_hailo_sink, GST_TYPE_BASE_SINK)

enum {
    PROP_0,
    PROP_HEF,
    PROP_BATCH,
    PROP_SCORE,
    PROP_DUMP_DIR,
    PROP_DRAIN_TIMEOUT,
    PROP_FRAMES,
    PROP_SUBMITTED,
    PROP_ERRORS,
    PROP_DETECTIONS,
    PROP_IN_FLIGHT_MAX,
    PROP_MAPPED_SLOTS,
    PROP_NPU_MS_MEAN,
    PROP_E2E_MS_MEAN,
};

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(YVO_TENSOR_SINK_CAPS));

static Impl& impl_of(gpointer p) { return *reinterpret_cast<YvoHailoSink*>(p)->impl; }

// Same line format as yvo_pipeline --dump-dir (pipeline/main.cpp flush_results) → compare_detections.py works on both.
static void write_jsonl(std::FILE* f, std::uint64_t frame, const std::vector<yvo::Detection>& dets) {
    std::fprintf(f, "{\"frame\":%lld,\"detections\":[", static_cast<long long>(frame));
    for (std::size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        std::fprintf(f, "%s{\"class\":%d,\"name\":\"%s\",\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}", i ? "," : "",
                     d.class_id, yvo::class_name(d.class_id), static_cast<double>(d.score), static_cast<double>(d.x1),
                     static_cast<double>(d.y1), static_cast<double>(d.x2), static_cast<double>(d.y2));
    }
    std::fprintf(f, "]}\n");
}

// Moves finished completions into seq order and emits every result whose turn has come (submit order). With final,
// also emits results behind a frame that never completed (drain timeout). Streaming thread, EOS or stop().
static void flush_results(Impl& im, bool final) {
    std::deque<Done> got;
    if (im.sh) {
        std::lock_guard<std::mutex> lk(im.sh->m);
        got.swap(im.sh->done);
    }
    std::lock_guard<std::mutex> lk(im.out_m);
    for (auto& d : got) {
        const std::uint64_t s = d.seq;
        im.reorder.emplace(s, std::move(d));
    }
    while (!im.order.empty()) {
        auto it = im.reorder.find(im.order.front());
        if (it == im.reorder.end()) {
            if (!final) break;
            im.order.pop_front();  // never completed (only after a drain timeout)
            continue;
        }
        Done d = std::move(it->second);
        im.reorder.erase(it);
        im.order.pop_front();
        if (im.jsonl) write_jsonl(im.jsonl, d.seq, d.detections);
        yvo::gst::HailoSinkResultCallback cb;
        yvo::gst::HailoSinkFrameResult r;
        {
            std::lock_guard<std::mutex> sl(im.stats_m);
            auto& st = im.stats;
            ++st.completed;
            if (d.status != HAILO_SUCCESS) ++st.errors;
            st.npu_ms.push_back(d.npu_ms);
            if (d.e2e_ms >= 0.0) st.e2e_ms.push_back(d.e2e_ms);
            st.detections += d.detections.size();
            cb = im.cb;
        }
        if (d.status != HAILO_SUCCESS)
            GST_ERROR("frame %llu: HailoRT status %d", static_cast<unsigned long long>(d.seq), d.status);
        if (cb) {
            r.seq = d.seq;
            r.status = d.status;
            r.detections = std::move(d.detections);
            r.npu_ms = d.npu_ms;
            r.e2e_ms = d.e2e_ms;
            cb(r);
        }
    }
    if (final && im.jsonl) std::fflush(im.jsonl);
}

// Waits until no inference is in flight. false on deadline.
static bool drain_wait(Impl& im, guint timeout_ms) {
    if (!im.sh) return true;
    std::unique_lock<std::mutex> lk(im.sh->m);
    return im.sh->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return im.sh->in_flight == 0; });
}

static gboolean yvo_hailo_sink_start(GstBaseSink* bs) {
    Impl& im = impl_of(bs);
    GST_OBJECT_LOCK(bs);
    const std::string hef = im.hef;
    const guint batch = im.batch;
    const float score = im.score;
    const std::string dump_dir = im.dump_dir;
    GST_OBJECT_UNLOCK(bs);
    if (hef.empty()) {
        GST_ELEMENT_ERROR(bs, RESOURCE, NOT_FOUND, ("property \"hef\" is not set"), (nullptr));
        return FALSE;
    }
    {
        std::lock_guard<std::mutex> sl(im.stats_m);
        im.stats = yvo::gst::HailoSinkStats{};
        im.stats.batch = static_cast<int>(batch);
        im.stats.hailo_t0 = im.stats.hailo_t1 = std::nanf("");
    }
    {
        std::lock_guard<std::mutex> lk(im.out_m);
        im.order.clear();
        im.reorder.clear();
    }
    im.mapped_fd.assign(yvo::gst::kHailoSinkRunnerSlots, -1);
    {
        auto sh = std::make_shared<Shared>();
        GST_OBJECT_LOCK(bs);
        im.sh = std::move(sh);
        GST_OBJECT_UNLOCK(bs);
    }

    yvo::HailoRunnerConfig c;
    c.hef_path = hef;
    c.input_layout = yvo::TensorLayout::Nhcw;  // = yvo_pipeline --layout nhcw (dma-buf input only)
    c.batch_size = static_cast<std::uint16_t>(batch);
    c.num_slots = yvo::gst::kHailoSinkRunnerSlots;
    c.score_threshold = score;
    try {
        im.runner = std::make_unique<yvo::HailoRunner>(c);
    } catch (const std::exception& e) {
        GST_ELEMENT_ERROR(bs, LIBRARY, INIT, ("HailoRT initialisation failed"), ("%s", e.what()));
        GST_OBJECT_LOCK(bs);
        im.sh.reset();
        GST_OBJECT_UNLOCK(bs);
        return FALSE;
    }
    {
        std::lock_guard<std::mutex> sl(im.stats_m);
        im.stats.hailo_t0 = im.runner->chip_temperature();
        im.stats.async_queue_size = im.runner->async_queue_size();
    }
    if (!dump_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        const std::string p = dump_dir + "/detections.jsonl";
        std::lock_guard<std::mutex> lk(im.out_m);
        im.jsonl = std::fopen(p.c_str(), "w");
        if (!im.jsonl) {
            GST_ELEMENT_ERROR(bs, RESOURCE, OPEN_WRITE, ("cannot open %s", p.c_str()), (nullptr));
            im.runner.reset();
            GST_OBJECT_LOCK(bs);
            im.sh.reset();
            GST_OBJECT_UNLOCK(bs);
            return FALSE;
        }
    }
    GST_INFO_OBJECT(bs, "HailoRunner ready: hef %s batch %u score %.3f async queue %zu input %zu B", hef.c_str(), batch,
                    static_cast<double>(score), im.runner->async_queue_size(), im.runner->input_frame_size());
    return TRUE;
}

static gboolean yvo_hailo_sink_stop(GstBaseSink* bs) {
    Impl& im = impl_of(bs);
    GST_OBJECT_LOCK(bs);
    const guint timeout_ms = im.drain_timeout_ms;
    GST_OBJECT_UNLOCK(bs);
    const bool drained = drain_wait(im, timeout_ms);
    flush_results(im, true);
    if (im.runner) {
        const float t1 = im.runner->chip_temperature();
        std::lock_guard<std::mutex> sl(im.stats_m);
        im.stats.hailo_t1 = t1;
        if (!drained) im.stats.drain_timeout = true;
    }
    if (drained) {
        // shutdown + dma_unmap_dmabuf of every mapped slot fd; the upstream element still owns the fds here
        im.runner.reset();
    } else {
        GST_ERROR_OBJECT(bs, "stop: inferences still in flight after %u ms — HailoRunner leaked (its destructor would "
                         "wait forever), their slot buffers stay referenced", timeout_ms);
        (void)im.runner.release();
    }
    im.mapped_fd.assign(yvo::gst::kHailoSinkRunnerSlots, -1);
    if (im.sh) {
        int ifm = 0;
        {
            std::lock_guard<std::mutex> lk(im.sh->m);
            ifm = im.sh->in_flight_max;
        }
        std::lock_guard<std::mutex> sl(im.stats_m);
        im.stats.in_flight_max = std::max(im.stats.in_flight_max, ifm);
    }
    GST_OBJECT_LOCK(bs);
    im.sh.reset();  // in-flight callbacks (leak case) keep their own reference
    GST_OBJECT_UNLOCK(bs);
    std::lock_guard<std::mutex> lk(im.out_m);
    if (im.jsonl) {
        std::fclose(im.jsonl);
        im.jsonl = nullptr;
    }
    return TRUE;
}

static GstFlowReturn yvo_hailo_sink_render(GstBaseSink* bs, GstBuffer* buf) {
    Impl& im = impl_of(bs);
    if (!im.runner || !im.sh) {
        GST_ELEMENT_ERROR(bs, CORE, STATE_CHANGE, ("buffer before start / after stop"), (nullptr));
        return GST_FLOW_ERROR;
    }
    yvo::gst::TensorFrameInfo ti;
    if (!yvo::gst::tensor_meta_get(buf, ti)) {
        GST_ELEMENT_ERROR(bs, STREAM, FORMAT, ("buffer without YvoTensorMeta (not a yvosandletterbox slot buffer)"),
                          (nullptr));
        return GST_FLOW_ERROR;
    }
    const int slot = yvo::gst::tensor_buffer_slot(buf);
    const int fd = yvo::gst::tensor_buffer_fd(buf);
    const std::size_t in_size = im.runner->input_frame_size();
    if (ti.layout != yvo::TensorLayout::Nhcw || ti.tensor_bytes != in_size || ti.dmabuf_bytes < ti.tensor_bytes ||
        slot != ti.slot || slot < 0 || slot >= yvo::gst::kHailoSinkRunnerSlots || fd < 0 || fd != ti.dmabuf_fd) {
        GST_ELEMENT_ERROR(bs, STREAM, FORMAT, ("tensor buffer does not match the HailoRT input"),
                          ("frame %llu: layout %s, tensor %u B (HEF input %zu B), dma-buf %llu B, slot %d/%d (meta %d), "
                           "fd %d (meta %d)",
                           static_cast<unsigned long long>(ti.seq), ti.layout == yvo::TensorLayout::Nhcw ? "nhcw" : "nhwc",
                           ti.tensor_bytes, in_size, static_cast<unsigned long long>(ti.dmabuf_bytes), slot,
                           yvo::gst::kHailoSinkRunnerSlots, ti.slot, fd, ti.dmabuf_fd));
        return GST_FLOW_ERROR;
    }
    // Map each slot fd once. Size = HailoRT transfer size (input_frame_size), exactly like HostNmsStage::bind_input_dmabuf.
    int& mfd = im.mapped_fd[static_cast<std::size_t>(slot)];
    if (mfd < 0) {
        for (int other : im.mapped_fd) {
            if (other == fd) {
                GST_ELEMENT_ERROR(bs, STREAM, FORMAT, ("slot dma-buf fd already mapped for another slot"),
                                  ("slot %d fd %d", slot, fd));
                return GST_FLOW_ERROR;
            }
        }
        try {
            im.runner->map_dmabuf(fd, in_size);
        } catch (const std::exception& e) {
            GST_ELEMENT_ERROR(bs, LIBRARY, FAILED, ("dma_map_dmabuf failed"), ("slot %d fd %d: %s", slot, fd, e.what()));
            return GST_FLOW_ERROR;
        }
        mfd = fd;
        std::lock_guard<std::mutex> sl(im.stats_m);
        ++im.stats.mapped_slots;
        im.stats.ring_slots = ti.num_slots;
    } else if (mfd != fd) {
        // The element contract keeps the fd per slot stable from start() to stop(); HailoRunner cannot unmap one fd.
        GST_ELEMENT_ERROR(bs, STREAM, FORMAT, ("slot dma-buf fd changed while mapped"),
                          ("slot %d: mapped fd %d, buffer fd %d", slot, mfd, fd));
        return GST_FLOW_ERROR;
    }
    if (ti.gpu_ms >= 0.0) {
        std::lock_guard<std::mutex> sl(im.stats_m);
        im.stats.gpu_ms.push_back(ti.gpu_ms);
    }

    yvo::PreprocOutput in;
    in.dmabuf_fd = fd;
    in.size = in_size;  // NOT the dma-buf size: a transfer of the full dma-buf size never completes
    in.layout = yvo::TensorLayout::Nhcw;
    in.lb = ti.lb;
    in.pts = static_cast<std::int64_t>(ti.seq);
    in.slot = slot;

    std::shared_ptr<Shared> sh = im.sh;
    GstBuffer* held = gst_buffer_ref(buf);  // held until the completion callback (slot stays out of the ring)
    const std::uint64_t seq = ti.seq;
    const std::uint64_t arrival = ti.arrival_mono_ns;
    {
        std::lock_guard<std::mutex> lk(sh->m);
        ++sh->in_flight;
        if (sh->in_flight > sh->in_flight_max) sh->in_flight_max = sh->in_flight;
    }
    {
        std::lock_guard<std::mutex> lk(im.out_m);
        im.order.push_back(seq);
    }
    try {
        im.runner->submit(in, [sh, held, seq, arrival](yvo::InferResult&& r) {
            const std::uint64_t now = mono_ns();
            Done d;
            d.seq = seq;
            d.status = static_cast<int>(r.status);
            d.detections = std::move(r.detections);
            d.npu_ms = r.latency_ms;
            d.e2e_ms = arrival > 0 && now >= arrival ? static_cast<double>(now - arrival) * 1e-6 : -1.0;
            {
                std::lock_guard<std::mutex> lk(sh->m);
                sh->done.push_back(std::move(d));
            }
            gst_buffer_unref(held);  // the slot returns to yvosandletterbox's ring (thread-safe pool release)
            {
                std::lock_guard<std::mutex> lk(sh->m);
                --sh->in_flight;
            }
            sh->cv.notify_all();
        });
    } catch (const std::exception& e) {
        gst_buffer_unref(held);
        {
            std::lock_guard<std::mutex> lk(sh->m);
            --sh->in_flight;
        }
        sh->cv.notify_all();
        {
            std::lock_guard<std::mutex> lk(im.out_m);
            if (!im.order.empty() && im.order.back() == seq) im.order.pop_back();
        }
        GST_ELEMENT_ERROR(bs, LIBRARY, FAILED, ("HailoRT submit failed"),
                          ("frame %llu slot %d: %s", static_cast<unsigned long long>(seq), slot, e.what()));
        return GST_FLOW_ERROR;
    }
    {
        std::lock_guard<std::mutex> sl(im.stats_m);
        ++im.stats.submitted;
    }
    flush_results(im, false);
    return GST_FLOW_OK;
}

// EOS: the application must see EOS only after the last NPU completion (and its jsonl line).
static gboolean yvo_hailo_sink_event(GstBaseSink* bs, GstEvent* event) {
    if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
        Impl& im = impl_of(bs);
        GST_OBJECT_LOCK(bs);
        const guint timeout_ms = im.drain_timeout_ms;
        GST_OBJECT_UNLOCK(bs);
        const bool drained = drain_wait(im, timeout_ms);
        flush_results(im, true);
        if (!drained) {
            {
                std::lock_guard<std::mutex> sl(im.stats_m);
                im.stats.drain_timeout = true;
            }
            GST_ELEMENT_ERROR(bs, LIBRARY, FAILED, ("inferences still in flight at EOS"),
                              ("no completion within %u ms", timeout_ms));
        }
    }
    return GST_BASE_SINK_CLASS(yvo_hailo_sink_parent_class)->event(bs, event);
}

// Advertise the tensor meta in the ALLOCATION answer (yvosandletterbox ignores the answer; informative only).
static gboolean yvo_hailo_sink_propose_allocation(GstBaseSink* /*bs*/, GstQuery* query) {
    gst_query_add_allocation_meta(query, static_cast<GType>(yvo::gst::tensor_meta_api_type()), nullptr);
    return TRUE;
}

static double mean_of(const std::vector<double>& v) {
    if (v.empty()) return -1.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

static void yvo_hailo_sink_set_property(GObject* obj, guint id, const GValue* v, GParamSpec* pspec) {
    Impl& im = impl_of(obj);
    GST_OBJECT_LOCK(obj);
    switch (id) {
    case PROP_HEF: {
        const gchar* s = g_value_get_string(v);
        im.hef = s ? s : "";
        break;
    }
    case PROP_BATCH: im.batch = g_value_get_uint(v); break;
    case PROP_SCORE: im.score = g_value_get_float(v); break;
    case PROP_DUMP_DIR: {
        const gchar* s = g_value_get_string(v);
        im.dump_dir = s ? s : "";
        break;
    }
    case PROP_DRAIN_TIMEOUT: im.drain_timeout_ms = g_value_get_uint(v); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, pspec); break;
    }
    GST_OBJECT_UNLOCK(obj);
}

static void yvo_hailo_sink_get_property(GObject* obj, guint id, GValue* v, GParamSpec* pspec) {
    Impl& im = impl_of(obj);
    switch (id) {
    case PROP_HEF:
    case PROP_BATCH:
    case PROP_SCORE:
    case PROP_DUMP_DIR:
    case PROP_DRAIN_TIMEOUT: {
        GST_OBJECT_LOCK(obj);
        if (id == PROP_HEF) g_value_set_string(v, im.hef.empty() ? nullptr : im.hef.c_str());
        else if (id == PROP_BATCH) g_value_set_uint(v, im.batch);
        else if (id == PROP_SCORE) g_value_set_float(v, im.score);
        else if (id == PROP_DUMP_DIR) g_value_set_string(v, im.dump_dir.empty() ? nullptr : im.dump_dir.c_str());
        else g_value_set_uint(v, im.drain_timeout_ms);
        GST_OBJECT_UNLOCK(obj);
        return;
    }
    default: break;
    }
    int in_flight_max = 0;
    {
        // Shared outlives stop() only through callbacks; read the live value when running.
        std::shared_ptr<Shared> sh;
        GST_OBJECT_LOCK(obj);
        sh = im.sh;
        GST_OBJECT_UNLOCK(obj);
        if (sh) {
            std::lock_guard<std::mutex> lk(sh->m);
            in_flight_max = sh->in_flight_max;
        }
    }
    std::lock_guard<std::mutex> sl(im.stats_m);
    const auto& st = im.stats;
    switch (id) {
    case PROP_FRAMES: g_value_set_uint64(v, st.completed); break;
    case PROP_SUBMITTED: g_value_set_uint64(v, st.submitted); break;
    case PROP_ERRORS: g_value_set_uint64(v, st.errors); break;
    case PROP_DETECTIONS: g_value_set_uint64(v, st.detections); break;
    case PROP_IN_FLIGHT_MAX: g_value_set_uint(v, static_cast<guint>(std::max(in_flight_max, st.in_flight_max))); break;
    case PROP_MAPPED_SLOTS: g_value_set_uint(v, static_cast<guint>(st.mapped_slots)); break;
    case PROP_NPU_MS_MEAN: g_value_set_double(v, mean_of(st.npu_ms)); break;
    case PROP_E2E_MS_MEAN: g_value_set_double(v, mean_of(st.e2e_ms)); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, pspec); break;
    }
}

static void yvo_hailo_sink_finalize(GObject* obj) {
    auto* self = reinterpret_cast<YvoHailoSink*>(obj);
    // stop() already ran for any started sink (NULL state required before finalize); nothing is in flight here.
    delete self->impl;
    self->impl = nullptr;
    G_OBJECT_CLASS(yvo_hailo_sink_parent_class)->finalize(obj);
}

static void yvo_hailo_sink_class_init(YvoHailoSinkClass* klass) {
    GST_DEBUG_CATEGORY_INIT(yvo_hailo_sink_debug, "yvohailosink", 0, "YVO HailoRT inference sink");
    GObjectClass* oc = G_OBJECT_CLASS(klass);
    oc->set_property = yvo_hailo_sink_set_property;
    oc->get_property = yvo_hailo_sink_get_property;
    oc->finalize = yvo_hailo_sink_finalize;

    const auto rw = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY);
    const auto ro = static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(oc, PROP_HEF,
        g_param_spec_string("hef", "HEF", "HEF file path (required)", nullptr, rw));
    g_object_class_install_property(oc, PROP_BATCH,
        g_param_spec_uint("batch", "Batch", "InferModel batch size (scheduler burst), as yvo_pipeline --batch", 1,
                          kMaxBatch, kDefaultBatch, rw));
    g_object_class_install_property(oc, PROP_SCORE,
        g_param_spec_float("score-threshold", "Score threshold",
                           "NMS score threshold; < 0 keeps the HEF value (0.2), as yvo_pipeline --score", -1.0f, 1.0f,
                           -1.0f, rw));
    g_object_class_install_property(oc, PROP_DUMP_DIR,
        g_param_spec_string("dump-dir", "Dump dir",
                            "Write <dir>/detections.jsonl (yvo_pipeline --dump-dir line format); empty = off", nullptr, rw));
    g_object_class_install_property(oc, PROP_DRAIN_TIMEOUT,
        g_param_spec_uint("drain-timeout-ms", "Drain timeout", "EOS/stop wait for in-flight inferences (ms)", 100,
                          600000, kDefaultDrainTimeoutMs, rw));
    g_object_class_install_property(oc, PROP_FRAMES,
        g_param_spec_uint64("frames", "Frames", "Results delivered (seq order) since start", 0, G_MAXUINT64, 0, ro));
    g_object_class_install_property(oc, PROP_SUBMITTED,
        g_param_spec_uint64("submitted", "Submitted", "Frames submitted to HailoRT since start", 0, G_MAXUINT64, 0, ro));
    g_object_class_install_property(oc, PROP_ERRORS,
        g_param_spec_uint64("errors", "Errors", "Completions with a HailoRT error status", 0, G_MAXUINT64, 0, ro));
    g_object_class_install_property(oc, PROP_DETECTIONS,
        g_param_spec_uint64("detections", "Detections", "Total detections", 0, G_MAXUINT64, 0, ro));
    g_object_class_install_property(oc, PROP_IN_FLIGHT_MAX,
        g_param_spec_uint("in-flight-max", "In flight max", "Max frames submitted and not yet completed", 0, 64, 0, ro));
    g_object_class_install_property(oc, PROP_MAPPED_SLOTS,
        g_param_spec_uint("mapped-slots", "Mapped slots", "Slot dma-bufs mapped with dma_map_dmabuf", 0, 64, 0, ro));
    g_object_class_install_property(oc, PROP_NPU_MS_MEAN,
        g_param_spec_double("npu-ms-mean", "NPU ms mean", "Mean run_async → completion time (ms), -1 if none", -1.0,
                            G_MAXDOUBLE, -1.0, ro));
    g_object_class_install_property(oc, PROP_E2E_MS_MEAN,
        g_param_spec_double("e2e-ms-mean", "e2e ms mean", "Mean letterbox arrival → completion time (ms), -1 if none",
                            -1.0, G_MAXDOUBLE, -1.0, ro));

    GstElementClass* ec = GST_ELEMENT_CLASS(klass);
    gst_element_class_set_static_metadata(ec, "YVO HailoRT inference sink", "Sink/Analyzer/Hardware",
                                          "yvosandletterbox NHCW slot dma-buf -> HailoRT InferModel (HEF NMS on host)",
                                          "yolo_vulkan_optimize");
    gst_element_class_add_static_pad_template(ec, &sink_template);

    GstBaseSinkClass* bc = GST_BASE_SINK_CLASS(klass);
    bc->start = yvo_hailo_sink_start;
    bc->stop = yvo_hailo_sink_stop;
    bc->render = yvo_hailo_sink_render;
    bc->event = yvo_hailo_sink_event;
    bc->propose_allocation = yvo_hailo_sink_propose_allocation;
}

static void yvo_hailo_sink_init(YvoHailoSink* self) {
    self->impl = new Impl();
    GstBaseSink* bs = GST_BASE_SINK(self);
    gst_base_sink_set_sync(bs, FALSE);
    gst_base_sink_set_qos_enabled(bs, FALSE);
    gst_base_sink_set_last_sample_enabled(bs, FALSE);  // a cached last sample would hold a ring slot
}

namespace yvo::gst {

bool register_hailo_sink(GstPlugin* plugin) {
    return gst_element_register(plugin, kHailoSinkElementName, GST_RANK_NONE, yvo_hailo_sink_get_type()) != FALSE;
}

static YvoHailoSink* as_sink(GstElement* e) {
    if (!e || !G_TYPE_CHECK_INSTANCE_TYPE(e, yvo_hailo_sink_get_type())) return nullptr;
    return reinterpret_cast<YvoHailoSink*>(e);
}

bool hailo_sink_set_result_callback(GstElement* element, HailoSinkResultCallback cb) {
    YvoHailoSink* s = as_sink(element);
    if (!s) return false;
    std::lock_guard<std::mutex> sl(s->impl->stats_m);
    s->impl->cb = std::move(cb);
    return true;
}

bool hailo_sink_get_stats(GstElement* element, HailoSinkStats& out) {
    YvoHailoSink* s = as_sink(element);
    if (!s) return false;
    Impl& im = *s->impl;
    int in_flight_max = 0;
    GST_OBJECT_LOCK(element);
    std::shared_ptr<Shared> sh = im.sh;
    GST_OBJECT_UNLOCK(element);
    if (sh) {
        std::lock_guard<std::mutex> lk(sh->m);
        in_flight_max = sh->in_flight_max;
    }
    std::lock_guard<std::mutex> sl(im.stats_m);
    if (in_flight_max > im.stats.in_flight_max) im.stats.in_flight_max = in_flight_max;
    out = im.stats;
    return true;
}

} // namespace yvo::gst
