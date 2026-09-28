#include "gstelem/sand_letterbox_element.hpp"

#include <gst/allocators/allocators.h>
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/frame_types.hpp"
#include "gstelem/gst_drm_frame.hpp"
#include "gstelem/tensor_meta.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vulkan_context.hpp"

GST_DEBUG_CATEGORY_STATIC(yvo_sand_letterbox_debug);
#define GST_CAT_DEFAULT yvo_sand_letterbox_debug

// DMA_DRM NV12 + Broadcom SAND128 (modifier parameter 0) only: no system-memory caps, so nothing else can link.
#define YVO_SAND_SINK_CAPS                                                                        \
    "video/x-raw(memory:DMABuf), format=(string)DMA_DRM, drm-format=(string)NV12:0x0700000000000004, " \
    "width=(int)[ 1, 2147483647 ], height=(int)[ 1, 2147483647 ]"
#define YVO_TENSOR_SRC_CAPS \
    "application/x-yvo-tensor, layout=(string)nhcw, format=(string)RGB8, width=(int)640, height=(int)640"

namespace {

constexpr int kDstW = 640;
constexpr int kDstH = 640;
constexpr int kDefaultSlots = 16;
constexpr guint kDefaultDecoderPoolMin = 4;
constexpr std::uint64_t kGpuWaitTimeoutNs = 5'000'000'000ull;  // a GPU hang becomes an element error
// stop(): grace period for slot buffers still on their way back. Longer waits cannot help: appsink keeps the last
// pulled buffer (its cached GstSample) until READY→NULL, which comes after this element's PAUSED→READY stop().
constexpr auto kDrainGrace = std::chrono::milliseconds(200);

std::uint64_t mono_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<std::uint64_t>(ts.tv_nsec);
}

// Vulkan objects and the exported slot fds. Co-owned by the element (start→stop), the slot pool (kept alive by every
// outstanding slot buffer's GstBuffer::pool ref) and every live slot GstMemory (destroy-notify qdata). The last owner
// is either stop() (normal case) or whichever thread drops the last slot buffer/memory after stop() (deferred teardown). The destructor follows the
// pipeline/main.cpp shutdown order: clear_import_cache → SandLetterbox → export fds → VulkanContext.
struct Resources {
    std::unique_ptr<yvo::vk::VulkanContext> ctx;
    std::unique_ptr<yvo::vk::SandLetterbox> pre;
    std::vector<int> fds;            // per-slot exported dma-buf fd (memories wrap them with DONT_CLOSE)
    ~Resources() {
        GST_INFO("releasing Vulkan resources (%zu slot fds)", fds.size());
        if (pre) {
            try {
                pre->clear_import_cache();
            } catch (const std::exception& e) {
                GST_WARNING("Resources teardown: clear_import_cache: %s", e.what());
            }
        }
        pre.reset();
        for (int fd : fds)
            if (fd >= 0) ::close(fd);
        fds.clear();
        ctx.reset();
    }
};

// ── Slot pool ────────────────────────────────────────────────────────────────────────────────
// Shared between the element (property reads, drain wait) and the pool object (vfuncs). Every live slot GstMemory
// also holds a reference (memory destroy-notify qdata), so Resources and the fds outlive the last slot memory even if
// downstream moved a memory out of its slot buffer.
struct SlotPoolState {
    std::shared_ptr<Resources> res;  // keeps the dma-bufs and Vulkan memory alive while slot memories exist
    std::vector<int> fds;            // = res->fds (numbers only)
    std::size_t tensor_bytes = 0;    // logical memory size
    std::size_t dmabuf_bytes = 0;    // true dma-buf size (memory maxsize)
    GstAllocator* allocator = nullptr;
    std::mutex m;
    std::condition_variable cv;
    std::vector<bool> alive;              // slot GstMemory exists (cleared only by the memory destroy notify)
    std::vector<GstBuffer*> buffer_of;    // pool buffer created for the slot's current memory (nullptr once freed).
                                          // An older buffer of the same slot (its memory was replaced) must not
                                          // clear this, so free_buffer compares identities, never just slot numbers.
    std::map<GstBuffer*, int> acquired;   // buffers handed out by acquire and not yet released → slot
    int outstanding() const { return static_cast<int>(acquired.size()); }
    int detached() const {                // slot memories alive without their pool buffer (moved/shared downstream)
        int n = 0;
        for (std::size_t i = 0; i < alive.size(); ++i)
            if (alive[i] && !buffer_of[i]) ++n;
        return n;
    }
    ~SlotPoolState() {
        if (allocator) gst_object_unref(allocator);
    }
};

// alloc_buffer: how long to wait for a slot memory that is still referenced downstream before failing explicitly.
constexpr auto kSlotMemoryWait = std::chrono::seconds(2);

GQuark slot_owner_quark() {
    static GQuark q = 0;
    static gsize once = 0;
    if (g_once_init_enter(&once)) {
        q = g_quark_from_static_string("yvo-slot-memory-owner");
        g_once_init_leave(&once, 1);
    }
    return q;
}

struct SlotMemoryOwner {
    std::shared_ptr<SlotPoolState> st;
    int slot;
};

// Destroy notify of a slot GstMemory: the only place a slot becomes free for a new memory (S1). Dropping the owner
// may destroy SlotPoolState → Resources (fds closed only after the last slot memory is gone).
void slot_memory_died(gpointer p) {
    auto* o = static_cast<SlotMemoryOwner*>(p);
    {
        std::lock_guard<std::mutex> lk(o->st->m);
        o->st->alive[static_cast<std::size_t>(o->slot)] = false;
        o->st->cv.notify_all();
    }
    delete o;
}

bool valid_slot(const SlotPoolState& s, int slot) { return slot >= 0 && static_cast<std::size_t>(slot) < s.alive.size(); }

} // namespace

struct YvoSlotPool {
    GstBufferPool parent;
    std::shared_ptr<SlotPoolState>* st;
};
struct YvoSlotPoolClass {
    GstBufferPoolClass parent_class;
};

G_DEFINE_TYPE(YvoSlotPool, yvo_slot_pool, GST_TYPE_BUFFER_POOL)

// Wraps the first slot whose memory is gone. If every slot memory is still alive (a discarded buffer's memory is
// referenced downstream), waits up to kSlotMemoryWait and then fails: a slot that may still be read is never re-used.
static GstFlowReturn yvo_slot_pool_alloc_buffer(GstBufferPool* pool, GstBuffer** buffer,
                                                GstBufferPoolAcquireParams* /*params*/) {
    const std::shared_ptr<SlotPoolState> sp = *reinterpret_cast<YvoSlotPool*>(pool)->st;
    SlotPoolState& s = *sp;
    int slot = -1;
    {
        std::unique_lock<std::mutex> lk(s.m);
        const auto deadline = std::chrono::steady_clock::now() + kSlotMemoryWait;
        while (true) {
            for (std::size_t i = 0; i < s.alive.size(); ++i)
                if (!s.alive[i]) {
                    slot = static_cast<int>(i);
                    break;
                }
            if (slot >= 0 || GST_BUFFER_POOL_IS_FLUSHING(pool) || std::chrono::steady_clock::now() >= deadline) break;
            s.cv.wait_for(lk, std::chrono::milliseconds(50));
        }
        if (slot >= 0) s.alive[static_cast<std::size_t>(slot)] = true;
    }
    if (slot < 0) {
        if (GST_BUFFER_POOL_IS_FLUSHING(pool)) return GST_FLOW_FLUSHING;
        GST_ERROR_OBJECT(pool, "no free slot: every slot memory is still referenced downstream (%zu slots)", s.alive.size());
        return GST_FLOW_ERROR;
    }
    // DONT_CLOSE: Resources owns the fd; the owner qdata below keeps Resources alive as long as this memory lives.
    GstMemory* mem = gst_dmabuf_allocator_alloc_with_flags(s.allocator, s.fds[static_cast<std::size_t>(slot)],
                                                           s.dmabuf_bytes, GST_FD_MEMORY_FLAG_DONT_CLOSE);
    if (!mem) {
        std::lock_guard<std::mutex> lk(s.m);
        s.alive[static_cast<std::size_t>(slot)] = false;
        GST_ERROR_OBJECT(pool, "gst_dmabuf_allocator_alloc failed for slot %d", slot);
        return GST_FLOW_ERROR;
    }
    gst_memory_resize(mem, 0, s.tensor_bytes); // logical tensor = first tensor_bytes of the dma-buf
    yvo::gst::tensor_memory_set_slot(mem, slot);
    gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(mem), slot_owner_quark(), new SlotMemoryOwner{sp, slot},
                              slot_memory_died);
    GstBuffer* buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);
    yvo::gst::tensor_buffer_set_slot(buf, slot); // S2: the pool tracks the buffer by its own tag, not its memory
    {
        std::lock_guard<std::mutex> lk(s.m);
        s.buffer_of[static_cast<std::size_t>(slot)] = buf;
    }
    GST_DEBUG_OBJECT(pool, "allocated slot %d buffer (fd %d, %zu/%zu bytes)", slot, s.fds[static_cast<std::size_t>(slot)],
                     s.tensor_bytes, s.dmabuf_bytes);
    *buffer = buf;
    return GST_FLOW_OK;
}

// Buffer freed (pool stop, or discarded because downstream touched/shared its memory). The slot's memory may live on
// elsewhere, so `alive` is left to the memory destroy notify.
static void yvo_slot_pool_free_buffer(GstBufferPool* pool, GstBuffer* buffer) {
    SlotPoolState& s = **reinterpret_cast<YvoSlotPool*>(pool)->st;
    const int slot = yvo::gst::tensor_buffer_own_slot(buffer);
    {
        std::lock_guard<std::mutex> lk(s.m);
        if (valid_slot(s, slot) && s.buffer_of[static_cast<std::size_t>(slot)] == buffer)
            s.buffer_of[static_cast<std::size_t>(slot)] = nullptr;
        s.acquired.erase(buffer);
        s.cv.notify_all();
    }
    GST_BUFFER_POOL_CLASS(yvo_slot_pool_parent_class)->free_buffer(pool, buffer);
}

static GstFlowReturn yvo_slot_pool_acquire_buffer(GstBufferPool* pool, GstBuffer** buffer,
                                                  GstBufferPoolAcquireParams* params) {
    const GstFlowReturn r = GST_BUFFER_POOL_CLASS(yvo_slot_pool_parent_class)->acquire_buffer(pool, buffer, params);
    if (r == GST_FLOW_OK) {
        SlotPoolState& s = **reinterpret_cast<YvoSlotPool*>(pool)->st;
        const int slot = yvo::gst::tensor_buffer_own_slot(*buffer);
        std::lock_guard<std::mutex> lk(s.m);
        s.acquired[*buffer] = slot;
        GST_LOG_OBJECT(pool, "acquired slot %d (outstanding %d)", slot, s.outstanding());
    }
    return r;
}

// Also called directly by the base class for preallocated buffers (default_start), which were never acquired:
// only buffers present in `acquired` are counted.
static void yvo_slot_pool_release_buffer(GstBufferPool* pool, GstBuffer* buffer) {
    {
        SlotPoolState& s = **reinterpret_cast<YvoSlotPool*>(pool)->st;
        std::lock_guard<std::mutex> lk(s.m);
        if (s.acquired.erase(buffer)) s.cv.notify_all();
        GST_LOG_OBJECT(pool, "released slot %d (outstanding %d)", yvo::gst::tensor_buffer_own_slot(buffer),
                       s.outstanding());
    }
    GST_BUFFER_POOL_CLASS(yvo_slot_pool_parent_class)->release_buffer(pool, buffer);
}

static void yvo_slot_pool_finalize(GObject* obj) {
    GST_DEBUG_OBJECT(obj, "finalize");
    delete reinterpret_cast<YvoSlotPool*>(obj)->st;
    G_OBJECT_CLASS(yvo_slot_pool_parent_class)->finalize(obj);
}

static void yvo_slot_pool_class_init(YvoSlotPoolClass* klass) {
    G_OBJECT_CLASS(klass)->finalize = yvo_slot_pool_finalize;
    GstBufferPoolClass* pc = GST_BUFFER_POOL_CLASS(klass);
    pc->alloc_buffer = yvo_slot_pool_alloc_buffer;
    pc->free_buffer = yvo_slot_pool_free_buffer;
    pc->acquire_buffer = yvo_slot_pool_acquire_buffer;
    pc->release_buffer = yvo_slot_pool_release_buffer;
}

static void yvo_slot_pool_init(YvoSlotPool* self) { self->st = new std::shared_ptr<SlotPoolState>(); }

// ── Element ─────────────────────────────────────────────────────────────────────────────────
namespace {

struct Impl {
    // properties (GST_OBJECT_LOCK)
    int slots = kDefaultSlots;
    guint decoder_pool_min = kDefaultDecoderPoolMin;
    bool validation = false;

    // runtime: created in start(), destroyed in stop(); used by the streaming thread in between
    std::shared_ptr<Resources> res;
    yvo::vk::SandLetterbox* pre = nullptr;      // = res->pre (streaming thread only)
    std::size_t dmabuf_bytes = 0;
    GstBufferPool* pool = nullptr;
    std::shared_ptr<SlotPoolState> pool_state;  // also read by property getters (GST_OBJECT_LOCK)
    yvo::gst::SandCaps sc;
    bool caps_ok = false;
    std::uint64_t seq = 0;
    std::uint64_t arrival_ns = 0;      // set in prepare_output_buffer, consumed in transform
    std::uint64_t acquire_wait_ns = 0;
    std::vector<GstBuffer*> failed_out;  // N4: outbufs whose GPU wait failed (slot still pending) — held until teardown

    // stats (GST_OBJECT_LOCK)
    std::uint64_t frames = 0;
    double gpu_sum = 0.0;
    std::uint64_t gpu_n = 0;
    std::uint64_t val_errors = 0;
    bool val_active = false;
};

} // namespace

struct YvoSandLetterbox {
    GstBaseTransform parent;
    Impl* impl;
};
struct YvoSandLetterboxClass {
    GstBaseTransformClass parent_class;
};

G_DEFINE_TYPE(YvoSandLetterbox, yvo_sand_letterbox, GST_TYPE_BASE_TRANSFORM)

enum {
    PROP_0,
    PROP_SLOTS,
    PROP_DECODER_POOL_MIN,
    PROP_VALIDATION,
    PROP_FRAMES,
    PROP_GPU_MS_MEAN,
    PROP_SLOTS_IN_USE,
    PROP_SLOTS_DETACHED,
    PROP_VALIDATION_ERRORS,
    PROP_VALIDATION_ACTIVE,
};

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(YVO_SAND_SINK_CAPS));
static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(YVO_TENSOR_SRC_CAPS));

static Impl& impl_of(GstBaseTransform* t) { return *reinterpret_cast<YvoSandLetterbox*>(t)->impl; }

// Deactivates the pool, releases the decoder imports, then drops the element's share of the Vulkan resources.
// If every slot buffer is back, the resources are destroyed right here; otherwise when the last one returns.
static void teardown(YvoSandLetterbox* self) {
    Impl& im = *self->impl;
    int held_n = 0;
    std::string held;
    if (im.pool) {
        gst_buffer_pool_set_flushing(im.pool, TRUE);
        gst_buffer_pool_set_active(im.pool, FALSE);
        // Slots whose GPU wait failed: the GPU work is still covered by vkDeviceWaitIdle in ~SandLetterbox.
        for (GstBuffer* b : im.failed_out) gst_buffer_unref(b);
        im.failed_out.clear();
        SlotPoolState& s = *im.pool_state;
        std::unique_lock<std::mutex> lk(s.m);
        s.cv.wait_for(lk, kDrainGrace, [&] { return s.outstanding() == 0; });
        held_n = s.outstanding();
        for (const auto& kv : s.acquired) held += " " + std::to_string(kv.second);
    }
    // No pending GPU work here (transform always waits), so the dup'ed decoder fds can go now in any case.
    if (im.pre) {
        try {
            im.pre->clear_import_cache();
        } catch (const std::exception& e) {
            GST_WARNING_OBJECT(self, "clear_import_cache: %s", e.what());
        }
    }
    GST_OBJECT_LOCK(self);
    if (im.res && im.res->ctx) im.val_errors = im.res->ctx->validation_errors();
    GstBufferPool* pool = im.pool;
    im.pool = nullptr;
    im.pool_state.reset();
    GST_OBJECT_UNLOCK(self);
    im.pre = nullptr;
    im.res.reset();
    if (held_n > 0)
        GST_INFO_OBJECT(self, "stop(): %d slot buffer(s) still held downstream (slots%s); Vulkan teardown deferred "
                              "until they are released", held_n, held.c_str());
    if (pool) gst_object_unref(pool); // last ref if nothing is held → pool finalize → Resources destroyed here
    im.caps_ok = false;
}

static gboolean yvo_sand_letterbox_start(GstBaseTransform* t) {
    auto* self = reinterpret_cast<YvoSandLetterbox*>(t);
    Impl& im = *self->impl;
    GST_OBJECT_LOCK(self);
    const int slots = im.slots;
    const bool validation = im.validation;
    im.frames = 0;
    im.gpu_sum = 0.0;
    im.gpu_n = 0;
    im.val_errors = 0;
    GST_OBJECT_UNLOCK(self);
    im.seq = 0;
    try {
        yvo::vk::VulkanContextOptions o;
        o.enable_validation = validation;
        o.device_name_hint = "V3D";
        im.res = std::make_shared<Resources>();
        Resources& r = *im.res;
        r.ctx = std::make_unique<yvo::vk::VulkanContext>(o);
        yvo::vk::SandLetterboxConfig c;
        c.num_slots = slots;
        c.dst_w = kDstW;
        c.dst_h = kDstH;
        c.out_layout = yvo::TensorLayout::Nhcw;
        c.export_dmabuf = true;
        r.pre = std::make_unique<yvo::vk::SandLetterbox>(*r.ctx, c);
        im.pre = r.pre.get();

        // Export each slot once: these fds (and their numbers) live as long as Resources (stop() or the last slot buffer).
        for (int i = 0; i < slots; ++i) r.fds.push_back(r.pre->export_dmabuf_fd(i));
        im.dmabuf_bytes = 0;
        for (int fd : r.fds) {
            const off_t end = lseek(fd, 0, SEEK_END);
            (void)lseek(fd, 0, SEEK_SET);
            if (end < static_cast<off_t>(r.pre->output_size()))
                throw std::runtime_error("exported slot dma-buf smaller than the tensor (lseek " + std::to_string(end) + ")");
            if (im.dmabuf_bytes == 0) im.dmabuf_bytes = static_cast<std::size_t>(end);
            if (static_cast<std::size_t>(end) != im.dmabuf_bytes)
                throw std::runtime_error("exported slot dma-bufs differ in size");
        }

        auto st = std::make_shared<SlotPoolState>();
        st->res = im.res;
        st->fds = r.fds;
        st->tensor_bytes = r.pre->output_size();
        st->dmabuf_bytes = im.dmabuf_bytes;
        st->allocator = gst_dmabuf_allocator_new();
        st->alive.assign(static_cast<std::size_t>(slots), false);
        st->buffer_of.assign(static_cast<std::size_t>(slots), nullptr);
        GstBufferPool* pool = GST_BUFFER_POOL(g_object_new(yvo_slot_pool_get_type(), nullptr));
        gst_object_ref_sink(pool);
        *reinterpret_cast<YvoSlotPool*>(pool)->st = st;
        GstStructure* cfg = gst_buffer_pool_get_config(pool);
        gst_buffer_pool_config_set_params(cfg, nullptr, static_cast<guint>(st->tensor_bytes), static_cast<guint>(slots),
                                          static_cast<guint>(slots));
        if (!gst_buffer_pool_set_config(pool, cfg) || !gst_buffer_pool_set_active(pool, TRUE)) {
            gst_object_unref(pool);
            throw std::runtime_error("slot pool configuration/activation failed");
        }
        GST_OBJECT_LOCK(self);
        im.pool = pool;
        im.pool_state = st;
        im.val_active = r.ctx->validation_layer_active() && r.ctx->debug_messenger_active();
        GST_OBJECT_UNLOCK(self);
        GST_INFO_OBJECT(self, "Vulkan device %s, %d slots, tensor %zu B, slot dma-buf %zu B, validation %s",
                        r.ctx->device_name(), slots, st->tensor_bytes, im.dmabuf_bytes,
                        r.ctx->validation_layer_active() ? "on" : "off");
    } catch (const std::exception& e) {
        GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("Vulkan SAND letterbox initialisation failed"), ("%s", e.what()));
        teardown(self);
        return FALSE;
    }
    return TRUE;
}

static gboolean yvo_sand_letterbox_stop(GstBaseTransform* t) {
    teardown(reinterpret_cast<YvoSandLetterbox*>(t));
    return TRUE;
}

// Tensor caps from video caps (sink → src) or the SAND template from tensor caps (src → sink); framerate carried over.
static GstCaps* yvo_sand_letterbox_transform_caps(GstBaseTransform* /*t*/, GstPadDirection dir, GstCaps* caps,
                                                  GstCaps* filter) {
    GstCaps* tmpl = gst_static_caps_get(dir == GST_PAD_SINK ? &src_template.static_caps : &sink_template.static_caps);
    GstCaps* res = nullptr;
    if (gst_caps_is_any(caps)) {
        res = gst_caps_ref(tmpl);
    } else {
        res = gst_caps_new_empty();
        const GstStructure* ts = gst_caps_get_structure(tmpl, 0);
        const GstCapsFeatures* tf = gst_caps_get_features(tmpl, 0);
        for (guint i = 0; i < gst_caps_get_size(caps); ++i) {
            GstStructure* s = gst_structure_copy(ts);
            if (const GValue* fr = gst_structure_get_value(gst_caps_get_structure(caps, i), "framerate"))
                gst_structure_set_value(s, "framerate", fr);
            res = gst_caps_merge_structure_full(res, s, tf ? gst_caps_features_copy(tf) : nullptr);
        }
    }
    gst_caps_unref(tmpl);
    if (filter) {
        GstCaps* x = gst_caps_intersect_full(filter, res, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(res);
        res = x;
    }
    return res;
}

static gboolean yvo_sand_letterbox_set_caps(GstBaseTransform* t, GstCaps* incaps, GstCaps* outcaps) {
    Impl& im = impl_of(t);
    std::string why;
    yvo::gst::SandCaps sc;
    if (!yvo::gst::sand_caps_from_caps(incaps, sc, &why)) {
        gchar* s = gst_caps_to_string(incaps);
        GST_ELEMENT_ERROR(t, CORE, NEGOTIATION, ("unsupported input caps for the SAND letterbox"), ("%s: %s", why.c_str(), s));
        g_free(s);
        im.caps_ok = false;
        return FALSE;
    }
    // N6: a new input geometry/format makes the cached decoder imports useless → release them now (no pending slot:
    // transform always waits) instead of keeping those decoder BOs pinned until LRU eviction.
    const bool changed = im.caps_ok && (sc.width != im.sc.width || sc.height != im.sc.height ||
                                        sc.drm_fourcc != im.sc.drm_fourcc || sc.drm_modifier != im.sc.drm_modifier);
    if (changed && im.pre) {
        try {
            im.pre->clear_import_cache();
            GST_INFO_OBJECT(t, "input caps changed: import cache cleared");
        } catch (const std::exception& e) {
            GST_WARNING_OBJECT(t, "input caps changed, clear_import_cache: %s", e.what());
        }
    }
    im.sc = sc;
    im.caps_ok = true;
    gchar* a = gst_caps_to_string(incaps);
    gchar* b = gst_caps_to_string(outcaps);
    GST_INFO_OBJECT(t, "caps: in %s -> out %s", a, b);
    g_free(a);
    g_free(b);
    return TRUE;
}

// Upstream (decoder) ALLOCATION query: VideoMeta is what makes v4l2slh265dec accept DMA_DRM; a small min keeps the
// decoder pool small (stage 1: pool of 10 with min 4).
static gboolean yvo_sand_letterbox_propose_allocation(GstBaseTransform* t, GstQuery* /*decide_query*/, GstQuery* query) {
    Impl& im = impl_of(t);
    GstCaps* caps = nullptr;
    gboolean need_pool = FALSE;
    gst_query_parse_allocation(query, &caps, &need_pool);
    guint size = 0;
    if (caps) {
        GstVideoInfoDmaDrm di;
        if (gst_video_info_dma_drm_from_caps(&di, caps)) size = static_cast<guint>(GST_VIDEO_INFO_SIZE(&di.vinfo));
    }
    GST_OBJECT_LOCK(t);
    const guint min = im.decoder_pool_min;
    GST_OBJECT_UNLOCK(t);
    gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, nullptr);
    gst_query_add_allocation_pool(query, nullptr, size, min, 0);
    GST_DEBUG_OBJECT(t, "proposed upstream allocation: VideoMeta + pool(size %u, min %u)", size, min);
    return TRUE;
}

// Downstream ALLOCATION answer: output memory must be our exported slots, so pools/allocators offered downstream are
// dropped (BaseTransform then keeps no pool; prepare_output_buffer uses the slot pool).
static gboolean yvo_sand_letterbox_decide_allocation(GstBaseTransform* t, GstQuery* query) {
    while (gst_query_get_n_allocation_pools(query) > 0) gst_query_remove_nth_allocation_pool(query, 0);
    while (gst_query_get_n_allocation_params(query) > 0) gst_query_remove_nth_allocation_param(query, 0);
    const bool meta = gst_query_find_allocation_meta(query, static_cast<GType>(yvo::gst::tensor_meta_api_type()), nullptr);
    GST_DEBUG_OBJECT(t, "downstream allocation ignored (downstream %s YvoTensorMeta)", meta ? "advertises" : "does not advertise");
    return TRUE;
}

static GstFlowReturn yvo_sand_letterbox_prepare_output_buffer(GstBaseTransform* t, GstBuffer* inbuf, GstBuffer** outbuf) {
    Impl& im = impl_of(t);
    im.arrival_ns = mono_ns();
    GST_OBJECT_LOCK(t);
    GstBufferPool* pool = im.pool ? GST_BUFFER_POOL(gst_object_ref(im.pool)) : nullptr;
    GST_OBJECT_UNLOCK(t);
    if (!pool) {
        GST_ELEMENT_ERROR(t, CORE, STATE_CHANGE, ("no slot pool (element not started)"), (nullptr));
        return GST_FLOW_ERROR;
    }
    // Blocks while all slots are held downstream (ring backpressure). Returns FLUSHING on shutdown/flush.
    const GstFlowReturn r = gst_buffer_pool_acquire_buffer(pool, outbuf, nullptr);
    gst_object_unref(pool);
    im.acquire_wait_ns = mono_ns() - im.arrival_ns;
    if (r != GST_FLOW_OK) {
        GST_DEBUG_OBJECT(t, "slot acquire returned %s", gst_flow_get_name(r));
        if (r == GST_FLOW_ERROR)
            GST_ELEMENT_ERROR(t, RESOURCE, BUSY, ("no free output slot: slot memory still referenced downstream"),
                              ("a slot GstMemory outlived its slot buffer (moved/shared into another buffer) for more "
                               "than %lld s; the slot is not overwritten while it may still be read",
                               static_cast<long long>(kSlotMemoryWait.count())));
        return r;
    }
    GstBuffer* out = *outbuf;
    GST_BUFFER_PTS(out) = GST_BUFFER_PTS(inbuf);
    GST_BUFFER_DTS(out) = GST_BUFFER_DTS(inbuf);
    GST_BUFFER_DURATION(out) = GST_BUFFER_DURATION(inbuf);
    GST_BUFFER_OFFSET(out) = im.seq;
    GST_BUFFER_OFFSET_END(out) = im.seq + 1;
    if (GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT)) GST_BUFFER_FLAG_SET(out, GST_BUFFER_FLAG_DISCONT);
    return GST_FLOW_OK;
}

static GstFlowReturn yvo_sand_letterbox_transform(GstBaseTransform* t, GstBuffer* inbuf, GstBuffer* outbuf) {
    Impl& im = impl_of(t);
    if (!im.caps_ok || !im.pre) {
        GST_ELEMENT_ERROR(t, CORE, NEGOTIATION, ("buffer before caps / after stop"), (nullptr));
        return GST_FLOW_NOT_NEGOTIATED;
    }
    yvo::DrmFrame f;
    std::string why;
    if (!yvo::gst::drm_frame_from_buffer(inbuf, im.sc, f, &why)) {
        const std::string d = yvo::gst::describe_buffer(inbuf);
        GST_ELEMENT_ERROR(t, STREAM, FORMAT, ("input buffer rejected by the SAND adapter"),
                          ("frame %llu: %s\n%s", static_cast<unsigned long long>(im.seq), why.c_str(), d.c_str()));
        return GST_FLOW_ERROR;
    }
    const int slot = yvo::gst::tensor_buffer_slot(outbuf);
    if (slot < 0 || slot >= im.pre->num_slots()) {
        GST_ELEMENT_ERROR(t, CORE, FAILED, ("output buffer is not a slot buffer"), ("slot %d", slot));
        return GST_FLOW_ERROR;
    }
    yvo::PreprocOutput o;
    bool submitted = false;
    try {
        im.pre->submit(f, slot);
        submitted = true;
        f = yvo::DrmFrame{}; // SandLetterbox keeps its own keepalive copy until wait()
        o = im.pre->wait(slot, kGpuWaitTimeoutNs);
    } catch (const std::exception& e) {
        // N4: the slot stays pending in SandLetterbox → keep its buffer so the pool cannot hand the slot out again.
        if (submitted && im.pre->pending(slot)) im.failed_out.push_back(gst_buffer_ref(outbuf));
        GST_ELEMENT_ERROR(t, LIBRARY, FAILED, ("Vulkan SAND letterbox failed"),
                          ("frame %llu slot %d: %s", static_cast<unsigned long long>(im.seq), slot, e.what()));
        return GST_FLOW_ERROR;
    }
    const std::uint64_t done = mono_ns();

    yvo::gst::TensorFrameInfo info;
    info.slot = slot;
    info.num_slots = im.pre->num_slots();
    info.seq = im.seq;
    info.src_pts_ns = GST_BUFFER_PTS_IS_VALID(inbuf) ? static_cast<std::int64_t>(GST_BUFFER_PTS(inbuf)) : -1;
    info.src_width = im.sc.width;
    info.src_height = im.sc.height;
    info.lb = o.lb;
    info.layout = o.layout;
    info.tensor_bytes = static_cast<std::uint32_t>(o.size);
    info.dmabuf_bytes = im.dmabuf_bytes;
    info.dmabuf_fd = im.res->fds[static_cast<std::size_t>(slot)];
    info.gpu_ms = im.pre->last_gpu_ms();
    info.arrival_mono_ns = im.arrival_ns;
    info.acquire_wait_ns = im.acquire_wait_ns;
    info.gpu_done_mono_ns = done;
    if (!yvo::gst::tensor_meta_set(outbuf, info)) {
        GST_ELEMENT_ERROR(t, CORE, FAILED, ("could not attach YvoTensorMeta"), ("slot %d", slot));
        return GST_FLOW_ERROR;
    }
    ++im.seq;

    GST_OBJECT_LOCK(t);
    ++im.frames;
    if (info.gpu_ms >= 0.0) {
        im.gpu_sum += info.gpu_ms;
        ++im.gpu_n;
    }
    im.val_errors = im.res->ctx->validation_errors();
    GST_OBJECT_UNLOCK(t);
    return GST_FLOW_OK;
}

// Upstream metas (VideoMeta etc.) describe the video frame, not the tensor: none are copied.
static gboolean yvo_sand_letterbox_transform_meta(GstBaseTransform* /*t*/, GstBuffer* /*outbuf*/, GstMeta* /*meta*/,
                                                  GstBuffer* /*inbuf*/) {
    return FALSE;
}

static void set_pool_flushing(GstBaseTransform* t, bool flushing) {
    GST_OBJECT_LOCK(t);
    GstBufferPool* pool = impl_of(t).pool ? GST_BUFFER_POOL(gst_object_ref(impl_of(t).pool)) : nullptr;
    GST_OBJECT_UNLOCK(t);
    if (!pool) return;
    if (gst_buffer_pool_is_active(pool)) gst_buffer_pool_set_flushing(pool, flushing ? TRUE : FALSE);
    gst_object_unref(pool);
}

static gboolean yvo_sand_letterbox_sink_event(GstBaseTransform* t, GstEvent* event) {
    if (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_START) set_pool_flushing(t, true);
    else if (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_STOP) set_pool_flushing(t, false);
    return GST_BASE_TRANSFORM_CLASS(yvo_sand_letterbox_parent_class)->sink_event(t, event);
}

// PAUSED→READY: unblock a streaming thread waiting for a slot BEFORE the pads are deactivated (the sink pad
// deactivation takes the stream lock, which that thread holds).
static GstStateChangeReturn yvo_sand_letterbox_change_state(GstElement* e, GstStateChange transition) {
    if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) set_pool_flushing(GST_BASE_TRANSFORM(e), true);
    return GST_ELEMENT_CLASS(yvo_sand_letterbox_parent_class)->change_state(e, transition);
}

static void yvo_sand_letterbox_set_property(GObject* obj, guint id, const GValue* v, GParamSpec* pspec) {
    Impl& im = *reinterpret_cast<YvoSandLetterbox*>(obj)->impl;
    GST_OBJECT_LOCK(obj);
    switch (id) {
    case PROP_SLOTS: im.slots = g_value_get_int(v); break;
    case PROP_DECODER_POOL_MIN: im.decoder_pool_min = g_value_get_uint(v); break;
    case PROP_VALIDATION: im.validation = g_value_get_boolean(v) != FALSE; break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, pspec); break;
    }
    GST_OBJECT_UNLOCK(obj);
}

static void yvo_sand_letterbox_get_property(GObject* obj, guint id, GValue* v, GParamSpec* pspec) {
    Impl& im = *reinterpret_cast<YvoSandLetterbox*>(obj)->impl;
    GST_OBJECT_LOCK(obj);
    switch (id) {
    case PROP_SLOTS: g_value_set_int(v, im.slots); break;
    case PROP_DECODER_POOL_MIN: g_value_set_uint(v, im.decoder_pool_min); break;
    case PROP_VALIDATION: g_value_set_boolean(v, im.validation); break;
    case PROP_FRAMES: g_value_set_uint64(v, im.frames); break;
    case PROP_GPU_MS_MEAN: g_value_set_double(v, im.gpu_n ? im.gpu_sum / static_cast<double>(im.gpu_n) : -1.0); break;
    case PROP_SLOTS_IN_USE: {
        guint n = 0;
        if (im.pool_state) {
            std::lock_guard<std::mutex> lk(im.pool_state->m);
            n = static_cast<guint>(im.pool_state->outstanding());
        }
        g_value_set_uint(v, n);
        break;
    }
    case PROP_SLOTS_DETACHED: {
        guint n = 0;
        if (im.pool_state) {
            std::lock_guard<std::mutex> lk(im.pool_state->m);
            n = static_cast<guint>(im.pool_state->detached());
        }
        g_value_set_uint(v, n);
        break;
    }
    case PROP_VALIDATION_ERRORS: g_value_set_uint64(v, im.val_errors); break;
    case PROP_VALIDATION_ACTIVE: g_value_set_boolean(v, im.val_active); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, pspec); break;
    }
    GST_OBJECT_UNLOCK(obj);
}

static void yvo_sand_letterbox_finalize(GObject* obj) {
    auto* self = reinterpret_cast<YvoSandLetterbox*>(obj);
    teardown(self); // no-op after a normal stop()
    delete self->impl;
    self->impl = nullptr;
    G_OBJECT_CLASS(yvo_sand_letterbox_parent_class)->finalize(obj);
}

static void yvo_sand_letterbox_class_init(YvoSandLetterboxClass* klass) {
    GST_DEBUG_CATEGORY_INIT(yvo_sand_letterbox_debug, "yvosandletterbox", 0, "YVO Vulkan SAND128 letterbox");
    GObjectClass* oc = G_OBJECT_CLASS(klass);
    oc->set_property = yvo_sand_letterbox_set_property;
    oc->get_property = yvo_sand_letterbox_get_property;
    oc->finalize = yvo_sand_letterbox_finalize;

    const auto rw = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY);
    const auto ro = static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(oc, PROP_SLOTS,
        g_param_spec_int("slots", "Slots", "Number of output ring slot buffers (blocks when all are held downstream)",
                         2, 32, kDefaultSlots, rw));
    g_object_class_install_property(oc, PROP_DECODER_POOL_MIN,
        g_param_spec_uint("decoder-pool-min", "Decoder pool min",
                          "Min buffers proposed to the upstream (decoder) pool in the ALLOCATION answer", 1, 32,
                          kDefaultDecoderPoolMin, rw));
    g_object_class_install_property(oc, PROP_VALIDATION,
        g_param_spec_boolean("validation", "Validation", "Enable VK_LAYER_KHRONOS_validation (VulkanContextOptions)",
                             FALSE, rw));
    g_object_class_install_property(oc, PROP_FRAMES,
        g_param_spec_uint64("frames", "Frames", "Frames transformed since start", 0, G_MAXUINT64, 0, ro));
    g_object_class_install_property(oc, PROP_GPU_MS_MEAN,
        g_param_spec_double("gpu-ms-mean", "GPU ms mean", "Mean GPU dispatch time (ms), -1 if unknown", -1.0,
                            G_MAXDOUBLE, -1.0, ro));
    g_object_class_install_property(oc, PROP_SLOTS_IN_USE,
        g_param_spec_uint("slots-in-use", "Slots in use",
                          "Slot buffers acquired from the pool and not yet returned (tracked per buffer)", 0, 32, 0, ro));
    g_object_class_install_property(oc, PROP_SLOTS_DETACHED,
        g_param_spec_uint("slots-detached", "Slots detached",
                          "Slot memories alive without their slot buffer (moved/shared downstream; blocks re-use)", 0, 32,
                          0, ro));
    g_object_class_install_property(oc, PROP_VALIDATION_ERRORS,
        g_param_spec_uint64("validation-errors", "Validation errors",
                            "Vulkan debug-messenger error count (kept after stop)", 0, G_MAXUINT64, 0, ro));
    g_object_class_install_property(oc, PROP_VALIDATION_ACTIVE,
        g_param_spec_boolean("validation-active", "Validation active",
                             "VK_LAYER_KHRONOS_validation on and counted (property, YVO_VK_VALIDATION or VK_LOADER_LAYERS_ENABLE)", FALSE, ro));

    GstElementClass* ec = GST_ELEMENT_CLASS(klass);
    gst_element_class_set_static_metadata(ec, "YVO Vulkan SAND letterbox", "Filter/Converter/Video/Hardware",
                                          "NV12 SAND128 DMABuf -> 640x640 NHCW RGB8 tensor in exported Vulkan dma-bufs",
                                          "yolo_vulkan_optimize");
    gst_element_class_add_static_pad_template(ec, &sink_template);
    gst_element_class_add_static_pad_template(ec, &src_template);
    ec->change_state = yvo_sand_letterbox_change_state;

    GstBaseTransformClass* bc = GST_BASE_TRANSFORM_CLASS(klass);
    bc->passthrough_on_same_caps = FALSE;
    bc->transform_ip_on_passthrough = FALSE;
    bc->start = yvo_sand_letterbox_start;
    bc->stop = yvo_sand_letterbox_stop;
    bc->transform_caps = yvo_sand_letterbox_transform_caps;
    bc->set_caps = yvo_sand_letterbox_set_caps;
    bc->propose_allocation = yvo_sand_letterbox_propose_allocation;
    bc->decide_allocation = yvo_sand_letterbox_decide_allocation;
    bc->prepare_output_buffer = yvo_sand_letterbox_prepare_output_buffer;
    bc->transform = yvo_sand_letterbox_transform;
    bc->transform_meta = yvo_sand_letterbox_transform_meta;
    bc->sink_event = yvo_sand_letterbox_sink_event;
}

static void yvo_sand_letterbox_init(YvoSandLetterbox* self) {
    self->impl = new Impl();
    GstBaseTransform* t = GST_BASE_TRANSFORM(self);
    gst_base_transform_set_passthrough(t, FALSE);
    gst_base_transform_set_in_place(t, FALSE);
    gst_base_transform_set_qos_enabled(t, FALSE);
}

namespace yvo::gst {

bool register_sand_letterbox(GstPlugin* plugin) {
    return gst_element_register(plugin, kSandLetterboxElementName, GST_RANK_NONE, yvo_sand_letterbox_get_type()) != FALSE;
}

} // namespace yvo::gst
