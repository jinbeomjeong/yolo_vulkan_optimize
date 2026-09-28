#include "gstelem/tensor_meta.hpp"

#include <gst/allocators/allocators.h>
#include <gst/gst.h>

#include <new>
#include <type_traits>

namespace yvo::gst {

namespace {

static_assert(sizeof(GType) == sizeof(std::size_t), "GType is exposed as std::size_t");
static_assert(std::is_trivially_copyable_v<TensorFrameInfo>, "TensorFrameInfo is stored by value in a GstMeta");

struct YvoTensorMeta {
    GstMeta meta;
    TensorFrameInfo info;
};

gboolean meta_init(GstMeta* meta, gpointer /*params*/, GstBuffer* /*buffer*/) {
    auto* m = reinterpret_cast<YvoTensorMeta*>(meta);
    new (&m->info) TensorFrameInfo{};
    return TRUE;
}

void meta_free(GstMeta* /*meta*/, GstBuffer* /*buffer*/) {} // trivially destructible

// Copy on buffer copies (gst_buffer_copy / make_writable). Other transforms drop it: the meta describes these exact bytes.
gboolean meta_transform(GstBuffer* dest, GstMeta* meta, GstBuffer* /*buffer*/, GQuark type, gpointer /*data*/) {
    if (!GST_META_TRANSFORM_IS_COPY(type)) return FALSE;
    const auto* src = reinterpret_cast<const YvoTensorMeta*>(meta);
    return tensor_meta_set(dest, src->info) ? TRUE : FALSE;
}

GType api_type() {
    static GType type = 0;
    static gsize once = 0;
    if (g_once_init_enter(&once)) {
        static const gchar* tags[] = {nullptr}; // no tags: kept by transforms that only care about tagged metas
        type = gst_meta_api_type_register("YvoTensorMetaAPI", tags);
        g_once_init_leave(&once, 1);
    }
    return type;
}

const GstMetaInfo* meta_info() {
    static const GstMetaInfo* info = nullptr;
    static gsize once = 0;
    if (g_once_init_enter(&once)) {
        info = gst_meta_register(api_type(), "YvoTensorMeta", sizeof(YvoTensorMeta), meta_init, meta_free,
                                 meta_transform);
        g_once_init_leave(&once, 1);
    }
    return info;
}

GQuark slot_quark() {
    static GQuark q = 0;
    static gsize once = 0;
    if (g_once_init_enter(&once)) {
        q = g_quark_from_static_string("yvo-tensor-slot");
        g_once_init_leave(&once, 1);
    }
    return q;
}

} // namespace

std::size_t tensor_meta_api_type() { return static_cast<std::size_t>(api_type()); }

bool tensor_meta_set(GstBuffer* buf, const TensorFrameInfo& info) {
    if (!buf) return false;
    auto* m = reinterpret_cast<YvoTensorMeta*>(gst_buffer_get_meta(buf, api_type()));
    if (!m) {
        if (!gst_buffer_is_writable(buf)) return false;
        m = reinterpret_cast<YvoTensorMeta*>(gst_buffer_add_meta(buf, meta_info(), nullptr));
        if (!m) return false;
    }
    m->info = info;
    return true;
}

bool tensor_meta_get(GstBuffer* buf, TensorFrameInfo& out) {
    if (!buf) return false;
    const auto* m = reinterpret_cast<const YvoTensorMeta*>(gst_buffer_get_meta(buf, api_type()));
    if (!m) return false;
    out = m->info;
    return true;
}

int tensor_buffer_own_slot(GstBuffer* buf) {
    if (!buf) return -1;
    const gpointer p = gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(buf), slot_quark());
    return p ? GPOINTER_TO_INT(p) - 1 : -1;
}

int tensor_buffer_slot(GstBuffer* buf) {
    const int own = tensor_buffer_own_slot(buf);
    if (own >= 0 || !buf || gst_buffer_n_memory(buf) < 1) return own;
    GstMemory* mem = gst_buffer_peek_memory(buf, 0);
    const gpointer p = gst_mini_object_get_qdata(GST_MINI_OBJECT_CAST(mem), slot_quark());
    return p ? GPOINTER_TO_INT(p) - 1 : -1;
}

int tensor_buffer_fd(GstBuffer* buf) {
    if (!buf || gst_buffer_n_memory(buf) < 1) return -1;
    GstMemory* mem = gst_buffer_peek_memory(buf, 0);
    return gst_is_fd_memory(mem) ? gst_fd_memory_get_fd(mem) : -1;
}

void tensor_memory_set_slot(GstMemory* mem, int slot) {
    gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(mem), slot_quark(), GINT_TO_POINTER(slot + 1), nullptr);
}

void tensor_buffer_set_slot(GstBuffer* buf, int slot) {
    gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(buf), slot_quark(), GINT_TO_POINTER(slot + 1), nullptr);
}

} // namespace yvo::gst
