// Per-buffer tensor metadata carried by yvosandletterbox output buffers (research path, stage 2a)
//
// Output buffer contract (application/x-yvo-tensor, layout=nhcw, format=RGB8, width=640, height=640):
//   - exactly one GstMemory: a GstDmaBufMemory wrapping the ring slot's exported Vulkan dma-buf.
//     memory size = tensor bytes (640*640*3 = 1,228,800, the logical tensor starts at offset 0),
//     memory maxsize = true dma-buf size (lseek). The fd is owned by the element (GST_FD_MEMORY_FLAG_DONT_CLOSE)
//     and is the SAME fd number for a given slot from start() until the element's resources are released (stop(),
//     or the last slot buffer returned after stop()) → a consumer may map it once per slot and must unmap it
//     before it releases its last slot buffer / goes to NULL.
//   - the slot index is stored on the buffer and on the memory (tensor_buffer_slot) and in the meta below.
//   - the GPU has finished writing and the buffer was released to the external/FOREIGN queue family before the
//     buffer is pushed (SandLetterbox::wait), so a PCIe DMA reader needs no further Vulkan synchronization.
//   - releasing the buffer (gst_buffer_unref) returns the slot to the element's pool; holding it blocks the element.
// Consumer rules (a slot is reused as soon as its buffer is back in the pool):
//   - hold the GstBuffer itself (not only its GstMemory) until the tensor is consumed (e.g. until the HailoRT
//     completion callback); do not move/share the slot memory into another buffer. If that happens anyway the slot is
//     never overwritten: the element waits for that memory and then fails with "slot memory still referenced
//     downstream", and the dma-buf/fd stay alive until the last reference to the memory dies.
//   - the fd number is valid only while a slot buffer (or its memory) is held. A device mapping made from it
//     (HailoRT dma_map_dmabuf) must be unmapped before the consumer releases its last slot buffer and before the
//     element goes to READY.
//
// The public header stays free of GStreamer includes: GstBuffer is forward-declared and the meta is reached through
// accessor functions. The API GType is exposed as std::size_t (== GType on this ABI; static_assert in the .cpp) so a
// sink can advertise the meta in its ALLOCATION answer (gst_query_add_allocation_meta).
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/frame_types.hpp"

typedef struct _GstBuffer GstBuffer;
typedef struct _GstMemory GstMemory;

namespace yvo::gst {

// Trivially copyable: stored by value inside the GstMeta.
struct TensorFrameInfo {
    int slot = -1;                        // ring slot index, 0..num_slots-1
    int num_slots = 0;                    // ring size (element property "slots")
    std::uint64_t seq = 0;                // frame sequence number counted by the element (0-based, input order)
    std::int64_t src_pts_ns = -1;         // PTS of the decoded input buffer (ns, -1 if none). Not a frame index
    int src_width = 0;                    // display size of the source frame (1920)
    int src_height = 0;                   // (1080)
    LetterboxParams lb;                   // scale, pad_x, pad_y, dst 640x640 (make_letterbox(src_w, src_h))
    TensorLayout layout = TensorLayout::Nhcw;
    std::uint32_t tensor_bytes = 0;       // logical tensor size (1,228,800)
    std::uint64_t dmabuf_bytes = 0;       // true dma-buf size of the slot (lseek)
    int dmabuf_fd = -1;                   // slot fd (element-owned, stable per slot; do not close)
    double gpu_ms = -1.0;                 // GPU time of this frame's dispatch (timestamp query), < 0 if unavailable
    std::uint64_t arrival_mono_ns = 0;    // CLOCK_MONOTONIC when the input buffer reached the element
    std::uint64_t acquire_wait_ns = 0;    // time blocked waiting for a free output slot (ring backpressure)
    std::uint64_t gpu_done_mono_ns = 0;   // CLOCK_MONOTONIC right after the GPU fence wait returned
};

// Meta API GType (registered once, thread-safe). Name: "YvoTensorMetaAPI".
std::size_t tensor_meta_api_type();

// Adds (or overwrites an existing) tensor meta on buf. buf must be writable. Returns false if that fails.
bool tensor_meta_set(GstBuffer* buf, const TensorFrameInfo& info);
// Reads the tensor meta. Returns false if the buffer has none.
bool tensor_meta_get(GstBuffer* buf, TensorFrameInfo& out);

// Slot index of a yvosandletterbox slot buffer (-1 otherwise): the buffer's own tag, else the first memory's tag.
// Works even without the meta (e.g. before transform() filled it).
int tensor_buffer_slot(GstBuffer* buf);
// fd of the buffer's first memory if it is fd/dma-buf memory, else -1 (not owned by the caller).
int tensor_buffer_fd(GstBuffer* buf);

// Used by the element's pool: tag a slot memory / slot buffer with its slot index (qdata, survives pool recycling:
// GstBufferPool's reset_buffer does not touch qdata).
void tensor_memory_set_slot(GstMemory* mem, int slot);
void tensor_buffer_set_slot(GstBuffer* buf, int slot);
// Slot tag of the buffer object itself only (-1 if none). The pool uses this, so replaced memories cannot hide a slot.
int tensor_buffer_own_slot(GstBuffer* buf);

} // namespace yvo::gst
