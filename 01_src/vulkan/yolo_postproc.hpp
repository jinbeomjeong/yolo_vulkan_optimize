// YoloPostproc: NPU raw output, 6 tensors (UINT8 NHCW) → compacted candidate detections (02_shaders/yolo_decode.comp).
//               The GPU half of plan §4.3 "pass 1" + §4.4 hybrid option (A).
//
// The execution path follows the `SandLetterbox`/`DisplayConvert` template as-is (slot ring buffer · barriers · spec constants ·
// push constants + static_assert · per-slot timestamps · external SPIR-V injection via environment variable).
// `SandLetterbox` itself is the tuned 2.13 ms path, so it is left untouched — this class was written from scratch.
//
// ── What it does and what it does not do ────────────────────────────────────────────────────────────
// Does    : dequantize → DFL softmax → threshold → inverse letterbox → **compact candidates in original coordinates**
// Optional: **NMS suppression** (`enable_nms`). When on, the owned `EfficientNms` (02_shaders/efficient_nms.comp)
//          runs after the 3 decode dispatches **in the same command buffer** (1 barrier: DetBuf compute write → read).
//          The result is bit-identical to `cpu_nms_greedy_by_class()` (01_src/ref/cpu_yolov8_decode.hpp), and
//          frames for which equality cannot be guaranteed are signalled by `nms_fallback_reason()` != 0 → the caller handles them on the CPU.
//          When off, it emits only candidates as before and the caller does suppression on the CPU (`--nms cpu`).
//
// ── Tensor indexing convention ──────────────────────────────────────────────────────────────────────────
// The tensors are **numbered 0..5 (6 of them)**: for scale s (0=P3/stride 8, 1=P4/16, 2=P5/32),
//   class tensor = `yolo_cls_tensor(s)` = 2s,  box (DFL) tensor = `yolo_box_tensor(s)` = 2s+1.
// This number is the tensor argument of `raw_dmabuf_fd(slot, tensor)` / `raw_host_ptr(slot, tensor)`.
// The mapping between HailoRT stream names ↔ these numbers is decided **by the caller** when filling `YoloTensorLayout`
// (in this repository `HailoRawRunner` derives each HEF stream's scale/role from its stride and feature count and stores it
// in `RawModelInfo::t[6]` by this numbering; `layout_from_runner()` in 01_src/pipeline/gpu_postproc_stage.cpp reads it).
//
// ── Descriptors (plan §4.3) ──────────────────────────────────────────────────────────────────
// A single dispatch sees **only 3** SSBOs, `{cls_i, box_i, det}` (`maxPerStageDescriptorStorageBuffers = 8`).
// Descriptor sets are **slots x scales = 3 per slot, all static** — updated only once at initialization, and
// the frame loop only changes bindings. There are no barriers between the 3 dispatches (only det.count's device-scope atomic).
//
// ── dma-buf ownership direction (plan §4.2, appendix A.13) ────────────────────────────────────────────────
// With `input_alloc = ExportDmaBuf`, **Vulkan allocates and exports** (16384 alignment, appendix A.4).
// That fd is taken out with `raw_dmabuf_fd()` and Hailo maps it with `dma_map_dmabuf(fd, size, D2H)`. Size contract:
//   · `dma_map_dmabuf` takes `raw_dmabuf_size()` (= the whole dma-buf. hw_frame_size also works, but the whole is safer)
//   · `read_async` / `hailo_dma_buffer_t.size` take **only `raw_size()` (= hw_frame_size).**
//     ⚠ Passing a wrong size to `infer_async` returns HAILO_SUCCESS and **never invokes a single callback.**
// fd ownership belongs to this object (the internal `DmaBufExporter`). The caller does not close it.
//
// ── Synchronization (`input_sync`) ─────────────────────────────────────────────────────────────────────
//   Foreign   : the raw output is **external memory written by Hailo PCIe DMA** →
//               a `FOREIGN_EXT → compute` acquire before the command buffer, a `compute → FOREIGN_EXT` release after.
//               Same pattern as `sand_letterbox.cpp:274-298 / :312-356` (both **only read**).
//   HostWrite : fixture mode. Makes contents memcpy'd by the host available via `HOST_WRITE → SHADER_STORAGE_READ`.
//   None      : **inserts no barrier at all.** Only for plan §6.2 P3-④ (evidence that the barrier is required).
//               Not used in production.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "ref/cpu_yolov8_decode.hpp"
#include "vulkan/efficient_nms.hpp"
#include "core/dma_heap_alloc.hpp"
#include "core/frame_types.hpp"

namespace yvo::vk {

class VulkanContext;
class DmaBufExporter;
class DmaBufImportCache;

inline constexpr int kYoloNumScales = 3;
inline constexpr int kYoloNumTensors = 2 * kYoloNumScales;

inline int yolo_cls_tensor(int scale_id) { return 2 * scale_id; }
inline int yolo_box_tensor(int scale_id) { return 2 * scale_id + 1; }

// Geometry and quantization parameters of one scale. Same meaning as `RawScale` in `cpu_yolov8_decode.hpp`.
struct YoloScaleSpec {
    int grid_h = 0;               // number of valid rows (80 / 40 / 20)
    int grid_w = 0;               // number of valid columns (80 / 40 / 20)
    int w_padded = 0;             // number of columns for address computation (80 / 40 / 24)
    int stride = 0;               // 8 / 16 / 32
    std::size_t cls_bytes = 0;    // class tensor hw_frame_size
    std::size_t box_bytes = 0;    // box tensor hw_frame_size
    float cls_scale = 1.0f;
    float cls_zp = 0.0f;
    float box_scale = 1.0f;
    float box_zp = 0.0f;
};

struct YoloTensorLayout {
    YoloScaleSpec s[kYoloNumScales];   // P3, P4, P5 (stride 8, 16, 32)
    int num_classes = 80;
    int reg_max = kDflRegMax;          // 16. The shader holds it as a compile-time constant; a mismatch with this value throws.
};

// How the raw output buffers are allocated
enum class YoloInputAlloc {
    Plain,          // ordinary host-visible STORAGE buffer (fixture mode. no dma-buf)
    ExportDmaBuf,   // DmaBufExporter (16384 alignment, export). Hailo maps it as D2H
    // Allocated from dma-heap (`/dev/dma_heap/system`) and **imported** into Vulkan — plan appendix B step 4.
    //   · allocation, mmap and CPU cache sync are done by `yvo::DmaHeapPool` (core/dma_heap_alloc.hpp)
    //   · `t.host` is **the mmap pointer of the heap fd** (not `vkMapMemory` — it can fail on imported memory)
    //   · allocation size is `align_up(hw + device_headroom, 16384)` (B.12). The headroom is obtained by query in
    //     `vk_storage_headroom.hpp` — no hard-coded constant
    //   · imported with `DmaBufImportCache::get(fd, dmabuf_size, min_range = hw)`; throws if range < hw
    // **The default path is still Plain/ExportDmaBuf.** This is an option, and reverting is this one enum value (B.14 S5).
    ImportHeap,
};

// How visibility of the raw output buffers is ensured
enum class YoloInputSync {
    Foreign,     // FOREIGN_EXT ↔ compute acquire/release (default for the live path)
    HostWrite,   // HOST_WRITE → SHADER_STORAGE_READ (fixture path)
    None,        // no barrier — only for the P3-④ experiment
};

struct YoloPostprocConfig {
    int num_slots = 2;
    // Candidate cap. Measured candidates per frame are around 15 (P1: 15.1/frame), but it is set generously for the worst case (all anchors pass).
    // 32 B * 4096 = 128 KB/slot.
    int max_dets = 4096;
    // Workgroup size (spec constants 0/1). x is **word columns** (=4 anchors), y is rows. x*y <= 256.
    // Measurement basis (`yvo_postproc_probe --probe sweep`, fixture 5 frames as a continuous stream of 120 frames,
    //   3 interleaved runs, NPU idle. pass 1 median ms):
    //   20(x)x4(y)=20 inv 0.837 | 10x2=20 0.857 | 16x2=32 **2.281** | 8x4=32 **1.782**
    //   10x4=40 0.846 | 5x8=40 0.896 | **12x5=60 0.560** | **10x6=60 0.566 ← adopted**
    //   5x12=60 0.586 | 16x4=64 1.357 | 8x8=64 1.093 | 10x8=80 0.902 | **10x12=120 0.572**
    //   16x8=128 0.898 | 10x16=160 0.697 | 24x10=240 0.625 | 20x12=240 0.670 | 16x16=256 0.733
    //   32x8=256 0.755 | 40x6=240 0.820 | 64x4=256 0.987 | 80x3=240 1.269
    // ⚠ **Two things are the opposite of what the plan expected. Only the measurements are evidence; the mechanism is not confirmed.**
    //   ① `sand_letterbox`'s "larger invocation counts are better" (64 inv 3.65 → 256 inv 2.13 ms)
    //      does not hold here. 60 invocations are 1.3x faster than 256. Because the total work is small
    //      (2100 valid invocations), large workgroups drop the workgroup count to as few as 8, spreading less across cores —
    //      that is the **presumption** (10x6 gives 28 workgroups at P3). It is an unconfirmed hypothesis.
    //   ② Attempting to make reads within a warp (16 lanes) longer and more contiguous by enlarging x is **counterproductive**
    //      (80x3 is 1.269 ms). w_words is only 20/10/5, so all lanes with x > w_words return early,
    //      and that waste outweighs the coalescing gain.
    //   ③ 32·64·128 invocations (warp multiples) are **slower** than 20·40·60·120. Not explained.
    //      Must be re-measured if the resolution or number of classes changes (it is a spec constant, so no recompilation is needed).
    std::uint32_t local_size_x = 10;
    std::uint32_t local_size_y = 6;
    float score_thr = 0.2f;            // same value as the HEF's YOLOV8 post-process setting
    YoloInputAlloc input_alloc = YoloInputAlloc::Plain;
    YoloInputSync input_sync = YoloInputSync::HostWrite;
    // Meaningful only for ImportHeap. If true, CPU access is wrapped with `DMA_BUF_IOCTL_SYNC` (appendix B.10 R2/R3).
    // **false is for verification only** (the negative control of V2/V3). Do not turn it off in production —
    // the device tree has 0 `dma-coherent` entries, so the kernel treats V3D as a non-coherent device.
    bool heap_sync = true;
    // If true, also creates a **second pipeline** with the kChecksum spec constant enabled (verification only).
    // Being a spec constant, it does not add instructions to the normal path.
    bool checksum_pipeline = false;
    bool timestamps = true;
    bool verbose = false;
    // GPU NMS (efficient_nms.comp). If false, EfficientNms is not created (0 instructions, 0 memory).
    bool enable_nms = false;
    EfficientNmsConfig nms;
};

// Shader push constants (std430, same order and size as Params in 02_shaders/yolo_decode.comp)
struct YoloDecodePush {
    std::int32_t grid_h;
    std::int32_t grid_w;
    std::int32_t w_words;
    std::int32_t w_words_pad;
    std::int32_t num_classes;
    std::int32_t scale_id;
    std::uint32_t q_thr_low;
    std::uint32_t max_dets;
    float stride;
    float cls_scale;
    float cls_zp;
    float box_scale;
    float box_zp;
    float thr;
    float inv_scale;
    float pad_x;
    float pad_y;
    float max_x;
    float max_y;
};
static_assert(sizeof(YoloDecodePush) == 76, "push constant layout must match shader");

// ⚠ The two structs below must be **byte-for-byte identical** to `OverlayDetBufHeader`/`DetGpu` in
//   `01_src/display/overlay_renderer.hpp`. Only then can this buffer be passed as-is to the overlay via `vkCmdDispatchIndirect`.
//   The reason for writing the same definition twice is **link-level isolation** (plan §5.2 safeguard 4):
//   `yvo_display` depends on `yvo_vulkan`, so an include in the opposite direction cannot be made.
//   Offsets are pinned down too, so that compilation breaks if the static_asserts of the two headers diverge.
struct YoloDetBufHeader {
    std::uint32_t dispatch[3];   // VkDispatchIndirectCommand compatible (offset 0)
    std::uint32_t count;         // total emit **attempts**. May exceed max_dets (see overflow)
    std::uint32_t overflow;      // whether the cap was exceeded
    std::uint32_t text_bytes;    // overlay HUD length. The decode path leaves it 0
    std::uint32_t text[32];      // overlay HUD ASCII. Reused as byte-sum storage only in kChecksum mode
};
static_assert(sizeof(YoloDetBufHeader) == 152, "DetBuf header layout must match shader (std430)");
static_assert(offsetof(YoloDetBufHeader, count) == 12, "dispatch must be uvec3 at offset 0");
static_assert(offsetof(YoloDetBufHeader, overflow) == 16, "overflow offset must match shader");
static_assert(offsetof(YoloDetBufHeader, text_bytes) == 20, "text_bytes offset must match shader");
static_assert(offsetof(YoloDetBufHeader, text) == 24, "text offset must match shader");

struct YoloDetGpu {
    float x1, y1, x2, y2;
    float score;
    std::uint32_t class_id;
    std::uint32_t anchor_key;
    std::uint32_t keep;
};
static_assert(sizeof(YoloDetGpu) == 32, "DetGpu must be 32 B (contract shared with the overlay track)");
static_assert(offsetof(YoloDetGpu, score) == 16, "DetGpu.score offset must match shader");
static_assert(offsetof(YoloDetGpu, class_id) == 20, "DetGpu.class_id offset must match shader");
static_assert(offsetof(YoloDetGpu, anchor_key) == 24, "DetGpu.anchor_key offset must match shader");
static_assert(offsetof(YoloDetGpu, keep) == 28, "DetGpu.keep offset must match shader");

class YoloPostproc {
public:
    YoloPostproc(VulkanContext& ctx, const YoloTensorLayout& layout, const YoloPostprocConfig& cfg = {});
    ~YoloPostproc();
    YoloPostproc(const YoloPostproc&) = delete;
    YoloPostproc& operator=(const YoloPostproc&) = delete;

    // Feeds the slot's 6 raw outputs to pass 1. Header init (host write) → 3 dispatches → submit.
    // If checksum = true, runs with the kChecksum pipeline (checksum_pipeline required).
    void submit(int slot, const LetterboxParams& lb, int src_w, int src_h, bool checksum = false);
    bool ready(int slot) const;                                  // whether the fence has finished (does not wait)
    void wait(int slot, std::uint64_t timeout_ns = UINT64_MAX);   // fence wait + read timestamps

    // Call after wait(). Returns `min(count, max_dets)` entries (the shader count is the total attempts).
    std::vector<DetectionKeyed> read_candidates(int slot) const;
    std::uint32_t raw_count(int slot) const;      // count written by the shader (greater than max_dets when the cap is exceeded)
    std::uint32_t overflow(int slot) const;
    // Byte sums in kChecksum mode. out[2s] = class tensor of scale s, out[2s+1] = box tensor.
    void checksums(int slot, std::uint32_t out[kYoloNumTensors]) const;

    // ── CPU access section for ImportHeap (appendix B.10 R2/R3) ──────────────────────────────────────
    // Applies `DMA_BUF_IOCTL_SYNC` to all 6 raw outputs of the slot. In other alloc modes or
    // with `heap_sync = false` it **does nothing** (reverting is the value of a single flag — B.14 S5).
    //   R2 CPU read: **after** the completion signal, begin(Read) → read → end(Read)
    //   R3 CPU write: begin(Write) → memcpy/memset → end(Write). **Initialization is no exception**
    void raw_cpu_sync_begin(int slot, yvo::DmaSyncDir dir);
    void raw_cpu_sync_end(int slot, yvo::DmaSyncDir dir);
    bool raw_heap_sync_active() const {
        return cfg_.input_alloc == YoloInputAlloc::ImportHeap && cfg_.heap_sync;
    }
    // Driver headroom actually applied in ImportHeap (the B.12 query value). 0 in other modes.
    std::size_t heap_headroom() const { return heap_headroom_; }

    // ── Interface to hand to Hailo ──
    // dma-buf fd valid only in ExportDmaBuf mode. **Ownership belongs to this object** (do not close it).
    int raw_dmabuf_fd(int slot, int tensor) const;
    std::size_t raw_dmabuf_size(int slot, int tensor) const;   // dma-buf size measured with lseek
    std::size_t raw_size(int tensor) const;                    // tensor size = hw_frame_size
    std::uint8_t* raw_host_ptr(int slot, int tensor) const;    // persistent mapping (16384 alignment)

    // ── GPU NMS (only when enable_nms. Otherwise read_nms is false and the reason is 0) ──
    // Call after wait(). If false, out is empty and that frame must be processed on the CPU.
    bool read_nms(int slot, std::vector<DetectionKeyed>& out) const;
    std::uint32_t nms_fallback_reason(int slot) const;   // kNmsFallback* bits
    bool nms_enabled() const { return nms_ != nullptr; }
    const EfficientNms* nms() const { return nms_.get(); }

    double last_pass1_ms() const { return last_pass1_ms_; }   // the 3-dispatch interval
    double last_nms_ms() const { return last_nms_ms_; }       // NMS dispatch interval (-1 when off)
    double last_total_ms() const { return last_total_ms_; }   // whole command buffer (including barriers)

    int num_slots() const { return static_cast<int>(slots_.size()); }
    const YoloPostprocConfig& config() const { return cfg_; }
    const YoloTensorLayout& layout() const { return layout_; }
    // Integer prefilter bound `floor(thr/cls_scale)`. Appendix A.5: P3 57 / P4 53 / P5 52.
    std::uint32_t q_thr_low(int scale_id) const;

private:
    struct Tensor {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;   // owned only in Plain mode. In ExportDmaBuf the exporter holds it
        std::uint8_t* host = nullptr;
        int fd = -1;
        std::size_t dmabuf_size = 0;
        VkDeviceSize range = 0;
        bool from_heap = false;   // ImportHeap: buf/mem are owned by the import cache, ptr/fd by the heap pool
    };
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkDescriptorSet set[kYoloNumScales]{};
        Tensor t[kYoloNumTensors];
        VkBuffer det = VK_NULL_HANDLE;
        VkDeviceMemory det_mem = VK_NULL_HANDLE;
        std::uint8_t* det_host = nullptr;
        bool pending = false;
    };

    void validate();
    void setup_heap();
    void create_pipelines();
    void create_slots();
    void record(Slot& s, int slot, const LetterboxParams& lb, int src_w, int src_h, bool checksum);
    void destroy();
    Tensor alloc_tensor(std::size_t bytes);
    const YoloDetBufHeader* header(int slot) const;

    VulkanContext& ctx_;
    YoloTensorLayout layout_;
    YoloPostprocConfig cfg_;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipeline pipeline_sum_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    std::unique_ptr<DmaBufExporter> exporter_;
    // ImportHeap only. Destruction order matters — empty the import cache (VkBuffer/VkDeviceMemory) first,
    // then release the heap pool (mmap/fd). destroy() keeps that order.
    std::unique_ptr<yvo::DmaHeapPool> heap_;
    std::unique_ptr<DmaBufImportCache> import_cache_;
    // Holds the slot DetBufs as input descriptors → destroy() destroys it before the slots.
    std::unique_ptr<EfficientNms> nms_;
    std::size_t heap_headroom_ = 0;
    std::vector<Slot> slots_;
    VkDeviceSize det_size_ = 0;
    double last_pass1_ms_ = -1.0;
    double last_total_ms_ = -1.0;
    double last_nms_ms_ = -1.0;
};

} // namespace yvo::vk
