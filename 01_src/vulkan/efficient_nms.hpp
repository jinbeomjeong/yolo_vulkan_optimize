// EfficientNms: candidate DetBuf → per-class greedy NMS result DetBuf (02_shaders/efficient_nms.comp).
//
// Ports the structure of TensorRT EfficientNMS (Filter → Sort → tile + state-broadcast NMS → write results,
// `08_learn/11_tensorrt_efficient_nms.md`) to V3D compute. **The semantics are not TRT's but this repository's
// `cpu_nms_greedy_by_class()` (01_src/ref/cpu_yolov8_decode.hpp)** — IoU >= 0.7, per-class suppression,
// 100 per class, (score descending, anchor_key ascending) total order. The result is bit-identical to that function.
// Frames that might not match are flagged by the shader via overflow bits, and the caller processes those frames on the CPU.
//
// ── TRT mapping and what was not adopted ──────────────────────────────────────────────────────
//   Filter(§5.1)     : the atomicAdd emit in `yolo_decode.comp` already is the Filter.
//   DenseIndex(§5.2) : **not adopted.** It is a low-threshold path meant to avoid atomic contention, but here there are
//                      about 17 candidates per frame on average, so there is no atomicAdd contention. Also, passing every anchor (8400 x 80) to the sort
//                      would exceed kMaxSelected (the shared 32 KB limit), causing a CPU fallback every frame.
//   Sort(§5.3)       : a shared bitonic sort in a single workgroup instead of CUB radix (N padded to a power of two).
//   numSelected(4096): kMaxSelected = 1024 (shared budget). **Does not truncate** — if exceeded, overflow bit 1 → CPU.
//   Main kernel(§5.4): 1 workgroup = 1 frame. kTiles = ceil(kMaxSelected/local) candidates per invocation,
//                      and the leader broadcasts blockState (1 / -1 / -2).
//
// ── shared layout (kMaxSelected = 1024, local 256: 24.9 KB) ───────────────────────────────────
//   s_a, s_b   : sort keys (score bits, anchor_key) → reused as x1, y1 bits after sort  2 x 4 KB
//   s_idx      : sorted position → input index                                        4 KB
//   s_x2, s_y2 : rest of the box                                                      2 x 4 KB
//   s_st       : class | output position | selected | dropped                         4 KB
//   s_kept     : survivor count per class                                             kNumClasses x 4 B
// At creation `shared_bytes()` is compared against maxComputeSharedMemorySize, and exceeding it throws
// (VUID-RuntimeSpirv-Workgroup-06530, validusage.json Vulkan 1.4.309).
//
// ── Rationale for the IoU guard band G (iou_guard) ────────────────────────────────────────────
// With this repository's flags (-O3 -std=c++20 -mcpu=cortex-a76, GCC 14 default -ffp-contract=fast), CPU `iou_xyxy`
// contracts the first addition of `ua` into fmadd (verified with `c++ -S`: fmul areaB → fmadd areaA + → fsub inter → fdiv).
// The GPU does not guarantee fma as a single rounding (spec "Inherited from OpFMul followed by OpFAdd"),
// and division is 2.5 ULP. So the shader decides by the sign of `inter - (thr ± G) * ua` without dividing.
// Bound (u = 2^-24; 2u per operation so it holds whether the rounding mode is RTE or RTZ):
//   · ix, iy, inter, the two side lengths, and areaB have identical bits on both sides (min/max exact, sub/mul correctly rounded).
//   · s = areaA + areaB is within 4u of the exact value on both sides (no negative terms). Since inter <= min(areaA, areaB),
//     s - inter >= s/2 → cancellation amplification <= 2 → ua is within about 10u of the exact value.
//   · CPU iou = fl(inter/ua_c) is within about 22u of inter/ua_g, the shader's d computation error is about 4u → total about 26u ≈ 1.6e-6.
// G = 4e-6 (about 67u) gives a 2.5x margin. If even one pair with |iou - thr| <= G exists among **the pairs the CPU actually compares** (survivor i and
// a live j of the same class), that frame gets overflow bit 2 → CPU fallback.
// Areas below 1e-30, near denormals (where FTZ behavior differs), are sent to the same guard band.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "ref/cpu_yolov8_decode.hpp"

namespace yvo::vk {

class VulkanContext;

// fallback reason bits the shader writes to out_overflow
inline constexpr std::uint32_t kNmsFallbackCapacity = 1u;   // candidates > kMaxSelected or decode DetBuf overflow
inline constexpr std::uint32_t kNmsFallbackIouGuard = 2u;   // some pair had IoU inside the guard band
inline constexpr std::uint32_t kNmsFallbackBadClass = 4u;   // class_id >= num_classes

struct EfficientNmsConfig {
    std::uint32_t local_size = 256;     // spec 0. <= maxComputeWorkGroupInvocations
    std::uint32_t max_selected = 1024;  // spec 1. power of two, <= 4096. Determines the shared budget
    std::uint32_t num_classes = 80;     // spec 2
    bool score_sigmoid = false;         // spec 3. scores are already probabilities
    bool class_agnostic = false;        // spec 4
    float iou_thr = 0.7f;               // HEF setting
    float iou_guard = 4e-6f;            // G from the comment above
    int max_per_class = 100;            // HEF setting
    int max_output = 0;                 // 0 = unlimited (same as the CPU reference). >0 diverges from the CPU reference
};

// push constant (std430, same order as Params in efficient_nms.comp)
struct EfficientNmsPush {
    float iou_thr;
    float iou_lo;
    float iou_hi;
    std::uint32_t max_per_class;
    std::uint32_t max_output;
    std::uint32_t n_max;
};
static_assert(sizeof(EfficientNmsPush) == 24, "push constant layout must match shader");

class EfficientNms {
public:
    // inputs[slot] = the slot's decode DetBuf (YoloDetBufHeader + YoloDetGpu x input_max_dets).
    // Buffer ownership stays with the caller, and the buffers must outlive this object.
    EfficientNms(VulkanContext& ctx, const EfficientNmsConfig& cfg, const std::vector<VkBuffer>& inputs,
                 VkDeviceSize input_size, std::uint32_t input_max_dets);
    ~EfficientNms();
    EfficientNms(const EfficientNms&) = delete;
    EfficientNms& operator=(const EfficientNms&) = delete;

    // Records one dispatch (one workgroup). **Barriers are the caller's job**:
    //   before: shader writes to the input DetBuf → SHADER_STORAGE_READ
    //   after: shader writes to output(slot) → consumer (HOST_READ or indirect)
    void record(VkCommandBuffer cb, int slot) const;

    // Call after the fence wait. If there is a fallback reason, returns false and clears out.
    bool read_results(int slot, std::vector<DetectionKeyed>& out) const;
    std::uint32_t fallback_reason(int slot) const;   // if 0, use the GPU result as is
    std::uint32_t result_count(int slot) const;

    VkBuffer output(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).out; }
    VkDeviceSize output_size() const { return out_size_; }
    std::size_t shared_bytes() const;   // sum of the shader's shared declarations (compared against the limit at creation)
    const EfficientNmsConfig& config() const { return cfg_; }

private:
    struct Slot {
        VkBuffer out = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        std::uint8_t* host = nullptr;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    void validate();
    void create_pipeline();
    void create_slots(const std::vector<VkBuffer>& inputs, VkDeviceSize input_size);
    void destroy();

    VulkanContext& ctx_;
    EfficientNmsConfig cfg_;
    std::uint32_t input_max_dets_ = 0;
    EfficientNmsPush push_{};
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    std::vector<Slot> slots_;
    VkDeviceSize out_size_ = 0;
};

} // namespace yvo::vk
