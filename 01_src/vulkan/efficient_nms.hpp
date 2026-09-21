// EfficientNms: 후보 DetBuf → 클래스별 greedy NMS 결과 DetBuf (02_shaders/efficient_nms.comp).
//
// TensorRT EfficientNMS 의 구조(Filter → Sort → 타일+상태 방송 NMS → 결과 기록,
// `08_learn/11_tensorrt_efficient_nms.md`)를 V3D 컴퓨트로 옮겼다. **의미론은 TRT 가 아니라 이 저장소의
// `cpu_nms_greedy_by_class()`(01_src/ref/cpu_yolov8_decode.hpp) 다** — IoU >= 0.7, 클래스별 억제,
// 클래스당 100개, (score 내림차순, anchor_key 오름차순) 전순서. 결과는 그 함수와 비트 단위로 같다.
// 같지 않을 수 있는 프레임은 셰이더가 overflow 비트로 알리고, 호출자가 그 프레임을 CPU 로 처리한다.
//
// ── TRT 대응과 채택하지 않은 것 ───────────────────────────────────────────────────────────────
//   Filter(§5.1)     : `yolo_decode.comp` 의 atomicAdd emit 이 이미 Filter 다.
//   DenseIndex(§5.2) : **채택하지 않았다.** atomic 경쟁을 피하려는 저임계값 경로인데, 여기서는 후보가
//                      프레임당 평균 17개 수준이라 atomicAdd 경쟁이 없다. 또 모든 앵커(8400 x 80)를 정렬에
//                      넘기면 kMaxSelected(shared 32 KB 한도) 를 넘어 매 프레임 CPU fallback 이 된다.
//   Sort(§5.3)       : CUB radix 대신 워크그룹 하나의 shared bitonic sort (N 을 2의 거듭제곱으로 패딩).
//   numSelected(4096): kMaxSelected = 1024 (shared 예산). **자르지 않는다** — 넘으면 overflow 비트 1 → CPU.
//   메인 커널(§5.4)  : 워크그룹 1개 = 프레임 1장. invocation 당 kTiles = ceil(kMaxSelected/local) 후보,
//                      리더가 blockState(1 / -1 / -2)를 방송한다.
//
// ── shared 배치 (kMaxSelected = 1024, local 256 기준 24.9 KB) ─────────────────────────────────
//   s_a, s_b   : 정렬 키(score 비트, anchor_key) → 정렬 뒤 x1, y1 비트로 재사용       2 x 4 KB
//   s_idx      : 정렬 위치 → 입력 인덱스                                              4 KB
//   s_x2, s_y2 : 박스 나머지                                                          2 x 4 KB
//   s_st       : class | 출력 위치 | 선택 | 탈락                                      4 KB
//   s_kept     : 클래스별 생존 수                                                     kNumClasses x 4 B
// 생성 시 `shared_bytes()` 를 maxComputeSharedMemorySize 와 비교해 넘으면 예외다
// (VUID-RuntimeSpirv-Workgroup-06530, validusage.json Vulkan 1.4.309).
//
// ── IoU 보호대 G (iou_guard) 의 근거 ──────────────────────────────────────────────────────────
// CPU `iou_xyxy` 는 이 저장소의 플래그(-O3 -std=c++20 -mcpu=cortex-a76, GCC 14 기본 -ffp-contract=fast)
// 에서 `ua` 의 첫 덧셈이 fmadd 로 축약된다(`c++ -S` 로 확인: fmul areaB → fmadd areaA + → fsub inter → fdiv).
// GPU 는 fma 를 단일 반올림으로 보장하지 않고(스펙 "Inherited from OpFMul followed by OpFAdd"),
// 나눗셈은 2.5 ULP 다. 그래서 셰이더는 나눗셈 없이 `inter - (thr ± G) * ua` 의 부호로 판정한다.
// 상한 (u = 2^-24, 반올림 모드가 RTE/RTZ 어느 쪽이어도 성립하도록 연산당 2u 로 잡는다):
//   · ix, iy, inter, 두 변 길이, areaB 는 양쪽이 같은 비트다(min/max 는 정확, sub/mul 은 correctly rounded).
//   · s = areaA + areaB 는 양쪽 다 정확값의 4u 이내(음수 항이 없다). inter <= min(areaA, areaB) 라
//     s - inter >= s/2 → 상쇄 증폭 <= 2 → ua 는 정확값의 약 10u 이내.
//   · CPU iou = fl(inter/ua_c) 는 inter/ua_g 와 약 22u 이내, 셰이더의 d 계산 오차 약 4u → 합 약 26u ≈ 1.6e-6.
// G = 4e-6 (약 67u) 이면 여유 2.5배다. |iou - thr| <= G 인 쌍이 **CPU 가 실제로 비교하는 쌍**(생존자 i 와
// 살아 있는 같은 클래스 j) 중에 하나라도 있으면 그 프레임은 overflow 비트 2 → CPU fallback 이다.
// 면적이 1e-30 미만인 비정규수 근처(FTZ 여부가 갈린다)도 같은 보호대로 보낸다.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "ref/cpu_yolov8_decode.hpp"

namespace yvo::vk {

class VulkanContext;

// 셰이더가 out_overflow 에 쓰는 fallback 사유 비트
inline constexpr std::uint32_t kNmsFallbackCapacity = 1u;   // 후보 > kMaxSelected 또는 디코드 DetBuf overflow
inline constexpr std::uint32_t kNmsFallbackIouGuard = 2u;   // IoU 가 보호대 안에 든 쌍이 있었다
inline constexpr std::uint32_t kNmsFallbackBadClass = 4u;   // class_id >= num_classes

struct EfficientNmsConfig {
    std::uint32_t local_size = 256;     // spec 0. <= maxComputeWorkGroupInvocations
    std::uint32_t max_selected = 1024;  // spec 1. 2의 거듭제곱, <= 4096. shared 예산을 정한다
    std::uint32_t num_classes = 80;     // spec 2
    bool score_sigmoid = false;         // spec 3. 점수가 이미 확률이다
    bool class_agnostic = false;        // spec 4
    float iou_thr = 0.7f;               // HEF 설정
    float iou_guard = 4e-6f;            // 위 주석의 G
    int max_per_class = 100;            // HEF 설정
    int max_output = 0;                 // 0 = 무제한 (CPU 기준과 같다). >0 이면 CPU 기준과 달라진다
};

// push constant (std430, efficient_nms.comp 의 Params 와 같은 순서)
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
    // inputs[slot] = 슬롯의 디코드 DetBuf(YoloDetBufHeader + YoloDetGpu x input_max_dets).
    // 버퍼 소유권은 호출자에 있고 이 객체보다 오래 살아야 한다.
    EfficientNms(VulkanContext& ctx, const EfficientNmsConfig& cfg, const std::vector<VkBuffer>& inputs,
                 VkDeviceSize input_size, std::uint32_t input_max_dets);
    ~EfficientNms();
    EfficientNms(const EfficientNms&) = delete;
    EfficientNms& operator=(const EfficientNms&) = delete;

    // 디스패치 1개(워크그룹 1개)를 기록한다. **배리어는 호출자 몫이다**:
    //   앞: 입력 DetBuf 의 셰이더 쓰기 → SHADER_STORAGE_READ
    //   뒤: output(slot) 의 셰이더 쓰기 → 소비자(HOST_READ 또는 indirect)
    void record(VkCommandBuffer cb, int slot) const;

    // 펜스 대기 뒤에 호출한다. fallback 사유가 있으면 false 를 돌려주고 out 은 비운다.
    bool read_results(int slot, std::vector<DetectionKeyed>& out) const;
    std::uint32_t fallback_reason(int slot) const;   // 0 이면 GPU 결과를 그대로 쓴다
    std::uint32_t result_count(int slot) const;

    VkBuffer output(int slot) const { return slots_.at(static_cast<std::size_t>(slot)).out; }
    VkDeviceSize output_size() const { return out_size_; }
    std::size_t shared_bytes() const;   // 셰이더의 shared 선언 합계(생성 시 한도와 비교한다)
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
