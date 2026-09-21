// YoloPostproc: NPU 원시 출력 6텐서(UINT8 NHCW) → 후보 검출 압축 (02_shaders/yolo_decode.comp).
//               계획서 §4.3 "pass 1" + §4.4 하이브리드안(A)의 GPU 쪽 절반.
//
// 실행 경로는 `SandLetterbox`/`DisplayConvert` 의 템플릿(슬롯 링버퍼 · 배리어 · spec constant ·
// push constant + static_assert · 슬롯당 timestamp · 환경변수 외부 SPIR-V 주입)을 그대로 따른다.
// `SandLetterbox` 자체는 튜닝된 2.13 ms 경로라 손대지 않는다 — 이 클래스는 새로 썼다.
//
// ── 무엇을 하고 무엇을 하지 않는가 ────────────────────────────────────────────────────────────
// 한다  : dequantize → DFL softmax → threshold → letterbox 역변환 → **원본 좌표 후보 압축**
// 선택  : **NMS 억제** (`enable_nms`). 켜면 소유한 `EfficientNms`(02_shaders/efficient_nms.comp)가
//          **같은 커맨드 버퍼**에서 디코드 3디스패치 뒤에 돈다(DetBuf compute write → read 배리어 1개).
//          결과는 `cpu_nms_greedy_by_class()`(01_src/ref/cpu_yolov8_decode.hpp)와 비트 단위로 같고,
//          같다고 보장할 수 없는 프레임은 `nms_fallback_reason()` != 0 으로 알린다 → 호출자가 CPU 로 처리.
//          끄면 예전대로 후보만 내고 억제는 호출자가 CPU 에서 한다(`--nms cpu`).
//
// ── 텐서 인덱싱 규약 ──────────────────────────────────────────────────────────────────────────
// 텐서는 **6개를 0..5 로 번호 매긴다**: 스케일 s(0=P3/stride 8, 1=P4/16, 2=P5/32)의
//   클래스 텐서 = `yolo_cls_tensor(s)` = 2s,  box(DFL) 텐서 = `yolo_box_tensor(s)` = 2s+1.
// 이 번호가 `raw_dmabuf_fd(slot, tensor)` / `raw_host_ptr(slot, tensor)` 의 tensor 인자다.
// HailoRT 스트림 이름 ↔ 이 번호의 대응은 **호출자가** `YoloTensorLayout` 를 채울 때 정한다
// (P1 이 덤프한 `04_data/raw_tensors/fixtures.json` 의 `scale_id`/`role` 이 그 정보다).
//
// ── 디스크립터 (계획서 §4.3) ──────────────────────────────────────────────────────────────────
// 한 디스패치가 보는 SSBO 는 `{cls_i, box_i, det}` **3개뿐**이다(`maxPerStageDescriptorStorageBuffers = 8`).
// 디스크립터 셋은 **슬롯 x 스케일 = 3개/슬롯이고 전부 정적**이다 — 초기화 때 한 번만 갱신하고
// 프레임 루프에서는 바인딩만 바꾼다. 3 디스패치 사이에 배리어가 없다(det.count 의 device-scope atomic만).
//
// ── dma-buf 소유 방향 (계획서 §4.2, 부록 A.13) ────────────────────────────────────────────────
// `input_alloc = ExportDmaBuf` 면 **Vulkan 이 할당하고 export 한다**(16384 정렬, 부록 A.4).
// 그 fd 를 `raw_dmabuf_fd()` 로 꺼내 Hailo 가 `dma_map_dmabuf(fd, size, D2H)` 로 매핑한다. 크기 계약:
//   · `dma_map_dmabuf` 에는 `raw_dmabuf_size()` (= dma-buf 전체. hw_frame_size 도 되지만 전체가 안전하다)
//   · `read_async` / `hailo_dma_buffer_t.size` 에는 **`raw_size()` (= hw_frame_size) 뿐이다.**
//     ⚠ `infer_async` 에 틀린 크기를 넘기면 HAILO_SUCCESS 를 돌려주고 **콜백을 하나도 호출하지 않는다.**
// fd 소유권은 이 객체(내부 `DmaBufExporter`)에 있다. 호출자는 close 하지 않는다.
//
// ── 동기화 (`input_sync`) ─────────────────────────────────────────────────────────────────────
//   Foreign   : 원시 출력은 **Hailo PCIe DMA 가 쓴 외부 메모리**다 →
//               커맨드 버퍼 앞에 `FOREIGN_EXT → 컴퓨트` acquire, 뒤에 `컴퓨트 → FOREIGN_EXT` release.
//               `sand_letterbox.cpp:363-387 / :401-445` 와 같은 패턴이다(둘 다 **읽기만** 한다).
//   HostWrite : 픽스처 모드. 호스트가 memcpy 한 내용을 `HOST_WRITE → SHADER_STORAGE_READ` 로 가용화한다.
//   None      : **배리어를 아예 넣지 않는다.** 계획서 §6.2 P3-④(배리어가 필수임을 보이는 증거) 전용이다.
//               프로덕션에서 쓰지 않는다.
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

// 한 스케일의 기하·양자화 파라미터. `cpu_yolov8_decode.hpp` 의 `RawScale` 과 같은 의미다.
struct YoloScaleSpec {
    int grid_h = 0;               // 유효 행 수 (80 / 40 / 20)
    int grid_w = 0;               // 유효 열 수 (80 / 40 / 20)
    int w_padded = 0;             // 주소 계산용 열 수 (80 / 40 / 24)
    int stride = 0;               // 8 / 16 / 32
    std::size_t cls_bytes = 0;    // 클래스 텐서 hw_frame_size
    std::size_t box_bytes = 0;    // box 텐서 hw_frame_size
    float cls_scale = 1.0f;
    float cls_zp = 0.0f;
    float box_scale = 1.0f;
    float box_zp = 0.0f;
};

struct YoloTensorLayout {
    YoloScaleSpec s[kYoloNumScales];   // P3, P4, P5 (stride 8, 16, 32)
    int num_classes = 80;
    int reg_max = kDflRegMax;          // 16. 셰이더가 컴파일 타임 상수로 갖고 있어 이 값과 달라지면 예외다.
};

// 원시 출력 버퍼를 어떻게 할당하는가
enum class YoloInputAlloc {
    Plain,          // 평범한 host-visible STORAGE 버퍼 (픽스처 모드. dma-buf 없음)
    ExportDmaBuf,   // DmaBufExporter (16384 정렬, export). Hailo 가 D2H 로 매핑한다
    // dma-heap(`/dev/dma_heap/system`) 에서 잡아 Vulkan 에 **import** 한다 — 계획서 부록 B 4단계.
    //   · 할당·mmap·CPU 캐시 sync 는 `yvo::DmaHeapPool`(core/dma_heap_alloc.hpp) 이 한다
    //   · `t.host` 는 **힙 fd 의 mmap 포인터**다(`vkMapMemory` 가 아니다 — import 메모리에서 실패할 수 있다)
    //   · 할당 크기는 `align_up(hw + device_headroom, 16384)` (B.12). 헤드룸은 `vk_storage_headroom.hpp`
    //     가 질의로 구한다 — 상수를 박지 않는다
    //   · `DmaBufImportCache::get(fd, dmabuf_size, min_range = hw)` 로 import 하고 range < hw 면 예외
    // **기본 경로는 여전히 Plain/ExportDmaBuf 다.** 이것은 옵션이고 되돌림은 이 enum 값 하나다(B.14 S5).
    ImportHeap,
};

// 원시 출력 버퍼의 가시성 확보 방식
enum class YoloInputSync {
    Foreign,     // FOREIGN_EXT ↔ 컴퓨트 acquire/release (라이브 경로의 기본)
    HostWrite,   // HOST_WRITE → SHADER_STORAGE_READ (픽스처 경로)
    None,        // 배리어 없음 — P3-④ 실험 전용
};

struct YoloPostprocConfig {
    int num_slots = 2;
    // 후보 상한. 프레임당 실측 후보는 15개 수준(P1: 15.1/frame)이지만 최악(전 앵커 통과)을 대비해 넉넉히 둔다.
    // 32 B * 4096 = 128 KB/슬롯.
    int max_dets = 4096;
    // 워크그룹 크기(spec constant 0/1). x 는 **워드 열**(=4앵커), y 는 행이다. x*y <= 256.
    // 실측 근거 (`yvo_postproc_probe --probe sweep`, 픽스처 5프레임 연속 스트림 120프레임,
    //   3회 인터리브, NPU 유휴. pass 1 median ms):
    //   20(x)x4(y)=20 inv 0.837 | 10x2=20 0.857 | 16x2=32 **2.281** | 8x4=32 **1.782**
    //   10x4=40 0.846 | 5x8=40 0.896 | **12x5=60 0.560** | **10x6=60 0.566 ← 채택**
    //   5x12=60 0.586 | 16x4=64 1.357 | 8x8=64 1.093 | 10x8=80 0.902 | **10x12=120 0.572**
    //   16x8=128 0.898 | 10x16=160 0.697 | 24x10=240 0.625 | 20x12=240 0.670 | 16x16=256 0.733
    //   32x8=256 0.755 | 40x6=240 0.820 | 64x4=256 0.987 | 80x3=240 1.269
    // ⚠ **두 가지가 계획서의 예상과 반대다. 측정치만이 근거이고 기제는 확정하지 못했다.**
    //   ① `sand_letterbox` 의 "invocation 이 클수록 좋다"(64 inv 3.65 → 256 inv 2.13 ms)가
    //      여기서는 통하지 않는다. 60 invocation 이 256 보다 1.3배 빠르다. 총 작업량이 작아
    //      (유효 invocation 2100개) 큰 워크그룹으로는 워크그룹 수가 8개까지 떨어져 코어에 덜
    //      퍼지는 것으로 **추정**한다(10x6 은 P3 에서 28 워크그룹이다). 미확정 가설이다.
    //   ② x 를 크게 해서 워프(16 lane) 안의 읽기를 더 길게 연속시키려는 시도는 **역효과**다
    //      (80x3 은 1.269 ms). w_words 가 20/10/5 뿐이라 x > w_words 인 lane 이 전부 조기 return 하고,
    //      그 낭비가 coalescing 이득보다 크다.
    //   ③ 32·64·128 invocation(워프 배수)이 20·40·60·120 보다 **느리다**. 설명하지 못했다.
    //      해상도·클래스 수가 바뀌면 다시 재야 한다(spec constant 라 재컴파일이 필요 없다).
    std::uint32_t local_size_x = 10;
    std::uint32_t local_size_y = 6;
    float score_thr = 0.2f;            // HEF 의 YOLOV8 post-process 설정과 같은 값
    YoloInputAlloc input_alloc = YoloInputAlloc::Plain;
    YoloInputSync input_sync = YoloInputSync::HostWrite;
    // ImportHeap 에서만 의미가 있다. true 면 CPU 접근을 `DMA_BUF_IOCTL_SYNC` 로 감싼다(부록 B.10 R2/R3).
    // **false 는 검증 전용이다**(V2/V3 의 음성 대조). 프로덕션에서 끄지 않는다 —
    // 디바이스 트리에 `dma-coherent` 가 0건이라 커널이 V3D 를 비일관 장치로 취급한다.
    bool heap_sync = true;
    // true 면 kChecksum spec constant 를 켠 **두 번째 파이프라인**도 만든다(검증 전용).
    // spec constant 라 평상시 경로의 명령 수가 늘지 않는다.
    bool checksum_pipeline = false;
    bool timestamps = true;
    bool verbose = false;
    // GPU NMS (efficient_nms.comp). false 면 EfficientNms 를 만들지 않는다(명령·메모리 0).
    bool enable_nms = false;
    EfficientNmsConfig nms;
};

// 셰이더 push constant (std430, 02_shaders/yolo_decode.comp 의 Params 와 같은 순서·크기)
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

// ⚠ 아래 두 구조체는 `01_src/display/overlay_renderer.hpp` 의 `OverlayDetBufHeader`/`DetGpu` 와
//   **바이트 단위로 같아야 한다.** 그래야 이 버퍼를 그대로 `vkCmdDispatchIndirect` 로 오버레이에 넘길 수 있다.
//   같은 정의를 두 번 쓰는 이유는 **링크 수준 격리**다(계획서 §5.2 안전장치 4):
//   `yvo_display` 가 `yvo_vulkan` 에 의존하므로 반대 방향 include 를 만들 수 없다.
//   두 헤더의 static_assert 가 어긋나면 컴파일이 깨지도록 오프셋까지 못 박아 둔다.
struct YoloDetBufHeader {
    std::uint32_t dispatch[3];   // VkDispatchIndirectCommand 호환 (offset 0)
    std::uint32_t count;         // emit **시도** 총수. max_dets 를 넘을 수 있다(overflow 참고)
    std::uint32_t overflow;      // 상한 초과 여부
    std::uint32_t text_bytes;    // 오버레이 HUD 길이. 디코드 경로는 0 으로 둔다
    std::uint32_t text[32];      // 오버레이 HUD ASCII. kChecksum 모드에서만 바이트 합 저장소로 재사용
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
static_assert(sizeof(YoloDetGpu) == 32, "DetGpu must be 32 B (오버레이 트랙과 공유하는 계약)");
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

    // 슬롯의 원시 출력 6개를 pass 1 에 먹인다. 헤더 초기화(호스트 쓰기) → 3 디스패치 → submit.
    // checksum = true 면 kChecksum 파이프라인으로 돌린다(checksum_pipeline 필요).
    void submit(int slot, const LetterboxParams& lb, int src_w, int src_h, bool checksum = false);
    bool ready(int slot) const;                                  // 펜스가 끝났는가 (대기하지 않는다)
    void wait(int slot, std::uint64_t timeout_ns = UINT64_MAX);   // 펜스 대기 + timestamp 읽기

    // wait() 뒤에 호출한다. `min(count, max_dets)` 개를 돌려준다(셰이더 count 는 시도 총수다).
    std::vector<DetectionKeyed> read_candidates(int slot) const;
    std::uint32_t raw_count(int slot) const;      // 셰이더가 쓴 count (상한 초과 시 max_dets 보다 크다)
    std::uint32_t overflow(int slot) const;
    // kChecksum 모드의 바이트 합. out[2s] = 스케일 s 클래스 텐서, out[2s+1] = box 텐서.
    void checksums(int slot, std::uint32_t out[kYoloNumTensors]) const;

    // ── ImportHeap 의 CPU 접근 구간 (부록 B.10 R2/R3) ──────────────────────────────────────
    // 슬롯의 원시 출력 6개 전부에 대해 `DMA_BUF_IOCTL_SYNC` 를 건다. 다른 alloc 모드거나
    // `heap_sync = false` 면 **아무것도 하지 않는다**(되돌림이 플래그 한 개 값이다 — B.14 S5).
    //   R2 CPU 읽기: 완료 신호 **뒤에** begin(Read) → 읽기 → end(Read)
    //   R3 CPU 쓰기: begin(Write) → memcpy/memset → end(Write). **초기화도 예외가 아니다**
    void raw_cpu_sync_begin(int slot, yvo::DmaSyncDir dir);
    void raw_cpu_sync_end(int slot, yvo::DmaSyncDir dir);
    bool raw_heap_sync_active() const {
        return cfg_.input_alloc == YoloInputAlloc::ImportHeap && cfg_.heap_sync;
    }
    // ImportHeap 에서 실제 적용된 드라이버 헤드룸(B.12 질의값). 다른 모드면 0.
    std::size_t heap_headroom() const { return heap_headroom_; }

    // ── Hailo 에 넘길 인터페이스 ──
    // ExportDmaBuf 모드에서만 유효한 dma-buf fd. **소유권은 이 객체에 있다**(close 하지 마라).
    int raw_dmabuf_fd(int slot, int tensor) const;
    std::size_t raw_dmabuf_size(int slot, int tensor) const;   // lseek 로 실측한 dma-buf 크기
    std::size_t raw_size(int tensor) const;                    // 텐서 크기 = hw_frame_size
    std::uint8_t* raw_host_ptr(int slot, int tensor) const;    // 영구 매핑(16384 정렬)

    // ── GPU NMS (enable_nms 일 때만. 아니면 read_nms 는 false, 사유는 0) ──
    // wait() 뒤에 호출한다. false 면 out 은 비고 그 프레임은 CPU 로 처리해야 한다.
    bool read_nms(int slot, std::vector<DetectionKeyed>& out) const;
    std::uint32_t nms_fallback_reason(int slot) const;   // kNmsFallback* 비트
    bool nms_enabled() const { return nms_ != nullptr; }
    const EfficientNms* nms() const { return nms_.get(); }

    double last_pass1_ms() const { return last_pass1_ms_; }   // 3 디스패치 구간
    double last_nms_ms() const { return last_nms_ms_; }       // NMS 디스패치 구간 (끄면 -1)
    double last_total_ms() const { return last_total_ms_; }   // 커맨드 버퍼 전체(배리어 포함)

    int num_slots() const { return static_cast<int>(slots_.size()); }
    const YoloPostprocConfig& config() const { return cfg_; }
    const YoloTensorLayout& layout() const { return layout_; }
    // 정수 사전필터 경계 `floor(thr/cls_scale)`. 부록 A.5: P3 57 / P4 53 / P5 52.
    std::uint32_t q_thr_low(int scale_id) const;

private:
    struct Tensor {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;   // Plain 모드만 소유. ExportDmaBuf 는 exporter 가 쥔다
        std::uint8_t* host = nullptr;
        int fd = -1;
        std::size_t dmabuf_size = 0;
        VkDeviceSize range = 0;
        bool from_heap = false;   // ImportHeap: buf/mem 은 import 캐시, ptr/fd 는 힙 풀이 소유한다
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
    // ImportHeap 전용. 파괴 순서가 중요하다 — import 캐시(VkBuffer/VkDeviceMemory)를 먼저 비우고
    // 그다음 힙 풀(mmap/fd)을 놓는다. destroy() 가 그 순서를 지킨다.
    std::unique_ptr<yvo::DmaHeapPool> heap_;
    std::unique_ptr<DmaBufImportCache> import_cache_;
    // 슬롯 DetBuf 를 입력 디스크립터로 쥔다 → destroy() 에서 슬롯보다 먼저 파괴한다.
    std::unique_ptr<EfficientNms> nms_;
    std::size_t heap_headroom_ = 0;
    std::vector<Slot> slots_;
    VkDeviceSize det_size_ = 0;
    double last_pass1_ms_ = -1.0;
    double last_total_ms_ = -1.0;
    double last_nms_ms_ = -1.0;
};

} // namespace yvo::vk
