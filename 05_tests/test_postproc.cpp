// GPU 후처리(`--postproc gpu`) 회귀 고정 — 계획서 [G] 단계.
//
// `--postproc gpu` 는 정확도 게이트를 전부 통과했지만 그 판정이 수동 1회성 실행에만 의존했다.
// 이 파일이 그 판정을 ctest 로 박아 둔다. 두 묶음이 있고 **의존 장치가 다르다**.
//
//  PostprocFixture.*  (NPU 불필요, 수 초)
//      `04_data/raw_tensors/`(5프레임 x 6텐서 + fixtures.json)를 `YoloPostproc`
//      (input_alloc = Plain, input_sync = HostWrite)에 호스트 memcpy 로 먹이고
//      `01_src/ref/cpu_yolov8_decode.hpp` 의 `cpu_yolov8_candidates()` 와 비교한다.
//      게이트(계획서 §4.5):
//        G5 후보 개수 : CPU 와 **동일**(only_cpu = only_gpu = 0, overflow = 0)   알려진 값 115 / 115
//        G5 점수      : **비트 일치**(불일치 0)                                   알려진 값 0
//        G5 좌표      : **< 0.25 px**                                             알려진 값 0.000122 px
//        G6 결정성    : 10회 반복에서 `(score, anchor_key)` 전순서 정규형이 바이트 동일
//      ⚠ 이 묶음은 `/dev/hailo0` 을 열지 않는다. NPU 가 다른 작업에 잡혀 있어도 돌아야 한다.
//
//  PostprocFixture.EfficientNmsBitExact / EfficientNms.*  (NPU 불필요)
//      GPU NMS(`02_shaders/efficient_nms.comp`)가 `cpu_nms_greedy_by_class()` 와 **비트 일치**하는가
//      (개수·순서·class·anchor_key·score·좌표). fallback 비트가 선 프레임은 비교에서 빼고 수를 보고한다.
//      EfficientNms.* 는 합성 DetBuf 를 직접 먹인다(동점, 클래스 정원, 클래스 간 겹침, 용량 초과, 0/1개,
//      IoU 보호대, 입력 순서 섞기 x 10회 결정성).
//
//  PostprocLive.*  (Hailo-8 독점, RUN_SERIAL)
//      `test_detections.cpp` 와 같은 "Hailo 장치 1개 → 2패스" 구조다.
//        패스 1: SandLetterbox(NHCW, dma-buf) → HostNmsStage (HEF 내장 NMS + HailoRT 호스트 후처리)
//        패스 2: **객체 전부 파괴 후** SandLetterbox(NHCW, dma-buf) → GpuPostprocStage
//                (HailoRawRunner 원시 6스트림 → yolo_decode.comp → cpu_nms_greedy_by_class)
//      게이트: `detection_match.hpp` 의 규칙으로 매칭 비율 >= 0.95 (목표 0.99).
//
// 환경변수: YVO_POSTPROC_FRAMES(기본 100), YVO_POSTPROC_REPEATS(기본 10, G6),
//           YVO_FIXTURE_DIR, YVO_TEST_VIDEO.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "ref/detection_match.hpp"

#include "ref/cpu_yolov8_decode.hpp"
#include "core/frame_types.hpp"
#include "hailo/hailo_runner.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "pipeline/gpu_postproc_stage.hpp"
#include "pipeline/infer_stage.hpp"
#include "vulkan/efficient_nms.hpp"
#include "vulkan/sand_letterbox.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"
#include "testkit/postproc_fixture.hpp"
#include "vulkan/yolo_postproc.hpp"

namespace {

using yvo::DetectionKeyed;
using yvo::detmatch::kMinMatchRatio;
using yvo::detmatch::match;
using yvo::detmatch::MatchResult;
using yvo::vk::kYoloNumScales;
using yvo::vk::kYoloNumTensors;
using yvo::vk::YoloInputAlloc;
using yvo::vk::YoloInputSync;
using yvo::vk::YoloPostproc;
using yvo::vk::YoloPostprocConfig;
using yvo::vk::YoloTensorLayout;
using yvo::vk::yolo_box_tensor;
using yvo::vk::yolo_cls_tensor;
using namespace yvo::fixture;

// HEF 의 YOLOV8 post-process 설정과 같은 값이어야 한다. 다르면 host/gpu 경로가 원리적으로 갈라진다.
constexpr float kScoreThr = 0.2f;

long env_long(const char* name, long fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::max(1L, std::atol(v)) : fallback;
}

std::string env_str(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : std::string(fallback);
}


// ───────────────────────────────────────────────── 최소 JSON 리더 (fixtures.json 전용)
//
// nlohmann_json 은 이 시스템에서 conda prefix 에만 있어(다른 시스템 헤더를 가릴 위험) 테스트 하나를
// 위해 그 include 경로를 끌어오지 않는다. 스키마가 고정돼 있으므로 필요한 만큼만 직접 읽는다.
// (같은 이유로 `01_src/tools/vulkan/postproc_probe.cpp` 도 자체 리더를 쓴다 — 그 파일은 읽기만 했다.)
// 픽스처 읽기·비교는 testkit/postproc_fixture.hpp 하나뿐이다(프로브와 공유).

// ═══════════════════════════════════════════════ PostprocFixture (NPU 불필요)

class PostprocFixture : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            const std::string dir = env_str("YVO_FIXTURE_DIR", YVO_FIXTURE_DIR);
            const int repeats = static_cast<int>(env_long("YVO_POSTPROC_REPEATS", 10));
            const FixtureSet fs = load_fixtures(dir);
            const YoloTensorLayout layout = layout_from_fixtures(fs);
            std::vector<FrameTensors> frames;
            for (int i = 0; i < fs.frames; ++i) frames.push_back(load_frame(fs, i));

            yvo::vk::VulkanContext ctx;
            YoloPostprocConfig pc;
            pc.num_slots = fs.frames;          // 프레임 하나당 슬롯 하나 (프로브의 fixture 모드와 같다)
            pc.score_thr = kScoreThr;
            pc.input_alloc = YoloInputAlloc::Plain;    // dma-buf 없음 → NPU 도, dma-heap 도 건드리지 않는다
            pc.input_sync = YoloInputSync::HostWrite;  // 호스트 memcpy → 셰이더 읽기
            pc.checksum_pipeline = false;
            YoloPostproc pp(ctx, layout, pc);

            const int n = pp.num_slots();
            for (int s = 0; s < n; ++s)
                for (int t = 0; t < kYoloNumTensors; ++t)
                    std::memcpy(pp.raw_host_ptr(s, t), frames[static_cast<std::size_t>(s)][static_cast<std::size_t>(t)].data(),
                                frames[static_cast<std::size_t>(s)][static_cast<std::size_t>(t)].size());

            cmp_.assign(static_cast<std::size_t>(n), {});
            overflow_.assign(static_cast<std::size_t>(n), 0);
            std::vector<std::string> canon_ref(static_cast<std::size_t>(n));
            for (int s = 0; s < n; ++s) {
                pp.submit(s, fs.lb, fs.src_w, fs.src_h);
                pp.wait(s);
                const auto gpu = pp.read_candidates(s);
                const std::uint8_t* ptr[kYoloNumTensors];
                for (int t = 0; t < kYoloNumTensors; ++t) ptr[t] = pp.raw_host_ptr(s, t);
                const auto cpu = yvo::cpu_yolov8_candidates(make_raw_set(layout, ptr), kScoreThr, fs.lb, fs.src_w,
                                                            fs.src_h);
                cmp_[static_cast<std::size_t>(s)] = compare_candidates(cpu, gpu);
                overflow_[static_cast<std::size_t>(s)] = pp.overflow(s);
                canon_ref[static_cast<std::size_t>(s)] = canonical(gpu);
            }

            // G6: 같은 입력으로 repeats 회 반복 — 정규형이 바이트 단위로 같아야 한다.
            repeats_ = repeats;
            for (int rep = 1; rep < repeats && determinism_fail_.empty(); ++rep)
                for (int s = 0; s < n; ++s) {
                    pp.submit(s, fs.lb, fs.src_w, fs.src_h);
                    pp.wait(s);
                    if (canonical(pp.read_candidates(s)) != canon_ref[static_cast<std::size_t>(s)]) {
                        determinism_fail_ = "rep " + std::to_string(rep) + " slot " + std::to_string(s) +
                                            " 의 후보 집합이 1회차와 다르다";
                        break;
                    }
                }
        } catch (const std::exception& e) {
            setup_error_ = std::string("exception: ") + e.what();
        }
    }

    void SetUp() override {
        if (!setup_error_.empty()) FAIL() << setup_error_;
        ASSERT_FALSE(cmp_.empty()) << "픽스처 프레임이 0개다";
    }

    static void report() {
        std::size_t cpu = 0, gpu = 0;
        float worst = 0.0f;
        for (const Compare& c : cmp_) {
            cpu += c.cpu;
            gpu += c.gpu;
            worst = std::max(worst, c.max_coord_err);
        }
        std::printf("[fixture] frames %zu, 후보 합계 cpu %zu / gpu %zu, 좌표 최대오차 %.6f px\n", cmp_.size(), cpu,
                    gpu, static_cast<double>(worst));
    }

    static inline std::string setup_error_;
    static inline std::string determinism_fail_;
    static inline int repeats_ = 0;
    static inline std::vector<Compare> cmp_;
    static inline std::vector<std::uint32_t> overflow_;
};

// G5 ①: 후보 개수가 CPU 기준과 **동일**해야 한다. 알려진 통과값 115 / 115.
TEST_F(PostprocFixture, CandidateCountMatchesCpu) {
    report();
    for (std::size_t s = 0; s < cmp_.size(); ++s) {
        const Compare& c = cmp_[s];
        EXPECT_EQ(c.cpu, c.gpu) << "frame " << s << ": 후보 개수가 다르다";
        EXPECT_EQ(c.only_cpu, 0u) << "frame " << s << ": CPU 에만 있는 후보 " << c.only_cpu << "개";
        EXPECT_EQ(c.only_gpu, 0u) << "frame " << s << ": GPU 에만 있는 후보 " << c.only_gpu << "개";
        EXPECT_EQ(overflow_[s], 0u) << "frame " << s << ": max_dets 초과 (셰이더 overflow 플래그)";
    }
}

// G5 ②: 매칭된 후보의 점수는 **비트 일치**여야 한다(fp32 재현성). 알려진 통과값 불일치 0.
TEST_F(PostprocFixture, ScoreBitExactVsCpu) {
    for (std::size_t s = 0; s < cmp_.size(); ++s)
        EXPECT_EQ(cmp_[s].score_bit_mismatch, 0u) << "frame " << s << ": 점수 비트 불일치";
}

// G5 ③: 좌표 오차 < 0.25 px. 알려진 통과값 0.000122 px.
TEST_F(PostprocFixture, CoordinatesWithinQuarterPixel) {
    for (std::size_t s = 0; s < cmp_.size(); ++s)
        EXPECT_LT(cmp_[s].max_coord_err, kCoordTolPx) << "frame " << s << ": 좌표 최대오차 초과";
}

// G6: 같은 픽스처를 10회 돌려 후보 집합의 정규형이 바이트 동일해야 한다.
TEST_F(PostprocFixture, DeterministicAcrossRepeats) {
    ASSERT_GE(repeats_, 2) << "반복 횟수가 1 이하면 결정성을 검사하지 못한다";
    EXPECT_TRUE(determinism_fail_.empty()) << determinism_fail_;
    std::printf("[fixture] G6 %d회 x %zu프레임\n", repeats_, cmp_.size());
}

// ═══════════════════════════════════════════════ GPU NMS (efficient_nms.comp) vs cpu_nms_greedy_by_class

constexpr float kIouThr = 0.7f;      // HEF 설정
constexpr int kMaxPerClass = 100;    // HEF 설정

// 비트 비교. 다르면 첫 차이를 설명하는 문자열, 같으면 빈 문자열.
std::string diff_bits(const std::vector<DetectionKeyed>& gpu, const std::vector<DetectionKeyed>& cpu) {
    if (gpu.size() != cpu.size())
        return "개수 gpu " + std::to_string(gpu.size()) + " / cpu " + std::to_string(cpu.size());
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const auto& a = gpu[i];
        const auto& b = cpu[i];
        const float fa[5] = {a.score, a.x1, a.y1, a.x2, a.y2};
        const float fb[5] = {b.score, b.x1, b.y1, b.x2, b.y2};
        if (a.class_id != b.class_id || a.anchor_key != b.anchor_key || std::memcmp(fa, fb, sizeof(fa)) != 0)
            return "위치 " + std::to_string(i) + ": gpu (cls " + std::to_string(a.class_id) + ", key " +
                   std::to_string(a.anchor_key) + ") / cpu (cls " + std::to_string(b.class_id) + ", key " +
                   std::to_string(b.anchor_key) + ") 또는 float 비트 차이";
    }
    return {};
}

// 픽스처 5프레임 x score {0.2, 0.01}: 디코드 + GPU NMS 를 한 커맨드 버퍼로 돌리고,
// 같은 GPU 후보를 CPU 로 억제한 결과와 비트 비교한다. 10회 반복에서 결과가 바이트 동일해야 한다.
TEST_F(PostprocFixture, EfficientNmsBitExact) {
    const std::string dir = env_str("YVO_FIXTURE_DIR", YVO_FIXTURE_DIR);
    const int repeats = static_cast<int>(env_long("YVO_POSTPROC_REPEATS", 10));
    const FixtureSet fs = load_fixtures(dir);
    const YoloTensorLayout layout = layout_from_fixtures(fs);
    std::vector<FrameTensors> frames;
    for (int i = 0; i < fs.frames; ++i) frames.push_back(load_frame(fs, i));

    yvo::vk::VulkanContext ctx;
    for (const float thr : {0.2f, 0.01f}) {
        YoloPostprocConfig pc;
        pc.num_slots = fs.frames;
        pc.score_thr = thr;
        pc.input_alloc = YoloInputAlloc::Plain;
        pc.input_sync = YoloInputSync::HostWrite;
        pc.enable_nms = true;
        pc.nms.iou_thr = kIouThr;
        pc.nms.max_per_class = kMaxPerClass;
        YoloPostproc pp(ctx, layout, pc);
        const int n = pp.num_slots();
        for (int s = 0; s < n; ++s)
            for (int t = 0; t < kYoloNumTensors; ++t)
                std::memcpy(pp.raw_host_ptr(s, t), frames[static_cast<std::size_t>(s)][static_cast<std::size_t>(t)].data(),
                            frames[static_cast<std::size_t>(s)][static_cast<std::size_t>(t)].size());

        std::vector<std::vector<DetectionKeyed>> first(static_cast<std::size_t>(n));
        std::size_t compared = 0, fallback = 0, cand_total = 0, kept_total = 0;
        double nms_ms_max = 0.0;
        for (int rep = 0; rep < repeats; ++rep) {
            for (int s = 0; s < n; ++s) {
                pp.submit(s, fs.lb, fs.src_w, fs.src_h);
                pp.wait(s);
                nms_ms_max = std::max(nms_ms_max, pp.last_nms_ms());
                std::vector<DetectionKeyed> gpu;
                const bool ok = pp.read_nms(s, gpu);
                if (rep == 0) {
                    auto cpu = pp.read_candidates(s);
                    cand_total += cpu.size();
                    yvo::cpu_nms_greedy_by_class(cpu, kIouThr, kMaxPerClass);
                    if (!ok) {
                        ++fallback;
                        std::printf("[nms] thr %.2f frame %d: fallback reason %u (후보 %u)\n", static_cast<double>(thr),
                                    s, pp.nms_fallback_reason(s), pp.raw_count(s));
                        EXPECT_NE(pp.nms_fallback_reason(s), 0u);
                    } else {
                        ++compared;
                        kept_total += gpu.size();
                        EXPECT_EQ(diff_bits(gpu, cpu), "") << "thr " << thr << " frame " << s;
                    }
                    first[static_cast<std::size_t>(s)] = gpu;
                } else {
                    EXPECT_EQ(diff_bits(gpu, first[static_cast<std::size_t>(s)]), "")
                        << "thr " << thr << " rep " << rep << " frame " << s << ": 반복 결과가 1회차와 다르다";
                }
            }
        }
        std::printf("[nms] thr %.2f: frames %d, 후보 %zu → 생존 %zu, 비교 %zu, fallback %zu, gpu nms max %.4f ms, %d회\n",
                    static_cast<double>(thr), n, cand_total, kept_total, compared, fallback, nms_ms_max, repeats);
        if (thr == 0.2f) {
            EXPECT_EQ(fallback, 0u) << "score 0.2 픽스처는 fallback 없이 GPU 로 끝나야 한다";
        }
        EXPECT_GT(compared, 0u) << "비교한 프레임이 없다";
    }
}

// 합성 DetBuf 를 EfficientNms 에 직접 먹이는 하네스. 입력 버퍼는 HOST_VISIBLE|COHERENT 하나.
class NmsHarness {
public:
    static constexpr std::uint32_t kInputMax = 2048;

    explicit NmsHarness(yvo::vk::VulkanContext& ctx, const yvo::vk::EfficientNmsConfig& cfg = {}) : ctx_(ctx) {
        VkDevice dev = ctx_.device();
        size_ = sizeof(yvo::vk::YoloDetBufHeader) + kInputMax * sizeof(yvo::vk::YoloDetGpu);
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size_;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf_));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, buf_, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = ctx_.find_memory_type(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem_));
        YVO_VK_CHECK(vkBindBufferMemory(dev, buf_, mem_, 0));
        void* p = nullptr;
        YVO_VK_CHECK(vkMapMemory(dev, mem_, 0, VK_WHOLE_SIZE, 0, &p));
        host_ = static_cast<std::uint8_t*>(p);
        nms_ = std::make_unique<yvo::vk::EfficientNms>(ctx_, cfg, std::vector<VkBuffer>{buf_}, size_, kInputMax);

        VkCommandPoolCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = ctx_.queue_family();
        YVO_VK_CHECK(vkCreateCommandPool(dev, &cp, nullptr, &pool_));
        VkCommandBufferAllocateInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = pool_;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        YVO_VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cmd_));
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &fence_));
    }
    ~NmsHarness() {
        VkDevice dev = ctx_.device();
        vkDeviceWaitIdle(dev);
        nms_.reset();
        vkDestroyFence(dev, fence_, nullptr);
        vkDestroyCommandPool(dev, pool_, nullptr);
        vkUnmapMemory(dev, mem_);
        vkDestroyBuffer(dev, buf_, nullptr);
        vkFreeMemory(dev, mem_, nullptr);
    }
    NmsHarness(const NmsHarness&) = delete;
    NmsHarness& operator=(const NmsHarness&) = delete;

    // count 는 헤더에 쓸 값(기본 = in.size()). dets 는 min(in.size(), kInputMax) 개만 쓴다.
    bool run(const std::vector<DetectionKeyed>& in, std::vector<DetectionKeyed>& out, std::uint32_t* reason,
             long count = -1, std::uint32_t overflow = 0) {
        auto* h = reinterpret_cast<yvo::vk::YoloDetBufHeader*>(host_);
        std::memset(h, 0, sizeof(*h));
        h->count = count >= 0 ? static_cast<std::uint32_t>(count) : static_cast<std::uint32_t>(in.size());
        h->overflow = overflow;
        auto* d = reinterpret_cast<yvo::vk::YoloDetGpu*>(host_ + sizeof(yvo::vk::YoloDetBufHeader));
        for (std::size_t i = 0; i < in.size() && i < kInputMax; ++i)
            d[i] = {in[i].x1, in[i].y1, in[i].x2, in[i].y2, in[i].score, static_cast<std::uint32_t>(in[i].class_id),
                    in[i].anchor_key, 1u};

        YVO_VK_CHECK(vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        YVO_VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
        barrier(buf_, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        nms_->record(cmd_, 0);
        barrier(nms_->output(0), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        YVO_VK_CHECK(vkEndCommandBuffer(cmd_));
        YVO_VK_CHECK(vkResetFences(ctx_.device(), 1, &fence_));
        VkCommandBufferSubmitInfo cbs{};
        cbs.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cbs.commandBuffer = cmd_;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbs;
        YVO_VK_CHECK(vkQueueSubmit2(ctx_.queue(), 1, &si, fence_));
        YVO_VK_CHECK(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX));
        if (reason) *reason = nms_->fallback_reason(0);
        return nms_->read_results(0, out);
    }

private:
    void barrier(VkBuffer b, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                 VkAccessFlags2 da) {
        VkBufferMemoryBarrier2 m{};
        m.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        m.srcStageMask = ss;
        m.srcAccessMask = sa;
        m.dstStageMask = ds;
        m.dstAccessMask = da;
        m.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        m.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        m.buffer = b;
        m.offset = 0;
        m.size = VK_WHOLE_SIZE;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &m;
        vkCmdPipelineBarrier2(cmd_, &di);
    }

    yvo::vk::VulkanContext& ctx_;
    VkDeviceSize size_ = 0;
    VkBuffer buf_ = VK_NULL_HANDLE;
    VkDeviceMemory mem_ = VK_NULL_HANDLE;
    std::uint8_t* host_ = nullptr;
    std::unique_ptr<yvo::vk::EfficientNms> nms_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

DetectionKeyed make_det(float x1, float y1, float x2, float y2, float score, int cls, std::uint32_t key) {
    DetectionKeyed d;
    d.x1 = x1;
    d.y1 = y1;
    d.x2 = x2;
    d.y2 = y2;
    d.score = score;
    d.class_id = cls;
    d.anchor_key = key;
    return d;
}

// GPU 결과를 CPU 기준과 비트 비교한다. fallback 이면 false 를 돌려주고 사유를 reason 에 둔다.
bool expect_matches_cpu(NmsHarness& h, const std::vector<DetectionKeyed>& in, std::uint32_t* reason = nullptr) {
    std::vector<DetectionKeyed> gpu;
    std::uint32_t why = 0;
    const bool ok = h.run(in, gpu, &why);
    if (reason) *reason = why;
    if (!ok) return false;
    auto cpu = in;
    yvo::cpu_nms_greedy_by_class(cpu, kIouThr, kMaxPerClass);
    EXPECT_EQ(diff_bits(gpu, cpu), "");
    return true;
}

class EfficientNms : public ::testing::Test {
protected:
    static void SetUpTestSuite() { ctx_ = std::make_unique<yvo::vk::VulkanContext>(); }
    static void TearDownTestSuite() { ctx_.reset(); }
    static inline std::unique_ptr<yvo::vk::VulkanContext> ctx_;
};

// 0개·1개 입력: 0개면 count 0 / dispatch_x 1, 1개면 그대로 나온다.
TEST_F(EfficientNms, SyntheticEmptyAndSingle) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> out;
    std::uint32_t why = 99;
    ASSERT_TRUE(h.run({}, out, &why));
    EXPECT_EQ(why, 0u);
    EXPECT_TRUE(out.empty());
    ASSERT_TRUE(expect_matches_cpu(h, {make_det(10, 20, 110, 220, 0.5f, 3, 77)}));
}

// 동점 점수: (score 같음) → anchor_key 오름차순이 먼저 살아남는다. 입력은 key 내림차순으로 넣는다.
TEST_F(EfficientNms, SyntheticTiedScores) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> in;
    for (int i = 0; i < 40; ++i) {
        const float o = static_cast<float>(i % 4);   // 4개씩 거의 같은 자리 → 서로 억제
        in.push_back(make_det(100 + 50 * static_cast<float>(i / 4) + o, 100, 140 + 50 * static_cast<float>(i / 4) + o,
                              160, 0.5f, 1, static_cast<std::uint32_t>(1000 - i)));
    }
    ASSERT_TRUE(expect_matches_cpu(h, in));
    std::vector<DetectionKeyed> out;
    ASSERT_TRUE(h.run(in, out, nullptr));
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.front().anchor_key, 1000u - 39u) << "전부 동점이면 anchor_key 가 가장 작은 후보가 맨 앞이다";
}

// 클래스 정원: 서로 안 겹치는 같은 클래스 150개 → 100개만. 정원 초과 후보는 억제 연산을 하지 않는다
// (정원 밖 후보와 크게 겹치는 다른 후보가 살아남아야 한다 — CPU 와 같은 의미).
TEST_F(EfficientNms, SyntheticClassCapOver100) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> in;
    for (int i = 0; i < 150; ++i) {
        const float x = static_cast<float>(i % 30) * 60.0f, y = static_cast<float>(i / 30) * 60.0f;
        in.push_back(make_det(x, y, x + 40, y + 40, 0.9f - 0.001f * static_cast<float>(i), 7,
                              static_cast<std::uint32_t>(i)));
    }
    // 정원 밖(순위 120)의 후보와 거의 같은 박스를 더 낮은 점수로 넣는다 → CPU 는 억제하지 않는다
    in.push_back(make_det(in[120].x1 + 1, in[120].y1, in[120].x2 + 1, in[120].y2, 0.1f, 7, 5000));
    in.push_back(make_det(0, 0, 40, 40, 0.95f, 8, 6000));   // 다른 클래스는 따로 센다
    std::vector<DetectionKeyed> out;
    ASSERT_TRUE(expect_matches_cpu(h, in));
    ASSERT_TRUE(h.run(in, out, nullptr));
    EXPECT_EQ(std::count_if(out.begin(), out.end(), [](const DetectionKeyed& d) { return d.class_id == 7; }), 100);
}

// 서로 다른 클래스의 같은 박스는 억제하지 않는다.
TEST_F(EfficientNms, SyntheticCrossClassOverlap) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> in;
    for (int c = 0; c < 80; ++c)
        in.push_back(make_det(10, 10, 200, 200, 0.3f + 0.001f * static_cast<float>(c), c, static_cast<std::uint32_t>(c)));
    in.push_back(make_det(11, 10, 201, 200, 0.2f, 5, 999));   // 같은 클래스 5 는 억제된다
    std::vector<DetectionKeyed> out;
    ASSERT_TRUE(expect_matches_cpu(h, in));
    ASSERT_TRUE(h.run(in, out, nullptr));
    EXPECT_EQ(out.size(), 80u);
}

// N > kMaxSelected(1024) 이거나 디코드 DetBuf 가 넘쳤으면 자르지 않고 fallback 비트 1.
TEST_F(EfficientNms, SyntheticOverCapacityFallsBack) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> in;
    for (int i = 0; i < 1025; ++i)
        in.push_back(make_det(0, 0, 10, 10, 0.5f, i % 80, static_cast<std::uint32_t>(i)));
    std::vector<DetectionKeyed> out;
    std::uint32_t why = 0;
    EXPECT_FALSE(h.run(in, out, &why));
    EXPECT_EQ(why & yvo::vk::kNmsFallbackCapacity, yvo::vk::kNmsFallbackCapacity);
    EXPECT_TRUE(out.empty());
    in.resize(1024);   // 정확히 kMaxSelected 는 GPU 로 끝난다
    EXPECT_TRUE(expect_matches_cpu(h, in));
    in.resize(10);     // 디코드 overflow 플래그 / count > 입력 용량
    EXPECT_FALSE(h.run(in, out, &why, -1, 1u));
    EXPECT_EQ(why, yvo::vk::kNmsFallbackCapacity);
    EXPECT_FALSE(h.run(in, out, &why, NmsHarness::kInputMax + 1));
    EXPECT_EQ(why, yvo::vk::kNmsFallbackCapacity);
}

// IoU 가 임계값 0.7 에 딱 붙은 쌍(inter 70 / ua 100)은 보호대 → fallback 비트 2.
TEST_F(EfficientNms, SyntheticIouGuardBand) {
    NmsHarness h(*ctx_);
    const std::vector<DetectionKeyed> in = {make_det(0, 0, 10, 10, 0.9f, 0, 1), make_det(0, 0, 10, 7, 0.8f, 0, 2)};
    std::vector<DetectionKeyed> out;
    std::uint32_t why = 0;
    EXPECT_FALSE(h.run(in, out, &why));
    EXPECT_EQ(why, yvo::vk::kNmsFallbackIouGuard);
    // 보호대 밖(0.69 / 0.71)은 GPU 로 끝나고 CPU 와 같다
    EXPECT_TRUE(expect_matches_cpu(h, {make_det(0, 0, 10, 10, 0.9f, 0, 1), make_det(0, 0, 10, 6.9f, 0.8f, 0, 2)}));
    EXPECT_TRUE(expect_matches_cpu(h, {make_det(0, 0, 10, 10, 0.9f, 0, 1), make_det(0, 0, 10, 7.1f, 0.8f, 0, 2)}));
}

// 무작위 군집 박스: CPU 와 비트 일치, 입력 순서를 섞어 10회 돌려도 결과가 바이트 동일(결정성).
TEST_F(EfficientNms, SyntheticRandomDeterministic) {
    NmsHarness h(*ctx_);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    int compared = 0, fallback = 0;
    for (int trial = 0; trial < 20; ++trial) {
        const int n = 50 + trial * 45;   // 50 .. 905
        std::vector<DetectionKeyed> in;
        for (int i = 0; i < n; ++i) {
            const float cx = 100.0f + 1700.0f * U(rng) * 0.25f + static_cast<float>(i % 8) * 200.0f;
            const float cy = 100.0f + 800.0f * U(rng);
            const float w = 20.0f + 200.0f * U(rng), hh = 20.0f + 200.0f * U(rng);
            // 점수는 픽스처처럼 양자화된 값(동점이 생긴다)
            const float score = 0.2f + static_cast<float>(static_cast<int>(U(rng) * 60.0f)) * 0.0125f;
            in.push_back(make_det(std::max(0.0f, cx - w / 2), std::max(0.0f, cy - hh / 2), std::min(1920.0f, cx + w / 2),
                                  std::min(1080.0f, cy + hh / 2), score, static_cast<int>(U(rng) * 4.0f),
                                  static_cast<std::uint32_t>(trial * 10000 + i)));
        }
        std::uint32_t why = 0;
        if (!expect_matches_cpu(h, in, &why)) {
            ++fallback;
            EXPECT_EQ(why, yvo::vk::kNmsFallbackIouGuard) << "trial " << trial;
            continue;
        }
        ++compared;
        std::vector<DetectionKeyed> ref;
        ASSERT_TRUE(h.run(in, ref, nullptr));
        for (int rep = 0; rep < 10; ++rep) {
            std::shuffle(in.begin(), in.end(), rng);
            std::vector<DetectionKeyed> out;
            ASSERT_TRUE(h.run(in, out, nullptr));
            EXPECT_EQ(diff_bits(out, ref), "") << "trial " << trial << " rep " << rep << ": 입력 순서에 따라 달라졌다";
        }
    }
    std::printf("[nms] 무작위 20회: 비교 %d, 보호대 fallback %d\n", compared, fallback);
    EXPECT_GT(compared, 15);
}

// ═══════════════════════════════════════════════ PostprocLive (Hailo-8 독점)

enum LivePath { kHost = 0, kGpu = 1, kNumLivePaths = 2 };
const char* kLivePathName[kNumLivePaths] = {"host", "gpu"};

class PostprocLive : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::string video = env_str("YVO_TEST_VIDEO", YVO_TEST_VIDEO);
        const long want = env_long("YVO_POSTPROC_FRAMES", 100);
        // 장치 **부재**는 skip, 장치가 있는데 실패하면 FAIL 이다. 드라이버가 없는 보드에서
        // 회귀 실패처럼 보이지 않게 하되, 여기서 나는 다른 에러는 전부 진짜 실패로 다룬다.
        if (!std::filesystem::exists("/dev/hailo0")) {
            skip_reason_ = "/dev/hailo0 이 없다 — Hailo-8 이 없는 환경";
            return;
        }
        if (!std::filesystem::exists(video)) {
            setup_error_ = "video not found: " + video;
            return;
        }
        results_.assign(kNumLivePaths, {});
        for (auto& r : results_) r.resize(static_cast<std::size_t>(want));
        // 패스 1(host) 이 완전히 끝나고 객체가 파괴된 뒤에 패스 2(gpu) 를 연다 — Hailo 장치가 하나다.
        run_pass(kHost, video, want);
        if (setup_error_.empty() && frames_ > 0) run_pass(kGpu, video, frames_);
    }

    // 한 경로의 전체 패스. SandLetterbox(NHCW, dma-buf) → InferStage → 프레임마다 drain.
    static void run_pass(LivePath path, const std::string& video, long nframes) {
        std::string& err = path == kHost ? setup_error_ : gpu_error_;
        int export_fd = -1;
        try {
            yvo::vk::VulkanContext ctx;
            if (!ctx.dmabuf_export_supported()) {
                err = "Vulkan dma-buf export 미지원";
                return;
            }
            yvo::vk::SandLetterboxConfig pcfg;
            pcfg.num_slots = 1;
            pcfg.export_dmabuf = true;
            pcfg.out_layout = yvo::TensorLayout::Nhcw;   // 두 경로 모두 NHCW dma-buf 입력이다
            yvo::vk::SandLetterbox pre(ctx, pcfg);

            std::unique_ptr<yvo::pipeline::InferStage> stage;
            if (path == kGpu) {
                yvo::pipeline::GpuPostprocStageConfig gcfg;
                gcfg.hef_path = YVO_TEST_HEF;
                gcfg.batch_size = 1;
                gcfg.num_slots = 1;
                gcfg.score_thr = kScoreThr;
                gcfg.input_alloc = yvo::vk::YoloInputAlloc::ExportDmaBuf;
                stage = std::make_unique<yvo::pipeline::GpuPostprocStage>(ctx, gcfg);
            } else {
                yvo::HailoRunnerConfig hcfg;
                hcfg.hef_path = YVO_TEST_HEF;
                hcfg.input_layout = yvo::TensorLayout::Nhcw;
                hcfg.batch_size = 1;
                hcfg.num_slots = 1;
                stage = std::make_unique<yvo::pipeline::HostNmsStage>(hcfg);
            }
            if (stage->input_frame_size() != pre.output_size()) {
                err = "입력 크기 불일치: NPU " + std::to_string(stage->input_frame_size()) + ", GPU " +
                      std::to_string(pre.output_size());
                return;
            }
            export_fd = pre.export_dmabuf_fd(0);
            stage->bind_input_dmabuf(0, export_fd);

            yvo::FfmpegDrmDecoder dec;
            if (!dec.open(video)) {
                err = "decoder open failed: " + dec.error();
                return;
            }
            stage->set_source_size(dec.width(), dec.height());

            std::mutex m;
            long done = 0;
            for (long n = 0; n < nframes; ++n) {
                auto f = dec.next();
                if (!f) {
                    if (path == kGpu) err = "프레임 부족: " + std::to_string(n) + " / " + std::to_string(nframes);
                    break;
                }
                f->pts = n;
                yvo::PreprocOutput out = pre.process(*f, 0);
                f.reset();
                out.pts = n;
                out.slot = 0;
                out.dmabuf_fd = export_fd;
                stage->submit(out, [&, n](yvo::InferResult&& r) {
                    std::lock_guard<std::mutex> lk(m);
                    if (r.status != HAILO_SUCCESS) ++infer_errors_;
                    results_[path][static_cast<std::size_t>(n)] = std::move(r.detections);
                });
                // 슬롯이 하나뿐이다: 다음 프레임이 같은 dma-buf 를 덮어쓰기 전에 완료를 받아야 한다.
                stage->drain();
                ++done;
            }
            if (path == kHost) frames_ = done;
            pre.clear_import_cache();
            dec.close();
            stage.reset();   // 러너가 dma-buf 매핑을 쥐고 있다 — fd 를 닫기 전에 파괴한다
        } catch (const std::exception& e) {
            err = std::string("exception: ") + e.what();
        }
        if (export_fd >= 0) ::close(export_fd);
    }

    void SetUp() override {
        if (!skip_reason_.empty()) GTEST_SKIP() << skip_reason_;
        if (!setup_error_.empty()) FAIL() << setup_error_;
        ASSERT_TRUE(gpu_error_.empty()) << gpu_error_;
        ASSERT_GT(frames_, 0);
    }

    static inline std::string skip_reason_;
    static inline std::string setup_error_;
    static inline std::string gpu_error_;
    static inline long frames_ = 0;
    static inline long infer_errors_ = 0;
    static inline std::vector<std::vector<std::vector<yvo::Detection>>> results_;   // [path][frame]
};

// GPU 후처리 경로의 검출이 host 경로(HEF 내장 NMS + HailoRT 호스트 후처리)와 일치하는가.
// 게이트 0.95, 알려진 통과값 0.9990 (검출 총수 9457 로 host 와 동일).
TEST_F(PostprocLive, GpuPostprocMatchesHost) {
    MatchResult total;
    int printed = 0;
    for (long n = 0; n < frames_; ++n) {
        std::vector<std::string> detail;
        const auto m = match(results_[kHost][static_cast<std::size_t>(n)],
                             results_[kGpu][static_cast<std::size_t>(n)], &detail);
        total.ref += m.ref;
        total.test += m.test;
        total.matched += m.matched;
        if (!detail.empty() && printed < 20) {
            std::printf("[gpu] frame %ld: host %ld, gpu %ld, matched %ld\n", n, m.ref, m.test, m.matched);
            for (const auto& s : detail) std::printf("    %s\n", s.c_str());
            ++printed;
        }
    }
    const long denom = std::max(total.ref, total.test);
    const double ratio = denom > 0 ? static_cast<double>(total.matched) / static_cast<double>(denom) : 1.0;
    std::printf("[%s] frames %ld, detections host %ld / gpu %ld, matched %ld, ratio %.4f\n",
                kLivePathName[kGpu], frames_, total.ref, total.test, total.matched, ratio);
    EXPECT_GT(total.ref, 0) << "host 경로 검출이 0개";
    EXPECT_GE(ratio, kMinMatchRatio);
    EXPECT_EQ(infer_errors_, 0);
}

} // namespace
