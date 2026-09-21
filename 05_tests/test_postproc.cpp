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
#include "vulkan/sand_letterbox.hpp"
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
// (같은 이유로 `01_src/vulkan/tools/postproc_probe.cpp` 도 자체 리더를 쓴다 — 그 파일은 읽기만 했다.)
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
