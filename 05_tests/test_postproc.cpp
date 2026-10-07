// GPU postprocessing (`--postproc gpu`) regression lock — plan stage [G].
//
// `--postproc gpu` passed every accuracy gate, but that verdict depended only on a manual one-off run.
// This file pins that verdict into ctest. There are two groups, and **they depend on different devices**.
//
//  PostprocFixture.*  (no NPU needed, a few seconds)
//      Feeds `04_data/raw_tensors/` (5 frames x 6 tensors + fixtures.json) to `YoloPostproc`
//      (input_alloc = Plain, input_sync = HostWrite) via host memcpy and
//      compares against `cpu_yolov8_candidates()` in `01_src/ref/cpu_yolov8_decode.hpp`.
//      Gates (plan §4.5):
//        G5 candidate count : **identical** to CPU (only_cpu = only_gpu = 0, overflow = 0)   known value 115 / 115
//        G5 score      : **bit-exact** (0 mismatches)                                   known value 0
//        G5 coordinates      : **< 0.25 px**                                             known value 0.000122 px
//        G6 determinism    : over 10 repeats the `(score, anchor_key)` total-order canonical form is byte-identical
//      ⚠ This group does not open `/dev/hailo0`. It must run even while the NPU is held by another job.
//
//  PostprocFixture.EfficientNmsBitExact / EfficientNms.*  (no NPU needed)
//      Is GPU NMS (`02_shaders/efficient_nms.comp`) **bit-exact** with `cpu_nms_greedy_by_class()`
//      (count, order, class, anchor_key, score, coordinates). Frames with a fallback bit set are excluded from comparison and their count is reported.
//      EfficientNms.* feeds synthetic DetBufs directly (ties, per-class quota, cross-class overlap, over capacity, 0/1 inputs,
//      IoU guard band, input order shuffle x 10 determinism).
//
//  PostprocLive.*  (takes Hailo-8 exclusively, RUN_SERIAL)
//      Same "one Hailo device → 2 passes" structure as `test_detections.cpp`.
//        Pass 1: SandLetterbox(NHCW, dma-buf) → HostNmsStage (HEF built-in NMS + HailoRT host postprocessing)
//        Pass 2: **after destroying all objects** SandLetterbox(NHCW, dma-buf) → GpuPostprocStage
//                (HailoRawRunner raw 6 streams → yolo_decode.comp → cpu_nms_greedy_by_class)
//      Gate: match ratio >= 0.95 by the rules in `detection_match.hpp` (target 0.99).
//
// Env vars: YVO_POSTPROC_FRAMES (default 100), YVO_POSTPROC_REPEATS (default 10, G6),
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
#include "testkit/nms_harness.hpp"
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

// Must equal the HEF's YOLOV8 post-process settings. Otherwise the host/gpu paths diverge in principle.
constexpr float kScoreThr = 0.2f;

long env_long(const char* name, long fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::max(1L, std::atol(v)) : fallback;
}

std::string env_str(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : std::string(fallback);
}


// ───────────────────────────────────────────────── Minimal JSON reader (fixtures.json only)
//
// nlohmann_json exists only in the conda prefix on this system (risk of shadowing other system headers), so we do not pull in
// that include path for a single test. The schema is fixed, so we read only what we need directly.
// (For the same reason `01_src/tools/vulkan/postproc_probe.cpp` also uses its own reader — that file was only read.)
// Fixture reading/comparison lives only in testkit/postproc_fixture.hpp (shared with the probe).

// ═══════════════════════════════════════════════ PostprocFixture (no NPU needed)

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
            pc.num_slots = fs.frames;          // one slot per frame (same as the probe's fixture mode)
            pc.score_thr = kScoreThr;
            pc.input_alloc = YoloInputAlloc::Plain;    // no dma-buf → touches neither the NPU nor dma-heap
            pc.input_sync = YoloInputSync::HostWrite;  // host memcpy → shader read
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

            // G6: repeat `repeats` times with the same input — the canonical form must be byte-identical.
            repeats_ = repeats;
            for (int rep = 1; rep < repeats && determinism_fail_.empty(); ++rep)
                for (int s = 0; s < n; ++s) {
                    pp.submit(s, fs.lb, fs.src_w, fs.src_h);
                    pp.wait(s);
                    if (canonical(pp.read_candidates(s)) != canon_ref[static_cast<std::size_t>(s)]) {
                        determinism_fail_ = "rep " + std::to_string(rep) + " slot " + std::to_string(s) +
                                            " candidate set differs from the 1st run";
                        break;
                    }
                }
        } catch (const std::exception& e) {
            setup_error_ = std::string("exception: ") + e.what();
        }
    }

    void SetUp() override {
        if (!setup_error_.empty()) FAIL() << setup_error_;
        ASSERT_FALSE(cmp_.empty()) << "fixture has 0 frames";
    }

    static void report() {
        std::size_t cpu = 0, gpu = 0;
        float worst = 0.0f;
        for (const Compare& c : cmp_) {
            cpu += c.cpu;
            gpu += c.gpu;
            worst = std::max(worst, c.max_coord_err);
        }
        std::printf("[fixture] frames %zu, candidates total cpu %zu / gpu %zu, max coord error %.6f px\n", cmp_.size(), cpu,
                    gpu, static_cast<double>(worst));
    }

    static inline std::string setup_error_;
    static inline std::string determinism_fail_;
    static inline int repeats_ = 0;
    static inline std::vector<Compare> cmp_;
    static inline std::vector<std::uint32_t> overflow_;
};

// G5 ①: the candidate count must be **identical** to the CPU reference. Known passing value 115 / 115.
TEST_F(PostprocFixture, CandidateCountMatchesCpu) {
    report();
    for (std::size_t s = 0; s < cmp_.size(); ++s) {
        const Compare& c = cmp_[s];
        EXPECT_EQ(c.cpu, c.gpu) << "frame " << s << ": candidate count differs";
        EXPECT_EQ(c.only_cpu, 0u) << "frame " << s << ": CPU-only candidates: " << c.only_cpu << " found";
        EXPECT_EQ(c.only_gpu, 0u) << "frame " << s << ": GPU-only candidates: " << c.only_gpu << " found";
        EXPECT_EQ(overflow_[s], 0u) << "frame " << s << ": max_dets exceeded (shader overflow flag)";
    }
}

// G5 ②: scores of matched candidates must be **bit-exact** (fp32 reproducibility). Known passing value: 0 mismatches.
TEST_F(PostprocFixture, ScoreBitExactVsCpu) {
    for (std::size_t s = 0; s < cmp_.size(); ++s)
        EXPECT_EQ(cmp_[s].score_bit_mismatch, 0u) << "frame " << s << ": score bit mismatch";
}

// G5 ③: coordinate error < 0.25 px. Known passing value 0.000122 px.
TEST_F(PostprocFixture, CoordinatesWithinQuarterPixel) {
    for (std::size_t s = 0; s < cmp_.size(); ++s)
        EXPECT_LT(cmp_[s].max_coord_err, kCoordTolPx) << "frame " << s << ": max coord error exceeded";
}

// G6: run the same fixture 10 times; the canonical form of the candidate set must be byte-identical.
TEST_F(PostprocFixture, DeterministicAcrossRepeats) {
    ASSERT_GE(repeats_, 2) << "determinism cannot be checked with 1 or fewer repeats";
    EXPECT_TRUE(determinism_fail_.empty()) << determinism_fail_;
    std::printf("[fixture] G6 %d runs x %zu frames\n", repeats_, cmp_.size());
}

// ═══════════════════════════════════════════════ GPU NMS (efficient_nms.comp) vs cpu_nms_greedy_by_class

constexpr float kIouThr = 0.7f;      // HEF setting
constexpr int kMaxPerClass = 100;    // HEF setting

// Bit comparison. Returns a string describing the first difference if they differ, an empty string if equal.
std::string diff_bits(const std::vector<DetectionKeyed>& gpu, const std::vector<DetectionKeyed>& cpu) {
    if (gpu.size() != cpu.size())
        return "count gpu " + std::to_string(gpu.size()) + " / cpu " + std::to_string(cpu.size());
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const auto& a = gpu[i];
        const auto& b = cpu[i];
        const float fa[5] = {a.score, a.x1, a.y1, a.x2, a.y2};
        const float fb[5] = {b.score, b.x1, b.y1, b.x2, b.y2};
        if (a.class_id != b.class_id || a.anchor_key != b.anchor_key || std::memcmp(fa, fb, sizeof(fa)) != 0)
            return "position " + std::to_string(i) + ": gpu (cls " + std::to_string(a.class_id) + ", key " +
                   std::to_string(a.anchor_key) + ") / cpu (cls " + std::to_string(b.class_id) + ", key " +
                   std::to_string(b.anchor_key) + ") or float bit difference";
    }
    return {};
}

// Fixture 5 frames x score {0.2, 0.01}: run decode + GPU NMS in one command buffer, and
// bit-compare with the result of suppressing the same GPU candidates on the CPU. Results must be byte-identical over 10 repeats.
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
                        std::printf("[nms] thr %.2f frame %d: fallback reason %u (candidates %u)\n", static_cast<double>(thr),
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
                        << "thr " << thr << " rep " << rep << " frame " << s << ": repeated result differs from the 1st run";
                }
            }
        }
        std::printf("[nms] thr %.2f: frames %d, candidates %zu → kept %zu, compared %zu, fallback %zu, gpu nms max %.4f ms, %d runs\n",
                    static_cast<double>(thr), n, cand_total, kept_total, compared, fallback, nms_ms_max, repeats);
        if (thr == 0.2f) {
            EXPECT_EQ(fallback, 0u) << "the score 0.2 fixture must finish on the GPU without fallback";
        }
        EXPECT_GT(compared, 0u) << "no frames were compared";
    }
}

// NmsHarness (synthetic DetBuf → EfficientNms) lives in testkit/nms_harness.hpp — shared with yvo_postproc_probe --probe nms.

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

// Bit-compare the GPU result with the CPU reference. On fallback, returns false and puts the cause in reason.
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

// 0 and 1 inputs: 0 gives count 0 / dispatch_x 1, 1 comes out as is.
TEST_F(EfficientNms, SyntheticEmptyAndSingle) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> out;
    std::uint32_t why = 99;
    ASSERT_TRUE(h.run({}, out, &why));
    EXPECT_EQ(why, 0u);
    EXPECT_TRUE(out.empty());
    ASSERT_TRUE(expect_matches_cpu(h, {make_det(10, 20, 110, 220, 0.5f, 3, 77)}));
}

// Tied scores: (same score) → ascending anchor_key survives first. Inputs are inserted in descending key order.
TEST_F(EfficientNms, SyntheticTiedScores) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> in;
    for (int i = 0; i < 40; ++i) {
        const float o = static_cast<float>(i % 4);   // groups of 4 at nearly the same position → suppress each other
        in.push_back(make_det(100 + 50 * static_cast<float>(i / 4) + o, 100, 140 + 50 * static_cast<float>(i / 4) + o,
                              160, 0.5f, 1, static_cast<std::uint32_t>(1000 - i)));
    }
    ASSERT_TRUE(expect_matches_cpu(h, in));
    std::vector<DetectionKeyed> out;
    ASSERT_TRUE(h.run(in, out, nullptr));
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.front().anchor_key, 1000u - 39u) << "when all are tied, the candidate with the smallest anchor_key comes first";
}

// Per-class quota: 150 non-overlapping boxes of the same class → only 100. Candidates beyond the quota do not perform suppression
// (another candidate heavily overlapping an out-of-quota candidate must survive — same semantics as the CPU).
TEST_F(EfficientNms, SyntheticClassCapOver100) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> in;
    for (int i = 0; i < 150; ++i) {
        const float x = static_cast<float>(i % 30) * 60.0f, y = static_cast<float>(i / 30) * 60.0f;
        in.push_back(make_det(x, y, x + 40, y + 40, 0.9f - 0.001f * static_cast<float>(i), 7,
                              static_cast<std::uint32_t>(i)));
    }
    // Insert a box nearly identical to the out-of-quota (rank 120) candidate with a lower score → the CPU does not suppress it
    in.push_back(make_det(in[120].x1 + 1, in[120].y1, in[120].x2 + 1, in[120].y2, 0.1f, 7, 5000));
    in.push_back(make_det(0, 0, 40, 40, 0.95f, 8, 6000));   // other classes are counted separately
    std::vector<DetectionKeyed> out;
    ASSERT_TRUE(expect_matches_cpu(h, in));
    ASSERT_TRUE(h.run(in, out, nullptr));
    EXPECT_EQ(std::count_if(out.begin(), out.end(), [](const DetectionKeyed& d) { return d.class_id == 7; }), 100);
}

// Identical boxes of different classes do not suppress each other.
TEST_F(EfficientNms, SyntheticCrossClassOverlap) {
    NmsHarness h(*ctx_);
    std::vector<DetectionKeyed> in;
    for (int c = 0; c < 80; ++c)
        in.push_back(make_det(10, 10, 200, 200, 0.3f + 0.001f * static_cast<float>(c), c, static_cast<std::uint32_t>(c)));
    in.push_back(make_det(11, 10, 201, 200, 0.2f, 5, 999));   // same class 5 gets suppressed
    std::vector<DetectionKeyed> out;
    ASSERT_TRUE(expect_matches_cpu(h, in));
    ASSERT_TRUE(h.run(in, out, nullptr));
    EXPECT_EQ(out.size(), 80u);
}

// If N > kMaxSelected (1024) or the decode DetBuf overflowed, do not truncate; fallback bit 1.
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
    in.resize(1024);   // exactly kMaxSelected finishes on the GPU
    EXPECT_TRUE(expect_matches_cpu(h, in));
    in.resize(10);     // decode overflow flag / count > input capacity
    EXPECT_FALSE(h.run(in, out, &why, -1, 1u));
    EXPECT_EQ(why, yvo::vk::kNmsFallbackCapacity);
    EXPECT_FALSE(h.run(in, out, &why, NmsHarness::kInputMax + 1));
    EXPECT_EQ(why, yvo::vk::kNmsFallbackCapacity);
}

// A pair whose IoU sits right at the 0.7 threshold (inter 70 / ua 100) is in the guard band → fallback bit 2.
TEST_F(EfficientNms, SyntheticIouGuardBand) {
    NmsHarness h(*ctx_);
    const std::vector<DetectionKeyed> in = {make_det(0, 0, 10, 10, 0.9f, 0, 1), make_det(0, 0, 10, 7, 0.8f, 0, 2)};
    std::vector<DetectionKeyed> out;
    std::uint32_t why = 0;
    EXPECT_FALSE(h.run(in, out, &why));
    EXPECT_EQ(why, yvo::vk::kNmsFallbackIouGuard);
    // Outside the guard band (0.69 / 0.71) finishes on the GPU and matches the CPU
    EXPECT_TRUE(expect_matches_cpu(h, {make_det(0, 0, 10, 10, 0.9f, 0, 1), make_det(0, 0, 10, 6.9f, 0.8f, 0, 2)}));
    EXPECT_TRUE(expect_matches_cpu(h, {make_det(0, 0, 10, 10, 0.9f, 0, 1), make_det(0, 0, 10, 7.1f, 0.8f, 0, 2)}));
}

// Random clustered boxes: bit-exact with the CPU, and byte-identical results over 10 runs with shuffled input order (determinism).
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
            // Scores are quantized values like the fixture (ties occur)
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
            EXPECT_EQ(diff_bits(out, ref), "") << "trial " << trial << " rep " << rep << ": result depended on input order";
        }
    }
    std::printf("[nms] random 20 trials: compared %d, guard-band fallback %d\n", compared, fallback);
    EXPECT_GT(compared, 15);
}

// ═══════════════════════════════════════════════ PostprocLive (takes Hailo-8 exclusively)

enum LivePath { kHost = 0, kGpu = 1, kNumLivePaths = 2 };
const char* kLivePathName[kNumLivePaths] = {"host", "gpu"};

class PostprocLive : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::string video = env_str("YVO_TEST_VIDEO", YVO_TEST_VIDEO);
        const long want = env_long("YVO_POSTPROC_FRAMES", 100);
        // Device **absence** is a skip; a failure with the device present is a FAIL. So that boards without the driver
        // do not look like regression failures, while every other error here is treated as a real failure.
        if (!std::filesystem::exists("/dev/hailo0")) {
            skip_reason_ = "/dev/hailo0 not found — environment without Hailo-8";
            return;
        }
        if (!std::filesystem::exists(video)) {
            setup_error_ = "video not found: " + video;
            return;
        }
        results_.assign(kNumLivePaths, {});
        for (auto& r : results_) r.resize(static_cast<std::size_t>(want));
        // Open pass 2 (gpu) only after pass 1 (host) is fully finished and its objects destroyed — there is one Hailo device.
        run_pass(kHost, video, want);
        if (setup_error_.empty() && frames_ > 0) run_pass(kGpu, video, frames_);
    }

    // Full pass of one path. SandLetterbox(NHCW, dma-buf) → InferStage → drain per frame.
    static void run_pass(LivePath path, const std::string& video, long nframes) {
        std::string& err = path == kHost ? setup_error_ : gpu_error_;
        int export_fd = -1;
        try {
            yvo::vk::VulkanContext ctx;
            if (!ctx.dmabuf_export_supported()) {
                err = "Vulkan dma-buf export not supported";
                return;
            }
            yvo::vk::SandLetterboxConfig pcfg;
            pcfg.num_slots = 1;
            pcfg.export_dmabuf = true;
            pcfg.out_layout = yvo::TensorLayout::Nhcw;   // both paths take NHCW dma-buf input
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
                err = "input size mismatch: NPU " + std::to_string(stage->input_frame_size()) + ", GPU " +
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
                    if (path == kGpu) err = "not enough frames: " + std::to_string(n) + " / " + std::to_string(nframes);
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
                // There is only one slot: completion must be received before the next frame overwrites the same dma-buf.
                stage->drain();
                ++done;
            }
            if (path == kHost) frames_ = done;
            pre.clear_import_cache();
            dec.close();
            stage.reset();   // the runner holds the dma-buf mapping — destroy it before closing the fd
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

// Do the GPU postprocessing path's detections match the host path (HEF built-in NMS + HailoRT host postprocessing)?
// Gate 0.95, known passing value 0.9990 (total detections 9457, same as host).
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
    EXPECT_GT(total.ref, 0) << "host path has 0 detections";
    EXPECT_GE(ratio, kMinMatchRatio);
    EXPECT_EQ(infer_errors_, 0);
}

} // namespace
