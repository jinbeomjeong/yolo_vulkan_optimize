// Pure logic of the yvo_pipeline app (pipeline/slot_exchange.hpp, pipeline/input_pacer.hpp, toolkit/result_csv.hpp,
// toolkit/cli_parse.hpp, toolkit/frame_log.hpp)
// — no NPU, GPU or display needed.
//
//   SlotExchange.*        slot pool + completion queue: FIFO order, two return conditions (compute + display),
//                         duplicate / stale / out-of-range signals ignored, complete() from another thread
//   SlotBudget.*          validate_slot_budget(): the --display on startup gate on decoder frames held at once.
//                         Expected messages are the exact text the check printed when it was inline in main.cpp parse().
//                         plane_frames_held(): the plane-path count shared with yvo_pipeline's extra_hw_frames.
//   LeakyQueue.*          --input-pace source queue: FIFO, a full queue drops (and hands back) the oldest
//   ArrivalSchedule.*     t_arr(k) = t0 + k/fps on an absolute base, the boundary counts as due
//   InputPacer.*          max_gap() (longest dropped run) and the validate_input_queue() startup gate
//   ResultCsv.*           the 22-field RESULT_CSV line of yvo_pipeline / yvo_gst_pipeline, byte-exact
//                         (expected lines are real lines printed by the per-tool printf it replaced).
//                         RESULT_CSV4 (paced only): field count and a real line, byte-exact.
//   CliParse.*            strict option-value parsing: same values as atoi/atol/atof for well-formed input,
//                         rejection of what they silently mangled, exit code 2 + message from arg_number().
//   FrameLog.*            --frame-log CSV: exact header, processed / dropped / abandoned / asap rows, empty fields.
//
// Links yvo_stages only for the header include paths (InferResult comes from hailo/hailo_runner.hpp);
// nothing here opens /dev/hailo0 or creates a Vulkan instance.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pipeline/input_pacer.hpp"
#include "pipeline/slot_exchange.hpp"
#include "toolkit/cli_parse.hpp"
#include "toolkit/frame_log.hpp"
#include "toolkit/result_csv.hpp"

namespace {

using yvo::pipeline::Completion;
using yvo::pipeline::SlotBudgetPath;
using yvo::pipeline::SlotExchange;
using yvo::pipeline::validate_slot_budget;

yvo::InferResult result_for(int slot, std::int64_t pts = 0) {
    yvo::InferResult r;
    r.slot = slot;
    r.pts = pts;
    return r;
}

// Acquires every free slot with try_acquire and returns them in acquisition order.
std::vector<int> drain_free(SlotExchange& ex) {
    std::vector<int> got;
    while (const auto s = ex.try_acquire()) got.push_back(*s);
    return got;
}

// ═══════════════════════════════════════════════ SlotExchange

TEST(SlotExchange, InitialSlotsComeOutInAscendingOrder) {
    SlotExchange ex(4, false);
    EXPECT_EQ(drain_free(ex), (std::vector<int>{0, 1, 2, 3}));
    EXPECT_FALSE(ex.try_acquire().has_value());
}

TEST(SlotExchange, ZeroSlotsNeverYieldsASlot) {
    SlotExchange ex(0, false);
    EXPECT_FALSE(ex.try_acquire().has_value());
    EXPECT_FALSE(ex.acquire_poll(std::chrono::milliseconds(1)).has_value());
}

TEST(SlotExchange, WithoutDisplayComputeAloneReturnsSlot) {
    SlotExchange ex(2, false);
    ASSERT_EQ(drain_free(ex), (std::vector<int>{0, 1}));
    ex.complete(result_for(1));
    const auto s = ex.try_acquire();
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(*s, 1);
    EXPECT_FALSE(ex.try_acquire().has_value());
}

TEST(SlotExchange, ReturnedSlotsAreReusedInReturnOrder) {
    SlotExchange ex(4, false);
    ASSERT_EQ(drain_free(ex).size(), 4u);
    ex.complete(result_for(2));
    ex.complete(result_for(0));
    ex.complete(result_for(3));
    EXPECT_EQ(drain_free(ex), (std::vector<int>{2, 0, 3}));
}

TEST(SlotExchange, WithDisplayBothConditionsAreRequired) {
    SlotExchange ex(1, true);
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    ex.complete(result_for(0));
    EXPECT_FALSE(ex.try_acquire().has_value()) << "compute done alone must not return the slot";
    ex.display_done(0);
    EXPECT_EQ(ex.try_acquire(), std::optional<int>(0));
}

TEST(SlotExchange, DisplayAckMayArriveBeforeCompute) {
    SlotExchange ex(1, true);
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    ex.display_done(0);
    EXPECT_FALSE(ex.try_acquire().has_value()) << "display done alone must not return the slot";
    ex.complete(result_for(0));
    EXPECT_EQ(ex.try_acquire(), std::optional<int>(0));
}

TEST(SlotExchange, DisplayAckIsIgnoredWithoutDisplayCondition) {
    // need_display == false: display_done only sets a flag that is never consulted; it must not free the slot early.
    SlotExchange ex(1, false);
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    ex.display_done(0);
    EXPECT_FALSE(ex.try_acquire().has_value());
    ex.complete(result_for(0));
    EXPECT_EQ(ex.try_acquire(), std::optional<int>(0));
}

TEST(SlotExchange, DuplicateComputeSignalDoesNotDuplicateSlot) {
    SlotExchange ex(2, false);
    ASSERT_EQ(drain_free(ex).size(), 2u);
    ex.complete(result_for(0, 10));
    ex.complete(result_for(0, 11));   // slot 0 is no longer out → slot signal ignored
    EXPECT_EQ(drain_free(ex), (std::vector<int>{0}));
    // Both results are still queued for the recorder.
    const auto done = ex.take_done();
    ASSERT_EQ(done.size(), 2u);
    EXPECT_EQ(done[0].result.pts, 10);
    EXPECT_EQ(done[1].result.pts, 11);
}

TEST(SlotExchange, DuplicateDisplaySignalDoesNotDuplicateSlot) {
    SlotExchange ex(2, true);
    ASSERT_EQ(drain_free(ex).size(), 2u);
    ex.display_done(1);
    ex.display_done(1);
    ex.complete(result_for(1));
    ex.display_done(1);   // after the return: ignored
    EXPECT_EQ(drain_free(ex), (std::vector<int>{1}));
}

TEST(SlotExchange, SignalForSlotThatIsNotOutDoesNotCarryOver) {
    // A stale display_done for a free slot must not pre-satisfy the next use of that slot.
    SlotExchange ex(1, true);
    ex.display_done(0);   // slot 0 is free (not out) → ignored
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    ex.complete(result_for(0));
    EXPECT_FALSE(ex.try_acquire().has_value()) << "stale display ack leaked into the new use of the slot";
    ex.display_done(0);
    EXPECT_EQ(ex.try_acquire(), std::optional<int>(0));
}

TEST(SlotExchange, ReacquiredSlotStartsWithClearedFlags) {
    SlotExchange ex(1, true);
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    ex.complete(result_for(0));
    ex.display_done(0);
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    ex.complete(result_for(0));   // second use: compute only
    EXPECT_FALSE(ex.try_acquire().has_value());
}

TEST(SlotExchange, OutOfRangeSlotIsIgnoredButResultIsQueued) {
    SlotExchange ex(2, false);
    ASSERT_EQ(drain_free(ex).size(), 2u);
    ex.complete(result_for(-1, 1));
    ex.complete(result_for(2, 2));
    ex.display_done(-1);
    ex.display_done(99);
    EXPECT_FALSE(ex.try_acquire().has_value());
    const auto done = ex.take_done();
    ASSERT_EQ(done.size(), 2u);
    EXPECT_EQ(done[0].result.slot, -1);
    EXPECT_EQ(done[1].result.slot, 2);
}

TEST(SlotExchange, TakeDoneKeepsCompletionOrderAndClears) {
    SlotExchange ex(3, true);
    ASSERT_EQ(drain_free(ex).size(), 3u);
    ex.complete(result_for(2, 200));
    ex.complete(result_for(0, 0));
    ex.complete(result_for(1, 100));
    const auto done = ex.take_done();
    ASSERT_EQ(done.size(), 3u);
    EXPECT_EQ(done[0].result.slot, 2);
    EXPECT_EQ(done[1].result.slot, 0);
    EXPECT_EQ(done[2].result.slot, 1);
    EXPECT_EQ(done[2].result.pts, 100);
    EXPECT_LE(done[0].t_done, done[1].t_done);
    EXPECT_LE(done[1].t_done, done[2].t_done);
    EXPECT_TRUE(ex.take_done().empty());
    // take_done does not return slots — display acks are still missing.
    EXPECT_FALSE(ex.try_acquire().has_value());
}

TEST(SlotExchange, AcquirePollReturnsImmediatelyWhenFree) {
    SlotExchange ex(1, false);
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(ex.acquire_poll(std::chrono::seconds(5)), std::optional<int>(0));
    EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(1));
}

TEST(SlotExchange, AcquirePollTimesOutWhenNothingIsReturned) {
    SlotExchange ex(1, false);
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_FALSE(ex.acquire_poll(std::chrono::milliseconds(20)).has_value());
    EXPECT_GE(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(15));
}

TEST(SlotExchange, AcquirePollWakesOnCompleteFromAnotherThread) {
    SlotExchange ex(1, false);
    ASSERT_EQ(ex.try_acquire(), std::optional<int>(0));
    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ex.complete(result_for(0, 7));
    });
    const auto t0 = std::chrono::steady_clock::now();
    const auto s = ex.acquire_poll(std::chrono::seconds(5));
    const auto waited = std::chrono::steady_clock::now() - t0;
    producer.join();
    ASSERT_EQ(s, std::optional<int>(0));
    EXPECT_LT(waited, std::chrono::seconds(4)) << "woke by timeout, not by notify";
    const auto done = ex.take_done();
    ASSERT_EQ(done.size(), 1u);
    EXPECT_EQ(done[0].result.pts, 7);
}

// The yvo_pipeline thread shape: main thread acquires slots and acks display; a worker thread (the HailoRT callback
// thread stand-in) calls complete(). Every frame must complete exactly once, and a slot must never be handed out
// while it is still out.
void run_threaded_pipeline(int slots, bool need_display, int frames) {
    SlotExchange ex(slots, need_display);
    std::mutex qm;
    std::condition_variable qcv;
    std::deque<yvo::InferResult> work;
    bool closed = false;
    // 1 while the slot is handed out. Main sets it on acquire; the worker clears it right before complete(),
    // which is the earliest point the slot may legally come back (without display it is freed inside complete()).
    std::vector<std::atomic<int>> held(static_cast<std::size_t>(slots));
    for (auto& h : held) h.store(0);
    std::atomic<bool> double_out{false};

    std::thread worker([&] {
        for (;;) {
            yvo::InferResult r;
            {
                std::unique_lock<std::mutex> lk(qm);
                qcv.wait(lk, [&] { return closed || !work.empty(); });
                if (work.empty()) return;
                r = std::move(work.front());
                work.pop_front();
            }
            held.at(static_cast<std::size_t>(r.slot)).store(0);
            ex.complete(std::move(r));
        }
    });

    std::vector<int> seen(static_cast<std::size_t>(frames), 0);
    int submitted = 0, finished = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    auto collect = [&] {
        for (auto& c : ex.take_done()) {
            ++seen.at(static_cast<std::size_t>(c.result.pts));
            ++finished;
            if (need_display) ex.display_done(c.result.slot);
        }
    };
    while (finished < frames && std::chrono::steady_clock::now() < deadline) {
        collect();
        if (submitted < frames) {
            if (const auto s = ex.acquire_poll(std::chrono::milliseconds(2))) {
                if (held.at(static_cast<std::size_t>(*s)).exchange(1) != 0) double_out = true;
                {
                    std::lock_guard<std::mutex> lk(qm);
                    work.push_back(result_for(*s, submitted));
                }
                qcv.notify_one();
                ++submitted;
            }
        } else {
            std::this_thread::yield();
        }
    }
    {
        std::lock_guard<std::mutex> lk(qm);
        closed = true;
    }
    qcv.notify_one();
    worker.join();
    collect();

    EXPECT_FALSE(double_out) << "a slot was handed out twice";
    ASSERT_EQ(finished, frames) << "timed out: " << finished << "/" << frames;
    for (int i = 0; i < frames; ++i) EXPECT_EQ(seen[static_cast<std::size_t>(i)], 1) << "frame " << i;
    // Everything returned: all slots are free again.
    EXPECT_EQ(drain_free(ex).size(), static_cast<std::size_t>(slots));
}

TEST(SlotExchange, ThreadedCompleteWithoutDisplay) { run_threaded_pipeline(8, false, 20000); }

// With display the slot is freed by the main thread's own display_done, so acquire_poll often sleeps its full 2 ms slice
// (the same shape as main.cpp's acquisition loop) — fewer frames keep the test around one second.
TEST(SlotExchange, ThreadedCompleteWithDisplayAck) { run_threaded_pipeline(4, true, 2000); }

// ═══════════════════════════════════════════════ validate_slot_budget

constexpr int kExtra = 5;   // PlaneDisplay::kExtraHeldBeyondSlots today (display/plane_display.hpp)

const char* kStall = "The decoder stalls without an error when more than about 13 frames are held at once. ";

std::string composite_msg(int slots, int depth, long long sum) {
    return "--display on requires slots + gpu_depth <= 10 (now " + std::to_string(slots) + " + " +
           std::to_string(depth) + " = " + std::to_string(sum) + ").\n" + kStall + "Reduce --slots or --gpu-depth.\n";
}

std::string planes_msg(const char* backend, long long hold, int slots, bool latest) {
    return std::string("--display-backend ") + backend + ": decoder frames held would be " + std::to_string(hold) +
           ", exceeding 10 (slots " + std::to_string(slots) + " + 1 awaiting decode" +
           (latest ? " + 5 held by display" : "") + ").\n" + kStall + "Reduce --slots.\n";
}

TEST(SlotBudget, LimitIsTen) { EXPECT_EQ(yvo::pipeline::kMaxDecoderFramesHeld, 10); }

TEST(SlotBudget, CompositeAllowsUpToTen) {
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, false, 4, 3, kExtra), "");
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, false, 7, 3, kExtra), "");
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, true, 9, 1, kExtra), "");
}

TEST(SlotBudget, CompositeRejectsElevenWithExactMessage) {
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, false, 8, 3, kExtra), composite_msg(8, 3, 11));
    // Default --batch 8 → slots 16: --display on without --slots is rejected.
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, false, 16, 3, kExtra), composite_msg(16, 3, 19));
}

TEST(SlotBudget, CompositeIgnoresSyncAndPlaneExtra) {
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, true, 7, 3, 1000), "");
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, true, 8, 3, 0), composite_msg(8, 3, 11));
}

TEST(SlotBudget, PlanesMatchedAllowsSlotsPlusOneUpToTen) {
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::WaylandPlanes, false, 9, 3, kExtra), "");
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Kms, false, 9, 3, kExtra), "");
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::WaylandPlanes, false, 10, 1, kExtra),
              planes_msg("wayland-planes", 11, 10, false));
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Kms, false, 10, 1, kExtra), planes_msg("kms", 11, 10, false));
}

TEST(SlotBudget, PlanesLatestAddsDisplayHeldFrames) {
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::WaylandPlanes, true, 4, 3, kExtra), "");
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::WaylandPlanes, true, 5, 3, kExtra),
              planes_msg("wayland-planes", 11, 5, true));
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Kms, true, 6, 3, kExtra), planes_msg("kms", 12, 6, true));
}

TEST(SlotBudget, PlanesIgnoreGpuDepth) {
    // gpu_depth frames are inside the slots on the plane paths.
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Kms, false, 6, 1000, kExtra), "");
}

TEST(SlotBudget, HugeValuesDoNotOverflow) {
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Composite, false, INT_MAX, INT_MAX, kExtra),
              composite_msg(INT_MAX, INT_MAX, 2LL * INT_MAX));
    EXPECT_EQ(validate_slot_budget(SlotBudgetPath::Kms, true, INT_MAX, 1, kExtra),
              planes_msg("kms", static_cast<long long>(INT_MAX) + 1 + kExtra, INT_MAX, true));
}

TEST(SlotBudget, PlaneFramesHeld) {
    using yvo::pipeline::plane_frames_held;
    EXPECT_EQ(plane_frames_held(false, 4, kExtra), 5);              // matched: slots + 1
    EXPECT_EQ(plane_frames_held(true, 4, kExtra), 5 + kExtra);      // latest: + frames held by the display
    EXPECT_EQ(plane_frames_held(true, INT_MAX, kExtra), static_cast<long long>(INT_MAX) + 1 + kExtra);
}

// ═══════════════════════════════════════════════ RESULT_CSV (toolkit/result_csv.hpp)

using yvo::tool::ResultCsv;
using yvo::tool::format_result_csv;

// Values behind a real yvo_pipeline line (--frames 300 --quiet --csv-label X, 2026-10-11, printed by the old printf).
ResultCsv pipeline_sample() {
    return {.label = "X", .mode = "dmabuf", .layout = "nhcw", .batch = 8, .slots = 16, .frames = 300,
            .wall_s = 5.8612, .fps = 51.186, .cpu_ms_per_frame = 0.8783, .decode_mean = 3.4351, .gpu_mean = 1.8432,
            .gpu_p90 = 1.8851, .npu_mean = 290.056, .npu_p90 = 301.493, .e2e_mean = 320.526, .e2e_p90 = 431.254,
            .detections = 911, .hailo_t0 = static_cast<double>(80.3f), .hailo_t1 = static_cast<double>(83.3f),
            .soc_t0 = 73.0, .soc_t1 = 72.5, .ok = true};
}

TEST(ResultCsv, PipelineLineIsByteExact) {
    EXPECT_EQ(format_result_csv(pipeline_sample()),
              "RESULT_CSV,X,dmabuf,nhcw,8,16,300,5.861,51.19,0.878,3.435,1.843,1.885,290.06,301.49,320.53,431.25,911,"
              "80.3,83.3,73.0,72.5,1\n");
}

TEST(ResultCsv, GstLineWithNanDecodeIsByteExact) {
    // yvo_gst_pipeline --sink hailo: decode_ms_mean is nan (no blocking decode call). Real line, ok flipped to 0 here.
    const ResultCsv r{.label = "GH", .mode = "dmabuf", .layout = "nhcw", .batch = 8, .slots = 16, .frames = 200,
                      .wall_s = 3.9001, .fps = 51.2856, .cpu_ms_per_frame = 2.7262, .decode_mean = std::nan(""),
                      .gpu_mean = 1.9271, .gpu_p90 = 2.0669, .npu_mean = 300.861, .npu_p90 = 307.248, .e2e_mean = 318.921,
                      .e2e_p90 = 446.662, .detections = 606, .hailo_t0 = 79.2, .hailo_t1 = 82.2, .soc_t0 = 73.0,
                      .soc_t1 = 72.5, .ok = false};
    EXPECT_EQ(format_result_csv(r),
              "RESULT_CSV,GH,dmabuf,nhcw,8,16,200,3.900,51.29,2.726,nan,1.927,2.067,300.86,307.25,318.92,446.66,606,"
              "79.2,82.2,73.0,72.5,0\n");
}

TEST(ResultCsv, HasTwentyTwoFields) {
    const std::string line = format_result_csv(pipeline_sample());
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.back(), '\n');
    std::size_t commas = 0;
    for (const char c : line) commas += c == ',';
    EXPECT_EQ(commas + 1, 23u);   // "RESULT_CSV" tag + 22 fields
}

TEST(ResultCsv, LongLabelIsNotTruncated) {
    ResultCsv r = pipeline_sample();
    r.label = std::string(5000, 'L');
    const std::string line = format_result_csv(r);
    EXPECT_EQ(line.rfind("RESULT_CSV," + r.label + ",dmabuf,", 0), 0u);
    EXPECT_EQ(line.substr(line.size() - 3), ",1\n");
}

TEST(ResultCsv, PrintWritesTheFormattedLineToStdout) {
    std::fflush(stdout);
    testing::internal::CaptureStdout();
    yvo::tool::print_result_csv(pipeline_sample());
    std::fflush(stdout);
    EXPECT_EQ(testing::internal::GetCapturedStdout(), format_result_csv(pipeline_sample()));
}

// RESULT_CSV4: --input-pace source only. Values behind a real line (b1s1, --input-queue 1, --frames 300, 2026-10-12).
yvo::tool::ResultCsv4 paced_sample() {
    return {.label = "b1s1_q1", .input_queue = 1, .src_fps = 30.0, .arrived = 300, .processed = 243, .dropped = 57,
            .abandoned = 0, .drop_rate = 57.0 / 300.0, .max_drop_run = 1, .queue_depth_max = 1, .e2e_arr_mean = 58.679,
            .e2e_arr_median = 58.254, .e2e_arr_p90 = 72.729, .e2e_arr_max = 92.393, .queue_wait_mean = 14.887,
            .queue_wait_p90 = 28.840, .queue_wait_max = 48.071, .decode_late_mean = 2.927, .decode_late_p90 = 4.495,
            .decode_late_max = 24.317, .held_frames_max = 2, .ok = true};
}

TEST(ResultCsv, Csv4LineIsByteExact) {
    EXPECT_EQ(yvo::tool::format_result_csv4(paced_sample()),
              "RESULT_CSV4,b1s1_q1,1,30.000,300,243,57,0,0.1900,1,1,58.68,58.25,72.73,92.39,14.887,28.840,48.071,"
              "2.927,4.495,24.317,2,1\n");
}

TEST(ResultCsv, Csv4HasTwentyTwoFields) {
    const std::string line = yvo::tool::format_result_csv4(paced_sample());
    ASSERT_FALSE(line.empty());
    EXPECT_EQ(line.back(), '\n');
    std::size_t commas = 0;
    for (const char c : line) commas += c == ',';
    EXPECT_EQ(commas + 1, 23u);   // "RESULT_CSV4" tag + 22 fields
    EXPECT_EQ(line.rfind("RESULT_CSV4,b1s1_q1,", 0), 0u);
    EXPECT_EQ(line.substr(line.size() - 3), ",1\n");
}

// ═══════════════════════════════════════════════ input pacing (pipeline/input_pacer.hpp)

using yvo::pipeline::ArrivalSchedule;
using yvo::pipeline::LeakyQueue;

TEST(LeakyQueue, PopsOldestFirst) {
    LeakyQueue<int> q(3);
    EXPECT_FALSE(q.push(1).has_value());
    EXPECT_FALSE(q.push(2).has_value());
    EXPECT_FALSE(q.push(3).has_value());
    EXPECT_EQ(q.size(), 3u);
    EXPECT_EQ(q.pop_oldest(), std::optional<int>(1));
    EXPECT_EQ(q.pop_oldest(), std::optional<int>(2));
    EXPECT_EQ(q.pop_oldest(), std::optional<int>(3));
    EXPECT_TRUE(q.empty());
}

TEST(LeakyQueue, OverflowDropsAndReturnsTheOldest) {
    LeakyQueue<int> q(2);
    q.push(10);
    q.push(11);
    EXPECT_EQ(q.push(12), std::optional<int>(10));
    EXPECT_EQ(q.push(13), std::optional<int>(11));
    EXPECT_EQ(q.size(), 2u);
    EXPECT_EQ(q.pop_oldest(), std::optional<int>(12));   // order is kept after drops
    EXPECT_EQ(q.pop_oldest(), std::optional<int>(13));
}

TEST(LeakyQueue, CapacityOneKeepsOnlyTheNewest) {
    LeakyQueue<int> q(1);
    for (int i = 0; i < 5; ++i) {
        const auto dropped = q.push(i);
        EXPECT_EQ(dropped.has_value(), i > 0);
        if (dropped) {
            EXPECT_EQ(*dropped, i - 1);
        }
    }
    EXPECT_EQ(q.size(), 1u);
    EXPECT_EQ(q.pop_oldest(), std::optional<int>(4));
}

TEST(LeakyQueue, DroppedFrameReferenceIsReleased) {
    // The queue holds decoder frames through a shared_ptr keepalive; a drop must give the frame back at once.
    const auto first = std::make_shared<int>(1);
    const auto second = std::make_shared<int>(2);
    LeakyQueue<std::shared_ptr<int>> q(1);
    q.push(first);
    EXPECT_EQ(first.use_count(), 2);
    {
        auto dropped = q.push(second);
        ASSERT_TRUE(dropped.has_value());
        EXPECT_EQ(dropped->get(), first.get());
    }
    EXPECT_EQ(first.use_count(), 1);    // the caller let the dropped one go: only the test holds it now
    EXPECT_EQ(second.use_count(), 2);
    q.clear();
    EXPECT_EQ(second.use_count(), 1);
}

TEST(LeakyQueue, EmptyPopAndZeroCapacity) {
    LeakyQueue<int> q(0);   // clamped to 1
    EXPECT_EQ(q.capacity(), 1u);
    EXPECT_FALSE(q.pop_oldest().has_value());
    q.push(1);
    EXPECT_EQ(q.push(2), std::optional<int>(1));
    q.clear();
    EXPECT_TRUE(q.empty());
    EXPECT_FALSE(q.pop_oldest().has_value());
}

TEST(ArrivalSchedule, FrameZeroArrivesAtT0) {
    ArrivalSchedule s(30.0);
    const auto t0 = std::chrono::steady_clock::now();
    s.start(t0);
    EXPECT_EQ(s.arrival(0), t0);
    EXPECT_EQ(s.arrival(1) - t0, std::chrono::nanoseconds(33333333));
}

TEST(ArrivalSchedule, Frame900At30FpsIsExactlyThirtySeconds) {
    ArrivalSchedule s(30.0);
    const std::chrono::steady_clock::time_point t0{std::chrono::seconds(1000)};
    s.start(t0);
    EXPECT_EQ(s.arrival(900) - t0, std::chrono::seconds(30));
    EXPECT_EQ(s.arrival(1816) - t0, std::chrono::nanoseconds(60533333333));
}

TEST(ArrivalSchedule, NoDriftAtNtscRate) {
    // 30000/1001 fps: frame 30000 arrives exactly 1001 s later (absolute base, no accumulated rounding).
    ArrivalSchedule s(30000.0 / 1001.0);
    const std::chrono::steady_clock::time_point t0{};
    s.start(t0);
    EXPECT_EQ(s.arrival(30000) - t0, std::chrono::seconds(1001));
}

TEST(ArrivalSchedule, BoundaryCountsAsDue) {
    ArrivalSchedule s(25.0);
    const std::chrono::steady_clock::time_point t0{std::chrono::seconds(5)};
    s.start(t0);
    EXPECT_TRUE(s.due(0, t0));
    EXPECT_TRUE(s.due(2, t0 + std::chrono::milliseconds(80)));
    EXPECT_FALSE(s.due(2, t0 + std::chrono::milliseconds(80) - std::chrono::nanoseconds(1)));
}

TEST(InputPacer, MaxGapIsTheLongestDroppedRun) {
    using yvo::pipeline::max_gap;
    EXPECT_EQ(max_gap({}), 0);
    EXPECT_EQ(max_gap({0, 1, 2, 3}), 0);
    EXPECT_EQ(max_gap({0, 2, 4, 5}), 1);
    EXPECT_EQ(max_gap({0, 1, 5, 6, 8}), 3);
    EXPECT_EQ(max_gap({2, 3}), 2);   // a leading run counts from frame 0
}

TEST(InputPacer, ValidateInputQueue) {
    using yvo::pipeline::validate_input_queue;
    EXPECT_EQ(validate_input_queue(1, 3), "");
    EXPECT_EQ(validate_input_queue(3, 3), "");
    EXPECT_EQ(validate_input_queue(3, 6), "");   // 6 + 3 + 1 = 10
    EXPECT_EQ(validate_input_queue(0, 3), "--input-queue must be 1..3 (now 0).\n");
    EXPECT_EQ(validate_input_queue(4, 1), "--input-queue must be 1..3 (now 4).\n");
    EXPECT_EQ(validate_input_queue(3, 7),
              "--input-pace source requires gpu_depth + input_queue + 1 <= 10 (now 7 + 3 + 1 = 11).\n"
              "The decoder stalls without an error when more than about 13 frames are held at once. "
              "Reduce --gpu-depth or --input-queue.\n");
}

// ═══════════════════════════════════════════════ option values (toolkit/cli_parse.hpp)

using yvo::tool::parse_number;

template <class T>
std::optional<T> parsed(const char* text) {
    T v{};
    if (!parse_number(text, v)) return std::nullopt;
    return v;
}

TEST(CliParse, WellFormedIntegersMatchAtoi) {
    for (const char* t : {"0", "42", "-1", "+10", " 4", "\t7", "007", "-2147483648", "2147483647"}) {
        SCOPED_TRACE(t);
        EXPECT_EQ(parsed<int>(t), std::atoi(t));
        EXPECT_EQ(parsed<long>(t), std::atol(t));
    }
    EXPECT_EQ(parsed<long>("9000000000"), 9000000000L);
}

TEST(CliParse, WellFormedFloatsMatchAtof) {
    for (const char* t : {"0", "0.2", "-1", "2e-1", "1E3", ".5", "+0.25", " 0.45", "0x1p-2", "1e-30"}) {
        SCOPED_TRACE(t);
        EXPECT_EQ(parsed<double>(t), std::atof(t));
        EXPECT_EQ(parsed<float>(t), static_cast<float>(std::atof(t)));   // the old static_cast<float>(std::atof(v))
    }
}

TEST(CliParse, RejectsWhatAtoiSilentlyMangled) {
    for (const char* t : {"", "abc", "10x", "4.5", "1e3", "0x10", "5 ", " ", "-", "+", "--1", "1,000",
                          "99999999999999999999", "2147483648", "-2147483649"}) {
        SCOPED_TRACE(t);
        EXPECT_FALSE(parsed<int>(t).has_value());
    }
    EXPECT_FALSE(parsed<long>("99999999999999999999").has_value());   // ERANGE
    EXPECT_FALSE(parsed<long>(nullptr).has_value());
}

TEST(CliParse, RejectsWhatAtofSilentlyMangled) {
    for (const char* t : {"", "abc", "0.5abc", "0.5 ", "nan", "inf", "-infinity", "1e999", "1e-400", "0.2.3"}) {
        SCOPED_TRACE(t);
        EXPECT_FALSE(parsed<double>(t).has_value());
    }
    EXPECT_FALSE(parsed<float>("1e39").has_value());   // finite double, not representable as float
    EXPECT_TRUE(parsed<double>("1e39").has_value());
}

TEST(CliParse, UnsignedRejectsMinusInsteadOfWrapping) {
    EXPECT_EQ(parsed<unsigned>("7"), 7u);
    EXPECT_EQ(parsed<unsigned>(" +7"), 7u);
    EXPECT_FALSE(parsed<unsigned>("-1").has_value());
    EXPECT_FALSE(parsed<unsigned>(" -0").has_value());
    EXPECT_FALSE(parsed<std::uint16_t>("65536").has_value());
    EXPECT_EQ(parsed<std::uint16_t>("65535"), 65535u);
}

TEST(CliParse, FailureLeavesOutputUnchanged) {
    int v = 123;
    EXPECT_FALSE(parse_number("x", v));
    EXPECT_EQ(v, 123);
    double d = 0.5;
    EXPECT_FALSE(parse_number("nan", d));
    EXPECT_EQ(d, 0.5);
}

TEST(CliParse, ProgNameDropsDirectory) {
    EXPECT_EQ(yvo::tool::prog_name("/home/pi/build/bin/yvo_pipeline"), "yvo_pipeline");
    EXPECT_EQ(yvo::tool::prog_name("yvo_pipeline"), "yvo_pipeline");
    EXPECT_EQ(yvo::tool::prog_name(nullptr), "");
}

TEST(CliParse, ArgNumberReturnsValidValue) {
    EXPECT_EQ(yvo::tool::arg_number<long>("prog", "--frames", "900"), 900L);
    EXPECT_EQ(yvo::tool::arg_number<float>("prog", "--score", "0.2"), static_cast<float>(std::atof("0.2")));
}

TEST(CliParseDeathTest, ArgNumberExitsWithCodeTwoAndNamesTheOption) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");   // other tests in this binary start threads
    EXPECT_EXIT(yvo::tool::arg_number<long>("/x/bin/yvo_pipeline", "--frames", "abc"), testing::ExitedWithCode(2),
                "^yvo_pipeline: invalid value for --frames: 'abc'\n$");
    EXPECT_EXIT(yvo::tool::arg_number<int>("tool", "YVO_DEC_THREADS", ""), testing::ExitedWithCode(2),
                "^tool: invalid value for YVO_DEC_THREADS: ''\n$");
}

// ── --frame-log (toolkit/frame_log.hpp) ─────────────────────────────────────────────────────────────────────────

TEST(FrameLog, HeaderIsExact) {
    EXPECT_EQ(yvo::tool::format_frame_log_header(),
              "src,seq,dropped,arrival_ms,decode_end_ms,submit_ms,npu_done_ms,gpu_post_done_ms,complete_ms\n");
}

TEST(FrameLog, ProcessedPacedRowHasEveryField) {
    yvo::tool::FrameLogRow r;
    r.src = 12;
    r.seq = 9;
    r.arrival_ms = 400.0;
    r.decode_end_ms = 401.25;
    r.submit_ms = 402.0004;     // rounds down
    r.npu_done_ms = 440.0006;   // rounds up
    r.gpu_post_done_ms = 441.5;
    r.complete_ms = 441.6789;
    EXPECT_EQ(yvo::tool::format_frame_log_row(r), "12,9,0,400.000,401.250,402.000,440.001,441.500,441.679\n");
}

TEST(FrameLog, DroppedAndAbandonedRowsKeepOnlyArrivalAndDecode) {
    yvo::tool::FrameLogRow r;
    r.src = 3;
    r.dropped = 1;
    r.arrival_ms = 100.0;
    r.decode_end_ms = 100.5;
    EXPECT_EQ(yvo::tool::format_frame_log_row(r), "3,,1,100.000,100.500,,,,\n");
    r.dropped = 2;
    EXPECT_EQ(yvo::tool::format_frame_log_row(r), "3,,2,100.000,100.500,,,,\n");
}

TEST(FrameLog, AsapHostRowLeavesArrivalAndGpuPostEmpty) {
    yvo::tool::FrameLogRow r;
    r.src = 0;
    r.seq = 0;
    r.decode_end_ms = 0.107;
    r.submit_ms = 0.2;
    r.npu_done_ms = 60.0;
    r.complete_ms = 62.5;
    EXPECT_EQ(yvo::tool::format_frame_log_row(r), "0,0,0,,0.107,0.200,60.000,,62.500\n");
}

TEST(FrameLog, NonFiniteTimesAreEmptyAndFieldCountIsFixed) {
    yvo::tool::FrameLogRow r;   // all times kNone
    r.src = 7;
    r.seq = 7;
    r.complete_ms = std::numeric_limits<double>::infinity();
    const std::string line = yvo::tool::format_frame_log_row(r);
    EXPECT_EQ(line, "7,7,0,,,,,,\n");
    // 9 fields like the header: 8 commas.
    EXPECT_EQ(std::count(line.begin(), line.end(), ','), 8);
    const std::string head = yvo::tool::format_frame_log_header();
    EXPECT_EQ(std::count(head.begin(), head.end(), ','), 8);
}

} // namespace
