// Pure logic of the yvo_pipeline app (pipeline/slot_exchange.hpp) — no NPU, GPU or display needed.
//
//   SlotExchange.*        slot pool + completion queue: FIFO order, two return conditions (compute + display),
//                         duplicate / stale / out-of-range signals ignored, complete() from another thread
//   SlotBudget.*          validate_slot_budget(): the --display on startup gate on decoder frames held at once.
//                         Expected messages are the exact text the check printed when it was inline in main.cpp parse().
//
// Links yvo_stages only for the header include paths (InferResult comes from hailo/hailo_runner.hpp);
// nothing here opens /dev/hailo0 or creates a Vulkan instance.
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pipeline/slot_exchange.hpp"

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

} // namespace
