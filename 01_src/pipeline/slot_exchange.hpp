// SlotExchange: thread-safe slot pool + completion queue shared by the inference completion side and the main thread
// (yvo_pipeline). validate_slot_budget: the startup gate on decoder frames held at once when --display is on.
//
// Both used to live in the anonymous namespace / parse() of pipeline/main.cpp. They were moved here unchanged
// (2026-10-11) so that they can be unit-tested without NPU or GPU (05_tests/test_pipeline_logic.cpp).
//
// ── Slot contract (slot numbers shared by GPU, NPU and display, 0..n-1) ──────────────────────────────────────
//   try_acquire / acquire_poll  → the slot is "out"
//   complete(result)            → compute done  (HailoRT callback thread for host post-processing, main thread for gpu)
//   display_done(slot)          → display done  (main thread only)
//   A slot goes back to the free list (FIFO) once compute is done **and** (display is done or need_display == false).
//   Signals for slots that are not out are ignored, so a duplicate complete/display_done is harmless.
//   complete() always queues the result for take_done(), even when its slot signal is ignored.
//   With need_display == false there is no display condition at all (the --display off regression path).
// ── Wake (event-driven main-thread waits) ─────────────────────────────────────────────────────────────────────
//   wake()                      → "an NPU completion is queued inside the stage" (GpuPostprocStage's HailoRT callback)
//   acquire_poll / wait_event_until return early on wake() or on a queued completion (done_ not empty), so the main
//   thread does not sleep out its slice. The wake flag is set under m_ → a wake() before the wait is not lost.
//   Every return clears it: the caller must pump the stage (and take_done()) before waiting again.
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <iterator>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hailo/hailo_runner.hpp"

namespace yvo::pipeline {

struct Completion {
    yvo::InferResult result;
    std::chrono::steady_clock::time_point t_done;
};

// There are two slot return conditions (compute done + display done). With need_display == false there is no display condition,
// giving **exactly the same behavior as before** (--display off regression path).
class SlotExchange {
public:
    using Clock = std::chrono::steady_clock;

    SlotExchange(int n, bool need_display)
        : need_display_(need_display), ready_(static_cast<std::size_t>(n), {false, false}),
          out_(static_cast<std::size_t>(n), false) {
        for (int i = 0; i < n; ++i) free_.push_back(i);
    }
    // HailoRT callback thread (host) or main thread (gpu post-processing)
    void complete(yvo::InferResult&& r) {
        Completion c{std::move(r), Clock::now()};
        {
            std::lock_guard<std::mutex> lk(m_);
            const int slot = c.result.slot;
            done_.push_back(std::move(c));
            mark(slot, 0);
        }
        cv_.notify_all();
    }
    // Main thread only. Display has finished (display completion callback) or display was skipped.
    void display_done(int slot) {
        {
            std::lock_guard<std::mutex> lk(m_);
            mark(slot, 1);
        }
        cv_.notify_all();
    }
    std::optional<int> try_acquire() {
        std::lock_guard<std::mutex> lk(m_);
        return pop_locked();
    }
    // Waits only for slice (no blocking — the one who sets display_done is the caller itself). slice <= 0: no wait.
    // Returns early (possibly without a slot) on wake() or a queued completion — take_done() before calling.
    std::optional<int> acquire_poll(Clock::duration slice) {
        std::unique_lock<std::mutex> lk(m_);
        if (free_.empty()) cv_.wait_for(lk, slice, [&] { return !free_.empty() || event_locked(); });
        woken_ = false;
        return pop_locked();
    }
    // Main-thread idle wait that wants no slot (paced: until the next arrival). Returns at deadline, or earlier on
    // wake() or a queued completion (take_done() before calling, or it returns at once). true = an event ended it.
    bool wait_event_until(Clock::time_point deadline) {
        std::unique_lock<std::mutex> lk(m_);
        const bool ev = cv_.wait_until(lk, deadline, [&] { return event_locked(); });
        woken_ = false;
        return ev;
    }
    // Any thread (GpuPostprocStage's HailoRT callback via InferStage::set_wake_hook). Ends the current or next wait.
    void wake() {
        {
            std::lock_guard<std::mutex> lk(m_);
            woken_ = true;
        }
        cv_.notify_all();
    }
    std::vector<Completion> take_done() {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<Completion> out(std::make_move_iterator(done_.begin()), std::make_move_iterator(done_.end()));
        done_.clear();
        return out;
    }

private:
    bool event_locked() const { return woken_ || !done_.empty(); }   // called with m_ held
    // Called with m_ held. Signals for slots that are not out are ignored (makes duplicate calls harmless).
    void mark(int slot, int which) {
        const auto i = static_cast<std::size_t>(slot);
        if (i >= out_.size() || !out_[i]) return;
        ready_[i][static_cast<std::size_t>(which)] = true;
        if (ready_[i][0] && (ready_[i][1] || !need_display_)) {
            ready_[i] = {false, false};
            out_[i] = false;
            free_.push_back(slot);
        }
    }
    std::optional<int> pop_locked() {
        if (free_.empty()) return std::nullopt;
        const int s = free_.front();
        free_.pop_front();
        const auto i = static_cast<std::size_t>(s);
        ready_[i] = {false, false};
        out_[i] = true;
        return s;
    }

    std::mutex m_;
    std::condition_variable cv_;
    bool need_display_;
    std::vector<std::array<bool, 2>> ready_;   // [0] = compute done, [1] = display done
    std::vector<bool> out_;                    // whether the slot is out (guards against duplicate signals)
    std::deque<int> free_;
    std::deque<Completion> done_;
    bool woken_ = false;                       // wake() since the last acquire_poll / wait_event_until return
};

// ── Decoder frame budget (--display on) ──────────────────────────────────────────────────────────────────────
// The decoder **stalls without an error** when more than about 13 frames are held at once; we block at 10 for a margin.
inline constexpr int kMaxDecoderFramesHeld = 10;

// Decoder frames held at once with --display on (PlaneDisplay, both sync modes — "matched-detach": the slot comes back
// right after show(), and the display keeps the frame outside the slots until the compositor releases it):
//   slots (one retained frame per slot: SandLetterbox / display retain) + 1 (one decoded frame awaiting a slot)
//   + extra_held_beyond_slots (PlaneDisplay::kExtraHeldBeyondSlots: pending + committed, unreleased)
//   + input_queue (frames waiting in --input-pace source's leaky queue, 0 with asap).
// Shared by validate_slot_budget() and yvo_pipeline's extra_hw_frames. long long so absurd --slots values cannot overflow int.
inline long long plane_frames_held(int slots, int extra_held_beyond_slots, int input_queue = 0) {
    return static_cast<long long>(slots) + 1 + extra_held_beyond_slots + input_queue;
}

// Startup gate, called by yvo_pipeline parse() only when --display is on. Returns "" when the budget fits,
// otherwise the complete stderr message.
// extra_held_beyond_slots is PlaneDisplay::kExtraHeldBeyondSlots, passed in so pipeline/ does not need display/ here.
// input_queue = --input-queue N with --input-pace source (0 with asap).
std::string validate_slot_budget(int slots, int extra_held_beyond_slots, int input_queue = 0);

} // namespace yvo::pipeline
