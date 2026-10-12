// Input pacing for yvo_pipeline --input-pace source: frames "arrive" at the source frame rate (like a camera) and
// wait in a small leaky queue after decode, the way GStreamer's `queue leaky=downstream max-size-buffers=N` does.
//
// Header-only, no hardware — unit-tested in 05_tests/test_pipeline_logic.cpp, like slot_exchange.hpp.
//
//   LeakyQueue<T>        FIFO with capacity N >= 1. push() on a full queue drops the **oldest** element and hands it
//                        back to the caller (who releases it right away, so the decoder gets its frame back).
//                        pop_oldest() takes the oldest too, so frame order is kept (N = 1 keeps only the newest).
//   ArrivalSchedule      t_arr(k) = t0 + k / fps on an absolute base (no accumulated drift). A frame is due when
//                        now >= t_arr(k) (the boundary counts as due).
//   max_gap()            longest run of consecutive source frames that never got submitted (dropped).
//   validate_input_queue the startup gate: 1 <= N <= 3 and gpu_depth + N + 1 <= kMaxDecoderFramesHeld
//                        (N queued + gpu_depth on the GPU + 1 just decoded while the queue is full).
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pipeline/slot_exchange.hpp"

namespace yvo::pipeline {

inline constexpr int kMinInputQueue = 1;
inline constexpr int kMaxInputQueue = 3;

template <class T>
class LeakyQueue {
public:
    explicit LeakyQueue(std::size_t capacity) : cap_(capacity < 1 ? 1 : capacity) {}

    // Appends v. When the queue was already full, the oldest element is removed and returned (dropped).
    std::optional<T> push(T v) {
        std::optional<T> dropped;
        if (q_.size() >= cap_) {
            dropped.emplace(std::move(q_.front()));
            q_.pop_front();
        }
        q_.push_back(std::move(v));
        return dropped;
    }
    // Removes and returns the oldest element (nullopt when empty).
    std::optional<T> pop_oldest() {
        if (q_.empty()) return std::nullopt;
        std::optional<T> out(std::move(q_.front()));
        q_.pop_front();
        return out;
    }
    std::size_t size() const { return q_.size(); }
    bool empty() const { return q_.empty(); }
    std::size_t capacity() const { return cap_; }
    void clear() { q_.clear(); }

private:
    std::size_t cap_;
    std::deque<T> q_;
};

class ArrivalSchedule {
public:
    using Clock = std::chrono::steady_clock;

    explicit ArrivalSchedule(double fps) : fps_(fps) {}
    void start(Clock::time_point t0) { t0_ = t0; }
    double fps() const { return fps_; }
    Clock::time_point t0() const { return t0_; }
    // Arrival time of source frame k, computed from t0 every time (no accumulated rounding).
    Clock::time_point arrival(std::int64_t k) const {
        const double ns = static_cast<double>(k) * 1e9 / fps_;
        return t0_ + std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(std::llround(ns)));
    }
    bool due(std::int64_t k, Clock::time_point now) const { return now >= arrival(k); }

private:
    double fps_;
    Clock::time_point t0_{};
};

// src: source frame indices that were submitted, ascending (FIFO keeps them ascending). Returns the longest run of
// missing indices, counting a leading run from frame 0. Frames after the last submitted one are not counted
// (with pacing they are either drained at EOF or abandoned on a stop, never dropped).
inline std::int64_t max_gap(const std::vector<std::int64_t>& src) {
    std::int64_t best = 0, prev = -1;
    for (const std::int64_t s : src) {
        if (s - prev - 1 > best) best = s - prev - 1;
        prev = s;
    }
    return best;
}

// Returns "" when the --input-queue value fits, otherwise the complete stderr message.
inline std::string validate_input_queue(int n, int gpu_depth) {
    if (n < kMinInputQueue || n > kMaxInputQueue)
        return "--input-queue must be " + std::to_string(kMinInputQueue) + ".." + std::to_string(kMaxInputQueue) +
               " (now " + std::to_string(n) + ").\n";
    const long long sum = static_cast<long long>(gpu_depth) + n + 1;
    if (sum > kMaxDecoderFramesHeld)
        return "--input-pace source requires gpu_depth + input_queue + 1 <= " + std::to_string(kMaxDecoderFramesHeld) +
               " (now " + std::to_string(gpu_depth) + " + " + std::to_string(n) + " + 1 = " + std::to_string(sum) +
               ").\nThe decoder stalls without an error when more than about 13 frames are held at once. "
               "Reduce --gpu-depth or --input-queue.\n";
    return {};
}

} // namespace yvo::pipeline
