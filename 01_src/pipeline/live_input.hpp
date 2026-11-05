// Live input for yvo_pipeline --input-live: packets reach the decoder only at their arrival time, like a camera or a
// network stream, instead of being read ahead from the file. Plain --input-pace source lets the decoder take later packets
// as soon as it wants them, so frame threads finish a frame before its arrival time and hide their output delay.
//
// Header-only, no hardware: the decoder is a template parameter so 05_tests/test_pipeline_logic.cpp can use a fake.
//
//   LiveDecodeThread<Decoder>  runs Decoder::next() on its own thread. The decoder's packet gate sleeps until
//                              t_arr(k) = t0 + k / fps before reading packet k, so EAGAIN from the decoder turns into a
//                              wait for the next arrival. Finished frames go to a small mailbox for the main thread.
//
// Why a thread: next() blocks for a whole packet interval or longer here, and the main thread must keep pumping NPU
// completions, GPU post-processing and display meanwhile (the --decoder sw main-thread stall in
// 07_results/decode_benchmark/runs/pipeline_sw_hw/summary.md). The decoder is used by this thread only from construction
// until the destructor returns; frames may be released on any thread (AVBufferPool and SandFramePool lock).
//
// End of stream: flush would release the frames held by the decoder all at once, which no live source does. Frames after
// the flush keep the arrival cadence (frame N - D + i leaves at t_arr(N + i)), so the last D frames have the same delay as
// the others and do not overflow the leaky queue.
//
// Decoder requirements: next() -> std::optional<Frame>, set_packet_gate(std::function<bool(std::int64_t)>),
// packets_sent(), input_flushed(), thread_count(), reorder_frames().
#pragma once

#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "pipeline/input_pacer.hpp"

namespace yvo::pipeline {

// Finished frames waiting for the main thread. Counted in the startup gates on decoder frames held at once. Full: the
// decode thread waits (it never drops: frame indices stay contiguous, the leaky queue is the only place that drops).
inline constexpr int kLiveMailboxFrames = 2;

struct LiveInputStats {
    long packets = 0;                // packets sent to the decoder
    long frames = 0;                 // frames returned by the decoder
    long output_delay_max = 0;       // packets sent minus frames returned, before the flush (frame threads - 1 + reorder)
    int decoder_threads = 0;         // resolved thread count of the decoder
    int reorder_frames = 0;          // reorder delay the stream declares (0 = no B-frame reordering)
    double send_late_mean_ms = 0;    // how late packets were read relative to t_arr(k): a decoder falling behind
    double send_late_max_ms = 0;
    long mailbox_full_waits = 0;     // the main thread did not collect in time
};

// Printed with --quiet too: live_input_bench.sh reads this line from stdout.log.
inline void print_live_stats(const LiveInputStats& s) {
    std::printf("input live: packets %ld, frames %ld, output delay max %ld packets (decoder threads %d, stream reorder %d), "
                "send late mean %.3f max %.3f ms, mailbox full waits %ld\n",
                s.packets, s.frames, s.output_delay_max, s.decoder_threads, s.reorder_frames, s.send_late_mean_ms,
                s.send_late_max_ms, s.mailbox_full_waits);
}

template <class Decoder>
class LiveDecodeThread {
public:
    using Clock = ArrivalSchedule::Clock;
    using Frame = typename decltype(std::declval<Decoder&>().next())::value_type;

    // sched must be started. wake runs on the decode thread when a frame or the end of the stream is ready.
    LiveDecodeThread(Decoder& dec, const ArrivalSchedule& sched, std::function<void()> wake)
        : dec_(dec), sched_(sched), wake_(std::move(wake)) {
        dec_.set_packet_gate([this](std::int64_t k) { return gate(k); });
        thread_ = std::thread([this] { run(); });
    }
    LiveDecodeThread(const LiveDecodeThread&) = delete;
    LiveDecodeThread& operator=(const LiveDecodeThread&) = delete;
    ~LiveDecodeThread() {
        halt();
        {
            const std::lock_guard<std::mutex> lk(m_);
            release_ = true;
        }
        cv_.notify_all();
        thread_.join();
        dec_.set_packet_gate({});
    }

    // Main thread, never blocks. false = nothing yet. true with f set = a frame, t_begin..t_end its decode work (waiting for
    // packets excluded) ending at decode return. true with f empty = end of stream, or the decoder failed (decoder error()
    // after the destructor). Rethrows what the decode thread caught.
    bool poll(std::optional<Frame>& f, Clock::time_point& t_begin, Clock::time_point& t_end) {
        std::unique_lock<std::mutex> lk(m_);
        if (mailbox_.empty()) {
            if (!ended_) return false;
            if (failure_) std::rethrow_exception(failure_);
            f.reset();
            return true;
        }
        const bool was_full = mailbox_.size() >= static_cast<std::size_t>(kLiveMailboxFrames);
        Item item = std::move(mailbox_.front());
        mailbox_.pop_front();
        lk.unlock();
        if (was_full) cv_.notify_all();
        f.emplace(std::move(item.frame));
        t_begin = item.t_begin;
        t_end = item.t_end;
        return true;
    }

    // Stops decoding and releases the frames in the mailbox. The thread stays alive, parked, until the destructor: per-thread
    // CPU can only be read while the thread exists (toolkit/tool_util.hpp thread_cpu_groups).
    void halt() {
        std::deque<Item> released;
        std::unique_lock<std::mutex> lk(m_);
        halt_ = true;
        cv_.notify_all();
        parked_cv_.wait(lk, [this] { return parked_; });
        released.swap(mailbox_);
    }

    // After halt().
    LiveInputStats stats() const {
        const std::lock_guard<std::mutex> lk(m_);
        return stats_;
    }

    // Idle wait of the main thread while no frame is ready; wake ends it earlier, this only bounds it.
    Clock::duration idle_wait() const {
        return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / sched_.fps()));
    }

private:
    struct Item {
        Frame frame;
        Clock::time_point t_begin, t_end;
    };

    // Decode thread. true when t arrived; false when halted first.
    bool wait_arrival(Clock::time_point t) {
        std::unique_lock<std::mutex> lk(m_);
        return !cv_.wait_until(lk, t, [this] { return halt_; });
    }

    // Decode thread, inside Decoder::next(): before the read of packet k (k = the count sent so far, also at end of file).
    bool gate(std::int64_t k) {
        const auto t_arr = sched_.arrival(k);
        const auto g0 = Clock::now();
        if (!wait_arrival(t_arr)) return false;
        const auto g1 = Clock::now();
        gate_wait_ += g1 - g0;
        if (k != last_gated_) {   // a non-video packet asks again with the same k
            last_gated_ = k;
            const double late = std::max(0.0, std::chrono::duration<double, std::milli>(g1 - t_arr).count());
            late_sum_ms_ += late;
            late_max_ms_ = std::max(late_max_ms_, late);
            ++late_samples_;
        }
        return true;
    }

    void decode_loop() {
        std::int64_t tail = 0;   // frames returned after the flush
        for (;;) {
            gate_wait_ = Clock::duration::zero();
            const auto n0 = Clock::now();
            std::optional<Frame> f = dec_.next();
            const auto busy = (Clock::now() - n0) - gate_wait_;
            if (!f) return;   // end of stream, decoder error or halt
            ++frames_;
            const std::int64_t sent = dec_.packets_sent();
            if (dec_.input_flushed()) {
                if (!wait_arrival(sched_.arrival(sent + tail))) return;
                ++tail;
            } else {
                output_delay_max_ = std::max(output_delay_max_, static_cast<long>(sent) - frames_);
            }
            const auto t_end = Clock::now();
            {
                std::unique_lock<std::mutex> lk(m_);
                if (mailbox_.size() >= static_cast<std::size_t>(kLiveMailboxFrames)) {
                    ++mailbox_full_waits_;
                    cv_.wait(lk, [this] { return halt_ || mailbox_.size() < static_cast<std::size_t>(kLiveMailboxFrames); });
                    if (halt_) return;
                }
                mailbox_.push_back(Item{std::move(*f), t_end - busy, t_end});
            }
            wake_();
        }
    }

    void run() {
        pthread_setname_np(pthread_self(), "yvo-live-dec");   // the "thread cpu" lines group by this name
        std::exception_ptr failure;
        try {
            decode_loop();
        } catch (...) {
            failure = std::current_exception();
        }
        LiveInputStats s;
        s.packets = static_cast<long>(dec_.packets_sent());
        s.frames = frames_;
        s.output_delay_max = output_delay_max_;
        s.decoder_threads = dec_.thread_count();
        s.reorder_frames = dec_.reorder_frames();
        s.send_late_mean_ms = late_samples_ > 0 ? late_sum_ms_ / static_cast<double>(late_samples_) : 0.0;
        s.send_late_max_ms = late_max_ms_;
        {
            const std::lock_guard<std::mutex> lk(m_);
            s.mailbox_full_waits = mailbox_full_waits_;
            stats_ = s;
            failure_ = failure;
            ended_ = true;
        }
        wake_();
        std::unique_lock<std::mutex> lk(m_);
        parked_ = true;
        parked_cv_.notify_all();
        cv_.wait(lk, [this] { return release_; });
    }

    Decoder& dec_;
    const ArrivalSchedule sched_;
    const std::function<void()> wake_;

    // Decode thread only.
    Clock::duration gate_wait_{};
    std::int64_t last_gated_ = -1;
    double late_sum_ms_ = 0, late_max_ms_ = 0;
    long late_samples_ = 0, frames_ = 0, output_delay_max_ = 0;

    // Guarded by m_. cv_: the decode thread waits (arrival, mailbox space, release). parked_cv_: halt() waits.
    mutable std::mutex m_;
    std::condition_variable cv_, parked_cv_;
    std::deque<Item> mailbox_;
    std::exception_ptr failure_;
    LiveInputStats stats_;
    long mailbox_full_waits_ = 0;
    bool halt_ = false, ended_ = false, parked_ = false, release_ = false;

    std::thread thread_;   // last: everything above exists before the thread starts
};

} // namespace yvo::pipeline
