// PlaneDisplay: HVS two-plane scanout display (second implementation of DisplaySink).
//
//   Video: puts the decoder NV12 SAND128 dma-buf **as-is** onto the HVS video plane (0 V3D conversion).
//   Boxes: puts the AB24 LINEAR dma-buf drawn by OverlayPlane on the GPU (overlay_boxes_buf.spv) onto the plane above the video.
//   The HVS does the YUV→RGB conversion and blending during scanout. Plan atomic-munching-kite §2.
//
// Transport (WaylandPlaneScanout = Wayland linux-dmabuf through labwc) sits behind the `PlaneScanout` interface. This class handles the
// transport-independent part — frame lifetime, box ring, commit gate (pacing), statistics, debugfs verification.
//
// ── Frame lifetime ("matched-detach": the pipeline slot and the decoder frame are released separately) ─────────
//   retain(slot)  : holds the keepalive after slot acquisition (glass-to-glass start point; decode-to-screen starts at t_decoded)
//   show(slot)    : retained[slot] → pending (not committed yet): the frame **moves** into an Item, so retained[slot] is empty
//                   afterwards and the caller returns the pipeline slot right away (display_done) for both sync modes.
//                   Draws the boxes (acquire → draw, non-blocking).
//   commit        : pending → committed (the transport holds it for scanout). Identified by the transport token.
//   release       : the transport released that buffer (wl_buffer.release) → the Item drops its keepalive (decoder frame back to the pool).
//   The decoder must not write the next frame into the same buffer **while the HVS is scanning that buffer out**, so the frame is
//   held until release (1~2 vsync after the next commit) — by the Item, not by the slot. Holding the slot instead deadlocks
//   with one slot: the on-screen frame is released only once a newer frame is committed, and that frame needs the slot.
//   Holding limit outside the slots: pending kMaxPending(2) + committed (unreleased) kMaxCommitted(3: 1 on screen + 1 waiting for the
//   next vblank + 1 release delay). pending is a FIFO committed one per vblank. On overflow the **oldest** pending is discarded (drop).
//
// ── Pacing (never blocks forever) ──────────────────────────────────────────────────────────
//   pace(source/asap) sets the frame timetable (pace_deadline, --input-pace asap only). With pace=source the commit interval is also
//   widened to at least source interval - 8 ms (evens out the bunching of matched + batch). On top of that there is one more vblank gate:
//   each commit requests a frame callback (wl_surface.frame), and the next commit is deferred until it arrives.
//   **If the callback does not arrive within kFrameCallbackTimeout, it just commits** (counted in frame_cb_timeouts). A commit is a non-blocking
//   request, so this process does not stall even if the compositor is stalled. pump()/show() never block in any case,
//   and only drain() waits for a bounded time (kDrainTimeoutMs).
//   (The defect of the former Vulkan swapchain path — when the compositor gives no frame callback, FIFO vkQueuePresentKHR waits
//    indefinitely inside Mesa WSI and even SIGTERM does not work — is caused by an unbounded wait inside the driver; this path has no such call.)
//
// ── Boxes ────────────────────────────────────────────────────────────────────────────────────────
//   In show(), OverlayPlane::acquire → draw (non-blocking submit). It commits only after ready() becomes true
//   (does not rely on dma-buf implicit fences). If acquire returns -1 (all 3 ring entries held by GPU/compositor), the box update is
//   skipped and the previous boxes are kept (overlay_skipped). Box wl_buffer.release → OverlayPlane::release.
//
// ── Statistics ────────────────────────────────────────────────────────────────────────────────────────
//   presented       : number of frames the transport reported as presented (wp_presentation.presented, else commit count)
//   display_dropped : skipped by every + pending overflow + rejected by transport + not shown at shutdown
//   overlay_ms      : OverlayPlane's last_overlay_ms + last_clear_ms (V3D timestamp)
//   present_interval_ms / glass_ms : based on presented time (CLOCK_MONOTONIC). glass = retain() → presented, d2s = t_decoded → presented.
//   convert_ms      : none (conversion happens in the HVS) — empty value.
//
// Threads: all on the main thread.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "display/display_config.hpp"
#include "display/display_sink.hpp"
#include "display/drm_state.hpp"
#include "display/overlay_plane.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

// Late-frame threshold of one vblank seq gap (ScanoutReport::late_frames): the largest gap of the normal cadence between
// two consecutive presentations. The frame shown first lasts (every + upstream_missing) source intervals, where
// upstream_missing = source frames between the two that never reached show() (dropped by --input-pace source's leaky
// queue — not the display's fault, so they widen the allowance instead of counting as late). Non-integer ratios
// alternate floor/ceil (60 Hz / 25 fps = 2.4 → 2,2,3,…), so round **up**; −0.01 keeps 60/30, 59.94/29.97 and
// 60/29.97 (2.002) at 2. upstream_missing = 0 gives the rule used before upstream drops existed.
// Returns 0 when refresh_hz or source_fps is unknown (<= 0).
inline int late_gap_threshold(int every, std::uint64_t upstream_missing, double refresh_hz, double source_fps) {
    if (!(refresh_hz > 0.0) || !(source_fps > 0.0)) return 0;
    const double frames = static_cast<double>(std::max(every, 1)) + static_cast<double>(upstream_missing);
    return std::max(1, static_cast<int>(std::ceil(frames * refresh_hz / source_fps - 0.01)));
}

// Transport → PlaneDisplay events. All arrive inside PlaneScanout::dispatch() (main thread).
class PlaneScanoutListener {
public:
    virtual ~PlaneScanoutListener() = default;
    virtual void on_video_released(std::uint64_t token) = 0;          // released the video buffer of that commit
    virtual void on_overlay_released(int overlay_idx) = 0;            // released a box ring entry
    virtual void on_frame_done() = 0;                                 // the next commit may be made (vblank)
    // Presented. t_ns = ns on CLOCK_MONOTONIC (if clock_monotonic=false it is a different clock and not used for glass).
    // seq = vertical retrace counter of that presentation (0 = unavailable), refresh_ns = refresh period (0 = unknown).
    virtual void on_presented(std::uint64_t token, std::int64_t t_ns, bool clock_monotonic, bool zero_copy,
                              std::uint64_t seq, std::uint32_t refresh_ns) = 0;
    virtual void on_discarded(std::uint64_t token) = 0;               // replaced without reaching the screen
};

// Plane scanout transport (implementation: WaylandPlaneScanout). Kept as an interface so PlaneDisplay does not see Wayland types.
class PlaneScanout {
public:
    virtual ~PlaneScanout() = default;
    virtual const char* name() const = 0;                 // backend_name() value
    virtual void set_listener(PlaneScanoutListener* l) = 0;
    virtual VkExtent2D output_extent() const = 0;         // box plane size (= screen size)
    // Once after creating the OverlayPlane: registers the ring entries as transport buffers (wl_buffer / FB).
    virtual void register_overlay(const OverlayPlane& ov) = 0;
    // Commits one frame. If overlay_idx = -1 the box plane is left as-is.
    // false = the transport did not accept this frame (buffer creation failure, busy reuse) — the caller treats it as a drop.
    virtual bool commit(const DrmFrame& video, std::uint64_t token, int overlay_idx) = 0;
    // Event handling. Waits at most timeout_ms (0 = non-blocking). false = connection lost or protocol error.
    virtual bool dispatch(int timeout_ms) = 0;
    virtual bool quit_requested() const = 0;              // quit key
    virtual bool should_close() const = 0;                // window closed / connection lost
    // Shutdown: takes it off the screen and waits up to timeout_ms for remaining releases. No release arrives after return.
    virtual void shutdown(int timeout_ms) = 0;
    // Empties the video buffer cache (before closing the decoder). Call after shutdown.
    virtual void clear_video_cache() = 0;
    virtual std::string describe() const = 0;             // startup line (planes, color request)
    virtual std::uint64_t busy_reuse() const { return 0; }
    // Whether it gives presentation time feedback (on_presented). If false, PlaneDisplay counts commits as presented.
    virtual bool reports_presentation() const = 0;
};

class PlaneDisplay final : public DisplaySink, private PlaneScanoutListener {
public:
    using Clock = DisplaySink::Clock;

    static constexpr int kMaxCommitted = 3;          // upper bound of frames committed but not yet released
    // With matched + batch 2, detections come out two at a time (flush_results calls show back-to-back). With pending 1
    // the second pushed out the first and ~9% were dropped even with source pacing (measured 26 of 300 frames). With 2 they go out
    // in turn on the next two vblanks.
    static constexpr int kMaxPending = 2;
    // Number of frames the display may hold outside the slots (both sync modes; used by main.cpp's decoder holding calculation)
    static constexpr int kExtraHeldBeyondSlots = kMaxCommitted + kMaxPending;
    static constexpr int kOverlayRing = 3;
    static constexpr auto kFrameCallbackTimeout = std::chrono::milliseconds(50);   // ≈ 3 vblank @60 Hz
    static constexpr int kDrainTimeoutMs = 500;
    static constexpr std::uint64_t kSnapshotAfterPresented = 60;   // when to read debugfs after start

    PlaneDisplay(yvo::vk::VulkanContext& ctx, std::unique_ptr<PlaneScanout> scanout, const DisplayConfig& cfg);
    ~PlaneDisplay() override;

    void set_source_size(int w, int h) override;
    void set_source_fps(double fps) override;
    void retain(int slot, const DrmFrame& f, Clock::time_point t_decoded) override;
    void set_source_index(int slot, std::int64_t src_index) override;
    bool show(int slot, const std::vector<Detection>& dets, std::string_view hud = {}) override;
    void pump() override;
    void drain() override;
    bool poll_window() override;
    bool quit_requested() const override;
    int frames_held() const override;
    std::optional<Clock::time_point> pace_deadline(std::int64_t frame_index) override;

    const char* backend_name() const override { return scanout_->name(); }
    std::uint64_t presented() const override { return presented_; }
    std::uint64_t display_dropped() const override { return skipped_ + replaced_ + rejected_ + shutdown_dropped_; }
    std::uint64_t overlay_overflows() const override { return ov_ ? ov_->overlay_overflows() : 0; }
    VkExtent2D extent() const override { return scanout_->output_extent(); }
    const std::vector<double>& overlay_ms() const override { return overlay_ms_; }
    const std::vector<double>& present_interval_ms() const override { return interval_ms_; }
    const std::vector<double>& glass_ms() const override { return glass_ms_; }
    const std::vector<double>& decode_to_screen_ms() const override { return d2s_ms_; }
    const std::vector<PresentRecord>& present_log() const override { return present_log_; }

    std::string describe() const override;
    ScanoutReport scanout_report() const override;

private:
    struct Retained {
        DrmFrame frame;
        Clock::time_point t{}, t_decoded{};
        std::int64_t src = -1;    // set_source_index()
        bool valid = false;
    };
    struct Item {                 // a pending or committed frame
        DrmFrame frame;           // keepalive until the transport releases the buffer (the slot was returned at show())
        int overlay = -1;         // box ring entry to put up with this frame (-1 = keep previous boxes)
        bool overlay_stat = false;
        Clock::time_point t_retain{}, t_decoded{};
        std::int64_t src = -1;
        std::uint64_t missing = 0;   // upstream_missing_ when this frame reached show() (cumulative)
        std::uint64_t token = 0;
    };
    struct AwaitPresent {        // committed, waiting for its presentation feedback
        Clock::time_point t_retain{}, t_decoded{};
        std::int64_t src = -1;
        std::uint64_t missing = 0;
    };

    // PlaneScanoutListener
    void on_video_released(std::uint64_t token) override;
    void on_overlay_released(int overlay_idx) override;
    void on_frame_done() override;
    void on_presented(std::uint64_t token, std::int64_t t_ns, bool clock_monotonic, bool zero_copy, std::uint64_t seq,
                      std::uint32_t refresh_ns) override;
    void on_discarded(std::uint64_t token) override;

    void try_commit(bool force);
    void drop_item(Item& it);          // discards a pending: return boxes + release keepalive
    void finish_item(Item& it);        // released: release keepalive
    bool overlay_ready(Item& it);      // box GPU completion (non-blocking) + stats once
    void snapshot(const char* tag, DrmStateSnapshot* out);

    yvo::vk::VulkanContext& ctx_;
    DisplayConfig cfg_;
    std::unique_ptr<PlaneScanout> scanout_;
    std::unique_ptr<OverlayPlane> ov_;

    std::vector<Retained> retained_;
    // Frames not yet committed (display order, at most kMaxPending). On overflow the oldest is discarded.
    std::deque<Item> pending_;
    std::unordered_map<std::uint64_t, Item> committed_;   // token → committed frame (waiting for release)
    std::unordered_map<std::uint64_t, AwaitPresent> await_present_;   // token → retain/decode time, source index
    std::uint64_t next_token_ = 1;

    bool frame_cb_pending_ = false;
    Clock::duration min_commit_interval_{0};   // pace=source: source interval - 8 ms (half a vblank). 0 = no gate
    Clock::time_point last_commit_{};
    bool overlay_ever_committed_ = false;

    Clock::time_point pace_t0_{};
    bool pace_started_ = false;
    std::int64_t shown_counter_ = 0;
    std::int64_t last_show_src_ = -1;      // source index of the previous show() (-1 = none / unknown)
    std::uint64_t upstream_missing_ = 0;   // source frames skipped between show() calls (never reached the display)

    std::uint64_t presented_ = 0, commits_ = 0;
    std::uint64_t skipped_ = 0, replaced_ = 0, rejected_ = 0, shutdown_dropped_ = 0;
    std::uint64_t overlay_skipped_ = 0, frame_cb_timeouts_ = 0, zero_copy_fb_ = 0, feedback_presented_ = 0;
    std::uint64_t discarded_ = 0;
    std::vector<std::uint32_t> seq_gaps_;   // vblank seq gaps after warmup (late frames are judged in scanout_report)
    std::vector<std::uint64_t> seq_gap_missing_;   // pairs with seq_gaps_: upstream-dropped source frames inside the gap
    std::uint64_t last_chain_missing_ = 0;         // Item::missing of the previous presentation
    std::uint64_t last_seq_ = 0;  std::uint32_t refresh_ns_ = 0;  bool seq_seen_ = false;
    std::uint64_t last_chain_token_ = 0, seq_breaks_ = 0;   // token of the previous presentation / broken seq chains
    std::vector<std::uint64_t> stall_tokens_;   // discarded / rejected tokens: real screen stalls, kept inside a gap
    bool drained_ = false;

    std::int64_t last_present_ns_ = 0;
    std::vector<double> overlay_ms_, interval_ms_, glass_ms_, d2s_ms_;
    std::vector<PresentRecord> present_log_;   // pairs with glass_ms_

    bool snap_start_taken_ = false;
    DrmStateSnapshot snap_start_, snap_end_;
    std::string dri_debug_dir_ = kDefaultDriDebugDir;
};

} // namespace yvo::display
