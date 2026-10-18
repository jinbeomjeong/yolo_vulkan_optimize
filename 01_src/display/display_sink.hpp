// DisplaySink: public API of the display layer as seen by the pipeline (pure virtual interface).
//
// The pipeline (main.cpp) holds only this interface. The one implementation is
//   · `PlaneDisplay` (plane_display.hpp) — plane scanout through labwc. Puts the decoder NV12 SAND dma-buf on the HVS video
//     plane and the GPU-drawn boxes (OverlayPlane, AB24 dma-buf) on the HVS overlay plane (transport: WaylandPlaneScanout).
//
// ── Contract (what implementations must honor) ─────────────────────────────────────────────────
//   · All slot arguments are **pipeline slots** (0..num_slots-1).
//   · `show()` takes the frame retained for the slot over (or drops it), so the caller returns the slot right after
//     `show()` whatever it returns (display_done). The implementation keeps the decoder frame alive on its own until the
//     compositor releases the buffer and never calls the display-done callback for a shown frame.
//   · Everything is called from the main thread.
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"

namespace yvo::display {

class DisplaySink {
public:
    using Clock = std::chrono::steady_clock;

    virtual ~DisplaySink() = default;
    DisplaySink() = default;
    DisplaySink(const DisplaySink&) = delete;
    DisplaySink& operator=(const DisplaySink&) = delete;

    // Source frame size. Called after opening the decoder, before the first show().
    virtual void set_source_size(int w, int h) = 0;
    // Source frame rate for pace = Source (decoder avg_frame_rate). Same timing as set_source_size(); <= 0 is ignored.
    virtual void set_source_fps(double fps) = 0;

    // After slot acquisition: additionally holds the keepalive of the frame this slot will use. Now = glass-to-glass start
    // point; t_decoded = decode return (decode-to-screen start point, includes the slot wait). No decode time → pass now.
    virtual void retain(int slot, const DrmFrame& f, Clock::time_point t_decoded) = 0;

    // Optional, right after retain(): the source frame index of the frame retained in this slot. It tags the
    // presentation in present_log(), and it tells source frames that never reached show() (dropped
    // upstream by --input-pace source's leaky queue) apart from late frames. Not called = -1 (unknown: no tag, and the
    // late-frame rule treats the frames as consecutive — the rule before this existed).
    virtual void set_source_index(int slot, std::int64_t src_index) = 0;

    // Presents one frame: takes the frame retained for this slot over. false = display drop (the frame is released).
    // Either way the slot no longer refers to anything inside the display, so the caller returns it right away.
    virtual bool show(int slot, const std::vector<Detection>& dets, std::string_view hud = {}) = 0;

    // Non-blocking: compositor events (releases, frame callbacks, presentation feedback) and the next commit. Called every loop iteration.
    virtual void pump() = 0;

    // Finishes all remaining display work (blocking). Once at shutdown.
    virtual void drain() = 0;

    // Window/device event handling. false = window closed, connection lost or quit key.
    virtual bool poll_window() = 0;

    // A quit key (Esc, q, Ctrl+C) was pressed.
    virtual bool quit_requested() const = 0;

    // Number of decoder frames this path is currently holding.
    virtual int frames_held() const = 0;

    // Processing deadline for this frame according to pace. Empty optional if there is no pacing.
    // yvo_pipeline consults it only with --input-pace asap: paced input already arrives at the source rate.
    virtual std::optional<Clock::time_point> pace_deadline(std::int64_t frame_index) = 0;

    // ── Statistics (common) ──
    virtual const char* backend_name() const = 0;         // display_backend value for the start line and CSV
    virtual std::uint64_t presented() const = 0;
    virtual std::uint64_t display_dropped() const = 0;
    virtual std::uint64_t overlay_overflows() const = 0;
    virtual VkExtent2D extent() const = 0;                // screen (or output) size
    virtual const std::vector<double>& overlay_ms() const = 0;
    virtual const std::vector<double>& present_interval_ms() const = 0;
    virtual const std::vector<double>& glass_ms() const = 0;              // retain() → reaches screen
    virtual const std::vector<double>& decode_to_screen_ms() const = 0;   // t_decoded → same screen time (pairs with glass_ms)
    // One record per glass_ms() sample, same order and same screen time (so the --display-warmup cut applies alike):
    // the set_source_index() value of that frame (-1 = not given) and when it reached the screen.
    struct PresentRecord {
        std::int64_t src = -1;
        Clock::time_point t{};
    };
    virtual const std::vector<PresentRecord>& present_log() const = 0;

    // ── Plane scanout report ──
    // Description appended after the start line `[display] backend=...`.
    virtual std::string describe() const = 0;
    struct ScanoutReport {
        std::string zero_copy = "-";       // debugfs judgement: "1" / "0" / "unverified"
        std::string color_ok = "-";        // video plane COLOR_ENCODING BT.709 && COLOR_RANGE limited
        long long underrun_delta = -1;     // hvs_underrun end - start (-1 = unreadable)
        std::uint64_t overlay_skipped = 0; // number of box updates skipped because the box ring was empty (all held by GPU/compositor)
        std::uint64_t frame_cb_timeouts = 0; // number of times waiting for the wl_surface.frame callback timed out and we committed anyway
        std::uint64_t zero_copy_feedback = 0; // number of presentations with the wp_presentation ZERO_COPY flag set
        std::uint64_t feedback_presented = 0; // number of presented feedbacks
        std::uint64_t busy_reuse = 0;      // number of attempts to re-attach a buffer still held by the compositor (must be 0)
        std::vector<std::string> lines;    // judgement lines printed verbatim in the summary (debugfs excerpt)
        // Late frames from the vblank counter (wp_presentation seq). Gaps touching the first
        // DisplayConfig::warmup presentations are skipped (same rule as the glass / present-interval series).
        std::uint64_t discarded = 0;       // wp_presentation.discarded count (replaced before reaching the screen)
        // seq gaps > the gap's threshold = late_gap_threshold(every, source frames dropped upstream inside that gap, …)
        // (-1 = no seq / refresh / gap left: unverified). Without upstream drops every threshold is late_expected_gap.
        long long late_frames = -1;
        std::uint64_t upstream_missing = 0;   // source frames that never reached show() (set_source_index gaps)
        std::uint64_t seq_gaps = 0, seq_gap_max = 0, seq_breaks = 0;   // evaluated gaps / largest (vblanks) / chain breaks
        int late_expected_gap = 0;         // largest normal gap = max(1, ceil(every*refresh_hz/source_fps - 0.01)); 0 = unknown
        double refresh_hz = 0.0;           // output refresh reported by the transport (0 = unknown)
    };
    virtual ScanoutReport scanout_report() const = 0;
};

} // namespace yvo::display
