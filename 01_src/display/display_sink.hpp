// DisplaySink: public API of the display layer as seen by the pipeline (pure virtual interface).
//
// The pipeline (main.cpp) holds only this interface. Two implementations are intended.
//   · `DisplayPipeline` — V3D composition (display_convert.comp + overlay_boxes.comp) → Vulkan swapchain/offscreen.
//   · `PlaneDisplay` (plane_display.hpp) — plane scanout. Puts the decoder NV12 SAND dma-buf on the HVS video plane and the GPU-drawn
//     boxes (OverlayPlane, AB24 dma-buf) on the HVS overlay plane (transport: WaylandPlaneScanout = Wayland linux-dmabuf, KmsScanout = KMS atomic).
//
// `DisplayBackend` (offscreen_backend.hpp) is an interface one level lower that "hands out the VkImage to draw into", so plane
// scanout does not fit it. Hence the polymorphism seam is placed at the `show()` level.
//
// ── Contract (what implementations must honor) ─────────────────────────────────────────────────
//   · All slot arguments are **pipeline slots** (0..num_slots-1). Do not mix them with the implementation's internal display slots.
//   · If `show()` returns false (display drop), the caller sets display_done(slot) immediately.
//     If true, the implementation later calls display_done(slot) via the callback inside `pump()`/`drain()`.
//   · Everything is called from the main thread.
//   · Statistics specific to Vulkan composition (convert_ms, uniform_scale, blit_mode, image_count, recreates …) return
//     0/empty values in the default implementation. Implementations without that concept do not override them.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
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

    // Callback signaling that display is finished and the pipeline slot may be released.
    virtual void set_display_done_callback(std::function<void(int)> cb) = 0;

    // Source frame size. Called after opening the decoder, before the first show().
    virtual void set_source_size(int w, int h) = 0;
    // Source frame rate for pace = Source (decoder avg_frame_rate). Same timing as set_source_size(); <= 0 is ignored.
    virtual void set_source_fps(double fps) = 0;

    // Right after decode: additionally holds the keepalive of the frame this slot will use. Glass-to-glass start point.
    virtual void retain(int slot, const DrmFrame& f) = 0;

    // Presents one frame. false = display drop (caller calls display_done immediately).
    virtual bool show(int slot, const std::vector<Detection>& dets, std::string_view hud = {}) = 0;

    // Non-blocking completion reaping → display_done callback. Called every loop iteration.
    virtual void pump() = 0;

    // Finishes all remaining display work (blocking). Once at shutdown.
    virtual void drain() = 0;

    // Window/device event handling. false = window closed or quit key. Always true if there is no window.
    virtual bool poll_window() = 0;

    // A quit key (Esc, q, Ctrl+C) was pressed.
    virtual bool quit_requested() const = 0;

    // Number of decoder frames this path is currently holding.
    virtual int frames_held() const = 0;

    // Processing deadline for this frame according to pace. Empty optional if there is no pacing.
    virtual std::optional<Clock::time_point> pace_deadline(std::int64_t frame_index) = 0;

    // ── Statistics (common) ──
    virtual const char* backend_name() const = 0;         // display_backend value for the start line and CSV
    virtual std::uint64_t presented() const = 0;
    virtual std::uint64_t display_dropped() const = 0;
    virtual std::uint64_t overlay_overflows() const = 0;
    virtual VkExtent2D extent() const = 0;                // screen (or output) size
    virtual const std::vector<double>& overlay_ms() const = 0;
    virtual const std::vector<double>& present_interval_ms() const = 0;
    virtual const std::vector<double>& glass_ms() const = 0;

    // ── Statistics (Vulkan composition path only — default 0/empty) ──
    virtual std::uint64_t recreates() const { return 0; }
    virtual std::uint32_t image_count() const { return 0; }
    virtual bool offscreen() const { return false; }
    virtual bool uniform_scale() const { return false; }
    virtual bool blit_mode() const { return false; }
    virtual const std::vector<double>& convert_ms() const { return empty_series(); }
    virtual const std::vector<double>& convert_total_ms() const { return empty_series(); }

    // ── Plane scanout only (default: not applicable) ──
    // Description appended after the start line `[display] backend=...`. If empty, the caller prints it in the Vulkan path format.
    virtual std::string describe() const { return {}; }
    struct ScanoutReport {
        bool applicable = false;           // false = not applicable to this path ("-" in CSV)
        std::string zero_copy = "-";       // debugfs judgement: "1" / "0" / "unverified"
        std::string color_ok = "-";        // video plane COLOR_ENCODING BT.709 && COLOR_RANGE limited
        long long underrun_delta = -1;     // hvs_underrun end - start (-1 = unreadable)
        std::uint64_t overlay_skipped = 0; // number of box updates skipped because the box ring was empty (all held by GPU/compositor)
        std::uint64_t frame_cb_timeouts = 0; // number of times waiting for the wl_surface.frame callback timed out and we committed anyway
        std::uint64_t zero_copy_feedback = 0; // number of presentations with the wp_presentation ZERO_COPY flag set
        std::uint64_t feedback_presented = 0; // number of presented feedbacks
        std::uint64_t busy_reuse = 0;      // number of attempts to re-attach a buffer still held by the compositor (must be 0)
        std::vector<std::string> lines;    // judgement lines printed verbatim in the summary (debugfs excerpt)
        std::vector<std::string> transport_lines;   // transport-specific lines (kms: flip_timeouts·master_lost·restore result at exit)
        std::string fatal;                 // unrecoverable transport error ("" = none) — the run is treated as failed
    };
    virtual ScanoutReport scanout_report() const { return {}; }

protected:
    static const std::vector<double>& empty_series() {
        static const std::vector<double> kEmpty;
        return kEmpty;
    }
};

} // namespace yvo::display
