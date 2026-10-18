// DisplaySink: public API of the display layer as seen by the pipeline (pure virtual interface).
//
// The pipeline (main.cpp) holds only this interface. The one implementation is
//   · `DisplayPipeline` — V3D composition (display_convert.comp + overlay_boxes.comp) → Vulkan swapchain on a Wayland window.
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

    // After slot acquisition: additionally holds the keepalive of the frame this slot will use. Now = glass-to-glass start
    // point; t_decoded = decode return (decode-to-screen start point, includes the slot wait). No decode time → pass now.
    virtual void retain(int slot, const DrmFrame& f, Clock::time_point t_decoded) = 0;

    // Optional, right after retain(): the source frame index of the frame retained in this slot. It tags the
    // presentation in present_log(). Not called = -1 (unknown: no tag).
    virtual void set_source_index(int slot, std::int64_t src_index) = 0;

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

    // ── Statistics (Vulkan composition path only — default 0/empty) ──
    virtual std::uint64_t recreates() const { return 0; }
    virtual std::uint32_t image_count() const { return 0; }
    virtual bool uniform_scale() const { return false; }
    virtual bool blit_mode() const { return false; }
    virtual const std::vector<double>& convert_ms() const { return empty_series(); }
    virtual const std::vector<double>& convert_total_ms() const { return empty_series(); }

protected:
    static const std::vector<double>& empty_series() {
        static const std::vector<double> kEmpty;
        return kEmpty;
    }
};

} // namespace yvo::display
