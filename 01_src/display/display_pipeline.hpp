// DisplayPipeline: decoder SAND dma-buf + detection list → one HDMI frame.
//
// The `run_convert_display()` loop of `yvo_display_probe --probe convert` moved as-is into a class.
// The per-frame order is **exactly the same** as the probe:
//     acquire(timeout=0) → begin → record_acquire_transition → DisplayConvert::record
//       → OverlayRenderer::set_frame/record → record_present_transition
//       → end_and_submit → present
// Only three things differ.
//   ① Frames come from a **pipeline slot**, not a decode loop (`retain` → `show`).
//   ② Completion is collected by **`pump()` fence polling + callback**, not a blocking `while(ready)`.
//   ③ Without Wayland it falls back to `OffscreenBackend` (`--display-backend offscreen`).
//
// ── Design decision: display_done is raised on the **submit fence**, not the present fence ────────────────
// The decoder frame (SAND dma-buf) may be released **once the GPU has finished reading it**, and that is
// the submit fence `DisplayConvert::finish(d)` waits on. The present fence (VkSwapchainPresentFenceInfoEXT) is
// the later event "the swapchain image may be rewritten", so it would hold the frame 1~2 vsync (16~33 ms) longer.
// Swapchain image reuse safety is checked separately by `VulkanSwapchain::acquire()` with the sync slot fence.
//
// ── Do not mix the numbering schemes of the two kinds of slots ──────────────────────────────────────────────────────
//   · **pipeline slot** (0..cfg.num_slots-1) : GPU/NPU shared slot of main.cpp. Argument of retain/show/callback.
//   · **display slot**  (0..image_count-1)   : `SwapchainFrame::sync_index`. Slot of DisplayConvert/Overlay.
//   The mapping is recorded in `inflight_` by `show()`, and `pump()` maps it back on fence completion and passes it to the callback.
//
// ── Threads ────────────────────────────────────────────────────────────────────────────────────
// Everything runs on the main thread. It calls `vkQueueSubmit`/`vkQueuePresentKHR`, so it must not be called from other threads
// (same reason as the threading discipline in `infer_stage.hpp` — there is only one VkQueue).
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "core/frame_types.hpp"
#include "display/display_convert.hpp"
#include "display/display_sink.hpp"
#include "display/offscreen_backend.hpp"
#include "display/overlay_renderer.hpp"
#include "display/vk_swapchain.hpp"
#include "display/wayland_window.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

enum class DisplayBackendKind {
    Auto,        // try Wayland first, fall back to offscreen on failure
    Wayland,     // strict — throw on failure (no silent fallback)
    Offscreen,
    WaylandPlanes,  // HVS two-plane scanout (PlaneDisplay + WaylandPlaneScanout). DisplayPipeline is not created
    Kms,            // commit the same two planes directly with KMS atomic (PlaneDisplay + KmsScanout). Requires DRM master
};

// COLOR_RANGE requested by plane scanout.
//   wayland-planes: requested from the compositor via wp_color_representation_v1. kms: put as-is into the video plane COLOR_RANGE property.
enum class ColorRangeRequest {
    Auto,      // wayland-planes = Full: labwc 0.20.1~0.20.2 / wlroots 0.20.2+rpt3~+rpt4 emit it **inverted** from the request ("fullhack") → actually limited
               // kms = Limited (no inversion)
    Full,
    Limited,
};

// Which frame to show together with which detections.
enum class DisplaySync {
    Matched,   // show together once that frame's detections are out (default). More latency but boxes match exactly
    Latest,    // show as soon as preprocessing finishes and overlay the **most recent** detections. Minimum display latency
};

enum class DisplayPace {
    None,      // no pacing (display off)
    Source,    // match the source frame rate (default 30 fps) — for latency/drop assessment
    Asap,      // maximum speed — for throughput assessment
};

struct DisplayPipelineConfig {
    DisplayBackendKind backend = DisplayBackendKind::Auto;
    DisplaySync sync = DisplaySync::Matched;
    DisplayPace pace = DisplayPace::Source;

    bool fullscreen = false;
    int width = 1280;                  // --display-size (overridden by the compositor when fullscreen)
    int height = 720;
    int every = 1;                     // show only 1 frame every K frames (1 = all)
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    bool force_blit = false;           // --display-blit

    int overlay_line_width = 2;
    int overlay_text_scale = 2;
    int max_dets = 80;

    int src_w = 1920;                  // decoder source size (overlay affine + 1:1 decision)
    int src_h = 1080;
    double source_fps = 30.0;          // pace = Source target; set_source_fps() overrides it with the decoder fps
    int num_slots = 4;                 // number of pipeline slots (retain array size)

    ColorRangeRequest color_range = ColorRangeRequest::Auto;   // wayland-planes / kms only
    std::string connector;             // kms only (--display-connector). "" = first connected HDMI-A

    std::string title = "yolo_vulkan_optimize";
    std::string app_id = "yvo_pipeline";
    bool verbose = false;
    // --display-warmup: the first N presentations are left out of glass / present-interval / late-frame statistics
    // (kms first commit is a blocking modeset, kms_scanout.hpp B8/B10). Same meaning for every backend. 0 = keep all.
    int warmup = 0;
};

class DisplayPipeline final : public DisplaySink {
public:
    using Clock = DisplaySink::Clock;

    // ⚠ Must be called **before VulkanContext**, because vkGetPhysicalDeviceWaylandPresentationSupportKHR
    //   requires a wl_display* (see the chicken-and-egg note in vulkan_context.hpp).
    //   If backend == Offscreen it does nothing and returns false.
    //   Even if it fails with backend == Wayland it **does not throw** — the caller checks error() and decides.
    static bool open_window(WaylandWindow& win, const DisplayPipelineConfig& cfg);

    // If win == nullptr or ctx.present_enabled()==false, it is built with the offscreen backend.
    DisplayPipeline(yvo::vk::VulkanContext& ctx, WaylandWindow* win, const DisplayPipelineConfig& cfg);
    ~DisplayPipeline() override;
    DisplayPipeline(const DisplayPipeline&) = delete;
    DisplayPipeline& operator=(const DisplayPipeline&) = delete;

    // Signals that display is done and the slot may be released (submit fence based). The argument is a **pipeline slot**.
    void set_display_done_callback(std::function<void(int)> cb) override { done_cb_ = std::move(cb); }

    // Source frame size. Call it **after opening the decoder, before the first show()** (same convention as `InferStage::set_source_size`).
    // Init order is window → context → display → decoder, so it is not yet known at constructor time
    // (`extra_hw_frames` depends on `image_count()`, so the decoder must come last).
    // This decides whether it is 1:1 or uniform scale — get it wrong and it silently takes a 2.6x slower path.
    void set_source_size(int w, int h) override;
    void set_source_fps(double fps) override { if (fps > 0.0) cfg_.source_fps = fps; }   // before the first show()

    // Right after decode: **additionally** holds the keepalive of the frame this slot will use (before main's f.reset()).
    // At the same time it records this moment as the glass-to-glass start point (t_decoded: decode-to-screen start point).
    void retain(int slot, const DrmFrame& f, Clock::time_point t_decoded) override;

    // Puts one frame on screen. false = display drop — the caller must raise display_done(slot) **immediately**.
    bool show(int slot, const std::vector<Detection>& dets, std::string_view hud = {}) override;

    // Fence polling → finish → display_done callback. Non-blocking. Call every iteration.
    void pump() override;

    // Finishes all remaining display work (blocking). Once at shutdown.
    void drain() override;

    // Wayland event handling + schedules window resize. false = window closed or quit key pressed (always true when offscreen).
    // Distinguish the two with quit_requested().
    bool poll_window() override;

    // A quit key (Esc, q, Ctrl+C) was pressed while the display window had focus — the caller treats it as an interrupt like SIGINT.
    bool quit_requested() const override;

    // Number of decoder frames this path is currently holding (retained + DisplayConvert in-flight).
    int frames_held() const override;

    // Processing deadline for this frame according to pace. Empty optional if there is no pacing.
    // The caller runs the pump loop until then (a blocking sleep would starve the NPU and GPU postprocess).
    std::optional<Clock::time_point> pace_deadline(std::int64_t frame_index) override;

    // ── Statistics ──
    const char* backend_name() const override { return offscreen_ ? "offscreen" : "wayland"; }
    std::uint64_t presented() const override { return backend_->presented(); }
    std::uint64_t display_dropped() const override { return backend_->display_dropped() + skipped_ + busy_dropped_; }
    std::uint64_t recreates() const override { return backend_->recreates(); }
    std::uint32_t image_count() const override { return backend_->image_count(); }
    bool offscreen() const override { return offscreen_; }
    bool uniform_scale() const override { return uniform_scale_; }
    bool blit_mode() const override { return backend_->blit_mode(); }
    VkExtent2D extent() const override { return backend_->extent(); }
    std::uint64_t overlay_overflows() const override { return overlay_overflow_; }

    const std::vector<double>& convert_ms() const override { return convert_ms_; }
    const std::vector<double>& convert_total_ms() const override { return convert_total_ms_; }
    const std::vector<double>& overlay_ms() const override { return overlay_ms_; }
    const std::vector<double>& present_interval_ms() const override { return interval_ms_; }
    const std::vector<double>& glass_ms() const override { return glass_ms_; }
    const std::vector<double>& decode_to_screen_ms() const override { return d2s_ms_; }

private:
    struct Pending {              // display slot → pipeline slot + latency measurement info
        int disp = -1;
        int pipe = -1;            // -1 = the caller raises display_done itself (latest mode)
        Clock::time_point t_retain{};
        bool have_retain = false;
    };

    void build_passes();            // builds DisplayConvert/Overlay to match the backend image count and size
    void apply_extent();            // after a resize, adjusts uniform_scale, LUT upper bound, and slot count
    void recreate_if_needed();
    void collect(const Pending& p); // finish + timestamp collection + display_done
    void drain_present_waits(bool blocking);

    yvo::vk::VulkanContext& ctx_;
    DisplayPipelineConfig cfg_;
    WaylandWindow* win_ = nullptr;
    bool offscreen_ = false;
    bool uniform_scale_ = false;

    std::unique_ptr<DisplayBackend> backend_;
    std::unique_ptr<DisplayConvert> conv_;
    std::unique_ptr<OverlayRenderer> ov_;

    std::function<void(int)> done_cb_;

    // Frames held by pipeline slots (consumed by show())
    struct Retained {
        DrmFrame frame;
        Clock::time_point t{}, t_decoded{};
        bool valid = false;
    };
    std::vector<Retained> retained_;

    std::deque<Pending> inflight_;     // display submit order
    struct PresentWait {
        std::uint64_t id = 0;
        Clock::time_point t_retain{}, t_decoded{};
        bool have_retain = false;
    };
    std::deque<PresentWait> present_ids_;
    Clock::time_point last_present_{};
    bool have_last_present_ = false;
    bool pace_blocking_ok_ = true;   // with pace=asap we must not throttle via present_wait

    std::int64_t shown_counter_ = 0;   // --display-every K counter
    std::uint64_t skipped_ = 0;        // number skipped due to every
    std::uint64_t busy_dropped_ = 0;   // number dropped because all display slots were tied up on the GPU
    std::uint64_t overlay_overflow_ = 0;

    Clock::time_point pace_t0_{};
    bool pace_started_ = false;

    std::vector<double> convert_ms_, convert_total_ms_, overlay_ms_, interval_ms_, glass_ms_, d2s_ms_;
};

const char* sync_name(DisplaySync s);
const char* pace_name(DisplayPace p);
bool parse_backend_kind(const std::string& s, DisplayBackendKind* out);
bool parse_color_range(const std::string& s, ColorRangeRequest* out);
const char* color_range_name(ColorRangeRequest r);
bool parse_display_sync(const std::string& s, DisplaySync* out);
bool parse_display_pace(const std::string& s, DisplayPace* out);

} // namespace yvo::display
