// DisplayConfig: settings of the HDMI display path (--display on) and their CLI name helpers.
//
// The only display implementation is PlaneDisplay + WaylandPlaneScanout (plane_display.hpp): the decoder NV12 SAND dma-buf
// goes onto the HVS video plane and the GPU-drawn boxes (AB24 dma-buf) onto the plane above it, through labwc.
// The window is always fullscreen: in windowed mode labwc composites with GL and the planes are not used.
#pragma once

#include <string>

namespace yvo::display {

// COLOR_RANGE requested from the compositor via wp_color_representation_v1.
enum class ColorRangeRequest {
    Auto,      // = Full: labwc 0.20.1~0.20.2 / wlroots 0.20.2+rpt3~+rpt4 emit it **inverted** from the request ("fullhack")
               //   → actually limited, matching the decoder output
    Full,
    Limited,
};

// Which frame to show together with which detections.
// Both return the pipeline slot right after show(): the display keeps the decoder frame alive on its own until the
// compositor releases the buffer (PlaneDisplay "Frame lifetime").
enum class DisplaySync {
    Matched,   // show together once that frame's detections are out (default). More latency but boxes match exactly
    Latest,    // show as soon as preprocessing finishes and overlay the **most recent** detections. Minimum display latency
};

enum class DisplayPace {
    None,      // no pacing (display off)
    Source,    // match the source frame rate (default 30 fps) — for latency/drop assessment
    Asap,      // maximum speed — for throughput assessment
};

struct DisplayConfig {
    DisplaySync sync = DisplaySync::Matched;
    DisplayPace pace = DisplayPace::Source;

    int width = 1280;                  // --display-size: window size before the fullscreen configure (the compositor overrides it)
    int height = 720;
    int every = 1;                     // show only 1 frame every K frames (1 = all)

    int overlay_line_width = 2;
    int overlay_text_scale = 2;
    int max_dets = 80;

    int src_w = 1920;                  // decoder source size (overlay affine)
    int src_h = 1080;
    double source_fps = 30.0;          // pace = Source target; set_source_fps() overrides it with the decoder fps
    int num_slots = 4;                 // number of pipeline slots (retain array size)

    ColorRangeRequest color_range = ColorRangeRequest::Auto;

    std::string title = "yolo_vulkan_optimize";
    std::string app_id = "yvo_pipeline";
    bool verbose = false;
    // --display-warmup: the first N presentations are left out of glass / present-interval / late-frame statistics. 0 = keep all.
    int warmup = 0;
};

const char* sync_name(DisplaySync s);
const char* pace_name(DisplayPace p);
bool parse_color_range(const std::string& s, ColorRangeRequest* out);
const char* color_range_name(ColorRangeRequest r);
bool parse_display_sync(const std::string& s, DisplaySync* out);
bool parse_display_pace(const std::string& s, DisplayPace* out);

} // namespace yvo::display
