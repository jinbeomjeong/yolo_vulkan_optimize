// drm_state: reads the active planes from debugfs `/sys/kernel/debug/dri/<n>/state` and judges the **actual scanout placement**.
//
// Why it is needed: the plane scanout backends (wayland-planes, kms) assume that "the HVS reads the decoder NV12 SAND buffer
// directly and blends the box plane on top of it". On Wayland the compositor (labwc/wlroots) can silently fall back to GL composition
// (windowed mode, format rejection …), and in labwc 0.20.1~0.20.2 the color range also goes out inverted from the request (wayland_plane_scanout.hpp
// "Color"). The protocol responses cannot tell us this — reading the kernel atomic state directly is the only way to confirm it.
//
// Origin: 06_tools/probes/hvs_probe/probe_common.hpp:137-216 (parse_active_planes / judge_placement).
// What changed in the move:
//   · modifier·fourcc·zpos are parsed as numbers (instead of string prefix comparison). SAND128 is judged on the modifier with
//     the column-height parameter removed (same rule as fourcc_mod_broadcom_mod).
//   · Instead of "number of RGB planes", we check **whether an AB24 plane is above (larger zpos) the video plane (NV12 SAND)** — labwc
//     puts the cursor on an AR24 plane (topmost zpos), so the number of RGB planes cannot identify the box plane.
//   · Judges COLOR_ENCODING / COLOR_RANGE of the video plane.
//   · Also reads the hvs_underrun counter (cumulative value — look at the start/end difference).
//   · A read failure is not an exception but readable=false ("unverified"). debugfs is root:sudo 0750, so permission may be missing.
//
// This file does not use Wayland·Vulkan·libdrm (it is a string parser). Unit tests: 05_tests/test_overlay.cpp DrmState.*
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace yvo::display {

inline constexpr const char* kDefaultDriDebugDir = "/sys/kernel/debug/dri/1";   // vc4 (axi:gpu)

// One active plane (only planes with a crtc attached)
struct DrmPlaneState {
    int id = 0;                 // plane[83] → 83
    std::string name;           // "plane-2"
    std::string crtc;           // "crtc-2"
    std::string alloc;          // "labwc" / "[fbcon]" / our process name
    std::string format;         // "NV12 little-endian (0x3231564e)"
    std::uint32_t fourcc = 0;   // hex value in parentheses. 0 if absent
    std::string modifier_str;   // "0x700000000000004"
    std::uint64_t modifier = 0;
    std::string size;           // fb size "1920x1080"
    std::string crtc_pos, src_pos;
    int zpos = -1;              // normalized-zpos
    std::string encoding;       // "ITU-R BT.709 YCbCr"
    std::string range;          // "YCbCr limited range"
};

// Extracts only planes with a crtc attached from the state text (fb=0 / crtc=(null) are dropped).
std::vector<DrmPlaneState> parse_drm_active_planes(std::string_view state);

bool is_nv12_sand128(const DrmPlaneState& p);   // NV12 + Broadcom SAND128 (column-height parameter ignored)
bool is_ab24_linear(const DrmPlaneState& p);    // ABGR8888 + LINEAR

// Plane scanout judgement
struct PlaneScanoutVerdict {
    int active_planes = 0;
    int video_plane = -1;       // NV12 SAND plane id (-1 if none)
    int video_zpos = -1;
    int overlay_plane = -1;     // AB24 LINEAR plane id above the video plane (-1 if none)
    int overlay_zpos = -1;
    bool video_on_plane = false;   // HVS reads NV12 SAND directly (zero-copy video)
    bool overlay_above = false;    // box plane is above the video
    bool encoding_bt709 = false;   // video plane COLOR_ENCODING
    bool range_limited = false;    // video plane COLOR_RANGE
    std::string text;              // one-line summary

    // expect_overlay: true if the box plane was committed even once (that plane must be above the video for zero-copy)
    bool zero_copy(bool expect_overlay) const { return video_on_plane && (!expect_overlay || overlay_above); }
    bool color_ok() const { return video_on_plane && encoding_bt709 && range_limited; }
};

PlaneScanoutVerdict judge_plane_scanout(const std::vector<DrmPlaneState>& planes);

std::string describe_planes(const std::vector<DrmPlaneState>& planes);

// Snapshot = one read of state + hvs_underrun
struct DrmStateSnapshot {
    bool readable = false;         // state was read (false = "unverified")
    std::string error;             // reason when readable=false
    std::vector<DrmPlaneState> planes;
    PlaneScanoutVerdict verdict;
    long long hvs_underrun = -1;   // cumulative counter. -1 if unreadable
};

DrmStateSnapshot take_drm_state_snapshot(const std::string& dri_debug_dir = kDefaultDriDebugDir);

// Parser for the hvs_underrun file contents (a single number). -1 on failure.
long long parse_hvs_underrun(std::string_view text);

} // namespace yvo::display
