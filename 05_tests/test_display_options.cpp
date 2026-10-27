// yvo_pipeline display option parsers (display/display_pipeline.hpp, display/vk_swapchain.hpp) — pure string → enum.
//
//   --display-backend      parse_backend_kind (+ is_plane_backend: auto is never a plane path, so yvo_pipeline's
//                          --display on gate for auto is the composition rule — O2's --slots 1 cannot deadlock there)
//   --display-color-range  parse_color_range  (+ color_range_name)
//   --display-sync         parse_display_sync (+ sync_name)
//   --display-pace         parse_display_pace (+ pace_name)
//   --present-mode         parse_present_mode (+ present_mode_name)
//   late frames (planes)   late_gap_threshold (display/plane_display.hpp): vblank gap allowance per presentation pair,
//                          widened by source frames the leaky input queue dropped (--input-pace source)
//   --display-commit-margin-ms  yvo::tool::parse_number<double> (toolkit/cli_parse.hpp, as main.cpp's arg_number)
//                          + commit_margin_valid, and the gate it sets: commit_gate_ms (display/plane_display.hpp)
//
// Needs no GPU or display: the functions are only linked from yvo_display, nothing creates a Vulkan instance or
// connects to Wayland. Built only when yvo_display exists (wayland-client/protocols present).
// Contract checked here: exact lowercase spelling only, and **on failure the output is left untouched**
// (main.cpp parse() relies on that: a rejected value must not overwrite the default).
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "display/display_pipeline.hpp"
#include "display/plane_display.hpp"
#include "display/vk_swapchain.hpp"
#include "toolkit/cli_parse.hpp"

namespace {

using namespace yvo::display;

// Values that no parser accepts: empty, case, whitespace, separators, prefixes, embedded NUL.
const char* const kGarbage[] = {"", " ", "AUTO", "Auto", "auto ", " auto", "on", "off ", "1", "-", "--"};

TEST(DisplayOptions, BackendKindValid) {
    const struct {
        const char* s;
        DisplayBackendKind k;
    } cases[] = {{"auto", DisplayBackendKind::Auto},
                 {"wayland", DisplayBackendKind::Wayland},
                 {"offscreen", DisplayBackendKind::Offscreen},
                 {"wayland-planes", DisplayBackendKind::WaylandPlanes},
                 {"kms", DisplayBackendKind::Kms}};
    for (const auto& c : cases) {
        DisplayBackendKind out = c.k == DisplayBackendKind::Auto ? DisplayBackendKind::Kms : DisplayBackendKind::Auto;
        EXPECT_TRUE(parse_backend_kind(c.s, &out)) << c.s;
        EXPECT_EQ(out, c.k) << c.s;
    }
}

TEST(DisplayOptions, BackendKindInvalidLeavesOutputUntouched) {
    for (const char* s : {"wayland_planes", "waylandplanes", "planes", "drm", "Kms", "wayland-plane", "kms2"}) {
        DisplayBackendKind out = DisplayBackendKind::Offscreen;
        EXPECT_FALSE(parse_backend_kind(s, &out)) << s;
        EXPECT_EQ(out, DisplayBackendKind::Offscreen) << s;
    }
    for (const char* s : kGarbage) {
        DisplayBackendKind out = DisplayBackendKind::Offscreen;
        EXPECT_FALSE(parse_backend_kind(s, &out)) << '"' << s << '"';
        EXPECT_EQ(out, DisplayBackendKind::Offscreen);
    }
    DisplayBackendKind out = DisplayBackendKind::Offscreen;
    EXPECT_FALSE(parse_backend_kind(std::string("kms\0", 4), &out)) << "embedded NUL";
    EXPECT_EQ(out, DisplayBackendKind::Offscreen);
}

TEST(DisplayOptions, PlaneBackendOnlyWhenNamedExplicitly) {
    EXPECT_FALSE(is_plane_backend(DisplayBackendKind::Auto));
    EXPECT_FALSE(is_plane_backend(DisplayBackendKind::Wayland));
    EXPECT_FALSE(is_plane_backend(DisplayBackendKind::Offscreen));
    EXPECT_TRUE(is_plane_backend(DisplayBackendKind::WaylandPlanes));
    EXPECT_TRUE(is_plane_backend(DisplayBackendKind::Kms));
    // The --display-backend default (DisplayPipelineConfig) and the parsed "auto" both stay off the plane paths.
    EXPECT_FALSE(is_plane_backend(DisplayPipelineConfig{}.backend));
    DisplayBackendKind out = DisplayBackendKind::Kms;
    ASSERT_TRUE(parse_backend_kind("auto", &out));
    EXPECT_FALSE(is_plane_backend(out));
}

TEST(DisplayOptions, ColorRangeValidAndRoundTrip) {
    const struct {
        const char* s;
        ColorRangeRequest r;
    } cases[] = {{"auto", ColorRangeRequest::Auto}, {"full", ColorRangeRequest::Full}, {"limited", ColorRangeRequest::Limited}};
    for (const auto& c : cases) {
        ColorRangeRequest out = c.r == ColorRangeRequest::Full ? ColorRangeRequest::Limited : ColorRangeRequest::Full;
        EXPECT_TRUE(parse_color_range(c.s, &out)) << c.s;
        EXPECT_EQ(out, c.r) << c.s;
        EXPECT_STREQ(color_range_name(c.r), c.s);
    }
}

TEST(DisplayOptions, ColorRangeInvalidLeavesOutputUntouched) {
    for (const char* s : {"Full", "limit", "lim", "tv", "pc", "full-range", "limited "}) {
        ColorRangeRequest out = ColorRangeRequest::Limited;
        EXPECT_FALSE(parse_color_range(s, &out)) << s;
        EXPECT_EQ(out, ColorRangeRequest::Limited) << s;
    }
    for (const char* s : kGarbage) {
        ColorRangeRequest out = ColorRangeRequest::Limited;
        EXPECT_FALSE(parse_color_range(s, &out)) << '"' << s << '"';
        EXPECT_EQ(out, ColorRangeRequest::Limited);
    }
}

TEST(DisplayOptions, SyncValidAndRoundTrip) {
    DisplaySync out = DisplaySync::Latest;
    EXPECT_TRUE(parse_display_sync("matched", &out));
    EXPECT_EQ(out, DisplaySync::Matched);
    EXPECT_TRUE(parse_display_sync("latest", &out));
    EXPECT_EQ(out, DisplaySync::Latest);
    for (const DisplaySync s : {DisplaySync::Matched, DisplaySync::Latest}) {
        DisplaySync back = s == DisplaySync::Matched ? DisplaySync::Latest : DisplaySync::Matched;
        EXPECT_TRUE(parse_display_sync(sync_name(s), &back)) << sync_name(s);
        EXPECT_EQ(back, s);
    }
}

TEST(DisplayOptions, SyncInvalidLeavesOutputUntouched) {
    for (const char* s : {"match", "Matched", "newest", "last", "sync"}) {
        DisplaySync out = DisplaySync::Latest;
        EXPECT_FALSE(parse_display_sync(s, &out)) << s;
        EXPECT_EQ(out, DisplaySync::Latest) << s;
    }
    for (const char* s : kGarbage) {
        DisplaySync out = DisplaySync::Latest;
        EXPECT_FALSE(parse_display_sync(s, &out)) << '"' << s << '"';
        EXPECT_EQ(out, DisplaySync::Latest);
    }
}

TEST(DisplayOptions, PaceValidIncludingOffAlias) {
    const struct {
        const char* s;
        DisplayPace p;
    } cases[] = {{"source", DisplayPace::Source},
                 {"asap", DisplayPace::Asap},
                 {"none", DisplayPace::None},
                 {"off", DisplayPace::None}};   // "off" is an alias of none
    for (const auto& c : cases) {
        DisplayPace out = c.p == DisplayPace::Asap ? DisplayPace::Source : DisplayPace::Asap;
        EXPECT_TRUE(parse_display_pace(c.s, &out)) << c.s;
        EXPECT_EQ(out, c.p) << c.s;
    }
}

TEST(DisplayOptions, PaceNameRoundTrip) {
    EXPECT_STREQ(pace_name(DisplayPace::Source), "source");
    EXPECT_STREQ(pace_name(DisplayPace::Asap), "asap");
    EXPECT_STREQ(pace_name(DisplayPace::None), "none");
    for (const DisplayPace p : {DisplayPace::None, DisplayPace::Source, DisplayPace::Asap}) {
        DisplayPace back = p == DisplayPace::Asap ? DisplayPace::Source : DisplayPace::Asap;
        EXPECT_TRUE(parse_display_pace(pace_name(p), &back)) << pace_name(p);
        EXPECT_EQ(back, p);
    }
}

TEST(DisplayOptions, PaceInvalidLeavesOutputUntouched) {
    for (const char* s : {"Source", "src", "fast", "vsync", "30", "none "}) {
        DisplayPace out = DisplayPace::Asap;
        EXPECT_FALSE(parse_display_pace(s, &out)) << s;
        EXPECT_EQ(out, DisplayPace::Asap) << s;
    }
    // kGarbage contains "on" (rejected) but not "off" (accepted alias) — "off " with a trailing space is rejected.
    for (const char* s : kGarbage) {
        DisplayPace out = DisplayPace::Asap;
        EXPECT_FALSE(parse_display_pace(s, &out)) << '"' << s << '"';
        EXPECT_EQ(out, DisplayPace::Asap);
    }
}

TEST(DisplayOptions, PresentModeValid) {
    const struct {
        const char* s;
        VkPresentModeKHR m;
        const char* name;
    } cases[] = {{"fifo", VK_PRESENT_MODE_FIFO_KHR, "FIFO"},
                 {"mailbox", VK_PRESENT_MODE_MAILBOX_KHR, "MAILBOX"},
                 {"immediate", VK_PRESENT_MODE_IMMEDIATE_KHR, "IMMEDIATE"}};
    for (const auto& c : cases) {
        VkPresentModeKHR out = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        EXPECT_TRUE(parse_present_mode(c.s, &out)) << c.s;
        EXPECT_EQ(out, c.m) << c.s;
        EXPECT_STREQ(present_mode_name(c.m), c.name);
    }
}

TEST(DisplayOptions, PresentModeInvalidLeavesOutputUntouched) {
    // fifo_relaxed exists in Vulkan (present_mode_name knows it) but is not accepted as an option.
    EXPECT_STREQ(present_mode_name(VK_PRESENT_MODE_FIFO_RELAXED_KHR), "FIFO_RELAXED");
    for (const char* s : {"fifo_relaxed", "fifo-relaxed", "relaxed", "FIFO", "Mailbox", "vsync", "shared"}) {
        VkPresentModeKHR out = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        EXPECT_FALSE(parse_present_mode(s, &out)) << s;
        EXPECT_EQ(out, VK_PRESENT_MODE_FIFO_RELAXED_KHR) << s;
    }
    for (const char* s : kGarbage) {
        VkPresentModeKHR out = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        EXPECT_FALSE(parse_present_mode(s, &out)) << '"' << s << '"';
        EXPECT_EQ(out, VK_PRESENT_MODE_FIFO_RELAXED_KHR);
    }
}

// ── late_gap_threshold (plane late frames) ──────────────────────────────────────────────────────────────────

TEST(LateGapThreshold, NoUpstreamDropsKeepsTheEveryRule) {
    // max(1, ceil(every × refresh / fps − 0.01)) — the rule ScanoutReport::late_expected_gap always used.
    EXPECT_EQ(late_gap_threshold(1, 0, 60.0, 30.0), 2);
    EXPECT_EQ(late_gap_threshold(1, 0, 59.94, 29.97), 2);
    EXPECT_EQ(late_gap_threshold(1, 0, 60.0, 29.97), 2);   // 2.002 stays 2
    EXPECT_EQ(late_gap_threshold(1, 0, 60.0, 25.0), 3);    // 2.4 → round up
    EXPECT_EQ(late_gap_threshold(2, 0, 60.0, 30.0), 4);    // --display-every 2
    EXPECT_EQ(late_gap_threshold(1, 0, 60.0, 120.0), 1);   // at least one vblank
    EXPECT_EQ(late_gap_threshold(0, 0, 60.0, 30.0), 2);    // every < 1 counts as 1
}

TEST(LateGapThreshold, UpstreamDropsWidenTheAllowance) {
    // One queue-dropped source frame between two presented ones: the first stays up for two source intervals.
    EXPECT_EQ(late_gap_threshold(1, 1, 60.0, 30.0), 4);
    EXPECT_EQ(late_gap_threshold(1, 3, 60.0, 30.0), 8);
    EXPECT_EQ(late_gap_threshold(2, 1, 60.0, 30.0), 6);    // every 2 + 1 dropped = 3 source intervals
    EXPECT_EQ(late_gap_threshold(1, 1, 60.0, 25.0), 5);    // 2 × 2.4 = 4.8 → 5
}

TEST(LateGapThreshold, UnknownRefreshOrFpsIsZero) {
    EXPECT_EQ(late_gap_threshold(1, 0, 0.0, 30.0), 0);
    EXPECT_EQ(late_gap_threshold(1, 0, 60.0, 0.0), 0);
    EXPECT_EQ(late_gap_threshold(1, 2, std::nan(""), 30.0), 0);
}

// ── --display-commit-margin-ms (plane commit gate) ──────────────────────────────────────────────────────────

// main.cpp parse(): arg_number<double> (exit 2 on a malformed number) then commit_margin_valid (usage, exit 2).
bool parse_commit_margin(const char* text, double& out) {
    double v = 0.0;
    if (!yvo::tool::parse_number(text, v) || !commit_margin_valid(v)) return false;
    out = v;
    return true;
}

TEST(CommitMargin, DefaultKeepsTheOldGate) {
    EXPECT_DOUBLE_EQ(DisplayPipelineConfig{}.commit_margin_ms, 8.0);
    // The gate before the option existed: max(1000/fps - 8, 0).
    EXPECT_DOUBLE_EQ(commit_gate_ms(30.0, 8.0), 1000.0 / 30.0 - 8.0);
    EXPECT_DOUBLE_EQ(commit_gate_ms(25.0, 8.0), 32.0);
}

TEST(CommitMargin, GateFollowsTheMargin) {
    EXPECT_DOUBLE_EQ(commit_gate_ms(30.0, 0.0), 1000.0 / 30.0);   // exactly the source interval
    EXPECT_DOUBLE_EQ(commit_gate_ms(30.0, 4.0), 1000.0 / 30.0 - 4.0);
    EXPECT_DOUBLE_EQ(commit_gate_ms(30.0, 12.0), 1000.0 / 30.0 - 12.0);
    EXPECT_DOUBLE_EQ(commit_gate_ms(30.0, 40.0), 0.0);            // >= interval: no gate
    EXPECT_DOUBLE_EQ(commit_gate_ms(0.0, 8.0), 0.0);              // unknown fps: no gate
    EXPECT_DOUBLE_EQ(commit_gate_ms(std::nan(""), 8.0), 0.0);
}

TEST(CommitMargin, ParseAcceptsNonNegativeFiniteUpTo1000) {
    const struct {
        const char* s;
        double v;
    } cases[] = {{"8", 8.0}, {"8.0", 8.0}, {"0", 0.0}, {"4", 4.0}, {"12", 12.0}, {"2.5", 2.5}, {"1000", 1000.0},
                 {"1e1", 10.0}, {"-0", 0.0}};
    for (const auto& c : cases) {
        double out = -1.0;
        EXPECT_TRUE(parse_commit_margin(c.s, out)) << c.s;
        EXPECT_DOUBLE_EQ(out, c.v) << c.s;
    }
}

TEST(CommitMargin, ParseRejectsAndLeavesOutputUntouched) {
    for (const char* s : {"", " ", "-1", "-0.5", "1000.5", "1e9", "abc", "8ms", "8 ", "nan", "inf", "-inf", "0x", "--"}) {
        double out = 7.0;
        EXPECT_FALSE(parse_commit_margin(s, out)) << '"' << s << '"';
        EXPECT_DOUBLE_EQ(out, 7.0) << s;
    }
    EXPECT_FALSE(commit_margin_valid(std::nan("")));
}

} // namespace
