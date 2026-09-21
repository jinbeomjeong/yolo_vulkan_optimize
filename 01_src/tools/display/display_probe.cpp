// yvo_display_probe: the visualization probe from plan §6.3.
//
//   context-regression (P6): creates two VulkanContexts, present off / on, and
//     prints the enabled instance extension list · enabled device extension list · selected queue family · timestamp valid bits ·
//     external queue family as **sorted text**.
//     The present=off block must match the pre-change golden state (regression check).
//
//   surface (P4): WaylandWindow → vkCreateWaylandSurfaceKHR →
//     vkGetPhysicalDeviceWaylandPresentationSupportKHR → surface caps · format list ·
//     dump of **minImageCount per present mode** +
//     whether vkGetPhysicalDeviceImageFormatProperties2(R8G8B8A8_UNORM, STORAGE) succeeds.
//     Prints each of the 5 pass criteria as PASS/FAIL. **Does not create a swapchain.**
//
//   present (P5, ★key gate): create swapchain → display_probe.comp writes a known test pattern
//     directly into the storage image → present N frames with FIFO → record vkWaitForPresentKHR intervals.
//     **The pixel check is done by the coordinator with grim -o HDMI-A-1** — the probe states the pattern coordinates and expected RGB on stdout
//     and keeps the screen for --hold-seconds (capture during that time).
//     One frame in the middle is deliberately discarded to also run the vkReleaseSwapchainImagesEXT path under the validation layer.
//
//   overlay (P7): runs overlay_boxes.comp with synthetic detections on OffscreenBackend → readback → compare with CPU reference.
//     ① box border bit-exact ② glyph bit-exact ③ GPU timestamp < 0.7 ms (80 detections) ④ validation errors 0.
//     **No NPU, display or decoder needed at all** (offscreen). Boundary conventions are in 01_src/ref/overlay_reference.hpp.
//
//   convert (P8, ★whether native resolution is feasible): converts the decoder SAND dma-buf to RGBA with display_convert.comp.
//     (a) **offscreen continuous stream** 300-frame timestamp median·p90 — 1920x1080 / 1600x900 / 1280x720.
//         Measured back-to-back without display so that present pacing does not idle the GPU and inflate times via DVFS
//         (same measurement conditions as the 2.13 ms preprocessing number).
//     (b) **pixel accuracy**: bit-exactness and max error vs the CPU reference (sand.hpp de-SAND + cpu_letterbox.hpp color formula).
//     (c) also runs on the **real swapchain path** to produce present statistics and total (including barriers+blit) time.
//
// Exit code: 0 success / 2 environment required by the probe is missing / 1 fatal error
//
// Example run (SSH):
//   XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0
//     build/bin/yvo_display_probe --probe present --display-fullscreen --hold-seconds 5
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include "display/display_convert.hpp"
#include "testkit/offscreen_target.hpp"
#include "display/overlay_renderer.hpp"
#include "display/vk_swapchain.hpp"
#include "display/wayland_window.hpp"
#include "vulkan/dmabuf_export.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

#ifdef YVO_HAVE_DECODER
#include "ref/cpu_letterbox.hpp"
#include "core/dmabuf_map.hpp"
#include "ref/sand.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "testkit/display_convert_check.hpp"
#endif
#include "display_probe_spv.h"

// Shared header used by the overlay (OverlayRenderer·overlay_reference.hpp). The static_asserts below pin the rendering convention
// (row/bit order) and the color formula once more **at compile time**.
#include "core/coco_labels.hpp"
#include "ref/glyph_font_8x16.hpp"
#include "toolkit/stats.hpp"

using yvo::Stats;
using yvo::stats;

static_assert(yvo::class_color(0)[0] == 255 && yvo::class_color(0)[1] == 80 && yvo::class_color(0)[2] == 0,
              "class_color() is RGB (the BGR rule reversed)");
static_assert(yvo::glyph_pixel(yvo::glyph_index('A'), 3, 2), "8x16 glyph convention");

namespace {

using yvo::display::SwapchainFrame;
using yvo::display::VulkanSwapchain;
using yvo::display::OverlayConfig;
using yvo::display::OverlayRenderer;

// Defined in the P7 section (below). P8's --with-overlay uses it first, so it is declared here.
std::vector<yvo::Detection> synth_dets(int n, int w, int h);

// ─────────────────────────────── common ───────────────────────────────

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char* pf(bool ok) { return ok ? "PASS" : "FAIL"; }

struct Args {
    std::string probe = "context-regression";
    std::string video = YVO_DEFAULT_VIDEO;
    int width = 1920, height = 1080;
    bool fullscreen = false;
    bool verbose = false;
    bool validation = false;
    bool sync_validation = false;
    bool blit = false;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    int frames = 0;         // 0 = probe default
    double hold_seconds = 3.0;
    std::uint32_t lx = 3, ly = 85;    // same as the DisplayConvertConfig default (measured optimum from the sweep)
    bool no_display = false; // convert: skip the swapchain section
    bool lean_images = false; // opt-in to per-mode minImageCount (FIFO 3) — the validation layer reports an error
    bool sweep = false;       // convert: workgroup size sweep on the 1920x1080 1:1 path
    int resize_every = 0;     // present: toggle fullscreen every N frames to cause a **window resize during execution**
    int line_width = 2;       // overlay: --overlay-line-width (1080p default 2)
    int text_scale = 2;       // overlay: --overlay-text-scale (1080p default 2, 8x16 → 16x32)
    std::uint32_t overlay_local = 128; // overlay: --overlay-local (same as the OverlayConfig default, measured optimum)
    bool with_overlay = false; // convert: record the overlay too, continuing in the same command buffer (integration check)
    int overlay_dets = 80;     // convert/--with-overlay: number of synthetic detections
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--probe context-regression|surface|present|convert|overlay|export]\n"
                 "          [--display-size WxH | --size WxH] [--display-fullscreen | --fullscreen]\n"
                 "          [--present-mode fifo|mailbox|immediate] [--display-blit] [--lean-images]\n"
                 "          [--frames N] [--hold-seconds N] [--local X Y] [--video PATH] [--no-display] [--sweep]\n"
                 "          [--resize-every N] [--overlay-line-width N] [--overlay-text-scale N] [--overlay-local N]\n"
                 "          [--with-overlay] [--overlay-dets N]\n"
                 "          [--validation] [--sync-validation] [--verbose]\n",
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--probe" && (v = next())) a.probe = v;
        else if ((s == "--size" || s == "--display-size") && (v = next())) {
            if (std::sscanf(v, "%dx%d", &a.width, &a.height) != 2 || a.width < 1 || a.height < 1) return false;
        }
        else if (s == "--fullscreen" || s == "--display-fullscreen") a.fullscreen = true;
        else if (s == "--present-mode" && (v = next())) {
            if (!yvo::display::parse_present_mode(v, &a.present_mode)) return false;
        }
        else if (s == "--display-blit") a.blit = true;
        else if (s == "--lean-images") a.lean_images = true;
        else if (s == "--sweep") a.sweep = true;
        else if (s == "--resize-every" && (v = next())) a.resize_every = std::atoi(v);
        else if (s == "--overlay-line-width" && (v = next())) a.line_width = std::atoi(v);
        else if (s == "--overlay-text-scale" && (v = next())) a.text_scale = std::atoi(v);
        else if (s == "--overlay-local" && (v = next())) a.overlay_local = static_cast<std::uint32_t>(std::atoi(v));
        else if (s == "--with-overlay") a.with_overlay = true;
        else if (s == "--overlay-dets" && (v = next())) a.overlay_dets = std::atoi(v);
        else if (s == "--frames" && (v = next())) a.frames = std::atoi(v);
        else if (s == "--hold-seconds" && (v = next())) a.hold_seconds = std::atof(v);
        else if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--no-display") a.no_display = true;
        else if (s == "--local" && i + 2 < argc) {
            a.lx = static_cast<std::uint32_t>(std::atoi(argv[++i]));
            a.ly = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        }
        else if (s == "--validation") a.validation = true;
        else if (s == "--sync-validation") a.validation = a.sync_validation = true;
        else if (s == "--verbose") a.verbose = true;
        else return false;
    }
    return true;
}

// Opens WaylandWindow + a present-enabled VulkanContext at once. On failure, false instead of nullptr.
struct DisplaySession {
    yvo::display::WaylandWindow win;
    std::unique_ptr<yvo::vk::VulkanContext> ctx;

    bool open(const Args& a, const char* title) {
        yvo::display::WaylandWindowOptions wo;
        wo.width = a.width;
        wo.height = a.height;
        wo.fullscreen = a.fullscreen;
        wo.title = title;
        wo.app_id = "yvo_display_probe";
        wo.verbose = a.verbose;
        if (!win.open(wo)) {
            std::printf("wayland       : failed — %s\n", win.error().c_str());
            return false;
        }
        yvo::vk::VulkanContextOptions co;
        co.enable_validation = a.validation;
        co.validation_sync = a.sync_validation;
        co.verbose = a.verbose;
        co.enable_present = true;
        co.wayland_display = win.native_display();
        co.require_present = true; // the probe does not want a silent fallback
        ctx = std::make_unique<yvo::vk::VulkanContext>(co);
        return true;
    }
};

void print_validation(const yvo::vk::VulkanContext& c) {
    std::printf("vk validation errors %llu warnings %llu (layer %s)\n",
                static_cast<unsigned long long>(c.validation_errors()),
                static_cast<unsigned long long>(c.validation_warnings()),
                c.validation_layer_active() ? "on" : "off");
}

// ─────────────────────────────── P6: context-regression ───────────────────────────────

struct ContextSnapshot {
    bool created = false;
    std::string device;
    std::uint32_t api_major = 0, api_minor = 0, api_patch = 0;
    std::uint32_t driver_version = 0;
    std::uint32_t queue_family = 0;
    bool graphics = false;
    bool timestamps = false;
    float timestamp_period = 0.0f;
    std::uint32_t timestamp_bits = 0;
    bool dmabuf_import = false, dmabuf_export = false, dmabuf_dedicated = false;
    std::string external_qf;
    bool present = false;
    std::string present_reason;
    bool surface_maintenance1 = false, swapchain_maintenance1 = false;
    bool present_id = false, present_wait = false;
    bool validation_layer = false;
    std::uint64_t validation_errors = 0, validation_warnings = 0;
    std::vector<std::string> inst_exts, dev_exts;
};

ContextSnapshot snapshot(const yvo::vk::VulkanContext& c) {
    ContextSnapshot s;
    s.created = true;
    s.device = c.device_name();
    const std::uint32_t api = c.properties().apiVersion;
    s.api_major = VK_API_VERSION_MAJOR(api);
    s.api_minor = VK_API_VERSION_MINOR(api);
    s.api_patch = VK_API_VERSION_PATCH(api);
    s.driver_version = c.properties().driverVersion;
    s.queue_family = c.queue_family();
    s.graphics = c.graphics_supported();
    s.timestamps = c.timestamps_supported();
    s.timestamp_period = c.timestamp_period();
    s.timestamp_bits = c.timestamp_valid_bits();
    s.dmabuf_import = c.dmabuf_import_supported();
    s.dmabuf_export = c.dmabuf_export_supported();
    s.dmabuf_dedicated = c.dmabuf_import_dedicated_only();
    s.external_qf = c.external_queue_family() == VK_QUEUE_FAMILY_FOREIGN_EXT ? "FOREIGN_EXT" : "EXTERNAL";
    s.present = c.present_enabled();
    s.present_reason = c.present_unavailable_reason();
    s.surface_maintenance1 = c.has_surface_maintenance1();
    s.swapchain_maintenance1 = c.has_swapchain_maintenance1();
    s.present_id = c.has_present_id();
    s.present_wait = c.has_present_wait();
    s.validation_layer = c.validation_layer_active();
    s.validation_errors = c.validation_errors();
    s.validation_warnings = c.validation_warnings();
    s.inst_exts = c.enabled_instance_extensions();
    s.dev_exts = c.enabled_device_extensions();
    // Lists are printed **sorted** — so the diff does not shake even if the push_back order in the code changes.
    std::sort(s.inst_exts.begin(), s.inst_exts.end());
    std::sort(s.dev_exts.begin(), s.dev_exts.end());
    return s;
}

void print_list(const char* label, const std::vector<std::string>& v) {
    std::printf("%-14s: %zu\n", label, v.size());
    for (const auto& e : v) std::printf("  %s\n", e.c_str());
}

void print_block(const char* tag, const ContextSnapshot& s) {
    std::printf("-- %s --\n", tag);
    std::printf("device        : %s (api %u.%u.%u, driver 0x%x)\n", s.device.c_str(), s.api_major, s.api_minor,
                s.api_patch, s.driver_version);
    std::printf("queue family  : %u\n", s.queue_family);
    std::printf("timestamps    : %s (period %.3f ns, %u bits)\n", s.timestamps ? "yes" : "no",
                static_cast<double>(s.timestamp_period), s.timestamp_bits);
    std::printf("dma-buf       : import=%d export=%d dedicated_only=%d\n", s.dmabuf_import, s.dmabuf_export,
                s.dmabuf_dedicated);
    std::printf("external qf   : %s\n", s.external_qf.c_str());
    std::printf("validation    : errors=%llu warnings=%llu (layer %s)\n",
                static_cast<unsigned long long>(s.validation_errors),
                static_cast<unsigned long long>(s.validation_warnings), s.validation_layer ? "on" : "off");
    std::printf("graphics qf   : %s\n", s.graphics ? "yes" : "no");
    std::printf("present       : %s\n", s.present ? "enabled" : "disabled");
    if (!s.present && !s.present_reason.empty()) std::printf("present why   : %s\n", s.present_reason.c_str());
    if (s.present)
        std::printf("present feats : surface_maintenance1=%d swapchain_maintenance1=%d present_id=%d "
                    "present_wait=%d\n",
                    s.surface_maintenance1, s.swapchain_maintenance1, s.present_id, s.present_wait);
    print_list("instance exts", s.inst_exts);
    print_list("device exts", s.dev_exts);
    std::printf("\n");
}

void print_added(const char* label, const std::vector<std::string>& off, const std::vector<std::string>& on) {
    std::vector<std::string> added;
    std::set_difference(on.begin(), on.end(), off.begin(), off.end(), std::back_inserter(added));
    std::printf("%s: %zu", label, added.size());
    for (const auto& e : added) std::printf(" %s", e.c_str());
    std::printf("\n");
}

int run_context_regression(const Args& a) {
    // ---- 1) present = off : the code path up to today. The block to compare against the golden ----
    ContextSnapshot off;
    {
        yvo::vk::VulkanContextOptions o;
        o.enable_validation = a.validation;
        o.verbose = a.verbose;
        yvo::vk::VulkanContext ctx(o);
        off = snapshot(ctx);
    }

    // ---- 2) WaylandWindow : created **before** VulkanContext (0 Vulkan symbols) ----
    yvo::display::WaylandWindowOptions wo;
    wo.width = a.width;
    wo.height = a.height;
    wo.fullscreen = a.fullscreen;
    wo.title = "yvo display probe";
    wo.app_id = "yvo_display_probe";
    wo.verbose = a.verbose;
    yvo::display::WaylandWindow win;
    const bool win_ok = win.open(wo);

    std::printf("== probe context-regression ==\n\n");
    print_block("present=off", off);

    if (!win_ok) {
        std::printf("-- present=on --\n");
        std::printf("wayland       : failed — %s\n", win.error().c_str());
        std::printf("=> cannot build the present=on block. This is a situation that needs the offscreen fallback.\n");
        return 2;
    }
    std::printf("-- wayland window --\n");
    std::printf("size          : %dx%d%s\n", win.width(), win.height(), a.fullscreen ? " (fullscreen)" : "");
    std::printf("configured    : %s\n", win.configured() ? "yes" : "no");
    std::printf("globals       : subcompositor=%d\n", win.has_subcompositor());
    std::printf("native        : wl_display=%p wl_surface=%p\n", win.native_display(), win.native_surface());
    std::printf("\n");

    // ---- 3) present = on : pass wl_display* as an opaque pointer. No VkSurfaceKHR is created ----
    ContextSnapshot on;
    try {
        yvo::vk::VulkanContextOptions o;
        o.enable_validation = a.validation;
        o.verbose = a.verbose;
        o.enable_present = true;
        o.wayland_display = win.native_display();
        o.require_present = true;
        yvo::vk::VulkanContext ctx(o);
        on = snapshot(ctx);
    } catch (const std::exception& e) {
        std::printf("-- present=on --\n");
        std::printf("vulkan        : failed — %s\n", e.what());
        return 2;
    }
    print_block("present=on", on);

    std::printf("-- diff (off -> on) --\n");
    print_added("added instance exts", off.inst_exts, on.inst_exts);
    print_added("added device exts  ", off.dev_exts, on.dev_exts);
    std::printf("queue family       : %u -> %u %s\n", off.queue_family, on.queue_family,
                off.queue_family == on.queue_family ? "(same)" : "(★changed)");
    std::printf("timestamp bits     : %u -> %u %s\n", off.timestamp_bits, on.timestamp_bits,
                off.timestamp_bits == on.timestamp_bits ? "(same)" : "(★changed)");
    std::printf("external qf        : %s -> %s %s\n", off.external_qf.c_str(), on.external_qf.c_str(),
                off.external_qf == on.external_qf ? "(same)" : "(★changed)");
    std::printf("dma-buf            : import=%d/%d export=%d/%d dedicated_only=%d/%d %s\n", off.dmabuf_import,
                on.dmabuf_import, off.dmabuf_export, on.dmabuf_export, off.dmabuf_dedicated, on.dmabuf_dedicated,
                (off.dmabuf_import == on.dmabuf_import && off.dmabuf_export == on.dmabuf_export &&
                 off.dmabuf_dedicated == on.dmabuf_dedicated)
                    ? "(same)"
                    : "(★changed)");

    const bool polled = win.poll();
    std::printf("poll()             : %s, should_close=%d\n", polled ? "ok" : win.error().c_str(),
                win.should_close());
    return 0;
}

// ─────────────────────────────── P4: surface ───────────────────────────────

void dump_surface(const yvo::display::SurfaceInfo& s) {
    using namespace yvo::display;
    std::printf("presentation support (family %u) : %s\n", s.queue_family,
                s.wayland_presentation_support ? "VK_TRUE" : "VK_FALSE");
    std::printf("vkGetPhysicalDeviceSurfaceSupportKHR: %s\n", s.surface_support ? "VK_TRUE" : "VK_FALSE");
    std::printf("currentExtent   : %ux%u%s\n", s.caps.currentExtent.width, s.caps.currentExtent.height,
                s.caps.currentExtent.width == 0xFFFFFFFFu ? "  (0xFFFFFFFF = we decide)" : "");
    std::printf("image extent    : min %ux%u  max %ux%u\n", s.caps.minImageExtent.width, s.caps.minImageExtent.height,
                s.caps.maxImageExtent.width, s.caps.maxImageExtent.height);
    std::printf("image count     : min %u max %u (default caps)\n", s.caps.minImageCount, s.caps.maxImageCount);
    std::printf("supportedUsage  : 0x%x  STORAGE=%d COLOR_ATTACHMENT=%d TRANSFER_DST=%d\n",
                s.caps.supportedUsageFlags, (s.caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) != 0,
                (s.caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
                (s.caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0);
    std::printf("formats         : %zu\n", s.formats.size());
    for (const auto& f : s.formats)
        std::printf("  %-26s %s%s\n", format_name(f.format), color_space_name(f.colorSpace),
                    (f.format == VK_FORMAT_R8G8B8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                        ? "   <-- the format we use"
                        : "");
    std::printf("present modes   : %zu\n", s.present_modes.size());
    for (VkPresentModeKHR m : s.present_modes) std::printf("  %s\n", present_mode_name(m));
    std::printf("per-mode minImageCount (VK_EXT_surface_maintenance1): %zu\n", s.mode_counts.size());
    for (const auto& c : s.mode_counts)
        std::printf("  %-12s min=%u max=%u\n", present_mode_name(c.mode), c.min_image_count, c.max_image_count);
    std::printf("ImageFormatProperties2(R8G8B8A8_UNORM, OPTIMAL, STORAGE|COLOR_ATTACHMENT|TRANSFER_DST): %s\n",
                yvo::vk::result_str(s.storage_image_result));
    if (s.storage_image_result == VK_SUCCESS)
        std::printf("  maxExtent %ux%u maxMipLevels %u maxArrayLayers %u\n", s.storage_image_props.maxExtent.width,
                    s.storage_image_props.maxExtent.height, s.storage_image_props.maxMipLevels,
                    s.storage_image_props.maxArrayLayers);
    std::printf("blit(R8G8B8A8_UNORM optimal)     : src=%d dst=%d\n", s.blit_src_supported, s.blit_dst_supported);
}

int run_surface(const Args& a) {
    std::printf("== probe surface (P4) ==\n\n");
    DisplaySession ses;
    if (!ses.open(a, "yvo display probe: surface")) return 2;
    std::printf("window          : %dx%d%s configured=%d\n\n", ses.win.width(), ses.win.height(),
                a.fullscreen ? " (fullscreen)" : "", ses.win.configured());

    yvo::display::SwapchainOptions so;
    so.wayland_display = ses.win.native_display();
    so.wayland_surface = ses.win.native_surface();
    so.width = ses.win.width();
    so.height = ses.win.height();
    so.present_mode = a.present_mode;
    so.force_blit = a.blit;
    so.lean_images = a.lean_images;
    so.verbose = a.verbose;
    VulkanSwapchain sc(*ses.ctx, so);   // creates only the surface
    const yvo::display::SurfaceInfo& s = sc.surface_info();
    dump_surface(s);

    std::printf("\n-- P4 pass criteria --\n");
    const bool c1 = s.wayland_presentation_support == VK_TRUE && s.surface_support == VK_TRUE;
    const bool c2 = s.format_available;
    const bool c3 = s.usage_storage;
    const bool c4 = s.fifo_available;
    const bool c5 = s.storage_image_result == VK_SUCCESS;
    std::printf("① family %u presentation support VK_TRUE      : %s\n", s.queue_family, pf(c1));
    std::printf("② R8G8B8A8_UNORM + SRGB_NONLINEAR available   : %s\n", pf(c2));
    std::printf("③ STORAGE in supportedUsageFlags              : %s\n", pf(c3));
    std::printf("④ FIFO present mode available                 : %s\n", pf(c4));
    std::printf("⑤ ImageFormatProperties2(UNORM, STORAGE) ok   : %s\n", pf(c5));
    std::printf("=> direct compute write %s (blit fallback %s)\n", (c3 && c5) ? "possible" : "not possible",
                (c3 && c5) ? "not needed" : "needed");
    print_validation(*ses.ctx);
    return (c1 && c2 && c3 && c4 && c5) ? 0 : 1;
}

// ─────────────────────────────── P5: present ───────────────────────────────

// Minimal pipeline dedicated to display_probe.comp (1 storage image + 12 B push constant)
struct ProbePattern {
    struct Push {
        std::int32_t width;
        std::int32_t height;
        std::uint32_t frame;
    };
    yvo::vk::VulkanContext& ctx;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets;
    std::vector<VkCommandBuffer> cmds;

    ProbePattern(yvo::vk::VulkanContext& c, std::uint32_t n) : ctx(c) {
        VkDevice dev = ctx.device();
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dl{};
        dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dl.bindingCount = 1;
        dl.pBindings = &b;
        YVO_VK_CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &set_layout));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set_layout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        YVO_VK_CHECK(vkCreatePipelineLayout(dev, &pl, nullptr, &layout));

        VkShaderModuleCreateInfo smi{};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = sizeof(kDisplayProbeSpv);
        smi.pCode = kDisplayProbeSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        YVO_VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &mod));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = layout;
        const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline);
        vkDestroyShaderModule(dev, mod, nullptr);
        YVO_VK_CHECK(r);

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, n};
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = n;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        YVO_VK_CHECK(vkCreateDescriptorPool(dev, &dp, nullptr, &pool));
        VkCommandPoolCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = ctx.queue_family();
        YVO_VK_CHECK(vkCreateCommandPool(dev, &cp, nullptr, &cmd_pool));
        sets.resize(n);
        cmds.resize(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            VkDescriptorSetAllocateInfo dai{};
            dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dai.descriptorPool = pool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts = &set_layout;
            YVO_VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &sets[i]));
            VkCommandBufferAllocateInfo cai{};
            cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cai.commandPool = cmd_pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            YVO_VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cmds[i]));
        }
    }
    ~ProbePattern() {
        VkDevice dev = ctx.device();
        if (dev == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(dev);
        if (cmd_pool) vkDestroyCommandPool(dev, cmd_pool, nullptr);
        if (pool) vkDestroyDescriptorPool(dev, pool, nullptr);
        if (pipeline) vkDestroyPipeline(dev, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(dev, layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
    }
    ProbePattern(const ProbePattern&) = delete;
    ProbePattern& operator=(const ProbePattern&) = delete;

    // Record the whole command buffer, including the swapchain barriers, and submit it.
    void draw(const VulkanSwapchain& sc, const SwapchainFrame& f, std::uint32_t frame_no) {
        VkDevice dev = ctx.device();
        const std::uint32_t i = f.sync_index;
        YVO_VK_CHECK(vkResetCommandBuffer(cmds[i], 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        YVO_VK_CHECK(vkBeginCommandBuffer(cmds[i], &bi));
        sc.record_acquire_transition(cmds[i], f);

        VkDescriptorImageInfo ii{};
        ii.imageView = f.target_view;
        ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets[i];
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

        vkCmdBindPipeline(cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[i], 0, nullptr);
        const Push pc{static_cast<std::int32_t>(f.extent.width), static_cast<std::int32_t>(f.extent.height),
                      frame_no};
        vkCmdPushConstants(cmds[i], layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmds[i], (f.extent.width + 15) / 16, (f.extent.height + 15) / 16, 1);

        sc.record_present_transition(cmds[i], f);
        YVO_VK_CHECK(vkEndCommandBuffer(cmds[i]));

        VkCommandBufferSubmitInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cbi.commandBuffer = cmds[i];
        VkSemaphoreSubmitInfo wi{}, so{};
        wi.sType = so.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        wi.semaphore = f.wait_semaphore;
        wi.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        so.semaphore = f.signal_semaphore;
        so.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.waitSemaphoreInfoCount = 1;
        si.pWaitSemaphoreInfos = &wi;
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos = &so;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        YVO_VK_CHECK(vkQueueSubmit2(ctx.queue(), 1, &si, f.submit_fence));
    }
};

void print_expected_pattern(int w, int h) {
    std::printf("-- expected pattern (coordinates and RGB to check with grim -o HDMI-A-1) --\n");
    std::printf("surface size            : %dx%d\n", w, h);
    std::printf("background              : (32,32,32)\n");
    std::printf("center block  x[%d,%d) y[%d,%d) : RGB(255,0,128)   center pixel (%d,%d)\n", (w - 400) / 2,
                (w - 400) / 2 + 400, (h - 300) / 2, (h - 300) / 2 + 300, w / 2, h / 2);
    std::printf("corner TL 64x64 (0,0)         : RGB(255,0,0)     sample (16,16)\n");
    std::printf("corner TR 64x64 (%d,0)      : RGB(0,255,0)     sample (%d,16)\n", w - 64, w - 16);
    std::printf("corner BL 64x64 (0,%d)      : RGB(0,0,255)     sample (16,%d)\n", h - 64, h - 16);
    std::printf("corner BR 64x64 (%d,%d)   : RGB(255,255,255) sample (%d,%d)\n", w - 64, h - 64, w - 16, h - 16);
    std::printf("moving bar y[%d,%d)          : RGB(255,255,0)   (proof of liveness; position differs every frame)\n",
                (h * 3) / 4, (h * 3) / 4 + 24);
    std::fflush(stdout);
}

int run_present(const Args& a) {
    const int frames = a.frames > 0 ? a.frames : 120;
    std::printf("== probe present (P5) ==\n\n");
    DisplaySession ses;
    if (!ses.open(a, "yvo display probe: present")) return 2;

    yvo::display::SwapchainOptions so;
    so.wayland_display = ses.win.native_display();
    so.wayland_surface = ses.win.native_surface();
    so.width = ses.win.width();
    so.height = ses.win.height();
    so.present_mode = a.present_mode;
    so.force_blit = a.blit;
    so.lean_images = a.lean_images;
    so.verbose = true;
    VulkanSwapchain sc(*ses.ctx, so);
    sc.create(ses.win.width(), ses.win.height());
    std::printf("vkCreateSwapchainKHR    : VK_SUCCESS\n");
    std::printf("swapchain               : %ux%u images=%u mode=%s path=%s\n", sc.extent().width,
                sc.extent().height, sc.image_count(), yvo::display::present_mode_name(a.present_mode),
                sc.blit_mode() ? "blit fallback" : "compute -> storage image");
    std::printf("present_wait            : %s\n", sc.present_wait_available() ? "used" : "unavailable (not measurable)");
    print_expected_pattern(static_cast<int>(sc.extent().width), static_cast<int>(sc.extent().height));

    ProbePattern pat(*ses.ctx, sc.image_count());
    std::deque<std::pair<std::uint64_t, double>> pending; // (present_id, present call time)
    std::vector<double> intervals, latencies;
    double last_shown = -1;
    std::uint64_t timeouts = 0, discarded = 0;
    std::uint32_t frame_no = 0;
    const std::size_t depth = std::max<std::size_t>(sc.image_count() - 1, 1);

    auto drain = [&](bool all) {
        while (!pending.empty() && (all || pending.size() >= depth)) {
            const auto [id, t_submit] = pending.front();
            pending.pop_front();
            if (!sc.present_wait_available()) continue;
            if (!sc.wait_present(id, 1000ull * 1000 * 1000)) { // 1 s
                ++timeouts;
                continue;
            }
            const double t = now_ms();
            latencies.push_back(t - t_submit);
            if (last_shown >= 0) intervals.push_back(t - last_shown);
            last_shown = t;
        }
    };

    const double t_start = now_ms();
    int presented = 0;
    int guard = 0;
    int last_toggle = -1;
    bool fs_state = a.fullscreen;
    while (presented < frames && guard < frames * 200) {
        ++guard;
        if (!ses.win.poll() || ses.win.should_close()) break;
        int nw = 0, nh = 0;
        if (ses.win.take_pending_size(&nw, &nh)) sc.request_recreate(nw, nh);
        if (sc.needs_recreate()) {
            drain(true);
            sc.create(ses.win.width(), ses.win.height());
            last_shown = -1;
            print_expected_pattern(static_cast<int>(sc.extent().width), static_cast<int>(sc.extent().height));
        }
        // Cause a **window resize during execution**: toggling fullscreen makes the compositor send a configure with the new size.
        // Repeatedly exercises swapchain recreation + semaphore lifetime + old swapchain retirement paths under the validation layer and ASan.
        if (a.resize_every > 0 && presented > 0 && presented % a.resize_every == 0 && presented != last_toggle) {
            last_toggle = presented;
            fs_state = !fs_state;
            ses.win.set_fullscreen(fs_state);
        }
        SwapchainFrame f;
        if (!sc.acquire(&f)) continue;
        // One frame is deliberately not drawn and returned — runs the vkReleaseSwapchainImagesEXT path under the validation layer.
        if (presented == 7 && discarded == 0) {
            sc.discard(f);
            ++discarded;
            continue;
        }
        pat.draw(sc, f, frame_no++);
        const double t_submit = now_ms();
        sc.present(f);
        if (f.present_id) pending.emplace_back(f.present_id, t_submit);
        ++presented;
        drain(false);
    }
    drain(true);
    const double wall = now_ms() - t_start;

    std::printf("\n-- result --\n");
    std::printf("presented               : %llu (requested %d), loop %.0f ms → %.1f fps\n",
                static_cast<unsigned long long>(sc.presented()), frames, wall,
                static_cast<double>(sc.presented()) * 1000.0 / wall);
    std::printf("display_dropped         : %llu (acquire NOT_READY %llu, in-flight shortage %llu, discard %llu)\n",
                static_cast<unsigned long long>(sc.display_dropped()),
                static_cast<unsigned long long>(sc.acquire_not_ready()),
                static_cast<unsigned long long>(sc.sync_busy()), static_cast<unsigned long long>(discarded));
    std::printf("swapchain_recreates     : %llu, vkReleaseSwapchainImagesEXT released %llu (pending %zu)\n",
                static_cast<unsigned long long>(sc.swapchain_recreates()),
                static_cast<unsigned long long>(sc.images_released()), sc.pending_releases());
    std::printf("present_wait timeouts   : %llu\n", static_cast<unsigned long long>(timeouts));
    const Stats iv = stats(intervals), la = stats(latencies);
    std::printf("present_wait interval ms: median=%.3f p90=%.3f mean=%.3f min=%.3f max=%.3f (n=%zu)\n", iv.median,
                iv.p90, iv.mean, iv.min, iv.max, iv.n);
    std::printf("present→screen delay ms : median=%.3f p90=%.3f max=%.3f (n=%zu)\n", la.median, la.p90, la.max, la.n);

    std::printf("\n-- P5 pass criteria --\n");
    const bool c1 = sc.created();
    const bool c2 = ses.ctx->validation_errors() == 0;
    const bool c3 = static_cast<int>(sc.presented()) >= frames && timeouts == 0;
    const bool c4 = iv.n > 0 && iv.median >= 14.7 && iv.median <= 18.7;
    std::printf("① vkCreateSwapchainKHR VK_SUCCESS             : %s\n", pf(c1));
    std::printf("② validation layer errors 0                   : %s (errors=%llu)\n", pf(c2),
                static_cast<unsigned long long>(ses.ctx->validation_errors()));
    std::printf("③ %d presents completed, 0 timeouts          : %s\n", frames, pf(c3));
    std::printf("④ present_wait interval median 16.7 ms ±2     : %s (%.3f ms)\n", pf(c4), iv.median);
    std::printf("⑤ grim pixel check                            : done by coordinator (capture during the HOLD below)\n");
    print_validation(*ses.ctx);

    // ── Keep the screen: the interval in which the coordinator captures with grim ──
    if (a.hold_seconds > 0) {
        std::printf("\nHOLD START %.1f s — capture now with grim -o HDMI-A-1\n", a.hold_seconds);
        std::fflush(stdout);
        const double t0 = now_ms();
        while (now_ms() - t0 < a.hold_seconds * 1000.0) {
            if (!ses.win.poll() || ses.win.should_close()) break;
            int nw = 0, nh = 0;
            if (ses.win.take_pending_size(&nw, &nh)) sc.request_recreate(nw, nh);
            if (sc.needs_recreate()) {
                drain(true);
                sc.create(ses.win.width(), ses.win.height());
            }
            SwapchainFrame f;
            if (!sc.acquire(&f)) continue;
            pat.draw(sc, f, frame_no++);
            sc.present(f);
            if (f.present_id) pending.emplace_back(f.present_id, now_ms());
            drain(false);
        }
        drain(true);
        std::printf("HOLD END\n");
        std::fflush(stdout);
    }
    print_validation(*ses.ctx);
    return (c1 && c2 && c3 && c4) ? 0 : 1;
}

// The offscreen render target, barriers and readback live only in testkit/offscreen_target.hpp (shared with test_overlay).
using yvo::display::barrier_image;
using yvo::display::OffscreenTarget;

// ─────────────────────────────── P8: convert ───────────────────────────────

#ifdef YVO_HAVE_DECODER

// Offscreen continuous stream measurement. There is no present pacing, so the GPU stays busy (minimal DVFS effect).
struct ConvertRun {
    Stats shader, total;
    long frames = 0;
    double wall_ms = 0;
};

ConvertRun run_offscreen(yvo::vk::VulkanContext& ctx, const Args& a, int dst_w, int dst_h, int frames, int depth) {
    ConvertRun out;
    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.extra_hw_frames = std::max(2 * yvo::kNumSlots, depth + 4);
    if (!dec.open(a.video, dopt)) throw std::runtime_error("decoder open failed: " + dec.error());

    yvo::display::DisplayConvertConfig cc;
    cc.num_slots = depth;
    cc.local_size_x = a.lx;
    cc.local_size_y = a.ly;
    cc.uniform_scale = !(dst_w == dec.width() && dst_h == dec.height() && (dec.width() % 4) == 0);
    cc.max_dst_w = std::max(dst_w, 1920);
    yvo::display::DisplayConvert conv(ctx, cc);

    std::vector<std::unique_ptr<OffscreenTarget>> tgt;
    for (int i = 0; i < depth; ++i)
        tgt.push_back(std::make_unique<OffscreenTarget>(ctx, static_cast<std::uint32_t>(dst_w),
                                                        static_cast<std::uint32_t>(dst_h), false));

    std::vector<double> sh, to;
    std::deque<int> inflight;
    auto finish = [&](int slot) {
        conv.finish(slot);
        if (conv.last_shader_ms() >= 0) sh.push_back(conv.last_shader_ms());
        if (conv.last_total_ms() >= 0) to.push_back(conv.last_total_ms());
    };

    const double t0 = now_ms();
    long n = 0;
    while (n < frames) {
        auto f = dec.next();
        if (!f) break;
        const int slot = static_cast<int>(n % depth);
        while (!inflight.empty() && (static_cast<int>(inflight.size()) >= depth || conv.pending(slot))) {
            finish(inflight.front());
            inflight.pop_front();
        }
        OffscreenTarget& t = *tgt[static_cast<std::size_t>(slot)];
        YVO_VK_CHECK(vkResetFences(ctx.device(), 1, &t.fence));
        VkCommandBuffer cb = conv.begin(slot);
        barrier_image(cb, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                      VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        conv.record(slot, *f, t.view, {t.w, t.h});
        conv.end_and_submit(slot, VK_NULL_HANDLE, VK_NULL_HANDLE, t.fence);
        inflight.push_back(slot);
        ++n;
    }
    while (!inflight.empty()) {
        finish(inflight.front());
        inflight.pop_front();
    }
    conv.clear_import_cache();
    out.wall_ms = now_ms() - t0;
    out.frames = n;
    out.shader = stats(sh);
    out.total = stats(to);
    return out;
}

// Pixel accuracy: compares against the CPU reference (sand.hpp de-SAND + cpu_letterbox.hpp color formula, scale=1).
// P8 accuracy: the check lives only in testkit/display_convert_check.hpp (shared with test_display_convert).
using AccuracyResult = yvo::display::ConvertAccuracy;

AccuracyResult run_accuracy(yvo::vk::VulkanContext& ctx, const Args& a, int frame_index) {
    yvo::FfmpegDrmDecoder dec;
    if (!dec.open(a.video)) throw std::runtime_error("decoder open failed: " + dec.error());
    std::optional<yvo::DrmFrame> f;
    for (int i = 0; i <= frame_index; ++i) {
        f = dec.next();
        if (!f) break;
    }
    if (!f) throw std::runtime_error("accuracy: failed to get a frame");
    return yvo::display::check_display_convert_1to1(ctx, *f, a.lx, a.ly);
}

// Real swapchain path (convert while displaying). Produces present statistics and total (including barriers+blit) time.
int run_convert_display(const Args& a, DisplaySession& ses, int frames) {
    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.extra_hw_frames = 2 * yvo::kNumSlots;
    if (!dec.open(a.video, dopt)) {
        std::printf("decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }
    yvo::display::SwapchainOptions so;
    so.wayland_display = ses.win.native_display();
    so.wayland_surface = ses.win.native_surface();
    so.width = ses.win.width();
    so.height = ses.win.height();
    so.present_mode = a.present_mode;
    so.force_blit = a.blit;
    so.lean_images = a.lean_images;
    so.verbose = true;
    VulkanSwapchain sc(*ses.ctx, so);
    sc.create(ses.win.width(), ses.win.height());

    yvo::display::DisplayConvertConfig cc;
    cc.num_slots = static_cast<int>(sc.image_count());
    cc.local_size_x = a.lx;
    cc.local_size_y = a.ly;
    cc.uniform_scale = !(static_cast<int>(sc.extent().width) == dec.width() &&
                         static_cast<int>(sc.extent().height) == dec.height() && (dec.width() % 4) == 0);
    cc.max_dst_w = std::max<int>(static_cast<int>(sc.extent().width), 1920);
    yvo::display::DisplayConvert conv(*ses.ctx, cc);
    std::printf("display path            : %ux%u %s / %s, images=%u\n", sc.extent().width, sc.extent().height,
                cc.uniform_scale ? "uniform-scale(bilinear)" : "1:1",
                sc.blit_mode() ? "blit fallback" : "compute -> swapchain storage image", sc.image_count());

    // --with-overlay: records the overlay **continuing in the same command buffer that DisplayConvert created**.
    // It has the same structure as the real integration (plan §5.4), so this path is what the validation layer (including sync) judges.
    std::unique_ptr<OverlayRenderer> ov;
    std::vector<yvo::Detection> ovdets;
    if (a.with_overlay) {
        OverlayConfig oc;
        oc.num_slots = static_cast<int>(sc.image_count());
        oc.local_size_x = a.overlay_local;
        oc.line_width = a.line_width;
        oc.text_scale = a.text_scale;
        ov = std::make_unique<OverlayRenderer>(*ses.ctx, oc);
        ovdets = synth_dets(a.overlay_dets, dec.width(), dec.height());
        std::printf("overlay                 : %d detections, line_width %d, text_scale %d, local_size_x %u\n",
                    a.overlay_dets, oc.line_width, oc.text_scale, oc.local_size_x);
    }

    std::vector<double> sh, to, iv, ovms;
    std::deque<int> inflight;
    std::deque<std::uint64_t> pending_ids;
    double last_shown = -1;
    long decoded = 0, drawn = 0;
    auto take_stats = [&](int s) {
        conv.finish(s);
        if (conv.last_shader_ms() >= 0) sh.push_back(conv.last_shader_ms());
        if (conv.last_total_ms() >= 0) to.push_back(conv.last_total_ms());
        if (ov) {
            ov->collect(s);
            if (ov->last_shader_ms() >= 0) ovms.push_back(ov->last_shader_ms());
        }
    };
    const double t0 = now_ms();
    while (drawn < frames) {
        if (!ses.win.poll() || ses.win.should_close()) break;
        int nw = 0, nh = 0;
        if (ses.win.take_pending_size(&nw, &nh)) sc.request_recreate(nw, nh);
        if (sc.needs_recreate()) {
            while (!inflight.empty()) {
                conv.finish(inflight.front());
                inflight.pop_front();
            }
            pending_ids.clear();
            sc.create(ses.win.width(), ses.win.height());
            const bool us = !(static_cast<int>(sc.extent().width) == dec.width() &&
                              static_cast<int>(sc.extent().height) == dec.height() && (dec.width() % 4) == 0);
            conv.set_uniform_scale(us);
            last_shown = -1;
        }
        auto f = dec.next();
        if (!f) break;
        ++decoded;
        // Reclaim previous slots (non-blocking first)
        while (!inflight.empty() && conv.ready(inflight.front())) {
            const int s = inflight.front();
            inflight.pop_front();
            take_stats(s);
        }
        SwapchainFrame fr;
        if (!sc.acquire(&fr)) continue;  // ★ dropping the display does not block decoding
        const int slot = static_cast<int>(fr.sync_index);
        if (conv.pending(slot)) {
            take_stats(slot);
            for (auto it = inflight.begin(); it != inflight.end(); ++it)
                if (*it == slot) {
                    inflight.erase(it);
                    break;
                }
        }
        VkCommandBuffer cb = conv.begin(slot);
        sc.record_acquire_transition(cb, fr);
        conv.record(slot, *f, fr.target_view, fr.extent);
        if (ov && slot < ov->num_slots()) {
            char hud[64];
            std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d", static_cast<double>(drawn) * 1000.0 / std::max(now_ms() - t0, 1.0), static_cast<int>(ovdets.size()));
            ov->set_frame(slot, ovdets, hud);
            ov->record(slot, cb, fr.target_view, fr.extent,
                       yvo::display::make_overlay_affine(dec.width(), dec.height(),
                                                         static_cast<int>(fr.extent.width),
                                                         static_cast<int>(fr.extent.height)),
                       true);
        }
        sc.record_present_transition(cb, fr);
        conv.end_and_submit(slot, fr.wait_semaphore, fr.signal_semaphore, fr.submit_fence);
        inflight.push_back(slot);
        sc.present(fr);
        if (fr.present_id) pending_ids.push_back(fr.present_id);
        ++drawn;
        while (pending_ids.size() >= std::max<std::size_t>(sc.image_count() - 1, 1)) {
            const std::uint64_t id = pending_ids.front();
            pending_ids.pop_front();
            if (!sc.wait_present(id, 1000ull * 1000 * 1000)) continue;
            const double t = now_ms();
            if (last_shown >= 0) iv.push_back(t - last_shown);
            last_shown = t;
        }
    }
    while (!inflight.empty()) {
        const int s = inflight.front();
        inflight.pop_front();
        take_stats(s);
    }
    conv.clear_import_cache();
    const double wall = now_ms() - t0;

    const Stats S = stats(sh), T = stats(to), I = stats(iv);
    std::printf("on-screen frames        : decoded %ld, displayed %ld, %.0f ms → %.1f fps\n", decoded, drawn, wall,
                static_cast<double>(drawn) * 1000.0 / wall);
    std::printf("on-screen shader ms     : median=%.3f p90=%.3f max=%.3f (n=%zu)\n", S.median, S.p90, S.max, S.n);
    std::printf("on-screen total ms      : median=%.3f p90=%.3f max=%.3f (n=%zu)  (including barriers+%s)\n", T.median,
                T.p90, T.max, T.n, sc.blit_mode() ? "blit" : "layout transitions");
    std::printf("present_wait interval ms: median=%.3f p90=%.3f (n=%zu)\n", I.median, I.p90, I.n);
    if (ov) {
        const Stats O = stats(ovms);
        std::printf("on-screen overlay ms    : median=%.3f p90=%.3f max=%.3f (n=%zu)  → convert+overlay median=%.3f\n",
                    O.median, O.p90, O.max, O.n, S.median + O.median);
    }
    std::printf("presented=%llu display_dropped=%llu (NOT_READY %llu, in-flight %llu) recreates=%llu\n",
                static_cast<unsigned long long>(sc.presented()),
                static_cast<unsigned long long>(sc.display_dropped()),
                static_cast<unsigned long long>(sc.acquire_not_ready()),
                static_cast<unsigned long long>(sc.sync_busy()),
                static_cast<unsigned long long>(sc.swapchain_recreates()));
    return 0;
}

int run_convert(const Args& a) {
    const int frames = a.frames > 0 ? a.frames : 300;
    std::printf("== probe convert (P8) ==\n\n");

    DisplaySession ses;
    const bool have_display = !a.no_display && ses.open(a, "yvo display probe: convert");
    std::unique_ptr<yvo::vk::VulkanContext> own_ctx;
    yvo::vk::VulkanContext* ctx = nullptr;
    if (have_display) {
        ctx = ses.ctx.get();
    } else {
        yvo::vk::VulkanContextOptions co;
        co.enable_validation = a.validation;
        co.validation_sync = a.sync_validation;
        co.verbose = a.verbose;
        own_ctx = std::make_unique<yvo::vk::VulkanContext>(co);
        ctx = own_ctx.get();
        std::printf("(measuring offscreen only, without display)\n");
    }

    // ---- (0) workgroup size sweep (--sweep): 1:1 path only, find the optimum before other measurements ----
    if (a.sweep) {
        // The sweep target size is chosen with --display-size (default 1920x1080 = 1:1 path).
        const bool sweep_scale = !(a.width == 1920 && a.height == 1080);
        std::printf("-- workgroup sweep (%dx%d %s, offscreen %d frames). x is in 4-pixel groups --\n", a.width,
                    a.height, sweep_scale ? "uniform scale" : "1:1", frames);
        // x*y <= 256 (V3D maxComputeWorkGroupInvocations). In measurements, **tall** workgroups won.
        // x*y <= 256 (V3D maxComputeWorkGroupInvocations). In measurements, **tall** tiles won.
        const std::uint32_t cand[][2] = {{16, 16}, {32, 8}, {8, 32}, {4, 64}, {2, 128}, {1, 256},
                                         {2, 108}, {5, 51}, {6, 42}, {3, 84}, {3, 85},  {3, 72}};
        for (const auto& c : cand) {
            Args b = a;
            b.lx = c[0];
            b.ly = c[1];
            const ConvertRun r = run_offscreen(*ctx, b, a.width, a.height, frames, 3);
            std::printf("local=%3ux%-3u (%4u inv, %4ux%-3u px)  shader ms: median=%.3f p90=%.3f mean=%.3f\n", c[0],
                        c[1], c[0] * c[1], c[0] * 4, c[1], r.shader.median, r.shader.p90, r.shader.mean);
        }
        std::printf("\n");
    }

    // ---- (b) pixel accuracy: bit-exact vs CPU reference ----
    std::printf("-- pixel accuracy (1:1, CPU reference = sand.hpp de-SAND + cpu_letterbox.hpp BT.709) --\n");
    const AccuracyResult acc = run_accuracy(*ctx, a, 100);
    std::printf("frame 100: max error = %d, differing pixels %lld / %lld, alpha!=255 %d\n", acc.max_err, acc.diff_px,
                acc.total_px, acc.alpha_bad);
    if (!acc.ok && acc.first_x >= 0)
        std::printf("  first mismatch (%d,%d): GPU(%d,%d,%d) vs CPU(%d,%d,%d)\n", acc.first_x, acc.first_y,
                    acc.gpu_rgb[0], acc.gpu_rgb[1], acc.gpu_rgb[2], acc.cpu_rgb[0], acc.cpu_rgb[1], acc.cpu_rgb[2]);
    std::printf("bit-exact: %s\n\n", acc.ok ? "yes (max error 0)" : "no");

    // ---- (a) offscreen continuous stream timing ----
    std::printf("-- offscreen continuous stream %d frames (no present pacing, local=%ux%u) --\n", frames, a.lx, a.ly);
    struct SizeCase {
        int w, h;
        const char* tag;
    };
    const SizeCase sizes[] = {{1920, 1080, "1920x1080 (1:1)"},
                              {1600, 900, "1600x900  (uniform scale)"},
                              {1280, 720, "1280x720  (uniform scale)"}};
    double median_1080 = -1, median_900 = -1, median_720 = -1;
    for (const SizeCase& sz : sizes) {
        const ConvertRun r = run_offscreen(*ctx, a, sz.w, sz.h, frames, 3);
        std::printf("%-28s frames=%ld  shader ms: median=%.3f p90=%.3f mean=%.3f max=%.3f | total ms: "
                    "median=%.3f p90=%.3f\n",
                    sz.tag, r.frames, r.shader.median, r.shader.p90, r.shader.mean, r.shader.max, r.total.median,
                    r.total.p90);
        if (sz.w == 1920) median_1080 = r.shader.median;
        else if (sz.w == 1600) median_900 = r.shader.median;
        else median_720 = r.shader.median;
    }

    // ---- (c) real swapchain path ----
    if (have_display) {
        std::printf("\n-- real swapchain path (%s) --\n", a.blit ? "--display-blit forced" : "default");
        run_convert_display(a, ses, std::min(frames, 300));
    }

    std::printf("\n-- P8 pass criteria --\n");
    const ConvertRun dummy{};
    (void)dummy;
    const bool c1 = median_1080 >= 0 && median_1080 <= 7.0;
    std::printf("① 1080p shader median ≤ 7.0 ms                : %s (%.3f ms)\n", pf(c1), median_1080);
    std::printf("② bit-exact vs CPU reference                  : %s (max error %d)\n", pf(acc.ok), acc.max_err);
    if (!c1) {
        if (median_1080 <= 10.0)
            std::printf("=> recommendation: lower to 1600x900 (median %.3f ms) or --display-every 2\n", median_900);
        else
            std::printf("=> recommendation: abandon option (I), 1280x720 (%.3f ms) or 640 upscale fallback. Box coordinate system stays native\n",
                        median_720);
    }
    print_validation(*ctx);
    return (c1 && acc.ok) ? 0 : 1;
}

#else  // !YVO_HAVE_DECODER

int run_convert(const Args&) {
    std::fprintf(stderr, "[error] --probe convert requires yvo_decoder (01_src/pipeline)\n");
    return 2;
}

#endif

// ─────────────────────────────── P7: overlay ───────────────────────────────
// Pass criteria (plan §6.3): ① box border bit-exact ② glyph bit-exact
//                        ③ timestamp GPU ms < 0.7 ms (80 detections) ④ validation layer errors 0
// **No NPU, display or decoder needed at all** (offscreen). Boundary conventions are overlay_reference.hpp R0~R8.

constexpr std::array<std::uint8_t, 4> kOvBg{32, 32, 32, 255};

std::vector<yvo::Detection> synth_dets(int n, int w, int h) {
    // Non-overlapping grid (borders do not overlap each other). Labels may overlap neighbors, so the accuracy check
    // uses a separate text case (overlay_accuracy below).
    std::vector<yvo::Detection> v;
    const int cols = 10;
    const int rows = (n + cols - 1) / cols;
    const int cw = w / cols, ch = h / std::max(rows, 1);
    for (int i = 0; i < n; ++i) {
        yvo::Detection d;
        d.class_id = i % 80;
        d.score = 0.1f + 0.01f * static_cast<float>(i % 80);
        d.x1 = static_cast<float>((i % cols) * cw + 4);
        d.y1 = static_cast<float>((i / cols) * ch + 20);
        d.x2 = static_cast<float>((i % cols) * cw + cw - 6);
        d.y2 = static_cast<float>((i / cols) * ch + ch - 6);
        v.push_back(d);
    }
    return v;
}

// Input where nothing overlaps even with text enabled (same intent as 05_tests/test_overlay.cpp)
std::vector<yvo::Detection> synth_text_dets(int w, int h) {
    std::vector<yvo::Detection> v;
    const int cols = 4, rows = 3;
    const int cw = w / cols, ch = h / rows;
    for (int i = 0; i < cols * rows; ++i) {
        yvo::Detection d;
        d.class_id = (i * 7) % 80;
        d.score = 0.05f + 0.07f * static_cast<float>(i);
        d.x1 = static_cast<float>((i % cols) * cw + 8);
        // Avoid the label area (16*2 = 32 px) + HUD area (y <= 36). Overlaps would make it order-dependent and break bit-exactness.
        d.y1 = static_cast<float>((i / cols) * ch + 80);
        d.x2 = static_cast<float>((i % cols) * cw + cw / 2);
        d.y2 = static_cast<float>((i / cols) * ch + ch - 8);
        v.push_back(d);
    }
    return v;
}

struct OverlayAccuracy {
    int max_err = -1;
    long long diff_px = 0;
    long long changed_gpu = 0, changed_cpu = 0;
    int first_x = -1, first_y = -1;
};

// Clear the background → overlay → read back one frame and compare with the CPU reference.
OverlayAccuracy overlay_accuracy(yvo::vk::VulkanContext& ctx, int w, int h, const OverlayConfig& cfg,
                                 const std::vector<yvo::Detection>& dets, const std::string& hud) {
    OverlayRenderer ov(ctx, cfg);
    OffscreenTarget t(ctx, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), true,
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    const yvo::display::OverlayAffine aff = yvo::display::make_overlay_affine(w, h, w, h);
    ov.set_frame(0, dets, hud);

    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = ctx.queue_family();
    VkCommandPool pool = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateCommandPool(ctx.device(), &cp, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkAllocateCommandBuffers(ctx.device(), &cai, &cb));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    YVO_VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    barrier_image(cb, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue cc{};
    for (int i = 0; i < 4; ++i) cc.float32[i] = static_cast<float>(kOvBg[static_cast<std::size_t>(i)]) / 255.0f;
    const VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1, &rng);
    barrier_image(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    ov.record(0, cb, t.view, {t.w, t.h}, aff, false);
    t.record_readback(cb);
    YVO_VK_CHECK(vkEndCommandBuffer(cb));

    YVO_VK_CHECK(vkResetFences(ctx.device(), 1, &t.fence));
    VkCommandBufferSubmitInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbi.commandBuffer = cb;
    VkSubmitInfo2 si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    YVO_VK_CHECK(vkQueueSubmit2(ctx.queue(), 1, &si, t.fence));
    YVO_VK_CHECK(vkWaitForFences(ctx.device(), 1, &t.fence, VK_TRUE, 5'000'000'000ull));
    ov.collect(0);

    // CPU reference
    std::vector<std::uint8_t> ref(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
    for (std::size_t i = 0; i < ref.size(); i += 4)
        for (int k = 0; k < 4; ++k) ref[i + static_cast<std::size_t>(k)] = kOvBg[static_cast<std::size_t>(k)];
    yvo::display::overlay_cpu_render(ref.data(), w, h, dets, hud, aff, ov.style(),
                                     static_cast<std::size_t>(cfg.max_dets));

    OverlayAccuracy r;
    r.max_err = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t o =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
            bool cg = false, ccp = false;
            int e = 0;
            for (int k = 0; k < 4; ++k) {
                const std::uint8_t g = t.host[o + static_cast<std::size_t>(k)];
                const std::uint8_t c = ref[o + static_cast<std::size_t>(k)];
                if (g != kOvBg[static_cast<std::size_t>(k)]) cg = true;
                if (c != kOvBg[static_cast<std::size_t>(k)]) ccp = true;
                e = std::max(e, std::abs(static_cast<int>(g) - static_cast<int>(c)));
            }
            if (cg) ++r.changed_gpu;
            if (ccp) ++r.changed_cpu;
            if (e > 0) {
                ++r.diff_px;
                if (r.first_x < 0) {
                    r.first_x = x;
                    r.first_y = y;
                }
                r.max_err = std::max(r.max_err, e);
            }
        }
    }
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    return r;
}

// Counts the actually changed pixels with the CPU reference. Used to convert GPU time to **ns/px** and
// compare with display_convert's sequential-write optimum (2.05 ns/px).
long long cpu_changed_px(int w, int h, const std::vector<yvo::Detection>& dets, const std::string& hud,
                         const yvo::display::OverlayStyle& st, int max_dets) {
    std::vector<std::uint8_t> ref(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
    for (std::size_t i = 0; i < ref.size(); i += 4)
        for (int k = 0; k < 4; ++k) ref[i + static_cast<std::size_t>(k)] = kOvBg[static_cast<std::size_t>(k)];
    yvo::display::overlay_cpu_render(ref.data(), w, h, dets, hud, yvo::display::make_overlay_affine(w, h, w, h), st,
                                     static_cast<std::size_t>(max_dets));
    long long n = 0;
    for (std::size_t i = 0; i < ref.size(); i += 4)
        for (int k = 0; k < 4; ++k)
            if (ref[i + static_cast<std::size_t>(k)] != kOvBg[static_cast<std::size_t>(k)]) {
                ++n;
                break;
            }
    return n;
}

// Continuous-stream timestamp measurement. No present pacing, so the GPU stays busy (minimal DVFS effect, same basis as 2.13 ms).
// The background clear is done only on the first frame — what is measured is **only the overlay dispatch interval**.
Stats overlay_timing(yvo::vk::VulkanContext& ctx, int w, int h, const OverlayConfig& base,
                     const std::vector<yvo::Detection>& dets, const std::string& hud, int frames, int depth) {
    OverlayConfig cfg = base;
    cfg.num_slots = depth;
    OverlayRenderer ov(ctx, cfg);
    const yvo::display::OverlayAffine aff = yvo::display::make_overlay_affine(w, h, w, h);

    std::vector<std::unique_ptr<OffscreenTarget>> tgt;
    for (int i = 0; i < depth; ++i)
        tgt.push_back(std::make_unique<OffscreenTarget>(ctx, static_cast<std::uint32_t>(w),
                                                        static_cast<std::uint32_t>(h), false));
    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = ctx.queue_family();
    VkCommandPool pool = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateCommandPool(ctx.device(), &cp, nullptr, &pool));
    std::vector<VkCommandBuffer> cbs(static_cast<std::size_t>(depth), VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = static_cast<std::uint32_t>(depth);
    YVO_VK_CHECK(vkAllocateCommandBuffers(ctx.device(), &cai, cbs.data()));

    std::vector<bool> initialized(static_cast<std::size_t>(depth), false);
    std::vector<double> ms;
    std::deque<int> inflight;
    auto finish = [&](int slot) {
        YVO_VK_CHECK(vkWaitForFences(ctx.device(), 1, &tgt[static_cast<std::size_t>(slot)]->fence, VK_TRUE,
                                     5'000'000'000ull));
        ov.collect(slot);
        if (ov.last_shader_ms() >= 0) ms.push_back(ov.last_shader_ms());
    };

    for (int n = 0; n < frames; ++n) {
        const int slot = n % depth;
        while (!inflight.empty() && (static_cast<int>(inflight.size()) >= depth || inflight.front() == slot)) {
            finish(inflight.front());
            inflight.pop_front();
        }
        OffscreenTarget& t = *tgt[static_cast<std::size_t>(slot)];
        ov.set_frame(slot, dets, hud);
        YVO_VK_CHECK(vkResetFences(ctx.device(), 1, &t.fence));
        VkCommandBuffer cb = cbs[static_cast<std::size_t>(slot)];
        YVO_VK_CHECK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        YVO_VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        if (!initialized[static_cast<std::size_t>(slot)]) {
            barrier_image(cb, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            initialized[static_cast<std::size_t>(slot)] = true;
        }
        // Recorded **including the WAW barrier against the preceding compute write**, same as the real path
        ov.record(slot, cb, t.view, {t.w, t.h}, aff, true);
        YVO_VK_CHECK(vkEndCommandBuffer(cb));
        VkCommandBufferSubmitInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cbi.commandBuffer = cb;
        VkSubmitInfo2 si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        YVO_VK_CHECK(vkQueueSubmit2(ctx.queue(), 1, &si, t.fence));
        inflight.push_back(slot);
    }
    while (!inflight.empty()) {
        finish(inflight.front());
        inflight.pop_front();
    }
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    return stats(ms);
}

int run_overlay(const Args& a) {
    std::printf("== probe overlay (P7) ==\n\n");
    yvo::vk::VulkanContextOptions co;
    co.enable_validation = a.validation;
    co.validation_sync = a.sync_validation;
    co.verbose = a.verbose;
    yvo::vk::VulkanContext ctx(co);
    const int frames = a.frames > 0 ? a.frames : 300;
    const int W = a.width, H = a.height;

    OverlayConfig cfg;
    cfg.num_slots = 1;
    cfg.line_width = a.line_width;
    cfg.text_scale = a.text_scale;
    cfg.local_size_x = a.overlay_local;

    // ── ① box border bit-exact (text off, 80 detections, line width 1/2/3) ──
    std::printf("-- ① box borders (bit-exact vs CPU reference, %dx%d, 80 detections) --\n", W, H);
    bool c1 = true;
    for (int lw : {1, 2, 3}) {
        OverlayConfig g = cfg;
        g.draw_text = false;
        g.line_width = lw;
        const OverlayAccuracy r = overlay_accuracy(ctx, W, H, g, synth_dets(80, W, H), "");
        c1 = c1 && r.max_err == 0 && r.diff_px == 0 && r.changed_gpu > 0;
        std::printf("line_width %d : max_err %d  diff_px %lld  changed px %lld (CPU %lld)  %s\n", lw, r.max_err,
                    r.diff_px, r.changed_gpu, r.changed_cpu, pf(r.max_err == 0 && r.diff_px == 0));
    }

    // ── ② glyph bit-exact (labels + HUD, text_scale 1 and 2) ──
    std::printf("\n-- ② glyphs (labels + HUD, bit-exact vs CPU reference) --\n");
    bool c2 = true;
    std::string ascii;
    for (int c = 32; c <= 126; ++c) ascii.push_back(static_cast<char>(c));
    for (int s : {1, 2}) {
        OverlayConfig g = cfg;
        g.draw_text = true;
        g.text_scale = s;
        const OverlayAccuracy r1 = overlay_accuracy(ctx, W, H, g, synth_text_dets(W, H), "FPS 51.6 DET 12");
        const OverlayAccuracy r2 = overlay_accuracy(ctx, W, H, g, {}, ascii);
        c2 = c2 && r1.max_err == 0 && r1.diff_px == 0 && r1.changed_gpu > 0 && r2.max_err == 0 &&
             r2.diff_px == 0 && r2.changed_gpu > 0;
        std::printf("text_scale %d : labels max_err %d diff_px %lld (changed px %lld) | "
                    "HUD ASCII 32~126 max_err %d diff_px %lld (changed px %lld)  %s\n",
                    s, r1.max_err, r1.diff_px, r1.changed_gpu, r2.max_err, r2.diff_px, r2.changed_gpu,
                    pf(r1.max_err == 0 && r2.max_err == 0));
    }

    // ── ③ GPU time (continuous stream) ──
    std::printf("\n-- ③ GPU timestamp (offscreen continuous stream %d frames, depth 3, local_size_x=%u) --\n", frames,
                a.overlay_local);
    std::printf("%-12s %-6s %-5s %-6s %8s %8s %8s %8s %10s %8s\n", "size", "dets", "text", "lw", "median", "p90",
                "min", "max", "px", "ns/px");
    struct Row {
        const char* tag;
        int w, h, n;
        bool text;
    };
    // 5 detections is based on the measured average of 5.21/frame (07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md)
    const Row rows[] = {{"1920x1080", 1920, 1080, 80, true}, {"1920x1080", 1920, 1080, 80, false},
                        {"1920x1080", 1920, 1080, 40, true}, {"1920x1080", 1920, 1080, 5, true},
                        {"1920x1080", 1920, 1080, 0, true},  {"640x640", 640, 640, 80, true},
                        {"640x640", 640, 640, 40, true},     {"640x640", 640, 640, 5, true}};
    double worst80 = -1.0, typical = -1.0;
    for (const Row& r : rows) {
        OverlayConfig g = cfg;
        g.draw_text = r.text;
        const std::vector<yvo::Detection> d = r.text ? synth_text_dets(r.w, r.h) : synth_dets(80, r.w, r.h);
        std::vector<yvo::Detection> use = d;
        if (r.n == 0) use.clear();
        else if (static_cast<int>(use.size()) > r.n) use.resize(static_cast<std::size_t>(r.n));
        else if (r.text) { // if more than 12 are needed, use the grid (labels overlap each other, but that is irrelevant for timing)
            use = synth_dets(80, r.w, r.h);
            use.resize(static_cast<std::size_t>(r.n));
        }
        const Stats s = overlay_timing(ctx, r.w, r.h, g, use, "FPS 51.6 DET 80", frames, 3);
        yvo::display::OverlayStyle st;
        st.line_width = g.line_width;
        st.text_scale = g.text_scale;
        st.draw_text = g.draw_text;
        const long long px = cpu_changed_px(r.w, r.h, use, "FPS 51.6 DET 80", st, g.max_dets);
        std::printf("%-12s %-6zu %-5s %-6d %8.3f %8.3f %8.3f %8.3f %10lld %8.2f\n", r.tag, use.size(),
                    r.text ? "on" : "off", g.line_width, s.median, s.p90, s.min, s.max, px,
                    px ? s.median * 1e6 / static_cast<double>(px) : 0.0);
        if (r.w == 1920 && use.size() == 80 && r.text) worst80 = s.median;
        if (r.w == 1920 && use.size() == 5 && r.text) typical = s.median;
    }
    const bool c3 = worst80 >= 0 && worst80 < 0.7;
    std::printf("\n80 detections (1080p, text on) median = %.3f ms  target < 0.7 ms → %s\n", worst80, pf(c3));
    std::printf("5 detections (measured average 5.21/frame) median = %.3f ms\n", typical);
    std::printf("note: display_convert.comp's sequential-write optimum is 2.05 ns/px (1920x1080 1:1, 4.23 ms).\n"
                "      the overlay writes scattered thin borders and glyphs, so its per-px cost is higher than that.\n");

    // ── ④ validation layer ──
    std::printf("\n");
    print_validation(ctx);
    const bool c4 = ctx.validation_errors() == 0;
    std::printf("\nverdict: ① borders %s  ② glyphs %s  ③ GPU < 0.7 ms %s  ④ validation errors 0 %s\n", pf(c1), pf(c2),
                pf(c3), pf(c4));
    return (c1 && c2 && c3 && c4) ? 0 : 1;
}

// ─────────────────────────────── extra: dmabuf-export helper check ───────────────────────────────
// Records whether 01_src/vulkan/dmabuf_export.{hpp,cpp}, to be used by the postprocessing track (P2), actually works, and
// how much **requested size / allocated size / measured dma-buf size** differ from each other.
// hailo-expert decides which value to pass to the HailoRT side (dma_map_dmabuf / read_async) by looking at these numbers.
int run_export(const Args& a) {
    std::printf("== probe export (dma-buf export helper) ==\n\n");
    yvo::vk::VulkanContextOptions co;
    co.enable_validation = a.validation;
    co.validation_sync = a.sync_validation;
    co.verbose = a.verbose;
    yvo::vk::VulkanContext ctx(co);

    yvo::vk::DmaBufExporterConfig ec;
    ec.align = yvo::vk::kHailoAsyncAlign;   // 16384 B — required by HailoRT read_async
    ec.verbose = a.verbose;
    yvo::vk::DmaBufExporter exp(ctx, ec);

    // yolov11m.hef raw 6-stream sizes + 640x640x3 preprocessed tensor
    const std::size_t sizes[] = {80u * 80 * 80, 80u * 80 * 64, 40u * 40 * 80, 40u * 40 * 64,
                                 20u * 24 * 80, 20u * 24 * 64, 640u * 640 * 3};
    std::printf("align = %zu B\n", ec.align);
    std::printf("dedicated allocation forced: %s\n", ctx.dmabuf_export_dedicated_only() ? "yes" : "no");
    std::printf("%-4s %12s %12s %12s %6s %-18s %-18s %s\n", "#", "requested", "alloc", "dmabuf", "fd", "host_ptr",
                "aligned_ptr", "align_offset");
    bool ok = true;
    for (std::size_t sz : sizes) {
        const int i = exp.allocate(sz);
        const yvo::vk::ExportedDmaBuf& e = exp.at(i);
        // Check that the mapping is actually writable and readable (the probe verifies the values).
        e.aligned_host_ptr()[0] = 0xa5;
        e.aligned_host_ptr()[sz - 1] = 0x5a;
        const bool rw = e.aligned_host_ptr()[0] == 0xa5 && e.aligned_host_ptr()[sz - 1] == 0x5a;
        const bool aligned = (reinterpret_cast<std::uintptr_t>(e.aligned_host_ptr()) % ec.align) == 0;
        // On devices where dedicated allocation is forced, allocationSize cannot be raised per spec (see header comment).
        const bool size_ok = e.dedicated || (e.alloc_size % ec.align) == 0;
        ok = ok && rw && aligned && size_ok && e.fd >= 0;
        std::printf("%-4d %12zu %12llu %12zu %6d %-18p %-18p %zu%s%s%s\n", i, e.requested_size,
                    static_cast<unsigned long long>(e.alloc_size), e.dmabuf_size, e.fd,
                    static_cast<void*>(e.host_ptr), static_cast<void*>(e.aligned_host_ptr()), e.align_offset,
                    rw ? "" : "  [RW FAIL]", aligned ? "" : "  [ALIGN FAIL]", size_ok ? "" : "  [SIZE FAIL]");
    }
    // Also create an fd to hand over via dup_fd (the caller closes it).
    const int dup = exp.dup_fd(0);
    std::printf("\ndup_fd(0) = %d (caller closes)\n", dup);
    if (dup >= 0) ::close(dup);
    std::printf("entries where requested size != measured dma-buf size: ");
    int mismatch = 0;
    for (std::size_t i = 0; i < exp.count(); ++i)
        if (exp.at(static_cast<int>(i)).requested_size != exp.at(static_cast<int>(i)).dmabuf_size) ++mismatch;
    std::printf("%d / %zu\n", mismatch, exp.count());
    std::printf("=> size candidates to pass to HailoRT dma_map_dmabuf(fd, size, D2H) / read_async(fd, size) are\n"
                "   two: ① requested_size ② dmabuf_size. hailo-expert must test both.\n");
    std::printf("verdict: %s\n", pf(ok));
    print_validation(ctx);
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 1;
    }
    try {
        if (a.probe == "context-regression") return run_context_regression(a);
        if (a.probe == "surface") return run_surface(a);
        if (a.probe == "present") return run_present(a);
        if (a.probe == "convert") return run_convert(a);
        if (a.probe == "overlay") return run_overlay(a);
        if (a.probe == "export") return run_export(a);
        std::fprintf(stderr, "[error] unknown probe: %s\n", a.probe.c_str());
        usage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 1;
    }
}
