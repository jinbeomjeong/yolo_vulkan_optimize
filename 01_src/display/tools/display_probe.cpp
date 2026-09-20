// yvo_display_probe: 계획서 §6.3 의 가시화 프로브.
//
//   context-regression (P6): present off / on 두 VulkanContext 를 만들어
//     활성 인스턴스 확장 목록 · 활성 디바이스 확장 목록 · 선택된 큐 패밀리 · timestamp 유효 비트 ·
//     external queue family 를 **정렬된 텍스트**로 출력한다.
//     present=off 블록이 변경 전 골든 상태와 같아야 한다(회귀 판정).
//
//   surface (P4): WaylandWindow → vkCreateWaylandSurfaceKHR →
//     vkGetPhysicalDeviceWaylandPresentationSupportKHR → surface caps · 포맷 목록 ·
//     **present mode 별 minImageCount** 덤프 +
//     vkGetPhysicalDeviceImageFormatProperties2(R8G8B8A8_UNORM, STORAGE) 성공 여부.
//     합격 기준 5개를 각각 PASS/FAIL 로 출력한다. **스왑체인은 만들지 않는다.**
//
//   present (P5, ★핵심 관문): 스왑체인 생성 → display_probe.comp 가 알려진 테스트 패턴을
//     storage image 에 직접 쓴다 → FIFO 로 N 프레임 present → vkWaitForPresentKHR 간격 기록.
//     **픽셀 검사는 조정자가 grim -o HDMI-A-1 로 한다** — 프로브는 패턴 좌표·기대 RGB 를 stdout 에
//     명시하고 --hold-seconds 동안 화면을 유지한다(그 사이에 캡처한다).
//     중간에 한 프레임을 일부러 discard 해 vkReleaseSwapchainImagesEXT 경로도 검증 레이어에 태운다.
//
//   overlay (P7): OffscreenBackend 에 합성 검출로 overlay_boxes.comp 실행 → 리드백 → CPU 기준과 비교.
//     ① 박스 테두리 비트 일치 ② 글리프 비트 일치 ③ GPU timestamp < 0.7 ms(검출 80개) ④ 검증 errors 0.
//     **NPU·디스플레이·디코더 전부 불필요**(오프스크린). 경계 규약은 01_src/display/overlay_reference.hpp.
//
//   convert (P8, ★원본 해상도 성립 여부): display_convert.comp 로 디코더 SAND dma-buf 를 RGBA 로 변환.
//     (a) **오프스크린 연속 스트림** 300프레임 timestamp 중앙값·p90 — 1920x1080 / 1600x900 / 1280x720.
//         present 페이싱이 GPU 를 놀려 DVFS 로 시간이 부풀지 않도록 표시 없이 back-to-back 으로 잰다
//         (2.13 ms 전처리 수치와 같은 측정 조건).
//     (b) **픽셀 정확도**: CPU 기준(sand.hpp de-SAND + cpu_letterbox.hpp 색 수식)과 비트 일치 여부·최대 오차.
//     (c) **실제 스왑체인 경로**로도 돌려 present 통계와 total(배리어+blit 포함) 시간을 낸다.
//
// 종료 코드: 0 성공 / 2 프로브가 요구하는 환경이 없음 / 1 치명적 오류
//
// 실행 예 (SSH):
//   XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0
//     build/01_src/display/yvo_display_probe --probe present --display-fullscreen --hold-seconds 5
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
#include "display/overlay_renderer.hpp"
#include "display/vk_swapchain.hpp"
#include "display/wayland_window.hpp"
#include "vulkan/dmabuf_export.hpp"
#include "vulkan/vk_check.hpp"
#include "vulkan/vulkan_context.hpp"

#ifdef YVO_HAVE_DECODER
#include "common/cpu_letterbox.hpp"
#include "common/dmabuf_map.hpp"
#include "common/sand.hpp"
#include "pipeline/ffmpeg_drm_decoder.hpp"
#include "display_probe_spv.h"
#else
#include "display_probe_spv.h"
#endif

// 오버레이(다음 단계)가 쓸 공용 헤더. 헤더 안의 static_assert 가 렌더링 규약(행/비트 순서)과 색 공식을
// **컴파일 시** 검증하므로, OverlayRenderer 가 붙기 전까지 이 프로브를 유일한 컴파일 지점으로 둔다.
#include "common/coco_labels.hpp"
#include "common/glyph_font_8x16.hpp"

static_assert(yvo::class_color(0)[0] == 255 && yvo::class_color(0)[1] == 80 && yvo::class_color(0)[2] == 0,
              "class_color()는 RGB다 (BGR 규칙을 뒤집은 것)");
static_assert(yvo::glyph_pixel(yvo::glyph_index('A'), 3, 2), "8x16 글리프 규약");

namespace {

using yvo::display::SwapchainFrame;
using yvo::display::VulkanSwapchain;
using yvo::display::OverlayConfig;
using yvo::display::OverlayRenderer;

// P7 섹션(아래)에서 정의한다. P8 의 --with-overlay 가 먼저 쓰므로 여기서 선언해 둔다.
std::vector<yvo::Detection> synth_dets(int n, int w, int h);

// ─────────────────────────────── 공통 ───────────────────────────────

struct Stats {
    double median = 0, p90 = 0, mean = 0, min = 0, max = 0;
    std::size_t n = 0;
};

Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) {
        const double idx = p * static_cast<double>(v.size() - 1);
        const auto lo = static_cast<std::size_t>(idx);
        const std::size_t hi = std::min(lo + 1, v.size() - 1);
        return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
    };
    s.median = pct(0.5);
    s.p90 = pct(0.9);
    s.min = v.front();
    s.max = v.back();
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    s.n = v.size();
    return s;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char* pf(bool ok) { return ok ? "PASS" : "FAIL"; }

struct Args {
    std::string probe = "context-regression";
    std::string video = "04_data/videos/video_h265.mp4";
    int width = 1920, height = 1080;
    bool fullscreen = false;
    bool verbose = false;
    bool validation = false;
    bool sync_validation = false;
    bool blit = false;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    int frames = 0;         // 0 = 프로브 기본값
    double hold_seconds = 3.0;
    std::uint32_t lx = 3, ly = 85;    // DisplayConvertConfig 기본값과 같게 (스윕 실측 최적)
    bool no_display = false; // convert: 스왑체인 구간을 건너뛴다
    bool lean_images = false; // 모드별 minImageCount(FIFO 3) opt-in — 검증 레이어가 error 를 낸다
    bool sweep = false;       // convert: 1920x1080 1:1 경로에서 워크그룹 크기 스윕
    int resize_every = 0;     // present: N 프레임마다 fullscreen 을 토글해 **실행 중 창 리사이즈**를 유발한다
    int line_width = 2;       // overlay: --overlay-line-width (1080p 기본 2)
    int text_scale = 2;       // overlay: --overlay-text-scale (1080p 기본 2, 8x16 → 16x32)
    std::uint32_t overlay_local = 128; // overlay: --overlay-local (OverlayConfig 기본값과 같게, 실측 최적)
    bool with_overlay = false; // convert: 같은 커맨드 버퍼에 오버레이까지 이어 기록한다(통합 검증)
    int overlay_dets = 80;     // convert/--with-overlay: 합성 검출 수
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

// WaylandWindow + present 가 켜진 VulkanContext 를 한 번에 연다. 실패하면 nullptr 대신 false.
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
            std::printf("wayland       : 실패 — %s\n", win.error().c_str());
            return false;
        }
        yvo::vk::VulkanContextOptions co;
        co.enable_validation = a.validation;
        co.validation_sync = a.sync_validation;
        co.verbose = a.verbose;
        co.enable_present = true;
        co.wayland_display = win.native_display();
        co.require_present = true; // 프로브는 조용한 폴백을 원하지 않는다
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
    // 목록은 **정렬해서** 출력한다 — 코드상의 push_back 순서가 바뀌어도 diff 가 흔들리지 않게.
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
    // ---- 1) present = off : 오늘까지의 코드 경로. 골든과 대조할 블록이다 ----
    ContextSnapshot off;
    {
        yvo::vk::VulkanContextOptions o;
        o.enable_validation = a.validation;
        o.verbose = a.verbose;
        yvo::vk::VulkanContext ctx(o);
        off = snapshot(ctx);
    }

    // ---- 2) WaylandWindow : VulkanContext 보다 **먼저** 만든다 (Vulkan 심볼 0개) ----
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
        std::printf("wayland       : 실패 — %s\n", win.error().c_str());
        std::printf("=> present=on 블록을 만들 수 없다. 오프스크린 폴백이 필요한 상황이다.\n");
        return 2;
    }
    std::printf("-- wayland window --\n");
    std::printf("size          : %dx%d%s\n", win.width(), win.height(), a.fullscreen ? " (fullscreen)" : "");
    std::printf("configured    : %s\n", win.configured() ? "yes" : "no");
    std::printf("globals       : viewporter=%d presentation=%d subcompositor=%d\n", win.has_viewporter(),
                win.has_presentation(), win.has_subcompositor());
    std::printf("native        : wl_display=%p wl_surface=%p\n", win.native_display(), win.native_surface());
    std::printf("\n");

    // ---- 3) present = on : wl_display* 를 불투명 포인터로 넘긴다. VkSurfaceKHR 는 만들지 않는다 ----
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
        std::printf("vulkan        : 실패 — %s\n", e.what());
        return 2;
    }
    print_block("present=on", on);

    std::printf("-- diff (off -> on) --\n");
    print_added("added instance exts", off.inst_exts, on.inst_exts);
    print_added("added device exts  ", off.dev_exts, on.dev_exts);
    std::printf("queue family       : %u -> %u %s\n", off.queue_family, on.queue_family,
                off.queue_family == on.queue_family ? "(동일)" : "(★변경됨)");
    std::printf("timestamp bits     : %u -> %u %s\n", off.timestamp_bits, on.timestamp_bits,
                off.timestamp_bits == on.timestamp_bits ? "(동일)" : "(★변경됨)");
    std::printf("external qf        : %s -> %s %s\n", off.external_qf.c_str(), on.external_qf.c_str(),
                off.external_qf == on.external_qf ? "(동일)" : "(★변경됨)");
    std::printf("dma-buf            : import=%d/%d export=%d/%d dedicated_only=%d/%d %s\n", off.dmabuf_import,
                on.dmabuf_import, off.dmabuf_export, on.dmabuf_export, off.dmabuf_dedicated, on.dmabuf_dedicated,
                (off.dmabuf_import == on.dmabuf_import && off.dmabuf_export == on.dmabuf_export &&
                 off.dmabuf_dedicated == on.dmabuf_dedicated)
                    ? "(동일)"
                    : "(★변경됨)");

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
                s.caps.currentExtent.width == 0xFFFFFFFFu ? "  (0xFFFFFFFF = 우리가 정한다)" : "");
    std::printf("image extent    : min %ux%u  max %ux%u\n", s.caps.minImageExtent.width, s.caps.minImageExtent.height,
                s.caps.maxImageExtent.width, s.caps.maxImageExtent.height);
    std::printf("image count     : min %u max %u (기본 caps)\n", s.caps.minImageCount, s.caps.maxImageCount);
    std::printf("supportedUsage  : 0x%x  STORAGE=%d COLOR_ATTACHMENT=%d TRANSFER_DST=%d\n",
                s.caps.supportedUsageFlags, (s.caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) != 0,
                (s.caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
                (s.caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0);
    std::printf("formats         : %zu\n", s.formats.size());
    for (const auto& f : s.formats)
        std::printf("  %-26s %s%s\n", format_name(f.format), color_space_name(f.colorSpace),
                    (f.format == VK_FORMAT_R8G8B8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                        ? "   <-- 우리가 쓰는 포맷"
                        : "");
    std::printf("present modes   : %zu\n", s.present_modes.size());
    for (VkPresentModeKHR m : s.present_modes) std::printf("  %s\n", present_mode_name(m));
    std::printf("mode별 minImageCount (VK_EXT_surface_maintenance1): %zu\n", s.mode_counts.size());
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
    VulkanSwapchain sc(*ses.ctx, so);   // surface 만 만든다
    const yvo::display::SurfaceInfo& s = sc.surface_info();
    dump_surface(s);

    std::printf("\n-- P4 합격 기준 --\n");
    const bool c1 = s.wayland_presentation_support == VK_TRUE && s.surface_support == VK_TRUE;
    const bool c2 = s.format_available;
    const bool c3 = s.usage_storage;
    const bool c4 = s.fifo_available;
    const bool c5 = s.storage_image_result == VK_SUCCESS;
    std::printf("① family %u presentation support VK_TRUE      : %s\n", s.queue_family, pf(c1));
    std::printf("② R8G8B8A8_UNORM + SRGB_NONLINEAR 존재        : %s\n", pf(c2));
    std::printf("③ supportedUsageFlags 에 STORAGE              : %s\n", pf(c3));
    std::printf("④ FIFO present mode 존재                      : %s\n", pf(c4));
    std::printf("⑤ ImageFormatProperties2(UNORM, STORAGE) 성공 : %s\n", pf(c5));
    std::printf("=> 컴퓨트 직접 쓰기 %s (blit 폴백 %s)\n", (c3 && c5) ? "가능" : "불가",
                (c3 && c5) ? "불필요" : "필요");
    print_validation(*ses.ctx);
    return (c1 && c2 && c3 && c4 && c5) ? 0 : 1;
}

// ─────────────────────────────── P5: present ───────────────────────────────

// display_probe.comp 전용 최소 파이프라인 (storage image 1개 + push constant 12 B)
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

    // 스왑체인 배리어까지 포함해 커맨드 버퍼를 통째로 기록하고 제출한다.
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
    std::printf("-- 기대 패턴 (grim -o HDMI-A-1 로 검사할 좌표와 RGB) --\n");
    std::printf("surface size            : %dx%d\n", w, h);
    std::printf("background              : (32,32,32)\n");
    std::printf("center block  x[%d,%d) y[%d,%d) : RGB(255,0,128)   중심 픽셀 (%d,%d)\n", (w - 400) / 2,
                (w - 400) / 2 + 400, (h - 300) / 2, (h - 300) / 2 + 300, w / 2, h / 2);
    std::printf("corner TL 64x64 (0,0)         : RGB(255,0,0)     샘플 (16,16)\n");
    std::printf("corner TR 64x64 (%d,0)      : RGB(0,255,0)     샘플 (%d,16)\n", w - 64, w - 16);
    std::printf("corner BL 64x64 (0,%d)      : RGB(0,0,255)     샘플 (16,%d)\n", h - 64, h - 16);
    std::printf("corner BR 64x64 (%d,%d)   : RGB(255,255,255) 샘플 (%d,%d)\n", w - 64, h - 64, w - 16, h - 16);
    std::printf("moving bar y[%d,%d)          : RGB(255,255,0)   (살아 있음의 증거, 위치는 프레임마다 다르다)\n",
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
    std::printf("present_wait            : %s\n", sc.present_wait_available() ? "사용" : "불가(측정 불가)");
    print_expected_pattern(static_cast<int>(sc.extent().width), static_cast<int>(sc.extent().height));

    ProbePattern pat(*ses.ctx, sc.image_count());
    std::deque<std::pair<std::uint64_t, double>> pending; // (present_id, present 호출 시각)
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
            if (!sc.wait_present(id, 1000ull * 1000 * 1000)) { // 1초
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
        // **실행 중 창 리사이즈**를 유발한다: fullscreen 을 토글하면 컴포지터가 새 크기로 configure 를 보낸다.
        // 스왑체인 재생성 + 세마포어 수명 + 구 스왑체인 폐기 경로를 검증 레이어·ASan 아래에서 반복 실행한다.
        if (a.resize_every > 0 && presented > 0 && presented % a.resize_every == 0 && presented != last_toggle) {
            last_toggle = presented;
            fs_state = !fs_state;
            ses.win.set_fullscreen(fs_state);
        }
        SwapchainFrame f;
        if (!sc.acquire(&f)) continue;
        // 한 프레임은 일부러 그리지 않고 되돌린다 — vkReleaseSwapchainImagesEXT 경로를 검증 레이어에 태운다.
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

    std::printf("\n-- 결과 --\n");
    std::printf("presented               : %llu (요청 %d), 루프 %.0f ms → %.1f fps\n",
                static_cast<unsigned long long>(sc.presented()), frames, wall,
                static_cast<double>(sc.presented()) * 1000.0 / wall);
    std::printf("display_dropped         : %llu (acquire NOT_READY %llu, 인플라이트 부족 %llu, discard %llu)\n",
                static_cast<unsigned long long>(sc.display_dropped()),
                static_cast<unsigned long long>(sc.acquire_not_ready()),
                static_cast<unsigned long long>(sc.sync_busy()), static_cast<unsigned long long>(discarded));
    std::printf("swapchain_recreates     : %llu, vkReleaseSwapchainImagesEXT 반납 %llu (대기 %zu)\n",
                static_cast<unsigned long long>(sc.swapchain_recreates()),
                static_cast<unsigned long long>(sc.images_released()), sc.pending_releases());
    std::printf("present_wait timeouts   : %llu\n", static_cast<unsigned long long>(timeouts));
    const Stats iv = stats(intervals), la = stats(latencies);
    std::printf("present_wait 간격 ms    : median=%.3f p90=%.3f mean=%.3f min=%.3f max=%.3f (n=%zu)\n", iv.median,
                iv.p90, iv.mean, iv.min, iv.max, iv.n);
    std::printf("present→화면 지연 ms    : median=%.3f p90=%.3f max=%.3f (n=%zu)\n", la.median, la.p90, la.max, la.n);

    std::printf("\n-- P5 합격 기준 --\n");
    const bool c1 = sc.created();
    const bool c2 = ses.ctx->validation_errors() == 0;
    const bool c3 = static_cast<int>(sc.presented()) >= frames && timeouts == 0;
    const bool c4 = iv.n > 0 && iv.median >= 14.7 && iv.median <= 18.7;
    std::printf("① vkCreateSwapchainKHR VK_SUCCESS             : %s\n", pf(c1));
    std::printf("② 검증 레이어 errors 0                        : %s (errors=%llu)\n", pf(c2),
                static_cast<unsigned long long>(ses.ctx->validation_errors()));
    std::printf("③ %d present 완료, 타임아웃 0                : %s\n", frames, pf(c3));
    std::printf("④ present_wait 간격 중앙값 16.7 ms ±2         : %s (%.3f ms)\n", pf(c4), iv.median);
    std::printf("⑤ grim 픽셀 검사                              : 조정자가 수행 (아래 HOLD 구간에서 캡처)\n");
    print_validation(*ses.ctx);

    // ── 화면 유지: 조정자가 grim 으로 캡처하는 구간 ──
    if (a.hold_seconds > 0) {
        std::printf("\nHOLD START %.1f s — 지금 grim -o HDMI-A-1 로 캡처하라\n", a.hold_seconds);
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

// ─────────── 공통: 오프스크린 렌더 타깃 (P7 overlay / P8 convert 가 같이 쓴다) ───────────

// STORAGE|TRANSFER_SRC 이미지 + 리드백 버퍼. 정확도 검사와 무페이싱 타이밍에 쓴다.
// extra_usage 로 TRANSFER_DST 등을 덧붙일 수 있다(P7 은 배경을 vkCmdClearColorImage 로 채운다).
// **P8 경로는 extra_usage = 0 으로 오늘까지와 완전히 같은 이미지를 만든다**(측정 회귀 방지).
struct OffscreenTarget {
    yvo::vk::VulkanContext& ctx;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory buf_mem = VK_NULL_HANDLE;
    std::uint8_t* host = nullptr;
    VkFence fence = VK_NULL_HANDLE;
    std::uint32_t w = 0, h = 0;

    OffscreenTarget(yvo::vk::VulkanContext& c, std::uint32_t width, std::uint32_t height, bool readback,
                    VkImageUsageFlags extra_usage = 0)
        : ctx(c), w(width), h(height) {
        VkDevice dev = ctx.device();
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | extra_usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        YVO_VK_CHECK(vkCreateImage(dev, &ici, nullptr, &image));
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(dev, image, &req);
        std::uint32_t type = ctx.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
        YVO_VK_CHECK(vkBindImageMemory(dev, image, mem, 0));
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        YVO_VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &view));

        if (readback) {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = static_cast<VkDeviceSize>(w) * h * 4;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
            vkGetBufferMemoryRequirements(dev, buf, &req);
            type = ctx.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = type;
            YVO_VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &buf_mem));
            YVO_VK_CHECK(vkBindBufferMemory(dev, buf, buf_mem, 0));
            void* p = nullptr;
            YVO_VK_CHECK(vkMapMemory(dev, buf_mem, 0, VK_WHOLE_SIZE, 0, &p));
            host = static_cast<std::uint8_t*>(p);
        }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        YVO_VK_CHECK(vkCreateFence(dev, &fi, nullptr, &fence));
    }
    ~OffscreenTarget() {
        VkDevice dev = ctx.device();
        if (dev == VK_NULL_HANDLE) return;
        if (fence) vkDestroyFence(dev, fence, nullptr);
        if (host) vkUnmapMemory(dev, buf_mem);
        if (buf) vkDestroyBuffer(dev, buf, nullptr);
        if (buf_mem) vkFreeMemory(dev, buf_mem, nullptr);
        if (view) vkDestroyImageView(dev, view, nullptr);
        if (image) vkDestroyImage(dev, image, nullptr);
        if (mem) vkFreeMemory(dev, mem, nullptr);
    }
    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;
};

void barrier_image(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                   VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = ss;
    b.srcAccessMask = sa;
    b.dstStageMask = ds;
    b.dstAccessMask = da;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

// ─────────────────────────────── P8: convert ───────────────────────────────

#ifdef YVO_HAVE_DECODER

// 오프스크린 연속 스트림 측정. present 페이싱이 없으므로 GPU 가 계속 바쁘다(DVFS 영향 최소).
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

// 픽셀 정확도: CPU 기준(sand.hpp de-SAND + cpu_letterbox.hpp 색 수식, scale=1)과 비교한다.
struct AccuracyResult {
    bool ok = false;
    int max_err = -1;
    long long diff_px = 0;
    long long total_px = 0;
    int alpha_bad = 0;
    int first_x = -1, first_y = -1;
    int gpu_rgb[3]{}, cpu_rgb[3]{};
};

AccuracyResult run_accuracy(yvo::vk::VulkanContext& ctx, const Args& a, int frame_index) {
    AccuracyResult r;
    yvo::FfmpegDrmDecoder dec;
    if (!dec.open(a.video)) throw std::runtime_error("decoder open failed: " + dec.error());
    std::optional<yvo::DrmFrame> f;
    for (int i = 0; i <= frame_index; ++i) {
        f = dec.next();
        if (!f) break;
    }
    if (!f) throw std::runtime_error("accuracy: 프레임을 얻지 못했다");
    const int w = f->width, h = f->height;

    yvo::display::DisplayConvertConfig cc;
    cc.num_slots = 1;
    cc.local_size_x = a.lx;
    cc.local_size_y = a.ly;
    cc.uniform_scale = false; // 1:1 경로를 검증한다
    cc.max_dst_w = std::max(w, 1920);
    yvo::display::DisplayConvert conv(ctx, cc);
    OffscreenTarget t(ctx, static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), true);

    YVO_VK_CHECK(vkResetFences(ctx.device(), 1, &t.fence));
    VkCommandBuffer cb = conv.begin(0);
    barrier_image(cb, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                  VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    conv.record(0, *f, t.view, {t.w, t.h});
    barrier_image(cb, t.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {t.w, t.h, 1};
    vkCmdCopyImageToBuffer(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.buf, 1, &region);
    VkBufferMemoryBarrier2 bb{};
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    bb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    bb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    bb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    bb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = t.buf;
    bb.size = VK_WHOLE_SIZE;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.bufferMemoryBarrierCount = 1;
    di.pBufferMemoryBarriers = &bb;
    vkCmdPipelineBarrier2(cb, &di);
    conv.end_and_submit(0, VK_NULL_HANDLE, VK_NULL_HANDLE, t.fence);
    conv.finish(0);

    // CPU 기준: dma-buf mmap → de-SAND → cpu_letterbox_nv12_bt709 (scale = 1, 패딩 없음)
    std::vector<std::uint8_t> y(static_cast<std::size_t>(w) * h), uv(static_cast<std::size_t>(w) * h / 2);
    {
        yvo::DmaBufReadMap my(f->fd, f->size), muv(f->uv_fd, f->uv_size);
        if (!my.data() || !muv.data() ||
            !yvo::desand_nv12(my.data(), f->size, muv.data(), f->uv_size, yvo::make_sand_layout(*f), y.data(),
                              static_cast<std::size_t>(w), uv.data(), static_cast<std::size_t>(w)))
            throw std::runtime_error("accuracy: de-SAND 실패");
    }
    yvo::LetterboxParams lb;
    lb.dst_w = w;
    lb.dst_h = h;
    lb.scale = 1.0f;
    lb.pad_x = 0;
    lb.pad_y = 0;
    std::vector<std::uint8_t> ref(static_cast<std::size_t>(w) * h * 3);
    yvo::cpu_letterbox_nv12_bt709(y.data(), static_cast<std::size_t>(w), uv.data(), static_cast<std::size_t>(w), w,
                                  h, lb, ref.data());

    r.total_px = static_cast<long long>(w) * h;
    r.max_err = 0;
    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            const std::uint8_t* g = t.host + (static_cast<std::size_t>(py) * w + px) * 4;
            const std::uint8_t* c = ref.data() + (static_cast<std::size_t>(py) * w + px) * 3;
            if (g[3] != 255) ++r.alpha_bad;
            int e = 0;
            for (int k = 0; k < 3; ++k) e = std::max(e, std::abs(static_cast<int>(g[k]) - static_cast<int>(c[k])));
            if (e > 0) {
                ++r.diff_px;
                if (e > r.max_err || r.first_x < 0) {
                    if (r.first_x < 0) {
                        r.first_x = px;
                        r.first_y = py;
                        for (int k = 0; k < 3; ++k) {
                            r.gpu_rgb[k] = g[k];
                            r.cpu_rgb[k] = c[k];
                        }
                    }
                    r.max_err = std::max(r.max_err, e);
                }
            }
        }
    }
    r.ok = r.max_err == 0 && r.alpha_bad == 0;
    return r;
}

// 실제 스왑체인 경로 (표시하면서 변환). present 통계와 total(배리어+blit 포함) 시간을 낸다.
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

    // --with-overlay: **DisplayConvert 가 만든 같은 커맨드 버퍼에 이어** 오버레이를 기록한다.
    // 실제 통합(계획서 §5.4)과 같은 구조이므로 이 경로가 검증 레이어(sync 포함)의 판정 대상이다.
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
        std::printf("overlay                 : 검출 %d개, line_width %d, text_scale %d, local_size_x %u\n",
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
        // 이전 슬롯 회수(논블로킹 우선)
        while (!inflight.empty() && conv.ready(inflight.front())) {
            const int s = inflight.front();
            inflight.pop_front();
            take_stats(s);
        }
        SwapchainFrame fr;
        if (!sc.acquire(&fr)) continue;  // ★ 표시를 버려도 디코드는 막히지 않는다
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
    std::printf("on-screen frames        : 디코드 %ld, 표시 %ld, %.0f ms → %.1f fps\n", decoded, drawn, wall,
                static_cast<double>(drawn) * 1000.0 / wall);
    std::printf("on-screen shader ms     : median=%.3f p90=%.3f max=%.3f (n=%zu)\n", S.median, S.p90, S.max, S.n);
    std::printf("on-screen total ms      : median=%.3f p90=%.3f max=%.3f (n=%zu)  (배리어+%s 포함)\n", T.median,
                T.p90, T.max, T.n, sc.blit_mode() ? "blit" : "레이아웃 전환");
    std::printf("present_wait 간격 ms    : median=%.3f p90=%.3f (n=%zu)\n", I.median, I.p90, I.n);
    if (ov) {
        const Stats O = stats(ovms);
        std::printf("on-screen overlay ms    : median=%.3f p90=%.3f max=%.3f (n=%zu)  → 변환+오버레이 median=%.3f\n",
                    O.median, O.p90, O.max, O.n, S.median + O.median);
    }
    std::printf("presented=%llu display_dropped=%llu (NOT_READY %llu, 인플라이트 %llu) recreates=%llu\n",
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
        std::printf("(표시 없이 오프스크린으로만 측정한다)\n");
    }

    // ---- (0) 워크그룹 크기 스윕 (--sweep): 1:1 경로만, 다른 측정 전에 최적값을 찾는다 ----
    if (a.sweep) {
        // 스윕 대상 크기는 --display-size 로 고른다(기본 1920x1080 = 1:1 경로).
        const bool sweep_scale = !(a.width == 1920 && a.height == 1080);
        std::printf("-- 워크그룹 스윕 (%dx%d %s, 오프스크린 %d프레임). x 는 4픽셀 묶음 단위다 --\n", a.width,
                    a.height, sweep_scale ? "uniform scale" : "1:1", frames);
        // x*y <= 256 (V3D maxComputeWorkGroupInvocations). 실측에서 **세로로 긴** 워크그룹이 이겼다.
        // x*y <= 256 (V3D maxComputeWorkGroupInvocations). 실측에서 **세로로 긴** 타일이 이겼다.
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

    // ---- (b) 픽셀 정확도: CPU 기준과 비트 일치 ----
    std::printf("-- 픽셀 정확도 (1:1, CPU 기준 = sand.hpp de-SAND + cpu_letterbox.hpp BT.709) --\n");
    const AccuracyResult acc = run_accuracy(*ctx, a, 100);
    std::printf("frame 100: max 오차 = %d, 다른 픽셀 %lld / %lld, alpha!=255 %d\n", acc.max_err, acc.diff_px,
                acc.total_px, acc.alpha_bad);
    if (!acc.ok && acc.first_x >= 0)
        std::printf("  첫 불일치 (%d,%d): GPU(%d,%d,%d) vs CPU(%d,%d,%d)\n", acc.first_x, acc.first_y,
                    acc.gpu_rgb[0], acc.gpu_rgb[1], acc.gpu_rgb[2], acc.cpu_rgb[0], acc.cpu_rgb[1], acc.cpu_rgb[2]);
    std::printf("비트 일치: %s\n\n", acc.ok ? "예 (max 오차 0)" : "아니오");

    // ---- (a) 오프스크린 연속 스트림 타이밍 ----
    std::printf("-- 오프스크린 연속 스트림 %d프레임 (present 페이싱 없음, local=%ux%u) --\n", frames, a.lx, a.ly);
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

    // ---- (c) 실제 스왑체인 경로 ----
    if (have_display) {
        std::printf("\n-- 실제 스왑체인 경로 (%s) --\n", a.blit ? "--display-blit 강제" : "기본");
        run_convert_display(a, ses, std::min(frames, 300));
    }

    std::printf("\n-- P8 합격 기준 --\n");
    const ConvertRun dummy{};
    (void)dummy;
    const bool c1 = median_1080 >= 0 && median_1080 <= 7.0;
    std::printf("① 1080p shader 중앙값 ≤ 7.0 ms               : %s (%.3f ms)\n", pf(c1), median_1080);
    std::printf("② CPU 기준과 비트 일치                        : %s (max 오차 %d)\n", pf(acc.ok), acc.max_err);
    if (!c1) {
        if (median_1080 <= 10.0)
            std::printf("=> 권고: 1600x900 로 낮춘다 (중앙값 %.3f ms) 또는 --display-every 2\n", median_900);
        else
            std::printf("=> 권고: 안 (I) 포기, 1280x720 (%.3f ms) 또는 640 업스케일 폴백. 박스 좌표계는 원본 유지\n",
                        median_720);
    }
    print_validation(*ctx);
    return (c1 && acc.ok) ? 0 : 1;
}

#else  // !YVO_HAVE_DECODER

int run_convert(const Args&) {
    std::fprintf(stderr, "[error] --probe convert 는 yvo_decoder 가 필요하다 (01_src/pipeline)\n");
    return 2;
}

#endif

// ─────────────────────────────── P7: overlay ───────────────────────────────
// 합격 기준(계획서 §6.3): ① 박스 테두리 비트 일치 ② 글리프 비트 일치
//                        ③ timestamp GPU ms < 0.7 ms (검출 80개) ④ 검증 레이어 errors 0
// **NPU·디스플레이·디코더 전부 필요 없다**(오프스크린). 경계 규약은 overlay_reference.hpp R0~R8.

constexpr std::array<std::uint8_t, 4> kOvBg{32, 32, 32, 255};

std::vector<yvo::Detection> synth_dets(int n, int w, int h) {
    // 겹치지 않는 격자(테두리끼리 안 겹친다). 라벨은 이웃과 겹칠 수 있으므로 정확도 판정은
    // 텍스트 케이스를 따로 둔다(아래 overlay_accuracy).
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

// 텍스트까지 켜도 아무것도 겹치지 않는 입력 (05_tests/test_overlay.cpp 와 같은 의도)
std::vector<yvo::Detection> synth_text_dets(int w, int h) {
    std::vector<yvo::Detection> v;
    const int cols = 4, rows = 3;
    const int cw = w / cols, ch = h / rows;
    for (int i = 0; i < cols * rows; ++i) {
        yvo::Detection d;
        d.class_id = (i * 7) % 80;
        d.score = 0.05f + 0.07f * static_cast<float>(i);
        d.x1 = static_cast<float>((i % cols) * cw + 8);
        // 라벨 자리(16*2 = 32 px) + HUD 영역(y <= 36) 회피. 겹치면 순서 의존이 되어 비트 일치가 깨진다.
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

// 한 프레임을 배경 클리어 → 오버레이 → 리드백 하고 CPU 기준과 비교한다.
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
    barrier_image(cb, t.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {t.w, t.h, 1};
    vkCmdCopyImageToBuffer(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.buf, 1, &region);
    VkBufferMemoryBarrier2 bb{};
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    bb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    bb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    bb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    bb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = t.buf;
    bb.size = VK_WHOLE_SIZE;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.bufferMemoryBarrierCount = 1;
    di.pBufferMemoryBarriers = &bb;
    vkCmdPipelineBarrier2(cb, &di);
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

    // CPU 기준
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

// CPU 기준으로 실제 변경 픽셀 수를 센다. GPU 시간을 **ns/px** 로 환산해
// display_convert 의 순차 쓰기 최적치(2.05 ns/px)와 비교하기 위한 것이다.
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

// 연속 스트림 timestamp 측정. present 페이싱이 없어 GPU 가 계속 바쁘다(DVFS 영향 최소, 2.13 ms 와 같은 기준).
// 배경 클리어는 첫 프레임에만 한다 — 재는 것은 **오버레이 디스패치 구간**뿐이다.
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
        // 실제 경로와 같게 **앞선 컴퓨트 쓰기와의 WAW 배리어를 포함**해 기록한다
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

    // ── ① 박스 테두리 비트 일치 (텍스트 끔, 검출 80개, 선 두께 1/2/3) ──
    std::printf("-- ① 박스 테두리 (CPU 기준과 비트 일치, %dx%d, 검출 80개) --\n", W, H);
    bool c1 = true;
    for (int lw : {1, 2, 3}) {
        OverlayConfig g = cfg;
        g.draw_text = false;
        g.line_width = lw;
        const OverlayAccuracy r = overlay_accuracy(ctx, W, H, g, synth_dets(80, W, H), "");
        c1 = c1 && r.max_err == 0 && r.diff_px == 0 && r.changed_gpu > 0;
        std::printf("line_width %d : max_err %d  diff_px %lld  변경 px %lld (CPU %lld)  %s\n", lw, r.max_err,
                    r.diff_px, r.changed_gpu, r.changed_cpu, pf(r.max_err == 0 && r.diff_px == 0));
    }

    // ── ② 글리프 비트 일치 (라벨 + HUD, text_scale 1 과 2) ──
    std::printf("\n-- ② 글리프 (라벨 + HUD, CPU 기준과 비트 일치) --\n");
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
        std::printf("text_scale %d : 라벨 max_err %d diff_px %lld (변경 px %lld) | "
                    "HUD ASCII 32~126 max_err %d diff_px %lld (변경 px %lld)  %s\n",
                    s, r1.max_err, r1.diff_px, r1.changed_gpu, r2.max_err, r2.diff_px, r2.changed_gpu,
                    pf(r1.max_err == 0 && r2.max_err == 0));
    }

    // ── ③ GPU 시간 (연속 스트림) ──
    std::printf("\n-- ③ GPU timestamp (오프스크린 연속 스트림 %d프레임, depth 3, local_size_x=%u) --\n", frames,
                a.overlay_local);
    std::printf("%-12s %-6s %-5s %-6s %8s %8s %8s %8s %10s %8s\n", "size", "dets", "text", "lw", "median", "p90",
                "min", "max", "px", "ns/px");
    struct Row {
        const char* tag;
        int w, h, n;
        bool text;
    };
    // 검출 5개는 실측 평균 5.21개/frame 기준이다(07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md)
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
        else if (r.text) { // 12개보다 많이 필요하면 격자를 쓴다(라벨끼리 겹치지만 타이밍에는 무관)
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
    std::printf("\n검출 80개(1080p, 텍스트 on) median = %.3f ms  기준 < 0.7 ms → %s\n", worst80, pf(c3));
    std::printf("검출 5개(실측 평균 5.21/frame) median = %.3f ms\n", typical);
    std::printf("참고: display_convert.comp 의 순차 쓰기 최적치는 2.05 ns/px 다(1920x1080 1:1, 4.23 ms).\n"
                "      오버레이는 얇은 테두리·글리프를 흩뿌려 쓰므로 px 당 비용이 그보다 높다.\n");

    // ── ④ 검증 레이어 ──
    std::printf("\n");
    print_validation(ctx);
    const bool c4 = ctx.validation_errors() == 0;
    std::printf("\n판정: ① 테두리 %s  ② 글리프 %s  ③ GPU < 0.7 ms %s  ④ 검증 errors 0 %s\n", pf(c1), pf(c2),
                pf(c3), pf(c4));
    return (c1 && c2 && c3 && c4) ? 0 : 1;
}

// ─────────────────────────────── 부가: dmabuf-export 헬퍼 확인 ───────────────────────────────
// 후처리 트랙(P2)이 쓸 01_src/vulkan/dmabuf_export.{hpp,cpp} 가 실제로 동작하는지, 그리고
// **요청 크기 / 할당 크기 / dma-buf 실측 크기** 가 서로 얼마나 다른지를 기록한다.
// HailoRT 쪽(dma_map_dmabuf / read_async)에 어느 값을 넘길지는 hailo-expert 가 이 숫자를 보고 정한다.
int run_export(const Args& a) {
    std::printf("== probe export (dma-buf 내보내기 헬퍼) ==\n\n");
    yvo::vk::VulkanContextOptions co;
    co.enable_validation = a.validation;
    co.validation_sync = a.sync_validation;
    co.verbose = a.verbose;
    yvo::vk::VulkanContext ctx(co);

    yvo::vk::DmaBufExporterConfig ec;
    ec.align = yvo::vk::kHailoAsyncAlign;   // 16384 B — HailoRT read_async 요구
    ec.verbose = a.verbose;
    yvo::vk::DmaBufExporter exp(ctx, ec);

    // yolov11m.hef 원시 6스트림 크기 + 640x640x3 전처리 텐서
    const std::size_t sizes[] = {80u * 80 * 80, 80u * 80 * 64, 40u * 40 * 80, 40u * 40 * 64,
                                 20u * 24 * 80, 20u * 24 * 64, 640u * 640 * 3};
    std::printf("align = %zu B\n", ec.align);
    std::printf("dedicated 할당 강제: %s\n", ctx.dmabuf_export_dedicated_only() ? "예" : "아니오");
    std::printf("%-4s %12s %12s %12s %6s %-18s %-18s %s\n", "#", "requested", "alloc", "dmabuf", "fd", "host_ptr",
                "aligned_ptr", "align_offset");
    bool ok = true;
    for (std::size_t sz : sizes) {
        const int i = exp.allocate(sz);
        const yvo::vk::ExportedDmaBuf& e = exp.at(i);
        // 매핑이 실제로 쓰고 읽히는지 확인한다(프로브가 값을 검증한다).
        e.aligned_host_ptr()[0] = 0xa5;
        e.aligned_host_ptr()[sz - 1] = 0x5a;
        const bool rw = e.aligned_host_ptr()[0] == 0xa5 && e.aligned_host_ptr()[sz - 1] == 0x5a;
        const bool aligned = (reinterpret_cast<std::uintptr_t>(e.aligned_host_ptr()) % ec.align) == 0;
        // 전용 할당이 강제되는 장치에서는 spec 상 allocationSize 를 올릴 수 없다(헤더 주석 참고).
        const bool size_ok = e.dedicated || (e.alloc_size % ec.align) == 0;
        ok = ok && rw && aligned && size_ok && e.fd >= 0;
        std::printf("%-4d %12zu %12llu %12zu %6d %-18p %-18p %zu%s%s%s\n", i, e.requested_size,
                    static_cast<unsigned long long>(e.alloc_size), e.dmabuf_size, e.fd,
                    static_cast<void*>(e.host_ptr), static_cast<void*>(e.aligned_host_ptr()), e.align_offset,
                    rw ? "" : "  [RW FAIL]", aligned ? "" : "  [ALIGN FAIL]", size_ok ? "" : "  [SIZE FAIL]");
    }
    // dup_fd 로 넘겨줄 fd 도 만들어 본다(호출자가 close 한다).
    const int dup = exp.dup_fd(0);
    std::printf("\ndup_fd(0) = %d (호출자가 close)\n", dup);
    if (dup >= 0) ::close(dup);
    std::printf("요청 크기 != dma-buf 실측 크기인 항목: ");
    int mismatch = 0;
    for (std::size_t i = 0; i < exp.count(); ++i)
        if (exp.at(static_cast<int>(i)).requested_size != exp.at(static_cast<int>(i)).dmabuf_size) ++mismatch;
    std::printf("%d / %zu\n", mismatch, exp.count());
    std::printf("=> HailoRT dma_map_dmabuf(fd, size, D2H) / read_async(fd, size) 에 넘길 size 후보는\n"
                "   ① requested_size ② dmabuf_size 두 가지다. hailo-expert 가 둘 다 시험해야 한다.\n");
    std::printf("판정: %s\n", pf(ok));
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
        std::fprintf(stderr, "[error] 알 수 없는 프로브: %s\n", a.probe.c_str());
        usage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 1;
    }
}
