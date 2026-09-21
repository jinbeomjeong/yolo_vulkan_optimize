// yvo_display_pipeline_probe: 마일스톤 7 검증 도구.
//
// `DisplayPipeline` 만 단독으로 돌려서 `yvo_display_probe --probe convert` 의 스왑체인 구간 수치
// (convert median 4.25 / overlay 0.151 / present 간격 16.661 / dropped 0)를 **DisplayPipeline 경유로**
// 재현하는 것이 목적이다. NPU 도 전처리도 쓰지 않는다 — 디코더와 표시 경로뿐이다.
//
// main.cpp 의 슬롯 규약을 그대로 흉내 낸다:
//   retain(slot, frame) → (지연 흉내: NPU 지연만큼 늦게) show(slot, dets, hud) → pump() → display_done
// display_done 콜백이 슬롯을 free 목록에 돌려주므로 2조건 반납의 **표시 쪽 절반**이 여기서 검증된다.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "core/coco_labels.hpp"
#include "display/display_pipeline.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "vulkan/vulkan_context.hpp"
#include "toolkit/stats.hpp"

using yvo::Stats;
using yvo::stats;

namespace {

using Clock = std::chrono::steady_clock;
using yvo::display::DisplayPipeline;
using yvo::display::DisplayPipelineConfig;

std::vector<yvo::Detection> synth_dets(int n, int w, int h) {
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

struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    int frames = 300;
    int slots = 4;
    int dets = 5;
    int lag = 4;                 // show 를 몇 프레임 늦출 것인가 (NPU 지연 흉내)
    int toggle_at = 0;           // N 프레임째에 fullscreen 을 토글해 **실행 중 리사이즈**를 강제한다
    bool validation = true;
    bool sync_validation = false;
    bool verbose = false;
    DisplayPipelineConfig cfg;
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--video PATH] [--frames N] [--slots N] [--dets N] [--lag N]\n"
                 "          [--display-backend auto|wayland|offscreen] [--display-sync matched|latest]\n"
                 "          [--display-pace source|asap|none] [--display-fullscreen] [--display-size WxH]\n"
                 "          [--display-every K] [--present-mode fifo|mailbox|immediate] [--display-blit]\n"
                 "          [--overlay-line-width N] [--overlay-text-scale N] [--toggle-fullscreen-at N]\n"
                 "          [--no-validation] [--sync-validation]\n",
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    a.cfg.pace = yvo::display::DisplayPace::Asap;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--frames" && (v = next())) a.frames = std::atoi(v);
        else if (s == "--slots" && (v = next())) a.slots = std::atoi(v);
        else if (s == "--dets" && (v = next())) a.dets = std::atoi(v);
        else if (s == "--lag" && (v = next())) a.lag = std::atoi(v);
        else if (s == "--toggle-fullscreen-at" && (v = next())) a.toggle_at = std::atoi(v);
        else if (s == "--display-backend" && (v = next())) {
            if (!yvo::display::parse_backend_kind(v, &a.cfg.backend)) return false;
        } else if (s == "--display-sync" && (v = next())) {
            if (!yvo::display::parse_display_sync(v, &a.cfg.sync)) return false;
        } else if (s == "--display-pace" && (v = next())) {
            if (!yvo::display::parse_display_pace(v, &a.cfg.pace)) return false;
        } else if (s == "--display-fullscreen") a.cfg.fullscreen = true;
        else if (s == "--display-size" && (v = next())) {
            int w = 0, h = 0;
            if (std::sscanf(v, "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) return false;
            a.cfg.width = w;
            a.cfg.height = h;
        } else if (s == "--display-every" && (v = next())) a.cfg.every = std::atoi(v);
        else if (s == "--present-mode" && (v = next())) {
            if (!yvo::display::parse_present_mode(v, &a.cfg.present_mode)) return false;
        } else if (s == "--display-blit") a.cfg.force_blit = true;
        else if (s == "--overlay-line-width" && (v = next())) a.cfg.overlay_line_width = std::atoi(v);
        else if (s == "--overlay-text-scale" && (v = next())) a.cfg.overlay_text_scale = std::atoi(v);
        else if (s == "--no-validation") a.validation = false;
        else if (s == "--sync-validation") a.sync_validation = true;
        else if (s == "--verbose") a.verbose = true;
        else return false;
    }
    if (a.frames < 1 || a.slots < 1 || a.cfg.every < 1) return false;
    a.cfg.num_slots = a.slots;
    a.cfg.verbose = a.verbose;
    a.cfg.app_id = "yvo_display_pipeline_probe";
    a.cfg.title = "yvo display pipeline probe";
    return true;
}

int run(const Args& a) {
    yvo::FfmpegDrmDecoder dec;
    yvo::FfmpegDrmDecoder::Options dopt;
    dopt.extra_hw_frames = std::max(8, a.slots + 4);
    if (!dec.open(a.video, dopt)) {
        std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
        return 1;
    }

    DisplayPipelineConfig cfg = a.cfg;
    cfg.src_w = dec.width();
    cfg.src_h = dec.height();

    // ⚠ 순서: WaylandWindow → VulkanContext(enable_present) → DisplayPipeline
    yvo::display::WaylandWindow win;
    const bool have_win = DisplayPipeline::open_window(win, cfg);
    if (!have_win && cfg.backend == yvo::display::DisplayBackendKind::Wayland) {
        std::fprintf(stderr, "wayland 창 열기 실패: %s\n", win.error().c_str());
        return 1;
    }

    yvo::vk::VulkanContextOptions co;
    co.enable_validation = a.validation;
    co.validation_sync = a.sync_validation;
    co.verbose = a.verbose;
    co.enable_present = have_win;
    co.wayland_display = have_win ? win.native_display() : nullptr;
    co.require_present = have_win && cfg.backend == yvo::display::DisplayBackendKind::Wayland;
    yvo::vk::VulkanContext ctx(co);

    DisplayPipeline disp(ctx, have_win ? &win : nullptr, cfg);
    std::printf("[cfg] backend=%s %ux%u %s / %s images=%u sync=%s pace=%s every=%d\n",
                disp.offscreen() ? "offscreen" : "wayland", disp.extent().width, disp.extent().height,
                disp.uniform_scale() ? "uniform-scale(bilinear)" : "1:1",
                disp.blit_mode() ? "blit fallback" : "compute -> storage image", disp.image_count(),
                yvo::display::sync_name(cfg.sync), yvo::display::pace_name(cfg.pace), cfg.every);

    // 슬롯 풀: display_done 콜백이 반납한다(main.cpp 2조건 반납의 표시 쪽 절반).
    std::deque<int> free_slots;
    for (int i = 0; i < a.slots; ++i) free_slots.push_back(i);
    disp.set_display_done_callback([&](int s) { free_slots.push_back(s); });

    const std::vector<yvo::Detection> dets = synth_dets(a.dets, dec.width(), dec.height());
    std::deque<std::pair<int, long>> lagq;   // (slot, 프레임 번호) — lag 프레임 뒤에 show
    long decoded = 0, shown = 0, dropped_show = 0;
    int held_max = 0;
    const auto t0 = Clock::now();

    bool toggled = false;
    while (decoded < a.frames) {
        if (!disp.poll_window()) break;
        // 실행 중 리사이즈: 컴포지터가 새 크기로 configure 하면 스왑체인이 재생성된다.
        if (have_win && a.toggle_at > 0 && !toggled && decoded >= a.toggle_at) {
            win.set_fullscreen(!cfg.fullscreen);
            toggled = true;
        }
        // 슬롯 확보 (main.cpp 의 pump 루프와 같은 모양)
        const auto deadline = Clock::now() + std::chrono::seconds(5);
        while (free_slots.empty()) {
            disp.pump();
            if (!lagq.empty() && (decoded - lagq.front().second) >= a.lag) {
                const int s = lagq.front().first;
                lagq.pop_front();
                char hud[64];
                std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d",
                              static_cast<double>(shown) * 1000.0 /
                                  std::max(std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), 1.0),
                              static_cast<int>(dets.size()));
                if (disp.show(s, dets, hud)) {
                    ++shown;
                    if (cfg.sync == yvo::display::DisplaySync::Latest) free_slots.push_back(s);
                } else {
                    ++dropped_show;
                    free_slots.push_back(s);
                }
                continue;
            }
            if (Clock::now() > deadline) {
                std::fprintf(stderr, "[error] 슬롯 대기 타임아웃 (표시 슬롯이 돌아오지 않는다)\n");
                return 1;
            }
        }
        auto f = dec.next();
        if (!f) break;
        const int slot = free_slots.front();
        free_slots.pop_front();
        f->pts = decoded;
        disp.retain(slot, *f);
        f.reset();
        lagq.emplace_back(slot, decoded);
        ++decoded;
        held_max = std::max(held_max, disp.frames_held());

        while (!lagq.empty() && (decoded - lagq.front().second) >= a.lag) {
            const int s = lagq.front().first;
            lagq.pop_front();
            char hud[64];
            std::snprintf(hud, sizeof(hud), "FPS %4.1f DET %d",
                          static_cast<double>(shown) * 1000.0 /
                              std::max(std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), 1.0),
                          static_cast<int>(dets.size()));
            if (disp.show(s, dets, hud)) {
                ++shown;
                if (cfg.sync == yvo::display::DisplaySync::Latest) free_slots.push_back(s);
            } else {
                ++dropped_show;
                free_slots.push_back(s);
            }
        }
        disp.pump();
        held_max = std::max(held_max, disp.frames_held());
    }
    // 드레인
    while (!lagq.empty()) {
        const int s = lagq.front().first;
        lagq.pop_front();
        if (disp.show(s, dets, "drain")) ++shown;
        else ++dropped_show;
        disp.pump();
    }
    disp.drain();
    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    const Stats C = stats(disp.convert_ms()), T = stats(disp.convert_total_ms()), O = stats(disp.overlay_ms()),
                I = stats(disp.present_interval_ms()), G = stats(disp.glass_ms());
    std::printf("frames                  : 디코드 %ld, show 성공 %ld, show 드롭 %ld, %.0f ms → %.1f fps\n", decoded,
                shown, dropped_show, wall, static_cast<double>(shown) * 1000.0 / wall);
    std::printf("convert shader ms       : median=%.3f p90=%.3f max=%.3f (n=%zu)\n", C.median, C.p90, C.max, C.n);
    std::printf("convert total ms        : median=%.3f p90=%.3f (n=%zu)\n", T.median, T.p90, T.n);
    std::printf("overlay shader ms       : median=%.3f p90=%.3f max=%.3f (n=%zu)  → 합 median=%.3f\n", O.median,
                O.p90, O.max, O.n, C.median + O.median);
    std::printf("present 간격 ms         : median=%.3f p90=%.3f (n=%zu)\n", I.median, I.p90, I.n);
    std::printf("glass-to-glass ms       : median=%.3f p90=%.3f (n=%zu)  (retain → present_wait 반환)\n", G.median,
                G.p90, G.n);
    std::printf("presented=%llu display_dropped=%llu recreates=%llu overlay_overflow=%llu frames_held_max=%d\n",
                static_cast<unsigned long long>(disp.presented()),
                static_cast<unsigned long long>(disp.display_dropped()),
                static_cast<unsigned long long>(disp.recreates()),
                static_cast<unsigned long long>(disp.overlay_overflows()), held_max);
    std::printf("vk validation errors %llu warnings %llu (layer %s)\n",
                static_cast<unsigned long long>(ctx.validation_errors()),
                static_cast<unsigned long long>(ctx.validation_warnings()),
                ctx.validation_layer_active() ? "on" : "off");
    std::fflush(stdout);
    return ctx.validation_errors() > 0 ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }
    try {
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
