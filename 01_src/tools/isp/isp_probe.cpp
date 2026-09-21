// yvo_isp_probe: feasibility check tool for BCM2712 PiSP Back End (ISP) resize
//
//   --dump            build the config only (without opening the device) and print describe().
//   --convert         decode the --frame N-th frame of the video → ISP resize → save to --png.
//                     With --frames K, processes K frames from there while cycling slots (to check slots, cache, fds).
//   --check-padding   check byte by byte that the output buffer's letterbox padding area is exactly pad_value.
//
// Check the riskiest things first: if the image comes out right and the padding stays intact, the rest is engineering.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef YVO_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "isp/isp_resizer.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"

namespace {

struct Args {
    std::string video = YVO_DEFAULT_VIDEO;
    std::string media = "/dev/media0";
    std::string png;
    std::string filter;         // resample filter (empty = smart selection)
    int frame = 0;
    int frames = 1;
    int slots = yvo::kNumSlots;
    int buffers = 12;
    unsigned int branch = 1;
    bool dump = false;
    bool convert = false;
    bool check_padding = false;
    bool no_letterbox = false;
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--dump] [--convert] [--check-padding]\n"
                 "          [--video PATH] [--frame N] [--frames K] [--png OUT.png] [--media /dev/mediaN]\n"
                 "          [--branch N] [--slots N] [--buffers N] [--filter NAME] [--no-letterbox]\n",
                 a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (s == "--video" && (v = next())) a.video = v;
        else if (s == "--media" && (v = next())) a.media = v;
        else if (s == "--png" && (v = next())) a.png = v;
        else if (s == "--filter" && (v = next())) a.filter = v;
        else if (s == "--frame" && (v = next())) a.frame = std::atoi(v);
        else if (s == "--frames" && (v = next())) a.frames = std::atoi(v);
        else if (s == "--slots" && (v = next())) a.slots = std::atoi(v);
        else if (s == "--buffers" && (v = next())) a.buffers = std::atoi(v);
        else if (s == "--branch" && (v = next())) a.branch = static_cast<unsigned int>(std::atoi(v));
        else if (s == "--dump") a.dump = true;
        else if (s == "--convert") a.convert = true;
        else if (s == "--check-padding") a.check_padding = true;
        else if (s == "--no-letterbox") a.no_letterbox = true;
        else return false;
    }
    if (!a.dump && !a.convert && !a.check_padding) a.dump = true;
    return a.slots > 0 && a.buffers >= a.slots && a.frames > 0;
}

bool save_rgb(const std::string& path, const std::uint8_t* rgb, int w, int h) {
#ifdef YVO_HAVE_OPENCV
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    cv::Mat m(h, w, CV_8UC3, const_cast<std::uint8_t*>(rgb));
    cv::Mat bgr;
    cv::cvtColor(m, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path, bgr)) return false;
#else
    std::string p = path;
    if (p.size() > 4 && p.substr(p.size() - 4) == ".png") p = p.substr(0, p.size() - 4) + ".ppm";
    std::ofstream o(p, std::ios::binary);
    o << "P6\n" << w << " " << h << "\n255\n";
    o.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(w) * h * 3);
    if (!o) return false;
    std::printf("(no OpenCV) saving as PPM: %s\n", p.c_str());
    return true;
#endif
    std::printf("saved: %s\n", path.c_str());
    return true;
}

// Check that the letterbox padding area (top/bottom rows, left/right columns) is all pad_value. Returns the violating pixel count.
long check_padding(const std::uint8_t* rgb, const yvo::LetterboxParams& lb, int w, int h, int content_w,
                   int content_h) {
    long bad = 0;
    long bad_rows = 0, bad_cols = 0;
    int first_bad_y = -1, first_bad_x = -1;
    for (int y = 0; y < h; ++y) {
        const bool row_is_pad = (y < lb.pad_y) || (y >= lb.pad_y + content_h);
        for (int x = 0; x < w; ++x) {
            const bool col_is_pad = (x < lb.pad_x) || (x >= lb.pad_x + content_w);
            if (!row_is_pad && !col_is_pad) continue;
            const std::uint8_t* px = rgb + (static_cast<std::size_t>(y) * w + x) * 3;
            if (px[0] != lb.pad_value || px[1] != lb.pad_value || px[2] != lb.pad_value) {
                ++bad;
                if (first_bad_y < 0) {
                    first_bad_y = y;
                    first_bad_x = x;
                }
                if (row_is_pad) ++bad_rows;
                else ++bad_cols;
            }
        }
    }
    std::printf("padding check: rows [0,%d) and [%d,%d), cols [0,%d) and [%d,%d), expected %u\n", lb.pad_y,
                lb.pad_y + content_h, h, lb.pad_x, lb.pad_x + content_w, w,
                static_cast<unsigned>(lb.pad_value));
    if (bad == 0) {
        std::printf("padding check: OK (0 violating pixels)\n");
    } else {
        const std::uint8_t* px = rgb + (static_cast<std::size_t>(first_bad_y) * w + first_bad_x) * 3;
        std::printf("padding check: FAIL %ld violating pixels (%ld in pad rows, %ld in pad cols); "
                    "first at (%d,%d) = %u,%u,%u\n",
                    bad, bad_rows, bad_cols, first_bad_x, first_bad_y, px[0], px[1], px[2]);
    }
    return bad;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }

    const bool need_device = a.convert || a.check_padding;

    try {
        yvo::isp::IspResizerConfig cfg;
        cfg.media_device = a.media;
        cfg.branch = a.branch;
        cfg.num_slots = a.slots;
        cfg.buffer_count = a.buffers;
        cfg.letterbox = !a.no_letterbox;
        cfg.resample_filter = a.filter;
        cfg.open_device = need_device;

        // Input geometry is read from the video (when opening the device). With --dump only, build the config from defaults (1080p).
        yvo::FfmpegDrmDecoder dec;
        if (need_device) {
            yvo::FfmpegDrmDecoder::Options dopt;
            dopt.extra_hw_frames = 2 * yvo::kNumSlots;
            if (!dec.open(a.video, dopt)) {
                std::fprintf(stderr, "decoder open failed: %s\n", dec.error().c_str());
                return 1;
            }
            cfg.src_w = dec.width();
            cfg.src_h = dec.height();
        }

        // Check the SAND column height and coded_height on the first frame, then create the resizer.
        std::optional<yvo::DrmFrame> frame;
        if (need_device) {
            while ((frame = dec.next())) {
                if (frame->pts >= a.frame) break;
            }
            if (!frame) {
                std::fprintf(stderr, "frame %d not found (eof, error: %s)\n", a.frame,
                             dec.error().empty() ? "none" : dec.error().c_str());
                return 1;
            }
            cfg.coded_height = frame->coded_height;
            cfg.sand_col_height = static_cast<int>(frame->sand_col_height);
            cfg.uv_sand_col_height = static_cast<int>(frame->uv_sand_col_height);
            std::printf("frame %lld: %dx%d coded %d, sand cols %u/%u, y fd %d (%zu B), uv fd %d (%zu B), "
                        "offsets %zu/%zu\n",
                        static_cast<long long>(frame->pts), frame->width, frame->height, frame->coded_height,
                        frame->sand_col_height, frame->uv_sand_col_height, frame->fd, frame->size, frame->uv_fd,
                        frame->uv_size, frame->y_offset, frame->uv_offset);
        }

        yvo::isp::IspResizer isp(cfg);
        std::printf("%s", isp.describe().c_str());

        if (!need_device) return 0;

        // --frames K: process K frames while cycling slots (checks slot reuse, input cache, fd leaks).
        std::vector<std::uint8_t> rgb;
        std::vector<double> isp_ms;
        int done = 0;
        for (; done < a.frames && frame; ++done) {
            const int slot = done % isp.num_slots();
            const yvo::PreprocOutput out = isp.process(*frame, slot);
            isp_ms.push_back(isp.last_isp_ms());
            if (done == 0) {
                std::printf("isp: %.3f ms, output %d x %d, %zu B, dmabuf fd %d, layout %s\n", isp.last_isp_ms(),
                            isp.out_width(), isp.out_height(), out.size, out.dmabuf_fd,
                            out.layout == yvo::TensorLayout::Nhwc ? "nhwc" : "nhcw");
                // Wrap CPU reads in dma-buf sync.
                rgb.resize(out.size);
                const yvo::isp::IspResizer::OutputRead lock(isp, slot);
                std::memcpy(rgb.data(), lock.data(), out.size);
            }
            if (done + 1 < a.frames) frame = dec.next();
        }
        if (done < a.frames)
            std::fprintf(stderr, "warning: video ended after %d of %d frames\n", done, a.frames);
        if (isp_ms.size() > 1) {
            std::vector<double> v = isp_ms;
            std::sort(v.begin(), v.end());
            std::printf("isp over %zu frames: median %.3f ms, min %.3f, max %.3f\n", v.size(), v[v.size() / 2],
                        v.front(), v.back());
            std::printf("%s", isp.describe().c_str());
        }

        int rc = 0;
        if (a.check_padding && cfg.letterbox) {
            const int content_w = static_cast<int>(static_cast<float>(cfg.src_w) * isp.letterbox().scale + 0.5f);
            const int content_h = static_cast<int>(static_cast<float>(cfg.src_h) * isp.letterbox().scale + 0.5f);
            if (check_padding(rgb.data(), isp.letterbox(), isp.out_width(), isp.out_height(), content_w, content_h) !=
                0)
                rc = 1;
        } else if (a.check_padding) {
            std::printf("padding check: skipped (letterbox disabled)\n");
        }

        if (!a.png.empty()) {
            if (!save_rgb(a.png, rgb.data(), isp.out_width(), isp.out_height())) {
                std::fprintf(stderr, "failed to save %s\n", a.png.c_str());
                rc = 1;
            }
        }
        // Release decoder frame references before the resizer.
        frame.reset();
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
