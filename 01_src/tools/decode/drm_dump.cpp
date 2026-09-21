// yvo_drm_dump: inspection tool for the decoder's DRM_PRIME output
//   descriptor dump, byte comparison of CPU de-SAND vs av_hwframe_transfer_data, full decode performance measurement
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include <drm_fourcc.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
}

#include "ref/sand.hpp"
#include "decode/ffmpeg_drm_decoder.hpp"
#include "toolkit/tool_util.hpp"

namespace {

using yvo::tool::cpu_ms;
struct Args {
    std::string video;
    int frames = 3;          // number of frames whose descriptor is printed
    std::string desand_dir;  // if non-empty, de-SAND compare and save
    int desand_frame = 0;    // frame index to compare
    bool count = false;      // full decode performance measurement
    int hold = 0;            // frames held at once in count mode
    int max_frames = 0;      // frame limit in count mode (0 = all)
    yvo::FfmpegDrmDecoder::Options dec_opt;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--frames N] [--desand OUTDIR [--desand-frame K]] "
                 "[--count [--hold N] [--max-frames N]] [--extra-hw-frames N] [--threads N] VIDEO\n",
                 argv0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto val = [&](int& dst) {
            if (i + 1 >= argc) return false;
            dst = std::atoi(argv[++i]);
            return true;
        };
        if (s == "--frames") {
            if (!val(a.frames)) return false;
        } else if (s == "--desand") {
            if (i + 1 >= argc) return false;
            a.desand_dir = argv[++i];
        } else if (s == "--desand-frame") {
            if (!val(a.desand_frame)) return false;
        } else if (s == "--count") {
            a.count = true;
        } else if (s == "--hold") {
            if (!val(a.hold)) return false;
        } else if (s == "--max-frames") {
            if (!val(a.max_frames)) return false;
        } else if (s == "--extra-hw-frames") {
            if (!val(a.dec_opt.extra_hw_frames)) return false;
        } else if (s == "--threads") {
            if (!val(a.dec_opt.threads)) return false;
        } else if (!s.empty() && s[0] == '-') {
            return false;
        } else {
            a.video = s;
        }
    }
    return !a.video.empty();
}

std::string fourcc_str(std::uint32_t f) {
    std::string s(4, ' ');
    for (int i = 0; i < 4; ++i) s[static_cast<std::size_t>(i)] = static_cast<char>((f >> (8 * i)) & 0xff);
    return s;
}

int count_open_fds() {
    int n = 0;
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/proc/self/fd", ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec))
        ++n;
    return n;
}

void dump_descriptor(const yvo::DrmFrame& f) {
    const AVFrame* av = yvo::FfmpegDrmDecoder::av_frame(f);
    const auto* d = reinterpret_cast<const AVDRMFrameDescriptor*>(av->data[0]);
    std::printf("frame %lld: AVFrame %dx%d crop(t=%zu b=%zu l=%zu r=%zu) pts=%lld\n",
                static_cast<long long>(f.pts), av->width, av->height, av->crop_top, av->crop_bottom,
                av->crop_left, av->crop_right, static_cast<long long>(av->pts));
    std::printf("  nb_objects=%d nb_layers=%d\n", d->nb_objects, d->nb_layers);
    for (int o = 0; o < d->nb_objects; ++o) {
        const auto& ob = d->objects[o];
        std::printf("  object[%d]: fd=%d size=%zu modifier=0x%016llx (vendor=0x%02llx mod=%llu param=%d)\n", o,
                    ob.fd, ob.size, static_cast<unsigned long long>(ob.format_modifier),
                    static_cast<unsigned long long>(ob.format_modifier >> 56),
                    static_cast<unsigned long long>(ob.format_modifier & 0xff),
                    fourcc_mod_broadcom_param(ob.format_modifier));
    }
    for (int l = 0; l < d->nb_layers; ++l) {
        const auto& ly = d->layers[l];
        std::printf("  layer[%d]: format=%s(0x%08x) nb_planes=%d\n", l, fourcc_str(ly.format).c_str(),
                    ly.format, ly.nb_planes);
        for (int p = 0; p < ly.nb_planes; ++p)
            std::printf("    plane[%d]: object=%d offset=%td pitch=%td\n", p, ly.planes[p].object_index,
                        ly.planes[p].offset, ly.planes[p].pitch);
    }
    std::printf("  DrmFrame: %dx%d coded_h=%d\n"
                "    Y : fd=%d size=%zu offset=%zu pitch=%zu sand_col_height=%u (col_stride=%zu B)\n"
                "    UV: fd=%d size=%zu offset=%zu pitch=%zu uv_sand_col_height=%u (col_stride=%zu B)\n",
                f.width, f.height, f.coded_height, f.fd, f.size, f.y_offset, f.y_pitch, f.sand_col_height,
                yvo::kSandColWidth * f.sand_col_height, f.uv_fd, f.uv_size, f.uv_offset, f.uv_pitch,
                f.uv_sand_col_height, yvo::kSandColWidth * f.uv_sand_col_height);
}

// dma-buf CPU mapping (read-only) + DMA_BUF_IOCTL_SYNC
class DmaBufReadMap {
public:
    DmaBufReadMap(int fd, std::size_t size) : fd_(fd), size_(size) {
        void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            std::perror("mmap(dma-buf)");
            return;
        }
        ptr_ = static_cast<const std::uint8_t*>(p);
        dma_buf_sync s{};
        s.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
        if (ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s) != 0) std::perror("DMA_BUF_IOCTL_SYNC start");
    }
    ~DmaBufReadMap() {
        if (!ptr_) return;
        dma_buf_sync s{};
        s.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
        if (ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s) != 0) std::perror("DMA_BUF_IOCTL_SYNC end");
        munmap(const_cast<std::uint8_t*>(ptr_), size_);
    }
    DmaBufReadMap(const DmaBufReadMap&) = delete;
    DmaBufReadMap& operator=(const DmaBufReadMap&) = delete;
    const std::uint8_t* data() const { return ptr_; }

private:
    int fd_;
    std::size_t size_;
    const std::uint8_t* ptr_ = nullptr;
};

bool write_file(const std::string& path, const void* data, std::size_t n, const std::string& header = {}) {
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    o.write(header.data(), static_cast<std::streamsize>(header.size()));
    o.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    return static_cast<bool>(o);
}

// Single-frame CPU de-SAND → save NV12/PGM, byte comparison with the av_hwframe_transfer_data result. Returns 0 on success.
int desand_and_compare(const yvo::DrmFrame& f, const std::string& dir) {
    const int w = f.width, h = f.height;
    const std::size_t ws = static_cast<std::size_t>(w), hs = static_cast<std::size_t>(h);
    std::vector<std::uint8_t> nv12(ws * hs * 3 / 2);
    std::uint8_t* y = nv12.data();
    std::uint8_t* uv = nv12.data() + ws * hs;
    {
        DmaBufReadMap map_y(f.fd, f.size);
        std::unique_ptr<DmaBufReadMap> map_uv_own;
        const DmaBufReadMap* map_uv = &map_y;
        if (f.uv_fd != f.fd) {
            map_uv_own = std::make_unique<DmaBufReadMap>(f.uv_fd, f.uv_size);
            map_uv = map_uv_own.get();
        }
        if (!map_y.data() || !map_uv->data()) return 1;
        const std::size_t uv_size = (f.uv_fd != f.fd) ? f.uv_size : f.size;
        if (!yvo::desand_nv12(map_y.data(), f.size, map_uv->data(), uv_size, yvo::make_sand_layout(f), y, ws, uv,
                              ws)) {
            std::fprintf(stderr, "desand_nv12: layout out of bounds\n");
            return 1;
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string base = dir + "/frame" + std::to_string(f.pts);
    const std::string nv12_path = base + "_" + std::to_string(w) + "x" + std::to_string(h) + ".nv12";
    const std::string pgm_path = base + "_y.pgm";
    if (!write_file(nv12_path, nv12.data(), nv12.size()) ||
        !write_file(pgm_path, y, ws * hs, "P5\n" + std::to_string(w) + " " + std::to_string(h) + "\n255\n")) {
        std::fprintf(stderr, "failed to write output files in %s\n", dir.c_str());
        return 1;
    }
    std::printf("saved %s, %s\n", nv12_path.c_str(), pgm_path.c_str());

    // Reference: av_hwframe_transfer_data (DRM → SW)
    AVFrame* hw = yvo::FfmpegDrmDecoder::av_frame(f);
    AVPixelFormat* fmts = nullptr;
    if (av_hwframe_transfer_get_formats(hw->hw_frames_ctx, AV_HWFRAME_TRANSFER_DIRECTION_FROM, &fmts, 0) >= 0 &&
        fmts) {
        std::printf("transfer formats:");
        for (AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p) std::printf(" %s", av_get_pix_fmt_name(*p));
        std::printf("\n");
        av_free(fmts);
    }
    std::unique_ptr<AVFrame, void (*)(AVFrame*)> sw(av_frame_alloc(), [](AVFrame* p) { av_frame_free(&p); });
    sw->format = AV_PIX_FMT_NV12;
    int ret = av_hwframe_transfer_data(sw.get(), hw, 0);
    if (ret < 0) {
        char e[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror(ret, e, sizeof(e));
        std::fprintf(stderr, "av_hwframe_transfer_data(NV12) failed: %s\n", e);
        return 1;
    }
    std::printf("transfer result: %s %dx%d linesize=%d/%d\n",
                av_get_pix_fmt_name(static_cast<AVPixelFormat>(sw->format)), sw->width, sw->height,
                sw->linesize[0], sw->linesize[1]);
    if (sw->format != AV_PIX_FMT_NV12 || sw->width < w || sw->height < h) {
        std::fprintf(stderr, "unexpected transfer result format/size\n");
        return 1;
    }
    std::size_t diff_y = 0, diff_uv = 0;
    long first_y = -1, first_uv = -1;
    for (std::size_t r = 0; r < hs; ++r)
        for (std::size_t c = 0; c < ws; ++c)
            if (sw->data[0][r * static_cast<std::size_t>(sw->linesize[0]) + c] != y[r * ws + c]) {
                if (first_y < 0) first_y = static_cast<long>(r * ws + c);
                ++diff_y;
            }
    for (std::size_t r = 0; r < hs / 2; ++r)
        for (std::size_t c = 0; c < ws; ++c)
            if (sw->data[1][r * static_cast<std::size_t>(sw->linesize[1]) + c] != uv[r * ws + c]) {
                if (first_uv < 0) first_uv = static_cast<long>(r * ws + c);
                ++diff_uv;
            }
    std::printf("compare desand vs av_hwframe_transfer_data: Y diff=%zu/%zu (first %ld), UV diff=%zu/%zu (first %ld) -> %s\n",
                diff_y, ws * hs, first_y, diff_uv, ws * hs / 2, first_uv,
                (diff_y == 0 && diff_uv == 0) ? "MATCH" : "MISMATCH");
    return (diff_y == 0 && diff_uv == 0) ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        usage(argv[0]);
        return 2;
    }
    const int fds_before = count_open_fds();
    int rc = 0;
    {
        yvo::FfmpegDrmDecoder dec;
        if (!dec.open(a.video, a.dec_opt)) {
            std::fprintf(stderr, "open failed: %s\n", dec.error().c_str());
            return 1;
        }
        std::printf("opened %s: %dx%d %.3f fps, nb_frames=%lld\n", a.video.c_str(), dec.width(), dec.height(),
                    dec.frame_rate(), static_cast<long long>(dec.nb_frames()));

        if (a.count) {
            std::deque<yvo::DrmFrame> held;
            std::set<int> fds;
            long n = 0;
            const double c0 = cpu_ms();
            const auto t0 = std::chrono::steady_clock::now();
            while (a.max_frames <= 0 || n < a.max_frames) {
                auto f = dec.next();
                if (!f) break;
                ++n;
                fds.insert(f->fd);
                if (a.hold > 0) {
                    held.push_back(std::move(*f));
                    if (static_cast<int>(held.size()) > a.hold) held.pop_front();
                }
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double c1 = cpu_ms();
            held.clear();
            if (!dec.error().empty()) {
                std::fprintf(stderr, "decode error after %ld frames: %s\n", n, dec.error().c_str());
                rc = 1;
            }
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            std::printf("count: frames=%ld hold=%d time=%.3f s fps=%.1f cpu=%.1f ms total, %.3f ms/frame, "
                        "distinct dma-buf fds=%zu\n",
                        n, a.hold, sec, n / sec, c1 - c0, (c1 - c0) / static_cast<double>(n), fds.size());
        } else {
            const int want = std::max(a.frames, a.desand_dir.empty() ? 0 : a.desand_frame + 1);
            for (int i = 0; i < want; ++i) {
                auto f = dec.next();
                if (!f) {
                    std::fprintf(stderr, "no frame %d: %s\n", i, dec.error().empty() ? "EOF" : dec.error().c_str());
                    rc = 1;
                    break;
                }
                if (i < a.frames) dump_descriptor(*f);
                if (!a.desand_dir.empty() && i == a.desand_frame) {
                    const int r = desand_and_compare(*f, a.desand_dir);
                    if (r != 0) rc = r;
                }
            }
        }
    }
    const int fds_after = count_open_fds();
    std::printf("open fds: before=%d after=%d%s\n", fds_before, fds_after,
                fds_after > fds_before ? " (LEAK?)" : "");
    return rc;
}
