#pragma once
// FFmpeg libswscale letterbox resize + rpi-ffmpeg SAND -> planar helpers (tools only, header-only).
// Used by yvo_preproc_bench --impl sws-rgb24 | sws-gbrp (2026-10-27). Include only when YVO_HAVE_SWSCALE is defined
// (libswscale is an optional dependency, 01_src/tools/CMakeLists.txt).
//
//   sand_to_yuv420p_rows()  av_rpi_sand_to_planar_y8 / _c8 (rpi-ffmpeg libavutil) on a band of luma rows
//   SwsResize               YUV420P (BT.709 limited, chroma left) -> RGB (full range) at the content size,
//                           driven through sws_scale_frame so the "threads" option (slice threads) takes effect
#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rpi_sand_fns.h>
#include <libswscale/swscale.h>
}

namespace yvo::tool {

// SAND128 NV12 -> YUV420P for luma rows [r0, r1) (r0, r1 even). y_base/uv_base point at the plane starts
// (buffer + offset); y_colh/uv_colh are the SAND column heights in rows (stride2 of rpi_sand_fns.h).
// rpi_sand_fns.h: x/width are in bytes for y8 and in chroma pels for c8, strides in bytes.
// The band start is applied by moving the source pointer by r0 rows (r0 * 128 bytes inside every column) and passing
// y = 0: the NEON path taken for x == 0 (ff_rpi_sand8_lines_to_planar_*) ignored a non-zero y here (2026-10-27,
// every band came out as rows 0..), while the column addressing (col * stride1 * stride2) is unchanged by the shift.
inline void sand_to_yuv420p_rows(const std::uint8_t* y_base, unsigned y_colh, const std::uint8_t* uv_base,
                                 unsigned uv_colh, int w, int r0, int r1, std::uint8_t* dst_y, std::uint8_t* dst_u,
                                 std::uint8_t* dst_v) {
    if (r0 >= r1) return;
    const unsigned uw = static_cast<unsigned>(w);
    const std::size_t c0 = static_cast<std::size_t>(r0 / 2);
    av_rpi_sand_to_planar_y8(dst_y + static_cast<std::size_t>(r0) * uw, uw,
                             y_base + static_cast<std::size_t>(r0) * 128, 128, y_colh, 0, 0, uw, static_cast<unsigned>(r1 - r0));
    av_rpi_sand_to_planar_c8(dst_u + c0 * (uw / 2), uw / 2, dst_v + c0 * (uw / 2), uw / 2, uv_base + c0 * 128, 128,
                             uv_colh, 0, 0, uw / 2, static_cast<unsigned>((r1 - r0) / 2));
}

class SwsResize {
public:
    // flags: sws_flags option string (e.g. "bilinear", "bilinear+full_chroma_int"). threads: swscale slice threads.
    SwsResize(int src_w, int src_h, int dst_w, int dst_h, AVPixelFormat dst_fmt, const std::string& flags, int threads)
        : src_w_(src_w), src_h_(src_h), dst_w_(dst_w), dst_h_(dst_h), dst_fmt_(dst_fmt) {
        ctx_ = sws_alloc_context();
        if (!ctx_) throw std::runtime_error("sws_alloc_context failed");
        check(av_opt_set_int(ctx_, "srcw", src_w, 0), "srcw");
        check(av_opt_set_int(ctx_, "srch", src_h, 0), "srch");
        check(av_opt_set_int(ctx_, "src_format", AV_PIX_FMT_YUV420P, 0), "src_format");
        check(av_opt_set_int(ctx_, "dstw", dst_w, 0), "dstw");
        check(av_opt_set_int(ctx_, "dsth", dst_h, 0), "dsth");
        check(av_opt_set_int(ctx_, "dst_format", dst_fmt, 0), "dst_format");
        check(av_opt_set(ctx_, "sws_flags", flags.c_str(), 0), "sws_flags");
        check(av_opt_set_int(ctx_, "threads", threads, 0), "threads");
        // Chroma siting of HEVC 4:2:0 "left" (ref/cpu_letterbox.hpp formula 4): horizontally co-sited with even luma
        // columns (0/256), vertically half way between two luma rows (128/256).
        check(av_opt_set_int(ctx_, "src_h_chr_pos", 0, 0), "src_h_chr_pos");
        check(av_opt_set_int(ctx_, "src_v_chr_pos", 128, 0), "src_v_chr_pos");
        check(av_opt_set_int(ctx_, "src_range", 0, 0), "src_range");
        check(av_opt_set_int(ctx_, "dst_range", 1, 0), "dst_range");
        if (sws_init_context(ctx_, nullptr, nullptr) < 0) throw std::runtime_error("sws_init_context failed");
        // BT.709 limited -> full-range RGB (after init: init resets the tables; propagated to slice contexts)
        if (sws_setColorspaceDetails(ctx_, sws_getCoefficients(SWS_CS_ITU709), 0, sws_getCoefficients(SWS_CS_ITU709),
                                     1, 0, 1 << 16, 1 << 16) < 0)
            throw std::runtime_error("sws_setColorspaceDetails(BT.709 limited -> full) not supported");
        // sws_scale_frame refs the frames; a frame without buf[0] would be deep-copied. Wrap caller memory with a
        // dummy, never-freeing buffer reference instead (the data pointers are copied as is).
        dummy_ = av_buffer_create(&dummy_byte_, 1, [](void*, std::uint8_t*) {}, nullptr, AV_BUFFER_FLAG_READONLY);
        src_ = av_frame_alloc();
        dst_ = av_frame_alloc();
        if (!dummy_ || !src_ || !dst_) throw std::runtime_error("SwsResize: allocation failed");
        src_->buf[0] = av_buffer_ref(dummy_);
        dst_->buf[0] = av_buffer_ref(dummy_);
        src_->format = AV_PIX_FMT_YUV420P;
        src_->width = src_w;
        src_->height = src_h;
        dst_->format = dst_fmt;
        dst_->width = dst_w;
        dst_->height = dst_h;
    }
    ~SwsResize() {
        av_frame_free(&src_);
        av_frame_free(&dst_);
        av_buffer_unref(&dummy_);
        sws_freeContext(ctx_);
    }
    SwsResize(const SwsResize&) = delete;
    SwsResize& operator=(const SwsResize&) = delete;

    // src: Y, U, V planes. dst: up to 3 planes (packed formats use plane 0; GBRP order is G, B, R).
    void scale(const std::uint8_t* const src[3], const int src_stride[3], std::uint8_t* const dst[3],
               const int dst_stride[3]) {
        for (int i = 0; i < 3; ++i) {
            src_->data[i] = const_cast<std::uint8_t*>(src[i]);
            src_->linesize[i] = src_stride[i];
            dst_->data[i] = dst[i];
            dst_->linesize[i] = dst_stride[i];
        }
        const int r = sws_scale_frame(ctx_, dst_, src_);
        if (r < 0) throw std::runtime_error("sws_scale_frame failed: " + std::to_string(r));
    }

    int threads() const {
        std::int64_t v = 0;
        av_opt_get_int(ctx_, "threads", 0, &v);
        return static_cast<int>(v);
    }
    std::string flags() const {
        std::uint8_t* s = nullptr;
        if (av_opt_get(ctx_, "sws_flags", 0, &s) < 0 || !s) return "?";
        std::string r(reinterpret_cast<char*>(s));
        av_free(s);
        return r;
    }

private:
    static void check(int r, const char* what) {
        if (r < 0) throw std::runtime_error(std::string("swscale option ") + what + " failed: " + std::to_string(r));
    }

    int src_w_, src_h_, dst_w_, dst_h_;
    AVPixelFormat dst_fmt_;
    SwsContext* ctx_ = nullptr;
    AVBufferRef* dummy_ = nullptr;
    AVFrame* src_ = nullptr;
    AVFrame* dst_ = nullptr;
    std::uint8_t dummy_byte_ = 0;
};

} // namespace yvo::tool
