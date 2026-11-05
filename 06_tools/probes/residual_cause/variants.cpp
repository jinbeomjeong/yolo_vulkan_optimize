// One-factor variants of the reference letterbox loop. Same -O3 -mcpu=cortex-a76 flags as yvo_preproc_bench.
#include "variants.hpp"
#include "ref/cpu_letterbox.hpp"

#if defined(__GNUC__) && __GNUC__ >= 14
#define RC_SCALAR _Pragma("GCC novector")
#else
#define RC_SCALAR
#endif

namespace rc {
using yvo::bilinear_tap;
using yvo::BilinearTap;
using yvo::letterbox_src_coord;
using yvo::detail::bilinear_u8;
using yvo::detail::to_u8;

namespace {
[[gnu::always_inline]] inline void color_px(u8* px, float Y, float U, float V) {
    const float yf = (Y - 16.0f) * (255.0f / 219.0f);
    const float uf = (U - 128.0f) * (255.0f / 224.0f);
    const float vf = (V - 128.0f) * (255.0f / 224.0f);
    px[0] = to_u8(yf + 1.5748f * vf);
    px[1] = to_u8(yf - 0.187324f * uf - 0.468124f * vf);
    px[2] = to_u8(yf + 1.8556f * uf);
}

// Row-level y tap for a plane, used by the hoisted variants.
struct YTap {
    const u8* r0;
    const u8* r1;
    float f;
};
[[gnu::always_inline]] inline YTap y_tap(const u8* plane, std::size_t pitch, float y, int h) {
    const BilinearTap t = bilinear_tap(y, h);
    return {plane + static_cast<std::size_t>(t.i0) * pitch, plane + static_cast<std::size_t>(t.i1) * pitch, t.f};
}
// x tap with the clamp limit given (the float conversion of n-1 is not redone per call).
[[gnu::always_inline]] inline BilinearTap x_tap(float s, float fmax, int nm1) {
    s = s < 0.0f ? 0.0f : (fmax < s ? fmax : s);
    BilinearTap t;
    t.i0 = static_cast<int>(s);
    t.i1 = t.i0 + 1 < nm1 ? t.i0 + 1 : nm1;
    t.f = s - static_cast<float>(t.i0);
    return t;
}
[[gnu::always_inline]] inline float sample(const YTap& yt, const BilinearTap& tx, int stride, int offset) {
    const float a = yt.r0[tx.i0 * stride + offset], b = yt.r0[tx.i1 * stride + offset];
    const float c = yt.r1[tx.i0 * stride + offset], d = yt.r1[tx.i1 * stride + offset];
    const float top = a + (b - a) * tx.f;
    const float bot = c + (d - c) * tx.f;
    return top + (bot - top) * yt.f;
}
} // namespace

// ref with lb fields in locals. Stores through u8* can no longer force reloads.
[[gnu::noinline]] void local_params(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    const int dst_w = lb.dst_w, dst_h = lb.dst_h, pad_x = lb.pad_x, pad_y = lb.pad_y;
    const u8 pad_value = lb.pad_value;
    for (int dy = 0; dy < dst_h; ++dy) {
        u8* row = dst + static_cast<std::size_t>(dy) * dst_w * 3;
        const bool row_in = dy >= pad_y && dy < pad_y + new_h;
        const float sy = letterbox_src_coord(dy, pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        for (int dx = 0; dx < dst_w; ++dx) {
            u8* px = row + dx * 3;
            if (!row_in || dx < pad_x || dx >= pad_x + new_w) {
                px[0] = px[1] = px[2] = pad_value;
                continue;
            }
            const float sx = letterbox_src_coord(dx, pad_x, inv);
            const float cx = sx * 0.5f;
            const float Y = bilinear_u8(y_plane, y_pitch, src_w, src_h, sx, sy);
            const float U = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 0);
            const float V = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 1);
            color_px(px, Y, U, V);
        }
    }
}

// ref + y taps per row; clamp limits per pixel as in ref (bilinear_tap on x), lb reloads and branch as in ref.
[[gnu::noinline]] void hoist_y(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    for (int dy = 0; dy < lb.dst_h; ++dy) {
        u8* row = dst + static_cast<std::size_t>(dy) * lb.dst_w * 3;
        const bool row_in = dy >= lb.pad_y && dy < lb.pad_y + new_h;
        const float sy = letterbox_src_coord(dy, lb.pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        const YTap ty = y_tap(y_plane, y_pitch, sy, src_h);
        const YTap tc = y_tap(uv_plane, uv_pitch, cy, ch);
        for (int dx = 0; dx < lb.dst_w; ++dx) {
            u8* px = row + dx * 3;
            if (!row_in || dx < lb.pad_x || dx >= lb.pad_x + new_w) {
                px[0] = px[1] = px[2] = lb.pad_value;
                continue;
            }
            const float sx = letterbox_src_coord(dx, lb.pad_x, inv);
            const float cx = sx * 0.5f;
            const BilinearTap tx = bilinear_tap(sx, src_w);
            const BilinearTap tcx = bilinear_tap(cx, cw);
            color_px(px, sample(ty, tx, 1, 0), sample(tc, tcx, 2, 0), sample(tc, tcx, 2, 1));
        }
    }
}

[[gnu::noinline]] void hoist_all(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    const float fw = static_cast<float>(src_w - 1), fcw = static_cast<float>(cw - 1);
    const int wm1 = src_w - 1, cwm1 = cw - 1;
    for (int dy = 0; dy < lb.dst_h; ++dy) {
        u8* row = dst + static_cast<std::size_t>(dy) * lb.dst_w * 3;
        const bool row_in = dy >= lb.pad_y && dy < lb.pad_y + new_h;
        const float sy = letterbox_src_coord(dy, lb.pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        const YTap ty = y_tap(y_plane, y_pitch, sy, src_h);
        const YTap tc = y_tap(uv_plane, uv_pitch, cy, ch);
        for (int dx = 0; dx < lb.dst_w; ++dx) {
            u8* px = row + dx * 3;
            if (!row_in || dx < lb.pad_x || dx >= lb.pad_x + new_w) {
                px[0] = px[1] = px[2] = lb.pad_value;
                continue;
            }
            const float sx = letterbox_src_coord(dx, lb.pad_x, inv);
            const float cx = sx * 0.5f;
            const BilinearTap tx = x_tap(sx, fw, wm1);
            const BilinearTap tcx = x_tap(cx, fcw, cwm1);
            color_px(px, sample(ty, tx, 1, 0), sample(tc, tcx, 2, 0), sample(tc, tcx, 2, 1));
        }
    }
}

[[gnu::noinline]] void local_hoist(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    const int dst_w = lb.dst_w, dst_h = lb.dst_h, pad_x = lb.pad_x, pad_y = lb.pad_y;
    const u8 pad_value = lb.pad_value;
    const float fw = static_cast<float>(src_w - 1), fcw = static_cast<float>(cw - 1);
    const int wm1 = src_w - 1, cwm1 = cw - 1;
    for (int dy = 0; dy < dst_h; ++dy) {
        u8* row = dst + static_cast<std::size_t>(dy) * dst_w * 3;
        const bool row_in = dy >= pad_y && dy < pad_y + new_h;
        const float sy = letterbox_src_coord(dy, pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        const YTap ty = y_tap(y_plane, y_pitch, sy, src_h);
        const YTap tc = y_tap(uv_plane, uv_pitch, cy, ch);
        RC_SCALAR
        for (int dx = 0; dx < dst_w; ++dx) {
            u8* px = row + dx * 3;
            if (!row_in || dx < pad_x || dx >= pad_x + new_w) {
                px[0] = px[1] = px[2] = pad_value;
                continue;
            }
            const float sx = letterbox_src_coord(dx, pad_x, inv);
            const float cx = sx * 0.5f;
            const BilinearTap tx = x_tap(sx, fw, wm1);
            const BilinearTap tcx = x_tap(cx, fcw, cwm1);
            color_px(px, sample(ty, tx, 1, 0), sample(tc, tcx, 2, 0), sample(tc, tcx, 2, 1));
        }
    }
}

[[gnu::noinline]] void rowsplit(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    for (int dy = 0; dy < lb.dst_h; ++dy) {
        u8* row = dst + static_cast<std::size_t>(dy) * lb.dst_w * 3;
        if (dy < lb.pad_y || dy >= lb.pad_y + new_h) {
            for (int dx = 0; dx < lb.dst_w; ++dx) {
                u8* px = row + dx * 3;
                px[0] = px[1] = px[2] = lb.pad_value;
            }
            continue;
        }
        const float sy = letterbox_src_coord(dy, lb.pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        for (int dx = 0; dx < lb.dst_w; ++dx) {
            u8* px = row + dx * 3;
            if (dx < lb.pad_x || dx >= lb.pad_x + new_w) {
                px[0] = px[1] = px[2] = lb.pad_value;
                continue;
            }
            const float sx = letterbox_src_coord(dx, lb.pad_x, inv);
            const float cx = sx * 0.5f;
            const float Y = bilinear_u8(y_plane, y_pitch, src_w, src_h, sx, sy);
            const float U = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 0);
            const float V = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 1);
            color_px(px, Y, U, V);
        }
    }
}

// One loop, no float buffers, no pad test. Content rows only. Needs pad_x == 0 (checked by the caller).
[[gnu::noinline]] void nobranch_content(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    const int dst_w = lb.dst_w, pad_x = lb.pad_x, pad_y = lb.pad_y;
    for (int r = 0; r < new_h; ++r) {
        const int dy = pad_y + r;
        u8* row = dst + (static_cast<std::size_t>(dy) * dst_w + pad_x) * 3;
        const float sy = letterbox_src_coord(dy, pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        RC_SCALAR
        for (int c = 0; c < new_w; ++c) {
            const float sx = letterbox_src_coord(c, 0, inv);
            const float cx = sx * 0.5f;
            const float Y = bilinear_u8(y_plane, y_pitch, src_w, src_h, sx, sy);
            const float U = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 0);
            const float V = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 1);
            color_px(row + c * 3, Y, U, V);
        }
    }
}

// nobranch_content, but the asm barriers make the y taps and clamp limits look different every pixel, so nothing is
// hoisted out of the pixel loop. Same arithmetic per pixel as the ref hot path, minus the pad test and lb reloads.
[[gnu::noinline]] void nobranch_nohoist_content(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    const int dst_w = lb.dst_w, pad_x = lb.pad_x, pad_y = lb.pad_y;
    for (int r = 0; r < new_h; ++r) {
        const int dy = pad_y + r;
        u8* row = dst + (static_cast<std::size_t>(dy) * dst_w + pad_x) * 3;
        const float sy = letterbox_src_coord(dy, pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        RC_SCALAR
        for (int c = 0; c < new_w; ++c) {
            float sy2 = sy, cy2 = cy;
            int w2 = src_w, h2 = src_h, cw2 = cw, ch2 = ch;
            asm volatile("" : "+w"(sy2), "+w"(cy2), "+r"(w2), "+r"(h2), "+r"(cw2), "+r"(ch2));
            const float sx = letterbox_src_coord(c, 0, inv);
            const float cx = sx * 0.5f;
            const float Y = bilinear_u8(y_plane, y_pitch, w2, h2, sx, sy2);
            const float U = bilinear_u8(uv_plane, uv_pitch, cw2, ch2, cx, cy2, 2, 0);
            const float V = bilinear_u8(uv_plane, uv_pitch, cw2, ch2, cx, cy2, 2, 1);
            color_px(row + c * 3, Y, U, V);
        }
    }
}

// split_resize, one row at a time.
[[gnu::noinline]] static void rb_resize(const u8* y_plane, std::size_t y_pitch, const u8* uv_plane, std::size_t uv_pitch,
                                        int src_w, int src_h, float sy, float cy, float inv, int new_w, float* out_y,
                                        float* out_u, float* out_v) {
    const int cw = src_w / 2, ch = src_h / 2;
    RC_SCALAR
    for (int c = 0; c < new_w; ++c) {
        const float sx = letterbox_src_coord(c, 0, inv);
        const float cx = sx * 0.5f;
        out_y[c] = bilinear_u8(y_plane, y_pitch, src_w, src_h, sx, sy);
        out_u[c] = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 0);
        out_v[c] = bilinear_u8(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 1);
    }
}
[[gnu::noinline]] static void rb_color(const float* in_y, const float* in_u, const float* in_v, int new_w, u8* px) {
    RC_SCALAR
    for (int c = 0; c < new_w; ++c) color_px(px + c * 3, in_y[c], in_u[c], in_v[c]);
}
[[gnu::noinline]] void rowbuf_content(RC_ARGS) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    alignas(64) float by[4096], bu[4096], bv[4096];
    for (int r = 0; r < new_h; ++r) {
        const int dy = lb.pad_y + r;
        const float sy = letterbox_src_coord(dy, lb.pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        rb_resize(y_plane, y_pitch, uv_plane, uv_pitch, src_w, src_h, sy, cy, inv, new_w, by, bu, bv);
        rb_color(by, bu, bv, new_w, dst + (static_cast<std::size_t>(dy) * lb.dst_w + lb.pad_x) * 3);
    }
}

// split_resize with nothing hoisted (reverse direction of the hoisting experiment).
[[gnu::noinline]] void split_nohoist_resize(const u8* y_plane, std::size_t y_pitch, const u8* uv_plane,
                                            std::size_t uv_pitch, int src_w, int src_h, const LetterboxParams& lb,
                                            float* out_y, float* out_u, float* out_v) {
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;
    for (int dy = lb.pad_y; dy < lb.pad_y + new_h; ++dy) {
        const float sy = letterbox_src_coord(dy, lb.pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        const std::size_t row = static_cast<std::size_t>(dy - lb.pad_y) * new_w;
        RC_SCALAR
        for (int c = 0; c < new_w; ++c) {
            float sy2 = sy, cy2 = cy;
            int w2 = src_w, h2 = src_h, cw2 = cw, ch2 = ch;
            asm volatile("" : "+w"(sy2), "+w"(cy2), "+r"(w2), "+r"(h2), "+r"(cw2), "+r"(ch2));
            const float sx = letterbox_src_coord(lb.pad_x + c, lb.pad_x, inv);
            const float cx = sx * 0.5f;
            out_y[row + c] = bilinear_u8(y_plane, y_pitch, w2, h2, sx, sy2);
            out_u[row + c] = bilinear_u8(uv_plane, uv_pitch, cw2, ch2, cx, cy2, 2, 0);
            out_v[row + c] = bilinear_u8(uv_plane, uv_pitch, cw2, ch2, cx, cy2, 2, 1);
        }
    }
}
} // namespace rc
