// Verbatim copy of the loop in 01_src/ref/cpu_letterbox.hpp:73-105 under another name. Helpers come from a private
// namespace so a TU built with different flags cannot merge COMDAT copies with the reference one.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "variants.hpp"

namespace {
inline float letterbox_src_coord_c(int d, int pad, float inv) { return (static_cast<float>(d - pad) + 0.5f) * inv - 0.5f; }
struct BilinearTapC { int i0 = 0, i1 = 0; float f = 0.0f; };
inline BilinearTapC bilinear_tap_c(float s, int n) {
    s = std::clamp(s, 0.0f, static_cast<float>(n - 1));
    BilinearTapC t;
    t.i0 = static_cast<int>(s);
    t.i1 = std::min(t.i0 + 1, n - 1);
    t.f = s - static_cast<float>(t.i0);
    return t;
}
inline float bilinear_u8_c(const std::uint8_t* plane, std::size_t pitch, int w, int h, float x, float y, int stride = 1, int offset = 0) {
    const BilinearTapC tx = bilinear_tap_c(x, w);
    const BilinearTapC ty = bilinear_tap_c(y, h);
    const std::uint8_t* r0 = plane + static_cast<std::size_t>(ty.i0) * pitch;
    const std::uint8_t* r1 = plane + static_cast<std::size_t>(ty.i1) * pitch;
    const float a = r0[tx.i0 * stride + offset], b = r0[tx.i1 * stride + offset];
    const float c = r1[tx.i0 * stride + offset], d = r1[tx.i1 * stride + offset];
    const float top = a + (b - a) * tx.f;
    const float bot = c + (d - c) * tx.f;
    return top + (bot - top) * ty.f;
}
inline std::uint8_t to_u8_c(float v) {
    v = std::clamp(v, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(v + 0.5f);
}
} // namespace

namespace rc {
RC_FN_ATTR void RC_FN_NAME(RC_ARGS) {
    RC_ENTRY_PAD
    const int new_w = static_cast<int>(src_w * lb.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * lb.scale + 0.5f);
    const float inv = 1.0f / lb.scale;
    const int cw = src_w / 2, ch = src_h / 2;

    for (int dy = 0; dy < lb.dst_h; ++dy) {
        std::uint8_t* row = dst + static_cast<std::size_t>(dy) * lb.dst_w * 3;
        const bool row_in = dy >= lb.pad_y && dy < lb.pad_y + new_h;
        const float sy = letterbox_src_coord_c(dy, lb.pad_y, inv);
        const float cy = (sy + 0.5f) * 0.5f - 0.5f;
        for (int dx = 0; dx < lb.dst_w; ++dx) {
            std::uint8_t* px = row + dx * 3;
            if (!row_in || dx < lb.pad_x || dx >= lb.pad_x + new_w) {
                px[0] = px[1] = px[2] = lb.pad_value;
                continue;
            }
            const float sx = letterbox_src_coord_c(dx, lb.pad_x, inv);
            const float cx = sx * 0.5f;
            const float Y = bilinear_u8_c(y_plane, y_pitch, src_w, src_h, sx, sy);
            const float U = bilinear_u8_c(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 0);
            const float V = bilinear_u8_c(uv_plane, uv_pitch, cw, ch, cx, cy, 2, 1);
            const float yf = (Y - 16.0f) * (255.0f / 219.0f);
            const float uf = (U - 128.0f) * (255.0f / 224.0f);
            const float vf = (V - 128.0f) * (255.0f / 224.0f);
            px[0] = to_u8_c(yf + 1.5748f * vf);
            px[1] = to_u8_c(yf - 0.187324f * uf - 0.468124f * vf);
            px[2] = to_u8_c(yf + 1.8556f * uf);
        }
    }
}
} // namespace rc
