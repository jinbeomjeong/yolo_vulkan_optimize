// Overlay (boxes, labels, HUD) rendering rules and CPU reference implementation — header-only, no OpenCV needed.
//
// Follows the convention of `01_src/ref/cpu_letterbox.hpp`: **the formulas and boundary rules are pinned down in the top comment,
// and the GPU shader (`02_shaders/overlay_boxes.comp`) refers to this comment.** The implementation below is merely a mechanical
// translation of that comment, and `05_tests/test_overlay.cpp` compares it against the GPU result with **max error 0**.
//
// Plan Appendix A.12 is why this file exists: the P5 test pattern drew 399x299 instead of the expected 400x300
// (inclusive/exclusive off-by-one). The overlay draws box borders, so the same mistake is an immediate test failure.
// **Write the rules down first and make GPU and CPU match the text. Doing it in the opposite order breaks bit-exactness.**
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════╗
// ║  Boundary rules (this block is the only reference. GPU and CPU must not deviate from it by a single character)  ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════╝
//
// R0. Coordinate system — affine absorption
//     `Detection` is already in **source 1920x1080 coordinates** (`hailo_runner.cpp:55-81` does the inverse transform).
//     The transform to canvas (swapchain image) coordinates is done only by the single push-constant affine:
//         fx = x * sx + tx,   fy = y * sy + ty      (all float32)
//     If the canvas is 1920x1080 it is the **identity** (sx=sy=1, tx=ty=0); if smaller, the same uniform scale + centering
//     as `make_letterbox()` applies (make_overlay_affine()). Even when switching to GPU NMS,
//     using normalized coordinates only requires changing the affine, so **shader changes are zero.**
//
// R1. float → integer pixel : **round (round half up), then inclusive range**
//         i = int(floor(clamp(f, -32768, 32767) + 0.5))
//     The clamp prevents undefined behavior in the int conversion (GLSL int(float) is undefined when out of range).
//     The box's integer rectangle is **a closed range including all four edges** [ix1, ix2] x [iy1, iy2].
//     Hence pixel width = ix2 - ix1 + 1. The "399 or 400" question is closed explicitly here.
//     Flipped input such as x1 > x2 is normalized with min/max after the transform.
//
// R2. NaN / Inf : if any of the four coordinates is not finite, **skip that detection entirely** (draw nothing).
//     It does not happen in practice, but without a rule the GPU's clamp(NaN) result is implementation-defined and breaks bit-exactness.
//
// R3. Border thickness : **grows inward only** (w >= 1, default 2~3 px at 1080p).
//     Pixel (x, y) is a border pixel if it lies inside the rectangle (the closed range of R1) and satisfies:
//         (x - ix1 < w) || (ix2 - x < w) || (y - iy1 < w) || (iy2 - y < w)
//     Since it does not grow outward, ① borders of boxes touching the screen edge are not cut off, and
//     ② a box does not spill into neighboring pixels, so clipping reduces to a single rectangle clip.
//     If the rectangle is thinner than 2w, the condition above is true everywhere and **the rectangle is filled** (natural degeneration).
//
// R4. Zero area : if x1 == x2 (or y1 == y2), the integer rectangle is **1 px** wide (or tall),
//     and by R3 that 1 px line is drawn. The detection is not dropped.
//     If it is exactly one point (x1==x2 && y1==y2), it is a single 1 px dot.
//
// R5. Clipping : the integer rectangle is **computed from unclipped coordinates** (so the border sits on the true edges),
//     and is clipped to the screen [0, dst_w-1] x [0, dst_h-1] right before the actual write.
//     Entirely off-screen writes no pixels. Partially off-screen leaves only the visible edges.
//
// R6. Label text anchor
//     string = "<class_name> <score>", with score formatted to two digits **in the integer domain**:
//         q = uint(floor(clamp(score, 0, 1) * 100 + 0.5))      // 0..100 (a non-finite score is 0)
//         chars = ['0' + q/100, '.', '0' + (q/10)%10, '0' + q%10]   → "0.87", "1.00"
//     The only safe way to format floats identically on both GPU and CPU (no printf).
//     Origin (top-left of the glyph):
//         tx0 = max(ix1, 0)
//         ty0 = (iy1 - 16*s >= 0) ? (iy1 - 16*s) : max(iy1, 0)      // s = text_scale
//     i.e. **by default just above the box's top-left**, and if there are not 16*s pixels above, move it down to **the inside top of the box**.
//     Overflow to the right is not cut; it is left to R5's screen clipping (no horizontal relayout).
//     Glyph rendering itself (row/bit order, integer scale, advance 8*s, line height 16*s, background untouched)
//     follows rules 1)~8) of `01_src/ref/glyph_font_8x16.hpp` as is.
//     **No label background box is drawn** — glyph rule 5) says "the caller does the background separately", and
//     filling backgrounds for 80 detections at 1080p would quadruple the pixel count and threaten P7's 0.7 ms budget.
//
// R7. Color : both box border and label use `yvo::class_color(cid)` (RGB). Alpha is **always 255**.
//     The class id is first clamped to [0, 79] (color and name both use the same cid).
//     The single HUD line is white (255,255,255) with a fixed origin (kHudOriginX, kHudOriginY) = (4, 4).
//
// R8. Drawing order : ascending detection index (CPU). **The GPU uses one workgroup per detection, so there is no order.**
//     → **The color of intersection pixels of overlapping boxes is order-dependent** (documented as an accepted limit. plan §7, §9).
//     Within one detection the border and label have the same color, so overlaps give the same result (idempotent).
//     On the GPU, workgroup 0 draws the HUD **together** with its own detection → boxes overlapping the HUD area
//     are also order-dependent. Tests and probes keep that area empty.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "core/coco_labels.hpp"
#include "core/frame_types.hpp"
#include "ref/glyph_font_8x16.hpp"

namespace yvo::display {

// R7: HUD origin (must equal the shader's kHudOriginX/Y)
inline constexpr int kHudOriginX = 4;
inline constexpr int kHudOriginY = 4;
// HUD string capacity of the DetBuf header (uint text[32] = 128 B)
inline constexpr std::size_t kHudMaxBytes = 128;
// Maximum label string length: longest COCO name (13) + " 0.00" (5)
inline constexpr int kMaxLabelChars = 18;

// R0: source coordinates → canvas coordinates affine
struct OverlayAffine {
    float sx = 1.0f, sy = 1.0f, tx = 0.0f, ty = 0.0f;
};

// Identity if the canvas equals the source; if smaller, the same uniform scale + centering as make_letterbox().
inline OverlayAffine make_overlay_affine(int src_w, int src_h, int dst_w, int dst_h) {
    OverlayAffine a;
    if (src_w == dst_w && src_h == dst_h) return a; // identity (display at source resolution = default path)
    const LetterboxParams lb = make_letterbox(src_w, src_h, dst_w, dst_h);
    a.sx = a.sy = lb.scale;
    a.tx = static_cast<float>(lb.pad_x);
    a.ty = static_cast<float>(lb.pad_y);
    return a;
}

struct OverlayStyle {
    int line_width = 2;   // R3. 1080p default 2
    int text_scale = 1;   // R6. 1080p default 2 (set by the host's OverlayConfig)
    bool draw_text = true;
};

// R1: float → integer pixel
inline int overlay_round(float v) {
    const float c = std::clamp(v, -32768.0f, 32767.0f);
    return static_cast<int>(std::floor(c + 0.5f));
}

// R1's closed (inclusive) integer rectangle
struct OverlayRect {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

inline bool overlay_finite(float v) { return std::isfinite(v); }

// R0 + R1 + R2. false if coordinates are not finite (that detection is skipped entirely).
inline bool overlay_rect_of(const Detection& d, const OverlayAffine& a, OverlayRect* out) {
    if (!overlay_finite(d.x1) || !overlay_finite(d.y1) || !overlay_finite(d.x2) || !overlay_finite(d.y2))
        return false;
    const float fx1 = d.x1 * a.sx + a.tx;
    const float fy1 = d.y1 * a.sy + a.ty;
    const float fx2 = d.x2 * a.sx + a.tx;
    const float fy2 = d.y2 * a.sy + a.ty;
    if (!overlay_finite(fx1) || !overlay_finite(fy1) || !overlay_finite(fx2) || !overlay_finite(fy2)) return false;
    out->x1 = overlay_round(std::min(fx1, fx2));
    out->x2 = overlay_round(std::max(fx1, fx2));
    out->y1 = overlay_round(std::min(fy1, fy2));
    out->y2 = overlay_round(std::max(fy1, fy2));
    return true;
}

// R7: class id used for color and name
inline int overlay_class_id(int class_id) { return std::clamp(class_id, 0, kNumCocoClasses - 1); }

// R6: score to two integer digits (same operation order as the GPU)
inline unsigned overlay_score_q(float score) {
    const float s = std::isfinite(score) ? std::clamp(score, 0.0f, 1.0f) : 0.0f;
    return static_cast<unsigned>(std::floor(s * 100.0f + 0.5f));
}

// R6: label string. The GPU builds the characters one by one with the same rule (it does not build a string).
inline std::string overlay_label(int class_id, float score) {
    const unsigned q = overlay_score_q(score);
    std::string s = class_name(overlay_class_id(class_id));
    s.push_back(' ');
    s.push_back(static_cast<char>('0' + q / 100));
    s.push_back('.');
    s.push_back(static_cast<char>('0' + (q / 10) % 10));
    s.push_back(static_cast<char>('0' + q % 10));
    return s;
}

// ─────────────────────────── CPU reference renderer ───────────────────────────
// rgba: w*h*4, R8G8B8A8_UNORM. Draws over the existing content (same as the GPU).

namespace detail {

inline void put_px(std::uint8_t* rgba, int w, int h, int x, int y, const std::array<std::uint8_t, 3>& c) {
    if (x < 0 || y < 0 || x >= w || y >= h) return; // R5
    std::uint8_t* p = rgba + (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
    p[0] = c[0];
    p[1] = c[1];
    p[2] = c[2];
    p[3] = 255; // R7
}

// R5: clip to the screen, then fill (inclusive range)
inline void fill_rect(std::uint8_t* rgba, int w, int h, int x0, int y0, int x1, int y1,
                      const std::array<std::uint8_t, 3>& c) {
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, w - 1);
    y1 = std::min(y1, h - 1);
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) put_px(rgba, w, h, x, y, c);
}

// Glyph rules 1)~8) + R6. Stamps each character of text starting at (x0, y0) with an 8*s stride.
inline void draw_glyphs(std::uint8_t* rgba, int w, int h, int x0, int y0, std::string_view text, int s,
                        const std::array<std::uint8_t, 3>& c) {
    for (std::size_t ci = 0; ci < text.size(); ++ci) {
        const int g = glyph_index(static_cast<unsigned char>(text[ci]));
        for (int row = 0; row < kGlyphHeight; ++row) {
            for (int col = 0; col < kGlyphWidth; ++col) {
                if (!glyph_pixel(g, col, row)) continue; // rule 5): pixels that are off are left untouched
                const int px0 = x0 + static_cast<int>(ci) * kGlyphWidth * s + col * s;
                const int py0 = y0 + row * s;
                for (int j = 0; j < s; ++j)
                    for (int i = 0; i < s; ++i) put_px(rgba, w, h, px0 + i, py0 + j, c);
            }
        }
    }
}

} // namespace detail

// Draws one detection (border + label). Does the same work as one GPU workgroup.
inline void overlay_cpu_draw_detection(std::uint8_t* rgba, int w, int h, const Detection& d,
                                       const OverlayAffine& a, const OverlayStyle& st) {
    OverlayRect r{};
    if (!overlay_rect_of(d, a, &r)) return; // R2
    const int cid = overlay_class_id(d.class_id);
    const std::array<std::uint8_t, 3> col = class_color(cid);
    const int lw = std::max(st.line_width, 1);

    // R3: four edges growing inward. The union of the four bands is exactly the set given by R3's condition.
    // Overlapping corner pixels written twice with the same color give the same result (idempotent).
    detail::fill_rect(rgba, w, h, r.x1, r.y1, r.x2, std::min(r.y1 + lw - 1, r.y2), col);                 // top
    detail::fill_rect(rgba, w, h, r.x1, std::max(r.y2 - lw + 1, r.y1), r.x2, r.y2, col);                 // bottom
    detail::fill_rect(rgba, w, h, r.x1, r.y1, std::min(r.x1 + lw - 1, r.x2), r.y2, col);                 // left
    detail::fill_rect(rgba, w, h, std::max(r.x2 - lw + 1, r.x1), r.y1, r.x2, r.y2, col);                 // right

    if (!st.draw_text) return;
    const int s = std::max(st.text_scale, 1);
    const int tx0 = std::max(r.x1, 0);                                        // R6
    const int ty0 = (r.y1 - kGlyphHeight * s >= 0) ? (r.y1 - kGlyphHeight * s) : std::max(r.y1, 0);
    detail::draw_glyphs(rgba, w, h, tx0, ty0, overlay_label(d.class_id, d.score), s, col);
}

// Single HUD line (R7: white, fixed origin)
inline void overlay_cpu_draw_hud(std::uint8_t* rgba, int w, int h, std::string_view hud, const OverlayStyle& st) {
    if (!st.draw_text || hud.empty()) return;
    const std::array<std::uint8_t, 3> white{255, 255, 255};
    detail::draw_glyphs(rgba, w, h, kHudOriginX, kHudOriginY, hud.substr(0, kHudMaxBytes),
                        std::max(st.text_scale, 1), white);
}

// Full reference render (R8: ascending detection index, HUD last)
inline void overlay_cpu_render(std::uint8_t* rgba, int w, int h, const std::vector<Detection>& dets,
                               std::string_view hud, const OverlayAffine& a, const OverlayStyle& st,
                               std::size_t max_dets = 80) {
    const std::size_t n = std::min(dets.size(), max_dets);
    for (std::size_t i = 0; i < n; ++i) overlay_cpu_draw_detection(rgba, w, h, dets[i], a, st);
    overlay_cpu_draw_hud(rgba, w, h, hud, st);
}

} // namespace yvo::display
