// CPU overlay painter: draws boxes, labels, and HUD into a transparent RGBA buffer (memory byte order R,G,B,A = DRM AB24).
// Drawing rules reuse the main body's CPU reference implementation src/display/overlay_reference.hpp as-is (= bit-exact with overlay_boxes.comp).
//
// Cost-saving trick: do not clear 1920x1080x4 (8.3 MB) every frame. Each buffer remembers a list of "rectangles drawn last time"
// and clears only those rectangles to 0 (fully transparent, also 0 in premultiplied).
// Pixels are either alpha 255 (opaque) or (0,0,0,0), so straight/premultiplied interpretation gives the same result.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/glyph_font_8x16.hpp"
#include "display/overlay_reference.hpp"

namespace hvsprobe {

struct DirtyRect {
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1; // inclusive range
};

class OverlayPainter {
public:
    OverlayPainter(int w, int h, int nbufs) : w_(w), h_(h), dirty_(static_cast<std::size_t>(nbufs)) {
        affine_ = yvo::display::make_overlay_affine(1920, 1080, w, h);
        style_.line_width = w >= 1920 ? 2 : 2;
        style_.text_scale = w >= 1920 ? 2 : 1;
    }

    // buf: packed w*h*4 (stride = w*4) buffer. idx: ring index.
    // returns: estimated number of bytes touched (cleared + drawn) this time
    std::size_t paint(std::uint8_t* buf, int idx, const std::vector<yvo::Detection>& dets, const std::string& hud) {
        auto& d = dirty_.at(static_cast<std::size_t>(idx));
        std::size_t touched = 0;
        for (const auto& r : d) touched += clear(buf, r);
        d.clear();
        const std::size_t n = std::min<std::size_t>(dets.size(), 80);
        const int s = std::max(style_.text_scale, 1);
        for (std::size_t i = 0; i < n; ++i) {
            yvo::display::OverlayRect r{};
            if (!yvo::display::overlay_rect_of(dets[i], affine_, &r)) continue;
            d.push_back(DirtyRect{r.x1, r.y1, r.x2, r.y2});
            const int tx0 = std::max(r.x1, 0);
            const int ty0 = (r.y1 - yvo::kGlyphHeight * s >= 0) ? (r.y1 - yvo::kGlyphHeight * s) : std::max(r.y1, 0);
            const int len = static_cast<int>(yvo::display::overlay_label(dets[i].class_id, dets[i].score).size());
            d.push_back(DirtyRect{tx0, ty0, tx0 + len * yvo::kGlyphWidth * s - 1, ty0 + yvo::kGlyphHeight * s - 1});
        }
        if (!hud.empty()) {
            const int len = static_cast<int>(std::min<std::size_t>(hud.size(), yvo::display::kHudMaxBytes));
            d.push_back(DirtyRect{yvo::display::kHudOriginX, yvo::display::kHudOriginY,
                                  yvo::display::kHudOriginX + len * yvo::kGlyphWidth * s - 1,
                                  yvo::display::kHudOriginY + yvo::kGlyphHeight * s - 1});
        }
        yvo::display::overlay_cpu_render(buf, w_, h_, dets, hud, affine_, style_);
        for (const auto& r : d) touched += area(r) * 4;
        return touched;
    }

    int width() const { return w_; }
    int height() const { return h_; }

private:
    std::size_t area(const DirtyRect& r) const {
        const int x0 = std::max(r.x0, 0), y0 = std::max(r.y0, 0);
        const int x1 = std::min(r.x1, w_ - 1), y1 = std::min(r.y1, h_ - 1);
        if (x1 < x0 || y1 < y0) return 0;
        return static_cast<std::size_t>(x1 - x0 + 1) * static_cast<std::size_t>(y1 - y0 + 1);
    }

    std::size_t clear(std::uint8_t* buf, const DirtyRect& r) const {
        const int x0 = std::max(r.x0, 0), y0 = std::max(r.y0, 0);
        const int x1 = std::min(r.x1, w_ - 1), y1 = std::min(r.y1, h_ - 1);
        if (x1 < x0 || y1 < y0) return 0;
        const std::size_t row_bytes = static_cast<std::size_t>(x1 - x0 + 1) * 4;
        for (int y = y0; y <= y1; ++y)
            std::memset(buf + (static_cast<std::size_t>(y) * static_cast<std::size_t>(w_) + static_cast<std::size_t>(x0)) * 4, 0,
                        row_bytes);
        return row_bytes * static_cast<std::size_t>(y1 - y0 + 1);
    }

    int w_, h_;
    yvo::display::OverlayAffine affine_;
    yvo::display::OverlayStyle style_;
    std::vector<std::vector<DirtyRect>> dirty_;
};

} // namespace hvsprobe
