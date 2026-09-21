// CPU 오버레이 페인터: 박스·라벨·HUD 를 투명 RGBA 버퍼(메모리 바이트 순서 R,G,B,A = DRM AB24)에 그린다.
// 그리기 규약은 본체의 CPU 기준 구현 src/display/overlay_reference.hpp 를 그대로 쓴다(= overlay_boxes.comp 와 비트 일치).
//
// 비용을 줄이는 요령: 매 프레임 1920x1080x4(8.3 MB)를 지우지 않는다. 버퍼마다 "지난번에 그린 사각형" 목록을
// 기억해 두었다가 그 사각형만 0(완전 투명, premultiplied 에서도 0)으로 지운다.
// 픽셀은 알파 255(불투명) 아니면 (0,0,0,0) 이므로 straight/premultiplied 어느 쪽으로 해석해도 같다.
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
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1; // 포함 범위
};

class OverlayPainter {
public:
    OverlayPainter(int w, int h, int nbufs) : w_(w), h_(h), dirty_(static_cast<std::size_t>(nbufs)) {
        affine_ = yvo::display::make_overlay_affine(1920, 1080, w, h);
        style_.line_width = w >= 1920 ? 2 : 2;
        style_.text_scale = w >= 1920 ? 2 : 1;
    }

    // buf: 패킹된 w*h*4 (stride = w*4) 버퍼. idx: 링 인덱스.
    // 반환값: 이번에 건드린(지운 + 그린) 바이트 수 추정치
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
