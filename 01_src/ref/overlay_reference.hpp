// 오버레이(박스·라벨·HUD) 렌더링 규약과 CPU 기준 구현 — 헤더 전용, OpenCV 불필요.
//
// `01_src/ref/cpu_letterbox.hpp` 의 관례를 따른다: **수식과 경계 규칙을 상단 주석에 못 박고
// GPU 셰이더(`02_shaders/overlay_boxes.comp`)가 이 주석을 참조한다.** 아래 구현은 그 주석의
// 기계적 번역일 뿐이며, `05_tests/test_overlay.cpp` 가 GPU 결과와 **max 오차 0** 으로 비교한다.
//
// 계획서 부록 A.12 가 이 파일이 존재하는 이유다: P5 테스트 패턴이 기대 400x300 대신 399x299 를
// 그렸다(포함/배제 off-by-one). 오버레이는 박스 테두리를 그리므로 같은 실수가 곧 테스트 실패다.
// **규칙을 먼저 글로 정하고 GPU·CPU 를 그 글에 맞춘다. 순서를 반대로 하면 비트 일치가 안 된다.**
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════╗
// ║  경계 규약 (이 블록이 유일한 기준이다. GPU 와 CPU 가 여기서 한 글자도 벗어나면 안 된다)  ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════╝
//
// R0. 좌표계 — affine 흡수
//     `Detection` 은 이미 **원본 1920x1080 좌표**다(`hailo_runner.cpp:54-82` 가 역변환해 준다).
//     캔버스(스왑체인 이미지) 좌표로 가는 변환은 push constant 의 affine 하나로만 한다:
//         fx = x * sx + tx,   fy = y * sy + ty      (모두 float32)
//     캔버스가 1920x1080 이면 **항등**(sx=sy=1, tx=ty=0)이고, 더 작으면 `make_letterbox()` 와
//     같은 균일 스케일 + 중앙 정렬이 들어간다(make_overlay_affine()). GPU NMS 로 바뀌어도
//     정규화 좌표를 쓰고 affine 만 바꾸면 되므로 **셰이더 변경은 0 이다.**
//
// R1. float → 정수 픽셀 : **반올림(round half up) 후 포함 범위**
//         i = int(floor(clamp(f, -32768, 32767) + 0.5))
//     클램프는 int 변환의 미정의 동작을 막는 것이다(GLSL 의 int(float) 는 범위를 넘으면 미정의).
//     그리고 박스의 정수 사각형은 **네 변을 모두 포함하는 닫힌 범위** [ix1, ix2] x [iy1, iy2] 다.
//     따라서 픽셀 폭 = ix2 - ix1 + 1 이다. "399 냐 400 이냐"를 여기서 명시적으로 닫는다.
//     x1 > x2 처럼 뒤집힌 입력은 변환 후 min/max 로 정규화한다.
//
// R2. NaN / Inf : 네 좌표 중 하나라도 유한하지 않으면 **그 검출을 통째로 건너뛴다**(아무것도 안 그린다).
//     실무에서는 오지 않지만, 규칙이 없으면 GPU 의 clamp(NaN) 결과가 구현 정의라 비트 일치가 깨진다.
//
// R3. 테두리 두께 : **안쪽으로만 자란다**(w >= 1, 기본 1080p 2~3 px).
//     픽셀 (x, y) 는 사각형 안(R1 의 닫힌 범위)에 있고 다음을 만족할 때 테두리다:
//         (x - ix1 < w) || (ix2 - x < w) || (y - iy1 < w) || (iy2 - y < w)
//     바깥으로 자라지 않으므로 ① 화면 경계에 붙은 박스의 테두리가 잘리지 않고
//     ② 박스가 이웃 픽셀을 침범하지 않아 클리핑이 사각형 클리핑 하나로 끝난다.
//     사각형이 2w 보다 얇으면 위 조건이 전부 참이 되어 **사각형이 꽉 찬다**(자연스러운 퇴화).
//
// R4. 면적 0 : x1 == x2 (또는 y1 == y2)면 정수 사각형은 폭(또는 높이) **1 px** 이고,
//     R3 에 의해 그 1 px 선이 그려진다. 검출을 버리지 않는다.
//     완전히 같은 점(x1==x2 && y1==y2)이면 1 px 점 하나다.
//
// R5. 클리핑 : 정수 사각형은 **클리핑하지 않은 좌표로 계산**하고(그래야 테두리가 진짜 변에 붙는다),
//     실제 쓰기 직전에 화면 [0, dst_w-1] x [0, dst_h-1] 로 자른다.
//     완전히 화면 밖이면 아무 픽셀도 쓰지 않는다. 일부만 밖이면 보이는 변만 남는다.
//
// R6. 라벨 텍스트 기준점
//     문자열 = "<class_name> <score>", score 는 **정수 도메인에서** 두 자리로 만든다:
//         q = uint(floor(clamp(score, 0, 1) * 100 + 0.5))      // 0..100 (유한하지 않은 score 는 0)
//         문자 = ['0' + q/100, '.', '0' + (q/10)%10, '0' + q%10]   → "0.87", "1.00"
//     float 서식화를 GPU·CPU 양쪽에서 똑같이 하는 유일하게 안전한 방법이다(printf 를 쓰지 않는다).
//     원점(글리프 좌상단):
//         tx0 = max(ix1, 0)
//         ty0 = (iy1 - 16*s >= 0) ? (iy1 - 16*s) : max(iy1, 0)      // s = text_scale
//     즉 **기본은 박스 좌상단 바로 위**, 위쪽에 16*s 픽셀이 없으면 **박스 안쪽 상단**으로 내린다.
//     오른쪽으로 넘치는 부분은 자르지 않고 R5 의 화면 클리핑에 맡긴다(가로 재배치 없음).
//     글리프 렌더링 자체(행/비트 순서, 정수 배율, 진행폭 8*s, 줄 높이 16*s, 배경 미터치)는
//     `01_src/ref/glyph_font_8x16.hpp` 의 규약 1)~8) 을 그대로 따른다.
//     **라벨 배경 상자는 그리지 않는다** — 글리프 규약 5)가 "배경은 호출자가 따로"라고 했고,
//     1080p 검출 80개에서 배경을 채우면 픽셀 수가 4배 늘어 P7 의 0.7 ms 예산을 위협한다.
//
// R7. 색 : 박스 테두리와 라벨 모두 `yvo::class_color(cid)`(RGB). 알파는 **항상 255**.
//     클래스 번호는 먼저 [0, 79] 로 클램프한다(색과 이름 모두 같은 cid 를 쓴다).
//     HUD 한 줄은 흰색 (255,255,255), 원점 고정 (kHudOriginX, kHudOriginY) = (4, 4).
//
// R8. 그리는 순서 : 검출 인덱스 오름차순(CPU). **GPU 는 검출 하나당 워크그룹 하나이므로 순서가 없다.**
//     → **겹치는 박스의 교차 픽셀 색은 순서 의존적이다**(허용 한계로 문서화. 계획서 §7, §9).
//     한 검출 안에서는 테두리와 라벨이 같은 색이므로 겹쳐도 결과가 같다(멱등).
//     HUD 는 GPU 에서 워크그룹 0 이 자기 검출과 **함께** 그린다 → HUD 영역과 겹치는 박스도
//     순서 의존이다. 테스트와 프로브는 그 영역을 비워 둔다.
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

// R7: HUD 원점 (셰이더의 kHudOriginX/Y 와 같은 값이어야 한다)
inline constexpr int kHudOriginX = 4;
inline constexpr int kHudOriginY = 4;
// DetBuf 헤더의 HUD 문자열 용량 (uint text[32] = 128 B)
inline constexpr std::size_t kHudMaxBytes = 128;
// 라벨 문자열 최대 길이: 가장 긴 COCO 이름(13) + " 0.00"(5)
inline constexpr int kMaxLabelChars = 18;

// R0: 원본 좌표 → 캔버스 좌표 affine
struct OverlayAffine {
    float sx = 1.0f, sy = 1.0f, tx = 0.0f, ty = 0.0f;
};

// 캔버스가 원본과 같으면 항등, 작으면 make_letterbox() 와 같은 균일 스케일 + 중앙 정렬.
inline OverlayAffine make_overlay_affine(int src_w, int src_h, int dst_w, int dst_h) {
    OverlayAffine a;
    if (src_w == dst_w && src_h == dst_h) return a; // 항등 (원본 해상도 표시 = 기본 경로)
    const LetterboxParams lb = make_letterbox(src_w, src_h, dst_w, dst_h);
    a.sx = a.sy = lb.scale;
    a.tx = static_cast<float>(lb.pad_x);
    a.ty = static_cast<float>(lb.pad_y);
    return a;
}

struct OverlayStyle {
    int line_width = 2;   // R3. 1080p 기본 2
    int text_scale = 1;   // R6. 1080p 기본 2 (호스트 OverlayConfig 가 정한다)
    bool draw_text = true;
};

// R1: float → 정수 픽셀
inline int overlay_round(float v) {
    const float c = std::clamp(v, -32768.0f, 32767.0f);
    return static_cast<int>(std::floor(c + 0.5f));
}

// R1 의 닫힌(포함) 정수 사각형
struct OverlayRect {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

inline bool overlay_finite(float v) { return std::isfinite(v); }

// R0 + R1 + R2. 유한하지 않은 좌표면 false (그 검출은 통째로 건너뛴다).
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

// R7: 색·이름에 쓰는 클래스 번호
inline int overlay_class_id(int class_id) { return std::clamp(class_id, 0, kNumCocoClasses - 1); }

// R6: score 를 정수 두 자리로 (GPU 와 같은 연산 순서)
inline unsigned overlay_score_q(float score) {
    const float s = std::isfinite(score) ? std::clamp(score, 0.0f, 1.0f) : 0.0f;
    return static_cast<unsigned>(std::floor(s * 100.0f + 0.5f));
}

// R6: 라벨 문자열. GPU 는 같은 규칙으로 문자를 하나씩 만든다(문자열을 만들지 않는다).
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

// ─────────────────────────── CPU 기준 렌더러 ───────────────────────────
// rgba: w*h*4, R8G8B8A8_UNORM. 기존 내용 위에 덮어 그린다(GPU 와 같다).

namespace detail {

inline void put_px(std::uint8_t* rgba, int w, int h, int x, int y, const std::array<std::uint8_t, 3>& c) {
    if (x < 0 || y < 0 || x >= w || y >= h) return; // R5
    std::uint8_t* p = rgba + (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
    p[0] = c[0];
    p[1] = c[1];
    p[2] = c[2];
    p[3] = 255; // R7
}

// R5: 화면으로 자른 뒤 채운다 (포함 범위)
inline void fill_rect(std::uint8_t* rgba, int w, int h, int x0, int y0, int x1, int y1,
                      const std::array<std::uint8_t, 3>& c) {
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    x1 = std::min(x1, w - 1);
    y1 = std::min(y1, h - 1);
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) put_px(rgba, w, h, x, y, c);
}

// 글리프 규약 1)~8) + R6. text 의 각 문자를 (x0, y0) 부터 8*s 간격으로 찍는다.
inline void draw_glyphs(std::uint8_t* rgba, int w, int h, int x0, int y0, std::string_view text, int s,
                        const std::array<std::uint8_t, 3>& c) {
    for (std::size_t ci = 0; ci < text.size(); ++ci) {
        const int g = glyph_index(static_cast<unsigned char>(text[ci]));
        for (int row = 0; row < kGlyphHeight; ++row) {
            for (int col = 0; col < kGlyphWidth; ++col) {
                if (!glyph_pixel(g, col, row)) continue; // 규약 5): 꺼진 픽셀은 건드리지 않는다
                const int px0 = x0 + static_cast<int>(ci) * kGlyphWidth * s + col * s;
                const int py0 = y0 + row * s;
                for (int j = 0; j < s; ++j)
                    for (int i = 0; i < s; ++i) put_px(rgba, w, h, px0 + i, py0 + j, c);
            }
        }
    }
}

} // namespace detail

// 검출 하나를 그린다 (테두리 + 라벨). GPU 의 워크그룹 하나와 같은 일을 한다.
inline void overlay_cpu_draw_detection(std::uint8_t* rgba, int w, int h, const Detection& d,
                                       const OverlayAffine& a, const OverlayStyle& st) {
    OverlayRect r{};
    if (!overlay_rect_of(d, a, &r)) return; // R2
    const int cid = overlay_class_id(d.class_id);
    const std::array<std::uint8_t, 3> col = class_color(cid);
    const int lw = std::max(st.line_width, 1);

    // R3: 안쪽으로 자라는 네 변. 네 밴드의 합집합이 R3 의 조건식 집합과 정확히 같다.
    // 겹치는 코너 픽셀은 같은 색으로 두 번 써도 결과가 같다(멱등).
    detail::fill_rect(rgba, w, h, r.x1, r.y1, r.x2, std::min(r.y1 + lw - 1, r.y2), col);                 // 위
    detail::fill_rect(rgba, w, h, r.x1, std::max(r.y2 - lw + 1, r.y1), r.x2, r.y2, col);                 // 아래
    detail::fill_rect(rgba, w, h, r.x1, r.y1, std::min(r.x1 + lw - 1, r.x2), r.y2, col);                 // 왼
    detail::fill_rect(rgba, w, h, std::max(r.x2 - lw + 1, r.x1), r.y1, r.x2, r.y2, col);                 // 오른

    if (!st.draw_text) return;
    const int s = std::max(st.text_scale, 1);
    const int tx0 = std::max(r.x1, 0);                                        // R6
    const int ty0 = (r.y1 - kGlyphHeight * s >= 0) ? (r.y1 - kGlyphHeight * s) : std::max(r.y1, 0);
    detail::draw_glyphs(rgba, w, h, tx0, ty0, overlay_label(d.class_id, d.score), s, col);
}

// HUD 한 줄 (R7: 흰색, 고정 원점)
inline void overlay_cpu_draw_hud(std::uint8_t* rgba, int w, int h, std::string_view hud, const OverlayStyle& st) {
    if (!st.draw_text || hud.empty()) return;
    const std::array<std::uint8_t, 3> white{255, 255, 255};
    detail::draw_glyphs(rgba, w, h, kHudOriginX, kHudOriginY, hud.substr(0, kHudMaxBytes),
                        std::max(st.text_scale, 1), white);
}

// 전체 기준 렌더 (R8: 검출 인덱스 오름차순, HUD 를 마지막에)
inline void overlay_cpu_render(std::uint8_t* rgba, int w, int h, const std::vector<Detection>& dets,
                               std::string_view hud, const OverlayAffine& a, const OverlayStyle& st,
                               std::size_t max_dets = 80) {
    const std::size_t n = std::min(dets.size(), max_dets);
    for (std::size_t i = 0; i < n; ++i) overlay_cpu_draw_detection(rgba, w, h, dets[i], a, st);
    overlay_cpu_draw_hud(rgba, w, h, hud, st);
}

} // namespace yvo::display
