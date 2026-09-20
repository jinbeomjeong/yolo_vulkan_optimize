// CPU 기준 YOLOv8/v11 원시 출력 디코드 (헤더 전용, OpenCV 불필요).
// `02_shaders/yolo_decode.comp`(GPU 후처리)와 하이브리드안의 호스트 NMS가 이 파일의 수식을 참조한다.
// `cpu_letterbox.hpp`와 같은 관례다: 수식을 여기에 못 박고, 셰이더 주석이 이 주석을 가리킨다.
//
// 입력: yolov11m.hef의 장치 출력 6개(UINT8 NHCW). 스케일별로 클래스 텐서 1개 + box(DFL) 텐서 1개.
//   P3 stride 8  : cls 80×80×80,  box 80×80×64
//   P4 stride 16 : cls 40×40×80,  box 40×40×64
//   P5 stride 32 : cls 20×24×80,  box 20×24×64   ← W가 20→24로 패딩돼 있다. 유효 w는 0..19뿐이고
//                                                  열 20~23은 쓰레기이므로 반드시 건너뛴다.
//
// 수식 (GPU 셰이더도 같은 규칙을 따라야 한다):
//
//   1) NHCW 인덱싱 — 이 파일의 모든 텐서 접근은 nhcw_index() 하나만 쓴다.
//        idx = ((h * C) + c) * W + w        (C = 채널 수, W = w_padded)
//      (h, c)를 고정하면 w가 연속이다. GPU에서 invocation을 w축에 매핑하면 완전히 coalesce된다.
//
//   2) 역양자화 (hailort.h hailo_quant_info_t 규약: "qp_zp를 뺀 뒤 qp_scale을 곱한다")
//        value = (q - qp_zp) * qp_scale
//      클래스 텐서는 qp_zp = 0, limvals_min = 0.0, limvals_max ≈ 0.88~0.97이다.
//      → **이미 sigmoid를 지난 확률이므로 sigmoid를 다시 적용하지 않는다**(DecodeOptions::apply_sigmoid
//        기본 false). 근거: pre-sigmoid 로짓이면 box 텐서처럼 limvals_min이 음수여야 하고,
//        호스트가 sigmoid를 한 번 더 하면 dequant 0 → 0.5 > 0.2로 8400×80 앵커가 전부 통과하는데
//        실제 검출은 5.21개/frame이다. P1 전수 시도가 이 값을 실측으로 확정했다
//        (DecodeOptions 위의 표 참고).
//
//   3) 클래스 점수: score = dequant(cls[h][c][w]).  threshold 비교는
//      DecodeOptions::threshold_inclusive로 `>=`(true) / `>`(false)를 고른다.
//      앵커당 threshold를 넘는 **모든** 클래스를 내보내는 것이 기본이고(argmax_only = false),
//      argmax_only = true면 앵커당 최대 점수 클래스 하나만 본다.
//
//   4) DFL 박스 디코드 (reg_max = 16, 변 4개: 0=left, 1=top, 2=right, 3=bottom)
//        채널 번호 ch는 DecodeOptions::dfl_order로 고른다
//          CoordMajor: ch = coord * reg_max + bin      (기본)
//          BinMajor  : ch = bin * 4 + coord
//        v[bin] = dequant(box[h][ch][w])
//        softmax: m = max(v);  e[b] = exp(v[b] - m);  d = Σ b·e[b] / Σ e[b]
//        (max를 빼지 않으면 UINT8 역양자화 값이 최대 19.1까지 올라가 exp가 부풀어 정밀도를 잃는다)
//
//   5) letterbox(640×640) 좌표
//        x1 = (w + 0.5 - d_left ) * stride,  y1 = (h + 0.5 - d_top   ) * stride
//        x2 = (w + 0.5 + d_right) * stride,  y2 = (h + 0.5 + d_bottom) * stride
//
//   6) 원본 좌표 역변환 — `01_src/hailo/hailo_runner.cpp`의 parse_nms_by_class()와 **같은 식**이다.
//        x = clamp((x_640 - pad_x) / scale, 0, src_w),  y = clamp((y_640 - pad_y) / scale, 0, src_h)
//      (HailoRT는 0~1 정규화 좌표 × dst를 쓰는데 그것이 곧 x_640이다.)
//      ⚠ 클램프를 NMS **전에** 적용한다는 점은 HailoRT와 다르다(HailoRT는 정규화 좌표로 NMS한 뒤
//        클램프한다). 프레임 경계에 걸친 박스에서만 IoU가 미세하게 달라질 수 있다. 균일 스케일 +
//        평행이동은 IoU를 보존하므로 그 외에는 차이가 없다.
//
//   7) anchor_key = (scale_id << 21) | (h << 14) | (w << 7) | class
//      결정성의 근거다(계획서 §4.5 G6). GPU pass 1의 atomicAdd 출력 순서는 실행마다 다르므로
//      정렬·억제·출력이 (score, anchor_key) **전순서**만 쓰면 결과가 순서에 무관해진다.
//      h, w, class는 각각 0..79로 7비트에 들어가고 scale_id는 0..2다.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/frame_types.hpp"

namespace yvo {

inline constexpr int kDflRegMax = 16;   // DFL 빈 수. box 채널 수 = 4 * kDflRegMax = 64
inline constexpr int kDflSides = 4;     // left, top, right, bottom

// DFL 채널 배치. 어느 쪽인지는 P1이 전수 시도로 판정한다.
enum class DflChannelOrder {
    CoordMajor,  // ch = coord * reg_max + bin
    BinMajor,    // ch = bin * 4 + coord
};

// 디코드 변형. **기본값은 P1 전수 시도(2x2x2x2 = 16조합)로 확정한 조합이다.**
// 근거: 07_results/gpu_postprocess/evidence/p1_combination_sweep.txt
//   (video_h265.mp4 40프레임, HailoRT NMS 검출 81개 기준, match() IoU>=0.9)
//
//   argmax_only  sigmoid  dfl_order    thr   cand/frame   test  matched   ratio
//   off          off      coord*16+b   >=          15.1     81       81  1.0000  ← 채택
//   on           off      coord*16+b   >=          15.0     80       80  0.9877  앵커당 다중 클래스가 있다
//   off          off      b*4+coord    >=          15.1    462        0  0.0000  DFL 채널 순서 오답
//   off          on       coord*16+b   >=       16800.0      -        -  기각    sigmoid 재적용 시 전 앵커 통과
//   on           on       coord*16+b   >=        8400.0   5094       80  0.0157
//
// `>` 와 `>=` 는 이 HEF에서 **원리적으로 구분되지 않는다**: threshold 0.2를 qp_scale로 나눈 값이
// 어느 스케일에서도 정수가 아니다(57.6409 / 53.1098 / 52.3728) → 점수가 정확히 0.2인 경우가 없다.
// 실측도 두 값의 ratio가 동일했다. HailoRT 소스 관례에 맞춰 `>=`를 기본값으로 둔다.
struct DecodeOptions {
    bool argmax_only = false;                               // false = 앵커당 threshold를 넘는 모든 클래스
    bool apply_sigmoid = false;                             // 클래스 출력은 이미 확률이다
    DflChannelOrder dfl_order = DflChannelOrder::CoordMajor;   // ch = coord * reg_max + bin
    bool threshold_inclusive = true;                        // true: score >= thr, false: score > thr
};

// 후보 검출 + 결정적 전순서용 키. 좌표는 원본 프레임 픽셀(Detection과 같다).
struct DetectionKeyed : Detection {
    std::uint32_t anchor_key = 0;
};

// 한 스케일의 원시 텐서 2개와 그 양자화·기하 파라미터.
// cls/box는 스트림 버퍼의 시작 주소(UINT8 NHCW)다. w_padded는 hw_shape.width(P5는 24).
struct RawScale {
    const std::uint8_t* cls = nullptr;
    const std::uint8_t* box = nullptr;
    int grid_h = 0;      // 유효 행 수 (80 / 40 / 20)
    int grid_w = 0;      // 유효 열 수 (80 / 40 / 20)
    int w_padded = 0;    // 주소 계산에 쓰는 열 수 (80 / 40 / 24)
    int stride = 0;      // 8 / 16 / 32
    float cls_scale = 1.0f;
    float cls_zp = 0.0f;
    float box_scale = 1.0f;
    float box_zp = 0.0f;
};

struct RawOutputSet {
    RawScale s[3];           // P3, P4, P5 (stride 8, 16, 32 순서)
    int num_classes = 80;
    int reg_max = kDflRegMax;
};

// NHCW 인덱싱. 이 파일에서 텐서에 접근하는 유일한 경로다.
inline std::size_t nhcw_index(int h, int c, int w, int C, int W) {
    return (static_cast<std::size_t>(h) * static_cast<std::size_t>(C) + static_cast<std::size_t>(c)) *
               static_cast<std::size_t>(W) +
           static_cast<std::size_t>(w);
}

// (scale_id, h, w, class)의 전순서 키. 각 필드는 7비트(0..127)에 들어간다.
inline std::uint32_t make_anchor_key(int scale_id, int h, int w, int cls) {
    return (static_cast<std::uint32_t>(scale_id) << 21) | (static_cast<std::uint32_t>(h) << 14) |
           (static_cast<std::uint32_t>(w) << 7) | static_cast<std::uint32_t>(cls);
}

// 저장소 유일의 IoU 식이다. common/detection_match.hpp 의 매칭 판정이 이 함수를 그대로 쓰고,
// 06_tools/python/compare_detections.py 가 float32 로 같은 식을 흉내 낸다.
inline float iou_xyxy(const Detection& a, const Detection& b) {
    const float ix = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float iy = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = ix * iy;
    const float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}

namespace detail {

inline float dequant(std::uint8_t q, float zp, float scale) {
    return (static_cast<float>(q) - zp) * scale;
}

inline int dfl_channel(DflChannelOrder order, int coord, int bin, int reg_max) {
    return order == DflChannelOrder::CoordMajor ? coord * reg_max + bin : bin * kDflSides + coord;
}

} // namespace detail

// 원시 텐서 → 후보 검출. dequantize → DFL softmax → threshold → letterbox 역변환까지 하고
// **NMS는 하지 않는다**(cpu_nms_greedy_by_class가 맡는다).
inline std::vector<DetectionKeyed> cpu_yolov8_candidates(const RawOutputSet& raw, float thr,
                                                        const LetterboxParams& lb, int src_w, int src_h,
                                                        const DecodeOptions& opt = {}) {
    std::vector<DetectionKeyed> out;
    const int num_classes = raw.num_classes;
    const int reg_max = raw.reg_max;
    const int box_channels = kDflSides * reg_max;
    const float inv_scale = 1.0f / lb.scale;
    const float max_x = static_cast<float>(src_w);
    const float max_y = static_cast<float>(src_h);

    std::vector<float> bins(static_cast<std::size_t>(reg_max));
    float dist[kDflSides];

    for (int scale_id = 0; scale_id < 3; ++scale_id) {
        const RawScale& s = raw.s[scale_id];
        if (s.cls == nullptr || s.box == nullptr || s.grid_h <= 0 || s.grid_w <= 0) continue;
        const int W = s.w_padded;
        const float stride = static_cast<float>(s.stride);

        for (int h = 0; h < s.grid_h; ++h) {
            for (int w = 0; w < s.grid_w; ++w) {   // 유효 열만: P5는 0..19 (열 20~23은 패딩)
                // 1) 클래스 스캔. 정수 도메인에서 비교할 수 있게 먼저 q의 최대를 찾는다.
                int argmax_c = 0;
                std::uint8_t argmax_q = 0;
                for (int c = 0; c < num_classes; ++c) {
                    const std::uint8_t q = s.cls[nhcw_index(h, c, w, num_classes, W)];
                    if (q > argmax_q) {
                        argmax_q = q;
                        argmax_c = c;
                    }
                }
                auto score_of = [&](std::uint8_t q) {
                    float v = detail::dequant(q, s.cls_zp, s.cls_scale);
                    if (opt.apply_sigmoid) v = 1.0f / (1.0f + std::exp(-v));
                    return v;
                };
                auto passes = [&](float v) { return opt.threshold_inclusive ? (v >= thr) : (v > thr); };

                if (!passes(score_of(argmax_q))) continue;   // 앵커 전체 탈락

                // 2) DFL 디코드 (앵커당 1회)
                for (int k = 0; k < kDflSides; ++k) {
                    float m = -std::numeric_limits<float>::infinity();
                    for (int b = 0; b < reg_max; ++b) {
                        const int ch = detail::dfl_channel(opt.dfl_order, k, b, reg_max);
                        const float v =
                            detail::dequant(s.box[nhcw_index(h, ch, w, box_channels, W)], s.box_zp, s.box_scale);
                        bins[static_cast<std::size_t>(b)] = v;
                        if (v > m) m = v;
                    }
                    float sum = 0.0f;
                    float acc = 0.0f;
                    for (int b = 0; b < reg_max; ++b) {
                        const float e = std::exp(bins[static_cast<std::size_t>(b)] - m);
                        bins[static_cast<std::size_t>(b)] = e;
                        sum += e;
                        acc += static_cast<float>(b) * e;
                    }
                    dist[k] = acc / sum;
                }

                // 3) letterbox 640 좌표 → 4) 원본 좌표 역변환 (parse_nms_by_class와 같은 식)
                const float cx = static_cast<float>(w) + 0.5f;
                const float cy = static_cast<float>(h) + 0.5f;
                const float x1_640 = (cx - dist[0]) * stride;
                const float y1_640 = (cy - dist[1]) * stride;
                const float x2_640 = (cx + dist[2]) * stride;
                const float y2_640 = (cy + dist[3]) * stride;
                const float x1 = std::clamp((x1_640 - static_cast<float>(lb.pad_x)) * inv_scale, 0.0f, max_x);
                const float y1 = std::clamp((y1_640 - static_cast<float>(lb.pad_y)) * inv_scale, 0.0f, max_y);
                const float x2 = std::clamp((x2_640 - static_cast<float>(lb.pad_x)) * inv_scale, 0.0f, max_x);
                const float y2 = std::clamp((y2_640 - static_cast<float>(lb.pad_y)) * inv_scale, 0.0f, max_y);

                // 5) emit
                auto emit = [&](int c, float score) {
                    DetectionKeyed d;
                    d.class_id = c;
                    d.score = score;
                    d.x1 = x1;
                    d.y1 = y1;
                    d.x2 = x2;
                    d.y2 = y2;
                    d.anchor_key = make_anchor_key(scale_id, h, w, c);
                    out.push_back(d);
                };
                if (opt.argmax_only) {
                    emit(argmax_c, score_of(argmax_q));
                } else {
                    for (int c = 0; c < num_classes; ++c) {
                        const std::uint8_t q = s.cls[nhcw_index(h, c, w, num_classes, W)];
                        const float v = score_of(q);
                        if (passes(v)) emit(c, v);
                    }
                }
            }
        }
    }
    return out;
}

// 클래스별 greedy NMS. 기준 구현이면서 하이브리드안(§4.4 A)의 프로덕션 코드이기도 하다
// (두 구현이 갈라지지 않게 하나로 둔다).
//   1) (score 내림차순, anchor_key 오름차순) **전순서**로 정렬 → 입력 순서에 무관한 결정적 결과
//   2) 같은 클래스에서 IoU >= iou_thr인 낮은 점수 박스를 억제
//   3) 클래스별 최대 max_per_class개 (HEF 설정 100)
// dets는 제자리에서 살아남은 검출만 남기고, 위 전순서로 정렬된 상태가 된다.
inline void cpu_nms_greedy_by_class(std::vector<DetectionKeyed>& dets, float iou_thr, int max_per_class) {
    std::sort(dets.begin(), dets.end(), [](const DetectionKeyed& a, const DetectionKeyed& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.anchor_key < b.anchor_key;
    });
    const std::size_t n = dets.size();
    std::vector<char> keep(n, 1);
    int max_class = -1;
    for (const auto& d : dets) max_class = std::max(max_class, d.class_id);
    std::vector<int> kept(static_cast<std::size_t>(max_class + 1), 0);

    for (std::size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        const int c = dets[i].class_id;
        if (kept[static_cast<std::size_t>(c)] >= max_per_class) {
            keep[i] = 0;   // 클래스 정원 초과. 뒤의 같은 클래스도 모두 초과이므로 억제만 건너뛴다
            continue;
        }
        ++kept[static_cast<std::size_t>(c)];
        for (std::size_t j = i + 1; j < n; ++j) {
            if (!keep[j] || dets[j].class_id != c) continue;
            if (iou_xyxy(dets[i], dets[j]) >= iou_thr) keep[j] = 0;
        }
    }
    std::size_t w = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (keep[i]) dets[w++] = dets[i];
    }
    dets.resize(w);
}

// 기준값: 후보 추출 + 클래스별 greedy NMS. HEF 설정은 thr 0.2 / IoU 0.7 / 클래스당 100개다.
inline std::vector<Detection> cpu_yolov8_postprocess(const RawOutputSet& raw, float score_thr, float iou_thr,
                                                     int max_per_class, const LetterboxParams& lb, int src_w,
                                                     int src_h, const DecodeOptions& opt = {}) {
    auto cand = cpu_yolov8_candidates(raw, score_thr, lb, src_w, src_h, opt);
    cpu_nms_greedy_by_class(cand, iou_thr, max_per_class);
    std::vector<Detection> out;
    out.reserve(cand.size());
    for (const auto& d : cand) out.push_back(static_cast<const Detection&>(d));
    return out;
}

} // namespace yvo
