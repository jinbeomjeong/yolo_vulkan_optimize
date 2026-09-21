// CPU reference YOLOv8/v11 raw output decode (header-only, no OpenCV needed).
// `02_shaders/yolo_decode.comp` (GPU postprocessing) and the host NMS of the hybrid option refer to the formulas in this file.
// Same convention as `cpu_letterbox.hpp`: the formulas are pinned down here, and shader comments point to this comment.
//
// Input: the 6 device outputs of yolov11m.hef (UINT8 NHCW). Per scale, 1 class tensor + 1 box (DFL) tensor.
//   P3 stride 8  : cls 80×80×80,  box 80×80×64
//   P4 stride 16 : cls 40×40×80,  box 40×40×64
//   P5 stride 32 : cls 20×24×80,  box 20×24×64   ← W is padded from 20 to 24. Only w 0..19 are valid, and
//                                                  columns 20~23 are garbage, so they must be skipped.
//
// Formulas (the GPU shader must follow the same rules):
//
//   1) NHCW indexing — every tensor access in this file goes through nhcw_index() only.
//        idx = ((h * C) + c) * W + w        (C = channel count, W = w_padded)
//      With (h, c) fixed, w is contiguous. Mapping GPU invocations onto the w axis gives fully coalesced access.
//
//   2) Dequantization (hailort.h hailo_quant_info_t convention: "subtract qp_zp, then multiply by qp_scale")
//        value = (q - qp_zp) * qp_scale
//      Class tensors have qp_zp = 0, limvals_min = 0.0, limvals_max ≈ 0.88~0.97.
//      → **They are already post-sigmoid probabilities, so sigmoid is not applied again** (DecodeOptions::apply_sigmoid
//        defaults to false). Rationale: pre-sigmoid logits would need a negative limvals_min like the box tensors,
//        and if the host applied sigmoid once more, dequant 0 → 0.5 > 0.2 would let all 8400×80 anchors pass, whereas
//        actual detections are 5.21/frame. The P1 exhaustive sweep confirmed this by measurement
//        (see the table above DecodeOptions).
//
//   3) Class score: score = dequant(cls[h][c][w]).  The threshold comparison
//      picks `>=` (true) / `>` (false) via DecodeOptions::threshold_inclusive.
//      The default is to emit **every** class per anchor that exceeds the threshold (argmax_only = false);
//      with argmax_only = true only the single highest-scoring class per anchor is considered.
//
//   4) DFL box decode (reg_max = 16, 4 sides: 0=left, 1=top, 2=right, 3=bottom)
//        the channel index ch is chosen by DecodeOptions::dfl_order
//          CoordMajor: ch = coord * reg_max + bin      (default)
//          BinMajor  : ch = bin * 4 + coord
//        v[bin] = dequant(box[h][ch][w])
//        softmax: m = max(v);  e[b] = exp(v[b] - m);  d = Σ b·e[b] / Σ e[b]
//        (without subtracting max, UINT8 dequantized values reach up to 19.1, exp blows up and precision is lost)
//
//   5) letterbox (640×640) coordinates
//        x1 = (w + 0.5 - d_left ) * stride,  y1 = (h + 0.5 - d_top   ) * stride
//        x2 = (w + 0.5 + d_right) * stride,  y2 = (h + 0.5 + d_bottom) * stride
//
//   6) Inverse transform to source coordinates — **the same formula** as parse_nms_by_class() in `01_src/hailo/hailo_runner.cpp`.
//        x = clamp((x_640 - pad_x) / scale, 0, src_w),  y = clamp((y_640 - pad_y) / scale, 0, src_h)
//      (HailoRT uses 0~1 normalized coordinates × dst, which is exactly x_640.)
//      ⚠ Applying the clamp **before** NMS differs from HailoRT (HailoRT runs NMS on normalized coordinates and
//        clamps afterwards). IoU can differ slightly only for boxes straddling the frame border. Uniform scaling +
//        translation preserves IoU, so there is no difference otherwise.
//
//   7) anchor_key = (scale_id << 21) | (h << 14) | (w << 7) | class
//      This is the basis of determinism (plan §4.5 G6). The atomicAdd output order of GPU pass 1 differs per run, so
//      if sorting, suppression and output use only the (score, anchor_key) **total order**, the result is order-independent.
//      h, w and class are each 0..79 and fit in 7 bits; scale_id is 0..2.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/frame_types.hpp"

namespace yvo {

inline constexpr int kDflRegMax = 16;   // number of DFL bins. box channel count = 4 * kDflRegMax = 64
inline constexpr int kDflSides = 4;     // left, top, right, bottom

// DFL channel layout. Which one applies is decided by P1's exhaustive sweep.
enum class DflChannelOrder {
    CoordMajor,  // ch = coord * reg_max + bin
    BinMajor,    // ch = bin * 4 + coord
};

// Decode variants. **The defaults are the combination confirmed by the P1 exhaustive sweep (2x2x2x2 = 16 combinations).**
// Evidence: 07_results/gpu_postprocess/evidence/p1_combination_sweep.txt
//   (video_h265.mp4 40 frames, against 81 HailoRT NMS detections, match() IoU>=0.9)
//
//   argmax_only  sigmoid  dfl_order    thr   cand/frame   test  matched   ratio
//   off          off      coord*16+b   >=          15.1     81       81  1.0000  ← chosen
//   on           off      coord*16+b   >=          15.0     80       80  0.9877  multiple classes per anchor exist
//   off          off      b*4+coord    >=          15.1    462        0  0.0000  wrong DFL channel order
//   off          on       coord*16+b   >=       16800.0      -        -  rejected  re-applying sigmoid passes every anchor
//   on           on       coord*16+b   >=        8400.0   5094       80  0.0157
//
// `>` and `>=` are **indistinguishable in principle** for this HEF: threshold 0.2 divided by qp_scale
// is not an integer at any scale (57.6409 / 53.1098 / 52.3728) → no score is ever exactly 0.2.
// Measurements also gave identical ratios for both. `>=` is the default, following HailoRT source conventions.
struct DecodeOptions {
    bool argmax_only = false;                               // false = every class per anchor that exceeds the threshold
    bool apply_sigmoid = false;                             // class outputs are already probabilities
    DflChannelOrder dfl_order = DflChannelOrder::CoordMajor;   // ch = coord * reg_max + bin
    bool threshold_inclusive = true;                        // true: score >= thr, false: score > thr
};

// Candidate detection + key for the deterministic total order. Coordinates are source frame pixels (same as Detection).
struct DetectionKeyed : Detection {
    std::uint32_t anchor_key = 0;
};

// The 2 raw tensors of one scale and their quantization and geometry parameters.
// cls/box are the start addresses of the stream buffers (UINT8 NHCW). w_padded is hw_shape.width (24 for P5).
struct RawScale {
    const std::uint8_t* cls = nullptr;
    const std::uint8_t* box = nullptr;
    int grid_h = 0;      // valid row count (80 / 40 / 20)
    int grid_w = 0;      // valid column count (80 / 40 / 20)
    int w_padded = 0;    // column count used for address computation (80 / 40 / 24)
    int stride = 0;      // 8 / 16 / 32
    float cls_scale = 1.0f;
    float cls_zp = 0.0f;
    float box_scale = 1.0f;
    float box_zp = 0.0f;
};

struct RawOutputSet {
    RawScale s[3];           // P3, P4, P5 (in stride 8, 16, 32 order)
    int num_classes = 80;
    int reg_max = kDflRegMax;
};

// NHCW indexing. The only path through which this file accesses tensors.
inline std::size_t nhcw_index(int h, int c, int w, int C, int W) {
    return (static_cast<std::size_t>(h) * static_cast<std::size_t>(C) + static_cast<std::size_t>(c)) *
               static_cast<std::size_t>(W) +
           static_cast<std::size_t>(w);
}

// Total order key of (scale_id, h, w, class). Each field fits in 7 bits (0..127).
inline std::uint32_t make_anchor_key(int scale_id, int h, int w, int cls) {
    return (static_cast<std::uint32_t>(scale_id) << 21) | (static_cast<std::uint32_t>(h) << 14) |
           (static_cast<std::uint32_t>(w) << 7) | static_cast<std::uint32_t>(cls);
}

// The repository's only IoU formula. The matching decision in ref/detection_match.hpp uses this function as is,
// and 06_tools/python/compare_detections.py mimics the same formula in float32.
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

// Raw tensors → candidate detections. Does dequantize → DFL softmax → threshold → inverse letterbox transform, and
// **does not do NMS** (cpu_nms_greedy_by_class handles that).
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
            for (int w = 0; w < s.grid_w; ++w) {   // valid columns only: 0..19 for P5 (columns 20~23 are padding)
                // 1) Class scan. Find the max of q first so the comparison can be done in the integer domain.
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

                if (!passes(score_of(argmax_q))) continue;   // the whole anchor is rejected

                // 2) DFL decode (once per anchor)
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

                // 3) letterbox 640 coordinates → 4) inverse transform to source coordinates (same formula as parse_nms_by_class)
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

// Per-class greedy NMS. It is both the reference implementation and the production code of the hybrid option (§4.4 A)
// (kept as one so the two implementations cannot diverge).
//   1) sort by the (score descending, anchor_key ascending) **total order** → deterministic result independent of input order
//   2) within the same class, suppress lower-scoring boxes with IoU >= iou_thr
//   3) at most max_per_class per class (HEF setting 100)
// dets is left in place with only the surviving detections, sorted in the total order above.
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
            keep[i] = 0;   // class quota exceeded. Later boxes of the same class are also over quota, so only suppression is skipped
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

// Reference: candidate extraction + per-class greedy NMS. The HEF settings are thr 0.2 / IoU 0.7 / 100 per class.
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
