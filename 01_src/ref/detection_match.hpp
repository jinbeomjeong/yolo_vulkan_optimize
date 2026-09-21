// Detection result comparison rule — **the single implementation for the whole repository**.
//
// Used by: `05_tests/test_detections.cpp`, `05_tests/test_postproc.cpp`,
//          `01_src/tools/hailo/hailo_raw_probe.cpp`. The Python copy is `06_tools/python/compare_detections.py`.
//
// ⚠ **Do not change the rule.** It must be **equivalent** to the Python copy. If equivalence breaks, the C++ gate (ctest) and
//   the Python aggregation give different verdicts.
//
// The incident before 2026-09-21: the same algorithm existed in 4 copies (here, raw_probe, cpu_yolov8_decode for the IoU formula only, Python)
//   and **they were not actually equivalent.** Here used `std::sort` (unstable), Python used `sorted` (stable), so
//   **the greedy choice differed on score ties.** So the sort was pinned to a total order (below).
//
// Decision rule:
//   · Match only within the same class.
//   · Walk the reference (ref) detections in the **(score descending, original index ascending)** total order, and among the
//     not-yet-used test detections pick the one with the highest IoU (first one on ties). If that IoU is
//     at least kIouThreshold, count it as a match (greedy).
//   · IoU is computed in float32 (`iou_xyxy`). The Python copy mimics float32 too.
//   · Match ratio = matched / max(ref total, test total). The pass line is kMinMatchRatio.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "ref/cpu_yolov8_decode.hpp"  // iou_xyxy — the IoU formula lives only there
#include "core/frame_types.hpp"

namespace yvo::detmatch {

inline constexpr float kIouThreshold = 0.9f;
inline constexpr double kMinMatchRatio = 0.95;

inline float iou(const yvo::Detection& a, const yvo::Detection& b) { return yvo::iou_xyxy(a, b); }

struct MatchResult {
    long ref = 0, test = 0, matched = 0;
};

// Score-ordered greedy matching (same class, max IoU, at or above threshold)
inline MatchResult match(const std::vector<yvo::Detection>& ref, const std::vector<yvo::Detection>& test,
                         std::vector<std::string>* detail) {
    MatchResult m;
    m.ref = static_cast<long>(ref.size());
    m.test = static_cast<long>(test.size());
    std::vector<std::size_t> order(ref.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    // Total order: score descending, original index ascending on ties. The result is deterministic even on ties.
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (ref[a].score != ref[b].score) return ref[a].score > ref[b].score;
        return a < b;
    });
    std::vector<bool> used(test.size(), false);
    for (std::size_t i : order) {
        const auto& r = ref[i];
        float best = 0.0f;
        std::size_t best_j = test.size();
        for (std::size_t j = 0; j < test.size(); ++j) {
            if (used[j] || test[j].class_id != r.class_id) continue;
            const float v = iou(r, test[j]);
            if (v > best) {
                best = v;
                best_j = j;
            }
        }
        if (best_j < test.size() && best >= kIouThreshold) {
            used[best_j] = true;
            ++m.matched;
        } else if (detail) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "unmatched ref class %d score %.3f box (%.1f,%.1f)-(%.1f,%.1f) best IoU %.3f",
                          r.class_id, static_cast<double>(r.score), static_cast<double>(r.x1),
                          static_cast<double>(r.y1), static_cast<double>(r.x2), static_cast<double>(r.y2),
                          static_cast<double>(best));
            detail->push_back(buf);
        }
    }
    if (detail) {
        for (std::size_t j = 0; j < test.size(); ++j) {
            if (used[j]) continue;
            char buf[160];
            std::snprintf(buf, sizeof(buf), "extra test class %d score %.3f box (%.1f,%.1f)-(%.1f,%.1f)",
                          test[j].class_id, static_cast<double>(test[j].score), static_cast<double>(test[j].x1),
                          static_cast<double>(test[j].y1), static_cast<double>(test[j].x2),
                          static_cast<double>(test[j].y2));
            detail->push_back(buf);
        }
    }
    return m;
}

} // namespace yvo::detmatch
