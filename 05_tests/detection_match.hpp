// 검출 결과 비교 규칙 (테스트 공용). `test_detections.cpp` 에서 추출했다.
//
// ⚠ **규칙을 바꾸지 마라.** 이 구현은 `06_tools/python/compare_detections.py` 와 **의도적으로 동치**다.
//   동치성이 깨지면 C++ 게이트(ctest)와 파이썬 게이트(집계 스크립트)가 서로 다른 판정을 낸다.
//
// 판정 규칙:
//   · 같은 클래스끼리만 매칭한다.
//   · 참조(ref) 검출을 **점수 내림차순**으로 훑으며, 아직 쓰이지 않은 테스트(test) 검출 중
//     IoU 가 최대인 것을 고른다. 그 IoU 가 kIouThreshold 이상이면 매칭으로 센다 (greedy).
//   · 매칭 비율 = matched / max(ref 총수, test 총수). 합격선은 kMinMatchRatio.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "common/frame_types.hpp"

namespace yvo::detmatch {

inline constexpr float kIouThreshold = 0.9f;
inline constexpr double kMinMatchRatio = 0.95;

inline float iou(const yvo::Detection& a, const yvo::Detection& b) {
    const float ix = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float iy = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = ix * iy;
    const float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0.0f ? inter / ua : 0.0f;
}

struct MatchResult {
    long ref = 0, test = 0, matched = 0;
};

// 점수 순 greedy 매칭 (같은 클래스, IoU 최대, 임계 이상)
inline MatchResult match(const std::vector<yvo::Detection>& ref, const std::vector<yvo::Detection>& test,
                         std::vector<std::string>* detail) {
    MatchResult m;
    m.ref = static_cast<long>(ref.size());
    m.test = static_cast<long>(test.size());
    std::vector<std::size_t> order(ref.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ref[a].score > ref[b].score; });
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
