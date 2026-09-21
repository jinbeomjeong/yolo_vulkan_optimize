// 검출 결과 비교 규칙 — **저장소 전체의 단일 구현**.
//
// 쓰는 곳: `05_tests/test_detections.cpp`, `05_tests/test_postproc.cpp`,
//          `01_src/tools/hailo/hailo_raw_probe.cpp`. 파이썬 사본은 `06_tools/python/compare_detections.py`.
//
// ⚠ **규칙을 바꾸지 마라.** 파이썬 사본과 **동치**여야 한다. 동치성이 깨지면 C++ 게이트(ctest)와
//   파이썬 집계가 서로 다른 판정을 낸다.
//
// 2026-09-21 이전의 사고: 같은 알고리즘이 4벌(여기·raw_probe·IoU 식만 cpu_yolov8_decode·파이썬)
//   있었고 **실제로 동치가 아니었다.** 여기는 `std::sort`(불안정), 파이썬은 `sorted`(안정)라
//   **점수 동점에서 greedy 선택이 달라졌다.** 그래서 정렬을 전순서로 못 박았다(아래).
//
// 판정 규칙:
//   · 같은 클래스끼리만 매칭한다.
//   · 참조(ref) 검출을 **(점수 내림차순, 원래 순번 오름차순)** 전순서로 훑으며, 아직 쓰이지 않은
//     테스트(test) 검출 중 IoU 가 최대인 것을 고른다(동률이면 먼저 나온 것). 그 IoU 가
//     kIouThreshold 이상이면 매칭으로 센다 (greedy).
//   · IoU 는 float32 로 계산한다(`iou_xyxy`). 파이썬 사본도 float32 로 흉내 낸다.
//   · 매칭 비율 = matched / max(ref 총수, test 총수). 합격선은 kMinMatchRatio.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "ref/cpu_yolov8_decode.hpp"  // iou_xyxy — IoU 식은 여기 하나뿐이다
#include "core/frame_types.hpp"

namespace yvo::detmatch {

inline constexpr float kIouThreshold = 0.9f;
inline constexpr double kMinMatchRatio = 0.95;

inline float iou(const yvo::Detection& a, const yvo::Detection& b) { return yvo::iou_xyxy(a, b); }

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
    // 전순서: 점수 내림차순, 동점이면 원래 순번 오름차순. 동점에서도 결과가 결정적이다.
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
