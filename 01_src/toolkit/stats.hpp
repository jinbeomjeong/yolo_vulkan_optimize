#pragma once
// 요약 통계(중앙값·p90·평균·최소·최대)의 **단일 구현**.
//
// 왜 모았나 (2026-09-21):
//   백분위수가 저장소 안에 **두 가지 알고리즘**으로 존재했다.
//     ① 선형 보간  : idx = q(n-1), v[lo] + (v[hi]-v[lo])(idx-lo)   — 8벌
//     ② 최근접 반올림: idx = (size_t)(q(n-1)+0.5)                  — 2벌
//       (`hailo_selftest.cpp`, `hailo_raw_probe.cpp`)
//   n 이 짝수일 때 두 알고리즘의 median 이 다르다. 그런데 **두 계열 모두**
//   `p50`/`p90`/`lat_median` 이라는 **같은 이름의 CSV 열**로 `07_results/` 에 들어갔다.
//   "모든 수치는 중앙값"이 이 프로젝트의 규약인데 그 중앙값의 정의가 둘이었다.
//
//   ①로 통일한다. 근거: 다수(8벌)이고, 벤치 스크립트의 집계(`statistics.median`)와
//   `06_tools/python/compare_detections.py` 의 백분위수와도 같은 정의다.
//   → ②를 쓰던 두 도구의 출력은 이 변경으로 **아주 조금 달라질 수 있다**(n 이 짝수일 때).
//     둘 다 단독 진단 도구이고 `07_results/` 의 표는 스크립트 집계가 만들므로 영향은 없다.
//
// 헤더 전용이다. `yvo_toolkit` 에 링크하면 쓸 수 있다.
#include <algorithm>
#include <cstddef>
#include <vector>

namespace yvo {

struct Stats {
    double mean = 0;
    double median = 0;
    double p90 = 0;
    double min = 0;
    double max = 0;
    std::size_t n = 0;
};

// 이미 오름차순으로 정렬된 표본에서 백분위수를 구한다. q 는 0..1.
inline double percentile_sorted(const std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    if (v.size() == 1) return v[0];
    const double idx = q * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

// 정렬되지 않은 표본을 받아 백분위수를 구한다(사본을 정렬한다).
inline double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return percentile_sorted(v, q);
}

inline Stats stats(std::vector<double> v) {
    Stats s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    s.median = percentile_sorted(v, 0.5);
    s.p90 = percentile_sorted(v, 0.9);
    s.min = v.front();
    s.max = v.back();
    return s;
}

}  // namespace yvo
