#pragma once
// The **single implementation** of summary statistics (median, p90, mean, min, max).
//
// Why consolidate (2026-09-21):
//   Percentiles existed in the repository with **two different algorithms**.
//     (1) linear interpolation : idx = q(n-1), v[lo] + (v[hi]-v[lo])(idx-lo)   — 8 copies
//     (2) nearest rounding     : idx = (size_t)(q(n-1)+0.5)                  — 2 copies
//       (`hailo_selftest.cpp`, `hailo_raw_probe.cpp`)
//   For even n the two algorithms give different medians. Yet **both families**
//   went into `07_results/` under **the same CSV column names** `p50`/`p90`/`lat_median`.
//   "All figures are medians" is this project's convention, yet that median had two definitions.
//
//   Unified on (1). Rationale: it is the majority (8 copies), and it is the same definition as the bench scripts' aggregation (`statistics.median`)
//   and the percentiles in `06_tools/python/compare_detections.py`.
//   → The output of the two tools that used (2) **may differ very slightly** after this change (for even n).
//     Both are standalone diagnostic tools and the tables in `07_results/` come from script aggregation, so there is no impact.
//
// Header-only. Available by linking `yvo_toolkit`.
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

// Percentile of a sample already sorted in ascending order. q is 0..1.
inline double percentile_sorted(const std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    if (v.size() == 1) return v[0];
    const double idx = q * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

// Percentile of an unsorted sample (sorts a copy).
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
