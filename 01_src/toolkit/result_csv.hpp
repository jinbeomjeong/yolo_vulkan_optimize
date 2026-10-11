#pragma once
// RESULT_CSV: the 22-field summary line printed with --csv-label by yvo_pipeline (pipeline/main.cpp) and by
// yvo_gst_pipeline --sink hailo (tools/gst/gst_pipeline.cpp). 06_tools/bench/ parses it, so the bytes must not change:
// field order, count and printf formats are fixed here, in one place (before 2026-10-11 each tool had its own printf).
//
//   label,mode,layout,batch,slots,frames,wall_s,fps,cpu_ms_per_frame,decode_mean,gpu_mean,gpu_p90,npu_mean,
//   npu_p90,e2e_mean,e2e_p90,detections,hailo_t0,hailo_t1,soc_t0,soc_t1,ok
//
// Library code does not include this header — it is for tools only.
#include <cstdio>
#include <string>

namespace yvo::tool {

struct ResultCsv {
    std::string label;
    const char* mode = "";     // dmabuf | ptr
    const char* layout = "";   // nhcw | nhwc
    int batch = 0;
    int slots = 0;
    long frames = 0;           // completed frames
    double wall_s = 0.0;
    double fps = 0.0;
    double cpu_ms_per_frame = 0.0;
    double decode_mean = 0.0;  // nan where there is no blocking decode call (yvo_gst_pipeline)
    double gpu_mean = 0.0, gpu_p90 = 0.0;
    double npu_mean = 0.0, npu_p90 = 0.0;
    double e2e_mean = 0.0, e2e_p90 = 0.0;
    long detections = 0;
    double hailo_t0 = 0.0, hailo_t1 = 0.0;
    double soc_t0 = 0.0, soc_t1 = 0.0;
    bool ok = false;
};

// The complete line including the trailing '\n'.
inline std::string format_result_csv(const ResultCsv& r) {
    const auto print = [&r](char* buf, std::size_t size) {
        return std::snprintf(
            buf, size,
            "RESULT_CSV,%s,%s,%s,%d,%d,%ld,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%ld,%.1f,%.1f,%.1f,%.1f,%d\n",
            r.label.c_str(), r.mode, r.layout, r.batch, r.slots, r.frames, r.wall_s, r.fps, r.cpu_ms_per_frame,
            r.decode_mean, r.gpu_mean, r.gpu_p90, r.npu_mean, r.npu_p90, r.e2e_mean, r.e2e_p90, r.detections,
            r.hailo_t0, r.hailo_t1, r.soc_t0, r.soc_t1, r.ok ? 1 : 0);
    };
    const int n = print(nullptr, 0);   // two passes: measure, then write (never truncates a long label)
    if (n <= 0) return {};
    std::string out(static_cast<std::size_t>(n) + 1, '\0');
    print(out.data(), out.size());
    out.resize(static_cast<std::size_t>(n));
    return out;
}

// Writes the line to stdout (same bytes as the printf it replaced).
inline void print_result_csv(const ResultCsv& r) { std::fputs(format_result_csv(r).c_str(), stdout); }

}  // namespace yvo::tool
