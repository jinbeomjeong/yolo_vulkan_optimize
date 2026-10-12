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

// RESULT_CSV4: yvo_pipeline --input-pace source only (with --csv-label). 22 fields:
//
//   label,input_queue,src_fps,arrived,processed,dropped,abandoned,drop_rate,max_drop_run,queue_depth_max,
//   e2e_arr_mean,e2e_arr_median,e2e_arr_p90,e2e_arr_max,queue_wait_mean,queue_wait_p90,queue_wait_max,
//   decode_late_mean,decode_late_p90,decode_late_max,held_frames_max,ok
//
//   e2e_arr      source arrival time → NPU completion callback (ms). RESULT_CSV's e2e stays decode return → completion.
//   queue_wait   decode return → submit to the GPU (ms), per processed frame.
//   decode_late  source arrival time → decode return (ms), per arrived frame.
//   drop_rate    dropped / arrived.
struct ResultCsv4 {
    std::string label;
    int input_queue = 0;
    double src_fps = 0.0;
    long arrived = 0, processed = 0, dropped = 0, abandoned = 0;
    double drop_rate = 0.0;
    long max_drop_run = 0, queue_depth_max = 0;
    double e2e_arr_mean = 0.0, e2e_arr_median = 0.0, e2e_arr_p90 = 0.0, e2e_arr_max = 0.0;
    double queue_wait_mean = 0.0, queue_wait_p90 = 0.0, queue_wait_max = 0.0;
    double decode_late_mean = 0.0, decode_late_p90 = 0.0, decode_late_max = 0.0;
    long held_frames_max = 0;
    bool ok = false;
};

// The complete line including the trailing '\n'.
inline std::string format_result_csv4(const ResultCsv4& r) {
    const auto print = [&r](char* buf, std::size_t size) {
        return std::snprintf(
            buf, size,
            "RESULT_CSV4,%s,%d,%.3f,%ld,%ld,%ld,%ld,%.4f,%ld,%ld,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%ld,%d\n",
            r.label.c_str(), r.input_queue, r.src_fps, r.arrived, r.processed, r.dropped, r.abandoned, r.drop_rate,
            r.max_drop_run, r.queue_depth_max, r.e2e_arr_mean, r.e2e_arr_median, r.e2e_arr_p90, r.e2e_arr_max,
            r.queue_wait_mean, r.queue_wait_p90, r.queue_wait_max, r.decode_late_mean, r.decode_late_p90,
            r.decode_late_max, r.held_frames_max, r.ok ? 1 : 0);
    };
    const int n = print(nullptr, 0);
    if (n <= 0) return {};
    std::string out(static_cast<std::size_t>(n) + 1, '\0');
    print(out.data(), out.size());
    out.resize(static_cast<std::size_t>(n));
    return out;
}

}  // namespace yvo::tool
