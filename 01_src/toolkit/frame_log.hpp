#pragma once
// --frame-log: one CSV row per frame, written once by yvo_pipeline (pipeline/main.cpp) at the end of the run
// (the run loop only collects time points; no file I/O happens inside it).
//
//   src,seq,dropped,arrival_ms,decode_end_ms,submit_ms,npu_done_ms,gpu_post_done_ms,complete_ms
//
//   src              source frame index (arrival order with --input-pace source; = seq with asap)
//   seq              submit sequence (the internal pts); empty for frames that were never submitted
//   dropped          0 = processed, 1 = dropped by the leaky input queue, 2 = abandoned (still queued at a stop/failure)
//   *_ms             milliseconds since the run's t0 (the wall-clock start, also the arrival schedule's base), "%.3f";
//                    empty when the event does not apply (arrival_ms with asap, gpu_post_done_ms with --postproc host,
//                    everything after decode_end_ms for dropped/abandoned frames, events a failed run never reached)
//
//   With --display on a 10th column is appended (with_present = true; runs without display keep the 9 columns):
//   present_ms       when the display backend reports the frame on screen (the glass-to-glass screen time); empty for
//                    frames that never reached the screen (display drops, --display-every skips, no feedback)
//
// Library code does not include this header — it is for tools only.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace yvo::tool {

inline constexpr const char* kFrameLogHeader =
    "src,seq,dropped,arrival_ms,decode_end_ms,submit_ms,npu_done_ms,gpu_post_done_ms,complete_ms";
inline constexpr const char* kFrameLogPresentColumn = "present_ms";   // appended with --display on

struct FrameLogRow {
    static constexpr double kNone = std::numeric_limits<double>::quiet_NaN();   // empty field

    std::int64_t src = 0;
    std::int64_t seq = -1;   // < 0 = never submitted (empty field)
    int dropped = 0;         // 0 processed, 1 dropped, 2 abandoned
    double arrival_ms = kNone;
    double decode_end_ms = kNone;
    double submit_ms = kNone;
    double npu_done_ms = kNone;
    double gpu_post_done_ms = kNone;
    double complete_ms = kNone;
    double present_ms = kNone;   // written only with with_present
};

// The header line including the trailing '\n'. with_present appends the present_ms column.
inline std::string format_frame_log_header(bool with_present = false) {
    return std::string(kFrameLogHeader) + (with_present ? std::string(",") + kFrameLogPresentColumn : std::string()) +
           "\n";
}

// One row including the trailing '\n'. Non-finite times print as empty fields. with_present must match the header.
inline std::string format_frame_log_row(const FrameLogRow& r, bool with_present = false) {
    std::string out;
    char buf[32];
    const auto put_ms = [&](double v) {
        out += ',';
        if (!std::isfinite(v)) return;
        const int n = std::snprintf(buf, sizeof(buf), "%.3f", v);
        if (n > 0 && static_cast<std::size_t>(n) < sizeof(buf)) out.append(buf, static_cast<std::size_t>(n));
    };
    out += std::to_string(r.src);
    out += ',';
    if (r.seq >= 0) out += std::to_string(r.seq);
    out += ',';
    out += std::to_string(r.dropped);
    put_ms(r.arrival_ms);
    put_ms(r.decode_end_ms);
    put_ms(r.submit_ms);
    put_ms(r.npu_done_ms);
    put_ms(r.gpu_post_done_ms);
    put_ms(r.complete_ms);
    if (with_present) put_ms(r.present_ms);
    out += '\n';
    return out;
}

} // namespace yvo::tool
