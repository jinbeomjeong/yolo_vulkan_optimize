#include "pipeline/slot_exchange.hpp"

#include <cstdarg>
#include <cstdio>

namespace yvo::pipeline {

namespace {

// printf-style formatting into std::string (two passes: measure, then write — never truncates).
[[gnu::format(printf, 1, 2)]] std::string format(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string out;
    if (n > 0) {
        out.resize(static_cast<std::size_t>(n) + 1);
        std::vsnprintf(out.data(), out.size(), fmt, ap2);
        out.resize(static_cast<std::size_t>(n));
    }
    va_end(ap2);
    return out;
}

} // namespace

std::string validate_slot_budget(int slots, int extra_held_beyond_slots, int input_queue) {
    const long long hold = plane_frames_held(slots, extra_held_beyond_slots, input_queue);
    if (hold > kMaxDecoderFramesHeld) {
        const std::string queued = input_queue > 0 ? format(" + %d in the input queue", input_queue) : "";
        return format("--display on: decoder frames held would be %lld, exceeding %d "
                      "(slots %d + 1 awaiting decode + %d held by display%s).\nThe decoder stalls without an error when more than about 13 frames are held at once. "
                      "Reduce --slots%s.\n",
                      hold, kMaxDecoderFramesHeld, slots, extra_held_beyond_slots, queued.c_str(),
                      input_queue > 0 ? " or --input-queue" : "");
    }
    return {};
}

} // namespace yvo::pipeline
