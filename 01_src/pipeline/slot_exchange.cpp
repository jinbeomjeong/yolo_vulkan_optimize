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

std::string validate_slot_budget(int slots, int gpu_depth, int input_queue) {
    // Sums are taken in long long so absurd --slots/--gpu-depth values cannot overflow int (identical result otherwise).
    if (input_queue > 0) {
        const long long sum = static_cast<long long>(slots) + gpu_depth + input_queue;
        if (sum > kMaxDecoderFramesHeld)
            return format("--display on --input-pace source requires slots + gpu_depth + input_queue <= %d (now %d + %d + %d = %lld).\n"
                          "The decoder stalls without an error when more than about 13 frames are held at once. "
                          "Reduce --slots, --gpu-depth or --input-queue.\n",
                          kMaxDecoderFramesHeld, slots, gpu_depth, input_queue, sum);
        return {};
    }
    const long long sum = static_cast<long long>(slots) + gpu_depth;
    if (sum > kMaxDecoderFramesHeld) {
        return format("--display on requires slots + gpu_depth <= %d (now %d + %d = %lld).\n"
                      "The decoder stalls without an error when more than about 13 frames are held at once. Reduce --slots or --gpu-depth.\n",
                      kMaxDecoderFramesHeld, slots, gpu_depth, sum);
    }
    return {};
}

} // namespace yvo::pipeline
