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

std::string validate_slot_budget(SlotBudgetPath path, bool latest_sync, int slots, int gpu_depth,
                                 int extra_held_beyond_slots, int input_queue) {
    // Sums are taken in long long so absurd --slots/--gpu-depth values cannot overflow int (identical result otherwise).
    if (path == SlotBudgetPath::WaylandPlanes || path == SlotBudgetPath::Kms) {
        // matched: every frame held by the display occupies its own slot → slots + 1 (one decoded frame awaiting a slot).
        // latest : the slot comes back right after show, and the display holds extra_held_beyond_slots more outside the slots.
        const long long hold = plane_frames_held(latest_sync, slots, extra_held_beyond_slots, input_queue);
        if (hold > kMaxDecoderFramesHeld) {
            const std::string by_display = latest_sync ? format(" + %d held by display", extra_held_beyond_slots) : "";
            const std::string queued = input_queue > 0 ? format(" + %d in the input queue", input_queue) : "";
            return format("--display-backend %s: decoder frames held would be %lld, exceeding %d "
                          "(slots %d + 1 awaiting decode%s%s).\nThe decoder stalls without an error when more than about 13 frames are held at once. "
                          "Reduce --slots%s.\n",
                          path == SlotBudgetPath::Kms ? "kms" : "wayland-planes", hold, kMaxDecoderFramesHeld, slots,
                          by_display.c_str(), queued.c_str(), input_queue > 0 ? " or --input-queue" : "");
        }
        return {};
    }
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
