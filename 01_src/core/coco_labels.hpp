// COCO 80-class labels and per-class colors — the definitions live in one place.
//
// Why this header exists: the same table was copied separately into 01_src/pipeline/main.cpp and
// 01_src/tools/hailo/hailo_selftest.cpp, and the class color formula was scattered in two places too. The GPU overlay shader (including its CPU reference implementation)
// must produce the same colors, so this structurally removes any room for the definitions to diverge.
//
// Color convention:
//   class_color() is in RGB order (the GPU storage image is R8G8B8A8_UNORM, so RGB is natural).
//   The original rule was written in OpenCV BGR:
//     cv::Scalar((c*67)%256, (c*131+80)%256, 255-(c*29)%256)   // (B, G, R)
//   so in RGB it is
//     R = 255 - (c*29)%256,  G = (c*131+80)%256,  B = (c*67)%256
//   To go back to OpenCV, flip it with cv::Scalar(col[2], col[1], col[0]).
#pragma once

#include <array>
#include <cstdint>

namespace yvo {

inline constexpr int kNumCocoClasses = 80;

inline constexpr const char* kCoco[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"};

// An out-of-range class id returns "?" (same as the previous behavior in main.cpp).
inline constexpr const char* class_name(int c) {
    return (c >= 0 && c < kNumCocoClasses) ? kCoco[c] : "?";
}

// Per-class box color (R, G, B). A direct transcription of the formula in the comment above.
inline constexpr std::array<std::uint8_t, 3> class_color(int c) {
    const int u = c < 0 ? 0 : c;
    return {static_cast<std::uint8_t>(255 - (u * 29) % 256), static_cast<std::uint8_t>((u * 131 + 80) % 256),
            static_cast<std::uint8_t>((u * 67) % 256)};
}

} // namespace yvo
