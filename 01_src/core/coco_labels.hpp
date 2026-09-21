// COCO 80클래스 라벨과 클래스별 색 — 정의를 한 곳에 모은다.
//
// 이 헤더가 존재하는 이유: 같은 표가 01_src/pipeline/main.cpp 와 01_src/tools/hailo/hailo_selftest.cpp 에
// 각각 복사돼 있었고, 클래스 색 공식도 두 곳에 흩어져 있었다. GPU 오버레이 셰이더(CPU 기준 구현 포함)가
// 같은 색을 내야 하므로 정의가 갈라질 여지를 구조적으로 없앤다.
//
// 색 규약:
//   class_color() 는 RGB 순서다(GPU storage image 가 R8G8B8A8_UNORM 이므로 RGB 가 자연스럽다).
//   원본 규칙은 OpenCV BGR 로 쓰여 있었다:
//     cv::Scalar((c*67)%256, (c*131+80)%256, 255-(c*29)%256)   // (B, G, R)
//   따라서 RGB 로는
//     R = 255 - (c*29)%256,  G = (c*131+80)%256,  B = (c*67)%256
//   OpenCV 로 되돌릴 때는 cv::Scalar(col[2], col[1], col[0]) 로 뒤집는다.
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

// 범위를 벗어난 클래스 번호는 "?" 로 돌려준다(기존 main.cpp 의 동작과 같다).
inline constexpr const char* class_name(int c) {
    return (c >= 0 && c < kNumCocoClasses) ? kCoco[c] : "?";
}

// 클래스별 박스 색 (R, G, B). 위 주석의 공식을 그대로 옮긴 것이다.
inline constexpr std::array<std::uint8_t, 3> class_color(int c) {
    const int u = c < 0 ? 0 : c;
    return {static_cast<std::uint8_t>(255 - (u * 29) % 256), static_cast<std::uint8_t>((u * 131 + 80) % 256),
            static_cast<std::uint8_t>((u * 67) % 256)};
}

} // namespace yvo
