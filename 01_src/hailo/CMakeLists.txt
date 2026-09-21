# HailoRT 추론 러너
#   hailo_runner      : InferModel 비동기 API + HEF 내장 NMS (호스트 후처리 기준 경로)
#   hailo_raw_runner  : 원시 출력 6스트림 + 입출력 dma-buf (GPU 후처리 경로, F0 관문 통과분)
# **yvo_hailo 는 yvo_vulkan 에 의존하지 않는다** — 계획서 §5.2의 링크 수준 격리.
# hailo_raw_runner 는 dma-buf fd 와 크기만 받고 Vulkan 타입을 알지 못한다.
add_library(yvo_hailo STATIC hailo_runner.cpp hailo_raw_runner.cpp)
target_link_libraries(yvo_hailo PUBLIC HailoRT::libhailort)
target_link_libraries(yvo_hailo PUBLIC yvo_core PRIVATE yvo_ref)
yvo_target_defaults(yvo_hailo)
