# HailoRT 추론 러너
#   hailo_runner      : InferModel 비동기 API + HEF 내장 NMS (호스트 후처리 기준 경로)
#   hailo_raw_runner  : 원시 출력 6스트림 + 입출력 dma-buf (GPU 후처리 경로, F0 관문 통과분)
# **yvo_hailo 는 yvo_vulkan 에 의존하지 않는다** — 계획서 §5.2의 링크 수준 격리.
# hailo_raw_runner 는 dma-buf fd 와 크기만 받고 Vulkan 타입을 알지 못한다.
add_library(yvo_hailo STATIC hailo_runner.cpp hailo_raw_runner.cpp)
target_link_libraries(yvo_hailo PUBLIC HailoRT::libhailort)
target_link_libraries(yvo_hailo PUBLIC yvo_common)
yvo_target_defaults(yvo_hailo)

# 자체 검증 도구: CPU letterbox(NV12 BT.709) → HailoRunner → 검출 출력·PNG·처리량
yvo_add_tool(yvo_hailo_selftest tools/hailo_selftest.cpp
  REQUIRES yvo_opencv
  LIBS yvo_hailo yvo_opencv)

# 원시 출력 6스트림 프로브 (계획서 §6.2 P1 + P2).
#   --probe p1: 호스트 malloc 버퍼만 쓴다(GPU 미사용)
#   --probe p2: 출력을 dma-buf 로 받는다 → --alloc vulkan 이 yvo_vulkan(VulkanContext, DmaBufExporter)을 쓴다
# yvo_vulkan 은 01_src/vulkan 에서 정의되고 그 디렉터리는 여기보다 뒤에 추가되지만,
# CMake 는 타깃 이름을 generate 시점에 해석하므로 이렇게 링크해도 된다.
# **yvo_hailo 라이브러리 자체는 yvo_vulkan 에 의존하지 않는다** — 계획서 §5.2의 링크 수준 격리를 지킨다.
yvo_add_tool(yvo_hailo_raw_probe tools/hailo_raw_probe.cpp
  LIBS yvo_hailo yvo_vulkan)
