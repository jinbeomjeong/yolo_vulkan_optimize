# Vulkan 컴퓨트 전처리: V3D 컨텍스트, dma-buf import, SAND128 → letterbox RGB888
add_library(yvo_vulkan STATIC
  vulkan_context.cpp
  dmabuf_import.cpp
  dmabuf_export.cpp
  sand_letterbox.cpp
  yolo_postproc.cpp)
target_include_directories(yvo_vulkan PUBLIC ${PROJECT_SOURCE_DIR}/01_src)
target_link_libraries(yvo_vulkan
  PUBLIC yvo_common Vulkan::Vulkan
  PRIVATE yvo_shaders)
yvo_target_defaults(yvo_vulkan)

# 성능·검증 도구: 전체 디코드 + GPU 전처리 (디코더가 있을 때만)
if(EXISTS ${PROJECT_SOURCE_DIR}/01_src/pipeline/CMakeLists.txt)  # yvo_decoder는 01_src/pipeline에서 정의
  yvo_add_tool(yvo_preproc_bench tools/preproc_bench.cpp
    LIBS yvo_vulkan yvo_decoder
    OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)
endif()

# 후처리 프로브 (계획서 §6.2 P3).
#   --probe fixture|sweep : NPU 불필요 (04_data/raw_tensors 픽스처)
#   --probe live|stale    : NPU 독점. HailoRT 가 있을 때만 켜진다.
# yvo_hailo 라이브러리에 링크해도 **yvo_vulkan 자체의 의존성은 늘지 않는다**
# (계획서 §5.2 링크 수준 격리 — test_preproc / yvo_preproc_bench 의 링크 그래프가 바뀌지 않는다).
# yvo_shaders: --probe heapimport 가 쓰는 heap_sum.comp 임베드 헤더 (부록 B 1단계)
yvo_add_tool(yvo_postproc_probe tools/postproc_probe.cpp
  LIBS yvo_vulkan yvo_shaders
  OPTIONAL yvo_hailo=YVO_HAVE_HAILO)
