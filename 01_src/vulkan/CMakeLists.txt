# Vulkan 컴퓨트 전처리: V3D 컨텍스트, dma-buf import, SAND128 → letterbox RGB888
add_library(yvo_vulkan STATIC
  vulkan_context.cpp
  dmabuf_import.cpp
  dmabuf_export.cpp
  sand_letterbox.cpp
  yolo_postproc.cpp)
target_link_libraries(yvo_vulkan
  PUBLIC yvo_core yvo_ref Vulkan::Vulkan
  PRIVATE yvo_shaders)
yvo_target_defaults(yvo_vulkan)
