# Vulkan compute preprocessing: V3D context, dma-buf import, SAND128 → letterbox RGB888
add_library(yvo_vulkan STATIC
  vulkan_context.cpp
  dmabuf_import.cpp
  dmabuf_export.cpp
  sand_letterbox.cpp
  yolo_postproc.cpp
  efficient_nms.cpp)
target_link_libraries(yvo_vulkan
  PUBLIC yvo_core yvo_ref Vulkan::Vulkan
  PRIVATE yvo_shaders)
yvo_target_defaults(yvo_vulkan)
