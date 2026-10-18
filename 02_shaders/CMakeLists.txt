# GLSL compute shaders → SPIR-V(build/02_shaders/*.spv) → embedded headers(build/02_shaders/generated/*_spv.h)
# Usage: target_link_libraries(<target> PRIVATE yvo_shaders), then #include "sand_letterbox_spv.h"
# 5 main pipeline shaders (+ 1 buffer variant of overlay_boxes).
yvo_add_shaders(yvo_shaders
  sand_letterbox.comp
  display_convert.comp
  overlay_boxes.comp
  # storage-buffer variant of the same source: draws into the packed RGBA8 dma-buf for the HVS overlay plane (OverlayPlane)
  overlay_boxes.comp@overlay_boxes_buf@YVO_OVERLAY_BUFFER
  yolo_decode.comp
  efficient_nms.comp
)
