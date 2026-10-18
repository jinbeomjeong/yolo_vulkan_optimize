# GLSL compute shaders → SPIR-V(build/02_shaders/*.spv) → embedded headers(build/02_shaders/generated/*_spv.h)
# Usage: target_link_libraries(<target> PRIVATE yvo_shaders), then #include "sand_letterbox_spv.h"
# 4 main pipeline shaders. overlay_boxes draws into the packed RGBA8 dma-buf of the HVS overlay plane (OverlayPlane).
yvo_add_shaders(yvo_shaders
  sand_letterbox.comp
  overlay_boxes.comp
  yolo_decode.comp
  efficient_nms.comp
)
