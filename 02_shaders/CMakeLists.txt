# GLSL compute shaders → SPIR-V(build/02_shaders/*.spv) → embedded headers(build/02_shaders/generated/*_spv.h)
# Usage: target_link_libraries(<target> PRIVATE yvo_shaders), then #include "sand_letterbox_spv.h"
# 5 main pipeline shaders.
yvo_add_shaders(yvo_shaders
  sand_letterbox.comp
  display_convert.comp
  overlay_boxes.comp
  yolo_decode.comp
  efficient_nms.comp
)
