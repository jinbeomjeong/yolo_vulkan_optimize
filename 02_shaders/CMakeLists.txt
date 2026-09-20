# GLSL 컴퓨트 셰이더 → SPIR-V(build/02_shaders/*.spv) → 임베드 헤더(build/02_shaders/generated/*_spv.h)
# 사용: target_link_libraries(<target> PRIVATE yvo_shaders) 후 #include "sand_letterbox_spv.h"
yvo_add_shaders(yvo_shaders
  sand_letterbox.comp
  display_convert.comp
  display_probe.comp
  overlay_boxes.comp
  yolo_decode.comp
  heap_sum.comp
)
