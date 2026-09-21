# GLSL 컴퓨트 셰이더 → SPIR-V(build/02_shaders/*.spv) → 임베드 헤더(build/02_shaders/generated/*_spv.h)
# 사용: target_link_libraries(<target> PRIVATE yvo_shaders) 후 #include "sand_letterbox_spv.h"
# 본 파이프라인 셰이더 5개. 진단 전용 셰이더는 diag/ (yvo_shaders_diag, 도구만 링크한다).
yvo_add_shaders(yvo_shaders
  sand_letterbox.comp
  display_convert.comp
  overlay_boxes.comp
  yolo_decode.comp
  efficient_nms.comp
)

add_subdirectory(diag)
