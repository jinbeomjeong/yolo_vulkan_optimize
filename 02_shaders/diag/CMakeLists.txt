# 진단 전용 셰이더 → build/02_shaders/diag/generated/*_spv.h (생성 헤더 이름은 본 셰이더와 같은 규칙)
#   display_probe.comp : P5 present 관문용 고정 테스트 패턴 (yvo_display_probe)
#   heap_sum.comp      : dma-heap 왕복 바이트 합 리덕션 (yvo_postproc_probe --probe heapimport)
# **라이브러리는 링크하지 않는다** — 도구(01_src/tools/)만 링크한다. 파이프라인에는 쓰이지 않는다.
yvo_add_shaders(yvo_shaders_diag
  display_probe.comp
  heap_sum.comp
)
