# 테스트·프로브가 공유하는 검증 하네스(헤더 전용). 라이브러리는 include 하지 않는다.
#   yvo_testkit_postproc : postproc_fixture.hpp — 04_data/raw_tensors 픽스처 파서 + GPU/CPU 비교
#                          (test_postproc 와 yvo_postproc_probe 가 공유)
#   yvo_testkit_display  : offscreen_target.hpp, display_convert_check.hpp — 오프스크린 렌더 타깃·되읽기,
#                          표시 변환 비트 일치 판정 (test_overlay·test_display_convert 와 yvo_display_probe 가 공유)
add_library(yvo_testkit_postproc INTERFACE)
target_link_libraries(yvo_testkit_postproc INTERFACE yvo_vulkan yvo_ref)

if(TARGET yvo_display)
  add_library(yvo_testkit_display INTERFACE)
  target_link_libraries(yvo_testkit_display INTERFACE yvo_display yvo_ref)
endif()
