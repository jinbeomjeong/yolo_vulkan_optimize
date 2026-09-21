# yvo_decoder 는 01_src/decode/ 로 옮겼다(2026-09-22).

# 추론 단계(InferStage 구현). **OpenCV 에 의존하지 않는다** — 그래야 05_tests 가 링크할 수 있다.
#   infer_stage.hpp   : 헤더 전용(HostNmsStage 포함)
#   gpu_postproc_stage: HailoRawRunner + YoloPostproc + cpu_nms 글루
# 원래 yvo_pipeline 실행 파일에만 컴파일돼 있어 테스트가 이 글루를 링크할 수 없었다.
# **로직은 그대로 두고 배선만 라이브러리로 뽑았다**(05_tests/test_postproc.cpp 의 PostprocLive 가 쓴다).
if(TARGET yvo_vulkan AND TARGET yvo_hailo)
  add_library(yvo_stages STATIC gpu_postproc_stage.cpp)
  target_link_libraries(yvo_stages PUBLIC yvo_vulkan yvo_hailo PRIVATE yvo_ref)
  yvo_target_defaults(yvo_stages)
endif()

# 통합 앱: HW 디코드 → Vulkan letterbox → HailoRT 추론 (ptr/dmabuf 모드)
if(TARGET yvo_stages AND TARGET yvo_opencv)
  add_executable(yvo_pipeline main.cpp)
  target_link_libraries(yvo_pipeline PRIVATE yvo_decoder yvo_stages yvo_opencv yvo_toolkit yvo_assets)
  yvo_target_defaults(yvo_pipeline)
  # HDMI 표시는 **조건부 링크**다. yvo_display 는 계속 별도 라이브러리로 두어
  # test_preproc / yvo_preproc_bench 의 링크 그래프에 wayland 가 끌려오지 않게 한다.
  # yvo_display 가 없으면(wayland-client/protocols 부재) --display 옵션 자체가 컴파일되지 않는다.
  if(TARGET yvo_display)
    target_link_libraries(yvo_pipeline PRIVATE yvo_display)
    target_compile_definitions(yvo_pipeline PRIVATE YVO_HAVE_DISPLAY=1)
  endif()
endif()
