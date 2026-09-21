# 도구·앱 전용 헬퍼(헤더 전용): tool_util(CLI·타이머), tool_image(PNG/PPM 저장), stats(중앙값·백분위).
# **라이브러리(yvo_hailo·yvo_vulkan·yvo_display·yvo_isp·yvo_decoder·yvo_stages)는 include 하지 않는다.**
# 도구(01_src/tools/)·앱(pipeline/main.cpp)·테스트만 링크한다.
add_library(yvo_toolkit INTERFACE)
target_link_libraries(yvo_toolkit INTERFACE yvo_core yvo_ref)
