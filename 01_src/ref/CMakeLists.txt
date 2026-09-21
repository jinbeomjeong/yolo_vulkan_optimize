# CPU 기준 구현(헤더 전용): cpu_letterbox, sand, cpu_yolov8_decode, detection_match, overlay_reference, glyph_font_8x16.
# **수식의 단일 진실 원천**이다. GPU 셰이더(02_shaders/)는 이 헤더들의 번역본이고 05_tests 가 비트 단위로 대조한다.
add_library(yvo_ref INTERFACE)
target_link_libraries(yvo_ref INTERFACE yvo_core)
