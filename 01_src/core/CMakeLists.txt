# 공용 계약 타입·OS 글루(헤더 전용): frame_types, coco_labels, dma_heap_alloc, dmabuf_map, hailo_align.
# **include 루트(${PROJECT_SOURCE_DIR}/01_src)는 여기서만 지정한다.** 모든 타깃이 yvo_core 를 거쳐
# "폴더/파일.hpp" 표기로 include 한다.
# 옛 이름 yvo_common / 01_src/common/ 은 2026-09-22 에 core·ref·toolkit 으로 나눴다
# (libpisp 내부 헤더가 "common/logging.hpp" 를 상대 경로 없이 include 해서 이름 충돌 위험이 있었다).
add_library(yvo_core INTERFACE)
target_include_directories(yvo_core INTERFACE ${PROJECT_SOURCE_DIR}/01_src)
target_compile_features(yvo_core INTERFACE cxx_std_20)
