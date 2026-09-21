# 공용 타깃 설정: 경고·CPU 플래그·OpenCV·도구 등록 헬퍼.
# 루트 CMakeLists.txt 가 find_package(OpenCV) **뒤에** include 한다(yvo_opencv 가 OpenCV_FOUND 를 본다).
#
# 왜 한 곳에 모았나 (2026-09-22):
#   -mcpu=cortex-a76 이 target_compile_options 로 20번 반복돼 있었고, yvo_decoder·yvo_drm_dump 에는 빠져 있었다.
#   OpenCV 링크·include·YVO_HAVE_OPENCV 3줄 블록도 타깃마다 복제돼 있었다.

# ── 경고 ─────────────────────────────────────────────────────────────────────
# 각 타깃에서 target_link_libraries(... PRIVATE yvo_warnings)로 쓴다(yvo_target_defaults 가 붙여 준다).
add_library(yvo_warnings INTERFACE)
target_compile_options(yvo_warnings INTERFACE -Wall -Wextra -Wpedantic)

# ── CPU 플래그 ───────────────────────────────────────────────────────────────
# Pi 5 = Cortex-A76. 다른 보드에서 빌드할 때는 -DYVO_CPU_FLAGS=-mcpu=native 처럼 바꾼다.
set(YVO_CPU_FLAGS "-mcpu=cortex-a76" CACHE STRING "C++ 타깃에 붙는 CPU 튜닝 플래그 (세미콜론/공백 구분)")

# yvo_target_defaults(<target>): 경고(yvo_warnings) + YVO_CPU_FLAGS 를 PRIVATE 로 붙인다.
function(yvo_target_defaults tgt)
  separate_arguments(_yvo_cpu_flags UNIX_COMMAND "${YVO_CPU_FLAGS}")
  target_link_libraries(${tgt} PRIVATE yvo_warnings)
  target_compile_options(${tgt} PRIVATE ${_yvo_cpu_flags})
endfunction()

# ── OpenCV ───────────────────────────────────────────────────────────────────
# OpenCV 가 있을 때만 정의된다 → 쓰는 쪽은 if(TARGET yvo_opencv) 로 분기한다.
# YVO_HAVE_OPENCV 는 여기서 정의하지 않는다: OpenCV 를 선택적으로 쓰는 타깃만 그 매크로를 받는다
# (필수로 쓰는 타깃에 붙이면 tool_image.hpp 의 #ifdef 경로가 바뀌어 코드가 달라진다).
if(OpenCV_FOUND)
  add_library(yvo_opencv INTERFACE)
  target_link_libraries(yvo_opencv INTERFACE ${OpenCV_LIBS})
  target_include_directories(yvo_opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
endif()

# ── 도구 등록 ─────────────────────────────────────────────────────────────────
# yvo_add_tool(<name> <src>
#   [REQUIRES <target>...]        이 중 하나라도 없으면 도구를 만들지 않는다(STATUS 메시지만)
#   [LIBS <lib>...]               PRIVATE 링크(순서 유지)
#   [DEFINES <def>...]            PRIVATE 컴파일 정의
#   [OPTIONAL <target>=<MACRO>...] 타깃이 있으면 링크하고 <MACRO>=1 을 정의한다(LIBS 뒤에 링크)
# )
# 모든 도구에 yvo_target_defaults 가 붙는다.
function(yvo_add_tool name src)
  cmake_parse_arguments(PARSE_ARGV 2 T "" "" "REQUIRES;LIBS;DEFINES;OPTIONAL")
  foreach(req IN LISTS T_REQUIRES)
    if(NOT TARGET ${req})
      message(STATUS "${name}: ${req} 없음 — 건너뛴다")
      return()
    endif()
  endforeach()
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE ${T_LIBS})
  yvo_target_defaults(${name})
  if(T_DEFINES)
    target_compile_definitions(${name} PRIVATE ${T_DEFINES})
  endif()
  foreach(opt IN LISTS T_OPTIONAL)
    string(REPLACE "=" ";" kv "${opt}")
    list(GET kv 0 opt_tgt)
    list(GET kv 1 opt_macro)
    if(TARGET ${opt_tgt})
      target_link_libraries(${name} PRIVATE ${opt_tgt})
      target_compile_definitions(${name} PRIVATE ${opt_macro}=1)
    endif()
  endforeach()
endfunction()
