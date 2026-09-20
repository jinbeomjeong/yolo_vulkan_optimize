# Vulkan HDMI 표시 경로 (Wayland xdg-shell 창 + 스왑체인). 회귀 격리를 위해 yvo_vulkan 과 **별도 라이브러리**다
# (계획서 §5.2 안전장치 4): test_preproc / yvo_preproc_bench 의 링크 그래프가 바뀌지 않는다.
option(YVO_BUILD_DISPLAY "Vulkan HDMI 표시 경로 빌드" ON)
if(NOT YVO_BUILD_DISPLAY)
  message(STATUS "yvo_display: YVO_BUILD_DISPLAY=OFF — 건너뛴다")
  return()
endif()

find_package(PkgConfig QUIET)
if(NOT PkgConfig_FOUND)
  message(STATUS "yvo_display: pkg-config 없음 — 건너뛴다")
  return()
endif()

pkg_check_modules(YVO_WAYLAND IMPORTED_TARGET wayland-client)     # REQUIRED 아님: 없으면 조용히 건너뛴다
if(NOT YVO_WAYLAND_FOUND)
  message(STATUS "yvo_display: wayland-client 없음 — 건너뛴다")
  return()
endif()

find_program(YVO_WAYLAND_SCANNER wayland-scanner)
pkg_get_variable(YVO_WL_PROTO_DIR wayland-protocols pkgdatadir)
if(NOT YVO_WAYLAND_SCANNER OR NOT YVO_WL_PROTO_DIR)
  message(STATUS "yvo_display: wayland-scanner 또는 wayland-protocols 없음 — 건너뛴다")
  return()
endif()
# pkg-config 이 pkgdatadir 을 "//usr/share/wayland-protocols"(선행 슬래시 2개)로 돌려준다 → 정규화한다.
string(REGEX REPLACE "^//+" "/" YVO_WL_PROTO_DIR "${YVO_WL_PROTO_DIR}")
file(TO_CMAKE_PATH "${YVO_WL_PROTO_DIR}" YVO_WL_PROTO_DIR)

# wayland-scanner 로 클라이언트 스텁 생성 (yvo_add_shaders() 가 glslc 를 add_custom_command 로 돌리는 것과 같은 패턴).
# 프로토콜을 추가할 때는 아래 목록에 "<이름>=<xml 상대경로>" 한 줄만 더한다.
set(yvo_wl_protocols
  "xdg-shell=stable/xdg-shell/xdg-shell.xml"           # 창 (필수)
  "viewporter=stable/viewporter/viewporter.xml"        # HVS 하드웨어 확대 (다음 단계)
  "presentation-time=stable/presentation-time/presentation-time.xml")  # 표시 시각 (present_wait 폴백)

set(yvo_wl_gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${yvo_wl_gen_dir})
set(yvo_wl_sources "")
set(yvo_wl_headers "")
foreach(entry IN LISTS yvo_wl_protocols)
  string(REPLACE "=" ";" parts "${entry}")
  list(GET parts 0 name)
  list(GET parts 1 rel)
  set(xml "${YVO_WL_PROTO_DIR}/${rel}")
  if(NOT EXISTS "${xml}")
    message(STATUS "yvo_display: ${xml} 없음 — 건너뛴다")
    return()
  endif()
  set(hdr ${yvo_wl_gen_dir}/${name}-client-protocol.h)
  set(src ${yvo_wl_gen_dir}/${name}-protocol.c)
  add_custom_command(OUTPUT ${hdr}
    COMMAND ${YVO_WAYLAND_SCANNER} client-header ${xml} ${hdr}
    DEPENDS ${xml} COMMENT "wayland-scanner client-header ${name}" VERBATIM)
  add_custom_command(OUTPUT ${src}
    COMMAND ${YVO_WAYLAND_SCANNER} private-code ${xml} ${src}
    DEPENDS ${xml} COMMENT "wayland-scanner private-code ${name}" VERBATIM)
  list(APPEND yvo_wl_headers ${hdr})
  list(APPEND yvo_wl_sources ${src})
endforeach()

# 생성된 C 는 C 언어로 컴파일해야 한다: private-code 가 file-scope 에 extern "C" 없이
# `extern const struct wl_interface wl_surface_interface;` 같은 선언을 내놓으므로 C++ 로 빌드하면
# libwayland-client 의 C 심볼과 링크되지 않는다.
enable_language(C)

# ⚠ 생성된 C 는 -Wall -Wextra -Wpedantic 을 통과하지 못한다(빈 초기화 항목 등).
#   따라서 **yvo_warnings 를 붙이지 않고** include 디렉터리를 SYSTEM 으로 준다.
#   "빌드 경고 0" 기준을 지키는 유일한 방법이다.
add_library(yvo_wayland_protocols STATIC ${yvo_wl_sources})
target_include_directories(yvo_wayland_protocols SYSTEM PUBLIC ${yvo_wl_gen_dir})
target_link_libraries(yvo_wayland_protocols PUBLIC PkgConfig::YVO_WAYLAND)
set_target_properties(yvo_wayland_protocols PROPERTIES POSITION_INDEPENDENT_CODE ON)
# 헤더는 소스와 별개 커스텀 커맨드라 명시적으로 빌드 순서를 걸어 준다.
add_custom_target(yvo_wayland_protocols_headers DEPENDS ${yvo_wl_headers})
add_dependencies(yvo_wayland_protocols yvo_wayland_protocols_headers)

add_library(yvo_display STATIC
  wayland_window.cpp
  vk_swapchain.cpp
  display_convert.cpp
  overlay_renderer.cpp
  offscreen_backend.cpp
  display_pipeline.cpp)
target_include_directories(yvo_display PUBLIC ${PROJECT_SOURCE_DIR}/01_src)
target_link_libraries(yvo_display
  PUBLIC yvo_vulkan
  PRIVATE PkgConfig::YVO_WAYLAND yvo_wayland_protocols yvo_shaders yvo_warnings)
target_compile_options(yvo_display PRIVATE -mcpu=cortex-a76)

# 프로브: 계획서 §6.3 (P4 surface / P5 present / P6 context-regression / P8 convert).
# P8 은 디코더 SAND dma-buf 를 입력으로 쓰므로 yvo_decoder 가 있을 때만 그 서브커맨드가 켜진다.
add_executable(yvo_display_probe tools/display_probe.cpp)
target_link_libraries(yvo_display_probe PRIVATE yvo_display yvo_shaders yvo_warnings)
target_compile_options(yvo_display_probe PRIVATE -mcpu=cortex-a76)
if(EXISTS ${PROJECT_SOURCE_DIR}/01_src/pipeline/CMakeLists.txt)   # yvo_decoder 는 01_src/pipeline 에서 정의
  target_link_libraries(yvo_display_probe PRIVATE yvo_decoder)
  target_compile_definitions(yvo_display_probe PRIVATE YVO_HAVE_DECODER=1)
endif()

# DisplayPipeline 단독 프로브 (마일스톤 7). yvo_pipeline 을 건드리지 않고 표시 경로만 재현한다:
# display_probe 의 run_convert_display() 수치를 DisplayPipeline 경유로 다시 내는 것이 목적이다.
if(EXISTS ${PROJECT_SOURCE_DIR}/01_src/pipeline/CMakeLists.txt)
  add_executable(yvo_display_pipeline_probe tools/display_pipeline_probe.cpp)
  target_link_libraries(yvo_display_pipeline_probe PRIVATE yvo_display yvo_decoder yvo_warnings)
  target_compile_options(yvo_display_pipeline_probe PRIVATE -mcpu=cortex-a76)
endif()
