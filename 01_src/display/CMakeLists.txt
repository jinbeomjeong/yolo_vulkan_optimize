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
  "xdg-shell=stable/xdg-shell/xdg-shell.xml"                   # 창 (필수)
  "linux-dmabuf=stable/linux-dmabuf/linux-dmabuf-v1.xml"       # wayland-planes: 디코더 SAND dma-buf → wl_buffer
  "presentation-time=stable/presentation-time/presentation-time.xml"  # wayland-planes: 표시 시각(glass-to-glass)
  "viewporter=stable/viewporter/viewporter.xml")               # wayland-planes: 출력 ≠ 원본 크기일 때 HVS 스케일
# 선택 프로토콜: XML 이 없어도 yvo_display 는 빌드된다. 있으면 YVO_HAVE_<매크로>=1 을 정의한다.
#   color-representation-v1 은 staging 이라 wayland-protocols 버전에 따라 없을 수 있다(1.44+).
set(yvo_wl_optional_protocols
  "color-representation=staging/color-representation/color-representation-v1.xml=YVO_HAVE_COLOR_REPRESENTATION")
set(yvo_wl_defines "")
foreach(entry IN LISTS yvo_wl_optional_protocols)
  string(REPLACE "=" ";" parts "${entry}")
  list(GET parts 0 name)
  list(GET parts 1 rel)
  list(GET parts 2 macro)
  if(EXISTS "${YVO_WL_PROTO_DIR}/${rel}")
    list(APPEND yvo_wl_protocols "${name}=${rel}")
    list(APPEND yvo_wl_defines "${macro}=1")
  else()
    message(STATUS "yvo_display: 선택 프로토콜 ${rel} 없음 — ${macro} 없이 빌드한다")
  endif()
endforeach()

# 헤더는 generated/display/ 에 둔다: 새 소스는 "display/<이름>-client-protocol.h" 로 include 해서 include 계층 규칙
# (STRUCTURE.md §2.2, check_structure C4 "<계층>/<파일>")을 지킨다. 기존 xdg-shell 은 접두사 없이 쓰던 그대로
# (structure_rules.toml include_unprefixed 예외) generated/display 도 include 경로에 넣는다.
set(yvo_wl_gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(yvo_wl_hdr_dir ${yvo_wl_gen_dir}/display)
file(MAKE_DIRECTORY ${yvo_wl_hdr_dir})
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
  set(hdr ${yvo_wl_hdr_dir}/${name}-client-protocol.h)
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
target_include_directories(yvo_wayland_protocols SYSTEM PUBLIC ${yvo_wl_gen_dir} ${yvo_wl_hdr_dir})
target_link_libraries(yvo_wayland_protocols PUBLIC PkgConfig::YVO_WAYLAND)
set_target_properties(yvo_wayland_protocols PROPERTIES POSITION_INDEPENDENT_CODE ON)
# 헤더는 소스와 별개 커스텀 커맨드라 명시적으로 빌드 순서를 걸어 준다.
add_custom_target(yvo_wayland_protocols_headers DEPENDS ${yvo_wl_headers})
add_dependencies(yvo_wayland_protocols yvo_wayland_protocols_headers)

# kms 백엔드(kms_scanout.cpp)는 libdrm atomic API 를 쓴다. libdrm 은 decode/ 에서 이미 REQUIRED 로 찾았다.
# pkg_check_modules 의 IMPORTED 타깃은 디렉터리 범위라 여기서 다시 부른다(캐시된 결과로 타깃만 만든다).
pkg_check_modules(YVO_LIBDRM REQUIRED IMPORTED_TARGET libdrm)

add_library(yvo_display STATIC
  wayland_window.cpp
  drm_state.cpp
  plane_display.cpp
  wayland_plane_scanout.cpp
  kms_scanout.cpp
  vk_swapchain.cpp
  display_convert.cpp
  overlay_renderer.cpp
  overlay_plane.cpp
  offscreen_backend.cpp
  display_pipeline.cpp)
target_link_libraries(yvo_display
  PUBLIC yvo_vulkan yvo_ref
  PRIVATE PkgConfig::YVO_WAYLAND PkgConfig::YVO_LIBDRM yvo_wayland_protocols yvo_shaders)
yvo_target_defaults(yvo_display)
# 선택 프로토콜 매크로는 이 라이브러리 소스 안에서만 쓴다(공개 헤더는 매크로에 의존하지 않는다).
target_compile_definitions(yvo_display PRIVATE ${yvo_wl_defines})
