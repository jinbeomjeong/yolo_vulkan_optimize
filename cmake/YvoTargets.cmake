# Shared target settings: warnings, CPU flags, OpenCV, tool registration helper.
# The root CMakeLists.txt includes this **after** find_package(OpenCV) (yvo_opencv checks OpenCV_FOUND).
#
# Why consolidate in one place (2026-09-22):
#   -mcpu=cortex-a76 was repeated 20 times via target_compile_options, and was missing from yvo_decoder and yvo_drm_dump.
#   The 3-line OpenCV link/include/YVO_HAVE_OPENCV block was also duplicated per target.

# ── Warnings ─────────────────────────────────────────────────────────────────────
# Each target uses it via target_link_libraries(... PRIVATE yvo_warnings) (yvo_target_defaults attaches it).
add_library(yvo_warnings INTERFACE)
# -Wshadow / -Wnon-virtual-dtor added 2026-10-11 (0 occurrences at the time; -Wconversion / -Wold-style-cast deliberately not).
# -Wnon-virtual-dtor is C++-only, so it is limited to CXX sources in case a C target ever links this.
target_compile_options(yvo_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow
  $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>)

# ── Sanitizers ───────────────────────────────────────────────────────────────────
# -DYVO_SANITIZE=ON adds ASan + UBSan to compile **and** link of every target defined after this include (all of
# 01_src/, 02_shaders/, 05_tests/; C and C++, so the generated wayland-scanner C stubs too). Default OFF: the default
# build is unchanged. Always use a separate build directory (e.g. build-asan/) — a sanitized build/ ruins measurements.
# TSan is not offered here: it cannot be combined with ASan.
option(YVO_SANITIZE "Build every target with -fsanitize=address,undefined (use a separate build directory)" OFF)
if(YVO_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
  message(STATUS "YVO_SANITIZE: ASan + UBSan on every target")
endif()

# ── CPU flags ───────────────────────────────────────────────────────────────
# Pi 5 = Cortex-A76. When building on another board, change it like -DYVO_CPU_FLAGS=-mcpu=native.
set(YVO_CPU_FLAGS "-mcpu=cortex-a76" CACHE STRING "CPU tuning flags added to C++ targets (semicolon/space separated)")

# yvo_target_defaults(<target>): attaches warnings (yvo_warnings) + YVO_CPU_FLAGS as PRIVATE.
function(yvo_target_defaults tgt)
  separate_arguments(_yvo_cpu_flags UNIX_COMMAND "${YVO_CPU_FLAGS}")
  target_link_libraries(${tgt} PRIVATE yvo_warnings)
  target_compile_options(${tgt} PRIVATE ${_yvo_cpu_flags})
endfunction()

# ── OpenCV ───────────────────────────────────────────────────────────────────
# Defined only when OpenCV is present → consumers branch with if(TARGET yvo_opencv).
# YVO_HAVE_OPENCV is not defined here: only targets that use OpenCV optionally receive that macro
# (attaching it to targets that require OpenCV would change the #ifdef path in tool_image.hpp and thus the code).
if(OpenCV_FOUND)
  add_library(yvo_opencv INTERFACE)
  target_link_libraries(yvo_opencv INTERFACE ${OpenCV_LIBS})
  target_include_directories(yvo_opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
endif()

# ── Asset paths ─────────────────────────────────────────────────────────────────
# Gives tools/apps/tests their default input/output paths as **absolute-path compile definitions** (2026-09-22).
# Previously CWD-relative string literals like 04_data/videos/video_h265.mp4 existed in 14 places, so they only worked when run from the repo root.
#   YVO_REPO_ROOT      repository root (= PROJECT_SOURCE_DIR). Default tool output is YVO_REPO_ROOT "/build/out/<tool>"
#   YVO_DEFAULT_HEF    default model
#   YVO_DEFAULT_VIDEO  default input video
#   YVO_FIXTURE_DIR    GPU postprocessing fixture (raw tensor) folder
# **Libraries do not link it** — libraries do not know asset paths; the caller passes them (e.g. HailoRunnerConfig::hef_path).
set(YVO_DEFAULT_HEF "${PROJECT_SOURCE_DIR}/03_models/yolov11m.hef" CACHE FILEPATH "Default HEF for tools/apps/tests")
set(YVO_DEFAULT_VIDEO "${PROJECT_SOURCE_DIR}/04_data/videos/video_h265.mp4" CACHE FILEPATH "Default input video for tools/apps/tests")
set(YVO_FIXTURE_DIR "${PROJECT_SOURCE_DIR}/04_data/raw_tensors" CACHE PATH "GPU postprocessing fixture (raw tensor) folder")
add_library(yvo_assets INTERFACE)
target_compile_definitions(yvo_assets INTERFACE
  YVO_REPO_ROOT="${PROJECT_SOURCE_DIR}"
  YVO_DEFAULT_HEF="${YVO_DEFAULT_HEF}"
  YVO_DEFAULT_VIDEO="${YVO_DEFAULT_VIDEO}"
  YVO_FIXTURE_DIR="${YVO_FIXTURE_DIR}")

# ── Tool registration ─────────────────────────────────────────────────────────────────
# yvo_add_tool(<name> <src>
#   [REQUIRES <target>...]        if any of these is missing, the tool is not built (STATUS message only)
#   [LIBS <lib>...]               PRIVATE link (order preserved)
#   [DEFINES <def>...]            PRIVATE compile definitions
#   [OPTIONAL <target>=<MACRO>...] if the target exists, link it and define <MACRO>=1 (linked after LIBS)
# )
# Every tool gets yvo_target_defaults and yvo_assets.
function(yvo_add_tool name src)
  cmake_parse_arguments(PARSE_ARGV 2 T "" "" "REQUIRES;LIBS;DEFINES;OPTIONAL")
  foreach(req IN LISTS T_REQUIRES)
    if(NOT TARGET ${req})
      message(STATUS "${name}: ${req} not found — skipping")
      return()
    endif()
  endforeach()
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE ${T_LIBS} yvo_assets)
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
