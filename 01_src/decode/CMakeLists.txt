# HW HEVC decoder (FFmpeg DRM_PRIME → DrmFrame).
# FFmpeg and libdrm are required dependencies, so yvo_decoder **always exists** — consumers need not wrap it in if(EXISTS)/if(TARGET).
# (Before 2026-09-22 it lived in 01_src/pipeline/, so other folders used the `if(EXISTS .../pipeline/CMakeLists.txt)` workaround in 5 places.)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YVO_FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
# libdrm: headers only (<drm_fourcc.h> format/modifier constants, no drm* calls) → include path, no -ldrm.
pkg_check_modules(YVO_LIBDRM REQUIRED libdrm)

add_library(yvo_decoder STATIC ffmpeg_drm_decoder.cpp)
target_link_libraries(yvo_decoder
  PUBLIC yvo_core PkgConfig::YVO_FFMPEG)
target_include_directories(yvo_decoder SYSTEM PRIVATE ${YVO_LIBDRM_INCLUDE_DIRS})
yvo_target_defaults(yvo_decoder)   # 2026-09-22: -mcpu applied for the first time (previously warnings only)
