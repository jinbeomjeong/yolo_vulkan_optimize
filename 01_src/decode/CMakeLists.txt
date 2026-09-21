# HW HEVC 디코더(FFmpeg DRM_PRIME → DrmFrame).
# FFmpeg·libdrm 은 필수 의존이라 yvo_decoder 는 **항상 존재한다** — 소비자가 if(EXISTS)/if(TARGET) 로 감쌀 필요가 없다.
# (2026-09-22 전에는 01_src/pipeline/ 에 있어서 다른 폴더가 `if(EXISTS .../pipeline/CMakeLists.txt)` 우회를 5곳에 썼다.)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YVO_FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
pkg_check_modules(YVO_LIBDRM REQUIRED IMPORTED_TARGET libdrm)

add_library(yvo_decoder STATIC ffmpeg_drm_decoder.cpp)
target_include_directories(yvo_decoder PUBLIC ${PROJECT_SOURCE_DIR}/01_src)
target_link_libraries(yvo_decoder
  PUBLIC yvo_common PkgConfig::YVO_FFMPEG PkgConfig::YVO_LIBDRM)
yvo_target_defaults(yvo_decoder)   # 2026-09-22: -mcpu 가 처음 붙었다(전에는 경고만)
