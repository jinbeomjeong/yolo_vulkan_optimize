# BCM2712 PiSP Back End (ISP) hardware resize: SAND128 NV12 → packed RGB888 letterbox
# Without libpisp, all targets in this folder are skipped (the existing build must not break).
find_package(PkgConfig QUIET)
if(NOT PkgConfig_FOUND)
  message(STATUS "pkg-config not found. Skipping yvo_isp.")
  return()
endif()

pkg_check_modules(YVO_PISP QUIET IMPORTED_TARGET libpisp)
if(NOT YVO_PISP_FOUND)
  message(STATUS "libpisp not found (libpisp-dev). Skipping yvo_isp.")
  return()
endif()
message(STATUS "libpisp ${YVO_PISP_VERSION} - building yvo_isp.")

add_library(yvo_isp STATIC
  isp_resizer.cpp
  nhcw_repack.cpp)
# PkgConfig::YVO_PISP must be PRIVATE. If the Cflags' -I/usr/include/libpisp leaks as PUBLIC,
# /usr/include/libpisp/backend/pisp_be_config.h and /usr/include/linux/media/raspberrypi/pisp_be_config.h
# collide by name. The public header (isp_resizer.hpp) exposes no libpisp types at all (PIMPL).
target_link_libraries(yvo_isp
  PUBLIC yvo_core
  PRIVATE yvo_ref PkgConfig::YVO_PISP)
# Caution: libpisp internal headers include "common/..." (pisp_common.h, logging.hpp, utils.hpp) without a relative path.
# A common/ folder directly under 01_src/ could be found first because of -I ordering — which is why the old 01_src/common/ was
# split into core/·ref/·toolkit/ (2026-09-22). Do not put a folder named like a system include folder under 01_src/.
yvo_target_defaults(yvo_isp)
