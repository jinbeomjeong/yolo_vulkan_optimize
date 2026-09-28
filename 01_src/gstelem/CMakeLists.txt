# GStreamer research path (optional): v4l2 stateless decode DMABuf (NV12 + SAND128) → DrmFrame adapter.
# Without GStreamer >= 1.24 dev packages, the whole folder is skipped (the existing build must not break),
# and tools that REQUIRE yvo_gstelem are skipped too.
find_package(PkgConfig QUIET)
if(NOT PkgConfig_FOUND)
  message(STATUS "pkg-config not found. Skipping yvo_gstelem.")
  return()
endif()

# GLOBAL: the tools directory (01_src/tools) links these imported targets directly (gst_parse_launch, appsink).
pkg_check_modules(YVO_GST QUIET IMPORTED_TARGET GLOBAL
  gstreamer-1.0>=1.24 gstreamer-base-1.0>=1.24 gstreamer-video-1.0>=1.24 gstreamer-allocators-1.0>=1.24)
if(NOT YVO_GST_FOUND)
  message(STATUS "GStreamer >= 1.24 (core/base/video/allocators dev) not found. Skipping yvo_gstelem.")
  return()
endif()
pkg_check_modules(YVO_GST_APP QUIET IMPORTED_TARGET GLOBAL gstreamer-app-1.0>=1.24)
pkg_check_modules(YVO_GSTELEM_DRM REQUIRED IMPORTED_TARGET libdrm)   # drm_fourcc.h (required by yvo_decoder anyway)
message(STATUS "GStreamer ${YVO_GST_gstreamer-1.0_VERSION} - building yvo_gstelem.")

add_library(yvo_gstelem STATIC
  gst_drm_frame.cpp)
# GStreamer stays PRIVATE: the public header only forward-declares GstBuffer/GstCaps.
target_link_libraries(yvo_gstelem
  PUBLIC yvo_core
  PRIVATE PkgConfig::YVO_GST PkgConfig::YVO_GSTELEM_DRM)
yvo_target_defaults(yvo_gstelem)
