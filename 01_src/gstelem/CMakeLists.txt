# GStreamer research path (optional): v4l2 stateless decode DMABuf (NV12 + SAND128) → DrmFrame adapter,
# yvosandletterbox (stage 2a), yvohailosink (stage 2b, needs yvo_hailo) and the tensor meta; elements are registered statically by the tools (register.hpp).
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
  gst_drm_frame.cpp
  tensor_meta.cpp
  sand_letterbox_element.cpp
  register.cpp)
# GStreamer stays PRIVATE: the public headers only forward-declare GstBuffer/GstCaps/GstMemory/GstPlugin.
# yvo_vulkan is PRIVATE too (the element owns its VulkanContext/SandLetterbox; no Vulkan type in the public headers).
target_link_libraries(yvo_gstelem
  PUBLIC yvo_core
  PRIVATE yvo_vulkan PkgConfig::YVO_GST PkgConfig::YVO_GSTELEM_DRM)
# Stage 2b: yvohailosink (HailoRunner, host NMS). PUBLIC define: register.cpp and the tools/tests check it.
if(TARGET yvo_hailo)
  target_sources(yvo_gstelem PRIVATE hailo_infer_sink.cpp)
  target_link_libraries(yvo_gstelem PRIVATE yvo_hailo)
  target_compile_definitions(yvo_gstelem PUBLIC YVO_GSTELEM_HAVE_HAILO=1)
endif()
yvo_target_defaults(yvo_gstelem)
