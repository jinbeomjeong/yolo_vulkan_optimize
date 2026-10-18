# HDMI display path: HVS plane scanout through labwc (Wayland xdg-shell fullscreen window + linux-dmabuf; boxes drawn
# with Vulkan into an AB24 dma-buf). It is a **separate library** from yvo_vulkan to isolate regressions
# (plan §5.2 safeguard 4): the link graph of test_preproc / yvo_preproc_bench does not change.
option(YVO_BUILD_DISPLAY "Build the HDMI display path (HVS planes via Wayland)" ON)
if(NOT YVO_BUILD_DISPLAY)
  message(STATUS "yvo_display: YVO_BUILD_DISPLAY=OFF — skipping")
  return()
endif()

find_package(PkgConfig QUIET)
if(NOT PkgConfig_FOUND)
  message(STATUS "yvo_display: no pkg-config — skipping")
  return()
endif()

pkg_check_modules(YVO_WAYLAND IMPORTED_TARGET wayland-client)     # not REQUIRED: silently skip if missing
if(NOT YVO_WAYLAND_FOUND)
  message(STATUS "yvo_display: no wayland-client — skipping")
  return()
endif()

find_program(YVO_WAYLAND_SCANNER wayland-scanner)
pkg_get_variable(YVO_WL_PROTO_DIR wayland-protocols pkgdatadir)
if(NOT YVO_WAYLAND_SCANNER OR NOT YVO_WL_PROTO_DIR)
  message(STATUS "yvo_display: no wayland-scanner or wayland-protocols — skipping")
  return()
endif()
# pkg-config returns pkgdatadir as "//usr/share/wayland-protocols" (two leading slashes) → normalize it.
string(REGEX REPLACE "^//+" "/" YVO_WL_PROTO_DIR "${YVO_WL_PROTO_DIR}")
file(TO_CMAKE_PATH "${YVO_WL_PROTO_DIR}" YVO_WL_PROTO_DIR)

# Generate client stubs with wayland-scanner (same pattern as yvo_add_shaders() running glslc via add_custom_command).
# To add a protocol, add just one line "<name>=<xml relative path>" to the list below.
set(yvo_wl_protocols
  "xdg-shell=stable/xdg-shell/xdg-shell.xml"                   # window (required)
  "linux-dmabuf=stable/linux-dmabuf/linux-dmabuf-v1.xml"       # wayland-planes: decoder SAND dma-buf → wl_buffer
  "presentation-time=stable/presentation-time/presentation-time.xml"  # wayland-planes: presentation time (glass-to-glass)
  "viewporter=stable/viewporter/viewporter.xml")               # wayland-planes: HVS scaling when output ≠ source size
# Optional protocols: yvo_display builds even without the XML. If present, define YVO_HAVE_<macro>=1.
#   color-representation-v1 is in staging, so it may be missing depending on the wayland-protocols version (1.44+).
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
    message(STATUS "yvo_display: optional protocol ${rel} not found — building without ${macro}")
  endif()
endforeach()

# Headers go in generated/display/: new sources include them as "display/<name>-client-protocol.h" to follow the include hierarchy rule
# (STRUCTURE.md §2.2, check_structure C4 "<layer>/<file>"). The existing xdg-shell is used without a prefix as before
# (structure_rules.toml include_unprefixed exception), so generated/display is also added to the include path.
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
    message(STATUS "yvo_display: ${xml} not found — skipping")
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

# The generated C must be compiled as C: private-code emits file-scope declarations such as
# `extern const struct wl_interface wl_surface_interface;` without extern "C", so building it as C++
# fails to link against the C symbols of libwayland-client.
enable_language(C)

# ⚠ The generated C does not pass -Wall -Wextra -Wpedantic (empty initializer entries, etc.).
#   Therefore **yvo_warnings is not attached** and the include directory is given as SYSTEM.
#   This is the only way to keep the "zero build warnings" standard.
add_library(yvo_wayland_protocols STATIC ${yvo_wl_sources})
target_include_directories(yvo_wayland_protocols SYSTEM PUBLIC ${yvo_wl_gen_dir} ${yvo_wl_hdr_dir})
target_link_libraries(yvo_wayland_protocols PUBLIC PkgConfig::YVO_WAYLAND)
set_target_properties(yvo_wayland_protocols PROPERTIES POSITION_INDEPENDENT_CODE ON)
# The headers are a custom command separate from the sources, so the build order is set explicitly.
add_custom_target(yvo_wayland_protocols_headers DEPENDS ${yvo_wl_headers})
add_dependencies(yvo_wayland_protocols yvo_wayland_protocols_headers)

add_library(yvo_display STATIC
  display_config.cpp
  wayland_window.cpp
  drm_state.cpp
  plane_display.cpp
  wayland_plane_scanout.cpp
  overlay_renderer.cpp
  overlay_plane.cpp)
target_link_libraries(yvo_display
  PUBLIC yvo_vulkan yvo_ref
  PRIVATE PkgConfig::YVO_WAYLAND yvo_wayland_protocols yvo_shaders)
yvo_target_defaults(yvo_display)
# Optional protocol macros are used only inside this library's sources (public headers do not depend on the macros).
target_compile_definitions(yvo_display PRIVATE ${yvo_wl_defines})
