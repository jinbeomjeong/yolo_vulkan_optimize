# Libraries and executables under 01_src. The order follows the dependency direction (CMake resolves target names at generate time, but
# if(TARGET ...) branches are evaluated at **configure time**, so the depended-on side must be added first).
add_subdirectory(core)
add_subdirectory(ref)
add_subdirectory(toolkit)
add_subdirectory(decode)
add_subdirectory(hailo)
add_subdirectory(vulkan)
add_subdirectory(display)
add_subdirectory(isp)
add_subdirectory(testkit)
add_subdirectory(pipeline)
add_subdirectory(tools)
