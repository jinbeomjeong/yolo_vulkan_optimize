# 01_src 하위 라이브러리·실행파일. 순서는 의존 방향을 따른다(CMake 는 타깃 이름을 generate 시점에 해석하지만,
# if(TARGET ...) 분기는 **configure 시점**에 평가되므로 의존되는 쪽을 먼저 추가해야 한다).
add_subdirectory(common)
add_subdirectory(decode)
add_subdirectory(hailo)
add_subdirectory(vulkan)
add_subdirectory(display)
add_subdirectory(isp)
add_subdirectory(pipeline)
