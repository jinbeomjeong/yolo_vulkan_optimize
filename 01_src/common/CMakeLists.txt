# 공용 헤더(인터페이스 타입, CPU 기준 구현) - 헤더 전용
add_library(yvo_common INTERFACE)
target_include_directories(yvo_common INTERFACE ${PROJECT_SOURCE_DIR}/01_src)
target_compile_features(yvo_common INTERFACE cxx_std_20)
# 공통 경고 옵션(yvo_warnings)은 cmake/YvoTargets.cmake 로 옮겼다.
