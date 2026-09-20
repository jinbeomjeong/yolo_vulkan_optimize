# 공용 헤더(인터페이스 타입, CPU 기준 구현) - 헤더 전용
add_library(yvo_common INTERFACE)
target_include_directories(yvo_common INTERFACE ${PROJECT_SOURCE_DIR}/01_src)
target_compile_features(yvo_common INTERFACE cxx_std_20)

# 공통 경고 옵션. 각 타깃에서 target_link_libraries(... PRIVATE yvo_warnings)로 사용한다.
add_library(yvo_warnings INTERFACE)
target_compile_options(yvo_warnings INTERFACE -Wall -Wextra -Wpedantic)
