cmake_minimum_required(VERSION 3.25)
project(yolo_vulkan_optimize LANGUAGES CXX)

# 컴파일러 캐시: 있으면 자동으로 쓴다(없으면 조용히 건너뛴다).
#   HailoRT/Vulkan/OpenCV 헤더 때문에 TU 하나가 전처리 후 7만~13만 줄로 팽창해 컴파일이 2~5초다.
#   verify 스킬은 단계마다 전체 빌드를 2번(--clean-first + 별도 build-asan/) 요구하는데, 그때
#   실제로 바뀐 파일은 1~3개뿐이다. ccache 가 나머지를 0초로 만든다(실측 4.1초 -> 0.0초).
#   빌드 디렉터리를 분리해도(build-display/ 등) 캐시를 공유하므로 병렬 작업 비용이 2배가 되지 않는다.
#   C 런처도 건다(wayland-scanner 가 만드는 C 스텁). -DYVO_USE_CCACHE=OFF 로 끌 수 있다.
option(YVO_USE_CCACHE "ccache 가 있으면 컴파일러 런처로 쓴다" ON)
if(YVO_USE_CCACHE)
  find_program(YVO_CCACHE ccache)
  if(YVO_CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${YVO_CCACHE})
    set(CMAKE_C_COMPILER_LAUNCHER ${YVO_CCACHE})
    message(STATUS "ccache 사용: ${YVO_CCACHE}")
  else()
    message(STATUS "ccache 없음 — 캐시 없이 빌드한다")
  endif()
endif()

# 언어 표준·도구 설정
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 실행파일은 모두 <build>/bin 에 모은다(2026-09-22). 소스 폴더를 옮겨도 벤치·스킬·문서의
# 바이너리 경로(build/bin/yvo_*)가 깨지지 않는다. 테스트의 WORKING_DIRECTORY 는 build/05_tests 그대로다.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 빌드 타입 미지정 시 Release (단일 구성 생성기에서만)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "빌드 타입" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(YVO_BUILD_TESTS "GoogleTest 기반 테스트 빌드" ON)
# Google Benchmark 옵션은 제거했다(2026-09-21). 06_tools/bench/(당시 06_bench/) 가 .gitkeep 하나뿐인 빈 폴더인데
# find_package(benchmark REQUIRED) 가 configure 하드 의존이었다 — 소비자가 0개인 의존이다.
# 마이크로 벤치가 실제로 필요해지면 05_tests/bench/ 하위로 되살린다.

# 필수 의존성
find_package(HailoRT REQUIRED)   # HailoRT::libhailort
find_package(Vulkan REQUIRED)    # Vulkan::Vulkan
find_package(OpenCV)             # ${OpenCV_LIBS}, 없으면 경고만
if(NOT OpenCV_FOUND)
  message(WARNING "OpenCV를 찾지 못했다. OpenCV를 쓰는 타깃은 빌드되지 않을 수 있다.")
endif()

# 선택 의존성: 옵션이 켜졌을 때만 탐색
if(YVO_BUILD_TESTS)
  find_package(GTest REQUIRED)   # GTest::gtest_main
  enable_testing()
endif()

# 공용 타깃·헬퍼: yvo_warnings, YVO_CPU_FLAGS, yvo_target_defaults(), yvo_opencv, yvo_add_tool()
include(${PROJECT_SOURCE_DIR}/cmake/YvoTargets.cmake)
# 셰이더 컴파일 헬퍼: yvo_add_shaders(<target> <shader>...)
include(${PROJECT_SOURCE_DIR}/cmake/YvoShaders.cmake)

# 하위 디렉터리. 필수 폴더라 조건 없이 추가한다 — 폴더가 없으면 configure 가 멈춘다
# (예전 yvo_add_subdirectory_if_exists 는 없는 폴더를 조용히 건너뛰었다).
add_subdirectory(01_src/common)
add_subdirectory(01_src/hailo)
add_subdirectory(01_src/vulkan)
add_subdirectory(01_src/display)
add_subdirectory(01_src/isp)
add_subdirectory(01_src/pipeline)
add_subdirectory(02_shaders)

if(YVO_BUILD_TESTS)
  add_subdirectory(05_tests)
endif()
