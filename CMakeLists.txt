cmake_minimum_required(VERSION 3.25)
project(yolo_vulkan_optimize LANGUAGES CXX)

# 컴파일러 캐시: 있으면 자동으로 쓴다(없으면 조용히 건너뛴다).
#   HailoRT/Vulkan/OpenCV 헤더 때문에 TU 하나가 전처리 후 7만~13만 줄로 팽창해 컴파일이 2~5초다.
#   verify 스킬은 단계마다 전체 빌드를 2번(--clean-first + 별도 build-asan/) 요구하는데, 그때
#   실제로 바뀐 파일은 1~3개뿐이다. ccache 가 나머지를 0초로 만든다(실측 4.1초 -> 0.0초).
#   빌드 디렉터리를 분리해도(build-display/ 등) 캐시를 공유하므로 병렬 작업 비용이 2배가 되지 않는다.
option(YVO_USE_CCACHE "ccache 가 있으면 컴파일러 런처로 쓴다" ON)
if(YVO_USE_CCACHE)
  find_program(YVO_CCACHE ccache)
  if(YVO_CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${YVO_CCACHE})
    set(CMAKE_C_COMPILER_LAUNCHER ${YVO_CCACHE})
    message(STATUS "ccache 사용: ${YVO_CCACHE}")
  endif()
endif()

# 언어 표준·도구 설정
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 빌드 타입 미지정 시 Release (단일 구성 생성기에서만)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "빌드 타입" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# 컴파일러 캐시 (ccache): 클린 빌드와 build-asan 별도 트리의 재컴파일 비용을 줄인다.
#   이 저장소는 클린 빌드가 단일코어 환산 약 385초다(파일 하나가 16~43초).
#   -DYVO_USE_CCACHE=OFF 로 끌 수 있다.
option(YVO_USE_CCACHE "ccache가 있으면 컴파일러 런처로 쓴다" ON)
if(YVO_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(YVO_CCACHE_PROGRAM ccache)
  if(YVO_CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${YVO_CCACHE_PROGRAM}" CACHE STRING "컴파일러 런처")
    message(STATUS "ccache 사용: ${YVO_CCACHE_PROGRAM}")
  else()
    message(STATUS "ccache 없음 — 캐시 없이 빌드한다")
  endif()
endif()

option(YVO_BUILD_TESTS "GoogleTest 기반 테스트 빌드" ON)
# Google Benchmark 옵션은 제거했다(2026-09-21). 06_bench/ 가 .gitkeep 하나뿐인 빈 폴더인데
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

# 셰이더 컴파일 헬퍼 (glslc: *.comp -> *.spv -> 임베드 헤더)
#   yvo_add_shaders(<target> <shader>...)
#   - glslc --target-env=vulkan1.3 -O -Werror 로 컴파일, -MD 의존성 파일로 #include 변경도 추적
#   - spirv-val --target-env vulkan1.3 로 검증 (실패하면 빌드 실패)
#   - 산출물: ${CMAKE_CURRENT_BINARY_DIR}/<name>.spv, generated/<name>_spv.h
#     (배열 이름: k<CamelName>Spv, 예: sand_letterbox.comp -> kSandLetterboxSpv)
#   - <target>은 INTERFACE 라이브러리. 링크하면 generated/가 include 경로에 추가되고 빌드 순서가 보장된다.
find_program(YVO_GLSLC glslc REQUIRED)
find_program(YVO_SPIRV_VAL spirv-val REQUIRED)
function(yvo_add_shaders target)
  set(gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${gen_dir})
  set(headers "")
  foreach(src IN LISTS ARGN)
    get_filename_component(abs ${src} ABSOLUTE)
    get_filename_component(name ${src} NAME_WE)
    set(spv ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv)
    set(hdr ${gen_dir}/${name}_spv.h)
    # snake_case -> CamelCase 배열 이름
    string(REPLACE "_" ";" parts ${name})
    set(sym "k")
    foreach(part IN LISTS parts)
      string(SUBSTRING ${part} 0 1 first)
      string(SUBSTRING ${part} 1 -1 rest)
      string(TOUPPER ${first} first)
      string(APPEND sym "${first}${rest}")
    endforeach()
    string(APPEND sym "Spv")
    add_custom_command(
      OUTPUT ${spv}
      COMMAND ${YVO_GLSLC} --target-env=vulkan1.3 -O -Werror -MD -MF ${spv}.d -MT ${spv} -o ${spv} ${abs}
      COMMAND ${YVO_SPIRV_VAL} --target-env vulkan1.3 ${spv}
      DEPENDS ${abs}
      DEPFILE ${spv}.d
      COMMENT "glslc + spirv-val ${src}"
      VERBATIM)
    add_custom_command(
      OUTPUT ${hdr}
      COMMAND ${CMAKE_COMMAND} -DSPV=${spv} -DOUT=${hdr} -DSYMBOL=${sym}
              -P ${PROJECT_SOURCE_DIR}/cmake/yvo_embed_spirv.cmake
      DEPENDS ${spv} ${PROJECT_SOURCE_DIR}/cmake/yvo_embed_spirv.cmake
      COMMENT "embed ${name}.spv -> ${name}_spv.h"
      VERBATIM)
    list(APPEND headers ${hdr})
  endforeach()
  add_custom_target(${target}_gen DEPENDS ${headers})
  add_library(${target} INTERFACE)
  target_include_directories(${target} INTERFACE ${gen_dir})
  add_dependencies(${target} ${target}_gen)
endfunction()

# 하위 디렉터리: CMakeLists.txt가 있을 때만 추가
function(yvo_add_subdirectory_if_exists dir)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    add_subdirectory(${dir})
  endif()
endfunction()

yvo_add_subdirectory_if_exists(01_src/common)
yvo_add_subdirectory_if_exists(01_src/hailo)
yvo_add_subdirectory_if_exists(01_src/vulkan)
yvo_add_subdirectory_if_exists(01_src/display)
yvo_add_subdirectory_if_exists(01_src/isp)
yvo_add_subdirectory_if_exists(01_src/pipeline)
yvo_add_subdirectory_if_exists(02_shaders)

if(YVO_BUILD_TESTS)
  yvo_add_subdirectory_if_exists(05_tests)
endif()

