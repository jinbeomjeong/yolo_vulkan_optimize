# BCM2712 PiSP Back End(ISP) 하드웨어 리사이즈: SAND128 NV12 → packed RGB888 letterbox
# libpisp이 없으면 이 폴더의 타깃을 통째로 건너뛴다(기존 빌드가 깨지면 안 된다).
find_package(PkgConfig QUIET)
if(NOT PkgConfig_FOUND)
  message(STATUS "pkg-config가 없다. yvo_isp를 건너뛴다.")
  return()
endif()

pkg_check_modules(YVO_PISP QUIET IMPORTED_TARGET libpisp)
if(NOT YVO_PISP_FOUND)
  message(STATUS "libpisp을 찾지 못했다(libpisp-dev). yvo_isp를 건너뛴다.")
  return()
endif()
message(STATUS "libpisp ${YVO_PISP_VERSION} - yvo_isp를 빌드한다.")

add_library(yvo_isp STATIC
  isp_resizer.cpp
  nhcw_repack.cpp
  dma_heap_buffer.cpp)
target_include_directories(yvo_isp PUBLIC ${PROJECT_SOURCE_DIR}/01_src)
# PkgConfig::YVO_PISP는 반드시 PRIVATE다. Cflags의 -I/usr/include/libpisp가 PUBLIC으로 새면
# /usr/include/libpisp/backend/pisp_be_config.h와 /usr/include/linux/media/raspberrypi/pisp_be_config.h가
# 이름 충돌한다. 공개 헤더(isp_resizer.hpp)는 libpisp 타입을 전혀 노출하지 않는다(PIMPL).
target_link_libraries(yvo_isp
  PUBLIC yvo_common
  PRIVATE PkgConfig::YVO_PISP yvo_warnings)
# 주의: libpisp 내부 헤더가 "common/..."(pisp_common.h, logging.hpp, utils.hpp)를 상대 경로 없이 include한다.
# 01_src/common/에 같은 이름의 헤더를 만들면 -I 순서 때문에 그쪽이 먼저 잡힌다.
target_compile_options(yvo_isp PRIVATE -mcpu=cortex-a76)

# 스파이크 도구: 설정 덤프 / 1프레임 변환 / 패딩 검사 (디코더가 있을 때만)
if(EXISTS ${PROJECT_SOURCE_DIR}/01_src/pipeline/CMakeLists.txt)  # yvo_decoder는 01_src/pipeline에서 정의
  add_executable(yvo_isp_probe tools/isp_probe.cpp)
  target_link_libraries(yvo_isp_probe PRIVATE yvo_isp yvo_decoder yvo_warnings)
  target_compile_options(yvo_isp_probe PRIVATE -mcpu=cortex-a76)
  if(OpenCV_FOUND)
    target_link_libraries(yvo_isp_probe PRIVATE ${OpenCV_LIBS})
    target_include_directories(yvo_isp_probe PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(yvo_isp_probe PRIVATE YVO_HAVE_OPENCV=1)
  endif()

  # 성능 측정: 전체 디코드 + ISP 전처리 (+ NHCW 전치). yvo_preproc_bench와 플래그 이름을 맞췄다.
  add_executable(yvo_isp_bench tools/isp_bench.cpp)
  target_link_libraries(yvo_isp_bench PRIVATE yvo_isp yvo_decoder yvo_warnings)
  target_compile_options(yvo_isp_bench PRIVATE -mcpu=cortex-a76)
  if(OpenCV_FOUND)
    target_link_libraries(yvo_isp_bench PRIVATE ${OpenCV_LIBS})
    target_include_directories(yvo_isp_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(yvo_isp_bench PRIVATE YVO_HAVE_OPENCV=1)
  endif()

  # 전체 파이프라인(ISP 전처리 + NPU 추론). yvo_pipeline과 같은 플래그·RESULT_CSV 형식을 쓴다.
  # yvo_pipeline(01_src/pipeline/main.cpp)은 건드리지 않는다 — 이건 별도 도구다.
  if(TARGET yvo_hailo AND OpenCV_FOUND)
    add_executable(yvo_isp_pipeline tools/isp_pipeline.cpp)
    target_link_libraries(yvo_isp_pipeline PRIVATE yvo_isp yvo_hailo yvo_decoder ${OpenCV_LIBS} yvo_warnings)
    target_include_directories(yvo_isp_pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_options(yvo_isp_pipeline PRIVATE -mcpu=cortex-a76)
  endif()
endif()
