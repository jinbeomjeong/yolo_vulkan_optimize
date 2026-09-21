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
  nhcw_repack.cpp)
# PkgConfig::YVO_PISP는 반드시 PRIVATE다. Cflags의 -I/usr/include/libpisp가 PUBLIC으로 새면
# /usr/include/libpisp/backend/pisp_be_config.h와 /usr/include/linux/media/raspberrypi/pisp_be_config.h가
# 이름 충돌한다. 공개 헤더(isp_resizer.hpp)는 libpisp 타입을 전혀 노출하지 않는다(PIMPL).
target_link_libraries(yvo_isp
  PUBLIC yvo_core
  PRIVATE yvo_ref PkgConfig::YVO_PISP)
# 주의: libpisp 내부 헤더가 "common/..."(pisp_common.h, logging.hpp, utils.hpp)를 상대 경로 없이 include한다.
# 01_src/ 바로 아래에 common/ 폴더를 만들면 -I 순서 때문에 그쪽이 먼저 잡힐 수 있다 — 그래서 옛 01_src/common/ 을
# core/·ref/·toolkit/ 으로 나눴다(2026-09-22). 시스템 include 폴더와 같은 이름의 폴더를 01_src/ 아래에 두지 말 것.
yvo_target_defaults(yvo_isp)

# 스파이크 도구: 설정 덤프 / 1프레임 변환 / 패딩 검사
yvo_add_tool(yvo_isp_probe tools/isp_probe.cpp
  LIBS yvo_isp yvo_decoder
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# 성능 측정: 전체 디코드 + ISP 전처리 (+ NHCW 전치). yvo_preproc_bench와 플래그 이름을 맞췄다.
yvo_add_tool(yvo_isp_bench tools/isp_bench.cpp
  LIBS yvo_isp yvo_decoder yvo_toolkit
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# 전체 파이프라인(ISP 전처리 + NPU 추론). yvo_pipeline과 같은 플래그·RESULT_CSV 형식을 쓴다.
# yvo_pipeline(01_src/pipeline/main.cpp)은 건드리지 않는다 — 이건 별도 도구다.
yvo_add_tool(yvo_isp_pipeline tools/isp_pipeline.cpp
  REQUIRES yvo_hailo yvo_opencv
  LIBS yvo_isp yvo_hailo yvo_decoder yvo_opencv yvo_toolkit)
