# GoogleTest: GPU/NPU 경로를 CPU 기준 구현과 비교
include(GoogleTest)

if(TARGET yvo_vulkan AND EXISTS ${PROJECT_SOURCE_DIR}/01_src/pipeline/CMakeLists.txt)
  add_executable(test_preproc test_preproc.cpp)
  target_link_libraries(test_preproc PRIVATE yvo_vulkan yvo_decoder GTest::gtest_main)
  yvo_target_defaults(test_preproc)
  target_compile_definitions(test_preproc PRIVATE
    YVO_TEST_VIDEO="${PROJECT_SOURCE_DIR}/04_data/videos/video_h265.mp4")
  if(TARGET yvo_opencv)
    target_link_libraries(test_preproc PRIVATE yvo_opencv)
    target_compile_definitions(test_preproc PRIVATE YVO_HAVE_OPENCV=1)
  endif()
  # 실패 시 차이 이미지는 build/05_tests/preproc_diff/ 에 저장된다
  gtest_discover_tests(test_preproc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
endif()

# ISP 전처리 화질: CPU 기준·GPU 셰이더와 비교 (PiSP BE 필요, NPU 불필요)
# 비트 일치가 아니라 통계 지표로 판정한다(리샘플 커널이 다르다). 패딩만 하드 게이트(오차 0)다.
# /dev/media0을 독점하므로 RUN_SERIAL로 돌린다.
if(TARGET yvo_isp AND TARGET yvo_vulkan AND TARGET yvo_decoder)
  add_executable(test_isp_preproc test_isp_preproc.cpp)
  target_link_libraries(test_isp_preproc PRIVATE yvo_isp yvo_vulkan yvo_decoder GTest::gtest_main)
  yvo_target_defaults(test_isp_preproc)
  target_compile_definitions(test_isp_preproc PRIVATE
    YVO_TEST_VIDEO="${PROJECT_SOURCE_DIR}/04_data/videos/video_h265.mp4")
  if(TARGET yvo_opencv)
    target_link_libraries(test_isp_preproc PRIVATE yvo_opencv)
    target_compile_definitions(test_isp_preproc PRIVATE YVO_HAVE_OPENCV=1)
  endif()
  # 실패 시 차이 이미지는 build/05_tests/isp_preproc_diff/ 에 저장된다
  gtest_discover_tests(test_isp_preproc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE
    DISCOVERY_TIMEOUT 60)
endif()

# 추론 결과 일치: CPU 전처리 경로 vs GPU 전처리 경로(ptr/dmabuf) 검출 비교 (Hailo-8 필요)
# 프레임 수는 환경변수 YVO_DET_FRAMES(기본 100). 전처리·추론을 한 번만 하도록 바이너리 전체를 한 테스트로 등록하고,
# NPU를 독점하므로 RUN_SERIAL로 돌린다.
if(TARGET yvo_vulkan AND TARGET yvo_hailo AND TARGET yvo_decoder)
  add_executable(test_detections test_detections.cpp)
  target_link_libraries(test_detections PRIVATE yvo_vulkan yvo_hailo yvo_decoder GTest::gtest_main)
  yvo_target_defaults(test_detections)
  target_compile_definitions(test_detections PRIVATE
    YVO_TEST_VIDEO="${PROJECT_SOURCE_DIR}/04_data/videos/video_h265.mp4"
    YVO_TEST_HEF="${PROJECT_SOURCE_DIR}/03_models/yolov11m.hef")
  add_test(NAME test_detections COMMAND test_detections)
  set_tests_properties(test_detections PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 600
    RUN_SERIAL TRUE)
endif()

# GPU 후처리(--postproc gpu) 회귀 고정 — 계획서 [G]
#   PostprocFixture.* : 04_data/raw_tensors 픽스처 vs cpu_yolov8_candidates. **NPU 불필요**(수 초)
#   PostprocLive.*    : host 경로 vs raw+GPU 경로 검출 매칭. **Hailo-8 독점** → RUN_SERIAL
# 두 묶음을 **별도 ctest 항목**으로 등록한다. 한 항목으로 묶으면 fixture 판정까지 NPU 를 기다리게 되고,
# NPU 가 다른 작업에 잡혀 있을 때 셰이더 회귀를 못 잡는다.
if(TARGET yvo_stages AND TARGET yvo_decoder)
  add_executable(test_postproc test_postproc.cpp)
  target_link_libraries(test_postproc PRIVATE yvo_stages yvo_decoder GTest::gtest_main)
  yvo_target_defaults(test_postproc)
  target_compile_definitions(test_postproc PRIVATE
    YVO_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/04_data/raw_tensors"
    YVO_TEST_VIDEO="${PROJECT_SOURCE_DIR}/04_data/videos/video_h265.mp4"
    YVO_TEST_HEF="${PROJECT_SOURCE_DIR}/03_models/yolov11m.hef")
  # fixture 쪽은 게이트마다 따로 등록한다 — 어느 게이트가 깨졌는지 ctest 줄만 보고 알 수 있게.
  # TEST_FILTER 로 PostprocLive 는 제외한다(아래에서 RUN_SERIAL 로 따로 등록한다).
  gtest_discover_tests(test_postproc
    TEST_FILTER PostprocFixture.*
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
  add_test(NAME test_postproc_live COMMAND test_postproc --gtest_filter=PostprocLive.*)
  set_tests_properties(test_postproc_live PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 600
    RUN_SERIAL TRUE)
endif()

# 오버레이(박스·라벨·HUD) GPU 셰이더 vs CPU 기준 구현 — 비트 일치 (NPU·디스플레이 불필요, 오프스크린)
# 계획서 §7: OverlayGeometry / OverlayText / OverlayDeterminism 은 max 오차 0 이 하드 게이트다.
# OverlayVsOpenCV 는 참고용(느슨한 기하 기준)이고 OpenCV 가 있을 때만 빌드된다.
if(TARGET yvo_display)
  add_executable(test_overlay test_overlay.cpp)
  target_link_libraries(test_overlay PRIVATE yvo_display GTest::gtest_main)
  yvo_target_defaults(test_overlay)
  if(TARGET yvo_opencv)
    target_link_libraries(test_overlay PRIVATE yvo_opencv)
    target_compile_definitions(test_overlay PRIVATE YVO_HAVE_OPENCV=1)
  endif()
  # 실패 시 차이 이미지는 build/05_tests/overlay_diff/ 에 저장된다
  gtest_discover_tests(test_overlay
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
endif()

# 표시 변환(display_convert.comp, 1:1) vs CPU 기준 구현 — 비트 일치 (GPU + HEVC HW 디코더, NPU 불필요)
# BT.709 계수의 세 번째 사본에 게이트가 없던 구멍을 막는다(2026-09-21). 판정은 display/display_convert_check.hpp.
if(TARGET yvo_display AND TARGET yvo_decoder)
  add_executable(test_display_convert test_display_convert.cpp)
  target_link_libraries(test_display_convert PRIVATE yvo_display yvo_decoder GTest::gtest_main)
  yvo_target_defaults(test_display_convert)
  target_compile_definitions(test_display_convert PRIVATE
    YVO_TEST_VIDEO="${PROJECT_SOURCE_DIR}/04_data/videos/video_h265.mp4")
  gtest_discover_tests(test_display_convert
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
endif()
