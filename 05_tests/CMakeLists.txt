# GoogleTest: compare GPU/NPU paths against the CPU reference implementation
include(GoogleTest)
# Asset paths: every test links yvo_assets (cmake/YvoTargets.cmake). YVO_FIXTURE_DIR comes from there, and
# YVO_TEST_VIDEO / YVO_TEST_HEF used by the test sources are aliases of YVO_DEFAULT_VIDEO / YVO_DEFAULT_HEF (names kept).

if(TARGET yvo_vulkan)
  add_executable(test_preproc test_preproc.cpp)
  target_link_libraries(test_preproc PRIVATE yvo_vulkan yvo_decoder yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_preproc)
  target_compile_definitions(test_preproc PRIVATE
    YVO_TEST_VIDEO=YVO_DEFAULT_VIDEO)
  if(TARGET yvo_opencv)
    target_link_libraries(test_preproc PRIVATE yvo_opencv)
    target_compile_definitions(test_preproc PRIVATE YVO_HAVE_OPENCV=1)
  endif()
  # On failure, diff images are saved to build/05_tests/preproc_diff/
  # SandRefVsFfmpeg.* (same executable, no GPU): ref/sand.hpp de-SAND vs rpi-ffmpeg av_hwframe_transfer_data(NV12),
  # byte-identical. It reuses this target's decoder + DmaBufReadMap + ref/sand.hpp; FFmpeg comes PUBLIC via yvo_decoder.
  gtest_discover_tests(test_preproc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
endif()

# ISP preprocessing quality: compare against the CPU reference and the GPU shader (needs PiSP BE, no NPU)
# Judged by statistical metrics, not bit-exactness (the resampling kernels differ). Only the padding is a hard gate (zero error).
# Runs with RUN_SERIAL because it takes /dev/media0 exclusively.
if(TARGET yvo_isp AND TARGET yvo_vulkan AND TARGET yvo_decoder)
  add_executable(test_isp_preproc test_isp_preproc.cpp)
  target_link_libraries(test_isp_preproc PRIVATE yvo_isp yvo_vulkan yvo_decoder yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_isp_preproc)
  target_compile_definitions(test_isp_preproc PRIVATE
    YVO_TEST_VIDEO=YVO_DEFAULT_VIDEO)
  if(TARGET yvo_opencv)
    target_link_libraries(test_isp_preproc PRIVATE yvo_opencv)
    target_compile_definitions(test_isp_preproc PRIVATE YVO_HAVE_OPENCV=1)
  endif()
  # On failure, diff images are saved to build/05_tests/isp_preproc_diff/
  gtest_discover_tests(test_isp_preproc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE
    DISCOVERY_TIMEOUT 60)
endif()

# Inference result agreement: compare detections of the CPU preprocessing path vs the GPU preprocessing path (ptr/dmabuf) (needs Hailo-8)
# Frame count from env var YVO_DET_FRAMES (default 100). The whole binary is registered as one test so preprocessing/inference run only once,
# and it runs with RUN_SERIAL because it takes the NPU exclusively.
if(TARGET yvo_vulkan AND TARGET yvo_hailo AND TARGET yvo_decoder)
  add_executable(test_detections test_detections.cpp)
  target_link_libraries(test_detections PRIVATE yvo_vulkan yvo_hailo yvo_decoder yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_detections)
  target_compile_definitions(test_detections PRIVATE
    YVO_TEST_VIDEO=YVO_DEFAULT_VIDEO
    YVO_TEST_HEF=YVO_DEFAULT_HEF)
  add_test(NAME test_detections COMMAND test_detections)
  set_tests_properties(test_detections PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 600
    RUN_SERIAL TRUE
    LABELS "npu")
endif()

# GPU postprocessing (--postproc gpu) regression lock — plan [G]
#   PostprocFixture.* : 04_data/raw_tensors fixtures vs cpu_yolov8_candidates / cpu_nms_greedy_by_class. **No NPU needed**
#   EfficientNms.*    : synthetic DetBuf → efficient_nms.comp vs cpu_nms_greedy_by_class (no NPU needed)
#   PostprocLive.*    : detection matching of host path vs raw+GPU path. **Takes Hailo-8 exclusively** → RUN_SERIAL
# The two groups are registered as **separate ctest entries**. If bundled into one entry, the fixture verdict would wait for the NPU,
# and shader regressions would go uncaught while the NPU is held by another job.
if(TARGET yvo_stages AND TARGET yvo_decoder)
  add_executable(test_postproc test_postproc.cpp)
  target_link_libraries(test_postproc PRIVATE yvo_stages yvo_decoder yvo_testkit_postproc yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_postproc)
  target_compile_definitions(test_postproc PRIVATE
    YVO_TEST_VIDEO=YVO_DEFAULT_VIDEO
    YVO_TEST_HEF=YVO_DEFAULT_HEF)
  # The fixture side is registered per gate — so the ctest line alone tells which gate broke.
  # TEST_FILTER excludes PostprocLive (registered separately with RUN_SERIAL below).
  gtest_discover_tests(test_postproc
    TEST_FILTER PostprocFixture.*:EfficientNms.*
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
  add_test(NAME test_postproc_live COMMAND test_postproc --gtest_filter=PostprocLive.*)
  set_tests_properties(test_postproc_live PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 600
    RUN_SERIAL TRUE
    LABELS "npu")
endif()

# yvo_pipeline app logic (pipeline/slot_exchange.hpp): SlotExchange slot pool + validate_slot_budget gate,
# and the toolkit pieces the app shares with the tools: RESULT_CSV formatter, strict option-value parsing.
# **No NPU, GPU or display needed** — yvo_stages is linked only for headers/symbols, nothing opens a device.
if(TARGET yvo_stages)
  add_executable(test_pipeline_logic test_pipeline_logic.cpp)
  target_link_libraries(test_pipeline_logic PRIVATE yvo_stages yvo_toolkit yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_pipeline_logic)
  gtest_discover_tests(test_pipeline_logic
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 120
    DISCOVERY_TIMEOUT 60)
endif()

# Display option parsers (--display-backend/-color-range/-sync/-pace, --present-mode): string → enum, no GPU or display.
# Built only with yvo_display (optional: needs wayland-client/protocols), like the other display tests.
if(TARGET yvo_display)
  add_executable(test_display_options test_display_options.cpp)
  target_link_libraries(test_display_options PRIVATE yvo_display yvo_toolkit yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_display_options)
  gtest_discover_tests(test_display_options
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 60
    DISCOVERY_TIMEOUT 60)
endif()

# Overlay (boxes, labels, HUD) GPU shader vs CPU reference implementation — bit-exact (no NPU needed, offscreen)
# Except KmsScanout.* uses /dev/dri/card1 (does not take master) and the HEVC HW decoder — skipped if absent.
# Plan §7: OverlayGeometry / OverlayText / OverlayDeterminism have max error 0 as a hard gate.
# OverlayVsOpenCV is for reference only (loose geometric criteria) and is built only when OpenCV is available.
if(TARGET yvo_display)
  add_executable(test_overlay test_overlay.cpp)
  # yvo_decoder: KmsScanout.* builds a video FB (AddFB2) from the decoder's NV12 SAND128 frames
  target_link_libraries(test_overlay PRIVATE yvo_display yvo_decoder yvo_testkit_display yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_overlay)
  if(TARGET yvo_opencv)
    target_link_libraries(test_overlay PRIVATE yvo_opencv)
    target_compile_definitions(test_overlay PRIVATE YVO_HAVE_OPENCV=1)
  endif()
  # On failure, diff images are saved to build/05_tests/overlay_diff/
  gtest_discover_tests(test_overlay
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
endif()

# Display conversion (display_convert.comp, 1:1) vs CPU reference implementation — bit-exact (GPU + HEVC HW decoder, no NPU needed)
# Closes the hole where the display copy of the BT.709 formula had no gate (2026-09-21; shared 02_shaders/bt709.glsl since 2026-10-11). Verdict: testkit/display_convert_check.hpp.
if(TARGET yvo_display AND TARGET yvo_decoder)
  add_executable(test_display_convert test_display_convert.cpp)
  target_link_libraries(test_display_convert PRIVATE yvo_display yvo_decoder yvo_testkit_display yvo_assets GTest::gtest_main)
  yvo_target_defaults(test_display_convert)
  target_compile_definitions(test_display_convert PRIVATE
    YVO_TEST_VIDEO=YVO_DEFAULT_VIDEO)
  gtest_discover_tests(test_display_convert
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
endif()

# GStreamer research path, stage 2a (GPU + HEVC HW decoder, no NPU). Built only with yvo_gstelem (GStreamer >= 1.24 dev)
# and gstreamer-app. Same registration style as test_preproc: one ctest entry per test, runs skipped without the video or
# v4l2slh265dec. Each entry decodes the file up to frame 1815 twice (FFmpeg reference + GStreamer).
#   GstAdapter.*          adapter DrmFrame vs FFmpeg DrmFrame (addressing) and NHCW bytes through SandLetterbox
#   GstElementPath*.*     yvosandletterbox output slot dma-buf vs the same reference; TensorMeta/slot/fd checks
#   GstElementCaps.*      sink pad accepts only DMA_DRM NV12 SAND128 (no GPU needed)
#   GstElementSlots.*     slot-lifetime misuse: moved slot memory is never overwritten, replaced memory does not leak
if(TARGET yvo_gstelem AND TARGET PkgConfig::YVO_GST_APP AND TARGET yvo_decoder)
  add_executable(test_gst_adapter test_gst_adapter.cpp)
  target_link_libraries(test_gst_adapter PRIVATE yvo_gstelem yvo_vulkan yvo_decoder yvo_toolkit yvo_assets
    PkgConfig::YVO_GST PkgConfig::YVO_GST_APP GTest::gtest_main)
  yvo_target_defaults(test_gst_adapter)
  target_compile_definitions(test_gst_adapter PRIVATE
    YVO_TEST_VIDEO=YVO_DEFAULT_VIDEO)
  gtest_discover_tests(test_gst_adapter
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 300
    DISCOVERY_TIMEOUT 60)
endif()

# GStreamer research path, stage 2b (Hailo-8 + GPU + HEVC HW decoder): yvohailosink detections vs the yvo_pipeline
# inference path (SandLetterbox NHCW dma-buf → HailoRunner, same batch) on the first YVO_GST_DET_FRAMES frames (default
# 120, batch YVO_GST_DET_BATCH default 4) — every frame's detection list must be equal. One ctest entry (the NPU work
# runs once), RUN_SERIAL because it takes the NPU exclusively. Skipped without /dev/hailo0, the video, v4l2slh265dec,
# or when the Hailo device is held by another process.
if(TARGET yvo_gstelem AND TARGET yvo_hailo AND TARGET yvo_vulkan AND TARGET yvo_decoder)
  add_executable(test_gst_detections test_gst_detections.cpp)
  target_link_libraries(test_gst_detections PRIVATE yvo_gstelem yvo_hailo yvo_vulkan yvo_decoder yvo_ref yvo_assets
    PkgConfig::YVO_GST GTest::gtest_main)
  yvo_target_defaults(test_gst_detections)
  target_compile_definitions(test_gst_detections PRIVATE
    YVO_TEST_VIDEO=YVO_DEFAULT_VIDEO
    YVO_TEST_HEF=YVO_DEFAULT_HEF)
  add_test(NAME test_gst_detections COMMAND test_gst_detections)
  set_tests_properties(test_gst_detections PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 600
    RUN_SERIAL TRUE
    LABELS "npu;gst")
endif()
