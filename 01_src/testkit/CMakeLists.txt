# Verification harness shared by tests and probes (header-only). Libraries do not include it.
#   yvo_testkit_postproc : postproc_fixture.hpp — 04_data/raw_tensors fixture parser + GPU/CPU comparison,
#                          nms_harness.hpp — synthetic DetBuf → EfficientNms harness with a timestamp pair
#                          (both shared by test_postproc and yvo_postproc_probe)
#   yvo_testkit_display  : offscreen_target.hpp, display_convert_check.hpp — offscreen render target and readback,
#                          display-conversion bit-exact judgment (shared by test_overlay, test_display_convert and yvo_display_probe)
add_library(yvo_testkit_postproc INTERFACE)
target_link_libraries(yvo_testkit_postproc INTERFACE yvo_vulkan yvo_ref)

if(TARGET yvo_display)
  add_library(yvo_testkit_display INTERFACE)
  target_link_libraries(yvo_testkit_display INTERFACE yvo_display yvo_ref)
endif()
