# yvo_decoder moved to 01_src/decode/ (2026-09-22).

# Inference stage (InferStage implementation). **Does not depend on OpenCV** — so that 05_tests can link it.
#   infer_stage.hpp   : header-only (includes HostNmsStage)
#   gpu_postproc_stage: HailoRawRunner + YoloPostproc + cpu_nms glue
# It used to be compiled only into the yvo_pipeline executable, so tests could not link this glue.
# **Logic left as is; only the wiring was pulled out into a library** (used by PostprocLive in 05_tests/test_postproc.cpp).
#   slot_exchange     : SlotExchange (slot pool + completion queue) + validate_slot_budget, moved out of main.cpp
#                       unchanged (2026-10-11) so 05_tests/test_pipeline_logic.cpp can test them without NPU/GPU.
if(TARGET yvo_vulkan AND TARGET yvo_hailo)
  add_library(yvo_stages STATIC gpu_postproc_stage.cpp slot_exchange.cpp)
  target_link_libraries(yvo_stages PUBLIC yvo_vulkan yvo_hailo PRIVATE yvo_ref)
  yvo_target_defaults(yvo_stages)
endif()

# Integrated app: HW decode → Vulkan letterbox → HailoRT inference (ptr/dmabuf mode)
if(TARGET yvo_stages AND TARGET yvo_opencv)
  add_executable(yvo_pipeline main.cpp)
  target_link_libraries(yvo_pipeline PRIVATE yvo_decoder yvo_stages yvo_opencv yvo_toolkit yvo_assets)
  yvo_target_defaults(yvo_pipeline)
  # HDMI display is a **conditional link**. yvo_display stays a separate library so that
  # wayland is not dragged into the link graph of test_preproc / yvo_preproc_bench.
  # Without yvo_display (wayland-client/protocols missing) the --display option itself is not compiled.
  if(TARGET yvo_display)
    target_link_libraries(yvo_pipeline PRIVATE yvo_display)
    target_compile_definitions(yvo_pipeline PRIVATE YVO_HAVE_DISPLAY=1)
  endif()
endif()
