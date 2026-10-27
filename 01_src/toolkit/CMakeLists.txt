# Helpers for tools/apps only (header-only): tool_util (CLI, timer), tool_image (PNG/PPM save), stats (median, percentiles),
# cli_parse (strict numeric option values), result_csv (the shared RESULT_CSV line), frame_log (--frame-log CSV rows),
# cpu_preproc / sws_preproc (CPU preprocessing comparison; sws_preproc needs the optional yvo_swscale target).
# **Libraries (yvo_hailo, yvo_vulkan, yvo_display, yvo_isp, yvo_decoder, yvo_stages) do not include it.**
# Only tools (01_src/tools/), the app (pipeline/main.cpp) and tests link it.
add_library(yvo_toolkit INTERFACE)
target_link_libraries(yvo_toolkit INTERFACE yvo_core yvo_ref)
