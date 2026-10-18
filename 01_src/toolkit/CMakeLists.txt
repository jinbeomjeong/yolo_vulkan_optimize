# Helpers for the app only (header-only): tool_util (CLI, timer), stats (median, percentiles),
# cli_parse (strict numeric option values), result_csv (the shared RESULT_CSV line), frame_log (--frame-log CSV rows).
# **Libraries (yvo_hailo, yvo_vulkan, yvo_display, yvo_decoder, yvo_stages) do not include it.**
# Only the app (pipeline/main.cpp) links it.
add_library(yvo_toolkit INTERFACE)
target_link_libraries(yvo_toolkit INTERFACE yvo_core yvo_ref)
