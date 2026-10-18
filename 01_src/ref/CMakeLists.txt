# CPU reference implementations (header-only): cpu_letterbox, cpu_yolov8_decode, overlay_reference, glyph_font_8x16.
# **The single source of truth for the formulas.** GPU shaders (02_shaders/) are translations of these headers.
add_library(yvo_ref INTERFACE)
target_link_libraries(yvo_ref INTERFACE yvo_core)
