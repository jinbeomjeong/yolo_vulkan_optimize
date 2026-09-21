# Registers the 10 built diagnostic/measurement tools in one place (2026-09-22; previously tools/ was scattered per domain folder).
# Folders are by **measured domain** (a tool spanning several domains goes by what it measures). File names keep their old names.
# All executables go to build/bin/. If a library target is missing (e.g. no libpisp or wayland), REQUIRES silently skips the tool.
#
# Link-level isolation (plan §5.2): **the yvo_hailo and yvo_vulkan libraries never link each other.** Only the tools (here)
# and yvo_stages use both.

# ── hailo ──────────────────────────────────────────────────────────────────────
# Self-test tool: CPU letterbox (NV12 BT.709) → HailoRunner → detection output, PNG, throughput
yvo_add_tool(yvo_hailo_selftest hailo/hailo_selftest.cpp
  REQUIRES yvo_opencv
  LIBS yvo_hailo yvo_opencv yvo_toolkit)

# Raw-output 6-stream probe (plan §6.2 P1 + P2).
#   --probe p1: uses host malloc buffers only (no GPU)
#   --probe p2: receives outputs as dma-buf → --alloc vulkan uses yvo_vulkan (VulkanContext, DmaBufExporter)
# **The yvo_hailo library itself does not depend on yvo_vulkan** — only the tool links both.
yvo_add_tool(yvo_hailo_raw_probe hailo/hailo_raw_probe.cpp
  LIBS yvo_hailo yvo_vulkan yvo_toolkit)

# ── vulkan ─────────────────────────────────────────────────────────────────────
# Performance/verification tool: full decode + GPU preprocessing
yvo_add_tool(yvo_preproc_bench vulkan/preproc_bench.cpp
  LIBS yvo_vulkan yvo_decoder yvo_toolkit
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# Postprocessing probe (plan §6.2 P3).
#   --probe fixture|sweep : no NPU needed (04_data/raw_tensors fixtures)
#   --probe live|stale    : exclusive NPU use. Enabled only when HailoRT is present.
# Linking the yvo_hailo library does **not add dependencies to yvo_vulkan itself**
# (plan §5.2 link-level isolation — the link graphs of test_preproc / yvo_preproc_bench do not change).
# yvo_shaders_diag: embedded header of heap_sum.comp used by --probe heapimport (Appendix B step 1)
yvo_add_tool(yvo_postproc_probe vulkan/postproc_probe.cpp
  LIBS yvo_vulkan yvo_shaders_diag yvo_toolkit yvo_testkit_postproc
  OPTIONAL yvo_hailo=YVO_HAVE_HAILO)

# ── display ────────────────────────────────────────────────────────────────────
# Probes: plan §6.3 (P4 surface / P5 present / P6 context-regression / P8 convert).
# P8 uses the decoder's SAND dma-buf as input. yvo_decoder always exists, so YVO_HAVE_DECODER is always 1 too
# (the #ifdef in the source was kept to preserve file:line references in the docs).
yvo_add_tool(yvo_display_probe display/display_probe.cpp
  REQUIRES yvo_display
  LIBS yvo_display yvo_shaders_diag yvo_decoder yvo_toolkit yvo_testkit_display
  DEFINES YVO_HAVE_DECODER=1)

# Standalone DisplayPipeline probe (milestone 7). Reproduces only the display path without touching yvo_pipeline:
# the goal is to reproduce display_probe's run_convert_display() figures via DisplayPipeline.
yvo_add_tool(yvo_display_pipeline_probe display/display_pipeline_probe.cpp
  REQUIRES yvo_display
  LIBS yvo_display yvo_decoder yvo_toolkit)

# ── isp ────────────────────────────────────────────────────────────────────────
# Spike tool: config dump / single-frame conversion / padding check
yvo_add_tool(yvo_isp_probe isp/isp_probe.cpp
  REQUIRES yvo_isp
  LIBS yvo_isp yvo_decoder
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# Performance measurement: full decode + ISP preprocessing (+ NHCW transpose). Flag names match yvo_preproc_bench.
yvo_add_tool(yvo_isp_bench isp/isp_bench.cpp
  REQUIRES yvo_isp
  LIBS yvo_isp yvo_decoder yvo_toolkit
  OPTIONAL yvo_opencv=YVO_HAVE_OPENCV)

# Full pipeline (ISP preprocessing + NPU inference). Uses the same flags and RESULT_CSV format as yvo_pipeline.
# yvo_pipeline (01_src/pipeline/main.cpp) is not touched — this is a separate tool.
yvo_add_tool(yvo_isp_pipeline isp/isp_pipeline.cpp
  REQUIRES yvo_isp yvo_hailo yvo_opencv
  LIBS yvo_isp yvo_hailo yvo_decoder yvo_opencv yvo_toolkit)

# ── decode ─────────────────────────────────────────────────────────────────────
# Inspection tool: descriptor dump, de-SAND comparison, decode performance
yvo_add_tool(yvo_drm_dump decode/drm_dump.cpp
  LIBS yvo_decoder yvo_toolkit)
