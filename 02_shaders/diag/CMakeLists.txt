# Diagnostic-only shaders → build/02_shaders/diag/generated/*_spv.h (generated header names follow the same rule as the main shaders)
#   display_probe.comp : fixed test pattern for the P5 present gate (yvo_display_probe)
#   heap_sum.comp      : byte-sum reduction for the dma-heap round trip (yvo_postproc_probe --probe heapimport)
# **Libraries do not link this** — only tools (01_src/tools/) link it. It is not used in the pipeline.
yvo_add_shaders(yvo_shaders_diag
  display_probe.comp
  heap_sum.comp
)
