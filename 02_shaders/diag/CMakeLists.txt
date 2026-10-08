# Diagnostic-only shaders → build/02_shaders/diag/generated/*_spv.h (generated header names follow the same rule as the main shaders)
#   display_probe.comp : fixed test pattern for the P5 present gate (yvo_display_probe)
#   heap_sum.comp      : byte-sum reduction for the dma-heap round trip (yvo_postproc_probe --probe heapimport)
# **Libraries do not link this** — only tools (01_src/tools/) link it. It is not used in the pipeline.
yvo_add_shaders(yvo_shaders_diag
  display_probe.comp
  heap_sum.comp
)

# Stage-cut variants of the production yolo_decode.comp (same source, -DYVO_DECODE_STOP=N) for
# yvo_postproc_probe --probe stages: 1 = stop after ② group rejection, 2 = after ③ DFL, 3 = after ④ letterbox inverse.
# The probe injects build/02_shaders/diag/yolo_decode_stop<N>.spv through YVO_YOLO_DECODE_SPV, so the directory
# is handed over as an absolute-path compile definition (same convention as yvo_assets).
yvo_add_shaders(yvo_shaders_decode_stages
  ../yolo_decode.comp@yolo_decode_stop1@YVO_DECODE_STOP=1
  ../yolo_decode.comp@yolo_decode_stop2@YVO_DECODE_STOP=2
  ../yolo_decode.comp@yolo_decode_stop3@YVO_DECODE_STOP=3
)
target_compile_definitions(yvo_shaders_decode_stages INTERFACE
  YVO_DECODE_STAGE_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}")
