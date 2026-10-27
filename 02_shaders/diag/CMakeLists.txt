# Diagnostic-only shaders → build/02_shaders/diag/generated/*_spv.h (generated header names follow the same rule as the main shaders)
#   display_probe.comp : fixed test pattern for the P5 present gate (yvo_display_probe)
#   heap_sum.comp      : byte-sum reduction for the dma-heap round trip (yvo_postproc_probe --probe heapimport)
#   mem_bw.comp        : bandwidth-bound read-sum / write / copy kernels, uint or uvec4 (yvo_postproc_probe --probe memasym)
# **Libraries do not link this** — only tools (01_src/tools/) link it. It is not used in the pipeline.
yvo_add_shaders(yvo_shaders_diag
  display_probe.comp
  heap_sum.comp
  mem_bw.comp@mem_bw_read_u32@YVO_MEMBW_READ
  mem_bw.comp@mem_bw_read_v4@YVO_MEMBW_READ,YVO_MEMBW_VEC4
  mem_bw.comp@mem_bw_write_u32@YVO_MEMBW_WRITE
  mem_bw.comp@mem_bw_write_v4@YVO_MEMBW_WRITE,YVO_MEMBW_VEC4
  mem_bw.comp@mem_bw_copy_u32@YVO_MEMBW_COPY
  mem_bw.comp@mem_bw_copy_v4@YVO_MEMBW_COPY,YVO_MEMBW_VEC4
)

# Diagnostic variants of the production yolo_decode.comp (same source, different -D) for yvo_postproc_probe
# --probe stages. The probe injects build/02_shaders/diag/<name>.spv through YVO_YOLO_DECODE_SPV, so the directory
# is handed over as an absolute-path compile definition (same convention as yvo_assets).
#   stage cuts   : 1 = stop after ② group rejection, 2 = after ③ DFL, 3 = after ④ letterbox inverse (emit 0 candidates)
#   emit split   : noatomic = non-atomic slot reservation, norescan = no per-anchor class rescan. Both still emit,
#                  with **intentionally wrong** values -- Full - noatomic = atomic share, Full - norescan = rescan share.
#   legacy       : YVO_DECODE_LEGACY_EMIT = the pre-argmax-shortcut ① + ⑤ (correct output; Full - legacy = shortcut Δ,
#                  S1 - stop1_legacy = the ① cost of the packed max + SWAR counter)
yvo_add_shaders(yvo_shaders_decode_stages
  ../yolo_decode.comp@yolo_decode_stop1@YVO_DECODE_STOP=1
  ../yolo_decode.comp@yolo_decode_stop2@YVO_DECODE_STOP=2
  ../yolo_decode.comp@yolo_decode_stop3@YVO_DECODE_STOP=3
  ../yolo_decode.comp@yolo_decode_noatomic@YVO_EMIT_NOATOMIC
  ../yolo_decode.comp@yolo_decode_norescan@YVO_EMIT_NORESCAN
  ../yolo_decode.comp@yolo_decode_legacy@YVO_DECODE_LEGACY_EMIT
  ../yolo_decode.comp@yolo_decode_stop1_legacy@YVO_DECODE_STOP=1,YVO_DECODE_LEGACY_EMIT
)
target_compile_definitions(yvo_shaders_decode_stages INTERFACE
  YVO_DECODE_STAGE_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}")
