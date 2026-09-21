# Shared contract types and OS glue (header-only): frame_types, coco_labels, dma_heap_alloc, dmabuf_map, hailo_align.
# **The include root (${PROJECT_SOURCE_DIR}/01_src) is set here and nowhere else.** Every target goes through yvo_core
# and includes with the "folder/file.hpp" form.
# The old name yvo_common / 01_src/common/ was split into core, ref and toolkit on 2026-09-22
# (libpisp's internal headers include "common/logging.hpp" without a relative path, so there was a name-collision risk).
add_library(yvo_core INTERFACE)
target_include_directories(yvo_core INTERFACE ${PROJECT_SOURCE_DIR}/01_src)
target_compile_features(yvo_core INTERFACE cxx_std_20)
