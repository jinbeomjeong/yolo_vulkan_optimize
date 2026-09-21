# HailoRT inference runners
#   hailo_runner      : InferModel async API + HEF-embedded NMS (host post-processing reference path)
#   hailo_raw_runner  : 6 raw output streams + input/output dma-buf (GPU post-processing path, passed the F0 gate)
# **yvo_hailo does not depend on yvo_vulkan** — link-level isolation per plan §5.2.
# hailo_raw_runner takes only a dma-buf fd and size and knows nothing about Vulkan types.
add_library(yvo_hailo STATIC hailo_runner.cpp hailo_raw_runner.cpp)
target_link_libraries(yvo_hailo PUBLIC HailoRT::libhailort)
target_link_libraries(yvo_hailo PUBLIC yvo_core PRIVATE yvo_ref)
yvo_target_defaults(yvo_hailo)
