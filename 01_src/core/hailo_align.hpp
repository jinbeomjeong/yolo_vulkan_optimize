#pragma once
// Alignment requirement for HailoRT async transfer buffers — **the single definition for the whole repository**.
//
// HailoRT's vdma_stream.cpp:379 requires 16384 B alignment for read_async(). With 4096 B,
// an "unaligned buffer ... causes performance degradation" warning fires per frame × stream, and
// measured FPS differed by 3 FPS, 48.5 → 51.7.
//
// Before 2026-09-21 this value was written separately in four places (vulkan/dmabuf_export.hpp, tools/hailo/p2_out_buffers.hpp,
// tools/hailo/hailo_raw_probe.cpp, a literal in core/dma_heap_alloc.hpp).
// The values matched but nothing kept them in sync. Being header-only, it does not break the yvo_hailo ⊥ yvo_vulkan isolation.
#include <cstddef>

namespace yvo {
inline constexpr std::size_t kHailoAsyncAlign = 16384;
}  // namespace yvo
