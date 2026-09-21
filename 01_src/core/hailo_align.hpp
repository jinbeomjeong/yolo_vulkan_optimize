#pragma once
// HailoRT 비동기 전송 버퍼의 정렬 요구 — **저장소 전체의 단일 정의**.
//
// HailoRT 의 vdma_stream.cpp:379 가 read_async() 에 16384 B 정렬을 요구한다. 4096 B 로 주면
// "unaligned buffer ... causes performance degradation" 경고가 프레임×스트림마다 뜨고
// 실측으로 FPS 가 48.5 → 51.7 로 3 FPS 차이가 났다.
//
// 2026-09-21 이전에는 이 값이 네 곳(vulkan/dmabuf_export.hpp, hailo/tools/p2_out_buffers.hpp,
// hailo/tools/hailo_raw_probe.cpp, core/dma_heap_alloc.hpp 리터럴)에 따로 적혀 있었다.
// 값은 같았지만 동기화 장치가 없었다. 헤더 전용이라 yvo_hailo ⊥ yvo_vulkan 격리를 깨지 않는다.
#include <cstddef>

namespace yvo {
inline constexpr std::size_t kHailoAsyncAlign = 16384;
}  // namespace yvo
