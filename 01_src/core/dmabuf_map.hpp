// dma-buf CPU 읽기 매핑 (헤더 전용). 생성 시 DMA_BUF_SYNC_START|READ, 소멸 시 END|READ로 CPU 캐시 일관성을 맞춘다.
// CPU 기준 경로(테스트·벤치)에서 디코더 dma-buf를 읽을 때 쓴다.
//
// **남의 fd를 읽기 전용으로 매핑하는 별개 용도다** — 할당하지 않는다. 직접 할당하는 쪽은
// core/dma_heap_alloc.hpp 의 DmaHeapPool 이다(부록 B.11 일원화).
// ioctl 두 곳은 그 헤더의 dma_sync_begin/end 에 위임한다. strict=false 로 불러
// **기존 동작(ioctl 실패를 무시)을 그대로 유지**한다 — 이 클래스의 공개 API와 동작은 바뀌지 않았다.
#pragma once

#include <cstddef>
#include <cstdint>

#include <sys/mman.h>

#include "core/dma_heap_alloc.hpp"

namespace yvo {

class DmaBufReadMap {
public:
    DmaBufReadMap(int fd, std::size_t size) : fd_(fd), size_(size) {
        void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) return;
        ptr_ = static_cast<const std::uint8_t*>(p);
        dma_sync_begin(fd_, DmaSyncDir::Read, /*strict=*/false);
    }
    ~DmaBufReadMap() {
        if (!ptr_) return;
        dma_sync_end(fd_, DmaSyncDir::Read, /*strict=*/false);
        munmap(const_cast<std::uint8_t*>(ptr_), size_);
    }
    DmaBufReadMap(const DmaBufReadMap&) = delete;
    DmaBufReadMap& operator=(const DmaBufReadMap&) = delete;
    const std::uint8_t* data() const { return ptr_; }

private:
    int fd_;
    std::size_t size_;
    const std::uint8_t* ptr_ = nullptr;
};

} // namespace yvo
