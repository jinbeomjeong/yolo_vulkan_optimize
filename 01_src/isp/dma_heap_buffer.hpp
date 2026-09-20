// dma-heap에서 잡은 CPU 캐시 매핑 버퍼 묶음 (슬롯마다 하나)
//
// C안(ISP 출력 → NHCW 전치)에서 전치 결과를 담을 버퍼가 필요하다. HailoRT에 NHCW로 넘기려면
// 그 버퍼를 dma-buf fd로 줘야 하고(HailoRunner가 set_dma_buffer + 사전 map_dmabuf를 쓴다),
// CPU가 쓰는 버퍼이므로 캐시 매핑이어야 한다. /dev/dma_heap/vidbuf_cached(→ system)가 그것이다.
//
// CPU로 쓴 뒤 장치(NPU)가 읽으므로 DMA_BUF_IOCTL_SYNC로 감싼다: begin_cpu_write → 쓰기 → end_cpu_write.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace yvo::isp {

class DmaHeapBuffers {
public:
    // name: dma-buf 이름(디버깅용). size: 버퍼 하나의 바이트 수. count: 개수.
    DmaHeapBuffers(const std::string& name, std::size_t size, int count);
    ~DmaHeapBuffers();
    DmaHeapBuffers(const DmaHeapBuffers&) = delete;
    DmaHeapBuffers& operator=(const DmaHeapBuffers&) = delete;

    int count() const { return static_cast<int>(bufs_.size()); }
    std::size_t size() const { return size_; }
    int fd(int i) const;
    std::uint8_t* ptr(int i) const;
    const std::string& heap() const { return heap_; }

    void begin_cpu_write(int i);
    void end_cpu_write(int i);

private:
    struct Buf {
        int fd = -1;
        std::uint8_t* ptr = nullptr;
    };
    std::vector<Buf> bufs_;
    std::size_t size_ = 0;
    std::string heap_;
};

} // namespace yvo::isp
