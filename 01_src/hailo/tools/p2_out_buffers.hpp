// P2(계획서 §6.2) 원시 출력 버퍼 풀 3종: malloc / dma-heap / Vulkan export dma-buf
//
// yvo_hailo_raw_probe --alloc {malloc,heap,vulkan} 이 쓰는 헤더다. 프로브 전용이므로 yvo_hailo
// 라이브러리에 넣지 않는다 — 넣으면 yvo_hailo 가 yvo_vulkan 에 의존하게 되어 계획서 §5.2의
// **링크 수준 격리**가 깨진다(test_preproc / yvo_preproc_bench 의 링크 그래프가 바뀐다).
//
// ── 세 방식의 차이 ────────────────────────────────────────────────────────────────────────────
//   malloc : std::aligned_alloc(16384). fd 가 없으므로 read_async(void*, size) 경로. 기준값이다.
//   heap   : /dev/dma_heap/system (DMA_HEAP_IOCTL_ALLOC). **캐시된** CPU 매핑이라 CPU 읽기가 싸다.
//            대가로 PCIe DMA 쓰기 ↔ CPU 캐시 일관성을 DMA_BUF_IOCTL_SYNC 로 직접 관리해야 한다.
//   vulkan : VulkanContext + DmaBufExporter. V3D 메모리라 셰이더가 직접 읽을 수 있다(계획서 §4.2 채택안).
//            V3D 메모리 타입이 DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT 하나뿐이라 HOST_CACHED 가 없다
//            → **CPU 읽기가 비싸다**(§2). 그래서 sync ioctl 은 필요 없지만 1.22 MB 전체 CPU 읽기는 느리다.
//
// ── 크기 계약 ─────────────────────────────────────────────────────────────────────────────────
// HailoRT read_async 는 16384 B 정렬을 요구한다(부록 A.4). 그래서 heap·vulkan 모두
// align_up(hw_frame_size, 16384) 를 요청한다 → **dma-buf 실제 크기 > hw_frame_size** 가 되고,
// dma_map_dmabuf / read_async 에 어느 크기를 넘길지가 불명확해진다. 그 4조합을 프로브가 판정한다.
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dma_heap_alloc.hpp"
#include "vulkan/dmabuf_export.hpp"
#include "vulkan/vulkan_context.hpp"

namespace yvo::p2 {

inline constexpr std::size_t kHailoAsyncAlign = 16384;   // vulkan/dmabuf_export.hpp 의 kHailoAsyncAlign 과 같다

inline std::size_t align_up(std::size_t v, std::size_t a) { return a <= 1 ? v : (v + a - 1) / a * a; }

// 버퍼 하나를 가리키는 값 타입. 수명은 풀이 쥔다.
struct OutBufRef {
    std::uint8_t* ptr = nullptr;    // CPU 읽기 가능한 시작 주소 (16384 정렬)
    int fd = -1;                    // dma-buf fd. malloc 모드는 -1
    std::size_t dmabuf_size = 0;    // lseek 로 실측한 dma-buf 크기. fd < 0 이면 0
    bool needs_sync = false;        // true 면 CPU 읽기 전후로 DMA_BUF_IOCTL_SYNC 가 필요하다(heap)
};

// CPU 읽기 구간을 여는/닫는다. heap 에서만 실제 ioctl 이 나간다.
// 실제 동작은 common/dma_heap_alloc.hpp 의 dma_sync_* 에 위임한다(부록 B.11 일원화).
// **strict=true** 다 — 기존 구현은 반환값을 버렸다(부록 B.10 R5).
inline void sync_cpu_read_begin(const OutBufRef& b) {
    if (!b.needs_sync || b.fd < 0) return;
    yvo::dma_sync_begin(b.fd, yvo::DmaSyncDir::Read, /*strict=*/true);
}
inline void sync_cpu_read_end(const OutBufRef& b) {
    if (!b.needs_sync || b.fd < 0) return;
    yvo::dma_sync_end(b.fd, yvo::DmaSyncDir::Read, /*strict=*/true);
}

class OutBufPool {
public:
    virtual ~OutBufPool() = default;
    // hw_frame_size 바이트를 담을 버퍼를 하나 더 할당해 돌려준다. 실패하면 예외.
    virtual OutBufRef acquire(std::size_t hw_frame_size) = 0;
    virtual const char* name() const = 0;
    // 지금까지 할당한 버퍼 수
    virtual std::size_t count() const = 0;
};

// ── malloc ───────────────────────────────────────────────────────────────────────────────────
class MallocOutPool final : public OutBufPool {
public:
    OutBufRef acquire(std::size_t hw_frame_size) override {
        const std::size_t rounded = align_up(hw_frame_size, kHailoAsyncAlign);
        void* p = std::aligned_alloc(kHailoAsyncAlign, rounded);
        if (!p) throw std::bad_alloc();
        std::memset(p, 0, rounded);
        bufs_.emplace_back(static_cast<std::uint8_t*>(p), [](std::uint8_t* q) { std::free(q); });
        OutBufRef r;
        r.ptr = bufs_.back().get();
        return r;
    }
    const char* name() const override { return "malloc"; }
    std::size_t count() const override { return bufs_.size(); }

private:
    std::vector<std::shared_ptr<std::uint8_t>> bufs_;
};

// ── /dev/dma_heap (= vidbuf_cached → system 심볼릭 링크) ─────────────────────────────────────
// 장치 노드는 video 그룹 권한이고 사용자 pi 는 video 그룹이다(실측).
//
// **`yvo::DmaHeapPool`(common/dma_heap_alloc.hpp) 위의 얇은 어댑터다**(부록 B.11 일원화).
// 직접 구현했던 것과 달라진 점 셋:
//   · `memset` 이 **WRITE sync 로 감싸진다** — 부록 B.6(c) 의 결함이 여기서 자동으로 고쳐진다
//   · 16384 정렬이 안 맞아도 **throw 하지 않는다** — 예약 후 MAP_FIXED 재매핑으로 맞춘다
//   · `DMA_HEAP_IOCTL_ALLOC` / `DMA_BUF_IOCTL_SYNC` 반환값을 전부 검사한다(B.10 R5)
// `device_headroom` 은 0 이다 — 이 풀의 버퍼는 HailoRT 로만 가고 Vulkan 에 import 하지 않는다.
// (import 하는 경로는 헤드룸을 질의해 넣어야 한다 — 부록 B.12, vulkan/vk_storage_headroom.hpp)
class HeapOutPool final : public OutBufPool {
public:
    explicit HeapOutPool(const std::string& heap = "/dev/dma_heap/system", std::size_t device_headroom = 0)
        : pool_(make_cfg(heap, device_headroom)) {}

    OutBufRef acquire(std::size_t hw_frame_size) override {
        const int i = pool_.allocate(hw_frame_size);
        const yvo::DmaHeapBuf& b = pool_.at(i);
        OutBufRef r;
        r.ptr = b.ptr;
        r.fd = b.fd;              // 소유권은 DmaHeapPool 에 있다. HailoRT 에는 이 fd 를 그대로 넘긴다.
        r.dmabuf_size = b.dmabuf_size;
        r.needs_sync = true;
        return r;
    }
    const char* name() const override { return "heap"; }
    std::size_t count() const override { return pool_.count(); }

    const std::string& heap_path() const { return pool_.heap(); }

private:
    static yvo::DmaHeapConfig make_cfg(const std::string& heap, std::size_t device_headroom) {
        yvo::DmaHeapConfig c;
        c.heaps = {heap};
        c.align = kHailoAsyncAlign;
        c.device_headroom = device_headroom;
        c.map = true;
        c.strict_sync = true;
        c.zero_init = true;
        c.name = "yvo_p2_out";
        return c;
    }
    yvo::DmaHeapPool pool_;
};

// ── Vulkan export dma-buf (계획서 §4.2 채택안) ────────────────────────────────────────────────
// VulkanContext 는 present=off 로 만든다(표시와 무관하고, off 면 확장 목록이 오늘과 같다 — §5.2).
class VulkanOutPool final : public OutBufPool {
public:
    VulkanOutPool(bool verbose = false) {
        yvo::vk::VulkanContextOptions co;
        co.verbose = verbose;
        co.enable_present = false;
        ctx_ = std::make_unique<yvo::vk::VulkanContext>(co);
        yvo::vk::DmaBufExporterConfig ec;
        ec.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;   // 다음 단계에서 yolo_decode.comp 가 읽는다
        ec.align = kHailoAsyncAlign;
        ec.map = true;
        ec.verbose = verbose;
        exp_ = std::make_unique<yvo::vk::DmaBufExporter>(*ctx_, ec);
    }

    OutBufRef acquire(std::size_t hw_frame_size) override {
        const int idx = exp_->allocate(hw_frame_size);
        const auto& e = exp_->at(idx);
        std::memset(e.aligned_host_ptr(), 0, hw_frame_size);
        OutBufRef r;
        r.ptr = e.aligned_host_ptr();
        r.fd = e.fd;             // 소유권은 DmaBufExporter 에 있다. HailoRT 에는 이 fd 를 그대로 넘긴다.
        r.dmabuf_size = e.dmabuf_size;
        r.needs_sync = false;    // HOST_COHERENT
        return r;
    }
    const char* name() const override { return "vulkan"; }
    std::size_t count() const override { return exp_->count(); }

    const yvo::vk::VulkanContext& ctx() const { return *ctx_; }
    // 첫 버퍼의 3종 크기(보고용). count()==0 이면 전부 0.
    void size_triplet(std::size_t& requested, std::size_t& alloc, std::size_t& dmabuf) const {
        if (exp_->count() == 0) {
            requested = alloc = dmabuf = 0;
            return;
        }
        const auto& e = exp_->at(0);
        requested = e.requested_size;
        alloc = static_cast<std::size_t>(e.alloc_size);
        dmabuf = e.dmabuf_size;
    }

private:
    std::unique_ptr<yvo::vk::VulkanContext> ctx_;
    std::unique_ptr<yvo::vk::DmaBufExporter> exp_;
};

} // namespace yvo::p2
