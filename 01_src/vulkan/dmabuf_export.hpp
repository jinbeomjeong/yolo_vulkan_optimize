// VkBuffer + VkDeviceMemory 를 dma-buf 로 내보내는 재사용 헬퍼 (후처리 트랙 P2 가 쓴다)
//
// sand_letterbox.cpp 의 출력 슬롯 할당 + export_dmabuf_fd() 경로를 클래스로 뽑아낸 것이다
// (원본은 튜닝된 2.13 ms 경로라 손대지 않는다 — 여기에 새로 썼다).
//
// 할당 절차:
//   1) VkExternalMemoryBufferCreateInfo(DMA_BUF) 를 붙여 VkBuffer 생성
//   2) VkExportMemoryAllocateInfo(DMA_BUF) 로 vkAllocateMemory (할당 하나 = 버퍼 하나 = dma-buf 하나.
//      버퍼를 오프셋 0 에 바인드하므로 dma-buf 시작 주소가 곧 버퍼 시작이다.
//      VkMemoryDedicatedAllocateInfo 는 드라이버가 요구할 때만 붙인다 — .cpp 의 정렬 주석 참고)
//   3) vkBindBufferMemory → vkMapMemory(영구 매핑) → vkGetMemoryFdKHR 로 fd 내보내기
//
// ── 정렬(align) ──────────────────────────────────────────────────────────────────────────────
// **HailoRT read_async 는 16384 B 정렬을 요구한다**(후처리 트랙 실측: 4096 이면 vdma_stream.cpp:379 가
// 프레임 x 스트림마다 "unaligned buffer causes performance degradation" 경고를 낸다).
// 따라서 기본 align 은 16384 이고 두 가지를 모두 맞춘다:
//   · **할당 크기**: align_up(VkMemoryRequirements::size, align). V3DV 는 버퍼 크기 + 256 B 를 요구하므로
//     전용 할당을 쓰면 이 값을 올릴 수 없다(spec 이 allocationSize == req.size 를 강제한다) → 전용 할당은
//     드라이버가 요구할 때만 쓴다. V3D 는 요구하지 않는다(실측).
//   · **시작 주소**: dma-buf 경로는 오프셋 0 이라 자동으로 맞는다. 호스트 포인터 경로는 vkMapMemory 가
//     돌려주는 주소가 mmap 정렬(보통 4096, 큰 할당이면 더 크다)이라 보장이 없으므로, 정렬이 안 맞으면
//     **align 만큼 여유를 더해 한 번 다시 할당하고** aligned_host_ptr() 로 정렬된 시작을 노출한다.
//     (첫 시도가 거의 항상 성공하므로 평상시 낭비는 0 이다.)
//
// ── 크기가 두 값인 이유 ───────────────────────────────────────────────────────────────────────
// V3DV 가 vkGetBufferMemoryRequirements 로 요구하는 크기는 실제 dma-buf 크기와 다를 수 있다
// (import 쪽에서도 같은 이유로 dmabuf_import.hpp 가 -256 B 범위 축소를 한다).
// 그리고 HailoRT 의 dma_map_dmabuf(fd, size, D2H) / read_async(fd, size) 에 **요청 크기**를 넘길지
// **dma-buf 실제 크기**를 넘길지가 미확정이라, 호출자가 둘 다 시험할 수 있게 두 값을 모두 노출한다:
//   requested_size : 호출자가 allocate() 에 넘긴 바이트 수 (텐서 크기)
//   alloc_size     : vkAllocateMemory 에 넘긴 크기 (align 배수)
//   dmabuf_size    : lseek(fd, 0, SEEK_END) 로 실측한 dma-buf 객체 크기
// V3D 실측(yvo_display_probe --probe export): alloc_size == dmabuf_size 이고 둘 다 16384 배수지만
// **requested_size 보다 항상 크다**(예: 1,228,800 요청 → 1,245,184). 7개 크기 모두 그랬다.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>
#include "core/hailo_align.hpp"

namespace yvo::vk {

class VulkanContext;

// HailoRT read_async 가 요구하는 정렬. 후처리 트랙이 실측한 값이다.
using yvo::kHailoAsyncAlign;  // 정의는 core/hailo_align.hpp 하나뿐이다

struct ExportedDmaBuf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    int fd = -1;                        // DmaBufExporter 소유. 넘겨줄 때는 dup_fd() 를 쓴다.
    std::size_t requested_size = 0;     // 호출자가 요청한 바이트 수
    VkDeviceSize alloc_size = 0;        // 실제 vkAllocateMemory 크기 (align 배수)
    std::size_t dmabuf_size = 0;        // lseek 로 실측한 dma-buf 크기 (0 이면 측정 실패)
    std::uint8_t* host_ptr = nullptr;   // 영구 매핑 (매핑 시작)
    std::size_t align_offset = 0;       // host_ptr 부터 align 정렬까지의 바이트 (보통 0)
    // 전용 할당(VkMemoryDedicatedAllocateInfo)을 썼는가. true 면 spec 이 allocationSize == req.size 를
    // 강제하므로 alloc_size 가 align 배수가 아닐 수 있다 (V3D 는 false 라 이 경우가 생기지 않는다).
    bool dedicated = false;
    // align 정렬된 시작 주소. align_offset 이 0 이면 host_ptr 과 같다.
    std::uint8_t* aligned_host_ptr() const { return host_ptr + align_offset; }
};

struct DmaBufExporterConfig {
    // 버퍼 usage. 후처리 트랙은 STORAGE_BUFFER(셰이더가 읽는다) 를 쓴다.
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    std::size_t align = kHailoAsyncAlign;   // 할당 크기·시작 주소 정렬
    bool map = true;                        // 영구 매핑 (host_ptr). false 면 nullptr
    bool verbose = false;
};

class DmaBufExporter {
public:
    explicit DmaBufExporter(VulkanContext& ctx, const DmaBufExporterConfig& cfg = {});
    ~DmaBufExporter();
    DmaBufExporter(const DmaBufExporter&) = delete;
    DmaBufExporter& operator=(const DmaBufExporter&) = delete;

    // size 바이트를 담을 버퍼를 새로 할당하고 dma-buf 로 export 한다. 인덱스를 돌려준다. 실패하면 예외.
    int allocate(std::size_t size);

    std::size_t count() const { return bufs_.size(); }
    const ExportedDmaBuf& at(int index) const;
    // 호출자에게 넘길 새 fd (호출자가 close 한다). 원본 fd 는 이 객체가 계속 들고 있다.
    int dup_fd(int index) const;

    const DmaBufExporterConfig& config() const { return cfg_; }

private:
    void destroy();
    // 한 번 할당을 시도한다. 매핑 주소 정렬까지 확인해 align_offset 을 채운다.
    ExportedDmaBuf allocate_once(std::size_t requested, VkDeviceSize min_alloc);
    void release(ExportedDmaBuf& e);

    VulkanContext& ctx_;
    DmaBufExporterConfig cfg_;
    std::vector<ExportedDmaBuf> bufs_;
};

} // namespace yvo::vk
