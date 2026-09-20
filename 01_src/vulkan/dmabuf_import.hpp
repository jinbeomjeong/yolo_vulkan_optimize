// dma-buf fd → VkBuffer(STORAGE_BUFFER) import와 캐시
//
// import 절차:
//   1) dup(fd) (vkAllocateMemory 성공 시 드라이버가 dup한 fd의 소유권을 가져간다. 실패하면 우리가 close)
//   2) vkGetMemoryFdPropertiesKHR(DMA_BUF)로 import 가능한 memoryTypeBits 확인
//   3) VkExternalMemoryBufferCreateInfo(DMA_BUF)를 붙여 VkBuffer 생성, 요구사항과 교집합인 메모리 타입 선택
//   4) VkImportMemoryFdInfoKHR(+ 필요하면 VkMemoryDedicatedAllocateInfo)로 vkAllocateMemory → bind
// 캐시 키: fd 번호. 디코더 버퍼 풀의 fd는 고정이지만, 디코더를 다시 열면 번호가 재사용될 수 있으므로
//   fstat(st_dev, st_ino)와 크기로 같은 dma-buf인지 확인하고, 다르면 새로 import한다.
// 수명: 캐시 항목은 dma-buf 참조(dup fd)를 들고 있다. 디코더를 닫은 뒤에는 clear()로 비울 것.
//   사용 중(GPU 대기 중인 커맨드가 참조) 항목은 호출자가 pin/unpin으로 표시하며, 교체·clear 대상에서 빠진다.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <sys/types.h>
#include <vulkan/vulkan.h>

namespace yvo::vk {

class VulkanContext;

struct ImportedDmaBuf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;     // dma-buf 크기 (= import allocationSize)
    VkDeviceSize range = 0;    // VkBuffer 크기. 드라이버 여유 요구 때문에 size보다 작을 수 있다(V3DV: -256 B)
    dev_t st_dev = 0;
    ino_t st_ino = 0;
    int pins = 0;              // 사용 중 참조 수
    std::uint64_t last_use = 0;
};

class DmaBufImportCache {
public:
    // max_entries: 캐시 상한(디코더 풀 크기 × plane 수보다 넉넉히). 초과하면 사용 중이 아닌 가장 오래된 항목 제거.
    explicit DmaBufImportCache(VulkanContext& ctx, std::size_t max_entries = 64);
    ~DmaBufImportCache();
    DmaBufImportCache(const DmaBufImportCache&) = delete;
    DmaBufImportCache& operator=(const DmaBufImportCache&) = delete;

    // fd(size 바이트)를 import한 항목. 캐시에 있으면 재사용. 실패하면 예외.
    //
    // min_range > 0 이면 **강제 검증**: 결과 range가 min_range보다 작으면 예외를 던진다(계획서 부록 B.12).
    // V3DV는 스토리지 버퍼 요구 크기에 여유(+256 B)를 더하고 그만큼 VkBuffer 범위를 줄이므로,
    // dma-buf가 실데이터 크기에 딱 맞으면 셰이더가 마지막 바이트를 못 읽는다. 조용히 넘기면
    // 셰이더가 잘못된 데이터를 읽으므로 여기서 끊는다. 호출자는 실데이터 바이트 수를 넘긴다.
    // 막으려면 할당 때 헤드룸을 더해야 한다(vulkan/vk_storage_headroom.hpp → DmaHeapConfig::device_headroom).
    ImportedDmaBuf& get(int fd, std::size_t size, std::size_t min_range = 0);
    void pin(ImportedDmaBuf& e) { ++e.pins; }
    void unpin(ImportedDmaBuf& e) { --e.pins; }

    // 사용 중이 아닌 항목 전부 해제 (디코더를 닫거나 다시 열 때 호출)
    void clear();

    std::size_t entries() const { return map_.size(); }
    std::uint64_t imports() const { return imports_; }   // 실제 import 횟수
    std::uint64_t hits() const { return hits_; }         // 캐시 적중 횟수

private:
    void release(ImportedDmaBuf& e);
    void evict_if_needed();

    VulkanContext& ctx_;
    std::size_t max_entries_;
    std::unordered_map<int, ImportedDmaBuf> map_;
    std::uint64_t clock_ = 0;
    std::uint64_t imports_ = 0;
    std::uint64_t hits_ = 0;
};

} // namespace yvo::vk
