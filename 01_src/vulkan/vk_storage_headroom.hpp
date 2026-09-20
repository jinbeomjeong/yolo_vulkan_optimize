// dma-buf import용 STORAGE_BUFFER 헤드룸 질의 (헤더 전용) — 계획서 부록 B.12
//
// 문제: V3DV는 스토리지 버퍼의 메모리 요구 크기에 TMU 선읽기용 여유를 더한다(실측 +256 B, 크기 무관).
// import 메모리 크기는 dma-buf 크기로 고정이므로, dma-buf가 요청 크기에 딱 맞으면
// `DmaBufImportCache::get()`이 VkBuffer 범위를 그만큼 **줄여** 대응한다 → 셰이더가 마지막 바이트를
// 못 읽는다. P0(부록 B.17)에서 `conv71_178`(409,600 = 16384×25, 여유 0)이 정확히 이 상태였다:
// `range 409,344 < hw 409,600`.
//
// 해법: dma-heap에서 잡을 때 **미리 헤드룸을 더해서** 잡는다.
//   1) 더미 VkBuffer 생성 (size = requested, usage = STORAGE_BUFFER,
//                          pNext = VkExternalMemoryBufferCreateInfo{DMA_BUF})
//   2) vkGetBufferMemoryRequirements2() → req.size
//   3) headroom = req.size - requested          (V3D 실측: 항상 256, 크기 무관)
//   4) 더미 파괴
// → 호출자가 `yvo::DmaHeapConfig::device_headroom`에 넣으면
//   `DmaHeapPool::allocate()`가 `align_up(requested + device_headroom, align)`으로 잡는다.
//
// **상수 256을 박지 않는다** — 드라이버 의존이다. 질의 결과는 크기별로 memoize한다.
//
// 과할당이 spec 위반이 되지 않는가: `VUID-VkMemoryDedicatedAllocateInfo-buffer-02965`는
// "buffer가 VK_NULL_HANDLE이 아니면 allocationSize가 그 버퍼의 VkMemoryRequirements::size와 같아야 한다"고
// 요구하지만, 이는 **전용 할당(VkMemoryDedicatedAllocateInfo)을 붙였을 때만** 적용된다.
// P0 실측에서 V3D는 6/6 `requiresDedicatedAllocation = prefersDedicatedAllocation = 0`이라
// `dmabuf_import.cpp`가 그 구조체를 붙이지 않는다 → 과할당이 걸리지 않는다.
// (validusage.json, Vulkan 1.4.309 / commit 43d153dd)
#pragma once

#include <cstddef>
#include <unordered_map>

#include <vulkan/vulkan.h>

#include "vulkan/vk_check.hpp"

namespace yvo::vk {

// requested 바이트짜리 외부 메모리(dma-buf) STORAGE_BUFFER의 메모리 요구를 묻는다. 할당하지 않는다.
// ded가 non-null이면 VkMemoryDedicatedRequirements도 함께 채운다.
inline VkMemoryRequirements query_dmabuf_storage_requirements(VkDevice dev, VkDeviceSize requested,
                                                              VkMemoryDedicatedRequirements* ded = nullptr) {
    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext;
    bci.size = requested;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer dummy = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &dummy));

    VkMemoryDedicatedRequirements local{};
    local.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    VkMemoryRequirements2 req2{};
    req2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    req2.pNext = &local;
    VkBufferMemoryRequirementsInfo2 ri{};
    ri.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    ri.buffer = dummy;
    vkGetBufferMemoryRequirements2(dev, &ri, &req2);
    vkDestroyBuffer(dev, dummy, nullptr);

    if (ded) *ded = local;
    return req2.memoryRequirements;
}

// 크기별 헤드룸 캐시. 같은 크기를 두 번 묻지 않는다(부록 B.12: "텐서마다 다시 묻지 않는다").
class StorageHeadroomQuery {
public:
    explicit StorageHeadroomQuery(VkDevice dev) : dev_(dev) {}

    // requested 바이트를 담을 dma-buf가 추가로 가져야 하는 바이트 수.
    // 드라이버가 여유를 요구하지 않으면 0이다.
    std::size_t headroom(std::size_t requested) {
        auto it = memo_.find(requested);
        if (it != memo_.end()) return it->second;
        const VkMemoryRequirements req = query_dmabuf_storage_requirements(dev_, requested);
        const std::size_t h = req.size > requested ? static_cast<std::size_t>(req.size - requested) : 0;
        last_alignment_ = req.alignment;
        memo_.emplace(requested, h);
        return h;
    }

    // 여러 크기 중 최대 헤드룸. `DmaHeapConfig::device_headroom`은 풀 하나에 값 하나라,
    // 여러 크기를 담는 풀에는 최대값을 넣는다. **과할당은 안전하다**(위 주석의 VUID-02965 설명 참고).
    template <typename It>
    std::size_t max_headroom(It first, It last) {
        std::size_t m = 0;
        for (It i = first; i != last; ++i) {
            const std::size_t h = headroom(static_cast<std::size_t>(*i));
            if (h > m) m = h;
        }
        return m;
    }

    VkDeviceSize last_alignment() const { return last_alignment_; }
    std::size_t queries() const { return memo_.size(); }

private:
    VkDevice dev_;
    std::unordered_map<std::size_t, std::size_t> memo_;
    VkDeviceSize last_alignment_ = 0;
};

} // namespace yvo::vk
