// DisplayBackend / OffscreenBackend: DisplayPipeline 이 보는 "표시 대상 이미지 공급자" 추상화.
//
// 왜 추상화하는가:
//   `DisplayPipeline` 의 프레임 루프(acquire → begin → 배리어 → convert → overlay → 배리어 →
//   submit → present → 펜스 폴링)는 화면이 있든 없든 **완전히 같다**. 다른 것은 이미지를 어디서
//   받고 결과를 어디로 보내느냐뿐이다. 그래서 그 둘만 가상 함수로 뽑고 루프는 하나만 둔다.
//     · `SwapchainBackend` (display_pipeline.cpp 내부) — `VulkanSwapchain` 어댑터. 실제 HDMI 경로.
//     · `OffscreenBackend`  (이 파일)                  — Wayland 가 없을 때의 헤드리스 폴백.
//
// OffscreenBackend 가 하는 일:
//   스왑체인 이미지 대신 **자체 R8G8B8A8_UNORM STORAGE 이미지 N장 + 펜스 N개**를 돌린다.
//   세마포어는 만들지 않는다 — `DisplayConvert::end_and_submit` 이 VK_NULL_HANDLE 을 받으면
//   waitSemaphoreInfoCount/signalSemaphoreInfoCount 를 0 으로 둔다(display_convert.cpp 확인).
//   present 는 아무 데도 보내지 않고 presented_ 만 센다. 따라서 present_wait 도 없다
//   (`present_wait_available()==false` → glass-to-glass 통계가 비어 있는 것이 정상이다).
//   이미지 레이아웃은 스왑체인과 같은 UNDEFINED → GENERAL → (PRESENT_SRC 대신) GENERAL 유지다.
//   PRESENT_SRC 로 보내면 안 된다 — 스왑체인이 아닌 이미지에는 그 레이아웃을 쓸 이유가 없고,
//   다음 프레임의 acquire 전환이 어차피 UNDEFINED 에서 시작한다.
#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "display/vk_swapchain.hpp"

namespace yvo::vk {
class VulkanContext;
}

namespace yvo::display {

// DisplayPipeline 이 쓰는 최소 표면. VulkanSwapchain 의 부분집합과 이름·의미를 맞춰 두었다.
class DisplayBackend {
public:
    virtual ~DisplayBackend() = default;

    // 절대 블로킹하지 않는다. 쓸 이미지가 없으면 false (표시 드롭).
    virtual bool acquire(SwapchainFrame* out) = 0;
    virtual void record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const = 0;
    virtual void record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const = 0;
    virtual void present(const SwapchainFrame& f) = 0;

    virtual bool needs_recreate() const { return false; }
    virtual void request_recreate(int /*w*/, int /*h*/) {}
    virtual void recreate(int /*w*/, int /*h*/) {}
    virtual void collect_retired() {}

    virtual VkExtent2D extent() const = 0;
    virtual std::uint32_t image_count() const = 0;
    virtual bool blit_mode() const { return false; }

    virtual bool present_wait_available() const { return false; }
    virtual bool wait_present(std::uint64_t /*id*/, std::uint64_t /*timeout_ns*/) { return false; }

    virtual std::uint64_t presented() const = 0;
    virtual std::uint64_t display_dropped() const = 0;
    virtual std::uint64_t recreates() const { return 0; }
    virtual const char* name() const = 0;
};

class OffscreenBackend final : public DisplayBackend {
public:
    OffscreenBackend(yvo::vk::VulkanContext& ctx, int width, int height, std::uint32_t images = 3);
    ~OffscreenBackend() override;
    OffscreenBackend(const OffscreenBackend&) = delete;
    OffscreenBackend& operator=(const OffscreenBackend&) = delete;

    bool acquire(SwapchainFrame* out) override;
    void record_acquire_transition(VkCommandBuffer cb, const SwapchainFrame& f) const override;
    void record_present_transition(VkCommandBuffer cb, const SwapchainFrame& f) const override;
    void present(const SwapchainFrame& f) override;

    VkExtent2D extent() const override { return extent_; }
    std::uint32_t image_count() const override { return static_cast<std::uint32_t>(images_.size()); }

    std::uint64_t presented() const override { return presented_; }
    std::uint64_t display_dropped() const override { return dropped_; }
    const char* name() const override { return "offscreen"; }

private:
    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool in_flight = false;
    };
    void destroy();

    yvo::vk::VulkanContext& ctx_;
    VkExtent2D extent_{};
    std::vector<Image> images_;
    std::uint32_t next_ = 0;
    std::uint64_t presented_ = 0, dropped_ = 0;
};

} // namespace yvo::display
