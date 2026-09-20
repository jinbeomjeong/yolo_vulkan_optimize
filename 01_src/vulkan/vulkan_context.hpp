// Vulkan 인스턴스·장치·컴퓨트 큐 (Pi 5 V3D 전용 선택, llvmpipe 제외)
//
// - 물리 장치: deviceType != CPU 이고 이름에 "V3D"가 들어간 장치(환경변수 YVO_VK_DEVICE로 부분 문자열 변경 가능).
// - 필수 확장: VK_KHR_external_memory_fd, VK_EXT_external_memory_dma_buf (dma-buf import/export).
//   선택 확장: VK_EXT_queue_family_foreign (있으면 외부 소유권 전환에 FOREIGN, 없으면 EXTERNAL).
// - Vulkan 1.3 synchronization2, maintenance4(셰이더 LocalSizeId) 기능을 켠다.
// - 검증 레이어: options.enable_validation 또는 환경변수 YVO_VK_VALIDATION=1이면 VK_LAYER_KHRONOS_validation을
//   명시적으로 켠다. VK_LOADER_LAYERS_ENABLE/VK_INSTANCE_LAYERS로 켠 경우에도 VK_EXT_debug_utils가 보이면
//   메신저를 등록해 에러·경고 개수를 센다(validation_errors()).
//
// ── present(표시) 옵션 ─────────────────────────────────────────────────────────────────────────
// enable_present == false(기본)면 **오늘까지의 동작과 완전히 같다**: 인스턴스 확장은 VK_EXT_debug_utils 하나,
// 디바이스 확장과 큐 패밀리 선택 결과도 그대로다. 회귀 판정은 yvo_display_probe --probe context-regression 이
// 한다.
//
// 닭-달걀 문제의 해법: vkGetPhysicalDeviceSurfaceSupportKHR 는 VkSurfaceKHR 를 요구하고 surface 는 wl_display 를
// 요구한다. 그러나 vkGetPhysicalDeviceWaylandPresentationSupportKHR(pd, qf, wl_display) 는 **surface가 필요
// 없다.** 따라서 WaylandWindow 를 VulkanContext 보다 먼저 만들고 wl_display* 만 불투명 포인터로 넘긴다.
// **VulkanContext 는 VkSurfaceKHR 를 영원히 모른다** — surface 생성은 VulkanSwapchain(01_src/display) 몫이다.
// 이 헤더가 <wayland-client.h> 나 vulkan_wayland.h 를 포함하지 않는 것도 같은 이유다(yvo_vulkan 의 의존성을
// 늘리지 않는다).
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace yvo::vk {

struct VulkanContextOptions {
    bool enable_validation = false;   // VK_LAYER_KHRONOS_validation 명시 활성화
    bool validation_sync = false;     // synchronization validation (enable_validation일 때만)
    std::string device_name_hint = "V3D";
    bool verbose = false;             // 선택된 장치 정보 출력

    // surface/swapchain 확장 + present 지원 큐를 요구한다. false면 아래 두 필드는 무시된다.
    bool enable_present = false;
    // struct wl_display* (불투명). enable_present일 때 필수다. 소유권은 호출자(WaylandWindow)에 있다.
    void* wayland_display = nullptr;
    // true면 present를 못 켤 때 예외를 던진다. false면 조용히 present 없이 계속한다(present_enabled()==false).
    bool require_present = false;
};

class VulkanContext {
public:
    explicit VulkanContext(const VulkanContextOptions& opt = {});
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return phys_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    std::uint32_t queue_family() const { return queue_family_; }
    const VkPhysicalDeviceProperties& properties() const { return props_; }
    const char* device_name() const { return props_.deviceName; }

    // 외부(dma-buf) 자원 소유권 전환에 쓸 큐 패밀리 (FOREIGN_EXT 또는 EXTERNAL)
    std::uint32_t external_queue_family() const { return external_qf_; }

    // ── present 관련 ──
    // present가 실제로 켜졌는가 (enable_present였고 필요한 확장·큐가 모두 있었는가)
    bool present_enabled() const { return present_enabled_; }
    // present를 못 켠 이유 (present_enabled()==true면 빈 문자열)
    const std::string& present_unavailable_reason() const { return present_reason_; }
    // 선택된 큐 패밀리가 GRAPHICS를 지원하는가 (그래픽스 폴백 경로 판정용)
    bool graphics_supported() const { return graphics_supported_; }
    // VK_EXT_swapchain_maintenance1 (디바이스). 인스턴스 확장 VK_EXT_surface_maintenance1을 전제로 한다.
    bool has_swapchain_maintenance1() const { return swapchain_maintenance1_; }
    // VK_EXT_surface_maintenance1 (인스턴스). present mode별 minImageCount 질의에 쓴다.
    bool has_surface_maintenance1() const { return surface_maintenance1_; }
    // VK_KHR_present_id (present_wait의 전제)
    bool has_present_id() const { return present_id_; }
    // VK_KHR_present_wait. glass-to-glass 지연을 추측이 아니라 실측하는 수단이다.
    bool has_present_wait() const { return present_wait_; }

    // 실제로 켠(enabled) 확장 목록. 회귀 판정(yvo_display_probe)이 이 목록을 그대로 출력한다.
    const std::vector<std::string>& enabled_instance_extensions() const { return enabled_inst_exts_; }
    const std::vector<std::string>& enabled_device_extensions() const { return enabled_dev_exts_; }

    // timestamp: 1 tick = timestamp_period() ns, 유효 비트 수는 timestamp_valid_bits()
    bool timestamps_supported() const { return timestamps_supported_; }
    float timestamp_period() const { return props_.limits.timestampPeriod; }
    std::uint32_t timestamp_valid_bits() const { return timestamp_valid_bits_; }

    // STORAGE_BUFFER 용도의 dma-buf import/export 지원 여부 (vkGetPhysicalDeviceExternalBufferProperties)
    bool dmabuf_import_supported() const { return dmabuf_import_; }
    bool dmabuf_export_supported() const { return dmabuf_export_; }
    bool dmabuf_import_dedicated_only() const { return dmabuf_import_dedicated_only_; }
    bool dmabuf_export_dedicated_only() const { return dmabuf_export_dedicated_only_; }

    // memoryTypeBits 중 required 플래그를 모두 가진 타입. preferred를 더 많이 가진 쪽 우선. 없으면 UINT32_MAX.
    std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred = 0) const;
    const VkPhysicalDeviceMemoryProperties& memory_properties() const { return mem_props_; }

    // 확장 함수 포인터
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR = nullptr;

    // 디버그 메신저가 받은 메시지 수 (메신저가 없으면 0)
    bool debug_messenger_active() const { return messenger_ != VK_NULL_HANDLE; }
    bool validation_layer_active() const { return validation_layer_active_; }
    std::uint64_t validation_errors() const { return errors_.load(); }
    std::uint64_t validation_warnings() const { return warnings_.load(); }

private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                         VkDebugUtilsMessageTypeFlagsEXT types,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                         void* user);
    void destroy();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    std::uint32_t external_qf_ = VK_QUEUE_FAMILY_EXTERNAL;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties mem_props_{};
    bool timestamps_supported_ = false;
    std::uint32_t timestamp_valid_bits_ = 0;
    bool dmabuf_import_ = false;
    bool dmabuf_export_ = false;
    bool dmabuf_import_dedicated_only_ = false;
    bool dmabuf_export_dedicated_only_ = false;
    bool validation_layer_active_ = false;
    bool present_enabled_ = false;
    bool graphics_supported_ = false;
    bool surface_maintenance1_ = false;
    bool swapchain_maintenance1_ = false;
    bool present_id_ = false;
    bool present_wait_ = false;
    std::string present_reason_;
    std::vector<std::string> enabled_inst_exts_;
    std::vector<std::string> enabled_dev_exts_;
    std::atomic<std::uint64_t> errors_{0};
    std::atomic<std::uint64_t> warnings_{0};
    std::atomic<std::uint64_t> printed_{0};
};

} // namespace yvo::vk
