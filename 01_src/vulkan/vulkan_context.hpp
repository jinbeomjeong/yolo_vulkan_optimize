// Vulkan instance, device, compute queue (Pi 5 V3D-specific selection, llvmpipe excluded)
//
// - Physical device: deviceType != CPU and name contains "V3D" (the substring can be changed with env var YVO_VK_DEVICE).
// - Required extensions: VK_KHR_external_memory_fd, VK_EXT_external_memory_dma_buf (dma-buf import/export).
//   Optional extension: VK_EXT_queue_family_foreign (if present, FOREIGN for external ownership transfer, otherwise EXTERNAL).
// - Enables the Vulkan 1.3 synchronization2 and maintenance4 (shader LocalSizeId) features.
// - Validation layer: if options.enable_validation or env var YVO_VK_VALIDATION=1, VK_LAYER_KHRONOS_validation is
//   enabled explicitly. Even when enabled via VK_LOADER_LAYERS_ENABLE/VK_INSTANCE_LAYERS, if VK_EXT_debug_utils is visible,
//   a messenger is registered to count errors and warnings (validation_errors()).
// - No surface/swapchain (present) extensions: nothing presents through Vulkan. The display path puts dma-bufs on HVS
//   planes through the compositor (01_src/display), and its boxes are drawn by a compute dispatch into an exported buffer.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace yvo::vk {

struct VulkanContextOptions {
    bool enable_validation = false;   // explicitly enable VK_LAYER_KHRONOS_validation
    bool validation_sync = false;     // synchronization validation (only with enable_validation)
    std::string device_name_hint = "V3D";
    bool verbose = false;             // print selected device info
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

    // Queue family used for ownership transfer of external (dma-buf) resources (FOREIGN_EXT or EXTERNAL)
    std::uint32_t external_queue_family() const { return external_qf_; }

    // List of extensions actually enabled.
    const std::vector<std::string>& enabled_instance_extensions() const { return enabled_inst_exts_; }
    const std::vector<std::string>& enabled_device_extensions() const { return enabled_dev_exts_; }

    // timestamp: 1 tick = timestamp_period() ns, number of valid bits is timestamp_valid_bits()
    bool timestamps_supported() const { return timestamps_supported_; }
    float timestamp_period() const { return props_.limits.timestampPeriod; }
    std::uint32_t timestamp_valid_bits() const { return timestamp_valid_bits_; }

    // Whether dma-buf import/export is supported for STORAGE_BUFFER usage (vkGetPhysicalDeviceExternalBufferProperties)
    bool dmabuf_import_supported() const { return dmabuf_import_; }
    bool dmabuf_export_supported() const { return dmabuf_export_; }
    bool dmabuf_import_dedicated_only() const { return dmabuf_import_dedicated_only_; }
    bool dmabuf_export_dedicated_only() const { return dmabuf_export_dedicated_only_; }

    // Type among memoryTypeBits that has all required flags. Prefer the one with more preferred flags. UINT32_MAX if none.
    std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred = 0) const;
    const VkPhysicalDeviceMemoryProperties& memory_properties() const { return mem_props_; }

    // Extension function pointers
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR = nullptr;

    // Number of messages received by the debug messenger (0 if there is no messenger)
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
    std::vector<std::string> enabled_inst_exts_;
    std::vector<std::string> enabled_dev_exts_;
    std::atomic<std::uint64_t> errors_{0};
    std::atomic<std::uint64_t> warnings_{0};
    std::atomic<std::uint64_t> printed_{0};
};

} // namespace yvo::vk
