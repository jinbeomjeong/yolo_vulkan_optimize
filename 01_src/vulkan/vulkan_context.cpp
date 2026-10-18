#include "vulkan/vulkan_context.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vulkan/vk_check.hpp"

namespace yvo::vk {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

bool env_contains(const char* name, const char* needle) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strstr(v, needle) != nullptr;
}

bool env_true(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

std::vector<VkExtensionProperties> instance_extensions(const char* layer) {
    std::uint32_t n = 0;
    if (vkEnumerateInstanceExtensionProperties(layer, &n, nullptr) != VK_SUCCESS) return {};
    std::vector<VkExtensionProperties> v(n);
    if (vkEnumerateInstanceExtensionProperties(layer, &n, v.data()) != VK_SUCCESS) return {};
    v.resize(n);
    return v;
}

bool has_ext(const std::vector<VkExtensionProperties>& v, const char* name) {
    for (const auto& e : v)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

bool layer_available(const char* name) {
    std::uint32_t n = 0;
    if (vkEnumerateInstanceLayerProperties(&n, nullptr) != VK_SUCCESS) return false;
    std::vector<VkLayerProperties> v(n);
    if (vkEnumerateInstanceLayerProperties(&n, v.data()) != VK_SUCCESS) return false;
    for (std::uint32_t i = 0; i < n; ++i)
        if (std::strcmp(v[i].layerName, name) == 0) return true;
    return false;
}

std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice pd) {
    std::uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr) != VK_SUCCESS) return {};
    std::vector<VkExtensionProperties> v(n);
    if (vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, v.data()) != VK_SUCCESS) return {};
    v.resize(n);
    return v;
}

// Query dma-buf external memory capabilities for STORAGE_BUFFER usage
VkExternalMemoryFeatureFlags dmabuf_buffer_features(VkPhysicalDevice pd) {
    VkPhysicalDeviceExternalBufferInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkExternalBufferProperties props{};
    props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(pd, &info, &props);
    return props.externalMemoryProperties.externalMemoryFeatures;
}

} // namespace

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                             void* user) {
    auto* self = static_cast<VulkanContext*>(user);
    const char* tag = "INFO";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        self->errors_.fetch_add(1);
        tag = "ERROR";
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        self->warnings_.fetch_add(1);
        tag = "WARNING";
    } else {
        return VK_FALSE;
    }
    // The same message may repeat every frame, so output is limited to the first 50
    if (self->printed_.fetch_add(1) < 50)
        std::fprintf(stderr, "[vulkan %s] %s\n", tag, data && data->pMessage ? data->pMessage : "(null)");
    return VK_FALSE;
}

VulkanContext::VulkanContext(const VulkanContextOptions& opt) {
    try {
        // ---- Instance ----
        const bool want_validation = opt.enable_validation || env_true("YVO_VK_VALIDATION");
        const bool env_validation = env_contains("VK_LOADER_LAYERS_ENABLE", "validation") ||
                                    env_contains("VK_INSTANCE_LAYERS", "validation");
        std::vector<const char*> layers;
        if (want_validation) {
            if (layer_available(kValidationLayer)) {
                layers.push_back(kValidationLayer);
            } else {
                std::fprintf(stderr, "[vulkan] %s not available, continuing without it\n", kValidationLayer);
            }
        }
        validation_layer_active_ = !layers.empty() || env_validation;

        auto inst_exts = instance_extensions(nullptr);
        if (validation_layer_active_) {
            auto le = instance_extensions(kValidationLayer);
            inst_exts.insert(inst_exts.end(), le.begin(), le.end());
        }
        std::vector<const char*> exts;
        const bool debug_utils = has_ext(inst_exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (debug_utils) exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        for (const char* e : exts) enabled_inst_exts_.emplace_back(e);

        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "yolo_vulkan_optimize";
        app.applicationVersion = 1;
        app.pEngineName = "yvo";
        app.apiVersion = VK_API_VERSION_1_3;

        VkDebugUtilsMessengerCreateInfoEXT dbg{};
        dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbg.pfnUserCallback = &VulkanContext::debug_callback;
        dbg.pUserData = this;

        VkValidationFeatureEnableEXT sync_enable = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
        VkValidationFeaturesEXT vfeat{};
        vfeat.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        vfeat.enabledValidationFeatureCount = 1;
        vfeat.pEnabledValidationFeatures = &sync_enable;

        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        ici.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
        ici.ppEnabledLayerNames = layers.data();
        ici.enabledExtensionCount = static_cast<std::uint32_t>(exts.size());
        ici.ppEnabledExtensionNames = exts.data();
        const void** tail = &ici.pNext;
        if (debug_utils) {
            *tail = &dbg; // also receive messages during instance creation/destruction
            tail = const_cast<const void**>(&dbg.pNext);
        }
        const bool want_sync = (opt.validation_sync || env_true("YVO_VK_SYNC_VALIDATION")) && validation_layer_active_ &&
                               has_ext(instance_extensions(kValidationLayer), VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        if (want_sync) *tail = &vfeat;
        YVO_VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));

        if (debug_utils) {
            auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
            if (create) {
                dbg.pNext = nullptr;
                YVO_VK_CHECK(create(instance_, &dbg, nullptr, &messenger_));
            }
        }

        // ---- Physical device: select V3D (exclude CPU devices such as llvmpipe) ----
        std::uint32_t n = 0;
        YVO_VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, nullptr));
        std::vector<VkPhysicalDevice> pds(n);
        YVO_VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, pds.data()));
        std::string hint = opt.device_name_hint;
        if (const char* e = std::getenv("YVO_VK_DEVICE"); e && *e) hint = e;
        for (VkPhysicalDevice pd : pds) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(pd, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
            if (!hint.empty() && std::strstr(p.deviceName, hint.c_str()) == nullptr) continue;
            phys_ = pd;
            props_ = p;
            break;
        }
        if (phys_ == VK_NULL_HANDLE)
            throw std::runtime_error("VulkanContext: no non-CPU physical device matching \"" + hint + "\"");
        vkGetPhysicalDeviceMemoryProperties(phys_, &mem_props_);
        if (props_.apiVersion < VK_API_VERSION_1_3)
            throw std::runtime_error(std::string("VulkanContext: device ") + props_.deviceName +
                                     " does not support Vulkan 1.3");

        // ---- Compute queue ----
        std::uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qfs.data());
        // Score-based selection: families without compute are excluded → +2 for those without GRAPHICS → on a tie,
        // the smaller index = "prefer a compute-only family, otherwise the first family that supports compute".
        // V3D has a single queue family, so it picks family 0.
        bool found = false;
        int best_score = -1;
        for (std::uint32_t i = 0; i < qn; ++i) {
            if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            const int score = (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 ? 2 : 0;
            if (score > best_score) {  // on a tie, keep the one seen first (= the smaller index)
                best_score = score;
                queue_family_ = i;
                found = true;
            }
        }
        if (!found) throw std::runtime_error("VulkanContext: no compute queue family");

        // ---- Extensions ----
        const auto dexts = device_extensions(phys_);
        std::vector<const char*> dev_exts;
        for (const char* req : {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME}) {
            if (!has_ext(dexts, req))
                throw std::runtime_error(std::string("VulkanContext: missing device extension ") + req);
            dev_exts.push_back(req);
        }
        if (has_ext(dexts, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
            dev_exts.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
            external_qf_ = VK_QUEUE_FAMILY_FOREIGN_EXT;
        }

        const VkExternalMemoryFeatureFlags f = dmabuf_buffer_features(phys_);
        dmabuf_import_ = (f & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        dmabuf_export_ = (f & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
        dmabuf_import_dedicated_only_ = dmabuf_export_dedicated_only_ =
            (f & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
        if (!dmabuf_import_)
            throw std::runtime_error("VulkanContext: dma-buf import not supported for storage buffers");

        timestamp_valid_bits_ = qfs[queue_family_].timestampValidBits;
        timestamps_supported_ = timestamp_valid_bits_ > 0 && props_.limits.timestampComputeAndGraphics;

        // ---- Logical device ----
        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f13;
        vkGetPhysicalDeviceFeatures2(phys_, &f2);
        if (!f13.synchronization2) throw std::runtime_error("VulkanContext: synchronization2 not supported");
        VkPhysicalDeviceVulkan13Features en13{};
        en13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        en13.synchronization2 = VK_TRUE;
        // Required for the shader's local_size_x_id (SPIR-V 1.6 LocalSizeId). A required feature in Vulkan 1.3.
        if (!f13.maintenance4) throw std::runtime_error("VulkanContext: maintenance4 not supported");
        en13.maintenance4 = VK_TRUE;
        VkPhysicalDeviceFeatures2 en2{};
        en2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        en2.pNext = &en13;

        for (const char* e : dev_exts) enabled_dev_exts_.emplace_back(e);

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &en2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<std::uint32_t>(dev_exts.size());
        dci.ppEnabledExtensionNames = dev_exts.data();
        YVO_VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &device_));
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
        vkGetMemoryFdPropertiesKHR = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryFdPropertiesKHR"));
        if (!vkGetMemoryFdKHR || !vkGetMemoryFdPropertiesKHR)
            throw std::runtime_error("VulkanContext: external memory fd entry points missing");

        if (opt.verbose) {
            std::printf("[vulkan] device: %s (api %u.%u.%u, driver 0x%x), queue family %u, timestamps %s "
                        "(period %.3f ns, %u bits), dma-buf import=%d export=%d dedicated_only=%d, "
                        "external qf=%s, validation=%s%s\n",
                        props_.deviceName, VK_API_VERSION_MAJOR(props_.apiVersion),
                        VK_API_VERSION_MINOR(props_.apiVersion), VK_API_VERSION_PATCH(props_.apiVersion),
                        props_.driverVersion, queue_family_, timestamps_supported_ ? "yes" : "no",
                        static_cast<double>(props_.limits.timestampPeriod), timestamp_valid_bits_, dmabuf_import_,
                        dmabuf_export_, dmabuf_import_dedicated_only_,
                        external_qf_ == VK_QUEUE_FAMILY_FOREIGN_EXT ? "FOREIGN_EXT" : "EXTERNAL",
                        validation_layer_active_ ? "on" : "off", want_sync ? " +sync" : "");
        }
    } catch (...) {
        destroy();
        throw;
    }
}

VulkanContext::~VulkanContext() { destroy(); }

void VulkanContext::destroy() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (messenger_ != VK_NULL_HANDLE) {
        auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_fn) destroy_fn(instance_, messenger_, nullptr);
        messenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

std::uint32_t VulkanContext::find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                              VkMemoryPropertyFlags preferred) const {
    std::uint32_t best = UINT32_MAX;
    int best_score = -1;
    for (std::uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        const VkMemoryPropertyFlags fl = mem_props_.memoryTypes[i].propertyFlags;
        if ((fl & required) != required) continue;
        const int score = __builtin_popcount(fl & preferred);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

} // namespace yvo::vk
