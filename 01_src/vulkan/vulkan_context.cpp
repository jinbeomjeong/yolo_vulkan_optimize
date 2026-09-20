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

// vulkan_wayland.h 를 포함하면 <wayland-client.h> 가 함께 끌려 들어와 yvo_vulkan 이 Wayland 헤더에 의존하게
// 된다. 필요한 것은 확장 이름 문자열과 함수 포인터 타입 하나뿐이므로 직접 선언한다.
// wl_display* 는 불투명 포인터(void*)로 다룬다 — ABI가 동일하다.
constexpr const char* kWaylandSurfaceExt = "VK_KHR_wayland_surface";
using PFN_yvoWaylandPresentationSupport = VkBool32(VKAPI_PTR*)(VkPhysicalDevice, std::uint32_t, void*);

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

// STORAGE_BUFFER 용도의 dma-buf 외부 메모리 기능 조회
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
    // 같은 메시지가 프레임마다 반복될 수 있으므로 출력은 앞쪽 50건으로 제한
    if (self->printed_.fetch_add(1) < 50)
        std::fprintf(stderr, "[vulkan %s] %s\n", tag, data && data->pMessage ? data->pMessage : "(null)");
    return VK_FALSE;
}

VulkanContext::VulkanContext(const VulkanContextOptions& opt) {
    try {
        // ---- 인스턴스 ----
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

        // ---- present: 인스턴스 확장 (enable_present일 때만. 끄면 목록이 위 한 줄로 끝난다) ----
        auto present_off = [&](std::string why) {
            if (present_reason_.empty()) present_reason_ = std::move(why);
            if (opt.require_present)
                throw std::runtime_error("VulkanContext: present required but unavailable: " + present_reason_);
        };
        bool want_present = opt.enable_present;
        if (want_present && opt.wayland_display == nullptr) {
            want_present = false;
            present_off("wayland_display가 null이다 (WaylandWindow를 먼저 열어야 한다)");
        }
        if (want_present) {
            for (const char* req : {VK_KHR_SURFACE_EXTENSION_NAME, kWaylandSurfaceExt}) {
                if (!has_ext(inst_exts, req)) {
                    want_present = false;
                    present_off(std::string("인스턴스 확장 없음: ") + req);
                }
            }
        }
        if (want_present) {
            exts.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
            exts.push_back(kWaylandSurfaceExt);
            // VK_EXT_surface_maintenance1 은 present mode별 minImageCount 질의(FIFO 3 / MAILBOX 4)를 준다.
            // 의존성: VK_KHR_get_surface_capabilities2 (인스턴스). 둘 다 있어야 켠다.
            // ⚠ 디바이스 확장 VK_EXT_swapchain_maintenance1 은 **이 인스턴스 확장을 전제로 한다.**
            //   여기서 못 켜면 아래 디바이스 확장도 켜지 않는다.
            if (has_ext(inst_exts, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME) &&
                has_ext(inst_exts, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) {
                exts.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
                exts.push_back(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
                surface_maintenance1_ = true;
            }
        }
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
            *tail = &dbg; // 인스턴스 생성·파괴 중 메시지도 받는다
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

        // ---- 물리 장치: V3D 선택 (llvmpipe 등 CPU 장치 제외) ----
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

        // ---- present: presentation support 질의 함수 (surface가 필요 없는 유일한 경로) ----
        PFN_yvoWaylandPresentationSupport wl_present_support = nullptr;
        if (want_present) {
            wl_present_support = reinterpret_cast<PFN_yvoWaylandPresentationSupport>(
                vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceWaylandPresentationSupportKHR"));
            if (!wl_present_support) {
                want_present = false;
                present_off("vkGetPhysicalDeviceWaylandPresentationSupportKHR 진입점을 얻지 못했다");
            }
        }

        // ---- 컴퓨트 큐 (present 여부가 선택에 영향을 주므로 확장 결정보다 먼저 한다) ----
        std::uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qfs.data());
        // 점수 기반 선택. **present == false 일 때 기존 2-pass 루프와 항상 같은 결과가 나온다**:
        //   컴퓨트 없는 패밀리는 제외 → GRAPHICS 없는 쪽에 +2 → 동점이면 인덱스가 작은 쪽
        //   = "컴퓨트 전용 패밀리 우선, 없으면 컴퓨트를 지원하는 첫 패밀리" 와 동일하다.
        // present == true 면 present 지원에 +4(사실상 필수), GRAPHICS 에 +1(그래픽스 폴백 여지).
        // V3D 는 큐 패밀리가 1개라 양쪽 모두 family 0 을 고른다 — 이 하드웨어에서 회귀 위험은 0이다.
        bool present_qf_found = false;
        auto select_queue_family = [&](bool present) {
            bool found = false;
            int best_score = -1;
            present_qf_found = false;
            for (std::uint32_t i = 0; i < qn; ++i) {
                if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
                const bool graphics = (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
                int score = 0;
                if (present) {
                    if (wl_present_support(phys_, i, opt.wayland_display) == VK_TRUE) {
                        score += 4;
                        present_qf_found = true;
                    }
                    if (graphics) score += 1;
                } else if (!graphics) {
                    score += 2;
                }
                if (score > best_score) {  // 동점이면 먼저 본 쪽(= 인덱스가 작은 쪽)을 유지한다
                    best_score = score;
                    queue_family_ = i;
                    graphics_supported_ = graphics;
                    found = true;
                }
            }
            return found;
        };
        bool found = select_queue_family(want_present);
        if (found && want_present && !present_qf_found) {
            want_present = false;
            present_off("present를 지원하는 컴퓨트 큐 패밀리가 없다 "
                        "(vkGetPhysicalDeviceWaylandPresentationSupportKHR == VK_FALSE)");
            found = select_queue_family(false);  // present 없는 기준 규칙으로 다시 고른다
        }
        if (!found) throw std::runtime_error("VulkanContext: no compute queue family");

        // ---- 확장 ----
        // present 선택 확장은 "확장이 있는가"와 "feature가 켜지는가"를 모두 통과해야 활성화된다.
        bool sm1_avail = false, pid_avail = false, pwait_avail = false;
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
        // ---- present: 디바이스 확장 (present가 살아 있을 때만. 끄면 위 3개로 끝난다) ----
        if (want_present && !has_ext(dexts, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            want_present = false;
            present_off(std::string("디바이스 확장 없음: ") + VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }
        if (want_present) {
            dev_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            // ⚠ VK_EXT_swapchain_maintenance1 은 인스턴스 확장 VK_EXT_surface_maintenance1 을 **전제로 한다**
            //   (모드별 minImageCount 질의와 VkSwapchainPresentModesCreateInfoEXT 가 한 쌍이다).
            //   surface_maintenance1_ 가 false면 여기서도 켜지 않는다. 계획서 §5.2 안전장치 3.
            //   (드라이버는 승격판 VK_KHR_swapchain_maintenance1 도 광고하지만 설치된 Vulkan 헤더 1.4.309 에
            //    KHR 쪽 feature 구조체가 없고 인스턴스 짝도 EXT 뿐이라 EXT 판을 쓴다.)
            sm1_avail = surface_maintenance1_ && has_ext(dexts, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
            // present_wait 는 present_id 를 전제로 한다(제출한 presentId 를 기다리는 구조).
            pid_avail = has_ext(dexts, VK_KHR_PRESENT_ID_EXTENSION_NAME);
            pwait_avail = pid_avail && has_ext(dexts, VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
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

        // ---- 논리 장치 ----
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
        // 셰이더의 local_size_x_id(SPIR-V 1.6 LocalSizeId)에 필요. Vulkan 1.3 필수 기능.
        if (!f13.maintenance4) throw std::runtime_error("VulkanContext: maintenance4 not supported");
        en13.maintenance4 = VK_TRUE;
        VkPhysicalDeviceFeatures2 en2{};
        en2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        en2.pNext = &en13;

        // ---- present 선택 기능: 확장이 광고돼 있고 feature 도 지원할 때만 확장·기능을 함께 켠다 ----
        // (확장만 켜고 feature 를 안 켜면 검증 레이어가 "feature not enabled" 를 내고, feature 만 켜면
        //  vkCreateDevice 가 VK_ERROR_FEATURE_NOT_PRESENT 를 낸다. 둘을 항상 쌍으로 다룬다.)
        VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT sm1_q{}, sm1_en{};
        sm1_q.sType = sm1_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
        VkPhysicalDevicePresentIdFeaturesKHR pid_q{}, pid_en{};
        pid_q.sType = pid_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
        VkPhysicalDevicePresentWaitFeaturesKHR pwait_q{}, pwait_en{};
        pwait_q.sType = pwait_en.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR;
        if (sm1_avail || pid_avail || pwait_avail) {
            VkPhysicalDeviceFeatures2 q2{};
            q2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            void** qtail = &q2.pNext;
            if (sm1_avail) { *qtail = &sm1_q; qtail = &sm1_q.pNext; }
            if (pid_avail) { *qtail = &pid_q; qtail = &pid_q.pNext; }
            if (pwait_avail) { *qtail = &pwait_q; qtail = &pwait_q.pNext; }
            vkGetPhysicalDeviceFeatures2(phys_, &q2);

            void** etail = &en13.pNext;
            if (sm1_avail && sm1_q.swapchainMaintenance1) {
                swapchain_maintenance1_ = true;
                dev_exts.push_back(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
                sm1_en.swapchainMaintenance1 = VK_TRUE;
                *etail = &sm1_en;
                etail = &sm1_en.pNext;
            }
            if (pid_avail && pid_q.presentId) {
                present_id_ = true;
                dev_exts.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
                pid_en.presentId = VK_TRUE;
                *etail = &pid_en;
                etail = &pid_en.pNext;
            }
            // present_wait 는 present_id 가 실제로 켜졌을 때만 의미가 있다.
            if (present_id_ && pwait_avail && pwait_q.presentWait) {
                present_wait_ = true;
                dev_exts.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
                pwait_en.presentWait = VK_TRUE;
                *etail = &pwait_en;
                etail = &pwait_en.pNext;
            }
            (void)etail;
        }
        present_enabled_ = want_present;
        if (opt.enable_present && !present_enabled_ && opt.require_present)
            throw std::runtime_error("VulkanContext: present required but unavailable: " + present_reason_);
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
            if (opt.enable_present) {
                if (present_enabled_)
                    std::printf("[vulkan] present: on (graphics=%d, surface_maintenance1=%d, "
                                "swapchain_maintenance1=%d, present_id=%d, present_wait=%d)\n",
                                graphics_supported_, surface_maintenance1_, swapchain_maintenance1_, present_id_,
                                present_wait_);
                else
                    std::printf("[vulkan] present: off (%s)\n", present_reason_.c_str());
            }
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
