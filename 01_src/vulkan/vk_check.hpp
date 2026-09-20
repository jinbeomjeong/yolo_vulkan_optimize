// VkResult 검사 헬퍼: 실패하면 호출 위치와 결과 코드 이름을 담은 예외를 던진다.
#pragma once

#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

namespace yvo::vk {

inline const char* result_str(VkResult r) {
    switch (r) {
#define YVO_VK_CASE(x) \
    case x: return #x;
        YVO_VK_CASE(VK_SUCCESS)
        YVO_VK_CASE(VK_NOT_READY)
        YVO_VK_CASE(VK_TIMEOUT)
        YVO_VK_CASE(VK_EVENT_SET)
        YVO_VK_CASE(VK_EVENT_RESET)
        YVO_VK_CASE(VK_INCOMPLETE)
        YVO_VK_CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
        YVO_VK_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
        YVO_VK_CASE(VK_ERROR_INITIALIZATION_FAILED)
        YVO_VK_CASE(VK_ERROR_DEVICE_LOST)
        YVO_VK_CASE(VK_ERROR_MEMORY_MAP_FAILED)
        YVO_VK_CASE(VK_ERROR_LAYER_NOT_PRESENT)
        YVO_VK_CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
        YVO_VK_CASE(VK_ERROR_FEATURE_NOT_PRESENT)
        YVO_VK_CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
        YVO_VK_CASE(VK_ERROR_TOO_MANY_OBJECTS)
        YVO_VK_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
        YVO_VK_CASE(VK_ERROR_FRAGMENTED_POOL)
        YVO_VK_CASE(VK_ERROR_UNKNOWN)
        YVO_VK_CASE(VK_ERROR_OUT_OF_POOL_MEMORY)
        YVO_VK_CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE)
        YVO_VK_CASE(VK_ERROR_FRAGMENTATION)
        YVO_VK_CASE(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)
#undef YVO_VK_CASE
    default: return "VK_RESULT_UNKNOWN";
    }
}

class VulkanError : public std::runtime_error {
public:
    VulkanError(const std::string& where, VkResult r)
        : std::runtime_error(where + " failed: " + result_str(r) + " (" + std::to_string(static_cast<int>(r)) + ")"),
          result_(r) {}
    VkResult result() const { return result_; }

private:
    VkResult result_;
};

inline void check(VkResult r, const char* call, const char* file, int line) {
    if (r != VK_SUCCESS)
        throw VulkanError(std::string(file) + ":" + std::to_string(line) + " " + call, r);
}

} // namespace yvo::vk

#define YVO_VK_CHECK(expr) ::yvo::vk::check((expr), #expr, __FILE__, __LINE__)
