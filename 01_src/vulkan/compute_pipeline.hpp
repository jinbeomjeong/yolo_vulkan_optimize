// Compute pipeline creation helpers shared by the library classes (SandLetterbox, YoloPostproc, EfficientNms,
// display::OverlayRenderer).
//
// Every class used the same fixed shape, so it lives here once:
//   - one descriptor set: binding i = types[i], descriptorCount 1, compute stage
//   - one push constant range [0, push_size) for the compute stage
//   - specialization constants: constant_id i = spec[i], 32-bit each (VkBool32 for bool), offset 4 * i
//   - entry point "main"; the shader module only lives during vkCreateComputePipelines
// The returned handles are owned by the caller (the classes destroy them in their destroy(), after vkDeviceWaitIdle).
// Every function throws VulkanError (vk_check.hpp) on failure and then has created nothing.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

namespace yvo::vk {

VkDescriptorSetLayout create_compute_set_layout(VkDevice dev, std::initializer_list<VkDescriptorType> types);

VkPipelineLayout create_compute_pipeline_layout(VkDevice dev, VkDescriptorSetLayout set_layout,
                                                std::uint32_t push_size);

// SPIR-V words to build from: the embedded array, or (development) the .spv file named by the environment variable
// env_var when it is set and non-empty. who prefixes the log line and error messages (e.g. "SandLetterbox").
std::vector<std::uint32_t> spirv_with_env_override(std::span<const std::uint32_t> embedded, const char* env_var,
                                                   const char* who);

VkPipeline create_compute_pipeline(VkDevice dev, VkPipelineLayout layout, std::span<const std::uint32_t> spirv,
                                   std::span<const std::uint32_t> spec);

} // namespace yvo::vk
