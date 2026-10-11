#include "vulkan/compute_pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "vulkan/vk_check.hpp"

namespace yvo::vk {

VkDescriptorSetLayout create_compute_set_layout(VkDevice dev, std::initializer_list<VkDescriptorType> types) {
    std::vector<VkDescriptorSetLayoutBinding> b(types.size());
    std::uint32_t i = 0;
    for (VkDescriptorType t : types) {
        b[i].binding = i;
        b[i].descriptorType = t;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        ++i;
    }
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = static_cast<std::uint32_t>(b.size());
    dl.pBindings = b.data();
    VkDescriptorSetLayout out = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &out));
    return out;
}

VkPipelineLayout create_compute_pipeline_layout(VkDevice dev, VkDescriptorSetLayout set_layout,
                                                std::uint32_t push_size) {
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    VkPipelineLayout out = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreatePipelineLayout(dev, &pl, nullptr, &out));
    return out;
}

std::vector<std::uint32_t> spirv_with_env_override(std::span<const std::uint32_t> embedded, const char* env_var,
                                                   const char* who) {
    const char* path = std::getenv(env_var);
    if (!path || !*path) return {embedded.begin(), embedded.end()};
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error(std::string(who) + ": cannot open " + path);
    const auto n = static_cast<std::size_t>(in.tellg());
    if (n == 0 || n % 4) throw std::runtime_error(std::string(who) + ": bad SPIR-V size: " + path);
    std::vector<std::uint32_t> words(n / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(n));
    std::fprintf(stderr, "[%s] using external SPIR-V %s\n", who, path);
    return words;
}

VkPipeline create_compute_pipeline(VkDevice dev, VkPipelineLayout layout, std::span<const std::uint32_t> spirv,
                                   std::span<const std::uint32_t> spec) {
    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = spirv.size_bytes();
    smi.pCode = spirv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    YVO_VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &module));

    std::vector<VkSpecializationMapEntry> spec_map(spec.size());
    for (std::uint32_t i = 0; i < spec_map.size(); ++i) spec_map[i] = {i, 4 * i, 4};
    VkSpecializationInfo si{};
    si.mapEntryCount = static_cast<std::uint32_t>(spec_map.size());
    si.pMapEntries = spec_map.data();
    si.dataSize = spec.size_bytes();
    si.pData = spec.data();

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    cpi.layout = layout;
    VkPipeline out = VK_NULL_HANDLE;
    const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &out);
    vkDestroyShaderModule(dev, module, nullptr); // the module is not needed once the pipeline exists
    YVO_VK_CHECK(r);
    return out;
}

} // namespace yvo::vk
